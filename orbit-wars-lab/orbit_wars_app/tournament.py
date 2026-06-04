"""Tournament entrypoints.

Execution now lives in `orbit_wars_app.scheduler` (the process-wide killable
match scheduler). This module keeps:

  * `Tournament` — a thin, blocking wrapper that spins up a transient Scheduler,
    submits one tournament, and waits for it to finish. Handy for the CLI, tests,
    and any synchronous caller. `config.parallel` becomes the transient pool's
    worker count (the API path instead uses the shared scheduler's system-wide
    concurrency setting and ignores `parallel`).
  * the CLI (`list` / `show` / `run` / `gauntlet` / `head-to-head`).

Pure expansion helpers (agent resolution, pair generation, tag filtering) live
in `orbit_wars_app.pairing`.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .discovery import scan_zoo
from .pairing import filter_agents_by_tags
from .scheduler import Scheduler, allocate_run_id
from .schemas import MatchResult, TournamentConfig
from .trueskill_store import TrueSkillStore
from .value_evaluator import default_value_model_path


class TournamentCancelled(Exception):
    """Raised when the caller requests a graceful tournament stop."""


# ============================================================
# ProcessPoolExecutor worker — top-level so it pickles cleanly.
# ============================================================


@dataclass
class _WorkerResult:
    """Pickle-friendly carrier for what the main process needs after a match.

    Avoids shipping the full replay dict back through the pipe — the worker
    persists it to disk locally (much smaller payload across the wire) and
    returns only the metadata + relative path string.

    `worker_pid` is included so tests can assert real parallelism (multiple
    distinct PIDs over N matches) without relying on flaky wallclock checks.

    `per_agent_turn_seconds` carries the per-turn duration samples kaggle-envs
    captured for each agent during the match — folded into RuntimeStore by the
    main process after the worker returns. Aligned to `agent_ids`.
    """
    match_counter: int
    agent_ids: list[str]
    winner: Optional[int]
    scores: list[float]
    turns: int
    duration_s: float
    seed: int
    status: str
    replay_path: str
    worker_pid: int = 0
    per_agent_turn_seconds: list[list[float]] = field(default_factory=list)
    # When the match ended in a non-ok status the runner attaches an error
    # string to `outcome.replay["error"]`. Carrying it through to the UI is
    # what turns "⚠ crashed" into "⚠ crashed — Permission denied: …" so
    # the failure is actually debuggable instead of mystery-meat.
    error: Optional[str] = None


def _quiet_kaggle_environments() -> None:
    """Silence kaggle-environments' chatty import-time logging in workers.

    The OpenSpiel env loader hard-codes `_log.setLevel(logging.INFO)` and
    streams ~30 lines per import to stdout. Bumping the
    `kaggle_environments` logger level above INFO suppresses those without
    affecting our own logs (we log under `orbit_wars_app.*`). Loggers are
    process-wide and persist across `ProcessPoolExecutor` worker reuse, so
    we run this once per worker via the executor's `initializer` rather
    than per match.
    """
    import logging
    logging.getLogger("kaggle_environments").setLevel(logging.WARNING)


def _run_match_in_worker(
    match_counter: int,
    agent_ids: list[str],
    agent_paths_str: list[str],
    mode: str,
    seed: int,
    replays_dir_str: Optional[str],
    save_replays: bool,
    logs_dir_str: Optional[str] = None,
    value_model_path: Optional[str] = None,
) -> _WorkerResult:
    """Run a single match in a worker process and persist the replay locally.

    Returns lightweight metadata (no replay dict) so cross-process pipe
    bandwidth isn't dominated by 5-10MB JSON payloads. Replay is written
    inside the worker because the file system is shared and the main process
    needs nothing about replay bytes other than the path.
    """
    agent_paths = [Path(p) for p in agent_paths_str]
    log_dir = Path(logs_dir_str) if logs_dir_str else None
    outcome = run_match(
        agent_ids=agent_ids,
        agent_paths=agent_paths,
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        log_dir=log_dir,
        log_prefix=f"{match_counter:03d}",
        value_model_path=value_model_path,
    )

    replay_rel = ""
    if save_replays and replays_dir_str and outcome.replay and "steps" in outcome.replay:
        replays_dir = Path(replays_dir_str)
        rp = save_replay(replays_dir, match_counter, agent_ids, outcome.replay)
        replay_rel = str(rp.relative_to(replays_dir.parent))

    err: Optional[str] = None
    if isinstance(outcome.replay, dict):
        e = outcome.replay.get("error")
        if isinstance(e, str) and e:
            err = e
    return _WorkerResult(
        match_counter=match_counter,
        agent_ids=agent_ids,
        winner=outcome.winner,
        scores=outcome.scores,
        turns=outcome.turns,
        duration_s=outcome.duration_s,
        seed=seed,
        status=outcome.status,
        replay_path=replay_rel,
        worker_pid=os.getpid(),
        per_agent_turn_seconds=outcome.per_agent_turn_seconds,
        error=err,
    )


def _filter_agents_by_tags(
    agents: list["AgentInfo"],
    include: list[str],
    exclude: list[str],
) -> list["AgentInfo"]:
    """Filter agents by tags.

    Semantics:
    - `include=[]` → all (then exclude can still trim)
    - `include=[a, b, ...]` → any tag from the list must be present (OR)
    - `exclude=[a, b, ...]` → none of these tags may be present (AND)
    - `disabled: true` → always skipped

    Returns the list in original order.
    """
    out = []
    inc_set = set(include)
    exc_set = set(exclude)
    for a in agents:
        if a.disabled:
            continue
        tag_set = set(a.tags)
        if inc_set and not (tag_set & inc_set):
            continue
        if exc_set and (tag_set & exc_set):
            continue
        out.append(a)
    return out


class Tournament:
    """Blocking single-tournament wrapper around the scheduler.

    Kept for the CLI and synchronous callers. Each `run()` creates a transient
    `Scheduler` (so the CLI doesn't depend on a long-lived process-wide one),
    submits the config, blocks until it finalizes, and returns the run id.
    """

    def __init__(
        self,
        config: TournamentConfig,
        *,
        runs_root: Path,
        zoo_root: Path,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.runs_root = runs_root
        self.zoo_root = zoo_root
        self.cancel_event = cancel_event
        # Throttle for run.json writes during a tournament. The UI polls this
        # file for progress, so we trade a small staleness budget (a few
        # matches / ~1s) for ~10× fewer JSON writes on large round-robins.
        # Terminal writes (status != "running") always go through.
        self._run_json_last_write_t: float = 0.0
        self._run_json_last_done: int = -1

    def next_run_id(self) -> str:
        """Next collision-safe run id without creating the directory."""
        return allocate_run_id(self.runs_root)

    # Retained name — some tests/call sites reference the private form.
    _new_run_id = next_run_id

    def run(
        self,
        on_match_done: Optional[Callable[["MatchResult", int, int], None]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Execute the tournament synchronously and return its run id.

        `run_id` is accepted for backward compatibility but ignored — the
        scheduler allocates the id atomically. `on_match_done` fires per match.
        """
        concurrency = max(1, getattr(self.config, "parallel", 1) or 1)
        sched = Scheduler(
            runs_root=self.runs_root,
            zoo_root=self.zoo_root,
            concurrency=concurrency,
        )
        sched.start()
        try:
            if self.config.parallel <= 1 or self.config.mode not in ("fast", "ultrafast", "value"):
                # Sequential path. Faithful mode also takes this branch:
                # subprocess agents already use OS-level concurrency, and
                # nesting ProcessPoolExecutor on top would multiply RAM and
                # lose the per-match HTTP retry isolation.
                for mc, aids, apaths, seed in jobs:
                    self._check_cancel()
                    outcome = run_match(
                        agent_ids=aids,
                        agent_paths=apaths,
                        mode=self.config.mode,
                        seed=seed,
                        log_dir=logs_dir,
                        log_prefix=f"{mc:03d}",
                        value_model_path=self.config.value_model_path,
                    )
                    replay_rel = ""
                    if (self.config.save_replays
                            and outcome.replay and "steps" in outcome.replay):
                        rp = save_replay(replays_dir, mc, aids, outcome.replay)
                        replay_rel = str(rp.relative_to(run_dir))
                    completed_count += 1
                    seq_err: Optional[str] = None
                    if isinstance(outcome.replay, dict):
                        e = outcome.replay.get("error")
                        if isinstance(e, str) and e:
                            seq_err = e
                    self._handle_match_outcome(
                        matches, store, runtime_store, run_dir, run_id, started_at,
                        total_matches, mc, aids, seed,
                        winner=outcome.winner, scores=outcome.scores,
                        turns=outcome.turns, duration_s=outcome.duration_s,
                        match_status=outcome.status, replay_path=replay_rel,
                        per_agent_turn_seconds=outcome.per_agent_turn_seconds,
                        on_match_done=on_match_done,
                        error=seq_err,
                    )
            else:
                # Parallel path keeps at most `parallel` matches in flight so a
                # cancel request can stop launching new work promptly, then wait
                # only for currently-running matches to finish.
                ex = ProcessPoolExecutor(
                    max_workers=self.config.parallel,
                    initializer=_quiet_kaggle_environments,
                )
                try:
                    job_iter = iter(jobs)
                    futs: dict[object, tuple[int, list[str], list[Path], int]] = {}

                    def submit_job(job: tuple[int, list[str], list[Path], int]) -> None:
                        mc, aids, apaths, seed = job
                        futs[ex.submit(
                            _run_match_in_worker,
                            mc, aids, [str(p) for p in apaths],
                            self.config.mode, seed,
                            str(replays_dir) if self.config.save_replays else None,
                            self.config.save_replays,
                            str(logs_dir),
                            self.config.value_model_path,
                        )] = (mc, aids, apaths, seed)

                    self._check_cancel()
                    for _ in range(min(self.config.parallel, len(jobs))):
                        submit_job(next(job_iter))

                    cancelling = False
                    while futs:
                        fut = next(as_completed(futs))
                        try:
                            wr = fut.result()
                        except BaseException as e:
                            # A worker exception not caught by run_match's own
                            # try/except (pickle failure, OOM, OSError on
                            # replay write) would otherwise abort the entire
                            # tournament and discard already-collected
                            # matches. Synthesize a `crashed` outcome so this
                            # match drops out of ratings but the rest of the
                            # run completes — same fault-tolerance contract
                            # as `run_match_fast`.
                            mc, aids, apaths, seed = futs[fut]
                            worker_err = f"{type(e).__name__}: {e}"
                            wr = _WorkerResult(
                                match_counter=mc, agent_ids=aids,
                                winner=None, scores=[], turns=0,
                                duration_s=0.0, seed=seed,
                                status="crashed", replay_path="",
                                worker_pid=0,
                                error=worker_err,
                            )
                            (run_dir / f"worker-error-{mc:03d}.txt").write_text(
                                worker_err + "\n"
                            )
                        finally:
                            futs.pop(fut, None)
                        completed_count += 1
                        self._handle_match_outcome(
                            matches, store, runtime_store, run_dir, run_id, started_at,
                            total_matches, wr.match_counter, wr.agent_ids,
                            wr.seed, winner=wr.winner, scores=wr.scores,
                            turns=wr.turns, duration_s=wr.duration_s,
                            match_status=wr.status, replay_path=wr.replay_path,
                            per_agent_turn_seconds=wr.per_agent_turn_seconds,
                            progress_count=completed_count,
                            on_match_done=on_match_done,
                            error=wr.error,
                        )
                        if self._cancel_requested():
                            cancelling = True
                        if not cancelling:
                            try:
                                submit_job(next(job_iter))
                            except StopIteration:
                                pass
                    if cancelling:
                        raise TournamentCancelled()
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
        except TournamentCancelled:
            status = "aborted"
        except BaseException:
            status = "aborted"
            raise
        finally:
            # Persist partial state no matter what
            store.save()
            store.snapshot_to(run_dir / "trueskill.json")
            runtime_store.save()

            finished_at = datetime.now(timezone.utc).isoformat()
            self._write_run_json(
                run_dir, run_id, started_at, finished_at, status,
                total_matches, completed_count,
            )

            # config + results always written (partial on abort)
            (run_dir / "config.json").write_text(json.dumps({
                "mode": self.config.mode,
                "format": self.config.format,
                "games_per_pair": self.config.games_per_pair,
                "agents": self.config.agents,
                "seed_base": self.config.seed_base,
                "seed_mode": self.config.seed_mode,
                "parallel": self.config.parallel,
                "save_replays": self.config.save_replays,
                "shape": self.config.shape,
                "challenger_id": self.config.challenger_id,
                "is_quick_match": self.config.is_quick_match,
                "started_at": started_at,
            }, indent=2))

            # Parallel execution completes out-of-order; sort by match_id so
            # results.json reads naturally and downstream consumers (UI run
            # detail, head-to-head matrix) get consistent ordering.
            matches.sort(key=lambda m: m.match_id)
            summary = self._build_summary(matches)
            (run_dir / "results.json").write_text(json.dumps({
                "started_at": started_at,
                "finished_at": finished_at,
                "total_matches": total_matches,
                "matches": [m.model_dump() for m in matches],
                "summary": summary,
                "status": status,
            }, indent=2))

            # 'latest' symlink (best-effort)
            latest = self.runs_root / "latest"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            try:
                latest.symlink_to(run_id, target_is_directory=True)
            except (OSError, NotImplementedError):
                (self.runs_root / "latest.txt").write_text(run_id)

        return run_id

    def next_run_id(self) -> str:
        """Return the next collision-safe run id without creating the directory."""
        return self._new_run_id()

    def _cancel_requested(self) -> bool:
        return self.cancel_event.is_set() if self.cancel_event is not None else False

    def _check_cancel(self) -> None:
        if self._cancel_requested():
            raise TournamentCancelled()

    def _handle_match_outcome(
        self,
        matches: list[MatchResult],
        store: TrueSkillStore,
        runtime_store: RuntimeStore,
        run_dir: Path,
        run_id: str,
        started_at: str,
        total_matches: int,
        match_counter: int,
        agent_ids: list[str],
        seed: int,
        *,
        winner: Optional[int],
        scores: list[float],
        turns: int,
        duration_s: float,
        match_status: str,
        replay_path: str,
        per_agent_turn_seconds: Optional[list[list[float]]] = None,
        progress_count: Optional[int] = None,
        on_match_done: Optional[Callable[["MatchResult", int, int], None]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Post-process a finished match: update store, append to results,
        write run.json, fire callback. Called from both sequential and
        parallel branches so the side-effect protocol stays in one place.

        `progress_count` is the high-water mark for run.json (parallel branch
        passes the as_completed counter; sequential passes None and falls
        back to match_counter). Without this distinction the parallel branch
        would write decreasing values when matches complete out-of-order.
        """
        if match_status != "agent_failed_to_start":
            store.update_match(
                agent_ids=agent_ids,
                winner=winner,
                format=self.config.format,
                scores=scores,
            )
        # Fold per-agent per-turn timing samples into the runtime store. Done
        # even for `agent_failed_to_start` because some agents may still have
        # produced samples (e.g. only one of four faithful subprocesses failed
        # to spawn — list will just be empty for the missing slot).
        if per_agent_turn_seconds:
            for aid, samples in zip(agent_ids, per_agent_turn_seconds):
                runtime_store.record(aid, samples)
        match_result = MatchResult(
            match_id=f"{match_counter:03d}",
            agent_ids=agent_ids,
            winner=winner,
            scores=scores,
            turns=turns,
            duration_s=duration_s,
            status=match_status,  # type: ignore[arg-type]
            seed=seed,
            replay_path=replay_path,
            error=error,
        )
        matches.append(match_result)
        progress = progress_count if progress_count is not None else match_counter
        self._write_run_json(
            run_dir, run_id, started_at, None, "running",
            total_matches, progress,
        )
        if on_match_done is not None:
            on_match_done(match_result, progress, total_matches)

    # Throttle policy for in-progress run.json writes.
    _RUN_JSON_THROTTLE_MATCHES = 10
    _RUN_JSON_THROTTLE_SECONDS = 1.0

    def _write_run_json(
        self,
        run_dir: Path,
        run_id: str,
        started_at: str,
        finished_at: Optional[str],
        status: RunStatus,
        total_matches: int,
        matches_done: int,
    ) -> None:
        """Write run.json lifecycle file.

        run.json is the UI's single source for run lifecycle state; it
        intentionally duplicates mode/format from config.json so the
        web UI doesn't need to read two files per run.

        Throttled while `status == "running"`: writes at most every
        `_RUN_JSON_THROTTLE_SECONDS` or every `_RUN_JSON_THROTTLE_MATCHES`
        match completions, whichever comes first. Terminal writes
        (completed/aborted/etc.) and the very first/last match always go
        through so progress bars don't appear stuck.
        """
        if status == "running" and matches_done not in (0, total_matches):
            now = time.monotonic()
            since_last = matches_done - self._run_json_last_done
            elapsed = now - self._run_json_last_write_t
            if (
                since_last < self._RUN_JSON_THROTTLE_MATCHES
                and elapsed < self._RUN_JSON_THROTTLE_SECONDS
            ):
                return
            self._run_json_last_write_t = now
            self._run_json_last_done = matches_done
        payload = {
            "id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "mode": self.config.mode,
            "format": self.config.format,
            "status": status,
            "total_matches": total_matches,
            "matches_done": matches_done,
            "is_quick_match": self.config.is_quick_match,
        }
        # Atomic write: temp-file + rename. Plain `write_text` truncates
        # first, then writes — a concurrent /api/runs/.../progress poll can
        # land between those steps and read an empty file → JSONDecodeError
        # → 500. The rename is atomic on POSIX (and on Windows via
        # Path.replace), so readers see either the old or new payload but
        # never an empty file.
        target = run_dir / "run.json"
        tmp = run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(target)

    def _new_run_id(self) -> str:
        """YYYY-MM-DD-NNN — N is `max(existing_NNN_today) + 1`.

        Uses max-of-existing rather than count-of-existing so a deleted run
        in the middle of today's sequence (e.g. user removed -013 from the UI)
        doesn't make `count + 1` collide with an existing later index.
        """
        now = datetime.now(timezone.utc)
        prefix = now.strftime("%Y-%m-%d")
        max_n = 0
        for p in self.runs_root.iterdir():
            if not p.is_dir() or not p.name.startswith(prefix + "-"):
                continue
            suffix = p.name[len(prefix) + 1:]
            try:
                max_n = max(max_n, int(suffix))
            except ValueError:
                continue
        return f"{prefix}-{max_n + 1:03d}"

    def _resolve_agents(self) -> list[dict]:
        """Look up AgentInfo for each requested agent_id.

        Returns list of dicts with `id` and `path` fields.
        """
        all_agents = {a.id: a for a in scan_zoo(self.zoo_root)}
        out: list[dict] = []
        for aid in self.config.agents:
            info = all_agents.get(aid)
            if info is None:
                raise ValueError(
                    f"Agent {aid!r} not found in zoo {self.zoo_root}. "
                    f"Available: {sorted(all_agents)}"
                )
            if info.disabled:
                raise ValueError(
                    f"Agent {aid!r} is disabled; remove from config or un-disable"
                )
            out.append({"id": info.id, "path": info.path})
        return out

    def _generate_pairs(self, agents: list[dict]) -> list[tuple[dict, ...]]:
        """round-robin: C(n,2) pairs (2p) or C(n,4) 4-tuples (4p).
        gauntlet: challenger × each opponent (2p), or challenger + C(n-1,3) (4p)."""
        if self.config.shape == "gauntlet":
            return self._generate_gauntlet_pairs(agents)
        if self.config.format == "2p":
            return list(itertools.combinations(agents, 2))
        # 4p round-robin
        if len(agents) < 4:
            raise ValueError(f"4p format needs ≥4 agents, got {len(agents)}")
        return list(itertools.combinations(agents, 4))

    def _generate_gauntlet_pairs(self, agents: list[dict]) -> list[tuple[dict, ...]]:
        cid = self.config.challenger_id
        if cid is None:
            raise ValueError("gauntlet requires challenger_id")
        challenger = next((a for a in agents if a["id"] == cid), None)
        if challenger is None:
            raise ValueError(f"challenger {cid!r} not in selected agents")
        opponents = [a for a in agents if a["id"] != cid]
        if self.config.format == "2p":
            if not opponents:
                raise ValueError("gauntlet needs ≥1 opponent")
            return [(challenger, opp) for opp in opponents]
        # 4p gauntlet: challenger + every 3-tuple of opponents
        if len(opponents) < 3:
            raise ValueError(f"4p gauntlet needs ≥3 opponents, got {len(opponents)}")
        return [(challenger,) + triple for triple in itertools.combinations(opponents, 3)]

    def _build_summary(self, matches: list[MatchResult]) -> dict:
        agent_stats: dict[str, dict] = {}
        total_duration = 0.0
        for m in matches:
            total_duration += m.duration_s
            for aid in m.agent_ids:
                stats = agent_stats.setdefault(
                    aid, {"wins": 0, "losses": 0, "draws": 0}
                )
                if m.winner is None:
                    stats["draws"] += 1
                elif m.winner == aid:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
        return {
            "total_matches": len(matches),
            "total_duration_s": round(total_duration, 3),
            "agent_stats": agent_stats,
        }


# =========================================================================
# CLI
# =========================================================================


def _default_runs_dir() -> Path:
    return Path(os.environ.get("ORBIT_WARS_RUNS_DIR", "runs"))


def _default_zoo_dir() -> Path:
    return Path(os.environ.get("ORBIT_WARS_ZOO_DIR", "agents"))


def _cmd_list(args):
    zoo = scan_zoo(args.zoo)
    store = TrueSkillStore(args.runs / "trueskill.json")
    lb_2p = {r.agent_id: r for r in store.leaderboard(format="2p")}

    print(f"{'ID':<40}  {'BUCKET':<12}  {'μ':>6}  {'σ':>6}  {'N':>4}  TAGS")
    print("-" * 100)
    for a in zoo:
        r = lb_2p.get(a.id)
        mu = f"{r.mu:.0f}" if r else "-"
        sigma = f"{r.sigma:.0f}" if r else "-"
        games = str(r.games_played) if r else "0"
        tags = ",".join(a.tags) if a.tags else ""
        marker = " [disabled]" if a.disabled else ""
        print(f"{a.id:<40}  {a.bucket:<12}  {mu:>6}  {sigma:>6}  {games:>4}  {tags}{marker}")


def _cmd_show(args):
    zoo = scan_zoo(args.zoo)
    match = next((a for a in zoo if a.id == args.agent_id), None)
    if match is None:
        print(f"Agent {args.agent_id!r} not found in {args.zoo}", file=sys.stderr)
        sys.exit(1)
    print(f"ID:          {match.id}")
    print(f"Name:        {match.name}")
    print(f"Bucket:      {match.bucket}")
    print(f"Path:        {match.path}")
    print(f"Description: {match.description or '-'}")
    print(f"Author:      {match.author or '-'}")
    print(f"Kernel:      {match.kernel_slug or '-'}")
    print(f"Version:     {match.kernel_version if match.kernel_version else '-'}")
    print(f"License:     {match.license or '-'}")
    print(f"LB claim:    {match.author_claimed_lb_score if match.author_claimed_lb_score else '-'}")
    print(f"Fetched:     {match.date_fetched or '-'}")
    if match.source_url:
        print(f"Source URL (DEPRECATED): {match.source_url}")
    if match.version:
        print(f"Version (DEPRECATED): {match.version}")
    print(f"Tags:        {', '.join(match.tags) if match.tags else '-'}")
    print(f"Disabled:    {match.disabled}")
    if match.last_error:
        print(f"Last error:  {match.last_error}")

    store = TrueSkillStore(args.runs / "trueskill.json")
    for fmt in ("2p", "4p"):
        r = store.get_rating(match.id, format=fmt)  # type: ignore[arg-type]
        print(f"Rating {fmt}:   μ={r.mu:.1f}  σ={r.sigma:.1f}  games={r.games_played}")


def _cmd_run(args):
    zoo = scan_zoo(args.zoo)
    agents = args.agents

    if agents:
        pass  # explicit list used as-is
    elif args.bucket:
        buckets = set(args.bucket.split(","))
        filtered = [a for a in zoo if a.bucket in buckets]
        filtered = filter_agents_by_tags(filtered, include=args.tag, exclude=args.exclude_tag)
        agents = [a.id for a in filtered]
    elif args.tag or args.exclude_tag:
        filtered = filter_agents_by_tags(zoo, include=args.tag, exclude=args.exclude_tag)
        agents = [a.id for a in filtered]
    else:
        agents = [a.id for a in zoo if not a.disabled]

    if not agents:
        print("No agents selected (check --agents / --bucket / --tag / --exclude-tag)", file=sys.stderr)
        sys.exit(1)
    min_agents = 4 if args.format == "4p" else 2
    if len(agents) < min_agents:
        print(
            f"Format {args.format} needs ≥{min_agents} agents, got {len(agents)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.parallel > 1 and args.mode not in ("fast", "ultrafast", "value"):
        print(
            "--parallel >1 only supported in fast/ultrafast/value modes; falling back to sequential.",
            file=sys.stderr,
        )
        args.parallel = 1

    args.runs.mkdir(parents=True, exist_ok=True)
    cfg = TournamentConfig(
        agents=agents,
        games_per_pair=args.games_per_pair,
        mode=args.mode,
        format=args.format,
        parallel=args.parallel,
        seed_base=args.seed,
        save_replays=not args.no_replays,
        value_model_path=args.value_model_path,
    )
    t = Tournament(config=cfg, runs_root=args.runs, zoo_root=args.zoo)
    run_id = t.run()
    print(f"Run {run_id} completed → {args.runs / run_id}")


def _cmd_gauntlet(args):
    zoo = scan_zoo(args.zoo)
    challenger_id = args.challenger
    if not any(a.id == challenger_id and not a.disabled for a in zoo):
        print(f"Challenger {challenger_id!r} not found or disabled in zoo", file=sys.stderr)
        sys.exit(1)

    if args.agents:
        opponents = [aid for aid in args.agents if aid != challenger_id]
    elif args.bucket:
        buckets = [b.strip() for b in args.bucket.split(",") if b.strip()]
        filtered = [a for a in zoo if not a.disabled and a.bucket in buckets]
        filtered = filter_agents_by_tags(filtered, include=args.tag, exclude=args.exclude_tag)
        opponents = [a.id for a in filtered if a.id != challenger_id]
    elif args.tag or args.exclude_tag:
        filtered = filter_agents_by_tags(zoo, include=args.tag, exclude=args.exclude_tag)
        opponents = [a.id for a in filtered if a.id != challenger_id and not a.disabled]
    else:
        opponents = [a.id for a in zoo if not a.disabled and a.id != challenger_id]

    if not opponents:
        print("No opponents selected (check --agents / --bucket / --tag)", file=sys.stderr)
        sys.exit(1)
    min_opponents = 3 if args.format == "4p" else 1
    if len(opponents) < min_opponents:
        print(f"Format {args.format} gauntlet needs ≥{min_opponents} opponents, got {len(opponents)}",
              file=sys.stderr)
        sys.exit(1)

    args.runs.mkdir(parents=True, exist_ok=True)
    cfg = TournamentConfig(
        agents=[challenger_id] + opponents,
        games_per_pair=args.games_per_pair,
        mode=args.mode,
        format=args.format,
        seed_base=args.seed,
        shape="gauntlet",
        challenger_id=challenger_id,
    )
    t = Tournament(config=cfg, runs_root=args.runs, zoo_root=args.zoo)
    run_id = t.run()
    print(f"Gauntlet {challenger_id} vs {len(opponents)} opponents: run {run_id}")


def _cmd_head_to_head(args):
    args.runs.mkdir(parents=True, exist_ok=True)
    cfg = TournamentConfig(
        agents=[args.agent_a, args.agent_b],
        games_per_pair=args.games,
        mode=args.mode,
        format="2p",
        seed_base=args.seed,
    )
    t = Tournament(config=cfg, runs_root=args.runs, zoo_root=args.zoo)
    run_id = t.run()
    print(f"Head-to-head {args.agent_a} vs {args.agent_b}: run {run_id}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m orbit_wars_app.tournament",
        description="Orbit Wars Lab — local tournament runner",
    )
    parser.add_argument("--zoo", type=Path, default=_default_zoo_dir())
    parser.add_argument("--runs", type=Path, default=_default_runs_dir())
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show zoo + TrueSkill")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show agent details")
    p_show.add_argument("agent_id")
    p_show.set_defaults(func=_cmd_show)

    p_run = sub.add_parser("run", help="Run a tournament")
    p_run.add_argument("--agents", nargs="*", default=[], help="Explicit agent IDs")
    p_run.add_argument("--bucket", default="", help="Comma-separated buckets (baselines,external,mine)")
    p_run.add_argument(
        "--tag", action="append", default=[],
        help="Include agents with this tag (repeatable = OR). "
             "Example: --tag benchmark --tag quick → benchmark OR quick",
    )
    p_run.add_argument(
        "--exclude-tag", action="append", default=[], dest="exclude_tag",
        help="Exclude agents with this tag (repeatable = AND). "
             "Example: --exclude-tag broken --exclude-tag slow",
    )
    p_run.add_argument("--games-per-pair", type=int, default=3, help="K games per pair (default 3)")
    p_run.add_argument("--mode", choices=["fast", "faithful", "ultrafast", "value"], default="fast",
                       help="fast=in-process kaggle-envs, faithful=subprocess+HTTP "
                            "(Kaggle protocol), ultrafast=native Rust engine (no replays), "
                            "value=native Rust engine with replay + XGBoost value trace")
    p_run.add_argument("--format", choices=["2p", "4p"], default="2p",
                       help="Match format — 2-player or 4-player FFA (default 2p)")
    p_run.add_argument("--parallel", type=int, default=1,
                       help="Concurrent matches for this CLI run (fast/ultrafast modes)")
    p_run.add_argument("--seed", type=int, default=42, help="Base seed for match randomness")
    p_run.add_argument("--no-replays", action="store_true", dest="no_replays",
                       help="Skip writing per-match replay JSON (5-10MB each); ratings still computed")
    p_run.add_argument("--value-model-path", default=str(default_value_model_path()),
                       help="XGBoost model path for --mode value")
    p_run.set_defaults(func=_cmd_run)

    p_g = sub.add_parser("gauntlet", help="One challenger vs every other agent (× K games)")
    p_g.add_argument("challenger", help="Challenger agent ID (e.g. mine/v1-my-bot)")
    p_g.add_argument("--agents", nargs="*", default=[], help="Explicit opponent IDs (excludes challenger)")
    p_g.add_argument("--bucket", default="", help="Comma-separated buckets for opponents")
    p_g.add_argument("--tag", action="append", default=[], help="Include opponents with this tag")
    p_g.add_argument("--exclude-tag", action="append", default=[], dest="exclude_tag",
                     help="Exclude opponents with this tag")
    p_g.add_argument("--games-per-pair", type=int, default=10, help="K games per opponent (default 10)")
    p_g.add_argument("--mode", choices=["fast", "faithful", "ultrafast", "value"], default="fast")
    p_g.add_argument("--format", choices=["2p", "4p"], default="2p",
                     help="2p: challenger vs 1 opponent. 4p: challenger + 3 opponents per match.")
    p_g.add_argument("--seed", type=int, default=42)
    p_g.set_defaults(func=_cmd_gauntlet)

    p_h2h = sub.add_parser("head-to-head", help="N games between exactly two agents (always 2p)")
    p_h2h.add_argument("agent_a", help="First agent ID (player 0)")
    p_h2h.add_argument("agent_b", help="Second agent ID (player 1)")
    p_h2h.add_argument("--games", type=int, default=10, help="Number of games (default 10)")
    p_h2h.add_argument("--mode", choices=["fast", "faithful", "ultrafast", "value"], default="fast",
                       help="fast=in-process, faithful=subprocess+HTTP, "
                            "ultrafast=native Rust engine (no replays), "
                            "value=native Rust engine with replay + XGBoost value trace")
    p_h2h.add_argument("--seed", type=int, default=42, help="Base seed")
    p_h2h.set_defaults(func=_cmd_head_to_head)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
