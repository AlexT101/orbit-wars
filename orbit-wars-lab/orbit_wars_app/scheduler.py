"""Global match scheduler.

A single process-wide `Scheduler` owns a queue of matches drawn from any number
of concurrently-running tournaments and executes them across a killable pool of
worker processes (pebble), sized by one system-wide concurrency setting.

Design notes
------------
* **Fair round-robin.** Each tournament keeps its own FIFO sub-queue; the
  dispatcher cycles one match from each active tournament in turn, so a large
  round-robin can't starve a Quick Match queued behind it.
* **Hard-kill.** pebble gives per-match `timeout` (terminates + respawns the
  worker) and `future.cancel()` on a *running* task (kills the worker). Stopping
  a tournament drops its queued matches and cancels its in-flight ones.
* **One process per pool slot, reused.** Workers stay warm across matches so
  fast-mode doesn't re-pay the ~150 MB `kaggle-environments` import each match.
* **Shared global stores.** A single `TrueSkillStore` + `RuntimeStore` are
  updated by every match under a lock — concurrent tournaments write the same
  global `runs/trueskill.json` / `runtimes.json`.

Locking discipline (no nesting → no deadlock):
* `self._lock` (RLock) guards the registry, per-tournament queues, the running
  map, the RR cursor, the pool, and the global stores.
* `ts.lock` guards one tournament's results list and its on-disk files.
Code acquires at most one of these at a time; the only ordering is
"`self._lock` released before touching `ts.lock`".
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pebble import ProcessExpired, ProcessPool

from .match import run_match, silence_kaggle_environments_logging
from .pairing import generate_pairs, resolve_agents
from .replay_store import save_replay
from .runtime_store import RuntimeStore
from .schemas import MatchResult, RunStatus, TournamentConfig
from .trueskill_store import TrueSkillStore
from .value_evaluator import validate_value_model_path


# Status values a tournament can be in. "queued" = registered, no match has
# started yet; "running" = at least one match dispatched; terminal states match
# the on-disk RunStatus.
TournamentStatus = str  # queued | running | completed | aborted
_COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def match_timeout_for(num_players: int) -> float:
    """Per-match wall-clock deadline (seconds), derived from the player count.

    Budget = 500 turns × ~1s/turn + 60s overage allowance per player + 20s slack,
    i.e. ``(500 + 60) * num_players + 20``. So a 2p match gets 1140s and a 4p
    match 2260s. Not user-configurable — it tracks the engine's own deadline.
    """
    return (500 + 60) * max(1, num_players) + 20


def side_order_for_seed(seed: int, num_players: int) -> list[int]:
    """Deterministically map a match seed to engine player slots."""
    order = list(range(num_players))
    if num_players == 2:
        return [1, 0] if seed % 2 else order
    rng = random.Random(seed)
    rng.shuffle(order)
    return order


def allocate_run_id(runs_root: Path, extra_taken: tuple[str, ...] = ()) -> str:
    """Return the next `YYYY-MM-DD-NNN` id (N = max existing today + 1).

    Scans `runs_root` for today's run dirs and also considers `extra_taken`
    (in-memory ids not yet on disk). Max-of-existing (not count) so deleting a
    middle run doesn't cause a later collision.
    """
    now = datetime.now(timezone.utc)
    prefix = now.strftime("%Y-%m-%d")
    max_n = 0
    if runs_root.is_dir():
        for p in runs_root.iterdir():
            if not p.is_dir() or not p.name.startswith(prefix + "-"):
                continue
            try:
                max_n = max(max_n, int(p.name[len(prefix) + 1 :]))
            except ValueError:
                continue
    for rid in extra_taken:
        if rid.startswith(prefix + "-"):
            try:
                max_n = max(max_n, int(rid[len(prefix) + 1 :]))
            except ValueError:
                continue
    return f"{prefix}-{max_n + 1:03d}"


def _quiet_kaggle_environments() -> None:
    """Silence kaggle-environments' chatty import-time logging in workers.

    Runs once per worker via the pool `initializer` (loggers are process-wide
    and persist across pebble worker reuse), so the filter is in place before
    the worker's first `import kaggle_environments`. See
    `match.silence_kaggle_environments_logging` for why a plain parent-level
    bump is not enough.
    """
    silence_kaggle_environments_logging()


# ============================================================
# Pickle-friendly job + result carriers (cross the process boundary)
# ============================================================


@dataclass
class QueuedMatch:
    """One match waiting to run. All fields are pickle-safe (str/int/list/bool)
    so the dataclass ships cleanly to a spawn-based pebble worker."""

    run_id: str
    match_counter: int
    agent_ids: list[str]
    agent_paths: list[str]
    mode: str
    seed: int
    save_replays: bool
    replays_dir: str
    logs_dir: str
    replay_map: Optional[dict] = None
    value_model_path: Optional[str] = None


@dataclass
class MatchJobResult:
    """What a worker returns to the scheduler — metadata only, no replay dict.

    The replay (5–10 MB) is written to disk inside the worker; we ship back just
    the relative path. `per_agent_turn_seconds` carries the per-turn timing
    samples to fold into the RuntimeStore. `error` surfaces a non-ok reason.
    """

    match_counter: int
    agent_ids: list[str]
    winner: Optional[str]
    scores: list[float]
    turns: int
    duration_s: float
    seed: int
    status: str
    replay_path: str
    worker_pid: int = 0
    per_agent_turn_seconds: list[list[float]] = field(default_factory=list)
    error: Optional[str] = None


def _execute_match_job(job: QueuedMatch) -> MatchJobResult:
    """Worker entrypoint: run one match and persist its replay locally.

    Top-level (picklable) so it survives the spawn boundary on Windows. Mirrors
    the lightweight-return contract of the old `_run_match_in_worker`.
    """
    outcome = run_match(
        agent_ids=job.agent_ids,
        agent_paths=[Path(p) for p in job.agent_paths],
        mode=job.mode,  # type: ignore[arg-type]
        seed=job.seed,
        log_dir=Path(job.logs_dir) if job.logs_dir else None,
        log_prefix=f"{job.match_counter:03d}",
        replay_map=job.replay_map,
        value_model_path=job.value_model_path,
    )

    replay_rel = ""
    if (
        job.save_replays
        and job.replays_dir
        and outcome.replay
        and "steps" in outcome.replay
    ):
        replays_dir = Path(job.replays_dir)
        rp = save_replay(replays_dir, job.match_counter, job.agent_ids, outcome.replay)
        # Relative to the run dir (replays_dir is <run_dir>/replays).
        replay_rel = str(rp.relative_to(replays_dir.parent))

    err: Optional[str] = None
    if isinstance(outcome.replay, dict):
        e = outcome.replay.get("error")
        if isinstance(e, str) and e:
            err = e

    return MatchJobResult(
        match_counter=job.match_counter,
        agent_ids=job.agent_ids,
        winner=outcome.winner,
        scores=outcome.scores,
        turns=outcome.turns,
        duration_s=outcome.duration_s,
        seed=job.seed,
        status=outcome.status,
        replay_path=replay_rel,
        worker_pid=os.getpid(),
        per_agent_turn_seconds=outcome.per_agent_turn_seconds,
        error=err,
    )


# ============================================================
# In-memory descriptors
# ============================================================


@dataclass
class RunningMatch:
    """A match currently occupying a pool slot. `started_mono` drives the
    elapsed-time readout in the introspection API."""

    run_id: str
    match_counter: int
    agent_ids: list[str]
    mode: str
    seed: int
    started_mono: float
    started_at: str
    # The originating job, so a pool restart can re-queue an in-flight match
    # instead of losing it.
    job: Optional["QueuedMatch"] = None

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "match_id": f"{self.match_counter:03d}",
            "agent_ids": list(self.agent_ids),
            "mode": self.mode,
            "started_at": self.started_at,
            "elapsed_s": round(time.monotonic() - self.started_mono, 2),
        }


# Throttle policy for in-progress run.json writes (matches the old Tournament).
_RUN_JSON_THROTTLE_MATCHES = 10
_RUN_JSON_THROTTLE_SECONDS = 1.0


class TournamentState:
    """Live state + on-disk writer for one tournament. Guarded by `self.lock`
    for everything touching `results` or the run directory's JSON files."""

    def __init__(
        self,
        *,
        run_id: str,
        config: TournamentConfig,
        run_dir: Path,
        replays_dir: Path,
        logs_dir: Path,
        pending: deque[QueuedMatch],
        total_matches: int,
        started_at: str,
    ):
        self.id = run_id
        self.config = config
        self.run_dir = run_dir
        self.replays_dir = replays_dir
        self.logs_dir = logs_dir
        self.pending = pending
        self.total_matches = total_matches
        self.started_at = started_at
        self.finished_at: Optional[str] = None
        self.status: TournamentStatus = "queued"
        self.results: list[MatchResult] = []
        self.completed_count = 0
        self.running: set = set()  # set[ProcessFuture]
        self.stop_requested = False
        # Optional per-match callback (match_result, matches_done, total). Fired
        # from a pool-manager thread after each match is recorded — used by the
        # CLI shim for progress; the API path leaves it None.
        self.on_match_done = None
        self.lock = threading.Lock()
        # Finalize guards (flipped under the scheduler lock).
        self._finalizing = False
        self._finalized = False
        # run.json write throttle state.
        self._run_json_last_write_t = 0.0
        self._run_json_last_done = -1

    # ---- introspection ----

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "mode": self.config.mode,
            "format": self.config.format,
            "shape": self.config.shape,
            "challenger_id": self.config.challenger_id,
            "is_quick_match": self.config.is_quick_match,
            "total_matches": self.total_matches,
            "matches_done": self.completed_count,
            "queued": len(self.pending),
            "running": len(self.running),
            "started_at": self.started_at,
        }

    # ---- on-disk writers (call under self.lock) ----

    def write_config(self) -> None:
        replay_map = (
            self.config.replay_map.model_dump()
            if self.config.replay_map is not None
            else None
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "mode": self.config.mode,
                    "format": self.config.format,
                    "games_per_pair": self.config.games_per_pair,
                    "agents": self.config.agents,
                    "seed_base": self.config.seed_base,
                    "seed_mode": self.config.seed_mode,
                    "replay_map": replay_map,
                    "value_model_path": self.config.value_model_path,
                    "save_replays": self.config.save_replays,
                    "shape": self.config.shape,
                    "challenger_id": self.config.challenger_id,
                    "is_quick_match": self.config.is_quick_match,
                    "started_at": self.started_at,
                },
                indent=2,
            )
        )

    def write_run_json(
        self,
        *,
        status: RunStatus,
        matches_done: int,
        finished_at: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Atomic run.json write, throttled while still running.

        Terminal writes and the first/last match always go through so the
        progress bar never looks stuck.
        """
        if not force and status == "running" and matches_done not in (0, self.total_matches):
            now = time.monotonic()
            since_last = matches_done - self._run_json_last_done
            elapsed = now - self._run_json_last_write_t
            if (
                since_last < _RUN_JSON_THROTTLE_MATCHES
                and elapsed < _RUN_JSON_THROTTLE_SECONDS
            ):
                return
            self._run_json_last_write_t = now
            self._run_json_last_done = matches_done
        payload = {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "mode": self.config.mode,
            "format": self.config.format,
            "status": status,
            "total_matches": self.total_matches,
            "matches_done": matches_done,
            "is_quick_match": self.config.is_quick_match,
        }
        target = self.run_dir / "run.json"
        tmp = self.run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(target)

    def build_summary(self) -> dict:
        agent_stats: dict[str, dict] = {}
        total_duration = 0.0
        for m in self.results:
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
            "total_matches": len(self.results),
            "total_duration_s": round(total_duration, 3),
            "agent_stats": agent_stats,
        }

    def write_results_json(self, status: RunStatus) -> None:
        # Parallel completion is out-of-order; sort by match_id so results.json
        # reads naturally and the head-to-head matrix stays consistent.
        self.results.sort(key=lambda m: m.match_id)
        (self.run_dir / "results.json").write_text(
            json.dumps(
                {
                    "started_at": self.started_at,
                    "finished_at": self.finished_at,
                    "total_matches": self.total_matches,
                    "matches": [m.model_dump() for m in self.results],
                    "summary": self.build_summary(),
                    "status": status,
                },
                indent=2,
            )
        )


class Scheduler:
    """Process-wide match scheduler. Construct once, `start()`, then `submit()`
    tournaments. Thread-safe."""

    DEFAULT_CONCURRENCY = 8

    def __init__(
        self,
        *,
        runs_root: Path,
        zoo_root: Path,
        concurrency: int = DEFAULT_CONCURRENCY,
        match_timeout_s: Optional[float] = None,
        job_fn: Callable[[QueuedMatch], MatchJobResult] = _execute_match_job,
    ):
        self._runs_root = runs_root
        self._zoo_root = zoo_root
        self._concurrency = max(1, int(concurrency))
        # Optional fixed override (mainly for tests). When None, each match gets
        # a deadline derived from its player count via `match_timeout_for`.
        self._match_timeout_override = match_timeout_s
        self._job_fn = job_fn

        self._lock = threading.RLock()
        self._tournaments: dict[str, TournamentState] = {}
        self._order: list[str] = []  # active run_ids, RR order
        self._rr_index = 0
        self._running: dict[object, RunningMatch] = {}  # future -> RunningMatch

        self._pool: Optional[ProcessPool] = None
        self._pool_size = 0

        # Shared global stores — every match feeds these under self._lock.
        self._trueskill = TrueSkillStore(runs_root / "trueskill.json")
        self._runtime = RuntimeStore(runs_root / "runtimes.json")

    # ---- lifecycle ----

    def start(self) -> None:
        with self._lock:
            if self._pool is None:
                self._open_pool_locked()

    def shutdown(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.stop()
            try:
                pool.join(timeout=5)
            except Exception:
                pass

    def _open_pool_locked(self) -> None:
        self._pool = ProcessPool(
            max_workers=self._concurrency,
            initializer=_quiet_kaggle_environments,
        )
        self._pool_size = self._concurrency

    def _recreate_pool_locked(self) -> None:
        old = self._pool
        self._open_pool_locked()
        if old is not None:
            # Caller guarantees no matches are tracked against the old pool; stop
            # it in the background so we don't block on worker teardown.
            threading.Thread(target=old.stop, daemon=True).start()

    def _timeout_for(self, num_players: int) -> float:
        """Per-match deadline: the fixed override if set, else the player-count
        formula."""
        if self._match_timeout_override is not None:
            return self._match_timeout_override
        return match_timeout_for(num_players)

    def restart_pool(self) -> None:
        """Recreate the worker pool, recycling all workers.

        Warm workers cache imported bot modules — including native extensions
        (`.so`/`.pyd`), which can't be hot-reloaded in-process. Restarting the
        pool is how a freshly-rebuilt bot binary gets picked up. Any in-flight
        matches are re-queued to the front of their tournament so the fresh
        workers re-run them (with the new code) rather than losing them.
        """
        with self._lock:
            for future, rm in list(self._running.items()):
                ts = self._tournaments.get(rm.run_id)
                if ts is not None and rm.job is not None and not ts.stop_requested:
                    ts.pending.appendleft(rm.job)
                    ts.running.discard(future)
                self._running.pop(future, None)
                future.cancel()  # kills the worker mid-match
            if self._pool is None:
                self._open_pool_locked()
            else:
                self._recreate_pool_locked()
            self._dispatch_locked()

    # ---- submission / control ----

    def submit(self, config: TournamentConfig, on_match_done=None) -> str:
        """Expand `config` into queued matches and register the tournament.

        Returns the run_id. Raises ValueError for a bad config (unknown/disabled
        agent, too few agents for the format) — surfaced before any run dir is
        created, so a bad request never leaves an orphan directory.

        `on_match_done(match_result, matches_done, total)` (optional) fires after
        each match is recorded, for progress streaming (CLI).
        """
        self._validate_replay_map_config(config)
        if config.mode == "value":
            if config.format != "2p":
                raise ValueError("value mode currently supports 2p XGBoost models only")
            config.value_model_path = str(validate_value_model_path(config.value_model_path))
        agents = resolve_agents(config, self._zoo_root)
        pairs = generate_pairs(config, agents)
        total = len(pairs) * config.games_per_pair
        replay_map = (
            config.replay_map.model_dump()
            if config.replay_map is not None
            else None
        )

        with self._lock:
            run_id = self._alloc_run_id_locked()
            run_dir = self._runs_root / run_id
            run_dir.mkdir(parents=True)
            replays_dir = run_dir / "replays"
            replays_dir.mkdir()
            logs_dir = run_dir / "logs"
            logs_dir.mkdir()

            # Deterministic per-match seeds, assigned in pair-iteration order so
            # outcomes don't depend on dispatch interleaving.
            rng = random.Random(config.seed_base)
            pending: deque[QueuedMatch] = deque()
            mc = 0
            for pair in pairs:
                for _ in range(config.games_per_pair):
                    mc += 1
                    seed = rng.randrange(10**9)
                    match_agents = list(pair)
                    if not config.is_quick_match:
                        match_agents = [
                            match_agents[i]
                            for i in side_order_for_seed(seed, len(match_agents))
                        ]
                    pending.append(
                        QueuedMatch(
                            run_id=run_id,
                            match_counter=mc,
                            agent_ids=[a["id"] for a in match_agents],
                            agent_paths=[
                                str(self._zoo_root.parent / a["path"])
                                for a in match_agents
                            ],
                            mode=config.mode,
                            seed=seed,
                            save_replays=config.save_replays,
                            replays_dir=str(replays_dir),
                            logs_dir=str(logs_dir),
                            replay_map=replay_map,
                            value_model_path=config.value_model_path,
                        )
                    )

            ts = TournamentState(
                run_id=run_id,
                config=config,
                run_dir=run_dir,
                replays_dir=replays_dir,
                logs_dir=logs_dir,
                pending=pending,
                total_matches=total,
                started_at=_now_iso(),
            )
            ts.on_match_done = on_match_done
            self._tournaments[run_id] = ts
            self._order.append(run_id)
            ts.write_config()
            ts.write_run_json(status="queued", matches_done=0, force=True)
            # Empty tournament (0 matches) finalizes immediately.
            if not pending:
                self._maybe_finalize_locked(ts)
            self._dispatch_locked()

        # An empty tournament was marked finalizing above; flush it outside lock.
        with self._lock:
            empty_done = ts._finalizing and not ts._finalized
        if empty_done:
            self._finalize(ts)
        return run_id

    def _validate_replay_map_config(self, config: TournamentConfig) -> None:
        if config.replay_map is None:
            if config.seed_mode == "replay":
                raise ValueError("seed_mode='replay' requires replay_map")
            return
        if config.seed_mode != "replay":
            raise ValueError("replay_map requires seed_mode='replay'")
        if config.mode == "ultrafast":
            raise ValueError("replay maps are supported in fast and faithful modes only")
        if not config.replay_map.planets:
            raise ValueError("replay_map.planets must not be empty")
        if (
            config.replay_map.initial_planets
            and len(config.replay_map.initial_planets) != len(config.replay_map.planets)
        ):
            raise ValueError("replay_map.initial_planets must match planets length")
        self._validate_replay_comet_schedule(config.replay_map.comet_schedule)
        expected_players = 4 if config.format == "4p" else 2
        source_players = config.replay_map.num_players
        if source_players is not None and source_players != expected_players:
            raise ValueError(
                f"Replay has {source_players} players but tournament format is "
                f"{config.format}; choose a {expected_players}p replay or change format"
            )

    def _validate_replay_comet_schedule(self, schedule: list) -> None:
        seen_steps: set[int] = set()
        for group_idx, group in enumerate(schedule):
            spawn_step = int(group.spawn_step)
            if spawn_step not in _COMET_SPAWN_STEPS:
                raise ValueError(
                    "replay_map.comet_schedule entries must use spawn steps "
                    "50, 150, 250, 350, or 450"
                )
            if spawn_step in seen_steps:
                raise ValueError(
                    f"replay_map.comet_schedule has duplicate spawn step {spawn_step}"
                )
            seen_steps.add(spawn_step)
            if not math.isfinite(float(group.ships)) or float(group.ships) <= 0:
                raise ValueError(
                    f"replay_map.comet_schedule[{group_idx}].ships must be positive"
                )
            if len(group.paths) != 4:
                raise ValueError(
                    f"replay_map.comet_schedule[{group_idx}].paths must contain 4 paths"
                )
            for path_idx, path in enumerate(group.paths):
                if not path:
                    raise ValueError(
                        f"replay_map.comet_schedule[{group_idx}].paths[{path_idx}] "
                        "must not be empty"
                    )
                for point_idx, point in enumerate(path):
                    if len(point) < 2:
                        raise ValueError(
                            f"replay_map.comet_schedule[{group_idx}].paths"
                            f"[{path_idx}][{point_idx}] must contain x/y"
                        )
                    x = float(point[0])
                    y = float(point[1])
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError(
                            f"replay_map.comet_schedule[{group_idx}].paths"
                            f"[{path_idx}][{point_idx}] must contain finite x/y"
                        )

    def stop(self, run_id: str) -> bool:
        """Drop a tournament's queued matches and kill its in-flight ones.

        Returns False if the tournament is unknown or already terminal.
        """
        with self._lock:
            ts = self._tournaments.get(run_id)
            if ts is None or ts.status in ("completed", "aborted"):
                return False
            ts.status = "aborted"
            ts.stop_requested = True
            ts.pending.clear()
            running = list(ts.running)

        for fut in running:
            fut.cancel()  # kills the worker if the match is already running

        # If nothing was in flight, no cancel callback will fire — finalize here.
        to_finalize = None
        with self._lock:
            if self._maybe_finalize_locked(ts):
                to_finalize = ts
            self._dispatch_locked()
        if to_finalize is not None:
            self._finalize(to_finalize)
        return True

    def set_concurrency(self, n: int) -> int:
        n = max(1, int(n))
        with self._lock:
            self._concurrency = n
            # Growing the pool requires recreation (pebble pre-spawns all
            # workers); only safe when idle. Shrinking takes effect immediately
            # via the dispatch gate below — surplus warm workers idle until the
            # next recreate.
            if self._pool is not None and not self._running and n != self._pool_size:
                self._recreate_pool_locked()
            self._dispatch_locked()
        return self._concurrency

    @property
    def runs_root(self) -> Path:
        return self._runs_root

    @property
    def zoo_root(self) -> Path:
        return self._zoo_root

    @property
    def concurrency(self) -> int:
        return self._concurrency

    # ---- introspection ----

    def status(self) -> dict:
        with self._lock:
            tournaments = [
                self._tournaments[rid].summary()
                for rid in self._order
                if rid in self._tournaments
            ]
            queued_total = sum(
                len(self._tournaments[rid].pending)
                for rid in self._order
                if rid in self._tournaments
            )
            running = [rm.summary() for rm in self._running.values()]
        return {
            "concurrency": self._concurrency,
            "running_count": len(running),
            "queued_total": queued_total,
            "tournaments": tournaments,
            "running": running,
        }

    def running_matches(self) -> list[dict]:
        with self._lock:
            return [rm.summary() for rm in self._running.values()]

    def tournament_progress(self, run_id: str) -> Optional[dict]:
        """In-memory progress for a live tournament, or None if not tracked
        (caller falls back to reading run.json off disk)."""
        with self._lock:
            ts = self._tournaments.get(run_id)
            if ts is None:
                return None
            return {
                "status": ts.status,
                "matches_done": ts.completed_count,
                "total_matches": ts.total_matches,
            }

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            ts = self._tournaments.get(run_id)
            return ts is not None and not ts._finalized

    def active_run_ids(self) -> list[str]:
        with self._lock:
            return [
                rid
                for rid in self._order
                if rid in self._tournaments and not self._tournaments[rid]._finalized
            ]

    # ---- run-id allocation (under self._lock) ----

    def _alloc_run_id_locked(self) -> str:
        """Next `YYYY-MM-DD-NNN`. Atomic w.r.t. concurrent submits because the
        caller holds self._lock and mkdir follows immediately. Also considers
        in-memory ids not yet on disk."""
        return allocate_run_id(self._runs_root, extra_taken=tuple(self._tournaments))

    # ---- dispatch (under self._lock) ----

    def _next_job_locked(self) -> Optional[tuple[TournamentState, QueuedMatch]]:
        n = len(self._order)
        for step in range(n):
            idx = (self._rr_index + step) % n
            run_id = self._order[idx]
            ts = self._tournaments.get(run_id)
            if ts is None or ts.status in ("aborted", "completed") or not ts.pending:
                continue
            job = ts.pending.popleft()
            self._rr_index = (idx + 1) % n
            return ts, job
        return None

    def _dispatch_locked(self) -> None:
        if self._pool is None:
            return
        limit = min(self._concurrency, self._pool_size)
        while len(self._running) < limit:
            nxt = self._next_job_locked()
            if nxt is None:
                break
            ts, job = nxt
            if ts.status == "queued":
                ts.status = "running"
            future = self._pool.schedule(
                self._job_fn, args=[job], timeout=self._timeout_for(len(job.agent_ids))
            )
            rm = RunningMatch(
                run_id=job.run_id,
                match_counter=job.match_counter,
                agent_ids=list(job.agent_ids),
                mode=job.mode,
                seed=job.seed,
                started_mono=time.monotonic(),
                started_at=_now_iso(),
                job=job,
            )
            self._running[future] = rm
            ts.running.add(future)
            future.add_done_callback(self._on_done)

    # ---- completion handling ----

    def _on_done(self, future) -> None:
        try:
            self._handle_done(future)
        except Exception as e:  # never let a callback exception break the pool
            try:
                (self._runs_root / "scheduler-error.log").open("a").write(
                    f"{_now_iso()} _on_done: {type(e).__name__}: {e}\n"
                )
            except OSError:
                pass

    def _handle_done(self, future) -> None:
        with self._lock:
            rm = self._running.pop(future, None)
            if rm is None:
                return
            ts = self._tournaments.get(rm.run_id)
            if ts is not None:
                ts.running.discard(future)

        result = self._classify(future, rm)
        if result is not None and ts is not None:
            self._record_result(ts, result)

        to_finalize = None
        with self._lock:
            if ts is not None and self._maybe_finalize_locked(ts):
                to_finalize = ts
            self._dispatch_locked()
        if to_finalize is not None:
            self._finalize(to_finalize)

    def _classify(self, future, rm: RunningMatch) -> Optional[MatchJobResult]:
        """Turn a finished future into a MatchJobResult, or None to drop it
        (a match cancelled by Stop is not recorded)."""
        try:
            return future.result()
        except CancelledError:
            return None  # stopped tournament — drop, don't record
        except FuturesTimeout:
            limit = self._timeout_for(len(rm.agent_ids))
            self._write_synth_log(rm, f"match exceeded {limit}s timeout")
            return self._synth_result(
                rm, status="timeout", error=f"exceeded {limit}s timeout"
            )
        except ProcessExpired as e:
            msg = f"worker process died: {e}"
            self._write_synth_log(rm, msg)
            return self._synth_result(rm, status="crashed", error=msg)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self._write_synth_log(rm, msg)
            return self._synth_result(rm, status="crashed", error=msg)

    def _synth_result(
        self, rm: RunningMatch, *, status: str, error: str
    ) -> MatchJobResult:
        return MatchJobResult(
            match_counter=rm.match_counter,
            agent_ids=list(rm.agent_ids),
            winner=None,
            scores=[],
            turns=0,
            duration_s=round(time.monotonic() - rm.started_mono, 3),
            seed=rm.seed,
            status=status,
            replay_path="",
            worker_pid=0,
            per_agent_turn_seconds=[],
            error=error,
        )

    def _write_synth_log(self, rm: RunningMatch, msg: str) -> None:
        ts = self._tournaments.get(rm.run_id)
        if ts is None:
            return
        try:
            ts.logs_dir.mkdir(parents=True, exist_ok=True)
            (ts.logs_dir / f"{rm.match_counter:03d}-scheduler.log").write_text(
                msg + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _record_result(self, ts: TournamentState, r: MatchJobResult) -> None:
        # Global stores under the scheduler lock (shared across tournaments).
        with self._lock:
            if r.status != "agent_failed_to_start":
                self._trueskill.update_match(
                    agent_ids=r.agent_ids,
                    winner=r.winner,
                    format=ts.config.format,
                    scores=r.scores,
                )
            for aid, samples in zip(r.agent_ids, r.per_agent_turn_seconds):
                if samples:
                    self._runtime.record(aid, samples)

        match_result = MatchResult(
            match_id=f"{r.match_counter:03d}",
            agent_ids=r.agent_ids,
            winner=r.winner,
            scores=r.scores,
            turns=r.turns,
            duration_s=r.duration_s,
            status=r.status,  # type: ignore[arg-type]
            seed=r.seed,
            replay_path=r.replay_path,
            error=r.error,
        )
        with ts.lock:
            ts.results.append(match_result)
            ts.completed_count += 1
            done = ts.completed_count
            ts.write_run_json(status="running", matches_done=done)
            cb = ts.on_match_done
            total = ts.total_matches
        if cb is not None:
            try:
                cb(match_result, done, total)
            except Exception:
                pass  # a buggy progress callback must not break the scheduler

    def _maybe_finalize_locked(self, ts: TournamentState) -> bool:
        """If a tournament has no pending or in-flight matches, claim the
        finalize (return True). The caller then runs `_finalize` outside the
        lock. Idempotent via `_finalizing`/`_finalized`."""
        if ts._finalized or ts._finalizing:
            return False
        if ts.pending or ts.running:
            return False
        ts._finalizing = True
        if ts.status != "aborted":
            ts.status = "completed"
        return True

    def _finalize(self, ts: TournamentState) -> None:
        """Write terminal artifacts + flush global stores. Runs outside the
        scheduler lock; only one caller ever reaches here per tournament."""
        status: RunStatus = "aborted" if ts.status == "aborted" else "completed"
        ts.finished_at = _now_iso()

        with self._lock:
            self._trueskill.save()
            self._runtime.save()

        with ts.lock:
            ts.write_results_json(status)
            ts.write_run_json(
                status=status,
                matches_done=ts.completed_count,
                finished_at=ts.finished_at,
                force=True,
            )
            self._snapshot_trueskill(ts)
            self._update_latest_pointer(ts)

        with self._lock:
            ts._finalized = True
            ts._finalizing = False
            # Drop from the active registry — the disk run dir is the archive.
            self._tournaments.pop(ts.id, None)
            if ts.id in self._order:
                self._order = [r for r in self._order if r != ts.id]
                if self._order:
                    self._rr_index %= len(self._order)
                else:
                    self._rr_index = 0

    def _snapshot_trueskill(self, ts: TournamentState) -> None:
        try:
            self._trueskill.snapshot_to(ts.run_dir / "trueskill.json")
        except OSError:
            pass

    def _update_latest_pointer(self, ts: TournamentState) -> None:
        latest = self._runs_root / "latest"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ts.id, target_is_directory=True)
        except (OSError, NotImplementedError):
            (self._runs_root / "latest.txt").write_text(ts.id)

    # ---- blocking helper (CLI) ----

    def run_blocking(
        self,
        config: TournamentConfig,
        *,
        on_match_done=None,
        cancel_event: "Optional[threading.Event]" = None,
        poll: float = 0.05,
    ) -> str:
        """Submit one tournament and block until it finalizes. For the CLI.

        If `cancel_event` is provided and gets set mid-run, the tournament is
        stopped (queued matches dropped, in-flight killed) and we return once it
        finalizes as aborted.
        """
        run_id = self.submit(config, on_match_done=on_match_done)
        stopped = False
        while self.is_active(run_id):
            if cancel_event is not None and cancel_event.is_set() and not stopped:
                self.stop(run_id)
                stopped = True
            time.sleep(poll)
        return run_id
