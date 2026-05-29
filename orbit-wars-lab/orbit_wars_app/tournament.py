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
from pathlib import Path
from typing import Callable, Optional

from .discovery import scan_zoo
from .pairing import filter_agents_by_tags
from .scheduler import Scheduler, allocate_run_id
from .schemas import MatchResult, TournamentConfig
from .trueskill_store import TrueSkillStore


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
            return sched.run_blocking(
                self.config,
                on_match_done=on_match_done,
                cancel_event=self.cancel_event,
            )
        finally:
            sched.shutdown()


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
    if args.parallel > 1 and args.mode not in ("fast", "ultrafast"):
        print(
            "--parallel >1 only supported in fast/ultrafast modes; falling back to sequential.",
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
    p_run.add_argument("--mode", choices=["fast", "faithful", "ultrafast"], default="fast",
                       help="fast=in-process kaggle-envs, faithful=subprocess+HTTP "
                            "(Kaggle protocol), ultrafast=native Rust engine (no replays)")
    p_run.add_argument("--format", choices=["2p", "4p"], default="2p",
                       help="Match format — 2-player or 4-player FFA (default 2p)")
    p_run.add_argument("--parallel", type=int, default=1,
                       help="Concurrent matches for this CLI run (fast/ultrafast modes)")
    p_run.add_argument("--seed", type=int, default=42, help="Base seed for match randomness")
    p_run.add_argument("--no-replays", action="store_true", dest="no_replays",
                       help="Skip writing per-match replay JSON (5-10MB each); ratings still computed")
    p_run.set_defaults(func=_cmd_run)

    p_g = sub.add_parser("gauntlet", help="One challenger vs every other agent (× K games)")
    p_g.add_argument("challenger", help="Challenger agent ID (e.g. mine/v1-my-bot)")
    p_g.add_argument("--agents", nargs="*", default=[], help="Explicit opponent IDs (excludes challenger)")
    p_g.add_argument("--bucket", default="", help="Comma-separated buckets for opponents")
    p_g.add_argument("--tag", action="append", default=[], help="Include opponents with this tag")
    p_g.add_argument("--exclude-tag", action="append", default=[], dest="exclude_tag",
                     help="Exclude opponents with this tag")
    p_g.add_argument("--games-per-pair", type=int, default=10, help="K games per opponent (default 10)")
    p_g.add_argument("--mode", choices=["fast", "faithful", "ultrafast"], default="fast")
    p_g.add_argument("--format", choices=["2p", "4p"], default="2p",
                     help="2p: challenger vs 1 opponent. 4p: challenger + 3 opponents per match.")
    p_g.add_argument("--seed", type=int, default=42)
    p_g.set_defaults(func=_cmd_gauntlet)

    p_h2h = sub.add_parser("head-to-head", help="N games between exactly two agents (always 2p)")
    p_h2h.add_argument("agent_a", help="First agent ID (player 0)")
    p_h2h.add_argument("agent_b", help="Second agent ID (player 1)")
    p_h2h.add_argument("--games", type=int, default=10, help="Number of games (default 10)")
    p_h2h.add_argument("--mode", choices=["fast", "faithful", "ultrafast"], default="fast",
                       help="fast=in-process, faithful=subprocess+HTTP, "
                            "ultrafast=native Rust engine (no replays)")
    p_h2h.add_argument("--seed", type=int, default=42, help="Base seed")
    p_h2h.set_defaults(func=_cmd_head_to_head)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
