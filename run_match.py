import argparse
import contextlib
import importlib.util
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"

MAX_STEPS = 500


@dataclass
class MatchResult:
    seed: int
    done: bool
    steps: int
    scores: list[int] | None
    rewards: list[float | None]
    avg_step_ms: list[float]
    statuses: list[str] | None = None


def scores_from_rows(planets: list[list[float]], fleets: list[list[float]], num_players: int) -> list[int]:
    scores = [0] * num_players
    for planet in planets:
        owner = int(planet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(planet[5])
    for fleet in fleets:
        owner = int(fleet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(fleet[6])
    return scores


@contextlib.contextmanager
def _silence_imports():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    logging.disable(logging.CRITICAL)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
        logging.disable(logging.NOTSET)


with _silence_imports():
    try:
        import kaggle_environments  # noqa: F401
    except ModuleNotFoundError:
        pass


def bot_entry(bot_name: str) -> Path:
    """
    Resolve a bot's main.py by searching:
    1. bots/<bot_name>/main.py
    2. bots/*/<bot_name>/main.py   (one layer deep)
    """

    # Direct path (top-level bot)
    direct = BOTS_DIR / bot_name / "main.py"
    if direct.is_file():
        return direct

    # Search one level deep
    for subdir in BOTS_DIR.iterdir():
        if not subdir.is_dir():
            continue

        candidate = subdir / bot_name / "main.py"
        if candidate.is_file():
            return candidate

    # Fallback (for error reporting consistency)
    return direct


def load_agent(main_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.agent


def run_rust_match(bot_paths: list[Path], bot_names: list[str], seed: int, game_index: int = 1) -> MatchResult:
    print(f"Match (rust engine): {bot_names[0]} vs {bot_names[1]}")

    from engine_parity_checker.candidates.rust import RustEngine

    agents = [load_agent(path, f"bot_{game_index}_{i}_main") for i, path in enumerate(bot_paths)]
    engine = RustEngine()
    obs = engine.reset(seed, len(agents))

    total_time = [0.0] * len(agents)
    call_counts = [0] * len(agents)

    done = False
    steps_run = 0
    for _ in range(MAX_STEPS):
        actions = []
        for i in range(len(agents)):
            start = perf_counter()
            action = agents[i](obs[i].as_dict())
            total_time[i] += perf_counter() - start
            call_counts[i] += 1
            actions.append(action)
        obs, done = engine.step(actions)
        steps_run += 1
        if done:
            break

    snap = engine.snapshot()
    print(f"Finished: done={done} steps={steps_run}")
    scores = scores_from_rows(snap.planets, snap.fleets, len(agents))
    if len(scores) == 2:
        print(f"Score: {scores[0]}-{scores[1]} diff={scores[0] - scores[1]:+d}")
    else:
        print("Score: " + ", ".join(str(score) for score in scores))
    rewards = snap.rewards if snap.rewards is not None else [None] * len(agents)
    avg_step_ms = []
    for i, reward in enumerate(rewards):
        avg_ms = (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        avg_step_ms.append(avg_ms)
        print(f"Player {i} ({bot_names[i]}): reward={reward}, avg_step_ms={avg_ms:.2f}")
    return MatchResult(
        seed=seed,
        done=done,
        steps=steps_run,
        scores=scores,
        rewards=rewards,
        avg_step_ms=avg_step_ms,
    )


def run_kaggle_match(bot_paths: list[Path], bot_names: list[str], seed: int, game_index: int = 1) -> MatchResult | None:
    try:
        from kaggle_environments import make
    except ModuleNotFoundError:
        print('Missing dependency: install with `pip install "kaggle-environments>=1.28.0"`')
        return None

    print(f"Match (kaggle engine): {bot_names[0]} vs {bot_names[1]}")

    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([str(path) for path in bot_paths])

    n_players = len(bot_names)
    total_time = [0.0] * n_players
    call_counts = [0] * n_players
    over_budget = [0] * n_players
    act_timeout = env.configuration.actTimeout

    for step_logs in env.logs:
        if not step_logs:
            continue
        for i in range(n_players):
            if i < len(step_logs) and "duration" in step_logs[i]:
                duration = step_logs[i]["duration"]
                total_time[i] += duration
                call_counts[i] += 1
                if duration > act_timeout:
                    over_budget[i] += 1

    final = env.steps[-1]
    rewards = []
    statuses = []
    avg_step_ms = []
    for i, state in enumerate(final):
        avg_ms = (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        rewards.append(state.reward)
        statuses.append(state.status)
        avg_step_ms.append(avg_ms)
        print(
            f"Player {i} ({bot_names[i]}): reward={state.reward}, "
            f"status={state.status}, avg_step_ms={avg_ms:.2f}"
        )

    timed_out = any(over_budget) or any(s.status == "TIMEOUT" for s in final)
    if timed_out:
        parts = [
            f"P{i} ({bot_names[i]})={over_budget[i]}{' [KILLED]' if final[i].status == 'TIMEOUT' else ''}"
            for i in range(n_players)
        ]
        print(f"Turns over actTimeout ({act_timeout}s): " + ", ".join(parts))
    return MatchResult(
        seed=seed,
        done=all(state.status == "DONE" for state in final),
        steps=max(0, len(env.steps) - 1),
        scores=None,
        rewards=rewards,
        avg_step_ms=avg_step_ms,
        statuses=statuses,
    )


def summarize_results(results: list[MatchResult], bot_names: list[str]) -> None:
    if len(results) <= 1:
        return
    n_players = len(bot_names)
    wins = [0] * n_players
    ties = 0
    total_steps = 0
    total_avg_ms = [0.0] * n_players
    score_diffs = []

    for result in results:
        total_steps += result.steps
        for i in range(n_players):
            if i < len(result.avg_step_ms):
                total_avg_ms[i] += result.avg_step_ms[i]
        if result.scores is not None and len(result.scores) == 2:
            score_diffs.append(result.scores[0] - result.scores[1])
            if result.scores[0] > result.scores[1]:
                wins[0] += 1
            elif result.scores[1] > result.scores[0]:
                wins[1] += 1
            else:
                ties += 1
        elif result.rewards:
            best = max(x for x in result.rewards if x is not None)
            winners = [i for i, reward in enumerate(result.rewards) if reward == best]
            if len(winners) == 1:
                wins[winners[0]] += 1
            else:
                ties += 1

    print()
    print(f"Summary: games={len(results)} mean_steps={total_steps / len(results):.1f}")
    if n_players == 2:
        print(f"W-L-T for {bot_names[0]}: {wins[0]}-{wins[1]}-{ties}")
        if score_diffs:
            print(f"Mean score diff for {bot_names[0]}: {sum(score_diffs) / len(score_diffs):+.1f}")
    else:
        print("Wins: " + ", ".join(f"{bot_names[i]}={wins[i]}" for i in range(n_players)) + f", ties={ties}")
    print(
        "Mean avg_step_ms: "
        + ", ".join(f"{bot_names[i]}={total_avg_ms[i] / len(results):.2f}" for i in range(n_players))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Orbit Wars matches between two bots.")
    parser.add_argument("bot1")
    parser.add_argument("bot2")
    parser.add_argument("-n", "--num-games", "--games", dest="games", type=int, default=1)
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use the Kaggle reference engine (env.run). Default: native Rust engine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the match. If omitted, a random seed is chosen and printed.",
    )
    args = parser.parse_args()
    if args.games < 1:
        print("--num-games/-n must be at least 1")
        return 1

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"Seed: {seed}")
    if args.games > 1:
        print(f"Games: {args.games} (seeds {seed}..{seed + args.games - 1})")

    bot_names = [args.bot1, args.bot2]
    bot_paths = [bot_entry(name) for name in bot_names]

    missing = [str(path.relative_to(ROOT)) for path in bot_paths if not path.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    results: list[MatchResult] = []
    for i in range(args.games):
        game_seed = seed + i
        if args.games > 1:
            print()
            print(f"Game {i + 1}: seed={game_seed}")
        if args.kaggle:
            result = run_kaggle_match(bot_paths, bot_names, game_seed, game_index=i + 1)
            if result is None:
                return 1
        else:
            result = run_rust_match(bot_paths, bot_names, game_seed, game_index=i + 1)
        results.append(result)
    summarize_results(results, bot_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
