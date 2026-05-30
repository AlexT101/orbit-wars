from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys

import torch
from torch.distributions import Categorical

from orbit_wars_rl.env import OrbitWarsDuelEnv
from orbit_wars_rl.features import decode_move, encode_obs
from orbit_wars_rl.model import build_policy, tensorize


_WORKER_MODEL = None
_WORKER_DEVICE = "cpu"
_WORKER_OPPONENT = None
_WORKER_SAMPLE = False


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_policy(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    model = build_policy(
        config.get("model", "mlp"),
        config.get("hidden", 128),
        config.get("transformer_layers", 3),
        config.get("transformer_heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def choose_action(model, obs, device: str, deterministic: bool) -> int:
    encoded = encode_obs(obs)
    batch = tensorize(encoded, device)
    with torch.no_grad():
        logits, _value = model(**batch)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


def policy_opponent(model, device: str, sample: bool):
    def opponent(obs):
        return decode_move(obs, choose_action(model, obs, device, deterministic=not sample))

    return opponent


def checkpoint_opponent(path: str, device: str, sample: bool):
    return policy_opponent(load_policy(path, device), device, sample)


def resolve_opponent(opponent_name: str, device: str):
    opponent = (
        policy_opponent(_WORKER_MODEL, device, sample=opponent_name in {"self_sample", "snapshot_sample"})
        if opponent_name in {"self", "self_sample", "snapshot", "snapshot_sample"}
        else opponent_name
    )
    if opponent_name.startswith("checkpoint:") or opponent_name.startswith("checkpoint_sample:"):
        prefix, path = opponent_name.split(":", 1)
        opponent = checkpoint_opponent(path, device, sample=prefix == "checkpoint_sample")
    return opponent


def init_worker(checkpoint: str, device: str, opponent_name: str, sample: bool) -> None:
    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_OPPONENT, _WORKER_SAMPLE
    torch.set_num_threads(1)
    _WORKER_DEVICE = device
    _WORKER_SAMPLE = sample
    _WORKER_MODEL = load_policy(checkpoint, device)
    _WORKER_OPPONENT = resolve_opponent(opponent_name, device)


def eval_one_game(seed: int) -> list[float]:
    if _WORKER_MODEL is None:
        raise RuntimeError("worker not initialized")
    env = OrbitWarsDuelEnv(seed=seed, opponent=_WORKER_OPPONENT)
    obs = env.reset(seed)
    done = False
    result = None
    while not done:
        action = choose_action(_WORKER_MODEL, obs, _WORKER_DEVICE, deterministic=not _WORKER_SAMPLE)
        result = env.step(action)
        obs = result.obs
        done = result.done
    assert result is not None
    return result.info["raw_rewards"]


def outcome_from_raw(raw: list[float]) -> str:
    if raw[0] > raw[1]:
        return "win"
    if raw[1] > raw[0]:
        return "loss"
    return "draw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy against scripted bots.")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--opponent",
        default="nearest",
        help="Opponent: built-in noop/random/nearest or any bot name, e.g. hellburner or heuristic.",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--json", action="store_true", help="Write final result as JSON to stdout.")
    parser.add_argument("--progress", action="store_true", help="Log per-game progress to stderr.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel game workers for evaluation. Use 0 for about half of CPU cores.",
    )
    args = parser.parse_args()

    wins = losses = draws = 0
    rewards = []
    if args.progress:
        log(
            f"{color('eval', BOLD + CYAN)} "
            f"checkpoint={color(args.checkpoint, DIM)} opponent={color(args.opponent, BLUE)} "
            f"games={args.games} seed={args.seed} workers={max(1, args.workers)}"
        )
    seeds = [args.seed + i for i in range(args.games)]
    completed_games = 0

    def record(raw):
        nonlocal wins, losses, draws, completed_games
        rewards.append(raw)
        completed_games += 1
        outcome_name = outcome_from_raw(raw)
        if outcome_name == "win":
            wins += 1
            outcome = color("W", GREEN)
        elif outcome_name == "loss":
            losses += 1
            outcome = color("L", RED)
        else:
            draws += 1
            outcome = color("D", YELLOW)
        if args.progress:
            wr = (wins + 0.5 * draws) / completed_games
            log(
                f"  {color(f'{completed_games:>3}/{args.games}', DIM)} {outcome} "
                f"wr={color(f'{wr:5.1%}', GREEN if wr >= 0.5 else RED)} "
                f"reward={raw[0]:.0f}:{raw[1]:.0f}"
            )

    requested_workers = (max(1, (os.cpu_count() or 2) // 2) if args.workers == 0 else args.workers)
    workers = max(1, min(requested_workers, args.games))
    if workers == 1:
        init_worker(args.checkpoint, args.device, args.opponent, args.sample)
        for seed in seeds:
            record(eval_one_game(seed))
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(args.checkpoint, args.device, args.opponent, args.sample),
            ) as pool:
                futures = [pool.submit(eval_one_game, seed) for seed in seeds]
                for future in as_completed(futures):
                    record(future.result())
        except PermissionError as exc:
            log(f"{color('eval warning', YELLOW)} multiprocessing unavailable ({exc}); falling back to one worker")
            init_worker(args.checkpoint, args.device, args.opponent, args.sample)
            for seed in seeds:
                record(eval_one_game(seed))
    result = {"games": args.games, "wins": wins, "losses": losses, "draws": draws, "rewards": rewards}
    win_rate = (wins + 0.5 * draws) / max(1, args.games)
    if args.json:
        print(json.dumps(result))
    else:
        status = GREEN if win_rate >= 0.5 else RED
        print(
            f"{color('result', BOLD)} opponent={color(args.opponent, BLUE)} "
            f"wr={color(f'{win_rate:.1%}', status)} "
            f"wins={wins} losses={losses} draws={draws}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
