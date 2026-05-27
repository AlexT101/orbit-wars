from __future__ import annotations

import argparse
import json

import torch
from torch.distributions import Categorical

from orbit_wars_rl.env import OrbitWarsDuelEnv
from orbit_wars_rl.features import decode_move, encode_obs
from orbit_wars_rl.model import build_policy, tensorize


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
    args = parser.parse_args()

    model = load_policy(args.checkpoint, args.device)
    opponent = (
        policy_opponent(model, args.device, sample=args.opponent in {"self_sample", "snapshot_sample"})
        if args.opponent in {"self", "self_sample", "snapshot", "snapshot_sample"}
        else args.opponent
    )
    if args.opponent.startswith("checkpoint:") or args.opponent.startswith("checkpoint_sample:"):
        prefix, path = args.opponent.split(":", 1)
        opponent = checkpoint_opponent(path, args.device, sample=prefix == "checkpoint_sample")
    wins = losses = draws = 0
    rewards = []
    for i in range(args.games):
        env = OrbitWarsDuelEnv(seed=args.seed + i, opponent=opponent)
        obs = env.reset(args.seed + i)
        done = False
        while not done:
            action = choose_action(model, obs, args.device, deterministic=not args.sample)
            result = env.step(action)
            obs = result.obs
            done = result.done
        raw = result.info["raw_rewards"]
        rewards.append(raw)
        if raw[0] > raw[1]:
            wins += 1
        elif raw[1] > raw[0]:
            losses += 1
        else:
            draws += 1
    print(json.dumps({"games": args.games, "wins": wins, "losses": losses, "draws": draws, "rewards": rewards}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
