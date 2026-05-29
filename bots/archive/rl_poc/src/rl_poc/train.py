"""Minimal PPO training loop.

One process, sequential episodes, CPU. After each `--updates` rounds of
`--games-per-update` episodes against the chosen opponent we PPO-update
and write `checkpoints/latest.pt`. wandb logs go to `orbit-wars-rl-poc`
unless overridden.

POC choices that should not survive contact with the full design:
- Returns are computed with γ on a sparse terminal reward only.
- We sum logprobs across all sources per turn (treats them as
  conditionally independent given state). Fine for POC, less fine for
  cross-source coordination.
- We share buffers across episodes within one update; no parallel workers.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wandb

from .features import parse_obs
from .model import ActorCritic
from .opponents import make_opponent
from .policy import ActionRecord, select_moves
from .env import run_episode


CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StepRecord:
    planet_table: np.ndarray
    planet_mask: np.ndarray
    mine_mask: np.ndarray
    globals_: np.ndarray
    record: ActionRecord
    value: float
    reward: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


def collect_episode(model: ActorCritic, opponent, seed: int | None, swap_sides: bool):
    """Roll out one game, returning the per-step records and final reward."""

    steps: list[StepRecord] = []

    def learner_agent(obs_dict):
        ov = parse_obs(obs_dict)
        with torch.no_grad():
            pt = torch.from_numpy(ov.planet_table).unsqueeze(0)
            pm = torch.from_numpy(ov.planet_mask).unsqueeze(0)
            mm = torch.from_numpy(ov.mine_mask).unsqueeze(0)
            gl = torch.from_numpy(ov.globals_).unsqueeze(0)
            logits, _, v = model(pt, pm, mm, gl)
            moves, record = select_moves(
                ov, logits[0], mm[0], sample=True
            )
        step = StepRecord(
            planet_table=ov.planet_table,
            planet_mask=ov.planet_mask,
            mine_mask=ov.mine_mask,
            globals_=ov.globals_,
            record=record,
            value=float(v.item()),
        )
        steps.append(step)
        return moves, None

    _, reward, n_steps = run_episode(
        learner_agent, opponent, seed=seed, swap_sides=swap_sides
    )
    return steps, reward, n_steps


def compute_returns(steps: list[StepRecord], final_reward: float, gamma: float, lam: float):
    """GAE-Lambda with terminal reward only.

    Treats every intermediate reward as 0; the final reward lands at the
    terminal step. Bootstrapped V at t = T is zero (episode ends)."""

    if not steps:
        return
    steps[-1].reward = final_reward
    next_value = 0.0
    next_advantage = 0.0
    for t in reversed(range(len(steps))):
        v = steps[t].value
        r = steps[t].reward
        delta = r + gamma * next_value - v
        next_advantage = delta + gamma * lam * next_advantage
        steps[t].advantage = next_advantage
        steps[t].return_ = next_advantage + v
        next_value = v


def ppo_update(
    model: ActorCritic,
    optimiser: torch.optim.Optimizer,
    batch: list[StepRecord],
    *,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    epochs: int,
):
    """Run `epochs` passes of PPO over the collected batch."""

    if not batch:
        return {}

    pt = torch.from_numpy(np.stack([s.planet_table for s in batch]))
    pm = torch.from_numpy(np.stack([s.planet_mask for s in batch]))
    mm = torch.from_numpy(np.stack([s.mine_mask for s in batch]))
    gl = torch.from_numpy(np.stack([s.globals_ for s in batch]))

    old_logprobs = torch.tensor(
        [s.record.logprob_sum for s in batch], dtype=torch.float32
    )
    advantages = torch.tensor([s.advantage for s in batch], dtype=torch.float32)
    returns = torch.tensor([s.return_ for s in batch], dtype=torch.float32)
    # Normalise advantages — standard PPO trick.
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics_acc = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}

    for _ in range(epochs):
        logits, _, values = model(pt, pm, mm, gl)
        # Recompute new logprob and entropy across each row of each batch element.
        new_logprob, entropy = _recompute_logprob_entropy(logits, batch)

        ratio = (new_logprob - old_logprobs).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimiser.step()

        with torch.no_grad():
            kl = (old_logprobs - new_logprob).mean().item()
        metrics_acc["policy_loss"] += float(policy_loss.item())
        metrics_acc["value_loss"] += float(value_loss.item())
        metrics_acc["entropy"] += float(entropy.mean().item())
        metrics_acc["kl"] += float(kl)

    for k in metrics_acc:
        metrics_acc[k] /= max(epochs, 1)
    return metrics_acc


def _recompute_logprob_entropy(logits: torch.Tensor, batch: list[StepRecord]):
    """Re-derive sum-logprob and mean-entropy per batch element for PPO.

    logits: (B, P, P+1). For each batch element, look up the source slots and
    chosen indices stored in the step's ActionRecord, gather logπ at those
    cells, sum them. Entropy is averaged across the source rows."""

    b = logits.shape[0]
    # Replace -inf with very-negative so log_softmax handles the masked rows.
    safe = torch.where(torch.isinf(logits), torch.full_like(logits, -1e9), logits)
    logp = F.log_softmax(safe, dim=-1)  # (B, P, P+1)

    new_logprob = torch.zeros(b, dtype=torch.float32)
    entropy = torch.zeros(b, dtype=torch.float32)
    probs = logp.exp()
    # Per-row entropy then mean across the sources we actually sampled at.
    for i, step in enumerate(batch):
        srcs = step.record.source_slots
        idxs = step.record.chosen_idx
        if len(srcs) == 0:
            continue
        srcs_t = torch.from_numpy(srcs)
        idxs_t = torch.from_numpy(idxs)
        new_logprob[i] = logp[i, srcs_t, idxs_t].sum()
        # entropy = -Σ p log p over each row, masked rows are 0
        row_logp = logp[i, srcs_t]
        row_probs = probs[i, srcs_t]
        # Replace very-negative entries with 0 contribution (e^-1e9 ≈ 0).
        entropy[i] = -(row_probs * torch.clamp(row_logp, min=-50.0)).sum(dim=-1).mean()
    return new_logprob, entropy


def evaluate_winrate(
    model: ActorCritic, opponent, n_games: int, seed_base: int
) -> tuple[float, float]:
    """Greedy eval — returns (winrate, drawrate)."""
    wins = 0
    draws = 0
    for k in range(n_games):
        swap = k % 2 == 1

        def learner_agent(obs_dict):
            ov = parse_obs(obs_dict)
            with torch.no_grad():
                pt = torch.from_numpy(ov.planet_table).unsqueeze(0)
                pm = torch.from_numpy(ov.planet_mask).unsqueeze(0)
                mm = torch.from_numpy(ov.mine_mask).unsqueeze(0)
                gl = torch.from_numpy(ov.globals_).unsqueeze(0)
                logits, _, _ = model(pt, pm, mm, gl)
                moves, _ = select_moves(ov, logits[0], mm[0], sample=False)
            return moves, None

        _, reward, _ = run_episode(
            learner_agent, opponent, seed=seed_base + k, swap_sides=swap
        )
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
    return wins / max(n_games, 1), draws / max(n_games, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", default="random",
                        choices=["random", "nearest-sniper"])
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--games-per-update", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="orbit-wars-rl-poc")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/latest.pt if present")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ActorCritic()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = CKPT_DIR / "latest.pt"
    start_update = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_update = int(ckpt.get("update", 0))
        print(f"Resumed from update {start_update}")

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
        mode=args.wandb_mode,
        dir=str(CKPT_DIR.parent / "wandb"),
    )

    opponent = make_opponent(args.opponent)

    seed_counter = args.seed * 1000
    for u in range(start_update, start_update + args.updates):
        batch: list[StepRecord] = []
        ep_rewards = []
        ep_lengths = []
        wins = 0
        for g in range(args.games_per_update):
            swap = g % 2 == 1
            steps, reward, length = collect_episode(
                model, opponent, seed=seed_counter, swap_sides=swap
            )
            seed_counter += 1
            compute_returns(steps, reward, args.gamma, args.lam)
            batch.extend(steps)
            ep_rewards.append(reward)
            ep_lengths.append(length)
            if reward > 0:
                wins += 1

        metrics = ppo_update(
            model,
            optim,
            batch,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            epochs=args.epochs,
        )

        eval_winrate, eval_drawrate = evaluate_winrate(
            model, opponent, args.eval_games, seed_base=10_000 + u * 100
        )

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        log = {
            "update": u + 1,
            "train/mean_reward": mean_reward,
            "train/mean_length": mean_length,
            "train/winrate_sampling": wins / max(len(ep_rewards), 1),
            "train/batch_size": len(batch),
            "eval/winrate_greedy": eval_winrate,
            "eval/drawrate_greedy": eval_drawrate,
            **{f"loss/{k}": v for k, v in metrics.items()},
        }
        wandb.log(log)
        print(
            f"update {u+1:4d} | "
            f"reward {mean_reward:+.2f} | "
            f"len {mean_length:.0f} | "
            f"win {log['train/winrate_sampling']:.2f} | "
            f"eval {eval_winrate:.2f} | "
            f"pl {metrics.get('policy_loss', float('nan')):+.4f} | "
            f"vl {metrics.get('value_loss', float('nan')):.4f} | "
            f"H {metrics.get('entropy', float('nan')):.3f}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "update": u + 1,
                "opponent": args.opponent,
            },
            ckpt_path,
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
