"""Minimal end-to-end training loop: Rust engine + Rust features + a /bots opponent.

This is a *plumbing* example, not a strong agent. It shows how the pieces fit:

    env_engine (OrbitWarsEngine)   ← training rollouts + shaped reward   (player 0 = learner)
    env_model  (encode_obs)        ← features computed in Rust           (one source of truth)
    bots/<...>/main.py : agent()   ← the opponent                        (player 1)
    torch                          ← a tiny REINFORCE policy we update

Feature / action interface (the real one)
------------------------------------------
`encode_obs(obs, player)` returns numpy arrays (zero-copy; `torch.from_numpy`-ready):
  - tokens   (NUM_FRAMES, 44, TOKEN_DIM)   per-planet features at t / t+1 / t+10 / t_resolved
  - globals  (GLOBAL_DIM,)                 board-level summary (scores, shares, …) at t
  - presence (NUM_FRAMES, 44)              which slots are real planets
  - turns    (44, 44, 7)                   turns-to-arrive per (src, tgt, action) at t
  - angles   (44, 44, 7)                   launch angle to actually issue the move
  - mask     (44, 44, 7)                   1 = legal action now
  - ship_counts / reachable_mask            integer ships + clean-arrival bit per action
  - planet_ids / frame_planets             slot→id map + raw per-frame planet state
The policy outputs a logit per action over the (44, 44, 7) space; we mask out the
illegal ones, sample (src, tgt, action), and convert it to a game move
`[source_id, angle, ships]` using `angles` and the action's ship count.

How the opponent works
----------------------
Every `/bots` agent is a module exposing `agent(obs)` over the kaggle observation
dict. `env_engine` emits that exact dict shape per player, so we feed
`observations[1]` (player 1's view) to the opponent's `agent` (loaded the way
`run_match.py` does). Stateful bots (apollo2, graph) keep state at module scope,
so we re-exec the module each episode for a fresh opponent.

Run (from experimental_arch/, in a venv with env_engine, orbit_wars_model, torch,
and — for bots that import it — kaggle_environments):

    python examples/train_loop_example.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from orbit_wars_engine import OrbitWarsEngine
from orbit_wars_model import encode_obs

# --- config (edit here) ----------------------------------------------------
OPPONENTS = ["random", "nearest-sniper"]  # bot names under pantheow/bots/
EPISODES = 20
LR = 1e-3
SEED = 0
# ---------------------------------------------------------------------------

# pantheow/ root, so we can find bots/. This file is experimental_arch/examples/.
REPO_ROOT = Path(__file__).resolve().parents[2]
BOTS_DIR = REPO_ROOT / "bots"

# Must match features.rs. Action 0 is per-source noop, actions 1-4 are
# send-fractions, action 5 is a constant, and action 6 is "resolved + 1".
ACTIONS_DIM = 7
SEND_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
CONST_SEND = 42
TOKEN_DIM = 11
GLOBAL_DIM = 16  # width of the encode_obs `globals` vector (board-level summary)


# --------------------------------------------------------------------------- #
# Opponent: load a /bots agent and (re)create it per episode.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _suppress_stdout():
    """Bots like graph/apollo2 print [LINE]/[DOT]/[TEXT] debug every turn."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _bot_main(name: str) -> Path:
    """Resolve bots/<name>/main.py or bots/*/<name>/main.py (run_match.py logic)."""
    direct = BOTS_DIR / name / "main.py"
    if direct.is_file():
        return direct
    for sub in BOTS_DIR.iterdir():
        cand = sub / name / "main.py"
        if sub.is_dir() and cand.is_file():
            return cand
    raise FileNotFoundError(f"no bot named {name!r} under {BOTS_DIR}")


class BotOpponent:
    """Wraps a /bots agent. `reset()` re-execs the module for fresh per-game state."""

    def __init__(self, name: str):
        self.name = name
        self.path = _bot_main(name)
        self._mod_name = f"opp__{name.replace('-', '_')}"
        self.agent = None
        self.reset()

    def reset(self) -> None:
        with _suppress_stdout():
            spec = importlib.util.spec_from_file_location(self._mod_name, self.path)
            assert spec and spec.loader, f"could not load {self.path}"
            module = importlib.util.module_from_spec(spec)
            sys.modules[self._mod_name] = module
            spec.loader.exec_module(module)
        self.agent = module.agent

    def act(self, obs: dict) -> list:
        with _suppress_stdout():
            return self.agent(obs)


# --------------------------------------------------------------------------- #
# Features -> fixed-size vector for the (toy) policy input.
# --------------------------------------------------------------------------- #
def global_vector(feat: dict) -> np.ndarray:
    """The board-level `globals` vector encode_obs already emits (normalized,
    fixed size). A real model would also consume the full token set + action
    grid; this keeps the example's policy input tiny and planet-count-agnostic."""
    return feat["globals"].astype(np.float32)


def action_to_move(feat: dict, flat_idx: int):
    """Convert a flat (src, tgt, action) index into a game move.

    Returns None for the per-source noop action; otherwise returns
    `[source_id, angle, ships]`. Assumes the action is legal (mask == 1), so the
    count is sendable."""
    ps, _, ad = feat["mask_shape"]
    a = flat_idx % ad
    sj = (flat_idx // ad) % ps
    si = flat_idx // (ad * ps)
    if a == 0:
        return None

    ids = feat["planet_ids"]
    id_i, id_j = ids[si], ids[sj]

    count = int(feat["ship_counts"][flat_idx])
    angle = float(feat["angles"][flat_idx])
    return [id_i, angle, count]


# --------------------------------------------------------------------------- #
# Toy policy: global features -> logit per action over (44, 44, 7). REINFORCE.
# --------------------------------------------------------------------------- #
class TinyPolicy(nn.Module):
    def __init__(self, in_dim: int = GLOBAL_DIM, hidden: int = 128, n_actions: int = 44 * 44 * ACTIONS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # flat (44*44*ACTIONS_DIM,) logits


def act(policy: TinyPolicy, obs: dict):
    """Pick one legal move from `obs` (player 0). Returns (moves, log_prob).

    The mask gates the logits; if no action is legal we pass (empty moves)."""
    feat = encode_obs(obs, 0)
    g = global_vector(feat)
    logits = policy(torch.from_numpy(g))
    mask = torch.from_numpy(feat["mask"]).bool()  # (44*44*ACTIONS_DIM,)
    if not bool(mask.any()):
        return [], None
    dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask, float("-inf")))
    idx = dist.sample()
    move = action_to_move(feat, int(idx))
    return ([] if move is None else [move]), dist.log_prob(idx)


def run_episode(engine: OrbitWarsEngine, policy: TinyPolicy, opponent: BotOpponent, seed: int):
    """One game: learner = player 0, opponent = player 1. Returns (log_probs, return)."""
    opponent.reset()
    obs = engine.reset(seed=seed)["observations"]
    log_probs = []
    total_reward = 0.0

    for _ in range(500):
        learner_moves, log_prob = act(policy, obs[0])
        if log_prob is not None:
            log_probs.append(log_prob)
        opp_moves = opponent.act(obs[1])

        out = engine.step([learner_moves, opp_moves])
        obs = out["observations"]
        total_reward += out["reward"][0]  # player 0's shaped reward
        if out["done"]:
            break

    return log_probs, total_reward


def main() -> int:
    torch.manual_seed(SEED)
    engine = OrbitWarsEngine(num_players=2)
    policy = TinyPolicy()
    optim = torch.optim.Adam(policy.parameters(), lr=LR)

    opponents = [BotOpponent(n) for n in OPPONENTS]
    opp_cycle = cycle(opponents)
    returns: list[float] = []

    for ep in range(EPISODES):
        opp = next(opp_cycle)
        log_probs, ret = run_episode(engine, policy, opp, seed=SEED + ep)
        returns.append(ret)

        # REINFORCE: maximize return -> minimize -(return * sum log_probs).
        # (Episode-level return; a real run would use GAE/PPO + a value head, but
        # this keeps the example to the integration story.)
        if log_probs:
            loss = -(ret * torch.stack(log_probs).sum())
            optim.zero_grad()
            loss.backward()
            optim.step()

        avg = np.mean(returns[-10:])
        print(f"ep {ep:3d}  vs {opp.name:<16}  return={ret:+.3f}  avg10={avg:+.3f}")

    print(f"\nFinished {EPISODES} episodes over opponents: "
          f"{', '.join(o.name for o in opponents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
