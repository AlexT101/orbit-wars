"""Minimal end-to-end training loop: Rust engine + Rust features + a /bots opponent.

This is a *plumbing* example, not a strong agent. It shows how the pieces fit:

    env_engine (OrbitWarsEngine)   ← training rollouts + shaped reward   (player 0 = learner)
    env_model  (encode_obs)        ← features computed in Rust           (one source of truth)
    bots/<...>/main.py : agent()   ← the opponent                        (player 1)
    torch                          ← a tiny REINFORCE policy we update

How the opponent works
----------------------
Every `/bots` agent is a module exposing `agent(obs)` that consumes the kaggle
observation dict. `env_engine` emits that exact dict shape per player, so we
just feed `observations[1]` (player 1's view) to the opponent's `agent`. We load
it the same way `run_match.py` does (importlib on its `main.py`).

Stateful bots (apollo2, graph) keep a Bot instance + turn counter at module
scope, so we **re-exec the module each episode** to give every game a fresh
opponent. Stateless bots (random, nearest-sniper) don't need this, but it's
cheap and harmless.

Run (from experimental_arch/, in a venv with env_engine, orbit_wars_model,
torch, and — for bots that import it — kaggle_environments):

    python examples/train_loop_example.py --opponents random,nearest-sniper --episodes 20

Add any bot under pantheow/bots/ by name: --opponents graph,hellburner,...
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
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
LR = 3e-3
SEED = 0
# ---------------------------------------------------------------------------

# pantheow/ root, so we can find bots/. This file is experimental_arch/examples/.
REPO_ROOT = Path(__file__).resolve().parents[2]
BOTS_DIR = REPO_ROOT / "bots"

# Macro-action space for the toy policy: a single "send fraction" per turn,
# applied to every owned planet (each sends that fraction of its ships toward
# its nearest non-owned planet). Replace with a real structured action head.
SEND_FRACTIONS = (0.5, 0.75, 1.0)
GLOBAL_FEATURES = 9


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
# Features: turn a Rust-encoded obs into a fixed-size vector for the policy.
# --------------------------------------------------------------------------- #
def summarize(obs: dict, player: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (global_feature_vector, distance_matrix, planet_ids).

    The distance matrix comes from Rust (`encode_obs`) — the SAME code the bot
    uses at test time. The global vector pools it + ownership into fixed dims so
    the MLP input size is independent of planet count.
    """
    feat = encode_obs(obs, player)
    ids = feat["planet_ids"]
    n = feat["n"]
    D = np.asarray(feat["distance_matrix"], dtype=np.float32).reshape(n, n) if n else np.zeros((0, 0), np.float32)

    planets = obs["planets"]
    owners = np.array([p[1] for p in planets])
    ships = np.array([p[5] for p in planets], dtype=np.float32)
    mine = owners == player
    enemy = (owners != player) & (owners != -1)
    neutral = owners == -1

    upper = D[np.triu_indices(n, k=1)] if n > 1 else np.array([0.0], np.float32)
    g = np.array([
        mine.sum(), enemy.sum(), neutral.sum(),
        np.log1p(ships[mine].sum()), np.log1p(ships[enemy].sum()),
        upper.mean() / 141.42, upper.min() / 141.42, upper.max() / 141.42,
        obs["step"] / 500.0,
    ], dtype=np.float32)
    return g, D, ids


def moves_for_fraction(obs: dict, player: int, D: np.ndarray, ids: list[int], frac: float) -> list:
    """From each owned planet, send `frac` of its ships toward its nearest
    non-owned planet (nearest chosen via the Rust distance matrix)."""
    planets = obs["planets"]
    idx_by_id = {pid: i for i, pid in enumerate(ids)}
    pos = {p[0]: (p[2], p[3]) for p in planets}
    owner = {p[0]: p[1] for p in planets}
    ships = {p[0]: p[5] for p in planets}

    moves = []
    targets = [p[0] for p in planets if p[1] != player]
    if not targets:
        return moves
    for p in planets:
        pid = p[0]
        if owner[pid] != player or ships[pid] < 2:
            continue
        i = idx_by_id[pid]
        nearest = min(targets, key=lambda t: D[i, idx_by_id[t]])
        send = int(frac * ships[pid])
        if send < 1:
            continue
        (sx, sy), (tx, ty) = pos[pid], pos[nearest]
        angle = math.atan2(ty - sy, tx - sx)
        moves.append([pid, angle, send])
    return moves


# --------------------------------------------------------------------------- #
# Toy policy: global features -> Categorical over SEND_FRACTIONS. REINFORCE.
# --------------------------------------------------------------------------- #
class TinyPolicy(nn.Module):
    def __init__(self, in_dim: int = GLOBAL_FEATURES, hidden: int = 64, n_actions: int = len(SEND_FRACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


def run_episode(engine: OrbitWarsEngine, policy: TinyPolicy, opponent: BotOpponent, seed: int):
    """One game: learner = player 0, opponent = player 1. Returns (log_probs, return)."""
    opponent.reset()
    obs = engine.reset(seed=seed)["observations"]
    log_probs = []
    total_reward = 0.0

    for _ in range(500):
        # --- learner (player 0): Rust features -> policy -> macro-action -> moves ---
        g, D, ids = summarize(obs[0], player=0)
        logits = policy(torch.from_numpy(g))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        learner_moves = moves_for_fraction(obs[0], 0, D, ids, SEND_FRACTIONS[int(action)])

        # --- opponent (player 1): a /bots agent on its own observation ---
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
        # (Episode-level return as the signal; a real run would use GAE/PPO and
        # the value head, but this keeps the example to the integration story.)
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
