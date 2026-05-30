from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from orbit_wars_engine import OrbitWarsEngine
from orbit_wars_model import encode_obs


NUM_FRAMES = 4
PLANET_SLOTS = 44
ACTIONS_DIM = 7
TOKEN_DIM = 11
GLOBAL_DIM = 16
ACTION_CHOICES_PER_SOURCE = PLANET_SLOTS * ACTIONS_DIM
ACTION_DIMS = [ACTION_CHOICES_PER_SOURCE] * PLANET_SLOTS

TRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
BOTS_DIR = REPO_ROOT / "bots"


@contextlib.contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _bot_main(name: str) -> Path:
    direct = BOTS_DIR / name / "main.py"
    if direct.is_file():
        return direct
    for sub in BOTS_DIR.iterdir():
        cand = sub / name / "main.py"
        if sub.is_dir() and cand.is_file():
            return cand
    raise FileNotFoundError(f"no bot named {name!r} under {BOTS_DIR}")


class BotOpponent:
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


class PlanetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TOKEN_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

    def forward(self, planet_tokens: th.Tensor) -> th.Tensor:
        return self.mlp(planet_tokens)


class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2 * ACTIONS_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.rho = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

    def forward(self, h: th.Tensor, distances: th.Tensor, reachable_mask: th.Tensor) -> th.Tensor:
        with th.no_grad():
            d_ji = distances.transpose(1, 2)
            reach_ji = reachable_mask.transpose(1, 2).float()
            edge_mask = reach_ji.any(dim=-1)

        phi_in = th.cat([d_ji, reach_ji], dim=-1)
        msgs = self.phi(phi_in)
        msgs = msgs * edge_mask.unsqueeze(-1)
        agg = msgs.sum(dim=2)
        return h + self.rho(agg)


class FrameFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(NUM_FRAMES * 64, 64)

    def forward(self, joined: th.Tensor) -> th.Tensor:
        batch = joined.shape[0]
        x = joined.permute(0, 2, 1, 3).reshape(batch, PLANET_SLOTS, NUM_FRAMES * 64)
        return self.proj(x)


class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

    def forward(self, global_features: th.Tensor) -> th.Tensor:
        return self.mlp(global_features)


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.planet_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, fused_planets: th.Tensor, encoded_globals: th.Tensor, planet_presence: th.Tensor) -> th.Tensor:
        p = planet_presence[:, 0].unsqueeze(-1)
        planets_projection = self.planet_proj(fused_planets)
        masked_planets = p * planets_projection
        pooled = masked_planets.sum(dim=1) / p.sum(dim=1).clamp(min=1)
        return self.mlp(th.cat([pooled, encoded_globals], dim=-1))


class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.planet_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.src_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, ACTIONS_DIM),
        )

    def forward(self, fused_planets: th.Tensor, encoded_globals: th.Tensor, valid_actions_mask: th.Tensor) -> th.Tensor:
        batch = fused_planets.shape[0]
        h = self.planet_proj(fused_planets)
        g = encoded_globals.unsqueeze(1).expand(batch, PLANET_SLOTS, -1)
        h_src = self.src_proj(th.cat([h, g], dim=-1))
        h_tgt = self.tgt_proj(h)

        src = h_src.unsqueeze(2).expand(batch, PLANET_SLOTS, PLANET_SLOTS, -1)
        tgt = h_tgt.unsqueeze(1).expand(batch, PLANET_SLOTS, PLANET_SLOTS, -1)
        logits = self.action_mlp(th.cat([src, tgt], dim=-1))
        logits = logits.masked_fill(~valid_actions_mask.bool(), -1e8)
        return logits.reshape(batch, PLANET_SLOTS * ACTION_CHOICES_PER_SOURCE)


class GalaxyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.planet_encoder = PlanetEncoder()
        self.gnn_layer = GNNLayer()
        self.frame_fusion = FrameFusion()
        self.global_encoder = GlobalEncoder()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, obs: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        encoded_planets = self.planet_encoder(obs["tokens"].float())
        gnn_out = self.gnn_layer(encoded_planets[:, 0], obs["turns"].float(), obs["reachable_mask"].float())
        joined = th.cat([gnn_out.unsqueeze(1), encoded_planets[:, 1:]], dim=1)
        fused = self.frame_fusion(joined)
        encoded_globals = self.global_encoder(obs["globals"].float())
        values = self.value_head(fused, encoded_globals, obs["presence"].float())
        flat_logits = self.policy_head(fused, encoded_globals, obs["valid_actions_mask"])
        return values, flat_logits


class DummyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        batch = next(iter(observations.values())).shape[0]
        device = next(iter(observations.values())).device
        return th.zeros((batch, 1), device=device)


class GalaxyMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs: Any):
        kwargs.setdefault("features_extractor_class", DummyExtractor)
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], ortho_init=False, **kwargs)
        self.galaxy_net = GalaxyNet()
        self.action_dist = make_masked_proba_distribution(self.action_space)
        self.optimizer = self.optimizer_class(
            self.galaxy_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _distribution(self, obs: dict[str, th.Tensor], action_masks=None):
        values, logits = self.galaxy_net(obs)
        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return values, distribution

    def forward(self, obs: dict[str, th.Tensor], deterministic: bool = False, action_masks=None):
        values, distribution = self._distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: dict[str, th.Tensor], actions: th.Tensor, action_masks=None):
        values, distribution = self._distribution(obs, action_masks)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs: dict[str, th.Tensor], action_masks=None):
        _, distribution = self._distribution(obs, action_masks)
        return distribution

    def predict_values(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        values, _ = self.galaxy_net(obs)
        return values


class OrbitWarsHellburnerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent: str = "hellburner", seed: int = 0):
        super().__init__()
        self.base_seed = seed
        self.next_seed = seed
        self.engine = OrbitWarsEngine(num_players=2)
        self.opponent = BotOpponent(opponent)
        self.obs_pair = None
        self.feat = None
        self.turn = 0

        self.action_space = spaces.MultiDiscrete(np.array(ACTION_DIMS, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "globals": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
                "tokens": spaces.Box(-np.inf, np.inf, shape=(NUM_FRAMES, PLANET_SLOTS, TOKEN_DIM), dtype=np.float32),
                "presence": spaces.Box(0.0, 1.0, shape=(NUM_FRAMES, PLANET_SLOTS), dtype=np.float32),
                "turns": spaces.Box(0.0, np.inf, shape=(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=np.float32),
                "reachable_mask": spaces.Box(0, 1, shape=(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=np.uint8),
                "valid_actions_mask": spaces.Box(0, 1, shape=(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=np.uint8),
            }
        )

    def _encode_current(self) -> dict[str, np.ndarray]:
        assert self.obs_pair is not None
        self.feat = encode_obs(self.obs_pair[0], 0)
        return {
            "globals": self.feat["globals"].astype(np.float32),
            "tokens": self.feat["tokens"].reshape(NUM_FRAMES, PLANET_SLOTS, TOKEN_DIM).astype(np.float32),
            "presence": self.feat["presence"].reshape(NUM_FRAMES, PLANET_SLOTS).astype(np.float32),
            "turns": self.feat["turns"].reshape(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM).astype(np.float32),
            "reachable_mask": self.feat["reachable_mask"].reshape(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM).astype(np.uint8),
            "valid_actions_mask": self.feat["mask"].reshape(PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM).astype(np.uint8),
        }

    def action_masks(self) -> np.ndarray:
        assert self.feat is not None
        return self.feat["mask"].reshape(PLANET_SLOTS, ACTION_CHOICES_PER_SOURCE).astype(bool).ravel()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        self.opponent.reset()
        self.obs_pair = self.engine.reset(int(seed))["observations"]
        self.turn = 0
        return self._encode_current(), {"seed": int(seed)}

    def _decode_action(self, action: np.ndarray) -> list[list[float]]:
        assert self.feat is not None
        moves: list[list[float]] = []
        planet_ids = self.feat["planet_ids"]
        angles = self.feat["angles"]
        ship_counts = self.feat["ship_counts"]
        mask = self.feat["mask"]
        for source_slot, flat in enumerate(np.asarray(action, dtype=np.int64).reshape(PLANET_SLOTS)):
            target_slot = int(flat) // ACTIONS_DIM
            action_bin = int(flat) % ACTIONS_DIM
            if action_bin == 0:
                continue
            idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + action_bin
            if idx < 0 or idx >= mask.shape[0] or not mask[idx]:
                continue
            source_id = int(planet_ids[source_slot])
            ships = int(ship_counts[idx])
            if source_id < 0 or ships <= 0:
                continue
            moves.append([source_id, float(angles[idx]), ships])
        return moves

    def step(self, action: np.ndarray):
        assert self.obs_pair is not None
        learner_moves = self._decode_action(action)
        opponent_moves = self.opponent.act(self.obs_pair[1])
        out = self.engine.step([learner_moves, opponent_moves])
        self.obs_pair = out["observations"]
        self.turn += 1
        terminated = bool(out["done"])
        reward = float(out["reward"][0])
        obs = self._encode_current()
        info = {
            "turn": self.turn,
            "learner_moves": len(learner_moves),
            "opponent_moves": len(opponent_moves),
        }
        return obs, reward, terminated, False, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="hellburner")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", default=str(TRAIN_DIR / "checkpoints" / "galaxy_hellburner_sb3"))
    args = parser.parse_args()

    env = OrbitWarsHellburnerEnv(opponent=args.opponent, seed=args.seed)
    model = MaskablePPO(
        GalaxyMaskablePolicy,
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_timesteps)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"saved {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
