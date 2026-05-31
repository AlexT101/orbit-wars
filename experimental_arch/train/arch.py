from __future__ import annotations

from typing import Any

import torch as th
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from constants import (
    ACTIONS_DIM,
    ACTION_CHOICES_PER_SOURCE,
    GLOBAL_DIM,
    NUM_FRAMES,
    PLANET_SLOTS,
    TOKEN_DIM,
)

D_MODEL = 64


class PlanetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TOKEN_DIM, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

    def forward(self, planet_tokens: th.Tensor) -> th.Tensor:
        return self.mlp(planet_tokens)


class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2 * ACTIONS_DIM, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.rho = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
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
        self.proj = nn.Linear(NUM_FRAMES * D_MODEL, D_MODEL)

    def forward(self, joined: th.Tensor) -> th.Tensor:
        batch = joined.shape[0]
        x = joined.permute(0, 2, 1, 3).reshape(batch, PLANET_SLOTS, NUM_FRAMES * D_MODEL)
        return self.proj(x)


class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(GLOBAL_DIM, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

    def forward(self, global_features: th.Tensor) -> th.Tensor:
        return self.mlp(global_features)


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.planet_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, 1),
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
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.src_proj = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, ACTIONS_DIM),
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
        batch = encoded_planets.shape[0]
        gnn_out = self.gnn_layer(
            encoded_planets.reshape(batch * NUM_FRAMES, PLANET_SLOTS, D_MODEL),
            obs["turns"].float().reshape(batch * NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM),
            obs["reachable_mask"].float().reshape(batch * NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM),
        ).reshape(batch, NUM_FRAMES, PLANET_SLOTS, D_MODEL)
        fused = self.frame_fusion(gnn_out)
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
