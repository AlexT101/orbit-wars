from __future__ import annotations

import torch as th
from torch import nn

from model import EntityTransformerPolicy, OrbitPolicy, build_policy, tensorize


D_MODEL = 128
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4


class GalaxyNet(nn.Module):
    """Compatibility wrapper around the copied reference EntityTransformerPolicy."""

    def __init__(
        self,
        hidden: int = D_MODEL,
        transformer_layers: int = TRANSFORMER_LAYERS,
        transformer_heads: int = TRANSFORMER_HEADS,
    ) -> None:
        super().__init__()
        self.policy = EntityTransformerPolicy(
            hidden=hidden,
            layers=transformer_layers,
            heads=transformer_heads,
        )

    def forward(self, obs: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        logits, value = self.policy(
            obs["tokens"].float()[:, 0],
            obs["presence"].float()[:, 0],
            obs["globals"].float(),
            discrete_action_mask(obs),
        )
        return value.unsqueeze(-1), logits


def discrete_action_mask(obs: dict[str, th.Tensor]) -> th.Tensor:
    mask = obs.get("discrete_action_mask")
    if mask is not None:
        return mask.bool()
    valid = obs["valid_actions_mask"].bool()
    send_targets = valid[:, :, 2:]
    noop = th.ones(valid.shape[0], 1, dtype=th.bool, device=valid.device)
    return th.cat([noop, send_targets.reshape(valid.shape[0], -1)], dim=1)
