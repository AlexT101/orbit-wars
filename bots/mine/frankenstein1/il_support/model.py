from __future__ import annotations

import torch
from torch import nn

from features import GLOBAL_FEATURES, MAX_PLANETS, PLANET_FEATURES, SEND_FRACTIONS


def with_timeline_features(planets, planet_timeline_features=None):
    if planet_timeline_features is None:
        extra = planets.new_zeros((*planets.shape[:-1], PLANET_FEATURES - planets.shape[-1]))
    else:
        extra = planet_timeline_features.to(dtype=planets.dtype, device=planets.device)
    return torch.cat([planets, extra], dim=-1)


class OrbitPolicy(nn.Module):
    """Entity-pair policy with a shared pooled value head."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.planet_encoder = nn.Sequential(
            nn.Linear(PLANET_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(SEND_FRACTIONS)),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        planets,
        planet_mask,
        globals_,
        action_mask=None,
        pair_turns=None,
        pair_reachable_mask=None,
        planet_timeline_features=None,
    ):
        batch = planets.shape[0]
        encoded = self.planet_encoder(with_timeline_features(planets, planet_timeline_features))
        mask = planet_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        src = encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., 7:9]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_logits = self.pair_head(torch.cat([pair, pair_geom, g], dim=-1))
        pair_logits = pair_logits.reshape(batch, -1)
        noop_logits = self.noop_head(torch.cat([pooled, globals_], dim=-1))
        logits = torch.cat([noop_logits, pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(torch.cat([pooled, globals_], dim=-1)).squeeze(-1)
        return logits, value


class EntityTransformerPolicy(nn.Module):
    """Small entity transformer over planet tokens plus one global token."""

    def __init__(self, hidden: int = 128, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.planet_encoder = nn.Sequential(
            nn.Linear(PLANET_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(hidden)
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(SEND_FRACTIONS)),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        planets,
        planet_mask,
        globals_,
        action_mask=None,
        pair_turns=None,
        pair_reachable_mask=None,
        planet_timeline_features=None,
    ):
        batch = planets.shape[0]
        planet_tokens = self.planet_encoder(with_timeline_features(planets, planet_timeline_features))
        global_token = self.global_encoder(globals_).unsqueeze(1)
        tokens = torch.cat([global_token, planet_tokens], dim=1)
        global_valid = torch.ones(batch, 1, dtype=torch.bool, device=planet_mask.device)
        valid = torch.cat([global_valid, planet_mask.bool()], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        encoded = self.final_norm(encoded)
        global_encoded = encoded[:, 0]
        planet_encoded = encoded[:, 1:]

        src = planet_encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = planet_encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., 7:9]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_logits = self.pair_head(torch.cat([pair, pair_geom, g], dim=-1)).reshape(batch, -1)
        state = torch.cat([global_encoded, globals_], dim=-1)
        logits = torch.cat([self.noop_head(state), pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(state).squeeze(-1)
        return logits, value


def build_policy(
    model_type: str = "mlp",
    hidden: int = 128,
    transformer_layers: int = 3,
    transformer_heads: int = 4,
) -> nn.Module:
    if model_type == "mlp":
        return OrbitPolicy(hidden=hidden)
    if model_type in {"entity_transformer", "entity_transformer_temporal"}:
        return EntityTransformerPolicy(
            hidden=hidden,
            layers=transformer_layers,
            heads=transformer_heads,
        )
    raise ValueError(f"unknown model type: {model_type}")


def tensorize(encoded, device="cpu"):
    return {
        "planets": torch.as_tensor(encoded.planets, dtype=torch.float32, device=device).unsqueeze(0),
        "planet_mask": torch.as_tensor(encoded.planet_mask, dtype=torch.float32, device=device).unsqueeze(0),
        "globals_": torch.as_tensor(encoded.globals, dtype=torch.float32, device=device).unsqueeze(0),
        "action_mask": torch.as_tensor(encoded.action_mask, dtype=torch.bool, device=device).unsqueeze(0),
        "planet_timeline_features": torch.as_tensor(
            encoded.planet_timeline_features, dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
