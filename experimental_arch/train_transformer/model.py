from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from features import GLOBAL_FEATURES, MAX_PLANETS, PLANET_FEATURES, SEND_ACTIONS, SEND_FRACTIONS

PAIR_OUTCOME_FEATURES = 4
PLANET_XY_SLICE = slice(11, 13)
ACTION_FEATURES = 8


def with_timeline_features(planets, planet_timeline_features=None):
    if planet_timeline_features is None:
        extra = planets.new_zeros((*planets.shape[:-1], PLANET_FEATURES - planets.shape[-1]))
    else:
        extra = planet_timeline_features.to(dtype=planets.dtype, device=planets.device)
    return torch.cat([planets, extra], dim=-1)


def hypersphere_norm(x):
    return F.normalize(x, p=2, dim=-1, eps=1.0e-6) * math.sqrt(x.shape[-1])


def action_feature_tensor(planets, pair_turns=None, pair_reachable_mask=None):
    batch = planets.shape[0]
    device = planets.device
    dtype = planets.dtype
    frac = torch.tensor(SEND_FRACTIONS, dtype=dtype, device=device).view(1, 1, 1, len(SEND_FRACTIONS), 1)
    is_half = torch.tensor([1.0, 0.0], dtype=dtype, device=device).view(1, 1, 1, len(SEND_FRACTIONS), 1)
    is_all = torch.tensor([0.0, 1.0], dtype=dtype, device=device).view(1, 1, 1, len(SEND_FRACTIONS), 1)
    if pair_turns is None:
        turns = planets.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1))
    else:
        turns = pair_turns[:, :, :, SEND_ACTIONS].unsqueeze(-1).to(dtype=dtype, device=device) / 20.0
    if pair_reachable_mask is None:
        reachable = planets.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1))
    else:
        reachable = pair_reachable_mask[:, :, :, SEND_ACTIONS].unsqueeze(-1).to(dtype=dtype, device=device)
    ships = planets[:, :, 10]
    src_ships = ships.unsqueeze(2).unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS)).unsqueeze(-1)
    tgt_ships = ships.unsqueeze(1).unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS)).unsqueeze(-1)
    sent_proxy = src_ships * frac
    return torch.cat(
        [
            frac.expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1),
            is_half.expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1),
            is_all.expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1),
            (1.0 - frac).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), 1),
            turns,
            reachable,
            sent_proxy,
            tgt_ships,
        ],
        dim=-1,
    )


class NormalizedEncoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.attn_alpha = nn.Parameter(torch.tensor(0.05))
        self.mlp_alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, key_padding_mask=None):
        x = hypersphere_norm(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = hypersphere_norm(x + self.attn_alpha * hypersphere_norm(attn_out))
        mlp_out = self.mlp(x)
        return hypersphere_norm(x + self.mlp_alpha * hypersphere_norm(mlp_out))


class NormalizedEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([NormalizedEncoderLayer(hidden, heads) for _ in range(layers)])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return x


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
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES + PAIR_OUTCOME_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
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
        pair_outcome_features=None,
        planet_timeline_features=None,
        owner_ids=None,
        player_ids=None,
        alive_players=None,
    ):
        batch = planets.shape[0]
        encoded = self.planet_encoder(with_timeline_features(planets, planet_timeline_features))
        mask = planet_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        src = encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., PLANET_XY_SLICE]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_base = torch.cat([pair, pair_geom, g], dim=-1)
        pair_base = pair_base.unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), -1)
        if pair_outcome_features is None:
            outcome = pair_base.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), PAIR_OUTCOME_FEATURES))
        else:
            outcome = pair_outcome_features[:, :, :, SEND_ACTIONS, :].to(dtype=pair_base.dtype, device=pair_base.device)
        pair_logits = self.pair_head(torch.cat([pair_base, outcome], dim=-1)).reshape(batch, -1)
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
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES + PAIR_OUTCOME_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
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
        pair_outcome_features=None,
        planet_timeline_features=None,
        owner_ids=None,
        player_ids=None,
        alive_players=None,
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

        xy = planets[..., PLANET_XY_SLICE]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_base = torch.cat([pair, pair_geom, g], dim=-1)
        pair_base = pair_base.unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), -1)
        if pair_outcome_features is None:
            outcome = pair_base.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), PAIR_OUTCOME_FEATURES))
        else:
            outcome = pair_outcome_features[:, :, :, SEND_ACTIONS, :].to(dtype=pair_base.dtype, device=pair_base.device)
        pair_logits = self.pair_head(torch.cat([pair_base, outcome], dim=-1)).reshape(batch, -1)
        state = torch.cat([global_encoded, globals_], dim=-1)
        logits = torch.cat([self.noop_head(state), pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(state).squeeze(-1)
        return logits, value


class NgptActionFeaturePolicy(nn.Module):
    """Entity transformer with normalized residual blocks and explicit launch-action features."""

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
        self.transformer = NormalizedEncoder(hidden, layers, heads)
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES + PAIR_OUTCOME_FEATURES + ACTION_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
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
        pair_outcome_features=None,
        planet_timeline_features=None,
        owner_ids=None,
        player_ids=None,
        alive_players=None,
    ):
        batch = planets.shape[0]
        planet_tokens = self.planet_encoder(with_timeline_features(planets, planet_timeline_features))
        global_token = self.global_encoder(globals_).unsqueeze(1)
        tokens = torch.cat([global_token, planet_tokens], dim=1)
        global_valid = torch.ones(batch, 1, dtype=torch.bool, device=planet_mask.device)
        valid = torch.cat([global_valid, planet_mask.bool()], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        global_encoded = encoded[:, 0]
        planet_encoded = encoded[:, 1:]

        src = planet_encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = planet_encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., PLANET_XY_SLICE]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_base = torch.cat([pair, pair_geom, g], dim=-1)
        pair_base = pair_base.unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), -1)
        if pair_outcome_features is None:
            outcome = pair_base.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), PAIR_OUTCOME_FEATURES))
        else:
            outcome = pair_outcome_features[:, :, :, SEND_ACTIONS, :].to(dtype=pair_base.dtype, device=pair_base.device)
        action_features = action_feature_tensor(planets, pair_turns, pair_reachable_mask)
        pair_logits = self.pair_head(torch.cat([pair_base, outcome, action_features], dim=-1)).reshape(batch, -1)
        state = torch.cat([global_encoded, globals_], dim=-1)
        logits = torch.cat([self.noop_head(state), pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(state).squeeze(-1)
        return logits, value


class NgptActionFeatureIdentityPolicy(nn.Module):
    """4p policy variant with absolute owner, acting-player, and alive-count embeddings."""

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
        self.owner_embedding = nn.Embedding(5, hidden)
        self.player_embedding = nn.Embedding(4, hidden)
        self.alive_embedding = nn.Embedding(5, hidden)
        self.transformer = NormalizedEncoder(hidden, layers, heads)
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES + PAIR_OUTCOME_FEATURES + ACTION_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
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
        pair_outcome_features=None,
        planet_timeline_features=None,
        owner_ids=None,
        player_ids=None,
        alive_players=None,
    ):
        batch = planets.shape[0]
        device = planets.device
        if owner_ids is None:
            owner_ids = torch.zeros((batch, MAX_PLANETS), dtype=torch.long, device=device)
        else:
            owner_ids = owner_ids.to(device=device, dtype=torch.long).clamp(0, 4)
        if player_ids is None:
            player_ids = torch.zeros((batch,), dtype=torch.long, device=device)
        else:
            player_ids = player_ids.to(device=device, dtype=torch.long).view(batch).clamp(0, 3)
        if alive_players is None:
            alive_players = torch.zeros((batch,), dtype=torch.long, device=device)
        else:
            alive_players = alive_players.to(device=device, dtype=torch.long).view(batch).clamp(0, 4)

        player_context = self.player_embedding(player_ids)
        alive_context = self.alive_embedding(alive_players)
        planet_tokens = (
            self.planet_encoder(with_timeline_features(planets, planet_timeline_features))
            + self.owner_embedding(owner_ids)
            + player_context.unsqueeze(1)
        )
        global_token = (self.global_encoder(globals_) + player_context + alive_context).unsqueeze(1)
        tokens = torch.cat([global_token, planet_tokens], dim=1)
        global_valid = torch.ones(batch, 1, dtype=torch.bool, device=planet_mask.device)
        valid = torch.cat([global_valid, planet_mask.bool()], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        global_encoded = encoded[:, 0]
        planet_encoded = encoded[:, 1:]

        src = planet_encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = planet_encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., PLANET_XY_SLICE]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)

        pair_base = torch.cat([pair, pair_geom, g], dim=-1)
        pair_base = pair_base.unsqueeze(3).expand(batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), -1)
        if pair_outcome_features is None:
            outcome = pair_base.new_zeros((batch, MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS), PAIR_OUTCOME_FEATURES))
        else:
            outcome = pair_outcome_features[:, :, :, SEND_ACTIONS, :].to(dtype=pair_base.dtype, device=pair_base.device)
        action_features = action_feature_tensor(planets, pair_turns, pair_reachable_mask)
        pair_logits = self.pair_head(torch.cat([pair_base, outcome, action_features], dim=-1)).reshape(batch, -1)
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
    if model_type == "entity_transformer_ngpt_action_features":
        return NgptActionFeaturePolicy(
            hidden=hidden,
            layers=transformer_layers,
            heads=transformer_heads,
        )
    if model_type == "entity_transformer_ngpt_action_features_identity":
        return NgptActionFeatureIdentityPolicy(
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
        "pair_turns": torch.as_tensor(encoded.pair_turns, dtype=torch.float32, device=device).unsqueeze(0),
        "pair_reachable_mask": torch.as_tensor(
            encoded.pair_reachable_mask, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "pair_outcome_features": torch.as_tensor(
            encoded.pair_outcome_features, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "planet_timeline_features": torch.as_tensor(
            encoded.planet_timeline_features, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "owner_ids": torch.as_tensor(encoded.owner_ids, dtype=torch.long, device=device).unsqueeze(0),
        "player_ids": torch.as_tensor([encoded.player_id], dtype=torch.long, device=device),
        "alive_players": torch.as_tensor([encoded.alive_players], dtype=torch.long, device=device),
    }
