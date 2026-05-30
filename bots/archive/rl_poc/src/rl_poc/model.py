"""Tiny actor-critic.

Per-planet encoder (shared MLP) → embedding for every slot. The actor
scores every (source, target) pair via a small pair MLP and adds a per-source
no-op logit. The critic mean-pools encoded planets and runs a 2-layer MLP.

No attention, no apollo features — the POC point is the training loop, not
the representation. The full design uses self-attention + apollo-grounded
features (see docs/rl_design.md)."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import EMBED_DIM, GLOBAL_FEATURES, PLANET_FEATURES


class ActorCritic(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 64):
        super().__init__()
        self.planet_enc = nn.Sequential(
            nn.Linear(PLANET_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.global_enc = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Pair scorer takes (src_embed, tgt_embed, globals_embed) -> logit.
        self.pair_scorer = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # No-op scorer takes (src_embed, globals_embed) -> logit.
        self.noop_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Value head: mean-pool encoded planets, concat globals, MLP -> scalar.
        self.value_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, planet_table, planet_mask, globals_):
        """Returns (planet_emb, board_emb, globals_emb).

        planet_table: (B, MAX_PLANETS, PLANET_FEATURES)
        planet_mask:  (B, MAX_PLANETS)
        globals_:     (B, GLOBAL_FEATURES)
        """
        planet_emb = self.planet_enc(planet_table)  # (B, P, E)
        # Mean-pool over valid slots for the board summary.
        mask_f = planet_mask.unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        board_emb = (planet_emb * mask_f).sum(dim=1) / denom  # (B, E)
        globals_emb = self.global_enc(globals_)  # (B, E)
        return planet_emb, board_emb, globals_emb

    def actor_logits(
        self,
        planet_emb,
        planet_mask,
        mine_mask,
        globals_emb,
    ):
        """Per-source logits over targets ∪ {no-op}.

        Returns:
          logits: (B, MAX_PLANETS, MAX_PLANETS + 1) — last column is no-op.
                  Masked positions are -inf.
          source_mask: (B, MAX_PLANETS) — 1 where source is owned by me.
        """
        b, p, e = planet_emb.shape
        # Broadcast: src is dim 1, tgt is dim 2.
        src = planet_emb.unsqueeze(2).expand(b, p, p, e)  # (B, P, P, E)
        tgt = planet_emb.unsqueeze(1).expand(b, p, p, e)  # (B, P, P, E)
        glb = globals_emb.unsqueeze(1).unsqueeze(2).expand(b, p, p, e)
        pair_in = torch.cat([src, tgt, glb], dim=-1)  # (B, P, P, 3E)
        pair_logits = self.pair_scorer(pair_in).squeeze(-1)  # (B, P, P)

        # No-op logit per source.
        src_only = planet_emb  # (B, P, E)
        glb_b = globals_emb.unsqueeze(1).expand(b, p, e)
        noop_in = torch.cat([src_only, glb_b], dim=-1)
        noop_logits = self.noop_scorer(noop_in)  # (B, P, 1)

        logits = torch.cat([pair_logits, noop_logits], dim=-1)  # (B, P, P+1)

        # Mask invalid targets (mask=0) and self-targeting.
        tgt_mask = planet_mask.unsqueeze(1).expand(b, p, p)  # (B, P, P)
        eye = torch.eye(p, device=planet_mask.device).unsqueeze(0)  # (1, P, P)
        valid_tgt = tgt_mask * (1.0 - eye)
        valid_full = torch.cat(
            [valid_tgt, torch.ones(b, p, 1, device=planet_mask.device)],
            dim=-1,
        )  # (B, P, P+1)
        neg_inf = torch.full_like(logits, float("-inf"))
        logits = torch.where(valid_full > 0.5, logits, neg_inf)
        return logits, mine_mask

    def value(self, board_emb, globals_emb):
        v_in = torch.cat([board_emb, globals_emb], dim=-1)
        return self.value_head(v_in).squeeze(-1)

    def forward(
        self,
        planet_table,
        planet_mask,
        mine_mask,
        globals_,
    ):
        planet_emb, board_emb, globals_emb = self.encode(
            planet_table, planet_mask, globals_
        )
        logits, source_mask = self.actor_logits(
            planet_emb, planet_mask, mine_mask, globals_emb
        )
        v = self.value(board_emb, globals_emb)
        return logits, source_mask, v
