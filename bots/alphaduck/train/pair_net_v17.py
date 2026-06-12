"""v17 model: planet+fleet tokens, structured attention mask, multi-head pair bias.

Token layout per batch: [global_token, planet_0..planet_{N-1}, fleet_0..fleet_{F-1}]
- N = n_planet_max, F = n_fleet_max — taken from the ckpt at load time
  (defaults: N=50, F=128). Total seq len T = 1 + N + F.

Attention mask:
- planet ↔ planet  : allowed (with multi-head bias from F_PAIR)
- planet ↔ global  : allowed
- global ↔ global  : allowed (trivial)
- fleet → target_planet : allowed
- target_planet → fleet : allowed
- fleet ↔ everything else : blocked

Heads:
- Pair head over planet outputs: bilinear (q_i · k_j) / sqrt(d) + b (scalar).
- Value head: tanh(MLP([cls_h, mean_pool(planet), max_pool(planet)])).
- Noop head: per-planet Linear(d → 1).

Use a custom MultiheadAttention wrapper to inject per-head additive bias on
planet-planet edges. The bias matrix is computed once from pair_feats and
shared across encoder layers.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredEncoderLayer(nn.Module):
    """Transformer encoder layer with optional additive attention bias on
    pre-softmax attention logits."""

    def __init__(self, d_model: int, nhead: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )

    def forward(self, x, attn_mask, attn_bias):
        """x: (B, T, D)
           attn_mask: (B, T, T) bool — True = blocked.
           attn_bias: (B, n_heads, T, T) float — added to attention logits.
        """
        B, T, D = x.shape
        H = self.nhead; Hd = self.head_dim
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, H, Hd).transpose(1, 2)  # (B,H,T,Hd)
        k = self.k_proj(h).view(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(h).view(B, T, H, Hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Hd)  # (B,H,T,T)
        if attn_bias is not None:
            scores = scores + attn_bias
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)             # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        x = x + self.drop(out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class PairNetV17(nn.Module):
    """Encoder + bilinear pair head + value head + noop head, with planet
    and fleet token types and structured attention.
    """

    def __init__(self, f_planet: int, f_fleet: int, f_global: int, f_pair: int,
                 n_planet_max: int, n_fleet_max: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.N = n_planet_max
        self.F = n_fleet_max

        self.planet_proj = nn.Linear(f_planet, d_model)
        self.fleet_proj = nn.Linear(f_fleet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        # Type embeddings so encoder can distinguish (cheap; learned).
        self.type_emb = nn.Embedding(3, d_model)   # 0=global, 1=planet, 2=fleet

        # Per-layer attention bias derived from pair_feats — single shared linear,
        # zero-init so the encoder starts as vanilla self-attention.
        self.pair_bias_proj = nn.Linear(f_pair, n_heads)
        nn.init.zeros_(self.pair_bias_proj.weight)
        nn.init.zeros_(self.pair_bias_proj.bias)

        self.layers = nn.ModuleList([
            StructuredEncoderLayer(d_model, n_heads, ff, dropout)
            for _ in range(n_layers)
        ])

        # Output heads (operate on planet outputs).
        self.q_pair = nn.Linear(d_model, d_model)
        self.k_pair = nn.Linear(d_model, d_model)
        self.pair_bias_scalar = nn.Parameter(torch.zeros(1))

        self.noop_head = nn.Linear(d_model, 1)

        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _build_attention(self, B, planet_mask, fleet_mask, fleet_tgt_idx,
                         pair_feats, device):
        """Returns (attn_mask, attn_bias) for the structured encoder.
        attn_mask: (B, T, T) bool True=blocked.
        attn_bias: (B, H, T, T) additive bias.
        T = 1 + N + F.
        """
        N, F_ = self.N, self.F
        T = 1 + N + F_
        attn_mask = torch.ones(B, T, T, dtype=torch.bool, device=device)  # True=blocked
        attn_bias = torch.zeros(B, self.n_heads, T, T, device=device)

        # Indices ranges:
        #   0           = global
        #   1..N        = planets
        #   N+1..N+F    = fleets
        G = 0
        p_start, p_end = 1, 1 + N
        f_start, f_end = 1 + N, 1 + N + F_

        # Allow global ↔ global
        attn_mask[:, G, G] = False
        # Allow planet ↔ planet (will further mask by planet_mask)
        attn_mask[:, p_start:p_end, p_start:p_end] = False
        # Allow planet ↔ global
        attn_mask[:, G, p_start:p_end] = False
        attn_mask[:, p_start:p_end, G] = False
        # Mask out padded planets:  block ANY edge involving an invalid planet
        # (we treat unmasked == invalid; planet_mask True means real)
        pm = planet_mask  # (B, N)
        invalid_p = ~pm   # (B, N)
        # any-with-invalid_p on planet rows
        attn_mask[:, p_start:p_end, :] = attn_mask[:, p_start:p_end, :] | invalid_p[:, :, None]
        attn_mask[:, :, p_start:p_end] = attn_mask[:, :, p_start:p_end] | invalid_p[:, None, :]
        # Fleet ↔ target planet only.
        # fleet_tgt_idx: (B, F) with -1 for invalid fleets.
        fm = fleet_mask    # (B, F)
        valid_f = fm & (fleet_tgt_idx >= 0) & (fleet_tgt_idx < N)
        # Vectorized: gather all (b, fleet_pos, target_planet_pos) where valid_f is True,
        # then scatter False into the attn_mask in two index_put calls (f→p and p→f).
        b_arange = torch.arange(B, device=device).unsqueeze(1).expand(B, F_)
        j_arange = torch.arange(F_, device=device).unsqueeze(0).expand(B, F_)
        ti_clamped = fleet_tgt_idx.clamp(min=0, max=N - 1)
        b_sel = b_arange[valid_f]
        fi_sel = f_start + j_arange[valid_f]
        pi_sel = p_start + ti_clamped[valid_f]
        attn_mask[b_sel, fi_sel, pi_sel] = False
        attn_mask[b_sel, pi_sel, fi_sel] = False
        # Also allow fleet → self (so attention always has at least one slot;
        # otherwise softmax produces NaN on fully-blocked rows).
        diag_idx = torch.arange(T, device=device)
        attn_mask[:, diag_idx, diag_idx] = False

        # Multi-head pair bias on planet-planet edges only.
        # bias_pp: (B, n_heads, N, N) from pair_feats (B, N, N, F_pair)
        bias_pp = self.pair_bias_proj(pair_feats)          # (B, N, N, H)
        bias_pp = bias_pp.permute(0, 3, 1, 2).contiguous()  # (B, H, N, N)
        attn_bias[:, :, p_start:p_end, p_start:p_end] = bias_pp

        return attn_mask, attn_bias

    def forward(self, planet_feats, planet_mask, fleet_feats, fleet_mask,
                fleet_tgt_idx, globals_, pair_feats,
                return_value: bool = True):
        """Returns (policy_logits[B, N, N+1], value[B]).

        policy_logits[:, src, 0]     = noop logit for src
        policy_logits[:, src, 1+tgt] = launch-from-src-to-tgt logit
        Joint distribution = softmax over last dim (size N+1) per src.
        """
        B = planet_feats.shape[0]
        N, F_ = self.N, self.F
        device = planet_feats.device

        # token embeddings + type embeddings
        g_tok = self.global_proj(globals_).unsqueeze(1) + self.type_emb(torch.zeros(1, dtype=torch.long, device=device))
        p_tok = self.planet_proj(planet_feats) + self.type_emb(torch.ones(1, dtype=torch.long, device=device))
        f_tok = self.fleet_proj(fleet_feats) + self.type_emb(torch.full((1,), 2, dtype=torch.long, device=device))
        x = torch.cat([g_tok, p_tok, f_tok], dim=1)  # (B, T, D)

        attn_mask, attn_bias = self._build_attention(
            B, planet_mask, fleet_mask, fleet_tgt_idx, pair_feats, device,
        )

        for layer in self.layers:
            x = layer(x, attn_mask, attn_bias)

        # Pull out global + planet hidden states
        cls_h = x[:, 0, :]                       # (B, D)
        h_planet = x[:, 1:1 + N, :]              # (B, N, D)

        # Per-src noop logit (B, N, 1)
        noop_logits = self.noop_head(h_planet)
        # Per-(src, tgt) launch logit (B, N, N) via bilinear
        q = self.q_pair(h_planet); k = self.k_pair(h_planet)
        pair_logits = torch.bmm(q, k.transpose(-2, -1)) / (self.d_model ** 0.5) + self.pair_bias_scalar
        # Unified policy logits: noop at col 0, launch-to-tgt at col 1+tgt
        policy_logits = torch.cat([noop_logits, pair_logits], dim=-1)  # (B, N, N+1)
        result = (policy_logits,)

        if return_value:
            mask_f = planet_mask.unsqueeze(-1).float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean_pool = (h_planet * mask_f).sum(dim=1) / denom
            h_max = h_planet.masked_fill(~planet_mask.unsqueeze(-1), torch.finfo(h_planet.dtype).min)
            max_pool = h_max.max(dim=1).values
            value = torch.tanh(self.value_head(torch.cat([cls_h, mean_pool, max_pool], dim=-1))).squeeze(-1)
            result = result + (value,)
        if len(result) == 1:
            return result[0]
        return result
