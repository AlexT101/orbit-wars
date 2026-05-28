"""Translate the policy's logits into Orbit Wars moves.

For each owned planet we sample (or argmax) over `targets ∪ {no-op}`,
then convert the chosen (src, tgt) into a fleet order with
`ships = target.ships + 1` (capped at `source.ships - 1`) and angle aimed
at the target's current position.

We keep a parallel record of (source_slot, chosen_idx, logprob) per
sampled source so the trainer can recompute new logprobs deterministically."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .features import ObsView


@dataclass
class ActionRecord:
    """What the trainer needs to recompute logprobs after a parameter update.

    `chosen_idx` is in [0, MAX_PLANETS]; MAX_PLANETS = no-op."""

    source_slots: np.ndarray  # (K,) int64 — source slot indices we sampled at
    chosen_idx: np.ndarray  # (K,) int64 — picked target slot or noop index
    logprob_sum: float  # sum of log π(a_k | s) across the K sources


def _target_ships(source_planet, target_planet) -> int:
    """How many ships to send. Just enough to capture + buffer of 1."""
    src_ships = int(source_planet[5])
    tgt_ships = int(target_planet[5])
    # `target.ships + 1` flips even neutrals; cap at all-but-1 to keep a seed.
    want = tgt_ships + 1
    return max(1, min(want, src_ships - 1))


def select_moves(
    obs_view: ObsView,
    logits_2d: torch.Tensor,  # (P, P+1) — already masked for one batch element
    mine_mask: torch.Tensor,  # (P,)
    *,
    sample: bool,
) -> tuple[list, ActionRecord]:
    """Pick a move per owned source planet.

    Returns the list of Kaggle [[from_id, angle, ships], ...] orders and a
    record of the sampled action distribution for PPO."""

    p = logits_2d.shape[0]
    moves: list = []
    source_slots: list[int] = []
    chosen_idx: list[int] = []
    logprob_sum = torch.zeros((), dtype=torch.float32, device=logits_2d.device)

    mine_indices = torch.nonzero(mine_mask > 0.5, as_tuple=False).flatten().tolist()
    for src_slot in mine_indices:
        if src_slot >= len(obs_view.planets_raw):
            continue
        source_planet = obs_view.planets_raw[src_slot]
        if int(source_planet[5]) <= 1:
            # Nothing to send — skip but still emit a no-op so the
            # gradient flows for this source. (Else the policy never
            # learns the no-op for low-ship sources.)
            row = logits_2d[src_slot]
            row_finite = torch.where(
                torch.isinf(row), torch.full_like(row, -1e9), row
            )
            logp = F.log_softmax(row_finite, dim=-1)
            noop_idx = p  # last column
            logprob_sum = logprob_sum + logp[noop_idx]
            source_slots.append(src_slot)
            chosen_idx.append(noop_idx)
            continue

        row = logits_2d[src_slot]
        # Replace -inf with very-negative so log_softmax is well-defined.
        row_finite = torch.where(
            torch.isinf(row), torch.full_like(row, -1e9), row
        )
        logp = F.log_softmax(row_finite, dim=-1)
        probs = logp.exp()
        if sample:
            idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            idx = int(torch.argmax(probs).item())
        logprob_sum = logprob_sum + logp[idx]
        source_slots.append(src_slot)
        chosen_idx.append(idx)

        if idx == p:  # no-op
            continue
        if idx >= len(obs_view.planets_raw):
            continue
        target_planet = obs_view.planets_raw[idx]
        if int(target_planet[0]) == int(source_planet[0]):
            continue
        ships = _target_ships(source_planet, target_planet)
        if ships <= 0:
            continue
        angle = math.atan2(
            float(target_planet[3]) - float(source_planet[3]),
            float(target_planet[2]) - float(source_planet[2]),
        )
        moves.append([int(source_planet[0]), float(angle), int(ships)])

    record = ActionRecord(
        source_slots=np.asarray(source_slots, dtype=np.int64),
        chosen_idx=np.asarray(chosen_idx, dtype=np.int64),
        logprob_sum=float(logprob_sum.detach().cpu().item()),
    )
    return moves, record
