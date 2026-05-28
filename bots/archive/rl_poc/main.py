"""Kaggle agent wrapper.

Loads `checkpoints/latest.pt` if present and runs the policy greedily. If
no checkpoint exists, falls back to a uniform-random policy over targets so
the bot is still loadable for sanity matches before any training."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_frame = inspect.currentframe()
if _frame is not None:
    _HERE = Path(_frame.f_code.co_filename).resolve().parent
else:
    _HERE = Path(__file__).resolve().parent

# Ensure the package layout (`src/rl_poc/`) is importable when this file is
# loaded directly by `run_match.py` (which `spec_from_file_location`s us in).
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from rl_poc.features import parse_obs
from rl_poc.model import ActorCritic
from rl_poc.policy import select_moves


_CKPT_PATH = _HERE / "checkpoints" / "latest.pt"
_MODEL: ActorCritic | None = None
_LOADED = False


def _load_model() -> ActorCritic | None:
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    _LOADED = True
    model = ActorCritic()
    if _CKPT_PATH.exists():
        try:
            ckpt = torch.load(_CKPT_PATH, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.eval()
            _MODEL = model
            print(f"[rl_poc] loaded {_CKPT_PATH.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[rl_poc] could not load checkpoint: {exc}", flush=True)
            _MODEL = None
    else:
        print(
            f"[rl_poc] no checkpoint at {_CKPT_PATH} — playing uniform random",
            flush=True,
        )
        _MODEL = None
    return _MODEL


def agent(obs):
    model = _load_model()
    ov = parse_obs(obs)
    if model is None:
        # No checkpoint — emit no moves so we at least don't throw ships into
        # the sun. Lets the harness verify wiring without a trained model.
        return []
    with torch.no_grad():
        pt = torch.from_numpy(ov.planet_table).unsqueeze(0)
        pm = torch.from_numpy(ov.planet_mask).unsqueeze(0)
        mm = torch.from_numpy(ov.mine_mask).unsqueeze(0)
        gl = torch.from_numpy(ov.globals_).unsqueeze(0)
        logits, _, _ = model(pt, pm, mm, gl)
        moves, _ = select_moves(ov, logits[0], mm[0], sample=False)
    return [list(m) for m in moves]
