from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.distributions import Categorical


_AGENT = None


def _repo_root() -> Path:
    env = os.environ.get("OSTEO_ORBIT_WARS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    starts = [Path.cwd().resolve()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve())
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / "run_match.py").is_file() and (parent / "bots").is_dir():
                return parent
    if "__file__" in globals():
        here = Path(__file__).resolve()
        return here.parents[3]
    for parent in Path.cwd().resolve().parents:
        if (parent / "run_match.py").is_file() and (parent / "bots").is_dir():
            return parent
    return Path.cwd().resolve()


ROOT = _repo_root()
EXPERIMENTAL_ARCH = ROOT / "experimental_arch"
TRAIN_TRANSFORMER = EXPERIMENTAL_ARCH / "train_transformer"
DEFAULT_CHECKPOINT = (
    EXPERIMENTAL_ARCH
    / "imitation_learning"
    / "checkpoints"
    / "osteo_bc_transformer"
    / "latest.pt"
)

if str(TRAIN_TRANSFORMER) not in sys.path:
    sys.path.insert(0, str(TRAIN_TRANSFORMER))

from features import decode_move, encode_obs  # noqa: E402
from model import build_policy  # noqa: E402

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")


def _checkpoint_path() -> Path:
    return Path(os.environ.get("OSTEO_IL_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser().resolve()


def _device() -> torch.device:
    return torch.device(os.environ.get("OSTEO_IL_DEVICE", "cpu"))


def _deterministic() -> bool:
    value = os.environ.get("OSTEO_IL_DETERMINISTIC", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    last_error = None
    for _ in range(5):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"failed to load IL checkpoint {path}: {last_error}")


class OsteoILAgent:
    def __init__(self) -> None:
        self.checkpoint_path = _checkpoint_path()
        self.device = _device()
        self.deterministic = _deterministic()
        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        config = checkpoint.get("config", {})
        self.model_type = config.get("model", "entity_transformer_temporal")
        self.model = build_policy(
            self.model_type,
            hidden=int(config.get("hidden", 128)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            transformer_heads=int(config.get("transformer_heads", 4)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def act(self, obs: dict) -> list[list[float]]:
        encoded = encode_obs(obs, player=int(obs.get("player", 0)))
        batch = {
            "planets": torch.as_tensor(encoded.planets, dtype=torch.float32, device=self.device).unsqueeze(0),
            "planet_mask": torch.as_tensor(encoded.planet_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            "globals_": torch.as_tensor(encoded.globals, dtype=torch.float32, device=self.device).unsqueeze(0),
            "action_mask": torch.as_tensor(encoded.action_mask, dtype=torch.bool, device=self.device).unsqueeze(0),
            "pair_turns": torch.as_tensor(encoded.pair_turns, dtype=torch.float32, device=self.device).unsqueeze(0),
            "pair_reachable_mask": torch.as_tensor(
                encoded.pair_reachable_mask,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "pair_outcome_features": torch.as_tensor(
                encoded.pair_outcome_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "planet_timeline_features": torch.as_tensor(
                encoded.planet_timeline_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
        }
        with torch.inference_mode():
            logits, _value = self.model(**batch)
            if self.deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        return decode_move(obs, int(action.item()))


def _get_agent() -> OsteoILAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = OsteoILAgent()
    return _AGENT


def agent(obs, config=None):
    return _get_agent().act(obs)
