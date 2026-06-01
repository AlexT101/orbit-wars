from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Protocol

from sb3_contrib import MaskablePPO

from arch import GalaxyMaskablePolicy  # noqa: F401 - needed when loading saved policies
from constants import BOTS_DIR
from features import decode_action, encode_features, flat_action_mask


class Opponent(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: dict) -> list[list[float]]:
        ...


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

    def act(self, obs: dict) -> list[list[float]]:
        with _suppress_stdout():
            return self.agent(obs)


class ModelOpponent:
    def __init__(
        self,
        checkpoint: str | Path | None,
        device: str = "auto",
        fallback: str = "hellburner",
        deterministic: bool = False,
    ):
        self.name = "self_play"
        self.device = device
        self.deterministic = deterministic
        self.fallback = BotOpponent(fallback)
        self.model: MaskablePPO | None = None
        self.checkpoint: Path | None = None
        if checkpoint is not None:
            self.set_checkpoint(checkpoint)

    def set_checkpoint(self, checkpoint: str | Path | None) -> None:
        if checkpoint is None:
            self.model = None
            self.checkpoint = None
            return
        path = Path(checkpoint)
        if not path.exists():
            self.model = None
            self.checkpoint = None
            return
        self.model = MaskablePPO.load(path, device=self.device)
        self.checkpoint = path

    def reset(self) -> None:
        self.fallback.reset()

    def act(self, obs: dict) -> list[list[float]]:
        if self.model is None:
            return self.fallback.act(obs)
        player = int(obs.get("player", 1))
        model_obs, feat = encode_features(obs, player=player)
        action, _ = self.model.predict(
            model_obs,
            deterministic=self.deterministic,
            action_masks=flat_action_mask(feat),
        )
        return decode_action(feat, action)
