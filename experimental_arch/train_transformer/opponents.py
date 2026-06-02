from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from constants import BOTS_DIR


AgentFn = Callable[[dict], list[list[float]]]


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


def noop_agent(obs: dict) -> list[list[float]]:
    return []


def random_agent(obs: dict) -> list[list[float]]:
    player = int(obs.get("player", 0))
    moves = []
    for p in obs.get("planets", []):
        pid, owner, _x, _y, _radius, ships, _production = p
        if int(owner) != player or int(ships) < 12:
            continue
        send = max(1, int(float(ships) * 0.35))
        moves.append([int(pid), random.random() * math.tau, send])
    return moves[:3]


def nearest_sniper_agent(obs: dict) -> list[list[float]]:
    player = int(obs.get("player", 0))
    planets = list(obs.get("planets", []) or [])
    targets = [p for p in planets if int(p[1]) != player]
    if not targets:
        return []

    moves = []
    for src in planets:
        pid, owner, x, y, _radius, ships, _production = src
        if int(owner) != player:
            continue
        target = min(targets, key=lambda t: (float(t[2]) - float(x)) ** 2 + (float(t[3]) - float(y)) ** 2)
        needed = int(target[5]) + 1
        if int(ships) >= needed:
            angle = math.atan2(float(target[3]) - float(y), float(target[2]) - float(x))
            moves.append([int(pid), angle, needed])
    return moves


SCRIPTED_OPPONENTS: dict[str, AgentFn] = {
    "noop": noop_agent,
    "random": random_agent,
    "nearest": nearest_sniper_agent,
}


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


def get_opponent(name_or_agent: str | AgentFn | Opponent) -> AgentFn | Opponent:
    if isinstance(name_or_agent, str):
        if name_or_agent in SCRIPTED_OPPONENTS:
            return SCRIPTED_OPPONENTS[name_or_agent]
        return BotOpponent(name_or_agent)
    return name_or_agent


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
        self.model: Any | None = None
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
        from sb3_contrib import MaskablePPO

        from arch import GalaxyMaskablePolicy  # noqa: F401 - needed when loading saved policies

        self.model = MaskablePPO.load(path, device=self.device)
        self.checkpoint = path

    def reset(self) -> None:
        self.fallback.reset()

    def act(self, obs: dict) -> list[list[float]]:
        if self.model is None:
            return self.fallback.act(obs)
        from features import decode_action, encode_features, flat_action_mask

        player = int(obs.get("player", 1))
        model_obs, feat = encode_features(obs, player=player)
        action, _ = self.model.predict(
            model_obs,
            deterministic=self.deterministic,
            action_masks=flat_action_mask(feat),
        )
        return decode_action(feat, action)
