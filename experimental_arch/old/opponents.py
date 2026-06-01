from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]


def noop_agent(obs):
    return []


def random_agent(obs):
    player = obs.get("player", 0)
    moves = []
    for p in obs.get("planets", []):
        pid, owner, _x, _y, _radius, ships, _production = p
        if owner != player or ships < 12:
            continue
        send = max(1, int(ships * 0.35))
        moves.append([int(pid), random.random() * math.tau, send])
    return moves[:3]


def nearest_sniper_agent(obs):
    player = obs.get("player", 0)
    planets = obs.get("planets", [])
    targets = [p for p in planets if int(p[1]) != player]
    if not targets:
        return []

    moves = []
    for src in planets:
        pid, owner, x, y, _radius, ships, _production = src
        if int(owner) != player:
            continue
        target = min(targets, key=lambda t: (float(t[2]) - x) ** 2 + (float(t[3]) - y) ** 2)
        needed = int(target[5]) + 1
        if int(ships) >= needed:
            angle = math.atan2(float(target[3]) - y, float(target[2]) - x)
            moves.append([int(pid), angle, needed])
    return moves


SCRIPTED_OPPONENTS = {
    "noop": noop_agent,
    "random": random_agent,
    "nearest": nearest_sniper_agent,
}


def bot_entry(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    if path.is_dir() and (path / "main.py").is_file():
        return path / "main.py"

    direct = ROOT / "bots" / name_or_path / "main.py"
    if direct.is_file():
        return direct

    # Prefer our curated baseline version when duplicate names exist.
    for group_name in ("mine", "baselines", "external"):
        candidate = ROOT / "bots" / group_name / name_or_path / "main.py"
        if candidate.is_file():
            return candidate

    for group in (ROOT / "bots").iterdir():
        candidate = group / name_or_path / "main.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find opponent bot '{name_or_path}'")


def load_bot_agent(name_or_path: str) -> Callable[[dict], list[list[float]]]:
    path = bot_entry(name_or_path)
    module_name = f"rl_opponent_{path.parent.parent.name}_{path.parent.name}_{abs(hash(path))}"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import opponent from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} does not define agent(obs)")
    return module.agent


def get_opponent(name_or_agent: str | Callable[[dict], list[list[float]]]):
    if callable(name_or_agent):
        return name_or_agent
    if name_or_agent in SCRIPTED_OPPONENTS:
        return SCRIPTED_OPPONENTS[name_or_agent]
    return load_bot_agent(name_or_agent)
