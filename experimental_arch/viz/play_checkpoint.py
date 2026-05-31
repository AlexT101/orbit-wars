"""Render a trained SB3 checkpoint game to Kaggle's HTML visualizer.

Run from repo root or `experimental_arch/`:

    python experimental_arch/viz/play_checkpoint.py
    python experimental_arch/viz/play_checkpoint.py --opponent random --seed 7
    python experimental_arch/viz/play_checkpoint.py --hero-agent experimental_arch/heuristic/closest_all.py
    python experimental_arch/viz/play_checkpoint.py --checkpoint train/checkpoints/galaxy_selfplay/final.zip
    python experimental_arch/viz/play_checkpoint.py --opponent-checkpoint train/checkpoints/galaxy_selfplay/self_play_opponent.zip

Default behavior is stochastic self-play: the hero model plays as player 0,
the opponent is `self` (the same checkpoint as player 1), and both checkpoint
agents sample from their policy distributions. Pass `--hero-deterministic`
and/or `--opponent-deterministic` to use argmax/eval-mode actions instead.

The opponent can also be a bot name under `bots/` or another checkpoint. We
drive `env_engine` and the Kaggle env with the same actions, then render the
Kaggle env so the animation matches the real rollout.

For feature-debugging, the script prints the encoder's `t` and `t_resolved`
values before every action. `t_resolved` is shown as an absolute turn:
`obs["step"] + frame_offsets[-1]`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from sb3_contrib import MaskablePPO

EXPERIMENTAL_ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train"
DEFAULT_OUT = EXPERIMENTAL_ARCH_DIR / "viz" / "orbit_wars_game.html"
DEFAULT_CHECKPOINT_DIR = TRAIN_DIR / "checkpoints" / "galaxy_selfplay"

sys.path.insert(0, str(TRAIN_DIR))

from arch import GalaxyMaskablePolicy  # noqa: F401 - needed by MaskablePPO.load
from env import scores_from_obs
from features import decode_action, encode_features, flat_action_mask
from opponents import BotOpponent
from orbit_wars_engine import OrbitWarsEngine

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make


PLAYERS = 2
MAX_STEPS = 500


class Agent(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: dict) -> list[list[float]]:
        ...


class CheckpointAgent:
    def __init__(self, checkpoint: Path, player: int, deterministic: bool, device: str):
        self.name = checkpoint.stem
        self.checkpoint = checkpoint
        self.player = player
        self.deterministic = deterministic
        self.model = MaskablePPO.load(checkpoint, device=device)

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> list[list[float]]:
        model_obs, feat = encode_features(obs, player=self.player)
        action, _ = self.model.predict(
            model_obs,
            deterministic=self.deterministic,
            action_masks=flat_action_mask(feat),
        )
        return decode_action(feat, np.asarray(action))


class PythonAgent:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self._module_name = f"viz_agent__{path.stem}"
        self._agent = None
        self.reset()

    def reset(self) -> None:
        spec = importlib.util.spec_from_file_location(self._module_name, self.path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Python agent: {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self._module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "agent"):
            raise RuntimeError(f"Python agent has no agent(obs) function: {self.path}")
        self._agent = module.agent

    def act(self, obs: dict) -> list[list[float]]:
        assert self._agent is not None
        return self._agent(obs)


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for base in (Path.cwd(), EXPERIMENTAL_ARCH_DIR, REPO_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return Path.cwd() / p


def default_checkpoint() -> Path:
    for name in ("latest.zip", "final.zip", "self_play_opponent.zip"):
        path = DEFAULT_CHECKPOINT_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint found in {DEFAULT_CHECKPOINT_DIR}; pass --checkpoint explicitly"
    )


def make_opponent(
    name: str,
    *,
    player: int,
    self_checkpoint: Path,
    deterministic: bool,
    device: str,
) -> Agent:
    if name == "self":
        return CheckpointAgent(self_checkpoint, player=player, deterministic=deterministic, device=device)
    path = resolve_path(name)
    if path.exists() and path.suffix == ".zip":
        return CheckpointAgent(path, player=player, deterministic=deterministic, device=device)
    if path.exists() and path.suffix == ".py":
        return PythonAgent(path)
    return BotOpponent(name)


def policy_label(agent: Agent) -> str:
    if isinstance(agent, CheckpointAgent):
        return "deterministic" if agent.deterministic else "stochastic"
    if isinstance(agent, PythonAgent):
        return "python"
    return "bot"


def frame_timing(obs: dict, player: int = 0) -> tuple[int, int, int, list[int]]:
    _model_obs, feat = encode_features(obs, player=player)
    offsets = [int(x) for x in feat["frame_offsets"]]
    t = int(obs.get("step", 0))
    t_resolved_offset = offsets[-1]
    return t, t_resolved_offset, t + t_resolved_offset, offsets


def print_frame_timing(label: str, obs: dict, player: int = 0) -> None:
    t, t_resolved_offset, t_resolved, offsets = frame_timing(obs, player=player)
    print(
        f"{label}: t={t} t_resolved={t_resolved} "
        f"(resolved_offset={t_resolved_offset}, frame_offsets={offsets})"
    )


def windows_link(path: Path) -> str | None:
    if not shutil.which("wslpath"):
        return None
    try:
        win = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if win.startswith("\\\\"):
        return "\nfile://" + win[2:].replace("\\", "/")
    return "\nfile:///" + win.replace("\\", "/")


def play_and_render(
    *,
    hero: Agent,
    opponent: Agent,
    seed: int,
    out: Path,
    max_steps: int,
) -> None:
    hero.reset()
    opponent.reset()

    engine = OrbitWarsEngine(num_players=PLAYERS)
    eng_obs = engine.reset(seed=seed)["observations"]
    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    kenv.reset(PLAYERS)

    steps = 0
    for _ in range(max_steps):
        if kenv.done:
            break
        print_frame_timing(f"step {steps} frames", eng_obs[0], player=0)
        actions = [hero.act(eng_obs[0]), opponent.act(eng_obs[1])]
        kenv.step(actions)
        out_step = engine.step(actions)
        eng_obs = out_step["observations"]
        steps += 1
        if out_step["done"]:
            break

    scores = scores_from_obs(eng_obs[0], num_players=PLAYERS)
    total = sum(scores)
    share0 = scores[0] / total if total else 0.0
    if scores[0] > scores[1]:
        verdict = "checkpoint wins"
    elif scores[1] > scores[0]:
        verdict = f"{opponent.name} wins"
    else:
        verdict = "tie"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(kenv.render(mode="html"), encoding="utf-8")

    print(f"hero:            {hero.name}")
    print(f"opponent:        {opponent.name}")
    print(f"seed:            {seed}")
    print(f"hero policy:     {policy_label(hero)}")
    print(f"opponent policy: {policy_label(opponent)}")
    print(f"result:          {verdict} after {steps} steps")
    print(f"ships:           p0={scores[0]} p1={scores[1]} p0_share={share0:.3f}")
    print(f"html:            {out}")
    link = windows_link(out)
    if link:
        print(f"windows:         {link}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--hero-agent", type=Path, default=None, help="Python agent file with agent(obs)")
    parser.add_argument(
        "--opponent",
        default="self",
        help="'self' by default, meaning the same checkpoint as player 1; can also be a bot name or checkpoint path",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--hero-deterministic",
        action="store_true",
        help="use deterministic argmax/eval-mode actions for the hero; default is stochastic sampling",
    )
    parser.add_argument(
        "--opponent-deterministic",
        action="store_true",
        help="use deterministic argmax/eval-mode actions for checkpoint opponents; default is stochastic sampling",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if args.hero_agent is not None:
        hero_path = resolve_path(args.hero_agent)
        if not hero_path.exists():
            raise FileNotFoundError(f"hero agent does not exist: {hero_path}")
        hero = PythonAgent(hero_path)
        self_checkpoint = resolve_path(args.checkpoint) if args.checkpoint is not None else hero_path
    else:
        checkpoint = resolve_path(args.checkpoint) if args.checkpoint is not None else default_checkpoint()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        hero = CheckpointAgent(
            checkpoint,
            player=0,
            deterministic=args.hero_deterministic,
            device=args.device,
        )
        self_checkpoint = checkpoint

    opponent_ref = str(args.opponent_checkpoint) if args.opponent_checkpoint is not None else args.opponent
    if opponent_ref == "self" and args.hero_agent is not None:
        opponent_ref = str(resolve_path(args.hero_agent))
    opponent = make_opponent(
        opponent_ref,
        player=1,
        self_checkpoint=self_checkpoint,
        deterministic=args.opponent_deterministic,
        device=args.device,
    )
    play_and_render(
        hero=hero,
        opponent=opponent,
        seed=args.seed,
        out=resolve_path(args.out),
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
