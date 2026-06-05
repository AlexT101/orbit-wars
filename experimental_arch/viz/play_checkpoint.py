from __future__ import annotations

import contextlib
import argparse
import importlib.util
import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import torch
from torch.distributions import Categorical

HERO = "/home/ec2-user/orbit-wars-osteo/experimental_arch/imitation_learning/checkpoints/isaiah_bc_transformer/latest.pt"
OPPONENT = "hellburner"  # "self", .zip, .py, or bot name

HERO_DETERMINISTIC = False
OPPONENT_DETERMINISTIC = False

SEED = 1
OUT_PATH = None  # e.g. "viz/game.html"
DEVICE = "cpu"
# =========================

EXPERIMENTAL_ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train_transformer"
DEFAULT_OUT = EXPERIMENTAL_ARCH_DIR / "viz" / "orbit_wars_game.html"
DEFAULT_CHECKPOINT_DIR = TRAIN_DIR / "checkpoints" / "galaxy_a2_p44_reference_ppo_transformer_v1"

sys.path.insert(0, str(TRAIN_DIR))

from env import scores_from_obs
from features import decode_move, encode_obs
from model import build_policy
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
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        self.model_type = config.get("model", "entity_transformer")
        self.model = build_policy(
            self.model_type,
            config.get("hidden", 128),
            config.get("transformer_layers", 3),
            config.get("transformer_heads", 4),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> list[list[float]]:
        encoded = encode_obs(obs, player=self.player)
        if self.model_type == "entity_transformer_temporal":
            planets = encoded.tokens
            planet_mask = encoded.presence
        else:
            planets = encoded.planets
            planet_mask = encoded.planet_mask
        batch = {
            "planets": torch.as_tensor(planets, dtype=torch.float32, device=self.device).unsqueeze(0),
            "planet_mask": torch.as_tensor(planet_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            "globals_": torch.as_tensor(encoded.globals, dtype=torch.float32, device=self.device).unsqueeze(0),
            "action_mask": torch.as_tensor(encoded.action_mask, dtype=torch.bool, device=self.device).unsqueeze(0),
        }
        if self.model_type == "entity_transformer_temporal":
            batch["pair_turns"] = torch.as_tensor(
                encoded.pair_turns,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            batch["pair_reachable_mask"] = torch.as_tensor(
                encoded.pair_reachable_mask,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            batch["takeover_features"] = torch.as_tensor(
                encoded.takeover_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self.model(**batch)
            if self.deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        return decode_move(obs, int(action.item()))


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
    for name in ("latest.pt", "best.pt", "final.pt"):
        path = DEFAULT_CHECKPOINT_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint found in {DEFAULT_CHECKPOINT_DIR}"
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
    if path.exists() and path.suffix == ".pt":
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
    final_reward = 0.0
    total_reward = 0.0
    for _ in range(max_steps):
        if kenv.done:
            break
        actions = [hero.act(eng_obs[0]), opponent.act(eng_obs[1])]
        kenv.step(actions)
        out_step = engine.step(actions)
        final_reward = float(out_step["reward"][0])
        total_reward += final_reward
        eng_obs = out_step["observations"]
        steps += 1
        if out_step["done"]:
            break

    scores = scores_from_obs(eng_obs[0], num_players=PLAYERS)
    total = sum(scores)
    share0 = scores[0] / total if total else 0.0
    if scores[0] > scores[1]:
        verdict = "hero wins"
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
    print(f"reward:          final={final_reward:.6g} total={total_reward:.6g}")
    print(f"html:            {out}")
    link = windows_link(out)
    if link:
        print(f"windows:         {link}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hero", default=HERO)
    parser.add_argument("--opponent", default=OPPONENT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--opponent-deterministic", action="store_true")
    args = parser.parse_args()

    # --- hero ---
    hero_path = resolve_path(args.hero)

    if hero_path.exists() and hero_path.suffix == ".pt":
        hero = CheckpointAgent(
            hero_path,
            player=0,
            deterministic=args.deterministic or HERO_DETERMINISTIC,
            device=args.device,
        )
        self_checkpoint = hero_path
    elif hero_path.exists() and hero_path.suffix == ".py":
        hero = PythonAgent(hero_path)
        self_checkpoint = hero_path
    else:
        raise FileNotFoundError(f"invalid HERO: {args.hero}")

    # --- opponent ---
    opponent_ref = args.opponent
    if opponent_ref == "self":
        opponent_ref = str(self_checkpoint)

    opponent = make_opponent(
        opponent_ref,
        player=1,
        self_checkpoint=self_checkpoint,
        deterministic=args.opponent_deterministic or OPPONENT_DETERMINISTIC,
        device=args.device,
    )

    # --- run ---
    out_path = resolve_path(args.out) if args.out else DEFAULT_OUT

    play_and_render(
        hero=hero,
        opponent=opponent,
        seed=args.seed,
        out=out_path,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
