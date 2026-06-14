from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.distributions import Categorical


IL_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_ARCH_DIR = IL_DIR.parent
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train_transformer"
BOTS_DIR = REPO_ROOT / "bots"
DEFAULT_CHECKPOINT = IL_DIR / "checkpoints" / "osteo_bc_transformer" / "latest.pt"
LATEST_CHECKPOINT_ALIASES = {"latest", "latest.pt", "latest-training", "training-latest"}

PLAYERS = 2
MAX_STEPS = 500

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from features import decode_move, encode_obs  # noqa: E402
from model import build_policy  # noqa: E402
from orbit_wars_model import encode_obs as raw_encode_obs  # noqa: E402
from orbit_wars_engine import OrbitWarsEngine  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make  # noqa: E402


class Agent(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: dict) -> list[list[float]]:
        ...


@dataclass
class GameResult:
    seed: int
    steps: int
    scores: list[int]
    winner: int | None
    reward: float


class CheckpointAgent:
    def __init__(self, checkpoint: Path, *, deterministic: bool, device: str):
        self.checkpoint = checkpoint
        self.name = checkpoint.stem
        self.deterministic = deterministic
        self.device = torch.device(device)

        ckpt = load_checkpoint_with_retry(checkpoint, map_location=self.device)
        config = ckpt.get("config", {})
        self.model_type = config.get("model", "entity_transformer_temporal")
        self.model = build_policy(
            self.model_type,
            hidden=int(config.get("hidden", 128)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            transformer_heads=int(config.get("transformer_heads", 4)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch = int(ckpt.get("epoch", 0))

    def reset(self) -> None:
        return None

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


def validate_live_feature_schema() -> None:
    engine = OrbitWarsEngine(num_players=PLAYERS)
    obs = engine.reset(seed=1)["observations"][0]
    feat = raw_encode_obs(obs, 0)
    tokens_shape = tuple(int(x) for x in feat.get("tokens_shape", ()))
    pair_shape = tuple(int(x) for x in feat.get("pair_outcome_features_shape", ()))
    if tokens_shape != (4, 44, 15) or pair_shape != (44, 44, 3, 4):
        raise RuntimeError(
            "live orbit_wars_model feature schema is stale; expected tokens_shape=(4, 44, 15) "
            "and pair_outcome_features_shape=(44, 44, 3, 4), got "
            f"tokens_shape={tokens_shape} pair_outcome_features_shape={pair_shape}. "
            "Rebuild/reinstall experimental_arch/env_model before playing current IL checkpoints."
        )


class PythonAgent:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self._module_name = f"il_play_agent__{path.stem}_{abs(hash(path))}"
        self._agent = None
        self.reset()

    def reset(self) -> None:
        spec = importlib.util.spec_from_file_location(self._module_name, self.path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Python agent: {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self._module_name] = module
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
        if not hasattr(module, "agent"):
            raise RuntimeError(f"Python agent has no agent(obs) function: {self.path}")
        self._agent = module.agent

    def act(self, obs: dict) -> list[list[float]]:
        assert self._agent is not None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self._agent(obs)


def resolve_ref(ref: str | Path) -> Path:
    path = Path(ref).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), IL_DIR, EXPERIMENTAL_ARCH_DIR, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return IL_DIR / path


def resolve_checkpoint_ref(ref: str | Path) -> Path:
    if str(ref) in LATEST_CHECKPOINT_ALIASES:
        return DEFAULT_CHECKPOINT
    return resolve_ref(ref)


def load_checkpoint_with_retry(path: Path, *, map_location: torch.device | str, attempts: int = 5) -> dict:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last_exc is not None
    raise RuntimeError(f"failed to load checkpoint after {attempts} attempts: {path}") from last_exc


def bot_main(name: str) -> Path:
    direct = BOTS_DIR / name / "main.py"
    if direct.is_file():
        return direct
    matches = sorted(BOTS_DIR.glob(f"*/{name}/main.py"))
    if matches:
        return matches[0]
    available = sorted(str(path.parent.relative_to(BOTS_DIR)) for path in BOTS_DIR.glob("*/*/main.py"))
    raise FileNotFoundError(f"no bot named {name!r} under {BOTS_DIR}; examples: {', '.join(available[:12])}")


def make_agent(ref: str, *, checkpoint: Path, deterministic: bool, device: str) -> Agent:
    if ref == "self":
        return CheckpointAgent(checkpoint, deterministic=deterministic, device=device)
    path = resolve_ref(ref)
    if path.exists() and path.suffix == ".pt":
        return CheckpointAgent(path, deterministic=deterministic, device=device)
    if path.exists() and path.suffix == ".py":
        return PythonAgent(path)
    return PythonAgent(bot_main(ref))


def scores_from_state(state: dict, num_players: int = PLAYERS) -> list[int]:
    scores = [0] * num_players
    for planet in state.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(planet[5])
    for fleet in state.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(fleet[6])
    return scores


def play_game(
    *,
    hero: Agent,
    opponent: Agent,
    seed: int,
    max_steps: int,
    render_out: Path | None,
    replay_index: int,
    replay_total: int,
) -> GameResult:
    hero.reset()
    opponent.reset()

    engine = OrbitWarsEngine(num_players=PLAYERS)
    eng_obs = engine.reset(seed=seed)["observations"]
    kenv = None
    if render_out is not None:
        kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
        kenv.reset(PLAYERS)

    reward = 0.0
    steps = 0
    for _ in range(max_steps):
        if kenv is not None and kenv.done:
            break
        actions = [hero.act(eng_obs[0]), opponent.act(eng_obs[1])]
        if kenv is not None:
            kenv.step(actions)
        result = engine.step(actions)
        reward = float(result["reward"][0])
        eng_obs = result["observations"]
        steps += 1
        if result["done"]:
            break

    state = engine.get_state()
    scores = scores_from_state(state)
    if scores[0] > scores[1]:
        winner = 0
    elif scores[1] > scores[0]:
        winner = 1
    else:
        winner = None

    if render_out is not None:
        assert kenv is not None
        render_out.parent.mkdir(parents=True, exist_ok=True)
        render_out.write_text(
            add_replay_links(kenv.render(mode="html"), index=replay_index, total=replay_total),
            encoding="utf-8",
        )

    return GameResult(seed=seed, steps=steps, scores=scores, winner=winner, reward=reward)


def add_replay_links(html: str, *, index: int, total: int) -> str:
    links = []
    if index > 1:
        links.append(f'<a href="{index - 1}.html">Previous</a>')
    links.append(f"<span>Game {index} / {total}</span>")
    if index < total:
        links.append(f'<a href="{index + 1}.html">Next</a>')
    nav = (
        '<div style="position:fixed;top:12px;right:12px;z-index:999999;'
        'display:flex;gap:10px;align-items:center;padding:8px 10px;'
        'background:rgba(0,0,0,0.72);color:#fff;border-radius:6px;'
        'font:14px system-ui,-apple-system,Segoe UI,sans-serif">'
        + "".join(
            item
            if item.startswith("<span")
            else item.replace("<a ", '<a style="color:#8fd3ff;text-decoration:none" ')
            for item in links
        )
        + "</div>"
    )
    marker = "</body>"
    if marker in html:
        return html.replace(marker, nav + marker, 1)
    return html + nav


def windows_link(path: Path) -> str | None:
    if not shutil.which("wslpath"):
        return None
    try:
        win = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if win.startswith("\\\\"):
        return "file://" + win[2:].replace("\\", "/")
    return "file:///" + win.replace("\\", "/")


def detect_remote_host() -> str:
    if os.environ.get("IL_SCP_HOST"):
        return os.environ["IL_SCP_HOST"]
    for url in (
        "http://169.254.169.254/latest/meta-data/public-ipv4",
        "https://checkip.amazonaws.com",
    ):
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                host = response.read().decode("utf-8", errors="replace").strip()
            if host:
                return host
        except Exception:
            pass
    try:
        host = subprocess.check_output(["hostname", "-f"], text=True, timeout=1.0).strip()
        if host:
            return host
    except Exception:
        pass
    return "YOUR_SERVER_HOST"


def print_wsl_scp_command(remote_paths: list[Path]) -> None:
    if not remote_paths:
        return
    remote = f"{os.environ.get('IL_SCP_USER', os.environ.get('USER', 'ubuntu'))}@{detect_remote_host()}"
    dest = remote_paths[0].parent.name or "replays"
    remote_args = " ".join(f"{remote}:{shlex.quote(str(path))}" for path in remote_paths)
    print("copy_from_pc_wsl:")
    print(
        f"  mkdir -p \"$PWD/{dest}\" && scp {remote_args} \"$PWD/{dest}/\" && "
        "python3 - <<'PY'\n"
        "import os\n"
        "from urllib.parse import quote\n"
        "from pathlib import Path\n"
        f"p = Path({str(Path(dest) / remote_paths[0].name)!r}).resolve()\n"
        "distro = os.environ.get('WSL_DISTRO_NAME') or 'Ubuntu-22.04'\n"
        "path_part = quote(p.as_posix(), safe='/:')\n"
        "distro_part = quote(distro, safe='')\n"
        "print('saved:', p)\n"
        "print('open:', f'file://wsl.localhost/{distro_part}{path_part}')\n"
        "PY"
    )


def replay_path(out: Path, game_index: int, games: int) -> Path:
    if out.suffix.lower() == ".html":
        if games != 1:
            raise ValueError("--out may be an .html file only when running one game")
        return out
    return out / f"{game_index}.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an osteo imitation-learning checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="IL .pt checkpoint, or 'latest' for the latest training checkpoint; defaults to latest",
    )
    parser.add_argument("--opponent", default="hellburner", help="'self', a bot name like hellburner, a .py agent, or a .pt checkpoint")
    parser.add_argument("-n", "--num-games", "--games", dest="games", type=int, default=1, help="number of games to run serially")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.set_defaults(deterministic=False)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="argmax for hero instead of sampling",
    )
    parser.add_argument(
        "--sample",
        dest="deterministic",
        action="store_false",
        help="sample hero actions from the policy; this is the default",
    )
    parser.add_argument("--opponent-deterministic", action="store_true", help="argmax for checkpoint opponents")
    parser.set_defaults(render=True)
    parser.add_argument("--render", dest="render", action="store_true", help="write numbered HTML replays for all games")
    parser.add_argument("--no-render", dest="render", action="store_false", help="skip HTML replay output")
    parser.add_argument("--out", type=Path, default=None, help="replay output directory; defaults to replays/ with 1.html, 2.html, ...")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--num-games/-n must be at least 1")
    validate_live_feature_schema()
    checkpoint = resolve_checkpoint_ref(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    out_path = args.out if args.out is not None else IL_DIR / "replays"

    hero = CheckpointAgent(checkpoint, deterministic=args.deterministic, device=args.device)
    opponent = make_agent(
        args.opponent,
        checkpoint=checkpoint,
        deterministic=args.opponent_deterministic,
        device=args.device,
    )

    wins = ties = losses = 0
    total_steps = 0
    score_diffs: list[int] = []
    rendered_paths: list[Path] = []
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint_step: {hero.global_step} epoch: {hero.epoch}")
    print(f"opponent: {args.opponent} ({opponent.name})")
    print(f"device: {args.device}")
    print(f"hero_policy: {'argmax' if args.deterministic else 'sample'}")
    for i in range(args.games):
        render_out = replay_path(out_path, i + 1, args.games) if args.render else None
        result = play_game(
            hero=hero,
            opponent=opponent,
            seed=args.seed + i,
            max_steps=args.max_steps,
            render_out=render_out,
            replay_index=i + 1,
            replay_total=args.games,
        )
        diff = result.scores[0] - result.scores[1]
        score_diffs.append(diff)
        total_steps += result.steps
        if result.winner == 0:
            wins += 1
        elif result.winner == 1:
            losses += 1
        else:
            ties += 1
        print(
            f"game {i + 1:03d} seed={result.seed} steps={result.steps} "
            f"score={result.scores[0]}-{result.scores[1]} diff={diff:+d} "
            f"result={'win' if result.winner == 0 else 'loss' if result.winner == 1 else 'tie'}"
        )
        if render_out is not None:
            rendered_paths.append(render_out.resolve())
            print(f"html: {render_out}")
            link = windows_link(render_out)
            if link:
                print(f"windows: {link}")

    games = max(1, args.games)
    mean_diff = sum(score_diffs) / games
    print(
        f"summary: games={args.games} W-L-T={wins}-{losses}-{ties} "
        f"score={(wins + 0.5 * ties) / games:.3f} mean_diff={mean_diff:.2f} "
        f"mean_steps={total_steps / games:.1f}"
    )
    if rendered_paths:
        print_wsl_scp_command(rendered_paths)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    raise SystemExit(main())
