from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import torch
from torch.distributions import Categorical


def _here() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    frame = inspect.currentframe()
    filename = frame.f_code.co_filename if frame is not None else ""
    if filename and filename != "<string>":
        return Path(filename).resolve().parent
    return Path.cwd().resolve()


HERE = _here()
IL_SUPPORT = HERE / "il_support"
CHECKPOINT = Path(
    os.environ.get(
        "TRIPLEPOINT1_IL_CHECKPOINT",
        os.environ.get("HYBRID1_IL_CHECKPOINT", HERE / "osteo_il_latest.pt"),
    )
).expanduser()

_APHRODITE_PROC = None
_APHRODITE_LOCK = threading.Lock()
_IL_ERROR: Exception | None = None

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(IL_SUPPORT) not in sys.path:
    sys.path.insert(0, str(IL_SUPPORT))

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")
torch.set_num_threads(int(os.environ.get("TRIPLEPOINT1_TORCH_THREADS", os.environ.get("HYBRID1_TORCH_THREADS", "1"))))

from features import decode_move, encode_obs  # noqa: E402
from model import build_policy  # noqa: E402


def _norm(obs) -> dict:
    get = obs.get if isinstance(obs, dict) else (lambda k, d=None: getattr(obs, k, d))
    return {
        "player": int(get("player", 0) or 0),
        "step": int(get("step", 0) or 0),
        "planets": list(get("planets", []) or []),
        "fleets": list(get("fleets", []) or []),
        "angular_velocity": float(get("angular_velocity", 0.0) or 0.0),
        "initial_planets": list(get("initial_planets", []) or []),
        "comets": list(get("comets", []) or []),
        "comet_planet_ids": list(get("comet_planet_ids", []) or []),
    }


def _alive_players(payload: dict) -> set[int]:
    alive: set[int] = set()
    for planet in payload.get("planets", []) or []:
        try:
            owner = int(planet[1])
        except Exception:
            continue
        if owner >= 0:
            alive.add(owner)
    for fleet in payload.get("fleets", []) or []:
        try:
            owner = int(fleet[1])
        except Exception:
            continue
        if owner >= 0:
            alive.add(owner)
    return alive


def _ensure_executable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def _pump_stderr(pipe) -> None:
    try:
        for line in iter(pipe.readline, b""):
            if os.environ.get("TRIPLEPOINT1_DEBUG") or os.environ.get("HYBRID1_DEBUG"):
                sys.stderr.write(line.decode("utf-8", "replace"))
                sys.stderr.flush()
    except Exception:
        pass


def _aphrodite_env() -> dict:
    env = dict(os.environ)
    env.setdefault(
        "APHRODITE_BUDGET_MS",
        env.get("TRIPLEPOINT1_APHRODITE_BUDGET_MS", env.get("HYBRID1_APHRODITE_BUDGET_MS", "500")),
    )
    env.setdefault("APHRODITE_VALUE_NET_PATH", str(HERE / "xgb_4p.json"))
    env.setdefault("APHRODITE_VALUE_NET_PATH_2P", str(HERE / "xgb_2p.json"))
    if not Path(env["APHRODITE_VALUE_NET_PATH"]).is_file():
        env["APHRODITE_VALUE_NET_PATH"] = str(HERE / "xgb_2p_old_top10.json")
    if not Path(env["APHRODITE_VALUE_NET_PATH_2P"]).is_file():
        env["APHRODITE_VALUE_NET_PATH_2P"] = str(HERE / "xgb_2p_old_top10.json")
    return env


def _start_aphrodite():
    binary = Path(
        os.environ.get("TRIPLEPOINT1_APHRODITE_BIN", os.environ.get("HYBRID1_APHRODITE_BIN", HERE / "aphrodite"))
    ).expanduser()
    if not binary.is_file():
        raise RuntimeError(f"aphrodite binary missing: {binary}")
    _ensure_executable(binary)
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(HERE),
        env=_aphrodite_env(),
        bufsize=0,
    )
    threading.Thread(target=_pump_stderr, args=(proc.stderr,), daemon=True).start()
    return proc


def _ensure_aphrodite():
    global _APHRODITE_PROC
    if _APHRODITE_PROC is not None and _APHRODITE_PROC.poll() is None:
        return _APHRODITE_PROC
    _APHRODITE_PROC = _start_aphrodite()
    return _APHRODITE_PROC


def _ask_aphrodite(payload: dict) -> object | None:
    global _APHRODITE_PROC
    with _APHRODITE_LOCK:
        proc = _ensure_aphrodite()
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
            proc.stdin.flush()
            raw = proc.stdout.readline()
        except (BrokenPipeError, OSError):
            _APHRODITE_PROC = None
            return None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _aphrodite_move(payload: dict) -> list:
    result = _ask_aphrodite(payload)
    return result if isinstance(result, list) else []


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    last_error = None
    for _ in range(5):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"failed to load IL checkpoint {path}: {last_error}")


class ILAgent:
    def __init__(self) -> None:
        self.device = torch.device(os.environ.get("TRIPLEPOINT1_IL_DEVICE", os.environ.get("HYBRID1_IL_DEVICE", "cpu")))
        deterministic = os.environ.get(
            "TRIPLEPOINT1_IL_DETERMINISTIC",
            os.environ.get("HYBRID1_IL_DETERMINISTIC", os.environ.get("OSTEO_IL_DETERMINISTIC", "")),
        )
        self.deterministic = deterministic.lower() in {"1", "true", "yes", "on"}
        ckpt = _load_checkpoint(CHECKPOINT, self.device)
        config = ckpt.get("config", {})
        self.model = build_policy(
            config.get("model", "entity_transformer_ngpt_action_features"),
            hidden=int(config.get("hidden", 128)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            transformer_heads=int(config.get("transformer_heads", 4)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def act(self, obs: dict) -> list:
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


def _load_il() -> ILAgent | None:
    global _IL_ERROR
    try:
        return ILAgent()
    except Exception as exc:
        _IL_ERROR = exc
        return None


with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    _IL = _load_il()

try:
    _ensure_aphrodite()
except Exception:
    pass


def _use_aphrodite(payload: dict) -> bool:
    alive = _alive_players(payload)
    if len(alive) > 2 or _IL is None:
        return True
    if len(alive) == 2:
        my_ships, opp_ships = _ship_totals_2p(payload)
        return my_ships > 1.5 * opp_ships
    return False


def _ship_totals_2p(payload: dict) -> tuple[float, float]:
    me = int(payload.get("player", 0) or 0)
    opponents = sorted(p for p in _alive_players(payload) if p != me)
    opp = opponents[0] if opponents else 1 - me
    totals: dict[int, float] = {}
    for planet in payload.get("planets", []) or []:
        try:
            owner = int(planet[1])
            ships = float(planet[5])
        except Exception:
            continue
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + ships
    for fleet in payload.get("fleets", []) or []:
        try:
            owner = int(fleet[1])
            ships = float(fleet[6])
        except Exception:
            continue
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + ships
    return totals.get(me, 0.0), totals.get(opp, 0.0)


def agent(obs, config=None):
    payload = _norm(obs)
    if config is not None:
        cfg = {}
        for key in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius", "boardSize", "cometSpeed"):
            value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
            if value is not None:
                cfg[key] = value
        if cfg:
            payload["config"] = cfg

    choice = "aphrodite" if _use_aphrodite(payload) else "il"
    if os.environ.get("TRIPLEPOINT1_DEBUG") or os.environ.get("HYBRID1_DEBUG"):
        my_ships, opp_ships = _ship_totals_2p(payload)
        sys.stderr.write(
            f"[triplepoint1] step={payload['step']} alive={len(_alive_players(payload))} "
            f"ships={my_ships:.0f}/{opp_ships:.0f} choice={choice}\n"
        )
        sys.stderr.flush()
    if choice == "il" and _IL is not None:
        try:
            return _IL.act(payload)
        except Exception as exc:
            if os.environ.get("TRIPLEPOINT1_DEBUG") or os.environ.get("HYBRID1_DEBUG"):
                sys.stderr.write(f"[triplepoint1] IL error, falling back to Aphrodite: {exc!r}\n")
                sys.stderr.flush()
            return _aphrodite_move(payload)
    return _aphrodite_move(payload)
