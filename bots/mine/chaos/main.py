"""chaos — aphrodite's DUCT search seeded with the osteo IL policy's moves.

Per turn:
  1. Run the osteo IL transformer once (~70ms) and take its top-k actions.
  2. Send the observation to the aphrodite binary with an extra
     `il_candidates` field; the Rust side splices them into the root
     candidate set (see `duct::inject_root_candidates`), where the search
     and the XGB value net arbitrate between apollo's plans and the IL
     policy's suggestions.

The aphrodite engine and subprocess management are reused from
bots/mine/aphrodite (same binary - the `il_candidates` field is optional and
aphrodite's own wrapper never sends it).

Time budgeting is dynamic per turn: Kaggle gives 1s/turn + a 60s overage pool
on unknown hardware, so the wrapper measures its own IL pass each turn and
sends the binary `budget_ms` = turn target minus elapsed. The target is a
conservative 700ms in dev and 1000ms in submission builds. The Rust side still
clamps the effective budget to 900ms when the remaining overage pool is low.

Failures are LOUD by design: a missing checkpoint, stale `orbit_wars_model`
schema, or IL runtime error raises and kills the bot. No silent degradation -
if chaos is running, the IL injection is running.

Env knobs:
  CHAOS_IL_K            max IL candidates injected per turn (default 5)
  CHAOS_IL_MIN_PROB     drop IL suggestions below this policy prob (default 0.02)
  CHAOS_TURN_TARGET_MS  total per-turn wall target (default 700 dev / 1000 prod)
  CHAOS_IL_CHECKPOINT   override the IL checkpoint path
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path


def _here() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    import inspect

    frame = inspect.currentframe()
    filename = frame.f_code.co_filename if frame is not None else ""
    if filename and filename != "<string>":
        return Path(filename).resolve().parent
    return Path.cwd().resolve()


def _repo_root() -> Path:
    env = os.environ.get("OSTEO_ORBIT_WARS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    starts = [Path.cwd().resolve(), _here()]
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / "run_match.py").is_file() and (parent / "bots").is_dir():
                return parent
    return Path.cwd().resolve()


HERE = _here()
# Flat Kaggle bundle: everything (wrapper copy, IL support files, .so modules,
# checkpoint, binary, weights) sits next to main.py. Dev: resolve through the
# repo layout instead.
_BUNDLE = (HERE / "aphrodite_wrapper.py").is_file()
if _BUNDLE:
    _APH_WRAPPER = HERE / "aphrodite_wrapper.py"
    _IL_SYS_PATH = HERE
    DEFAULT_CHECKPOINT = HERE / "osteo_il_latest.pt"
else:
    ROOT = _repo_root()
    _APH_WRAPPER = ROOT / "bots" / "mine" / "aphrodite" / "main.py"
    _IL_SYS_PATH = ROOT / "experimental_arch" / "train_transformer"
    DEFAULT_CHECKPOINT = (
        ROOT
        / "experimental_arch"
        / "imitation_learning"
        / "checkpoints"
        / "osteo_bc_transformer"
        / "latest.pt"
    )

# Reuse aphrodite's wrapper internals (binary location/build, weight selection,
# daemon lifecycle). Loaded by path because bot dirs are not packages. In the
# flat bundle its `_locate` finds the binary and weights next to itself.
_spec = importlib.util.spec_from_file_location("chaos_aphrodite_wrapper", _APH_WRAPPER)
_aph = importlib.util.module_from_spec(_spec)
sys.modules["chaos_aphrodite_wrapper"] = _aph
_spec.loader.exec_module(_aph)

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

# Total per-turn wall target (IL pass + search). Submission builds flip
# _USE_PROD_LIMITS to match aphrodite's production budget policy.
_DEV_TURN_TARGET_MS = 700
_SUBMISSION_TURN_TARGET_MS = 1000
_USE_PROD_LIMITS = False
# Never squeeze the search below this, no matter how slow the IL pass was.
_MIN_SEARCH_MS = 250
# Margin between (target - il_elapsed) and the budget we hand the binary,
# covering JSON encode + IPC + the binary's own dispatch overhead.
_DISPATCH_MARGIN_MS = 30


def _turn_target_ms() -> int:
    default = _SUBMISSION_TURN_TARGET_MS if _USE_PROD_LIMITS else _DEV_TURN_TARGET_MS
    return int(os.environ.get("CHAOS_TURN_TARGET_MS", default))


def _il_k() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_K", "5")))


def _il_min_prob() -> float:
    return float(os.environ.get("CHAOS_IL_MIN_PROB", "0.02"))


class _ILPolicy:
    """Loads the osteo IL transformer and yields top-k decoded actions."""

    def __init__(self) -> None:
        if str(_IL_SYS_PATH) not in sys.path:
            sys.path.insert(0, str(_IL_SYS_PATH))
        import torch  # deferred: only needed when IL is usable

        threads = os.environ.get("CHAOS_TORCH_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        from features import encode_obs  # noqa: F401  (validates import early)
        from model import build_policy
        from orbit_wars_engine import OrbitWarsEngine
        from orbit_wars_model import encode_obs as raw_encode_obs

        # Same live-schema check as osteo-il-latest: a stale orbit_wars_model
        # build silently produces garbage features, so fail (soft) instead.
        engine = OrbitWarsEngine(num_players=2)
        sample = engine.reset(seed=1)["observations"][0]
        feat = raw_encode_obs(sample, 0)
        tokens_shape = tuple(int(x) for x in feat.get("tokens_shape", ()))
        pair_shape = tuple(int(x) for x in feat.get("pair_outcome_features_shape", ()))
        if tokens_shape != (4, 44, 15) or pair_shape != (44, 44, 3, 4):
            raise RuntimeError(
                f"stale orbit_wars_model feature schema: tokens_shape={tokens_shape} "
                f"pair_outcome_features_shape={pair_shape}"
            )

        self._torch = torch
        path = Path(
            os.environ.get("CHAOS_IL_CHECKPOINT", DEFAULT_CHECKPOINT)
        ).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        self.model = build_policy(
            config.get("model", "entity_transformer_temporal"),
            hidden=int(config.get("hidden", 128)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            transformer_heads=int(config.get("transformer_heads", 4)),
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        # Warmup forward pass on the sample obs so torch's lazy init (thread
        # pools, kernel dispatch) is paid here — during game setup, where the
        # overage pool absorbs it — instead of inflating the first turns.
        sample.setdefault("player", 0)
        self.top_actions(sample, 1, 0.0)

    def top_actions(self, obs: dict, k: int, min_prob: float) -> list[list[float]]:
        torch = self._torch
        from features import decode_move, encode_obs

        encoded = encode_obs(obs, player=int(obs.get("player", 0)))
        t = lambda x, dt: torch.as_tensor(x, dtype=dt).unsqueeze(0)  # noqa: E731
        batch = {
            "planets": t(encoded.planets, torch.float32),
            "planet_mask": t(encoded.planet_mask, torch.float32),
            "globals_": t(encoded.globals, torch.float32),
            "action_mask": t(encoded.action_mask, torch.bool),
            "pair_turns": t(encoded.pair_turns, torch.float32),
            "pair_reachable_mask": t(encoded.pair_reachable_mask, torch.float32),
            "pair_outcome_features": t(encoded.pair_outcome_features, torch.float32),
            "planet_timeline_features": t(encoded.planet_timeline_features, torch.float32),
        }
        with torch.inference_mode():
            logits, _value = self.model(**batch)
            probs = torch.softmax(logits[0], dim=-1)
            # Oversample: some top entries decode to noop/dupes and get dropped.
            top = torch.topk(probs, min(k + 4, probs.numel()))
        actions: list[list[float]] = []
        seen: set[tuple] = set()
        for idx, prob in zip(top.indices.tolist(), top.values.tolist()):
            if len(actions) >= k or prob < min_prob:
                break
            move = decode_move(obs, int(idx))
            if not move:
                continue  # noop or invalid under the live mask
            src, angle, ships = move[0]
            key = (int(src), int(ships), round(float(angle) * 100))
            if key in seen:
                continue
            seen.add(key)
            actions.append([int(src), float(angle), int(ships)])
        return actions


_IL: _ILPolicy | None = None


def _il() -> _ILPolicy:
    """Construct the IL policy on first use. Failures propagate — chaos must
    never silently degrade to pure aphrodite."""
    global _IL
    if _IL is None:
        _IL = _ILPolicy()
    return _IL


def agent(obs, config=None):
    p = _aph._norm(obs)
    if config is not None:
        cfg = {}
        for k in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius", "boardSize", "cometSpeed"):
            v = config.get(k) if isinstance(config, dict) else getattr(config, k, None)
            if v is not None:
                cfg[k] = v
        if cfg:
            p["config"] = cfg

    t0 = time.perf_counter()
    # The IL net is 2p-trained; 4p games run as pure aphrodite. Any IL error
    # raises (fail loud) — no fallback.
    if _aph._infer_num_players(p) == 2:
        cands = _il().top_actions(p, _il_k(), _il_min_prob())
        if cands:
            p["il_candidates"] = cands
    else:
        cands = None
    # Dynamic split of the turn target: whatever the IL pass (or 4p skip) left
    # goes to the search. Sent per turn — the binary's env budget is unused.
    il_ms = (time.perf_counter() - t0) * 1000
    p["budget_ms"] = max(_MIN_SEARCH_MS, int(_turn_target_ms() - il_ms - _DISPATCH_MARGIN_MS))
    if os.environ.get("OW_DEBUG"):
        print(
            f"[chaos] step={p['step']} il_ms={il_ms:.0f} budget_ms={p['budget_ms']} "
            f"il_candidates={cands}",
            file=sys.stderr,
        )

    with _aph._LOCK:
        proc = _aph._ensure(p)
        try:
            proc.stdin.write((json.dumps(p, separators=(",", ":")) + "\n").encode())
            proc.stdin.flush()
            r = proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            _aph._PROC = None
            raise RuntimeError(f"aphrodite binary died at step {p['step']}") from exc
        if not r:
            raise RuntimeError(f"aphrodite binary closed stdout at step {p['step']}")
        return json.loads(r.decode())


# Eager init: pay the torch import + checkpoint load + warmup pass (~2.5s) at
# module load — Kaggle's agent setup window — instead of eating turn 1's search
# budget. Also the loudest possible failure: a stale orbit_wars_model schema or
# missing checkpoint kills the bot before it ever plays a turn.
_il()
