"""chaos: Aphrodite's DUCT search seeded with osteo IL policy moves.

Per turn:
  1. Run the osteo IL transformer and decode top legal actions.
  2. Send the observation to the aphrodite binary with an extra
     `il_candidates` field; the Rust side adds them to the root candidate set.

The aphrodite engine and subprocess management are reused from
bots/mine/aphrodite. The `il_candidates` field is optional, and
aphrodite's own wrapper never sends it.

Time budgeting is dynamic per turn: the wrapper measures its own IL pass and
sends the binary a `budget_ms` for the remaining search time.

Failures are loud: a missing checkpoint, stale `orbit_wars_model` schema, or IL
runtime error raises immediately.

Runtime IL is currently enabled for live two-player states. Three- and
four-player states pass through Aphrodite without IL injection.

Env knobs:
  CHAOS_IL_K            max IL candidates injected per turn; 0 disables IL
  CHAOS_IL_MIN_PROB     drop IL suggestions below this policy probability
  CHAOS_IL_SKIP_TURNS   skip IL injection before this step
  CHAOS_IL_BUSY_FAIL_MS fail loudly if a timed-out IL worker is still busy after
                         this many ms
  CHAOS_TORCH_THREADS   torch / OpenMP intra-op threads
  CHAOS_TURN_TARGET_MS  total per-turn wall target override
  CHAOS_IL_CHECKPOINT   override the 2p IL checkpoint path
  CHAOS_IL_CHECKPOINT_4P override the 4p IL checkpoint path
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
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
# Flat bundle: runtime files sit next to main.py. Dev resolves through the repo.
_BUNDLE = (HERE / "aphrodite_wrapper.py").is_file()
if _BUNDLE:
    _APH_WRAPPER = HERE / "aphrodite_wrapper.py"
    _IL_SYS_PATH = HERE
    DEFAULT_CHECKPOINT_2P = HERE / "osteo_il_2p_latest.pt"
    DEFAULT_CHECKPOINT_4P = HERE / "osteo_il_4p_latest.pt"
else:
    ROOT = _repo_root()
    _APH_WRAPPER = ROOT / "bots" / "mine" / "aphrodite" / "main.py"
    _IL_SYS_PATH = ROOT / "experimental_arch" / "train_transformer"
    _CHECKPOINT_DIR = ROOT / "experimental_arch" / "imitation_learning" / "checkpoints"
    DEFAULT_CHECKPOINT_2P = _CHECKPOINT_DIR / "osteo_bc_transformer" / "latest.pt"
    DEFAULT_CHECKPOINT_4P = _CHECKPOINT_DIR / "osteo_il_4p_latest.pt"

# Reuse aphrodite's wrapper internals (binary location/build, weight selection,
# daemon lifecycle). Loaded by path because bot dirs are not packages. In the
# flat bundle its `_locate` finds the binary and weights next to itself.
_spec = importlib.util.spec_from_file_location("chaos_aphrodite_wrapper", _APH_WRAPPER)
_aph = importlib.util.module_from_spec(_spec)
sys.modules["chaos_aphrodite_wrapper"] = _aph
_spec.loader.exec_module(_aph)

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

# Set before importing torch. Environment overrides still win.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_TORCH_THREADS = os.environ.get("CHAOS_TORCH_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _TORCH_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _TORCH_THREADS)

# Total per-turn wall target for IL plus search.
_DEV_TURN_TARGET_MS = 700
_SUBMISSION_TURN_TARGET_MS = 1000
_USE_PROD_LIMITS = False
# Never squeeze the search below this, no matter how slow the IL pass was.
_MIN_SEARCH_MS = 250
# Margin between (target - il_elapsed) and the budget we hand the binary,
# covering JSON encode + IPC + the binary's own dispatch overhead.
_DISPATCH_MARGIN_MS = 30
# Warmup architecture fallback. `_warm_arch()` prefers checkpoint config.
_WARM_MODEL = "entity_transformer_ngpt_action_features"
_WARM_HIDDEN = 128
_WARM_LAYERS = 3
_WARM_HEADS = 4


def _turn_target_ms() -> int:
    default = _SUBMISSION_TURN_TARGET_MS if _USE_PROD_LIMITS else _DEV_TURN_TARGET_MS
    return int(os.environ.get("CHAOS_TURN_TARGET_MS", default))


def _il_k() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_K", "5")))


def _il_min_prob() -> float:
    return float(os.environ.get("CHAOS_IL_MIN_PROB", "0.02"))


def _il_skip_turns() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_SKIP_TURNS", "1")))


def _il_busy_fail_ms() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_BUSY_FAIL_MS", "5000")))


def _checkpoint_path_2p() -> Path:
    override = os.environ.get("CHAOS_IL_CHECKPOINT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CHECKPOINT_2P


def _checkpoint_path_4p() -> Path:
    override = os.environ.get("CHAOS_IL_CHECKPOINT_4P")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CHECKPOINT_4P


def _load_checkpoint(torch, path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        if _BUNDLE:
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


class _ILPolicy:
    """Loads the osteo IL transformer and yields top-k decoded actions."""

    def __init__(self, path: Path) -> None:
        if str(_IL_SYS_PATH) not in sys.path:
            sys.path.insert(0, str(_IL_SYS_PATH))
        import torch
        from model import build_policy

        self._torch = torch
        checkpoint = _load_checkpoint(torch, path)
        config = checkpoint.get("config", {})
        self.checkpoint_path = path
        self.dataset_name = str(config.get("dataset_name", "unknown"))
        self.model = build_policy(
            config.get("model", "entity_transformer_temporal"),
            hidden=int(config.get("hidden", 128)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            transformer_heads=int(config.get("transformer_heads", 4)),
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def top_actions(self, obs: dict, k: int, min_prob: float) -> list[dict]:
        if k <= 0:
            return []
        torch = self._torch
        from features import decode_index_from_feat, encode_obs_and_feat
        from model import tensorize

        # Reuse the raw feature dict to decode every candidate index below.
        encoded, feat = encode_obs_and_feat(obs, player=int(obs.get("player", 0)))
        batch = tensorize(encoded)
        with torch.inference_mode():
            logits, _value = self.model(**batch)
            probs = torch.softmax(logits[0], dim=-1)
            # Oversample: some top entries decode to noop/dupes and get dropped.
            top = torch.topk(probs, min(k + 4, probs.numel()))
        actions: list[dict] = []
        seen: set[tuple] = set()
        for idx, prob in zip(top.indices.tolist(), top.values.tolist()):
            if len(actions) >= k or prob < min_prob:
                break
            move = decode_index_from_feat(feat, int(idx))
            if not move:
                continue  # noop or invalid under the live mask
            src, angle, ships = move[0]
            key = (int(src), int(ships), round(float(angle) * 100))
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "action": [int(src), float(angle), int(ships)],
                    "prob": float(prob),
                    "index": int(idx),
                }
            )
        return actions


_IL_2P: _ILPolicy | None = None
_IL_4P: _ILPolicy | None = None


def _il_2p() -> _ILPolicy:
    """Construct the 2p IL policy on first use."""
    global _IL_2P
    if _IL_2P is None:
        _IL_2P = _ILPolicy(_checkpoint_path_2p())
    return _IL_2P


def _il_4p() -> _ILPolicy:
    """Construct the 4p IL policy on first use (3- and 4-player states)."""
    global _IL_4P
    if _IL_4P is None:
        _IL_4P = _ILPolicy(_checkpoint_path_4p())
    return _IL_4P


def _il_for(num_players: int) -> _ILPolicy:
    return _il_2p() if num_players == 2 else _il_4p()


def _warm_arch() -> tuple[str, int, int, int]:
    """Return the checkpoint architecture used to size the warmup model."""
    import torch

    try:
        # weights_only=True + mmap reads just the config (no arbitrary pickle,
        # no weight materialization); the lazy _ILPolicy load pays the real load.
        ckpt = torch.load(
            _checkpoint_path_2p(), map_location="cpu", weights_only=True, mmap=True
        )
        cfg = ckpt.get("config", {}) or {}
        return (
            str(cfg.get("model", _WARM_MODEL)),
            int(cfg.get("hidden", _WARM_HIDDEN)),
            int(cfg.get("transformer_layers", _WARM_LAYERS)),
            int(cfg.get("transformer_heads", _WARM_HEADS)),
        )
    except Exception as exc:
        if os.environ.get("OW_DEBUG"):
            print(f"[chaos] warm-arch fell back to defaults: {exc}", file=sys.stderr)
        return (_WARM_MODEL, _WARM_HIDDEN, _WARM_LAYERS, _WARM_HEADS)


def _warm_torch() -> None:
    """Warm torch and validate the feature schema without loading checkpoints."""
    if str(_IL_SYS_PATH) not in sys.path:
        sys.path.insert(0, str(_IL_SYS_PATH))
    import torch

    torch.set_num_threads(int(_TORCH_THREADS))
    from features import encode_obs
    from model import build_policy, tensorize
    from orbit_wars_engine import OrbitWarsEngine
    from orbit_wars_model import encode_obs as raw_encode_obs

    engine = OrbitWarsEngine(num_players=2)
    sample = engine.reset(seed=1)["observations"][0]
    sample.setdefault("player", 0)
    feat = raw_encode_obs(sample, 0)
    tokens_shape = tuple(int(x) for x in feat.get("tokens_shape", ()))
    pair_shape = tuple(int(x) for x in feat.get("pair_outcome_features_shape", ()))
    if tokens_shape != (4, 44, 15) or pair_shape != (44, 44, 3, 4):
        raise RuntimeError(
            f"stale orbit_wars_model feature schema: tokens_shape={tokens_shape} "
            f"pair_outcome_features_shape={pair_shape}"
        )
    # Throwaway forward to initialize torch kernels/thread pools.
    model_type, hidden, layers, heads = _warm_arch()
    warm = build_policy(model_type, hidden=hidden, transformer_layers=layers, transformer_heads=heads)
    warm.eval()
    with torch.inference_mode():
        warm(**tensorize(encode_obs(sample, player=0)))


# IL forwards run in one worker. Timed-out results are discarded if they finish
# after their turn.
_IL_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chaos-il")
_IL_FUTURE = None  # in-flight task from a turn whose IL deadline expired
_IL_FUTURE_STARTED_AT: float | None = None
_IL_FUTURE_STEP = None


def _il_candidates(obs: dict, num_players: int) -> list[dict]:
    k = _il_k()
    if k <= 0:
        return []
    return _il_for(num_players).top_actions(obs, k, _il_min_prob())


def _il_pass(obs: dict, t0: float, num_players: int) -> tuple[list[dict] | None, str]:
    """Run the IL forward under a wall-clock deadline."""
    global _IL_FUTURE, _IL_FUTURE_STARTED_AT, _IL_FUTURE_STEP
    tag = "2p" if num_players == 2 else "4p"
    if _IL_FUTURE is not None:
        if _IL_FUTURE.done():
            stale = _IL_FUTURE
            _IL_FUTURE = None
            _IL_FUTURE_STARTED_AT = None
            _IL_FUTURE_STEP = None
            # Drain the stale previous-observation result. Success is discarded,
            # but any background exception must still be loud.
            stale.result()
        else:
            started_at = _IL_FUTURE_STARTED_AT or time.perf_counter()
            busy_ms = (time.perf_counter() - started_at) * 1000
            limit_ms = _il_busy_fail_ms()
            if busy_ms >= limit_ms:
                prior_step = _IL_FUTURE_STEP
                raise RuntimeError(
                    f"IL worker stuck after timeout: prior_step={prior_step} "
                    f"current_step={obs.get('step')} busy_ms={busy_ms:.0f} "
                    f"limit_ms={limit_ms}"
                )
            return None, f"{tag}:busy"

    deadline_s = (
        _turn_target_ms() - _MIN_SEARCH_MS - _DISPATCH_MARGIN_MS
    ) / 1000.0 - (time.perf_counter() - t0)
    if deadline_s <= 0.0:
        if os.environ.get("OW_DEBUG"):
            print(
                f"[chaos] IL skipped at step={obs.get('step')} {tag} "
                f"deadline_ms={deadline_s * 1000:.0f}; search runs without "
                f"IL candidates this turn",
                file=sys.stderr,
            )
        return None, f"{tag}:deadline0"

    submitted_at = time.perf_counter()
    fut = _IL_EXEC.submit(_il_candidates, obs, num_players)
    try:
        cands = fut.result(timeout=max(0.0, deadline_s))
    except _FutureTimeout:
        _IL_FUTURE = fut
        _IL_FUTURE_STARTED_AT = submitted_at
        _IL_FUTURE_STEP = obs.get("step")
        print(
            f"[chaos] IL deadline cut at step={obs.get('step')} {tag} "
            f"deadline_ms={max(0.0, deadline_s) * 1000:.0f}; search runs without "
            f"IL candidates this turn",
            file=sys.stderr,
        )
        return None, f"{tag}:deferred"
    return cands, f"{tag}:{_il_for(num_players).dataset_name}"


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
    num_players = _aph._infer_num_players(p)
    tag = "2p" if num_players == 2 else "4p"
    # Only live two-player states use IL.
    if num_players == 2 and p["step"] >= _il_skip_turns():
        if _il_k() <= 0:
            cands = None
            il_desc = f"{tag}:k0"
        else:
            cands, il_desc = _il_pass(p, t0, num_players)
            if cands:
                p["il_candidates"] = [c["action"] for c in cands]
                p["il_candidate_probs"] = [c["prob"] for c in cands]
                p["il_candidate_indices"] = [c["index"] for c in cands]
    elif num_players >= 3:
        cands = None
        il_desc = f"{tag}:off"
    else:
        cands = None
        il_desc = f"{tag}:skip" if num_players == 2 else None
    il_ms = (time.perf_counter() - t0) * 1000
    p["budget_ms"] = max(_MIN_SEARCH_MS, int(_turn_target_ms() - il_ms - _DISPATCH_MARGIN_MS))
    if os.environ.get("OW_DEBUG"):
        print(
            f"[chaos] step={p['step']} il_ms={il_ms:.0f} budget_ms={p['budget_ms']} "
            f"il={il_desc} il_candidates={cands}",
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


if _il_k() > 0:
    _warm_torch()
elif os.environ.get("OW_DEBUG"):
    print("[chaos] IL disabled by CHAOS_IL_K=0; skipping torch warmup", file=sys.stderr)
