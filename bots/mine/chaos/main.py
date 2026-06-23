"""chaos: Aphrodite's DUCT search seeded with osteo IL policy moves.

Per turn:
  1. Run the osteo IL transformer and decode top legal actions for us and,
     in live 2p states, for the enemy.
  2. Send the observation to the aphrodite binary with an extra
     `il_candidates` field; the Rust side adds them to our root candidate set.
     Enemy suggestions are sent as `opp_il_candidates` and become opponent root
     candidates.

The aphrodite engine and subprocess management are reused from
bots/mine/aphrodite. The `il_candidates` field is optional, and
aphrodite's own wrapper never sends it.

Time budgeting is dynamic per turn: the wrapper measures its own IL pass,
tracks prior Python/Rust overhead, and sends the binary base/hard deadlines for
the remaining search time.

Failures are loud: a missing checkpoint, stale `orbit_wars_model` schema, or IL
runtime error raises immediately.

Runtime IL is currently enabled for live two-player states. Three- and
four-player states pass through Aphrodite without IL injection.

Env knobs:
  CHAOS_IL_K            max IL candidates injected per turn; 0 disables IL
  CHAOS_IL_MIN_PROB     drop IL suggestions below this policy probability
  CHAOS_IL_SKIP_TURNS   skip IL injection before this step
  CHAOS_IL_LEAD_SKIP_STEP
                         skip IL at/after this step when far ahead
  CHAOS_IL_LEAD_SHIP_RATIO
                         own ships multiplier for the far-ahead IL skip
  CHAOS_IL_LEAD_PROD_RATIO
                         own production multiplier for the far-ahead IL skip
  CHAOS_IL_BUSY_FAIL_MS fail loudly if a timed-out IL worker is still busy after
                         this many ms
  CHAOS_TORCH_THREADS   torch / OpenMP intra-op threads
  CHAOS_TURN_TARGET_MS  total per-turn wall target override
  CHAOS_OVERAGE_RESERVE_MS
                         minimum overage pool reserve before spending extra
  CHAOS_PANIC_OVERAGE_MS
                         skip IL/search when overage pool falls below this
  CHAOS_OVERAGE_PER_TURN_CAP_MS
                         max overage ms granted to Rust on one turn
  CHAOS_PY_MARGIN_MS    minimum Python/IPC margin before Rust
  CHAOS_RUST_POST_MARGIN_MS
                         minimum Rust post-search margin for redirect/output
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
# The IL forward and the Rust DUCT search are sequential phases of a turn, but
# they share one CPU allocation (~1.6 cores on Kaggle). By default Intel OpenMP
# keeps the torch worker threads spinning for KMP_BLOCKTIME=200ms after the IL
# pass returns, which burns cores during the immediately following search. Make
# idle threads sleep at once so the search gets the full allocation. Must be set
# before the OpenMP runtime initializes (i.e. before torch is imported).
os.environ.setdefault("KMP_BLOCKTIME", "0")  # Intel OpenMP (libiomp): sleep immediately
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # libgomp fallback

# Total per-turn wall target for IL plus search.
_DEV_TURN_TARGET_MS = 700
_SUBMISSION_TURN_TARGET_MS = 1000
_USE_PROD_LIMITS = False
if os.environ.get("CHAOS_USE_PROD_LIMITS", "").strip().lower() not in ("", "0", "false"):
    _USE_PROD_LIMITS = True
    _aph._USE_PROD_LIMITS = True
# Prefer not to squeeze the search below this when the hard deadline has room.
_MIN_SEARCH_MS = 250
# Minimum margin between (target - il_elapsed) and the budget we hand the
# binary. Runtime telemetry can raise this when Python/IPC overhead is higher.
_DISPATCH_MARGIN_MS = 30
_DEFAULT_OVERAGE_RESERVE_MS = 2000
_DEFAULT_PANIC_OVERAGE_MS = 500
_OVERAGE_PER_TURN_CAP_MS = 800
_DEFAULT_RUST_POST_MARGIN_MS = 80
# Warmup architecture fallback. `_warm_arch()` prefers checkpoint config.
_WARM_MODEL = "entity_transformer_ngpt_action_features"
_WARM_HIDDEN = 128
_WARM_LAYERS = 3
_WARM_HEADS = 4
_MAX_OPPONENT_IL_CANDIDATES = 3

_py_overhead_margin_ms = float(os.environ.get("CHAOS_PY_MARGIN_MS", _DISPATCH_MARGIN_MS))
_rust_post_margin_ms = float(
    os.environ.get("CHAOS_RUST_POST_MARGIN_MS", _DEFAULT_RUST_POST_MARGIN_MS)
)
_overage_burn_margin_ms = 0.0
_last_remaining_overage_s: float | None = None
_last_overage_step: int | None = None


def _turn_target_ms() -> int:
    default = _SUBMISSION_TURN_TARGET_MS if _USE_PROD_LIMITS else _DEV_TURN_TARGET_MS
    return int(os.environ.get("CHAOS_TURN_TARGET_MS", default))


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(lo, default)


def _env_float(name: str, default: float, lo: float = 0.0) -> float:
    try:
        return max(lo, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(lo, default)


def _overage_reserve_ms() -> int:
    return _env_int("CHAOS_OVERAGE_RESERVE_MS", _DEFAULT_OVERAGE_RESERVE_MS)


def _panic_overage_ms() -> int:
    return _env_int("CHAOS_PANIC_OVERAGE_MS", _DEFAULT_PANIC_OVERAGE_MS)


def _overage_per_turn_cap_ms() -> int:
    return _env_int("CHAOS_OVERAGE_PER_TURN_CAP_MS", _OVERAGE_PER_TURN_CAP_MS)


def _py_margin_floor_ms() -> float:
    return _env_float("CHAOS_PY_MARGIN_MS", float(_DISPATCH_MARGIN_MS))


def _rust_post_margin_floor_ms() -> float:
    return _env_float("CHAOS_RUST_POST_MARGIN_MS", float(_DEFAULT_RUST_POST_MARGIN_MS))


def _update_peakish(current: float, sample: float, floor: float) -> float:
    sample = max(floor, float(sample))
    current = max(floor, float(current))
    alpha = 0.40 if sample > current else 0.05
    return max(floor, current * (1.0 - alpha) + sample * alpha)


def _act_timeout_ms(obs: dict) -> float:
    cfg = obs.get("config") if isinstance(obs.get("config"), dict) else {}
    raw = cfg.get("actTimeout")
    if raw is None:
        return float(_turn_target_ms())
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(_turn_target_ms())
    return value * 1000.0


def _remaining_overage_ms(obs: dict) -> float:
    try:
        return max(0.0, float(obs.get("remainingOverageTime", 0.0))) * 1000.0
    except (TypeError, ValueError):
        return 0.0


def _effective_reserve_ms() -> float:
    py_margin_ms = max(_py_margin_floor_ms(), _py_overhead_margin_ms)
    post_margin_ms = max(_rust_post_margin_floor_ms(), _rust_post_margin_ms)
    return max(
        float(_overage_reserve_ms()),
        py_margin_ms + post_margin_ms + _overage_burn_margin_ms,
    )


def _overage_panic(obs: dict) -> bool:
    return _USE_PROD_LIMITS and _remaining_overage_ms(obs) <= float(_panic_overage_ms())


def _observe_overage(obs: dict) -> None:
    global _last_remaining_overage_s, _last_overage_step, _overage_burn_margin_ms
    if not _USE_PROD_LIMITS:
        return
    try:
        step = int(obs.get("step", 0))
        remaining_s = max(0.0, float(obs.get("remainingOverageTime", 0.0)))
    except (TypeError, ValueError):
        return
    if (
        _last_remaining_overage_s is not None
        and _last_overage_step is not None
        and step > _last_overage_step
    ):
        burn_ms = max(0.0, (_last_remaining_overage_s - remaining_s) * 1000.0)
        burn_ms = min(burn_ms, float(_overage_per_turn_cap_ms() + 500))
        _overage_burn_margin_ms = _update_peakish(_overage_burn_margin_ms, burn_ms, 0.0)
    _last_remaining_overage_s = remaining_s
    _last_overage_step = step


def _budget_plan(obs: dict, elapsed_ms: float) -> tuple[int, int, int, float]:
    turn_target_ms = float(_turn_target_ms())
    act_timeout_ms = _act_timeout_ms(obs) if _USE_PROD_LIMITS else turn_target_ms
    remaining_ms = _remaining_overage_ms(obs) if _USE_PROD_LIMITS else 0.0
    py_margin_ms = max(_py_margin_floor_ms(), _py_overhead_margin_ms)
    post_margin_ms = max(_rust_post_margin_floor_ms(), _rust_post_margin_ms)
    reserve_ms = _effective_reserve_ms()
    if _USE_PROD_LIMITS and remaining_ms <= float(_panic_overage_ms()):
        return 0, 0, max(0, int(post_margin_ms)), 0.0
    extra_ms = 0.0
    if _USE_PROD_LIMITS:
        extra_ms = min(float(_overage_per_turn_cap_ms()), max(0.0, remaining_ms - reserve_ms))

    hard_total_ms = act_timeout_ms + extra_ms
    base_total_ms = min(turn_target_ms, hard_total_ms)
    base_available_ms = base_total_ms - elapsed_ms - py_margin_ms
    hard_available_ms = hard_total_ms - elapsed_ms - py_margin_ms

    hard_budget_ms = max(0, int(hard_available_ms))
    base_budget_ms = max(0, min(int(base_available_ms), hard_budget_ms))
    if base_budget_ms < _MIN_SEARCH_MS <= hard_budget_ms:
        base_budget_ms = _MIN_SEARCH_MS
    hard_budget_ms = max(base_budget_ms, hard_budget_ms)
    return base_budget_ms, hard_budget_ms, max(0, int(post_margin_ms)), extra_ms


def _update_timing_telemetry(timing: object, total_ms: float, il_ms: float) -> None:
    global _py_overhead_margin_ms, _rust_post_margin_ms
    if not isinstance(timing, dict):
        return
    try:
        rust_total_ms = float(timing.get("rust_total_ms", 0.0) or 0.0)
        redirect_ms = float(timing.get("redirect_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        return
    if rust_total_ms > 0.0:
        py_overhead_ms = max(0.0, total_ms - il_ms - rust_total_ms)
        _py_overhead_margin_ms = _update_peakish(
            _py_overhead_margin_ms,
            py_overhead_ms + 10.0,
            _py_margin_floor_ms(),
        )
    _rust_post_margin_ms = _update_peakish(
        _rust_post_margin_ms,
        redirect_ms + 25.0,
        _rust_post_margin_floor_ms(),
    )


def _il_k() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_K", "5")))


def _il_min_prob() -> float:
    return float(os.environ.get("CHAOS_IL_MIN_PROB", "0.02"))


def _il_skip_turns() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_SKIP_TURNS", "12")))


def _il_lead_skip_step() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_LEAD_SKIP_STEP", "100")))


def _il_lead_ship_ratio() -> float:
    return float(os.environ.get("CHAOS_IL_LEAD_SHIP_RATIO", "3.0"))


def _il_lead_prod_ratio() -> float:
    return float(os.environ.get("CHAOS_IL_LEAD_PROD_RATIO", "2.0"))


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


def _il_candidates(
    obs: dict,
    num_players: int,
    player: int | None = None,
    k: int | None = None,
) -> list[dict]:
    k = _il_k() if k is None else max(0, int(k))
    if k <= 0:
        return []
    if player is not None and int(obs.get("player", 0)) != int(player):
        obs = dict(obs)
        obs["player"] = int(player)
    return _il_for(num_players).top_actions(obs, k, _il_min_prob())


def _live_players(obs: dict) -> list[int]:
    seen: set[int] = set()
    for planet in obs.get("planets", []) or []:
        try:
            owner = int(planet[1])
        except (TypeError, ValueError, IndexError):
            continue
        if owner >= 0:
            seen.add(owner)
    for fleet in obs.get("fleets", []) or []:
        try:
            owner = int(fleet[1])
        except (TypeError, ValueError, IndexError):
            continue
        if owner >= 0:
            seen.add(owner)
    return sorted(seen)


def _opponent_player_2p(obs: dict) -> int | None:
    me = int(obs.get("player", 0))
    opponents = [p for p in _live_players(obs) if p != me]
    return opponents[0] if len(opponents) == 1 else None


def _il_candidate_bundle(obs: dict, num_players: int) -> dict:
    mine = _il_candidates(obs, num_players)
    opponent: list[dict] = []
    opponent_player = None
    if num_players == 2:
        opponent_player = _opponent_player_2p(obs)
        if opponent_player is not None:
            opponent = _il_candidates(
                obs,
                num_players,
                player=opponent_player,
                k=min(_il_k(), _MAX_OPPONENT_IL_CANDIDATES),
            )
    return {
        "mine": mine,
        "opponent": opponent,
        "opponent_player": opponent_player,
    }


def _ratio_met(mine: float, theirs: float, ratio: float) -> bool:
    if mine <= 0.0:
        return False
    if theirs <= 0.0:
        return True
    return mine >= ratio * theirs


def _il_winning_skip(obs: dict) -> bool:
    if int(obs.get("step", 0)) < _il_lead_skip_step():
        return False
    me = int(obs.get("player", 0))
    my_ships = 0.0
    other_ships = 0.0
    my_prod = 0.0
    other_prod = 0.0

    for planet in obs.get("planets", []) or []:
        if len(planet) < 7:
            continue
        owner = int(planet[1])
        ships = max(0.0, float(planet[5]))
        production = max(0.0, float(planet[6]))
        if owner == me:
            my_ships += ships
            my_prod += production
        elif owner >= 0:
            other_ships += ships
            other_prod += production

    for fleet in obs.get("fleets", []) or []:
        if len(fleet) < 7:
            continue
        owner = int(fleet[1])
        ships = max(0.0, float(fleet[6]))
        if owner == me:
            my_ships += ships
        elif owner >= 0:
            other_ships += ships

    return _ratio_met(my_ships, other_ships, _il_lead_ship_ratio()) and _ratio_met(
        my_prod, other_prod, _il_lead_prod_ratio()
    )


def _il_pass(obs: dict, t0: float, num_players: int) -> tuple[dict | None, str]:
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

    dispatch_margin_ms = max(_DISPATCH_MARGIN_MS, int(_py_overhead_margin_ms))
    deadline_s = (
        _turn_target_ms() - _MIN_SEARCH_MS - dispatch_margin_ms
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
    fut = _IL_EXEC.submit(_il_candidate_bundle, obs, num_players)
    try:
        bundle = fut.result(timeout=max(0.0, deadline_s))
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
    return bundle, f"{tag}:{_il_for(num_players).dataset_name}"


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
    _observe_overage(p)
    num_players = _aph._infer_num_players(p)
    tag = "2p" if num_players == 2 else "4p"
    panic = _overage_panic(p)
    cands = None
    opp_cands = None
    # Only live two-player states use IL.
    if panic:
        il_desc = f"{tag}:panic"
    elif num_players == 2 and p["step"] >= _il_skip_turns():
        if _il_k() <= 0:
            il_desc = f"{tag}:k0"
        elif _il_winning_skip(p):
            il_desc = f"{tag}:leadskip"
        else:
            bundle, il_desc = _il_pass(p, t0, num_players)
            cands = bundle.get("mine") if bundle else None
            opp_cands = bundle.get("opponent") if bundle else None
            if cands:
                p["il_candidates"] = [c["action"] for c in cands]
                p["il_candidate_probs"] = [c["prob"] for c in cands]
                p["il_candidate_indices"] = [c["index"] for c in cands]
            if opp_cands:
                p["opp_il_candidates"] = [c["action"] for c in opp_cands]
                p["opp_il_candidate_probs"] = [c["prob"] for c in opp_cands]
                p["opp_il_candidate_indices"] = [c["index"] for c in opp_cands]
    elif num_players >= 3:
        il_desc = f"{tag}:off"
    else:
        il_desc = f"{tag}:skip" if num_players == 2 else None
    il_ms = (time.perf_counter() - t0) * 1000
    base_budget_ms, hard_budget_ms, post_margin_ms, extra_ms = _budget_plan(p, il_ms)
    p["budget_ms"] = base_budget_ms
    p["base_budget_ms"] = base_budget_ms
    p["hard_budget_ms"] = hard_budget_ms
    p["post_search_margin_ms"] = post_margin_ms
    p["return_timing"] = True
    if os.environ.get("OW_DEBUG"):
        print(
            f"[chaos] step={p['step']} il_ms={il_ms:.0f} "
            f"base_ms={base_budget_ms} hard_ms={hard_budget_ms} "
            f"post_ms={post_margin_ms} extra_ms={extra_ms:.0f} "
            f"py_margin={_py_overhead_margin_ms:.0f} rust_post={_rust_post_margin_ms:.0f} "
            f"burn_margin={_overage_burn_margin_ms:.0f} "
            f"reserve={_effective_reserve_ms():.0f} panic_cutoff={_panic_overage_ms()} "
            f"panic={panic} "
            f"il={il_desc} il_candidates={cands} opp_il_candidates={opp_cands}",
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
        response = json.loads(r.decode())
        total_ms = (time.perf_counter() - t0) * 1000
        if isinstance(response, dict):
            _update_timing_telemetry(response.get("timing"), total_ms, il_ms)
            return response.get("moves", [])
        return response


if _il_k() > 0:
    _warm_torch()
elif os.environ.get("OW_DEBUG"):
    print("[chaos] IL disabled by CHAOS_IL_K=0; skipping torch warmup", file=sys.stderr)
