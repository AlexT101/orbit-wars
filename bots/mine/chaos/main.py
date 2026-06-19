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

IL injection runs in 2p games only. A native 4p game plays as pure aphrodite
(no IL); if it decays to two surviving players the 2p checkpoint loads lazily on
that turn. Startup is split so the costly, reusable part is paid once, with no
checkpoint loaded: `import torch` + threadpool/kernel warmup + the
orbit_wars_model schema check happen at module load (`_warm_torch`), while the 2p
checkpoint loads lazily on the first turn IL is actually used.

Env knobs:
  CHAOS_IL_K            max IL candidates injected per turn (default 4)
  CHAOS_IL_MIN_PROB     drop IL suggestions below this policy prob (default 0.02)
  CHAOS_IL_SKIP_TURNS   skip IL injection on the first N turns (default 1 =
                         skip only turn 0, where the binary spawn already lands)
  CHAOS_TORCH_THREADS   torch / OpenMP intra-op threads (default 2; set before
                         `import torch`)
  CHAOS_TURN_TARGET_MS  total per-turn wall target override
  CHAOS_IL_CHECKPOINT   override the 2p IL checkpoint path
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
# Flat Kaggle bundle: everything (wrapper copy, IL support files, .so modules,
# checkpoint, binary, weights) sits next to main.py. Dev: resolve through the
# repo layout instead.
_BUNDLE = (HERE / "aphrodite_wrapper.py").is_file()
if _BUNDLE:
    _APH_WRAPPER = HERE / "aphrodite_wrapper.py"
    _IL_SYS_PATH = HERE
    DEFAULT_CHECKPOINT_2P = HERE / "osteo_il_2p_latest.pt"
else:
    ROOT = _repo_root()
    _APH_WRAPPER = ROOT / "bots" / "mine" / "aphrodite" / "main.py"
    _IL_SYS_PATH = ROOT / "experimental_arch" / "train_transformer"
    DEFAULT_CHECKPOINT_2P = (
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

# Set BEFORE `import torch` (which happens in _warm_torch / _ILPolicy, both run
# after this). Kaggle ships a CUDA torch build; on a CPU-only sim node it still
# enumerates the (absent) CUDA driver during import, which costs seconds — all
# charged to turn 1. The IL forward is tiny, so an OpenMP pool sized to the full
# vCPU count just slows `import torch` and then contends with the aphrodite
# binary's DUCT search threads each turn. Both are setdefault, so an explicit
# GPU/thread override from the environment still wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_TORCH_THREADS = os.environ.get("CHAOS_TORCH_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _TORCH_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _TORCH_THREADS)

# Total per-turn wall target (IL pass + search). Submission builds flip
# _USE_PROD_LIMITS to match aphrodite's production budget policy.
_DEV_TURN_TARGET_MS = 600
_SUBMISSION_TURN_TARGET_MS = 900
_USE_PROD_LIMITS = False
# Never squeeze the search below this, no matter how slow the IL pass was.
_MIN_SEARCH_MS = 250
# Margin between (target - il_elapsed) and the budget we hand the binary,
# covering JSON encode + IPC + the binary's own dispatch overhead.
_DISPATCH_MARGIN_MS = 100
# Architecture of the deployed 2p IL checkpoint. Only used to warm torch's
# kernels/threadpool at startup with a throwaway model; the real architecture
# and weights come from the checkpoint's own config at load time.
_WARM_MODEL = "entity_transformer_ngpt_action_features"


def _turn_target_ms() -> int:
    default = _SUBMISSION_TURN_TARGET_MS if _USE_PROD_LIMITS else _DEV_TURN_TARGET_MS
    return int(os.environ.get("CHAOS_TURN_TARGET_MS", default))


def _il_k() -> int:
    return max(0, int(os.environ.get("CHAOS_IL_K", "4")))


def _il_min_prob() -> float:
    return float(os.environ.get("CHAOS_IL_MIN_PROB", "0.02"))


def _il_skip_turns() -> int:
    # Skip IL injection on the first N turns (default 1 = just turn 0). Turn 0
    # already eats the one-time binary spawn + XGB weight parse; deferring the
    # first checkpoint load off it keeps those costs on separate turns' budgets
    # instead of stacking on the one turn most likely to spill into overage.
    return max(0, int(os.environ.get("CHAOS_IL_SKIP_TURNS", "8")))


def _checkpoint_path() -> Path:
    override = os.environ.get("CHAOS_IL_CHECKPOINT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CHECKPOINT_2P


class _ILPolicy:
    """Loads the osteo IL transformer and yields top-k decoded actions."""

    def __init__(self) -> None:
        if str(_IL_SYS_PATH) not in sys.path:
            sys.path.insert(0, str(_IL_SYS_PATH))
        import torch  # already imported + warmed by _warm_torch() at module load
        from model import build_policy

        self._torch = torch
        path = _checkpoint_path()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        # No warmup forward and no schema check here: both are paid once in
        # _warm_torch() at module load. With torch's kernels/threadpool already
        # warm, the first top_actions() forward for this checkpoint is ~10ms even
        # when this policy is built mid-game (a 4p board decaying to two players).

    def top_actions(self, obs: dict, k: int, min_prob: float) -> list[dict]:
        torch = self._torch
        from features import decode_move, encode_obs
        from model import tensorize

        batch = tensorize(encode_obs(obs, player=int(obs.get("player", 0))))
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
            move = decode_move(obs, int(idx))
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
                    "logit": int(idx),
                }
            )
        return actions


_IL: _ILPolicy | None = None


def _il() -> _ILPolicy:
    """Construct the 2p IL policy on first use. Failures propagate — chaos must
    never silently degrade to pure aphrodite."""
    global _IL
    if _IL is None:
        _IL = _ILPolicy()
    return _IL


def _warm_torch() -> None:
    """Pay the IL startup cost once, at module load.

    This is the expensive, reusable part: `import torch`, the OpenMP/threadpool
    spin-up, torch's kernel dispatch caches, and the live `orbit_wars_model`
    schema check. Crucially it loads NO checkpoint — the 2p weights load lazily
    on the first turn IL is needed (see `agent`), which in a native 4p game only
    happens if/when it decays to two surviving players. The schema (token/pair
    shapes) is player-count independent, so validating it here once is enough.

    Failures are loud, exactly as before: a stale `orbit_wars_model` schema
    raises here, before chaos ever plays a turn.
    """
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
    # One throwaway forward (random weights, discarded) warms torch's kernels and
    # threadpool on the exact tensor shapes the real per-turn forwards use, so a
    # checkpoint loaded later — even mid-game — gets a ~10ms first forward.
    warm = build_policy(_WARM_MODEL, hidden=128, transformer_layers=3, transformer_heads=4)
    warm.eval()
    with torch.inference_mode():
        warm(**tensorize(encode_obs(sample, player=0)))


# IL forwards run in a single background worker so a cold/slow turn can be
# abandoned (its result discarded) without ever blocking the turn past the IL
# deadline — apollo candidate-gen + DUCT run inside the binary regardless, so a
# skipped injection only forfeits that turn's IL candidates, never the search.
_IL_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chaos-il")
_IL_FUTURE = None  # in-flight task from a turn whose IL deadline expired


def _il_candidates(obs: dict) -> list[dict]:
    return _il().top_actions(obs, _il_k(), _il_min_prob())


def _il_pass(obs: dict, t0: float) -> tuple[list[dict] | None, str]:
    """Run the IL forward under a wall-clock deadline; return (candidates, desc).

    The deadline is whatever the turn target leaves after reserving the search
    floor, so IL gets the slack but can never push the search below
    `_MIN_SEARCH_MS`. On a cold turn (first inference, or the 4p->2p checkpoint
    load) the worker keeps running in the background and populates the cache for
    the next turn; we never reuse its result for a later, different obs."""
    global _IL_FUTURE
    if _IL_FUTURE is not None:
        if _IL_FUTURE.done():
            _IL_FUTURE = None  # drain + discard the stale (previous-obs) result
        else:
            return None, "2p:busy"  # prior forward still running

    deadline_s = (
        _turn_target_ms() - _MIN_SEARCH_MS - _DISPATCH_MARGIN_MS
    ) / 1000.0 - (time.perf_counter() - t0)
    fut = _IL_EXEC.submit(_il_candidates, obs)
    try:
        cands = fut.result(timeout=max(0.0, deadline_s))
    except _FutureTimeout:
        _IL_FUTURE = fut  # let it finish; its checkpoint/cache warms the next turn
        # Loud by design: a deadline cut is rare post-warmup (a cold first
        # forward or the 4p->2p checkpoint load), so surface it unconditionally.
        print(
            f"[chaos] IL deadline cut at step={obs.get('step')} 2p "
            f"deadline_ms={max(0.0, deadline_s) * 1000:.0f}; search runs without "
            f"IL candidates this turn",
            file=sys.stderr,
        )
        return None, "2p:deferred"
    return cands, f"2p:{_il().dataset_name}"


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
    # IL injects in 2p only. _infer_num_players counts surviving owners, so a
    # native 4p game reports 4 here (pure aphrodite, no IL) and only flips to 2 —
    # lazily loading the 2p checkpoint on that turn — if it decays to two players.
    num_players = _aph._infer_num_players(p)
    if num_players == 2 and p["step"] >= _il_skip_turns():
        cands, il_desc = _il_pass(p, t0)
        if cands:
            p["il_candidates"] = [c["action"] for c in cands]
            p["il_candidate_probs"] = [c["prob"] for c in cands]
            p["il_candidate_logits"] = [c["logit"] for c in cands]
    else:
        cands = None
        il_desc = "2p:skip" if num_players == 2 else None
    # Dynamic split of the turn target: whatever the IL pass (or 4p skip) left
    # goes to the search. Sent per turn — the binary's env budget is unused.
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


# Eager init at module load (Kaggle's setup window / first-turn overage): pay
# `import torch` + threadpool/kernel warmup + the schema check ONCE, loading no
# checkpoint. The 2p checkpoint loads lazily on the first turn IL is needed, so a
# native 4p game never loads it unless it decays to two players.
# A stale orbit_wars_model schema still raises here, before any turn is played.
_warm_torch()
