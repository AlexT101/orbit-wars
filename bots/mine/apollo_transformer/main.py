"""
apollo_transformer = apollo (Rust planner) + target-predictor prior.

Each turn we:
  1. extract per-planet + global features (build_dataset_v0 schema, 38 planet
     feats x 12 globals = matches transformer_2k.pt).
  2. forward the PlanetTransformer once to get a sigmoid probability per
     planet ("P(my side targets this planet this turn)").
  3. pass {planet_id: prob} as `target_priors` to the Rust planner, which:
       - drops candidate targets below APOLLO_TF_PRIOR_FLOOR (default 0.05),
       - multiplies its selection key by (1 + APOLLO_TF_PRIOR_ALPHA * prob)
         (default alpha = 2.0).

If the model fails to load or inference errors out, we fall back silently to
stock apollo (priors = None) so a model regression can't take the bot offline.

Env vars:
  APOLLO_TF_CKPT          model checkpoint (default: target_predictor/train/weights/transformer_2k.pt)
  APOLLO_TF_DEVICE        torch device (default: cpu)
  APOLLO_TF_DISABLE       set to "1" to bypass the model entirely (acts as stock apollo)
  APOLLO_TF_PRIOR_FLOOR   per-target prefilter floor read by Rust (default: 0.05)
  APOLLO_TF_PRIOR_ALPHA   selection-key multiplier read by Rust (default: 2.0)
"""

import inspect
import os
import sys
import traceback

_frame = inspect.currentframe()
if _frame is not None:
    _HERE = os.path.dirname(os.path.abspath(_frame.f_code.co_filename))
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import apollo_transformer_native

# Locate the target_predictor train/ dir relative to the repo. apollo_transformer
# lives at bots/mine/apollo_transformer/; the predictor lives at
# bots/mine/target_predictor/train/. Resolve via parent paths so the bot works
# even when imported from a non-repo cwd (Kaggle exec()).
_TARGET_PRED_TRAIN = os.path.abspath(
    os.path.join(_HERE, "..", "target_predictor", "train")
)
if _TARGET_PRED_TRAIN not in sys.path:
    sys.path.insert(0, _TARGET_PRED_TRAIN)

_DEFAULT_CKPT = os.path.join(_TARGET_PRED_TRAIN, "weights", "transformer_2k.pt")
_CKPT_PATH = os.environ.get("APOLLO_TF_CKPT", _DEFAULT_CKPT)
_DEVICE_NAME = os.environ.get("APOLLO_TF_DEVICE", "cpu")
_DISABLE = os.environ.get("APOLLO_TF_DISABLE", "0") == "1"

_BOT = apollo_transformer_native.Bot()

# Lazy-loaded so import failures (missing torch, missing checkpoint) don't break
# the bot; we just fall back to stock apollo with no priors.
_CKPT = None
_LOAD_TRIED = False
_NMAX = 30  # transformer_2k.pt was trained with N_MAX=30


def _try_load_model():
    """Load the predictor checkpoint once. On any failure (torch missing,
    weights missing, schema mismatch) leave _CKPT as None so the bot acts
    like stock apollo."""
    global _CKPT, _LOAD_TRIED
    if _LOAD_TRIED:
        return _CKPT
    _LOAD_TRIED = True
    if _DISABLE:
        sys.stderr.write("apollo_transformer: model disabled via APOLLO_TF_DISABLE=1\n")
        return None
    try:
        import predict  # noqa: F401
        _CKPT = predict.load_checkpoint(_CKPT_PATH, _DEVICE_NAME)
        if _CKPT.get("f_planet") != 38 or _CKPT.get("f_global") != 12:
            sys.stderr.write(
                "apollo_transformer: checkpoint schema "
                f"f_planet={_CKPT.get('f_planet')} f_global={_CKPT.get('f_global')} "
                "does not match build_dataset_v0 (38, 12); disabling priors.\n"
            )
            _CKPT = None
    except Exception as exc:
        sys.stderr.write(f"apollo_transformer: model load failed: {exc!r}\n")
        traceback.print_exc(file=sys.stderr)
        _CKPT = None
    return _CKPT


def _compute_priors(obs_dict):
    """Run the transformer once, return {planet_id: prob} or None on failure /
    when the model is disabled."""
    ckpt = _try_load_model()
    if ckpt is None:
        return None
    try:
        import numpy as np
        import build_dataset_v0 as bd
        import predict
        state = bd.parse_state(obs_dict)
        player = int(obs_dict.get("player", 0))
        # owner_change_turn is normally tracked across the game; synthesizing
        # "everyone last changed at turn 0" matches what the target_predictor
        # bot does at inference time and the feature is low-importance per the
        # README permutation study.
        fake_change = {p["id"]: 0 for p in state["planets"]}
        feats, globals_, pids = bd.extract_per_player(state, player, fake_change)
        # v0 emits 46 planet features; transformer_2k.pt was trained on the
        # first 38 (they are a prefix - see PLANET_FEAT_NAMES order).
        feats38 = feats[:, :38].astype(np.float32, copy=False)
        probs = predict.predict(ckpt, feats38, globals_, n_max=_NMAX)
        return {int(pids[i]): float(probs[i]) for i in range(len(pids))}
    except Exception as exc:
        sys.stderr.write(f"apollo_transformer: inference failed: {exc!r}\n")
        traceback.print_exc(file=sys.stderr)
        return None


_OBS_FIELDS = (
    "player",
    "planets",
    "fleets",
    "angular_velocity",
    "initial_planets",
    "comets",
    "comet_planet_ids",
    "remainingOverageTime",
)


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in _OBS_FIELDS}


def agent(obs):
    try:
        obs_dict = _as_dict(obs)
        priors = _compute_priors(obs_dict)
        moves = _BOT.compute_moves_with_search(obs_dict, priors)
    except Exception:
        traceback.print_exc()
        raise
    return [list(move) for move in moves]
