"""prometheus_v1_transformer = prometheus_v1 + target-predictor prior.

Per turn we:
  1. extract per-planet + global features (build_dataset_v0 schema, 38 planet
     feats x 12 globals = matches transformer_2k.pt).
  2. forward the PlanetTransformer once to get a sigmoid probability per planet
     ("P(my side targets this planet this turn)").
  3. inject the {planet_id: prob} dict as a `target_priors` field on the JSON
     observation sent to the Rust binary on stdin. The binary stashes it
     thread-locally and apollo's root candidate generator (only when the player
     matches the stashed player) uses it to prefilter and re-weight targets.

If the predictor fails to load, we send observations without `target_priors`
and the binary falls back to upstream prometheus behavior. Env vars:

  APOLLO_TF_CKPT          model checkpoint (default: target_predictor/train/weights/transformer_2k.pt)
  APOLLO_TF_DEVICE        torch device (default: cpu)
  APOLLO_TF_DISABLE       set to "1" to bypass the model entirely (acts as stock prometheus_v1)
  APOLLO_TF_PRIOR_FLOOR   per-target prefilter floor read by Rust (default: 0.05)
  APOLLO_TF_PRIOR_ALPHA   selection-key multiplier read by Rust (default: 2.0)
"""

import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import traceback

_PROC = None
_LOCK = threading.Lock()
_BIN_NAME = "alphaow-bot"
_NET_2P_NAME = "xgb_46p12e88t11_latest.json"
_NET_4P_NAME = "xgb_4p_v2_rank4_latest.json"


def _pump_stderr(pipe):
    try:
        for line in iter(pipe.readline, b""):
            try:
                sys.stderr.write(line.decode("utf-8", "replace"))
                sys.stderr.flush()
            except Exception:
                pass
    except Exception:
        pass


def _bin_in(d):
    names = (_BIN_NAME + ".exe", _BIN_NAME) if sys.platform == "win32" else (_BIN_NAME,)
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return None


def _wrapper_dir():
    """Locate the directory this main.py lives in.

    Kaggle environments load bots via `exec()` which strips `__file__`, so we
    need a frame-based fallback. Order:
      1. `__file__` (normal Python imports / run_match.py)
      2. inspect.currentframe().f_code.co_filename (Kaggle exec)
      3. PROMETHEUS_DIR env var (explicit override)
      4. ./bots/mine/prometheus_v1_transformer relative to cwd
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    frame = inspect.currentframe()
    if frame is not None:
        path = frame.f_code.co_filename
        if path and path != "<string>" and os.path.isfile(path):
            return os.path.dirname(os.path.abspath(path))
    env = os.environ.get("PROMETHEUS_DIR")
    if env:
        return env
    return os.path.join(os.getcwd(), "bots", "mine", "prometheus_v1_transformer")


def _opt(path):
    return path if path and os.path.isfile(path) else None


def _locate():
    # Use a transformer-specific env var so when the kaggle-env runner shares
    # ALPHAOW_BOT_BIN with the prometheus_v1 wrapper (their main.py reads the
    # same var), our wrapper still finds its own binary and not v1's.
    env_bin = os.environ.get("PROMETHEUS_TRANSFORMER_BIN")
    if env_bin and os.path.isfile(env_bin):
        d = os.path.dirname(env_bin)
        return (env_bin, _opt(os.path.join(d, _NET_2P_NAME)),
                _opt(os.path.join(d, _NET_4P_NAME)), d, None)

    wd = _wrapper_dir()

    for d in (wd, "/kaggle_simulations/agent", os.getcwd()):
        if d:
            b = _bin_in(d)
            if b:
                return (b, _opt(os.path.join(d, _NET_2P_NAME)),
                        _opt(os.path.join(d, _NET_4P_NAME)), d, None)

    crate = wd or os.getcwd()
    release = os.path.join(crate, "target", "release")
    binary = _bin_in(release) or os.path.join(
        release, _BIN_NAME + (".exe" if sys.platform == "win32" else "")
    )
    wdir = os.path.join(crate, "train", "weights")
    return (binary, _opt(os.path.join(wdir, _NET_2P_NAME)),
            _opt(os.path.join(wdir, _NET_4P_NAME)), crate, crate)


def _build_if_needed(path, build_cwd):
    if os.path.isfile(path):
        return
    if not build_cwd:
        raise RuntimeError(f"binary not found at {path} and no build tree to build it from")
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError(f"binary not found at {path} and cargo not on PATH")
    subprocess.check_call([cargo, "build", "--release"], cwd=build_cwd)


def _ensure_executable(path):
    try:
        st = os.stat(path).st_mode
        os.chmod(path, st | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


# --- target-predictor inference -------------------------------------------------

_WRAPPER_DIR_PATH = _wrapper_dir() or os.getcwd()
# Predictor lives at bots/mine/target_predictor/train/ (parallel to
# bots/mine/prometheus_v1_transformer/).
_TARGET_PRED_TRAIN = os.path.abspath(
    os.path.join(_WRAPPER_DIR_PATH, "..", "target_predictor", "train")
)
if _TARGET_PRED_TRAIN not in sys.path:
    sys.path.insert(0, _TARGET_PRED_TRAIN)

_DEFAULT_CKPT = os.path.join(_TARGET_PRED_TRAIN, "weights", "transformer_2k.pt")
_CKPT_PATH = os.environ.get("APOLLO_TF_CKPT", _DEFAULT_CKPT)
_DEVICE_NAME = os.environ.get("APOLLO_TF_DEVICE", "cpu")
_DISABLE = os.environ.get("APOLLO_TF_DISABLE", "0") == "1"
_NMAX = 30
_CKPT = None
_LOAD_TRIED = False


def _try_load_model():
    global _CKPT, _LOAD_TRIED
    if _LOAD_TRIED:
        return _CKPT
    _LOAD_TRIED = True
    if _DISABLE:
        sys.stderr.write("prometheus_v1_transformer: model disabled via APOLLO_TF_DISABLE=1\n")
        return None
    try:
        import predict  # noqa: F401
        _CKPT = predict.load_checkpoint(_CKPT_PATH, _DEVICE_NAME)
        if _CKPT.get("f_planet") != 38 or _CKPT.get("f_global") != 12:
            sys.stderr.write(
                "prometheus_v1_transformer: checkpoint schema "
                f"f_planet={_CKPT.get('f_planet')} f_global={_CKPT.get('f_global')} "
                "does not match build_dataset_v0 (38, 12); disabling priors.\n"
            )
            _CKPT = None
    except Exception as exc:
        sys.stderr.write(f"prometheus_v1_transformer: model load failed: {exc!r}\n")
        traceback.print_exc(file=sys.stderr)
        _CKPT = None
    return _CKPT


def _compute_priors(obs_dict):
    ckpt = _try_load_model()
    if ckpt is None:
        return None
    try:
        import numpy as np
        import build_dataset_v0 as bd
        import predict
        state = bd.parse_state(obs_dict)
        player = int(obs_dict.get("player", 0))
        fake_change = {p["id"]: 0 for p in state["planets"]}
        feats, globals_, pids = bd.extract_per_player(state, player, fake_change)
        feats38 = feats[:, :38].astype(np.float32, copy=False)
        probs = predict.predict(ckpt, feats38, globals_, n_max=_NMAX)
        # JSON object keys must be strings; Rust parses each key with .parse::<i64>().
        return {str(int(pids[i])): float(probs[i]) for i in range(len(pids))}
    except Exception as exc:
        sys.stderr.write(f"prometheus_v1_transformer: inference failed: {exc!r}\n")
        traceback.print_exc(file=sys.stderr)
        return None


# --- normal prometheus wrapper plumbing -----------------------------------------


def _norm(o):
    g = o.get if isinstance(o, dict) else (lambda k, d=None: getattr(o, k, d))
    return {
        "player": g("player", 0),
        "step": g("step", 0),
        "planets": list(g("planets", []) or []),
        "fleets": list(g("fleets", []) or []),
        "angular_velocity": g("angular_velocity", 0.0),
        "initial_planets": list(g("initial_planets", []) or []),
        "comets": list(g("comets", []) or []),
        "comet_planet_ids": list(g("comet_planet_ids", []) or []),
    }


def _ensure():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    binary, net_2p, net_4p, run_cwd, build_cwd = _locate()
    _build_if_needed(binary, build_cwd)
    _ensure_executable(binary)
    env = dict(os.environ)
    for k in (
        "OW_PLANNER", "OW_PUCT_C", "OW_K_ROOT", "OW_K_NON_ROOT",
        "OW_ROLLOUT", "OW_ROLLOUT_DEPTH", "OW_ROLLOUT_REACTIVE",
        "OW_ROLLOUT_NOISE", "OW_DUCT_ENUMERATE", "OW_NO_COOP", "OW_NO_REUSE",
        "OW_FOCUSED_CANDIDATES", "OW_SELECTION", "OW_EXP3_ETA",
        "OW_EXP3_GAMMA", "OW_DEBUG", "OW_PROFILE",
    ):
        env.pop(k, None)
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    env.setdefault("ALPHAOW_BUDGET_MS", "500")

    if net_2p:
        env.setdefault("ALPHAOW_VALUE_NET_PATH_2P", net_2p)
        env.setdefault("ALPHAOW_VALUE_NET_PATH", net_2p)
    if net_4p:
        env.setdefault("ALPHAOW_VALUE_NET_PATH_4P", net_4p)
    _PROC = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=run_cwd,
        env=env,
        bufsize=0,
    )
    threading.Thread(
        target=_pump_stderr, args=(_PROC.stderr,), daemon=True
    ).start()
    return _PROC


def agent(obs, config=None):
    p = _norm(obs)
    if config is not None:
        cfg = {}
        for k in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius", "boardSize", "cometSpeed"):
            v = config.get(k) if isinstance(config, dict) else getattr(config, k, None)
            if v is not None:
                cfg[k] = v
        if cfg:
            p["config"] = cfg
    # Compute priors before serializing so they ride on the same JSON line.
    priors = _compute_priors(p)
    if priors:
        p["target_priors"] = priors
    with _LOCK:
        proc = _ensure()
        try:
            proc.stdin.write((json.dumps(p, separators=(",", ":")) + "\n").encode())
            proc.stdin.flush()
            r = proc.stdout.readline()
        except (BrokenPipeError, OSError):
            global _PROC
            _PROC = None
            return []
        if not r:
            return []
        try:
            return json.loads(r.decode())
        except json.JSONDecodeError:
            return []
