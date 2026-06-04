"""Self-play / cross-bot data collector for aphrodite's value net.

Drives the Rust engine, runs N matches between configurable bot pairings,
and for each side that runs as `aphrodite`, captures the per-turn feature
vectors that the bot dumps via `APHRODITE_DUMP_FEATURES_PATH`. Each turn's
features are labeled with the final-game reward of the player who saw
that state.

Output: a single NPZ with arrays
  features:   float32 [N, INPUT_DIM=2728]
  labels:     float32 [N]                  (final reward for that player)
  meta:       int32   [N, 4]               (game_idx, step, player, num_players)
  summary_v2: float32 [N, 46]
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# parents: train -> aphrodite -> mine -> bots -> repo
ROOT = Path(__file__).resolve().parents[4]
BOTS_DIR = ROOT / "bots"
APHRODITE_DIR = ROOT / "bots" / "mine" / "aphrodite"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BIN_PATH = APHRODITE_DIR / "target" / "release" / "aphrodite"

# Must match value_net.rs constants.
PER_OBJECT = 9
MAX_OBJECTS = 44
PER_BLOCK = MAX_OBJECTS * PER_OBJECT  # 396
DIST_BLOCK = MAX_OBJECTS * MAX_OBJECTS  # 1936
INPUT_DIM = 2 * PER_BLOCK + DIST_BLOCK  # 2728
SUMMARY_V2_DIM = 46
# New record layout: step(i64) + player(i32) + features(2728 f32) + summary_v2(46 f32)
RECORD_BYTES = 8 + 4 + 4 * INPUT_DIM + 4 * SUMMARY_V2_DIM
# Old layout (pre-v2): step(i64) + player(i32) + features only
LEGACY_RECORD_BYTES = 8 + 4 + 4 * INPUT_DIM
MAX_STEPS = 500


def label_for_rewards(rewards, slot: int) -> float:
    if len(rewards) >= 4:
        vals = [float(r) for r in rewards]
        best = max(vals)
        winners = [i for i, r in enumerate(vals) if r == best]
        if len(winners) != 1:
            return 0.0
        return 1.0 if slot == winners[0] else -1.0
    return float(rewards[slot])


def _silence():
    import contextlib

    @contextlib.contextmanager
    def s():
        sys.stdout.flush()
        sys.stderr.flush()
        out = os.dup(1)
        err = os.dup(2)
        n = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(n, 1)
            os.dup2(n, 2)
            yield
        finally:
            os.dup2(out, 1)
            os.dup2(err, 2)
            os.close(out)
            os.close(err)
            os.close(n)

    return s()


def bot_main_path(name: str) -> Path:
    direct = BOTS_DIR / name / "main.py"
    if direct.is_file():
        return direct
    for sub in BOTS_DIR.iterdir():
        if not sub.is_dir():
            continue
        cand = sub / name / "main.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(name)


class AphroditeDaemon:
    """Direct subprocess driver for aphrodite with per-process env control.

    Bypasses the Python wrapper, which captures os.environ at spawn time
    and so leaks env between concurrent aphrodite players in the same
    Python process.
    """

    def __init__(self, dump_path: Path | None, budget_ms: int, weights_path: Path | None,
                 weights_2p_path: Path | None = None):
        env = dict(os.environ)
        env.pop("APHRODITE_DUMP_FEATURES_PATH", None)
        env.pop("APHRODITE_VALUE_NET_PATH", None)
        env.pop("APHRODITE_VALUE_NET_PATH_2P", None)
        if dump_path is not None:
            env["APHRODITE_DUMP_FEATURES_PATH"] = str(Path(dump_path).resolve())
        env["APHRODITE_BUDGET_MS"] = str(budget_ms)
        if weights_path is not None:
            env["APHRODITE_VALUE_NET_PATH"] = str(Path(weights_path).resolve())
        # Secondary net used once a position has only 2 players left alive.
        if weights_2p_path is not None:
            env["APHRODITE_VALUE_NET_PATH_2P"] = str(Path(weights_2p_path).resolve())
        self.proc = subprocess.Popen(
            [str(BIN_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            cwd=str(APHRODITE_DIR),
            env=env,
            bufsize=0,
        )

    def __call__(self, obs_dict):
        import json

        # Match the bot's expected JSON shape (same as wrapper's _norm).
        p = {
            "player": int(obs_dict.get("player", 0)),
            "step": int(obs_dict.get("step", 0)),
            "planets": list(obs_dict.get("planets", []) or []),
            "fleets": list(obs_dict.get("fleets", []) or []),
            "angular_velocity": float(obs_dict.get("angular_velocity", 0.0)),
            "initial_planets": list(obs_dict.get("initial_planets", []) or []),
            "comets": list(obs_dict.get("comets", []) or []),
            "comet_planet_ids": list(obs_dict.get("comet_planet_ids", []) or []),
        }
        self.proc.stdin.write((json.dumps(p, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()
        r = self.proc.stdout.readline()
        if not r:
            return []
        try:
            return json.loads(r.decode())
        except json.JSONDecodeError:
            return []

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def load_other_agent(name: str):
    """Load a non-aphrodite bot via its main.py."""
    import importlib.util

    path = bot_main_path(name)
    mod_name = f"agent_{name}_{os.getpid()}_{int(time.time() * 1e6)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    with _silence():
        spec.loader.exec_module(module)
    return module.agent, module


def teardown_other(module):
    proc = getattr(module, "_PROC", None)
    if proc is None:
        return
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features [n, INPUT_DIM], steps [n], summary_v2 [n, 46]).
    Auto-detects new vs legacy record size."""
    empty = (
        np.zeros((0, INPUT_DIM), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, SUMMARY_V2_DIM), dtype=np.float32),
    )
    if not path.exists() or path.stat().st_size == 0:
        return empty
    raw = path.read_bytes()
    size = len(raw)
    # Try new format first (with v2 trailer).
    if size % RECORD_BYTES == 0 and size > 0:
        rec = RECORD_BYTES
        has_v2 = True
    elif size % LEGACY_RECORD_BYTES == 0:
        rec = LEGACY_RECORD_BYTES
        has_v2 = False
    else:
        # Mixed — be conservative, take floor of new format.
        rec = RECORD_BYTES
        has_v2 = True
    n = size // rec
    if n == 0:
        return empty
    arr = np.frombuffer(raw[: n * rec], dtype=np.uint8).reshape(n, rec)
    steps = arr[:, :8].view(np.int64).reshape(n).copy()
    feats = arr[:, 12 : 12 + 4 * INPUT_DIM].view(np.float32).reshape(n, INPUT_DIM).copy()
    if has_v2:
        v2 = arr[:, 12 + 4 * INPUT_DIM :].view(np.float32).reshape(n, SUMMARY_V2_DIM).copy()
    else:
        v2 = np.zeros((n, SUMMARY_V2_DIM), dtype=np.float32)
    return feats, steps, v2


def run_match(
    bots: list[str],
    seed: int,
    scratch: Path,
    budget_ms: int,
    weights_path: Path | None,
):
    from engine_parity_checker.candidates.rust import RustEngine

    n_players = len(bots)
    dumps: list[Path | None] = [None] * n_players
    agent_funcs: list = [None] * n_players
    closers: list = []  # callables that tear down a side

    for i, name in enumerate(bots):
        if name == "aphrodite":
            dumps[i] = scratch / f"feat_{seed}_p{i}.bin"
            dumps[i].write_bytes(b"")
            daemon = AphroditeDaemon(
                dump_path=dumps[i],
                budget_ms=budget_ms,
                weights_path=weights_path,
            )
            agent_funcs[i] = daemon
            closers.append(daemon.close)
        else:
            fn, mod = load_other_agent(name)
            agent_funcs[i] = fn
            closers.append(lambda m=mod: teardown_other(m))

    engine = RustEngine()
    obs = engine.reset(seed, n_players)
    done = False
    for _ in range(MAX_STEPS):
        actions = [agent_funcs[i](obs[i].as_dict()) for i in range(n_players)]
        obs, done = engine.step(actions)
        if done:
            break
    snap = engine.snapshot()
    rewards = snap.rewards or [0.0] * n_players

    for c in closers:
        try:
            c()
        except Exception:
            pass

    data = []
    for i in range(n_players):
        if dumps[i] is None:
            continue
        feats, steps, v2 = read_dump(dumps[i])
        if feats.size == 0:
            continue
        labels = np.full(feats.shape[0], label_for_rewards(rewards, i), dtype=np.float32)
        meta = np.stack(
            [
                np.zeros(feats.shape[0], dtype=np.int32),  # placeholder game_idx
                steps.astype(np.int32),
                np.full(feats.shape[0], i, dtype=np.int32),
                np.full(feats.shape[0], n_players, dtype=np.int32),
            ],
            axis=1,
        )
        data.append((feats, labels, meta, v2))
        try:
            dumps[i].unlink()
        except Exception:
            pass
    return data, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output NPZ path")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    p.add_argument("--budget-ms", type=int, default=200, help="aphrodite MCTS budget per turn")
    p.add_argument(
        "--pairings",
        default="aphrodite:aphrodite:1.0",
        help=(
            "Comma-separated bot specs. For 2p use bot0:bot1:weight; "
            "for 4p use bot0:bot1:bot2:bot3:weight. "
            "Use aphrodite on at least one side to collect daemon feature dumps."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="base seed (game_i uses seed+i)")
    p.add_argument("--weights", default=None, help="APHRODITE_VALUE_NET_PATH for collection rollouts (None = heuristic)")
    args = p.parse_args()

    pairings: list[tuple[list[str], float]] = []
    for spec in args.pairings.split(","):
        parts = spec.strip().split(":")
        expected = args.players + 1
        if len(parts) != expected:
            raise SystemExit(f"bad {args.players}p pairing: {spec} (expected {expected} colon-separated fields)")
        bots = parts[: args.players]
        w = float(parts[-1])
        if "aphrodite" not in bots:
            raise SystemExit(f"pairing has no aphrodite side to collect from: {spec}")
        pairings.append((bots, w))
    total_w = sum(p[1] for p in pairings)
    cum = []
    acc = 0.0
    for bots, w in pairings:
        acc += w / total_w
        cum.append((acc, bots))

    rng = random.Random(args.seed)
    scratch = Path(tempfile.mkdtemp(prefix="aphrodite_collect_"))

    all_feats = []
    all_labels = []
    all_meta = []
    all_v2 = []
    total_samples = 0
    t0 = time.time()

    def _flush(prefix=""):
        if not all_feats:
            return
        feats = np.concatenate(all_feats, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        meta = np.concatenate(all_meta, axis=0)
        v2 = np.concatenate(all_v2, axis=0)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, features=feats, labels=labels, meta=meta, summary_v2=v2)
        print(f"{prefix}wrote {feats.shape[0]} samples to {out}", flush=True)

    import signal

    def _on_sigint(signum, frame):
        _flush("[SIGINT] ")
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    flush_every = max(1, args.games // 10)
    for gi in range(args.games):
        r = rng.random()
        bots = None
        for thr, candidates in cum:
            if r < thr:
                bots = list(candidates)
                break
        if bots is None:
            bots = list(cum[-1][1])
        rng.shuffle(bots)
        seed = args.seed * 10_000 + gi
        print(
            f"[{gi+1}/{args.games}] {' vs '.join(bots)} seed={seed} "
            f"players={args.players} budget={args.budget_ms}ms",
            flush=True,
        )
        data, rewards = run_match(bots, seed, scratch, args.budget_ms, Path(args.weights) if args.weights else None)
        n_added = 0
        for feats, labels, meta, v2 in data:
            meta[:, 0] = gi
            all_feats.append(feats)
            all_labels.append(labels)
            all_meta.append(meta)
            all_v2.append(v2)
            n_added += feats.shape[0]
        total_samples += n_added
        elapsed = time.time() - t0
        rate = total_samples / max(elapsed, 1e-3)
        print(
            f"   rewards={rewards} samples_added={n_added} total={total_samples} elapsed={elapsed:.1f}s ({rate:.0f}/s)",
            flush=True,
        )
        if (gi + 1) % flush_every == 0:
            _flush("[checkpoint] ")

    if not all_feats:
        print("no samples collected", file=sys.stderr)
        sys.exit(1)
    _flush()


if __name__ == "__main__":
    main()
