"""Run a small batch of observations through both the Python derivation
(summary_features.summary_features) and the Rust extraction (the
`summary_parity` bin) and report any per-feature mismatch.

This guards against drift between the two extraction paths after
adding new features.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
APHRODITE_DIR = ROOT / "bots" / "mine" / "aphrodite"
BIN = APHRODITE_DIR / "target" / "release" / "summary_parity"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summary_features import summary_features, FEATURE_NAMES  # noqa: E402


def make_obs(seed):
    """Use the rust engine to generate one observation."""
    from engine_parity_checker.candidates.rust import RustEngine

    engine = RustEngine()
    obs = engine.reset(seed, 2)
    return obs[0].as_dict()


def py_summary_from_obs(obs):
    """Derive summary features the same way we derive them at training
    time: by going through the Rust 2728-d extraction (via the bot),
    then `summary_features`."""
    proc = subprocess.run(
        [str(APHRODITE_DIR / "target" / "release" / "summary_parity")],
        input=json.dumps(obs).encode(),
        capture_output=True,
        timeout=10,
    )
    out = proc.stdout.decode().strip().splitlines()
    if not out:
        return None, proc.stderr.decode()
    parts = out[0].split(",")
    step = int(parts[0])
    player = int(parts[1])
    rust_feats = np.array([float(x) for x in parts[2:]], dtype=np.float32)
    return (step, player, rust_feats), None


def main():
    seeds = [1, 7, 42, 100, 2025]
    print(f"checking parity on {len(seeds)} seeds")
    total_err = 0.0
    for seed in seeds:
        obs = make_obs(seed)
        result, err = py_summary_from_obs(obs)
        if result is None:
            print(f"seed={seed} rust call failed: {err[:200]}")
            continue
        step, player, rust_feats = result
        # Now derive Python summary using the rust 2728-d block.
        # Spawn the actual bot in dump mode to capture the 2728-d.
        # But we already have rust_feats — compare against Python deriving
        # from a feature vector. We need the 2728-d. Quickest path: use
        # the bot to dump 2728-d for this single observation.
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            dump_path = Path(f.name)
        env = {
            "PATH": "/usr/bin:/bin",
            "APHRODITE_DUMP_FEATURES_PATH": str(dump_path),
            "APHRODITE_BUDGET_MS": "5",
        }
        p = subprocess.Popen(
            [str(APHRODITE_DIR / "target" / "release" / "aphrodite")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        p.stdin.write((json.dumps(obs) + "\n").encode())
        p.stdin.flush()
        p.stdout.readline()  # discard move
        p.stdin.close()
        p.wait(timeout=10)
        raw = dump_path.read_bytes()
        dump_path.unlink()
        if len(raw) < 12 + 4 * 2728:
            print(f"seed={seed} dump empty/short")
            continue
        # Parse: step:i64, player:i32, features...
        feats_raw = np.frombuffer(raw[12 : 12 + 4 * 2728], dtype=np.float32).reshape(1, 2728)
        py_feats = summary_features(feats_raw)[0]
        # Compare
        if py_feats.shape[0] != rust_feats.shape[0]:
            print(f"seed={seed} DIM mismatch: py={py_feats.shape[0]} rust={rust_feats.shape[0]}")
            continue
        diffs = np.abs(py_feats - rust_feats)
        max_diff = diffs.max()
        total_err += max_diff
        if max_diff > 1e-3:
            print(f"seed={seed} max_diff={max_diff:.4f}  worst feature(s):")
            ranked = np.argsort(-diffs)
            for k in ranked[:5]:
                name = FEATURE_NAMES[k] if k < len(FEATURE_NAMES) else f"f{k}"
                print(f"   {name}: py={py_feats[k]:.4f} rust={rust_feats[k]:.4f} diff={diffs[k]:.4f}")
        else:
            print(f"seed={seed} OK (max_diff={max_diff:.2e})")
    print(f"\nsummed max_diff across seeds: {total_err:.4f}")


if __name__ == "__main__":
    main()
