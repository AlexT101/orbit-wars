"""Incremental decisiveness-aux extractor (no full re-extraction, no Rust).

The decided/decisiveness logic needs seat-invariant, player-count-aware state
quantities (per-player ship strength + production, neutral production) that the
original `summary_v2` extraction never persisted (it kept only derived ratios).
Rather than re-run the expensive Rust feature math, this re-reads the SAME replay
observations and computes ONLY those cheap quantities, then attaches them to the
existing gated NPZ as a `decisiveness_aux` block — leaving `summary_v2`, labels,
meta, etc. byte-for-byte unchanged (existing models unaffected).

Alignment is by KEY, not order: the gated NPZ stores `game_files` (local gid ->
"tag:entry" zip entry) and `meta[:, (gid, step)]`. Aux is seat-invariant (same for
all slots of a state, since observations are full-board), so we compute it once
per (gid, step) from any slot's observation and map it onto every matching row.

Per-row aux columns (9, float32):
    [0:4] ship_strength per player slot 0..3   (planet ships + in-flight, matches
          value_net.rs `strength[]`)
    [4:8] production    per player slot 0..3   (planet production)
    [8]   neutral production                    (owner == -1 planets)

Planet array layout (parse_state): [id, owner, x, y, radius, ships, production]
Fleet  array layout (parse_state): [id, owner, x, y, angle, from_planet_id, ships]

Usage (from train/):
    python build_aux.py --in data/2p/_ladder_work/gated_replays_6_07.npz \
        --zip ../../../../ladder_replays/replays_6_07.zip
  (rewrites the NPZ in place with an added `decisiveness_aux`; use --out to write
   a copy instead.)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

NP = 4               # engine MAX_PLAYERS
AUX_DIM = 2 * NP + 1  # 9: ships[4] + prod[4] + neutral_prod
P_OWNER, P_SHIPS, P_PROD = 1, 5, 6
F_OWNER, F_SHIPS = 1, 6


def state_aux(obs: dict) -> np.ndarray:
    """Seat-invariant aux for one observation (full board). Matches the Rust
    `strength[]` accumulation: ships = planet garrisons + in-flight fleets."""
    ships = np.zeros(NP, dtype=np.float64)
    prod = np.zeros(NP, dtype=np.float64)
    neutral_prod = 0.0
    for p in obs.get("planets") or ():
        owner = int(p[P_OWNER])
        if 0 <= owner < NP:
            ships[owner] += float(p[P_SHIPS])
            prod[owner] += float(p[P_PROD])
        elif owner == -1:
            neutral_prod += float(p[P_PROD])
    for f in obs.get("fleets") or ():
        owner = int(f[F_OWNER])
        if 0 <= owner < NP:
            ships[owner] += float(f[F_SHIPS])
    out = np.empty(AUX_DIM, dtype=np.float32)
    out[0:NP] = ships
    out[NP:2 * NP] = prod
    out[2 * NP] = neutral_prod
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--zip", required=True, type=Path,
                   help="the replay zip this NPZ was extracted from")
    p.add_argument("--out", type=Path, default=None,
                   help="output NPZ (default: rewrite --in in place)")
    args = p.parse_args()

    d = np.load(args.inp, allow_pickle=False)
    if "game_files" not in d.files:
        raise SystemExit(f"{args.inp} has no game_files; cannot map gid->replay entry")
    meta = d["meta"].astype(np.int64)
    game_files = [str(s) for s in d["game_files"]]
    gid_col, step_col = meta[:, 0], meta[:, 1]
    n_rows = meta.shape[0]

    # (gid -> set of steps we actually need an aux value for)
    need = {}
    for g, s in zip(gid_col.tolist(), step_col.tolist()):
        need.setdefault(g, set()).add(s)

    zf = zipfile.ZipFile(args.zip)
    # entry name in game_files is "tag:entry"; strip the tag prefix.
    aux_by_key: dict[tuple[int, int], np.ndarray] = {}
    missing_games = 0
    for gid, steps_needed in need.items():
        tagged = game_files[gid]
        entry = tagged.split(":", 1)[1] if ":" in tagged else tagged
        try:
            data = json.loads(zf.read(entry))
        except Exception:
            missing_games += 1
            continue
        steps = data.get("steps") or []
        for st in steps_needed:
            if st < 0 or st >= len(steps):
                continue
            row = steps[st]
            if not isinstance(row, list) or not row:
                continue
            # any slot works (full-board); find the first with an observation
            obs = None
            for slot_obj in row:
                if isinstance(slot_obj, dict):
                    o = slot_obj.get("observation")
                    if o and o.get("planets"):
                        obs = o
                        break
            if obs is None:
                continue
            aux_by_key[(gid, st)] = state_aux(obs)
    zf.close()

    aux = np.zeros((n_rows, AUX_DIM), dtype=np.float32)
    matched = 0
    for i in range(n_rows):
        a = aux_by_key.get((int(gid_col[i]), int(step_col[i])))
        if a is not None:
            aux[i] = a
            matched += 1
    print(f"  rows={n_rows:,} matched={matched:,} ({100*matched/max(n_rows,1):.2f}%)  "
          f"games_missing={missing_games}")
    if matched < n_rows:
        print(f"  WARN {n_rows - matched:,} rows had no aux (state not found)")

    out_path = args.out or args.inp
    payload = {k: d[k] for k in d.files}
    payload["decisiveness_aux"] = aux
    np.savez_compressed(out_path, **payload)
    print(f"  wrote {out_path}  (+decisiveness_aux {aux.shape})")


if __name__ == "__main__":
    main()
