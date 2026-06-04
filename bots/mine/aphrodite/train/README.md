# aphrodite training

We train **per-format XGBoost value nets** that aphrodite's DUCT search uses
for leaf evaluation:

- `weights/xgb_2p.json` — 2-player model (deployed)
- `weights/xgb_4p.json` — 4-player model (deployed)
- `weights/xgb_2p_old_top10.json` — legacy 2p model, kept only as the
  last-resort fallback in `main.py` (never reached when the above exist)

Each model is a `binary:logistic` gbtree over the 46-d **SummaryV2** features,
trained on Kaggle ladder replays with **recency** and **player-strength**
sample weighting. The whole flow is driven by `build_ladder.py`.

All commands below run from the **repo root**. On Windows use the project venv
interpreter `./venv/Scripts/python.exe`; on POSIX use `.venv/bin/python`.

---

## 0. Prerequisites (one-time)

Python deps in the venv (numpy + scipy ship already; xgboost does not):

```bash
./venv/Scripts/python.exe -m pip install xgboost
```

Rust binaries (the trainer shells out to `extract_v2`; eval drives `aphrodite`):

```bash
cd bots/mine/aphrodite && cargo build --release && cd ../../..
# builds target/release/{aphrodite, extract_v2, ...}
```

---

## 1. Download replays from Kaggle

Pull the daily episode dump for the `orbit-wars` competition and drop each
day's zip into the repo-root `ladder_replays/` folder, named `replays_M_DD.zip`:

```
ladder_replays/
  replays_5_14.zip
  replays_5_15.zip
  ...
  replays_6_03.zip
```

- Each zip is ~1.3 GB (~4–5k games). `ladder_replays/` is **gitignored** —
  never commit it (GitHub rejects >100 MB files; the data is re-downloadable).
- Roughly ~21% of games are 4-player, the rest 2-player. `--players` selects
  which to extract, so the same zips feed both the 2p and 4p runs.
- File names must contain `M_DD` (month_day); the trainer parses them to order
  days and to ramp the gate (see below). Consecutive days are assumed.

---

## 2. Train (the ladder pipeline)

`build_ladder.py` does the full run, one day per subprocess so memory stays
bounded (the full corpus is too large to extract at once):

```
for each day (oldest -> newest):
    build_from_zip.py        zip          -> raw_<day>.npz   (extract_v2: obs -> 46-d SummaryV2)
    filter_top10_..(--filter-only)        -> gated_<day>.npz (per-day quality gate)
    delete raw_<day>.npz
combine_npz.py  gated_*.npz (chronological) -> combined.npz  (source column = day rank)
filter_top10_..(--no-filter, weighted)      -> weights/xgb_<2p|4p>.json
```

### 2p

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/build_ladder.py \
  --replays-dir ladder_replays --players 2 \
  --recency-halflife 7 --rounds 2000 \
  --model-out bots/mine/aphrodite/train/weights/xgb_2p.json \
  --keep-temp
```

### 4p

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/build_ladder.py \
  --replays-dir ladder_replays --players 4 \
  --recency-halflife 7 --rounds 2000 \
  --model-out bots/mine/aphrodite/train/weights/xgb_4p.json \
  --keep-temp
```

What the defaults do:

- **Gate `strong-topn`** (per-perspective): a row is kept iff *that row's*
  player is in the day's top-N by win rate. This keeps the strong side of a
  strong-vs-weak game and drops only the weak side (a 1st-vs-30th game still
  contributes the strong player's positions).
- **Top-N ramp `--top-n-start 10` -> `--top-n-end 15`**: stricter on older
  days, looser on fresher days, interpolated by date. Override with `--top-n N`
  for a flat gate, or `--gate strong-median` / `--gate both-topn` / `--gate none`.
- **`--recency-halflife 7`**: down-weight older rows by 0.5 every 7 days (reads
  the `source` day-rank from `combine_npz`). `0` = uniform.
- **Quality weight** (on by default; `--no-quality-weight` to disable): weight
  each kept row by its player's strength percentile *within the kept set*,
  mapped to `[--quality-floor 0.25, 1.0]`. Composes multiplicatively with
  recency. (Meaningful only with a `strong-*` gate, which records `win_rate`.)
- **`--rounds 2000`** with `--early-stopping 50`: enough headroom for the
  booster to converge; early stopping picks the real count.

Useful flags:

- `--keep-temp` — keep the per-day NPZs and `combined.npz` so you can retrain
  with different weighting without re-extracting (see below). Without it, the
  scratch dir is cleaned at the end.
- `--resume` — skip days already gated (safe to re-run after an interruption).
- `--limit N` — cap games/day for a quick dry run.
- `--workers N` — passed through to `build_from_zip.py` extraction.

Scratch lives in `train/data/<2p|4p>/_ladder_work/` (gitignored).

### Retrain / sweep without re-extracting

With `--keep-temp`, the final train is one command over the kept
`combined.npz` — cheap to repeat with different knobs:

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/filter_top10_and_train_xgb.py \
  --data bots/mine/aphrodite/train/data/2p/_ladder_work/combined.npz \
  --no-filter --recency-halflife 7 --quality-weight --quality-floor 0.25 \
  --rounds 2000 --early-stopping 50 \
  --model-out bots/mine/aphrodite/train/weights/xgb_2p_try.json
```

Note: more boosting rounds improves offline logloss but yields a bigger model,
which costs more per leaf eval and so buys *fewer* MCTS iterations at a fixed
budget — always confirm a candidate with eval (§3), don't trust sign-acc alone.

---

## 3. Evaluate

`eval.py` plays real matches through the Rust engine and reports per-opponent
W/L/T from aphrodite's perspective. It is threaded (one match per worker
process).

### 2p

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/eval.py --players 2 \
  --weights bots/mine/aphrodite/train/weights/xgb_2p.json \
  --opponents apollo_fast producer \
  --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
  --no-swap --budget-ms 500 --threads 12
```

- `--swap` (default) plays both sides of each seed; `--no-swap` plays aphrodite
  as p0 only (use unique seeds for more seed diversity).

### 4p

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/eval.py --players 4 \
  --weights bots/mine/aphrodite/train/weights/xgb_4p.json \
  --weights-2p bots/mine/aphrodite/train/weights/xgb_2p.json \
  --opponents apollo_fast \
  --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
  --budget-ms 500 --threads 12
```

- In 4p, aphrodite plays **vs three copies of the opponent**. Seat order
  matters (it decides who you spawn next to), so seats are shuffled by a
  deterministic, seed-derived permutation and results are normalized back to
  aphrodite's perspective internally. `--swap` is ignored in 4p.
- `--weights-2p` enables the late-game 2p switchover (§5); omit it to eval the
  4p net alone. A/B the two to measure the switch's effect.

Opponents (resolved from `bots/**/<name>/main.py`): `heuristic`, `apollo_fast`
(in `bots/mine`), `owheuristic`, `producer` (in `bots/external`).

Threading caveat: each match uses ~1–2 cores. Keep `--threads <= cores/2`.
aphrodite's strength depends on fitting MCTS iterations into the wall-clock
`--budget-ms`, so **watch the reported avg ms** — if it climbs well above the
budget you're oversubscribed (contention), which understates strength. Use
`--threads 1` for a definitive timing/strength read.

---

## 4. Promote a model

`main.py` auto-selects weights by player count at runtime: `xgb_2p.json` for
2p, `xgb_4p.json` for 4p, else the `xgb_2p_old_top10.json` fallback. So a
trained model goes live simply by living at `weights/xgb_<2p|4p>.json`.

Convention when replacing a deployed model: archive the old one under a
descriptive name first, e.g.

```bash
cd bots/mine/aphrodite/train/weights
mv xgb_2p.json xgb_2p_fast.json       # archive the outgoing model
mv xgb_2p_candidate.json xgb_2p.json  # promote the winner
```

`build_submission.py` bundles `xgb_2p.json` / `xgb_4p.json` (plus the fallback)
into the Kaggle tarball.

---

## 5. Late-game 2p switchover (4p)

A 4p game that collapses to two survivors is effectively a 1v1, where the 2p
net is much stronger than the 4p net. So the bot can score such positions with
the 2p model:

- `value_net::predict` counts alive players in the **evaluated state** and, when
  exactly two are alive, uses the secondary net from `APHRODITE_VALUE_NET_PATH_2P`
  (falling back to the primary net if none is set). This is **per-leaf**, so a
  4p search scores its deep 2-survivor branches with the 2p net even before the
  real game has collapsed.
- **On by default at runtime**: `main.py` sets `APHRODITE_VALUE_NET_PATH_2P` to
  the resolved 2p net for every game. In a 2p game it just matches the primary
  (no behavior change); in 4p it engages once a position is down to two players.
- Both nets consume the same 46-d SummaryV2 features, so no feature/extraction
  changes are involved — only which booster scores the row.

To exercise it in eval, pass `--weights-2p` (see the 4p example in §3); the
daemon sets the env var for that match. Without `--weights-2p`, eval runs the
primary net alone — run both and compare to measure the switch's effect.

---

## Files in this directory

Pipeline (used by `build_ladder.py`):

- `build_ladder.py` — end-to-end driver (zips -> model).
- `build_from_zip.py` — stream replay JSONs out of zips through the `extract_v2`
  Rust binary into a SummaryV2 NPZ (`--players` selects 2p/4p games).
- `filter_top10_and_train_xgb.py` — quality gate (`--filter-only`) and the
  weighted XGBoost train (`--no-filter`, recency + quality + rounds).
- `combine_npz.py` — concatenate per-day NPZs with safe game-id offsets; writes
  the `source` day-rank column that recency weighting reads.

Eval & data:

- `eval.py` — threaded 2p/4p match eval vs an opponent set.
- `collect.py` — self-play / cross-bot data collection and the per-process
  aphrodite daemon driver (`eval.py` imports it).
- `train_xgb.py` — simple no-gate trainer for an already-combined NPZ (e.g.
  replay + self-play mixes).

Checks / utilities:

- `validate_extract.py` — independent pure-Python re-implementation of the
  extractor, cross-checked against the Rust output (guards against drift).
- `xgb_tune.py` — XGBoost hyperparameter sweep (offline metric only).
- `view_replay.py` — render a Kaggle episode JSON to a standalone HTML player.

---

## Data layout & git

| Path | Tracked? | Notes |
|---|---|---|
| `ladder_replays/` (repo root) | no | raw Kaggle zips, ~1.3 GB each |
| `train/data/2p/`, `train/data/4p/` (incl. `_ladder_work/`) | no | NPZ intermediates; `combined.npz` is hundreds of MB |
| `train/data/2p/old_top10.npz` | no | archived legacy 2p dataset (on disk for reference) |
| `train/weights/*.json` | yes | trained models (small, the actual artifacts) |

Everything under `data/` and `ladder_replays/` is regenerable from the zips via
the pipeline above, so it stays out of git.

---

## Labels (what the model predicts)

Per player, per turn, labeled by final game outcome:

- **2p**: raw reward — `+1` win, `-1` loss.
- **4p**: `+1` sole winner, `-1` if someone else won outright, `0` for a tie at
  the top. The trainer binarizes with `y > 0`, so the effective target is
  **P(sole win)** — `0` and `-1` both fall in the negative class. (4p win-rate
  base-rate is ~25%, so 4p sign-accuracy is not comparable to 2p's.)
