# target_predictor

Per-turn, per-planet binary model: `P(planet p is one of my targets this turn)`.
"Target" = my side dispatched ≥1 ship this turn whose ballistic-predicted
destination is `p`.

Goal: cheap inference for downstream uses (move-gen prior, candidate filter).
One forward pass per turn must emit `N` logits, so the deployed model is a
**set encoder** over planets, not a per-row XGBoost.

## Folder layout

```
target_predictor/
├── README.md
└── train/
    ├── manifest.csv             # Kaggle daily-episode dataset slugs (copied from prometheus)
    ├── download_kaggle.py       # manifest → kagglehub → /tmp/orbit_days/<slug>.zip
    ├── build_dataset.py         # zip → (turns × N_max × F) NPZ with per-planet labels
    ├── set_net.py               # DeepSets + Transformer backbones + train loop
    ├── predict.py               # load a checkpoint, run one forward pass (downstream API)
    ├── feature_importance.py    # per-feature permutation importance on val split
    ├── model_dashboard.py       # copied verbatim — gain/perm/calibration/HTML dashboard
    ├── dashboard.html           # HTML template paired with model_dashboard
    ├── engineered_features.py   # copied; legacy 46-d schema (kept for diagnostic XGB)
    ├── xgb_tune.py              # diagnostic XGB (row-wise) using model_dashboard
    ├── train_gbm.py             # alt XGB/LGBM driver
    ├── topn_experiments.py      # skill-band ablation
    ├── data/                    # generated NPZs (gitignored)
    ├── weights/                 # trained models (gitignored)
    └── requirements.txt
```

## Pipeline

```
manifest.csv ──► download_kaggle.py ──► /tmp/orbit_days/*.zip
                                              │
                                              ▼
                                      build_dataset.py         (2p filter inside)
                                              │
                                              ▼
                         data/targets.npz  (planet_feats, globals, mask, labels, meta)
                                              │
                                              ▼
                                         set_net.py
                                  (DeepSets baseline / Transformer primary)
                                              │
                                              ▼
                                      weights/setnet_latest.pt
```

## Setup

```
python3 -m pip install -r train/requirements.txt
python3 train/download_kaggle.py --limit-days 4 --out /tmp/orbit_days
```

`/tmp/orbit_days/` already has 4 days cached locally (~5.5GB,
2026-05-27..2026-05-30).

## Build the dataset

```
python3 train/build_dataset.py --max-games 2000 --workers 8 --out train/data/targets_2k.npz
```

Throughput on the local 4-day cache: ~0.66 s/game with 6 workers; ~44% of games
are 2p (4p dropped). Expect ~5MB NPZ per 1000 kept games.

Output NPZ keys:

| key | dtype | shape | notes |
|---|---|---|---|
| `planet_feats` | f32 | `(N_rows, N_max=30, F_planet=38)` | per-planet features |
| `globals` | f32 | `(N_rows, F_global=12)` | turn-level scalars |
| `mask` | bool | `(N_rows, N_max)` | True = real planet |
| `labels` | f32 | `(N_rows, N_max)` | 1.0 = my side targeted this planet at this turn |
| `meta` | i32 | `(N_rows, 4)` | (game_int_id, step, player, n_real_planets) |
| `planet_ids` | i32 | `(N_rows, N_max)` | engine planet ids (debug only) |
| `feat_names` / `global_names` | object | (F_planet,) / (F_global,) | human-readable names |

## Train

```
python3 train/set_net.py --data train/data/targets_2k.npz --arch transformer --epochs 8
python3 train/set_net.py --data train/data/targets_2k.npz --arch deepsets   --epochs 8
```

Game-level 12% val split (seed 42), masked BCE with auto pos-weight (~25× since
positives are ~4%), AdamW. Reports per-epoch:

- `acc`, `auc` over all real planet slots (val)
- `top1`, `top2` over rows that have ≥1 positive (val) — i.e., "among real
  planets in a turn, was the actual target in the top-K of my logits?"

Saves best (by val AUC) to `weights/setnet_latest.pt` including the normalization
stats fit on the train split, so inference is reproducible.

### Results — 2000-game build (1052 kept 2p games, 406k rows, 8 epochs)

| arch | params | val acc | val AUC | top1 | top2 | train epoch |
|---|---|---|---|---|---|---|
| Transformer | 70K | 0.831 | **0.939** | **0.519** | **0.710** | ~65 s |
| DeepSets    | 24K | 0.798 | 0.926 | 0.447 | 0.628 | ~8 s |

Chance-level top1 over ~20 planets is ~5%; both well above. Transformer beats
DeepSets by 7 pts top-1 / 8 pts top-2 — the cross-planet attention is paying
for itself. Transformer AUC was still climbing at epoch 8, more epochs should
help marginally.

### Feature importance — top 10 (transformer, permutation ΔAUC on 15k val rows)

| rank | feature | ΔAUC |
|---|---|---|
| 1 | `in_mine_count` | +0.060 |
| 2 | `min_eta_from_me` | +0.043 |
| 3 | `ships_log1p` | +0.037 |
| 4 | `is_mine` | +0.022 |
| 5 | `rank_min_eta_from_me` | +0.017 |
| 6 | `dist_nearest_enemy` | +0.016 |
| 7 | `planet_radius` | +0.013 |
| 8 | `is_neutral` | +0.012 |
| 9 | `dist_nearest_my` | +0.012 |
| 10 | `cos_theta` | +0.011 |

Sensible — top features cluster around "what is this planet, how fast can I
reach it, am I already campaigning toward it." Bottom features (~0 ΔAUC):
`omega`, `in_enemy_count`, `production`, `rank_production`,
`is_closest_enemy_to_me`. Production scoring near zero is the most surprising
result — likely correlated away by `ships`/`planet_radius`/owner one-hots.

### Inference

```
python3 train/predict.py --ckpt train/weights/transformer_2k.pt \
                         --npz train/data/targets_2k.npz --rows 55,86,97
```

The Python `predict` helper in `predict.py` is the downstream API:
```python
from predict import load_checkpoint, predict
ckpt = load_checkpoint("weights/transformer_2k.pt")
probs = predict(ckpt, planet_feats, globals_)  # (N_real,) sigmoid
```

## Model architecture (deployed)

**Primary: PlanetTransformer.** 2 encoder layers × 64-dim × 4 heads, FF 128,
GELU, dropout 0.1. The 12-d global vector is projected to a CLS-style token
prepended to the planet tokens; per-planet sigmoid head reads the post-encoder
planet rows. Padding handled via `src_key_padding_mask`.

**Floor: DeepSets.** Per-planet MLP → masked mean + max pool → concat with
global → broadcast back → per-planet head. No cross-planet attention.

Both produce N logits in a single forward pass for runtime efficiency.

## Feature schema (decided + implemented)

Per-planet (`F_planet = 38`):

| Group | Dims | Implemented |
|---|---|---|
| Owner one-hot (mine/neutral/enemy) | 3 | ✓ |
| Stock (ships, log1p, production) | 3 | ✓ |
| Orbit/geom (r, ω, cosθ, sinθ, planet_radius) | 5 | ✓ |
| Position at t, t+10, t+25 | 6 | ✓ (uses `planet_pos_at`) |
| Inbound fleets per side {mine, enemy} × {count, ships, min_eta, mean_eta} | 8 | ✓ (uses `predict_fleet_collision`) |
| Reachability from me (min_eta, surplus_at_src, arrivable_by_25/50) | 4 | ✓ (straight-line / fleet_speed proxy) |
| Frontier/geom (nearest my/enemy, two `is_closest` bits) | 4 | ✓ |
| Listwise rank within turn (prod, ships, eta, dist) | 4 | ✓ |
| Tempo (turns_since_owner_change) | 1 | ✓ |

Global (`F_global = 12`): turn_num, turn_norm, my/enemy ships/prod/planet_count,
economy_diff, phase one-hot (opening<30 / mid<120 / end).

**Deferred to v2** (would need apollo-equivalent helpers in Python or a Rust
extractor): capture-economics cost/hold via `min_ships_to_own_by` and
`reinforcement_needed_to_hold_until`; projected stock via full `PlanetTimeline`.
MVP uses straight-line ETA and direct counts as proxies.

## Scope

- **2p only.** Filter `len(rewards) == 2` per `kaggle_rebuild_v2.py` precedent.
- **Both perspectives kept.** Each 2p game contributes 2 rows per step — one
  with player=0 as "my side", one with player=1. Label always reflects what
  *that* player did at that step.
- **No specific bot imitation.** Train on whoever played. A skill filter
  (similar to `topn_experiments.py`) can be applied at training time if
  noisy labels hurt val AUC.

## Notes from the copy-over

The copied `model_dashboard.py` / `engineered_features.py` / `xgb_tune.py` are
the legacy 46-d `summary_v2` schema. They still run as-is against any matching
NPZ. They will be re-wired to the new per-planet schema after the set-net
baseline is solid; until then, treat them as the inherited diagnostic toolkit
for the optional flattened-XGB sanity check.
