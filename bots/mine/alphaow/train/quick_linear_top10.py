"""Quick standardized linear regression on top-10 players' games.
Reports per-feature weights with names so we can read off rough
importance. No leakage features used — pure summary_v2 (46-d):
current-state aggregates + extrapolated-state aggregates + neutral
aggregates. No tick, no future-outcome features.
"""

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler


# ---- 46-d summary_v2 feature names, by index ----
# Each per-player block is 10 fields for current, 9 fields for extrap
# (extrap omits ships_flying), 8 for neutral.
PLAYER_CUR = [
    "ships_on_planets", "ships_flying", "n_static", "n_orbit", "n_comet",
    "prod_static", "prod_orbit", "prod_comet",
    "n_neutrals_closer", "n_enemies_closer",
]
PLAYER_EXT = [
    "ships_on_planets", "n_static", "n_orbit", "n_comet",
    "prod_static", "prod_orbit", "prod_comet",
    "n_neutrals_closer", "n_enemies_closer",
]
NEUT = [
    "ships", "n_static", "n_orbit", "n_comet",
    "prod_static", "prod_orbit", "prod_comet", "comet_time_total",
]

NAMES = (
    [f"me_cur.{n}" for n in PLAYER_CUR]
    + [f"opp_cur.{n}" for n in PLAYER_CUR]
    + [f"me_ext.{n}" for n in PLAYER_EXT]
    + [f"opp_ext.{n}" for n in PLAYER_EXT]
    + [f"neut.{n}" for n in NEUT]
)
assert len(NAMES) == 46, f"expected 46 feature names, got {len(NAMES)}"


def main():
    print("Loading combined_top10.npz...")
    d = np.load("data/combined_top10.npz")
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    print(f"  X: {X.shape}, y: {y.shape}, y range=[{y.min():.2f},{y.max():.2f}], y mean={y.mean():.3f}")

    # Standardize so weights are directly comparable.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Train/val split for sanity.
    n = len(Xs)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(0.9 * n)
    tr, va = idx[:split], idx[split:]
    Xtr, Xva = Xs[tr], Xs[va]
    ytr, yva = y[tr], y[va]

    print("Fitting Ridge(alpha=1.0)...")
    m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    yhat_tr = m.predict(Xtr)
    yhat_va = m.predict(Xva)
    # Classification-style accuracy if labels are {-1,+1} (sign agreement).
    acc_tr = float(np.mean(np.sign(yhat_tr) == np.sign(ytr)))
    acc_va = float(np.mean(np.sign(yhat_va) == np.sign(yva)))
    # Mean squared error.
    mse_tr = float(np.mean((yhat_tr - ytr) ** 2))
    mse_va = float(np.mean((yhat_va - yva) ** 2))
    print(f"  train acc={acc_tr:.4f}  mse={mse_tr:.4f}")
    print(f"  val   acc={acc_va:.4f}  mse={mse_va:.4f}")
    print(f"  intercept (after standardization): {m.intercept_:+.4f}")
    print()

    coefs = m.coef_  # (46,)
    pairs = sorted(enumerate(coefs), key=lambda kv: abs(kv[1]), reverse=True)

    print("=== Feature weights (standardized; sorted by |coef|) ===")
    print(f"{'idx':>4} {'feature':<32}  {'coef':>10}")
    for i, c in pairs:
        bar = ""
        n_bars = int(round(abs(c) * 40))
        if c > 0:
            bar = "+" * n_bars
        else:
            bar = "-" * n_bars
        print(f"{i:>4} {NAMES[i]:<32}  {c:+10.4f}  {bar}")

    print()
    print("=== Grouped sums of |coef| by block ===")
    groups = [
        ("me_cur (10d)",  range(0, 10)),
        ("opp_cur (10d)", range(10, 20)),
        ("me_ext (9d)",   range(20, 29)),
        ("opp_ext (9d)",  range(29, 38)),
        ("neut (8d)",     range(38, 46)),
    ]
    for name, rng_ in groups:
        s = float(sum(abs(coefs[i]) for i in rng_))
        print(f"  {name:<15} sum|coef|={s:.4f}")


if __name__ == "__main__":
    main()
