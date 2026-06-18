"""Run the Aphrodite 2p player-quality weighting sweep.

This experiment deliberately does not re-extract ladder replay features. It
derives smaller top-N/day datasets from the existing top-20/day NPZs, trains six
2p XGBoost weights, evaluates them against producer_v2 on the same seed range,
and summarizes paired results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
APHRODITE = HERE.parent
REPO = HERE.parents[3]
PY = REPO / "venv" / "Scripts" / "python.exe"

WORK = HERE / "data" / "2p" / "_quality_sweep"
BASE_WORK = HERE / "data" / "2p" / "_ladder_work"
WEIGHTS = HERE / "weights"
LOGS = WORK / "logs"
COMBINED_TOP20 = BASE_WORK / "combined.npz"

SEED_RE = re.compile(
    r"seed=\s*(?P<seed>\d+)\s+steps=\s*(?P<steps>\d+)\s+"
    r"r=\((?P<r0>[^,]+),(?P<r1>[^)]+)\)\s+->\s+(?P<winner>\S+)\s+"
    r"ms=\((?P<ms0>[^,]+),(?P<ms1>[^)]+)\)"
)


@dataclass(frozen=True)
class Variant:
    name: str
    top_n: int
    quality_floor: float
    quality_weight: bool = True
    quality_shape: str = "decay"
    recency_halflife: float = 7.0
    drop_decided: bool = False
    decisiveness_weight: bool = False

    @property
    def model_path(self) -> Path:
        return WEIGHTS / f"xgb_2p_qsweep_{self.name}.json"

    @property
    def combined_path(self) -> Path:
        if self.top_n == 20:
            return COMBINED_TOP20
        return WORK / f"top{self.top_n}" / "combined.npz"


VARIANTS = [
    Variant("control_top20_floor005", top_n=20, quality_floor=0.05),
    Variant("top10_floor005", top_n=10, quality_floor=0.05),
    Variant("top5_floor005", top_n=5, quality_floor=0.05),
    Variant("top20_floor001", top_n=20, quality_floor=0.01),
    Variant("top20_floor020", top_n=20, quality_floor=0.20),
    Variant(
        "top20_floor005_dropdec",
        top_n=20,
        quality_floor=0.05,
        drop_decided=True,
        decisiveness_weight=True,
    ),
]

ROUND2_VARIANTS = [
    Variant("r2_top20_floor010", top_n=20, quality_floor=0.10),
    Variant("r2_top20_floor050", top_n=20, quality_floor=0.50),
    Variant("r2_top20_noquality", top_n=20, quality_floor=1.0, quality_weight=False),
    Variant("r2_top20_linear_floor005", top_n=20, quality_floor=0.05, quality_shape="linear"),
    Variant(
        "r2_top20_floor020_dropdec",
        top_n=20,
        quality_floor=0.20,
        drop_decided=True,
        decisiveness_weight=True,
    ),
    Variant("r2_top20_floor020_droponly", top_n=20, quality_floor=0.20, drop_decided=True),
    Variant(
        "r2_top20_floor020_decweight",
        top_n=20,
        quality_floor=0.20,
        decisiveness_weight=True,
    ),
    Variant("r2_top15_floor005", top_n=15, quality_floor=0.05),
    Variant("r2_top15_floor020", top_n=15, quality_floor=0.20),
    Variant("r2_top10_floor020", top_n=10, quality_floor=0.20),
]

ROUND3_VARIANTS = [
    Variant("r3_top20_noquality_dropdec", top_n=20, quality_floor=1.0, quality_weight=False, drop_decided=True),
    Variant(
        "r3_top20_noquality_decweight",
        top_n=20,
        quality_floor=1.0,
        quality_weight=False,
        decisiveness_weight=True,
    ),
    Variant(
        "r3_top20_noquality_dropdec_decweight",
        top_n=20,
        quality_floor=1.0,
        quality_weight=False,
        drop_decided=True,
        decisiveness_weight=True,
    ),
    Variant("r3_top20_floor050_dropdec", top_n=20, quality_floor=0.50, drop_decided=True),
    Variant("r3_top20_floor050_decweight", top_n=20, quality_floor=0.50, decisiveness_weight=True),
    Variant(
        "r3_top20_floor050_dropdec_decweight",
        top_n=20,
        quality_floor=0.50,
        drop_decided=True,
        decisiveness_weight=True,
    ),
    Variant("r3_top20_floor035", top_n=20, quality_floor=0.35),
    Variant("r3_top20_floor080", top_n=20, quality_floor=0.80),
    Variant("r3_top20_noquality_hl3", top_n=20, quality_floor=1.0, quality_weight=False, recency_halflife=3.0),
    Variant("r3_top20_noquality_hl14", top_n=20, quality_floor=1.0, quality_weight=False, recency_halflife=14.0),
]

ALL_VARIANTS = VARIANTS + ROUND2_VARIANTS + ROUND3_VARIANTS
ROUND3_BENCHMARK_VARIANTS = [
    VARIANTS[0],
    VARIANTS[5],
    ROUND2_VARIANTS[1],
    ROUND2_VARIANTS[2],
    *ROUND3_VARIANTS,
]


def ensure_dirs() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)


def run_logged(cmd: list[str | Path], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    merged_env = dict(os.environ, PYTHONUTF8="1")
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=REPO,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, [str(c) for c in cmd])


def source_days() -> list[tuple[str, Path, Path]]:
    days: list[tuple[str, Path, Path]] = []
    for npz in sorted(BASE_WORK.glob("gated_replays_6_*.npz")):
        m = re.search(r"(\d+_\d+)", npz.stem)
        if not m:
            continue
        day = m.group(1)
        if day < "6_08" or day > "6_14":
            continue
        top_json = BASE_WORK / f"topn_replays_{day}.json"
        if not top_json.is_file():
            raise FileNotFoundError(f"missing top-N JSON for {day}: {top_json}")
        days.append((day, npz, top_json))
    if len(days) != 7:
        raise SystemExit(f"expected 7 top-20 daily NPZs for 6_08..6_14, found {len(days)}")
    return days


def filter_daily_topn(src: Path, top_json: Path, top_n: int, out: Path, force: bool) -> None:
    if out.is_file() and not force:
        print(f"[skip] {out.relative_to(REPO)} exists")
        return
    keep = set(json.loads(top_json.read_text(encoding="utf-8"))[:top_n])
    d = np.load(src, allow_pickle=False)
    feat_key = "summary_v3" if "summary_v3" in d.files else "summary_v2"
    meta = d["meta"].astype(np.int32)
    game_names = d["game_names"]
    row_names = np.array(
        [str(game_names[int(gid), int(slot)]) for gid, _step, slot, _players in meta],
        dtype=object,
    )
    mask = np.fromiter((name in keep for name in row_names), dtype=bool, count=len(row_names))
    if not bool(mask.any()):
        raise SystemExit(f"{src} produced no rows for top-{top_n}")

    extra = {}
    for key in ("game_names", "game_rewards", "game_files"):
        if key in d.files:
            extra[key] = d[key]
    for key in ("is_strong", "win_rate", "decisiveness_aux"):
        if key in d.files:
            extra[key] = d[key][mask]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{
            feat_key: d[feat_key][mask].astype(np.float32),
            "labels": d["labels"][mask].astype(np.float32),
            "meta": meta[mask],
            **extra,
        },
    )
    print(
        f"wrote {out.relative_to(REPO)} rows={int(mask.sum()):,}/"
        f"{len(mask):,} keep_players={len(keep)}"
    )


def prepare(force: bool) -> None:
    ensure_dirs()
    if not COMBINED_TOP20.is_file():
        raise FileNotFoundError(f"missing top-20 combined dataset: {COMBINED_TOP20}")
    for top_n in sorted({v.top_n for v in ALL_VARIANTS if v.top_n != 20}):
        out_dir = WORK / f"top{top_n}"
        daily_outs: list[Path] = []
        for day, src, top_json in source_days():
            out = out_dir / f"gated_replays_{day}_top{top_n}.npz"
            filter_daily_topn(src, top_json, top_n, out, force)
            daily_outs.append(out)
        combined = out_dir / "combined.npz"
        if combined.is_file() and not force:
            print(f"[skip] {combined.relative_to(REPO)} exists")
            continue
        run_logged(
            [PY, HERE / "combine_npz.py", "--out", combined, *daily_outs],
            LOGS / f"combine_top{top_n}.log",
        )


def train(force: bool, variants: list[Variant]) -> None:
    ensure_dirs()
    for variant in variants:
        if variant.model_path.is_file() and not force:
            print(f"[skip] {variant.model_path.relative_to(REPO)} exists")
            continue
        cmd: list[str | Path] = [
            PY,
            HERE / "train_xgb.py",
            "--data",
            variant.combined_path,
            "--no-filter",
            "--recency-halflife",
            f"{variant.recency_halflife:g}",
            "--rounds",
            "2000",
            "--early-stopping",
            "50",
            "--model-out",
            variant.model_path,
        ]
        if variant.quality_weight:
            cmd += [
                "--quality-weight",
                "--quality-metric",
                "rating",
                "--quality-shape",
                variant.quality_shape,
                "--quality-floor",
                f"{variant.quality_floor:g}",
            ]
        if variant.decisiveness_weight:
            cmd.append("--decisiveness-weight")
        if variant.drop_decided:
            cmd.append("--drop-decided")
        run_logged(cmd, LOGS / f"train_{variant.name}.log")


def eval_log_path(
    variant: Variant,
    start_seed: int,
    matches: int,
    tag: str,
    opponent: str = "producer_v2",
) -> Path:
    opponent_part = "" if opponent == "producer_v2" else f"_{opponent}"
    suffix = f"_{tag}" if tag else ""
    if not opponent_part and not tag and start_seed == 20000 and matches == 100:
        return LOGS / f"eval_{variant.name}.log"
    return LOGS / f"eval_{variant.name}{opponent_part}_s{start_seed}_n{matches}{suffix}.log"


def evaluate(
    force: bool,
    start_seed: int,
    matches: int,
    threads: int,
    variants: list[Variant],
    tag: str = "",
    opponent: str = "producer_v2",
) -> None:
    ensure_dirs()
    for variant in variants:
        if not variant.model_path.is_file():
            raise FileNotFoundError(f"missing trained model for {variant.name}: {variant.model_path}")
        log_path = eval_log_path(variant, start_seed, matches, tag, opponent)
        if log_path.is_file() and not force:
            print(f"[skip] {log_path.relative_to(REPO)} exists")
            continue
        weight = str(variant.model_path.resolve())
        env = {
            "APHRODITE_VALUE_NET_PATH": weight,
            "APHRODITE_VALUE_NET_PATH_2P": weight,
            "APHRODITE_BUDGET_MS": "500",
            "APHRODITE_USE_OVERAGE": "0",
        }
        run_logged(
            [
                PY,
                REPO / "run_batched.py",
                "aphrodite",
                opponent,
                str(matches),
                "--start-seed",
                str(start_seed),
                "--threads",
                str(threads),
            ],
            log_path,
            env=env,
        )


def parse_float(text: str) -> float | None:
    text = text.strip()
    if text == "None":
        return None
    return float(text)


def parse_eval_log(path: Path) -> dict[int, dict[str, float | int | str | None]]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SEED_RE.search(line)
        if not m:
            continue
        seed = int(m.group("seed"))
        r0 = parse_float(m.group("r0"))
        r1 = parse_float(m.group("r1"))
        if r0 is None or r1 is None:
            score = 0.5
        elif r0 > r1:
            score = 1.0
        elif r1 > r0:
            score = 0.0
        else:
            score = 0.5
        rows[seed] = {
            "steps": int(m.group("steps")),
            "winner": m.group("winner"),
            "r0": r0,
            "r1": r1,
            "score": score,
            "ms0": float(m.group("ms0")),
            "ms1": float(m.group("ms1")),
        }
    return rows


def sign_test_p_two_sided(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def analyze_logs(
    variants: list[Variant],
    start_seed: int,
    matches: int,
    tag: str,
    out_name: str,
    opponent: str = "producer_v2",
) -> None:
    ensure_dirs()
    parsed = {}
    for variant in variants:
        log_path = eval_log_path(variant, start_seed, matches, tag, opponent)
        if not log_path.is_file():
            raise FileNotFoundError(f"missing eval log: {log_path}")
        parsed[variant.name] = parse_eval_log(log_path)

    control_name = variants[0].name
    control = parsed[control_name]
    summary = []
    for variant in variants:
        rows = parsed[variant.name]
        seeds = sorted(rows)
        wins = sum(1 for seed in seeds if rows[seed]["score"] == 1.0)
        losses = sum(1 for seed in seeds if rows[seed]["score"] == 0.0)
        draws = len(seeds) - wins - losses
        avg_ms = sum(float(rows[seed]["ms0"]) for seed in seeds) / len(seeds)
        avg_steps = sum(int(rows[seed]["steps"]) for seed in seeds) / len(seeds)
        better = worse = same = 0
        net_points = 0.0
        if variant.name != control_name:
            common = sorted(set(seeds) & set(control))
            for seed in common:
                delta = float(rows[seed]["score"]) - float(control[seed]["score"])
                net_points += delta
                if delta > 0:
                    better += 1
                elif delta < 0:
                    worse += 1
                else:
                    same += 1
        p_value = sign_test_p_two_sided(better, worse) if variant.name != control_name else 1.0
        summary.append(
            {
                "variant": variant.name,
                "matches": len(seeds),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "decided_win_rate": (wins / (wins + losses)) if wins + losses else 0.0,
                "avg_ms": avg_ms,
                "avg_steps": avg_steps,
                "better_seeds": better,
                "worse_seeds": worse,
                "same_seeds": same,
                "net_points_vs_control": net_points,
                "sign_p": p_value,
            }
        )

    out = WORK / out_name
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")


def analyze() -> None:
    analyze_logs(VARIANTS, 20000, 100, "", "summary.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "train",
            "train-round2",
            "train-round3",
            "eval",
            "eval-round2",
            "eval-round3",
            "analyze",
            "analyze-round2",
            "analyze-round3",
            "all",
        ),
        nargs="?",
        default="all",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--start-seed", type=int, default=20000)
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--opponent", default="producer_v2")
    args = p.parse_args()

    if args.stage in ("prepare", "all"):
        prepare(args.force)
    if args.stage in ("train", "all"):
        train(args.force, VARIANTS)
    if args.stage == "train-round2":
        train(args.force, ROUND2_VARIANTS)
    if args.stage == "train-round3":
        train(args.force, ROUND3_VARIANTS)
    if args.stage in ("eval", "all"):
        evaluate(args.force, args.start_seed, args.matches, args.threads, VARIANTS, opponent=args.opponent)
    if args.stage == "eval-round2":
        evaluate(
            args.force,
            args.start_seed,
            args.matches,
            args.threads,
            ALL_VARIANTS,
            "round2",
            args.opponent,
        )
    if args.stage == "eval-round3":
        evaluate(
            args.force,
            args.start_seed,
            args.matches,
            args.threads,
            ROUND3_BENCHMARK_VARIANTS,
            "round3",
            args.opponent,
        )
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage == "analyze-round2":
        suffix = "" if args.opponent == "producer_v2" else f"_{args.opponent}"
        analyze_logs(
            ALL_VARIANTS,
            args.start_seed,
            args.matches,
            "round2",
            f"summary_round2{suffix}.json",
            args.opponent,
        )
    if args.stage == "analyze-round3":
        suffix = "" if args.opponent == "producer_v2" else f"_{args.opponent}"
        analyze_logs(
            ROUND3_BENCHMARK_VARIANTS,
            args.start_seed,
            args.matches,
            "round3",
            f"summary_round3{suffix}.json",
            args.opponent,
        )


if __name__ == "__main__":
    main()
