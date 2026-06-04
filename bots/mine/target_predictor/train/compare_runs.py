"""Collect val_metrics from every checkpoint in train/weights/ and print a
sorted table. Useful for end-of-sweep summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path,
                    default=Path(__file__).resolve().parent / "weights")
    ap.add_argument("--sort", default="recall8",
                    help="metric to sort by (recall8, recall10, top1, auc, ...)")
    args = ap.parse_args()

    rows = []
    for p in sorted(args.weights_dir.glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        m = ck.get("val_metrics", {})
        if not m:
            continue
        rows.append(dict(
            name=p.stem,
            arch=ck.get("arch", "?"),
            f_planet=ck.get("f_planet", "?"),
            d_model=ck.get("d_model", "?"),
            opening=ck.get("opening_weight_schedule", "?"),
            max_step=ck.get("max_step", "?"),
            selected_by=ck.get("selected_by", "?"),
            acc=m.get("acc", float("nan")),
            auc=m.get("auc", float("nan")),
            top1=m.get("top1", float("nan")),
            r1=m.get("recall1", float("nan")),
            r3=m.get("recall3", float("nan")),
            r5=m.get("recall5", float("nan")),
            r8=m.get("recall8", float("nan")),
            r10=m.get("recall10", float("nan")),
            rows_w_pos=m.get("rows_with_pos", 0),
        ))
    rows.sort(key=lambda r: -r.get(args.sort, -1))

    name_w = max((len(r["name"]) for r in rows), default=20)
    print(f"  sorted by {args.sort}")
    print(f"  {'name':{name_w}}  {'arch':10s}  acc    auc    "
          f"r@1    r@3    r@5    r@8    r@10   rows")
    for r in rows:
        print(f"  {r['name']:{name_w}}  {r['arch']:10s}  "
              f"{r['acc']:.3f}  {r['auc']:.3f}  "
              f"{r['r1']:.3f}  {r['r3']:.3f}  {r['r5']:.3f}  "
              f"{r['r8']:.3f}  {r['r10']:.3f}  {r['rows_w_pos']}")


if __name__ == "__main__":
    raise SystemExit(main())
