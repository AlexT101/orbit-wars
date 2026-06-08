"""Quick action stats for a replay JSON. Useful to gauge bot quality without
watching every step:
  - tiny launches (< 5 ships) = bad
  - 100% noop turns = bad
  - few launches per non-noop turn = good
  - reasonable ship counts = good
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    args = ap.parse_args()

    g = json.loads(Path(args.replay).read_text())
    steps = g.get("steps") or []
    rewards = g.get("rewards") or []
    print(f"replay: {args.replay}  steps={len(steps)}  rewards={rewards}")

    for pid in (0, 1):
        n_launches = []
        ship_counts = []
        per_src = {}  # how many turns each src launched
        for t, step in enumerate(steps):
            if not step or pid >= len(step):
                continue
            a = step[pid].get("action") or []
            n_launches.append(len(a))
            for act in a:
                try:
                    src = int(act[0])
                    ships = int(act[2])
                except Exception:
                    continue
                ship_counts.append(ships)
                per_src[src] = per_src.get(src, 0) + 1
        nz = [c for c in n_launches if c > 0]
        total = sum(n_launches)
        ratio = len(nz) / max(1, len(n_launches))
        print(f"\n--- P{pid} {'(WIN)' if rewards and rewards[pid] > 0 else '(LOSS)' if rewards and rewards[pid] < 0 else ''} ---")
        print(f"  turns_with_action: {len(nz)}/{len(n_launches)} ({ratio*100:.0f}%)")
        print(f"  total_launches: {total}")
        if nz:
            print(f"  per_active_turn: avg={statistics.mean(nz):.1f} max={max(nz)}")
        if ship_counts:
            ss = sorted(ship_counts)
            tiny = sum(1 for s in ship_counts if s < 5)
            print(f"  ships_per_launch: min={min(ship_counts)} med={ss[len(ss)//2]} "
                  f"avg={statistics.mean(ship_counts):.1f} max={max(ship_counts)}  "
                  f"tiny(<5)={tiny} ({tiny/len(ship_counts)*100:.0f}%)")
            print(f"  unique sources launched from: {len(per_src)}")
        else:
            print(f"  no launches recorded")


if __name__ == "__main__":
    main()
