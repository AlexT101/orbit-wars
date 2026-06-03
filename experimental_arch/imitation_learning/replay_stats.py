from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path


REPLAY_DIR = Path("/home/sunrise/orbitwars/pantheow/experimental_arch/replays")
ISAIAH = "Isaiah @ Tufa Labs"
TOP_OPPONENTS = 30


def meta(path: Path) -> tuple[list[str], list[float]] | None:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        text = f.read(262_144)
    text = text[: text.find('"steps"')] if '"steps"' in text else text
    names = re.findall(r'"Name"\s*:\s*"([^"]*)"', text)
    rewards = re.search(r'"rewards"\s*:\s*\[([^\]]*)\]', text)
    if not rewards:
        return None
    return names, [float(x) for x in rewards.group(1).split(",") if x.strip()]


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a "


def main() -> int:
    paths = sorted(REPLAY_DIR.rglob("*.json"))
    by_players: Counter[int] = Counter()
    isaiah_by_players: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    opponent: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    skipped = 0

    for path in paths:
        item = meta(path)
        if item is None:
            skipped += 1
            continue
        names, rewards = item
        n = len(names)
        by_players[n] += 1
        if ISAIAH not in names or len(rewards) != n:
            continue
        i = names.index(ISAIAH)
        win = int(rewards[i] == max(rewards) and rewards.count(rewards[i]) == 1)
        isaiah_by_players[n][0] += win
        isaiah_by_players[n][1] += 1
        for name in names:
            if name != ISAIAH:
                opponent[name][0] += win
                opponent[name][1] += 1

    total = sum(by_players.values())
    print(f"replays: {total}  skipped: {skipped}  dir: {REPLAY_DIR}")
    print("\nplayer counts:")
    for n, count in sorted(by_players.items()):
        print(f"  {n}p: {count:5d} {pct(count, total)}")

    print("\nIsaiah winrate by player count:")
    for n, (wins, games) in sorted(isaiah_by_players.items()):
        print(f"  {n}p: {wins:5d}/{games:<5d} {pct(wins, games)}")

    print(f"\nIsaiah winrate by opponent, top {TOP_OPPONENTS} by games:")
    rows = sorted(opponent.items(), key=lambda kv: (-kv[1][1], kv[0]))[:TOP_OPPONENTS]
    for name, (wins, games) in rows:
        print(f"  {wins:5d}/{games:<5d} {pct(wins, games)}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
