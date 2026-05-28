"""Play one game (TinyPolicy as player 0 vs a /bots opponent) and write the
official kaggle animation to an HTML file you can open in a browser.

How it stays honest: we drive env_engine AND a kaggle env in lockstep with the
*same* actions each step (the exact scheme validate_against_kaggle.py proves
bit-identical), then render the kaggle env. So the picture you see is the real
game the training loop played — not a reimplementation.

Run (from experimental_arch/):
    python examples/visualize_game.py                       # seed 1 vs nearest-sniper
    python examples/visualize_game.py --opponent random --seed 3
    python examples/visualize_game.py --train 30            # REINFORCE-train first, then show

Writes to experimental_arch/replays/ and (on WSL) prints a file:// link you can
paste into a Windows browser.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import subprocess
from itertools import cycle
from pathlib import Path

import torch

from orbit_wars_engine import OrbitWarsEngine

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

from train_loop_example import BotOpponent, TinyPolicy, act, run_episode

PLAYERS = 2
MAX_STEPS = 500
# Default output: experimental_arch/replays/ (alongside the code, not /tmp).
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "replays" / "orbit_wars_game.html"


def windows_link(path: Path) -> str | None:
    """A file:// URL for the WSL path, openable from a Windows browser. Returns
    None if not on WSL / `wslpath` is unavailable."""
    if not shutil.which("wslpath"):
        return None
    try:
        win = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if win.startswith("\\\\"):
        # UNC (typical on WSL): \\wsl.localhost\Distro\... -> file://wsl.localhost/Distro/...
        return "file://" + win[2:].replace("\\", "/")
    # Drive path: C:\Users\me\x.html -> file:///C:/Users/me/x.html
    return "file:///" + win.replace("\\", "/")


def final_scores(state: dict, n: int = PLAYERS) -> list[int]:
    sc = [0] * n
    for p in state["planets"]:
        if 0 <= int(p[1]) < n:
            sc[int(p[1])] += int(p[5])
    for f in state["fleets"]:
        if 0 <= int(f[1]) < n:
            sc[int(f[1])] += int(f[6])
    return sc


def play_and_render(policy: TinyPolicy, opponent: BotOpponent, seed: int, out: Path) -> None:
    opponent.reset()
    engine = OrbitWarsEngine(num_players=PLAYERS)
    eng_obs = engine.reset(seed=seed)["observations"]
    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    kenv.reset(PLAYERS)

    steps = 0
    for _ in range(MAX_STEPS):
        if kenv.done:
            break
        # Moves are computed from env_engine's full per-player observations
        # (kaggle's non-player-0 obs is sparse), then applied to both engines.
        acts = [act(policy, eng_obs[0])[0], opponent.act(eng_obs[1])]
        kenv.step(acts)
        res = engine.step(acts)
        eng_obs = res["observations"]
        steps += 1
        if res["done"]:
            break

    sc = final_scores(engine.get_state())
    total = sum(sc)
    share0 = sc[0] / total if total else 0.0
    if sc[0] > sc[1]:
        verdict = "player 0 (TinyPolicy) WINS"
    elif sc[1] > sc[0]:
        verdict = "player 1 (opponent) WINS"
    else:
        verdict = "TIE"
    print(f"game over after {steps} steps: {verdict}")
    print(f"  final ships  p0={sc[0]}  p1={sc[1]}  (p0 ships-share = {share0:.2f})")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(kenv.render(mode="html"))
    print(f"  animation written to {out}")
    link = windows_link(out)
    if link:
        print(f"  open on Windows:  {link}")
        print(f"  or run:           explorer.exe '{out}'")
    else:
        print("  open it in a browser to watch.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="nearest-sniper", help="bot name under pantheow/bots/")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train", type=int, default=0,
                    help="REINFORCE episodes to train before recording (0 = fresh policy)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    policy = TinyPolicy()

    if args.train > 0:
        optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        engine = OrbitWarsEngine(num_players=PLAYERS)
        opp_cycle = cycle([BotOpponent(args.opponent)])
        for ep in range(args.train):
            log_probs, ret = run_episode(engine, policy, next(opp_cycle), seed=args.seed + ep)
            if log_probs:
                loss = -(ret * torch.stack(log_probs).sum())
                optim.zero_grad(); loss.backward(); optim.step()
        print(f"trained {args.train} episodes vs {args.opponent}")

    play_and_render(policy, BotOpponent(args.opponent), args.seed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
