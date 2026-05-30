from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from orbit_wars_rl.visualization import write_training_report


ROOT = Path(__file__).resolve().parents[1]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def log(message: str) -> None:
    print(message, flush=True)

START_BOTS = ["random", "nearest", "baselines/starter"]
DEFAULT_IGNORE = {"mine/apollo_backup"}

EMBEDDED_RATINGS = {
    "baselines/random": 227.51537502907,
    "baselines/nearest-sniper": 357.33075525380195,
    "baselines/starter": 375.25215292632345,
    "external/sigmaborov-reinforce": 546.1966441756399,
    "external/peaking-bot": 569.9010758200627,
    "external/yuriygreben-architect": 624.6160663153348,
    "external/structured-v4": 626.9749404966219,
    "external/pilkwang-structured": 631.7687496885535,
    "external/ykhnkf-distance-prioritized": 632.3453479196331,
    "external/smart-baseline": 634.5315638870542,
    "external/heuristic": 669.720096986768,
    "external/obnext": 672.5188707544976,
    "external/sim-search": 687.5579747625959,
    "external/marco-dg": 709.8857192554057,
    "external/owproto": 711.1075754816321,
    "external/tamrazov-starwars": 727.1547437737192,
    "external/ppo": 752.2565312541658,
    "external/hellburner": 758.1596660415727,
    "mine/bruteforcer": 789.1713630307494,
    "mine/apollo_bf": 830.6435778648433,
    "mine/apollo": 835.6250803027045,
}


@dataclass(frozen=True)
class BotRating:
    name: str
    source_name: str
    mu: float


@dataclass(frozen=True)
class Phase:
    index: int
    name: str
    train_opponents: list[str]
    gate_opponents: list[str]
    reward_mode: str
    threshold: float


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def normalize_name(name: str) -> str:
    if name == "baselines/random":
        return "random"
    if name == "baselines/nearest-sniper":
        return "nearest"
    return name


def names_csv(names: list[str]) -> str:
    return ",".join(names)


def read_ratings(path: Path | None, ignored: set[str]) -> list[BotRating]:
    raw: dict[str, float] = dict(EMBEDDED_RATINGS)
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = {}
        for bot_name, modes in data.get("ratings", {}).items():
            if bot_name in ignored:
                continue
            rating = modes.get("2p")
            if rating is None:
                continue
            raw[bot_name] = float(rating["mu"])

    bots = [
        BotRating(normalize_name(source_name), source_name, mu)
        for source_name, mu in raw.items()
        if source_name not in ignored
    ]
    by_name = {bot.name: bot for bot in bots}
    ordered: list[BotRating] = []
    for name in START_BOTS:
        bot = by_name.pop(name, None)
        if bot is not None:
            ordered.append(bot)
    ordered.extend(sorted(by_name.values(), key=lambda bot: bot.mu))
    return ordered


def make_phases(
    ladder: list[BotRating],
    bots_per_phase: int,
    carry: int,
    start_threshold: float,
    end_threshold: float,
    fixed_threshold: float | None = None,
) -> list[Phase]:
    if len(ladder) < len(START_BOTS):
        raise ValueError("ladder must contain at least random, nearest, and baselines/starter")

    phase_specs: list[tuple[str, list[BotRating], list[BotRating], str]] = [
        (
            "starter",
            ladder[: len(START_BOTS)],
            ladder[: len(START_BOTS)],
            "shaped",
        )
    ]
    phase_index = 1
    start = len(START_BOTS)
    while start < len(ladder):
        end = min(len(ladder), start + bots_per_phase)
        carry_start = max(0, start - carry)
        gates = ladder[start:end]
        train = ladder[carry_start:end]
        if phase_index <= 2:
            reward_mode = "shaped"
        elif phase_index <= 4:
            reward_mode = "score_delta"
        else:
            reward_mode = "terminal"
        phase_specs.append(
            (
                f"rating_{int(gates[0].mu)}_{int(gates[-1].mu)}",
                train,
                gates,
                reward_mode,
            )
        )
        phase_index += 1
        start = end

    phases: list[Phase] = []
    denom = max(1, len(phase_specs) - 1)
    for idx, (name, train, gates, reward_mode) in enumerate(phase_specs):
        threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else start_threshold + (end_threshold - start_threshold) * (idx / denom)
        )
        phases.append(
            Phase(
                index=idx,
                name=name,
                train_opponents=[bot.name for bot in train],
                gate_opponents=[bot.name for bot in gates],
                reward_mode=reward_mode,
                threshold=round(float(threshold), 4),
            )
        )
    return phases


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    path = checkpoint_dir / "latest.pt"
    return path if path.exists() else None


def latest_metric_step(checkpoint_dir: Path) -> int:
    path = checkpoint_dir / "metrics.jsonl"
    if not path.exists():
        return 0
    last_step = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_step = max(last_step, int(row.get("step", 0) or 0))
    return last_step


def run_command(cmd: list[str], dry_run: bool) -> subprocess.CompletedProcess[str]:
    log(f"{c('run', DIM)} {' '.join(cmd)}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=False)


def parse_json_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    preview = stdout[-1200:] if stdout else "<empty stdout>"
    raise RuntimeError(f"could not parse eval JSON result; stdout tail:\n{preview}")


def log_phase(phase: Phase, state: dict, total_budget_steps: int) -> None:
    progress = int(state.get("steps_requested", 0) or 0) / max(1, total_budget_steps)
    train = ", ".join(phase.train_opponents)
    gate = ", ".join(phase.gate_opponents)
    log(
        f"\n{c('phase', BOLD + CYAN)} {phase.index} {c(phase.name, MAGENTA)} "
        f"threshold={c(f'{phase.threshold:.1%}', YELLOW)} "
        f"budget={progress:.1%} ({state.get('steps_requested', 0)}/{total_budget_steps})"
    )
    log(f"  {c('train', BLUE)} {train}")
    log(f"  {c('gate ', BLUE)} {gate}")


def log_gate(label: str, phase: Phase, gate: dict) -> None:
    passed = bool(gate.get("passed"))
    status = c("PASS", GREEN + BOLD) if passed else c("WAIT", YELLOW + BOLD)
    log(
        f"{c(label, BOLD + MAGENTA)} phase={phase.index} {phase.name} "
        f"{status} min={float(gate.get('min_win_rate', 0.0)):.1%} "
        f"mean={float(gate.get('mean_win_rate', 0.0)):.1%} "
        f"target={float(gate.get('threshold', phase.threshold)):.1%}"
    )
    for name, result in sorted((gate.get("opponents") or {}).items()):
        wr = float(result.get("win_rate", 0.0) or 0.0)
        color = GREEN if wr >= phase.threshold else (YELLOW if wr >= 0.5 else RED)
        log(
            f"  {c(name, BLUE):<32} "
            f"wr={c(f'{wr:5.1%}', color)} "
            f"wins={int(result.get('wins', 0)):>2}/{int(result.get('games', 0)):<2} "
            f"losses={int(result.get('losses', 0)):>2} draws={int(result.get('draws', 0)):>2}"
        )


def train_chunk(args: argparse.Namespace, phase: Phase, checkpoint_dir: Path) -> None:
    opponents = list(phase.train_opponents)
    if args.use_snapshots and phase.index >= args.snapshot_start_phase:
        opponents.append("snapshot_sample")

    cmd = [
        sys.executable,
        str(ROOT / "rl_orbit_wars" / "train.py"),
        "--total-steps",
        str(args.chunk_steps),
        "--rollout-steps",
        str(args.rollout_steps),
        "--minibatch-size",
        str(args.minibatch_size),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--lr-schedule",
        args.lr_schedule,
        "--lr-warmup-steps",
        str(args.lr_warmup_steps),
        "--entropy-coef",
        str(args.entropy_coef),
        "--entropy-coef-final",
        str(args.entropy_coef_final),
        "--model",
        args.model,
        "--hidden",
        str(args.hidden),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--opponent",
        names_csv(opponents),
        "--reward-mode",
        phase.reward_mode,
        "--eval-every-updates",
        str(args.inner_eval_every_updates),
        "--eval-games",
        str(args.inner_eval_games),
        "--eval-opponents",
        names_csv(phase.gate_opponents),
        "--seed",
        str(args.seed + phase.index * 10_000),
        "--device",
        args.device,
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.use_snapshots and phase.index >= args.snapshot_start_phase:
        cmd.extend(["--snapshot-every-updates", str(args.snapshot_every_updates)])
        cmd.extend(["--snapshot-pool-size", str(args.snapshot_pool_size)])

    resume = latest_checkpoint(checkpoint_dir)
    if resume is not None:
        cmd.extend(["--resume-checkpoint", str(resume)])
    elif args.init_checkpoint:
        cmd.extend(["--init-checkpoint", args.init_checkpoint])

    run_command(cmd, args.dry_run)


def maybe_pretrain(args: argparse.Namespace) -> None:
    if not args.init_checkpoint:
        return
    out = Path(args.init_checkpoint)
    if out.exists():
        return
    if not args.pretrain_if_missing:
        raise FileNotFoundError(
            f"{out} does not exist. Run pretrain first or pass --pretrain-if-missing."
        )
    cmd = [
        sys.executable,
        str(ROOT / "rl_orbit_wars" / "pretrain_bc.py"),
        "--teacher",
        args.pretrain_teacher,
        "--samples",
        str(args.pretrain_samples),
        "--epochs",
        str(args.pretrain_epochs),
        "--batch-size",
        str(args.pretrain_batch_size),
        "--learning-rate",
        str(args.pretrain_learning_rate),
        "--model",
        args.model,
        "--hidden",
        str(args.hidden),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--opponents",
        args.pretrain_opponents,
        "--max-noop-fraction",
        str(args.pretrain_max_noop_fraction),
        "--out",
        str(out),
        "--device",
        args.device,
    ]
    run_command(cmd, args.dry_run)


def evaluate_gate(
    args: argparse.Namespace,
    phase: Phase,
    checkpoint_dir: Path,
    checkpoint: Path | None = None,
) -> dict:
    checkpoint = checkpoint or latest_checkpoint(checkpoint_dir)
    if checkpoint is None and args.dry_run:
        checkpoint = checkpoint_dir / "latest.pt"
    if checkpoint is None:
        raise FileNotFoundError(f"missing checkpoint in {checkpoint_dir}")

    per_bot: dict[str, dict] = {}
    win_rates: list[float] = []
    for idx, opponent in enumerate(phase.gate_opponents):
        cmd = [
            sys.executable,
            str(ROOT / "rl_orbit_wars" / "evaluate.py"),
            str(checkpoint),
            "--games",
            str(args.gate_games),
            "--opponent",
            opponent,
            "--seed",
            str(args.seed + 500_000 + phase.index * 10_000 + idx * 1_000),
            "--device",
            args.device,
            "--json",
            "--progress",
            "--workers",
            str(args.eval_workers),
        ]
        log(f"{c('eval', DIM)} {' '.join(cmd)}")
        if args.dry_run:
            result = {"games": args.gate_games, "wins": 0, "losses": 0, "draws": 0, "rewards": []}
        else:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            result = parse_json_result(completed.stdout)
        win_rate = (result["wins"] + 0.5 * result["draws"]) / max(1, result["games"])
        per_bot[opponent] = {**result, "win_rate": win_rate}
        win_rates.append(float(win_rate))

    mean_wr = sum(win_rates) / max(1, len(win_rates))
    min_wr = min(win_rates) if win_rates else 0.0
    passed_value = min_wr if args.gate_mode == "min" else mean_wr
    return {
        "mean_win_rate": mean_wr,
        "min_win_rate": min_wr,
        "passed": passed_value >= phase.threshold,
        "gate_mode": args.gate_mode,
        "threshold": phase.threshold,
        "opponents": per_bot,
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"phase_index": 0, "chunks": 0, "steps_requested": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gate_for(events_path: Path, phase_index: int, steps_requested: int) -> dict | None:
    for event in load_events(events_path):
        if event.get("kind") != "gate":
            continue
        phase = event.get("phase", {})
        state = event.get("state", {})
        if not isinstance(phase, dict) or not isinstance(state, dict):
            continue
        if int(phase.get("index", -1)) == phase_index and int(state.get("steps_requested", -1)) == steps_requested:
            gate = event.get("gate")
            return gate if isinstance(gate, dict) else {}
    return None


def has_gate_for(events_path: Path, phase_index: int, steps_requested: int) -> bool:
    return gate_for(events_path, phase_index, steps_requested) is not None


def load_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    events: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an overnight opponent-rating curriculum for Orbit Wars PPO.")
    parser.add_argument("--ratings-json", default=None, help="Optional bot ratings JSON. Defaults to embedded message (4) ratings.")
    parser.add_argument("--checkpoint-dir", default="rl_orbit_wars/checkpoints_curriculum")
    parser.add_argument("--init-checkpoint", default="rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt")
    parser.add_argument("--total-budget-steps", type=int, default=300_000)
    parser.add_argument("--chunk-steps", type=int, default=10_000)
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Optional fixed gate threshold for every phase. If omitted, thresholds decay from --start-gate-threshold to --end-gate-threshold.",
    )
    parser.add_argument("--start-gate-threshold", type=float, default=0.98)
    parser.add_argument("--end-gate-threshold", type=float, default=0.65)
    parser.add_argument("--gate-mode", choices=["mean", "min"], default="min")
    parser.add_argument("--gate-games", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--bots-per-phase", type=int, default=2)
    parser.add_argument("--carry", type=int, default=3)
    parser.add_argument("--max-runtime-hours", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", choices=["mlp", "entity_transformer"], default="entity_transformer")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--lr-schedule", choices=["linear", "cosine", "constant"], default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=10_000)
    parser.add_argument("--entropy-coef", type=float, default=0.08)
    parser.add_argument("--entropy-coef-final", type=float, default=0.025)
    parser.add_argument(
        "--inner-eval-every-updates",
        type=int,
        default=0,
        help="Trainer-side eval inside each chunk. Default 0 because curriculum gate eval handles promotion.",
    )
    parser.add_argument("--inner-eval-games", type=int, default=8)
    parser.add_argument("--pretrain-if-missing", action="store_true")
    parser.add_argument("--pretrain-teacher", default="hellburner")
    parser.add_argument("--pretrain-samples", type=int, default=20_000)
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--pretrain-opponents", default="random,nearest,baselines/starter")
    parser.add_argument("--pretrain-max-noop-fraction", type=float, default=0.08)
    parser.add_argument("--use-snapshots", action="store_true")
    parser.add_argument("--snapshot-start-phase", type=int, default=4)
    parser.add_argument("--snapshot-every-updates", type=int, default=20)
    parser.add_argument("--snapshot-pool-size", type=int, default=4)
    parser.add_argument("--ignore-bot", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    maybe_pretrain(args)

    ignored = set(DEFAULT_IGNORE)
    ignored.update(args.ignore_bot)
    ratings_path = Path(args.ratings_json).expanduser() if args.ratings_json else None
    ladder = read_ratings(ratings_path, ignored)
    phases = make_phases(
        ladder,
        args.bots_per_phase,
        args.carry,
        args.start_gate_threshold,
        args.end_gate_threshold,
        args.gate_threshold,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "curriculum_state.json"
    events_path = checkpoint_dir / "curriculum_events.jsonl"
    config_path = checkpoint_dir / "curriculum_config.json"
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "ignored": sorted(ignored),
                "ladder": [asdict(bot) for bot in ladder],
                "phases": [asdict(phase) for phase in phases],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_training_report(checkpoint_dir)

    state = load_state(state_path)
    start_time = time.time()
    append_jsonl(events_path, {"kind": "start", "time": start_time, "state": state})
    while state["steps_requested"] < args.total_budget_steps and state["phase_index"] < len(phases):
        if args.max_runtime_hours > 0 and (time.time() - start_time) / 3600.0 >= args.max_runtime_hours:
            append_jsonl(events_path, {"kind": "time_budget_reached", "state": state})
            break

        phase = phases[state["phase_index"]]
        log_phase(phase, state, args.total_budget_steps)
        append_jsonl(events_path, {"kind": "phase_chunk_start", "phase": asdict(phase), "state": state})
        write_training_report(checkpoint_dir)
        existing_initial_gate = gate_for(events_path, phase.index, int(state["steps_requested"]))
        if existing_initial_gate is None:
            initial_checkpoint = latest_checkpoint(checkpoint_dir)
            if initial_checkpoint is None and args.init_checkpoint:
                initial_checkpoint = Path(args.init_checkpoint)
            initial_gate = evaluate_gate(args, phase, checkpoint_dir, checkpoint=initial_checkpoint)
            append_jsonl(
                events_path,
                {
                    "kind": "gate",
                    "initial": True,
                    "phase": asdict(phase),
                    "gate": initial_gate,
                    "state": dict(state),
                },
            )
            log_gate("initial gate", phase, initial_gate)
            if initial_gate["passed"]:
                state["phase_index"] += 1
                save_state(state_path, state)
                append_jsonl(
                    events_path,
                    {
                        "kind": "phase_promoted",
                        "initial": True,
                        "next_phase_index": state["phase_index"],
                        "gate": initial_gate,
                    },
                )
                log(f"{c('promote', GREEN + BOLD)} phase {phase.index} -> {state['phase_index']} before training")
                write_training_report(checkpoint_dir)
                continue
            write_training_report(checkpoint_dir)
        target_after_chunk = state["steps_requested"] + args.chunk_steps
        already_trained = latest_metric_step(checkpoint_dir) >= target_after_chunk
        if already_trained:
            append_jsonl(
                events_path,
                {
                    "kind": "chunk_already_trained",
                    "phase": asdict(phase),
                    "target_step": target_after_chunk,
                    "latest_metric_step": latest_metric_step(checkpoint_dir),
                    "state": state,
                },
            )
        else:
            train_chunk(args, phase, checkpoint_dir)
        state["chunks"] += 1
        state["steps_requested"] = target_after_chunk
        existing_gate = gate_for(events_path, phase.index, int(state["steps_requested"]))
        if existing_gate is None:
            gate = evaluate_gate(args, phase, checkpoint_dir)
            append_jsonl(events_path, {"kind": "gate", "phase": asdict(phase), "gate": gate, "state": state})
        else:
            gate = existing_gate
            append_jsonl(
                events_path,
                {
                    "kind": "gate_reused",
                    "phase": asdict(phase),
                    "gate": gate,
                    "state": state,
                },
            )
        log_gate("gate", phase, gate)
        if gate["passed"]:
            state["phase_index"] += 1
            append_jsonl(events_path, {"kind": "phase_promoted", "next_phase_index": state["phase_index"], "gate": gate})
            log(f"{c('promote', GREEN + BOLD)} phase {phase.index} -> {state['phase_index']}")
        save_state(state_path, state)
        write_training_report(checkpoint_dir)

    append_jsonl(events_path, {"kind": "done", "time": time.time(), "state": state})
    write_training_report(checkpoint_dir)
    print(f"curriculum done; latest checkpoint: {checkpoint_dir / 'latest.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
