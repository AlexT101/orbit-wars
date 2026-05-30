from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from .env import OrbitWarsDuelEnv, RewardWeights
from .features import ACTION_DIM, MAX_PLANETS, SEND_FRACTIONS, decode_action_index, decode_move, encode_obs
from .model import OrbitPolicy, build_policy
from .visualization import append_jsonl, write_training_report


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


@dataclass
class PPOConfig:
    total_steps: int = 20_000
    rollout_steps: int = 32
    update_epochs: int = 1
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-5
    clip_coef: float = 0.20
    ent_coef: float = 0.05
    ent_coef_final: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    weight_decay: float = 1.0e-4
    seed: int = 7
    opponent: str = "nearest"
    device: str = "cpu"
    checkpoint_dir: str = "rl_orbit_wars/checkpoints"
    reward_mode: str = "terminal"
    eval_every_updates: int = 25
    eval_games: int = 4
    eval_opponents: str = "random,nearest,baselines/starter"
    report_every_updates: int = 1
    init_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    snapshot_every_updates: int = 0
    snapshot_pool_size: int = 4
    model: str = "mlp"
    hidden: int = 128
    transformer_layers: int = 3
    transformer_heads: int = 4
    lr_warmup_steps: int = 0
    lr_schedule: str = "linear"


def _last_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


def _max_jsonl_value(path: Path, key: str, default: float) -> float:
    if not path.exists():
        return default
    best = default
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key in row:
            best = max(best, float(row[key]))
    return best


def _stack_encoded(items, device):
    return {
        "planets": torch.as_tensor(np.stack([x.planets for x in items]), dtype=torch.float32, device=device),
        "planet_mask": torch.as_tensor(np.stack([x.planet_mask for x in items]), dtype=torch.float32, device=device),
        "globals_": torch.as_tensor(np.stack([x.globals for x in items]), dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(np.stack([x.action_mask for x in items]), dtype=torch.bool, device=device),
    }


def _sample_action(model, encoded, device):
    batch = _stack_encoded([encoded], device)
    with torch.no_grad():
        logits, value = model(**batch)
        dist = Categorical(logits=logits)
        action = dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def _greedy_action(model, obs, device) -> int:
    encoded = encode_obs(obs)
    batch = _stack_encoded([encoded], device)
    with torch.no_grad():
        logits, _value = model(**batch)
    return int(torch.argmax(logits, dim=-1).item())


def _scheduled_lr(base_lr: float, progress: float, step_delta: int, warmup_steps: int, schedule: str) -> float:
    if warmup_steps > 0 and step_delta < warmup_steps:
        return base_lr * max(0.05, step_delta / max(1, warmup_steps))
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        return base_lr * max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return base_lr * max(0.05, 1.0 - progress)


def _scheduled_entropy(start: float, final: float, progress: float) -> float:
    return final + (start - final) * max(0.0, 1.0 - progress)


def _component_entropies(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    noop_prob = probs[:, :1]
    pair_probs = probs[:, 1:].reshape(logits.shape[0], MAX_PLANETS, MAX_PLANETS, len(SEND_FRACTIONS))
    launch_prob = pair_probs.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    source_prob = pair_probs.sum(dim=(2, 3)) / launch_prob[:, None]
    target_prob = pair_probs.sum(dim=(1, 3)) / launch_prob[:, None]
    send_prob = pair_probs.sum(dim=(1, 2)) / launch_prob[:, None]

    def entropy(p):
        return -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1).mean().item()

    launch_binary = torch.cat([noop_prob, launch_prob[:, None]], dim=-1)
    return {
        "entropy_launch": float(entropy(launch_binary)),
        "entropy_source": float(entropy(source_prob)),
        "entropy_target": float(entropy(target_prob)),
        "entropy_send": float(entropy(send_prob)),
    }


def _names_csv(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or ["nearest"]


def _metric_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _format_train_metrics(metrics: dict, opponent_names: list[str]) -> str:
    ret = float(metrics.get("mean_return_25", 0.0) or 0.0)
    clip = float(metrics.get("clip_frac", 0.0) or 0.0)
    ev = float(metrics.get("explained_var", 0.0) or 0.0)
    reward = float(metrics.get("reward_mean", 0.0) or 0.0)
    launch = float(metrics.get("launch_rate", 0.0) or 0.0)
    entropy_launch = float(metrics.get("entropy_launch", 0.0) or 0.0)
    wr_parts = []
    for name in opponent_names:
        key = f"train_win_rate_{_metric_name(name)}"
        if key in metrics:
            wr = float(metrics[key])
            wr_parts.append(f"{_c(name, BLUE)}={_c(f'{wr:.0%}', GREEN if wr >= 0.5 else RED)}")
    wr_text = " ".join(wr_parts) if wr_parts else _c("no completed games yet", DIM)
    clip_color = GREEN if clip < 0.15 else (YELLOW if clip < 0.30 else RED)
    ev_color = GREEN if ev >= 0.5 else (YELLOW if ev >= 0.1 else RED)
    return (
        f"{_c('ppo', BOLD + CYAN)} "
        f"upd={int(metrics.get('update', 0)):>4} "
        f"step={int(metrics.get('step', 0)):>7} "
        f"opp={_c(str(metrics.get('current_opponent', '?')), MAGENTA)} "
        f"ret25={_c(f'{ret:7.2f}', GREEN if ret >= 0 else RED)} "
        f"r={reward:+.4f} "
        f"wr[{wr_text}] "
        f"sps={float(metrics.get('sps', 0.0) or 0.0):.1f} "
        f"launch={launch:.0%} "
        f"Hlaunch={entropy_launch:.3f} "
        f"clip={_c(f'{clip:.3f}', clip_color)} "
        f"ev={_c(f'{ev:.3f}', ev_color)} "
        f"lr={float(metrics.get('lr', 0.0) or 0.0):.2g}"
    )


def _format_eval_metrics(eval_metrics: dict) -> str:
    parts = []
    for key in sorted(eval_metrics):
        if key.startswith("win_rate_"):
            name = key.replace("win_rate_", "")
            wr = float(eval_metrics.get(key, 0.0) or 0.0)
            parts.append(f"{_c(name, BLUE)}={_c(f'{wr:.0%}', GREEN if wr >= 0.5 else RED)}")
    score = float(eval_metrics.get("eval_score", 0.0) or 0.0)
    return (
        f"{_c('eval', BOLD + MAGENTA)} "
        f"upd={int(eval_metrics.get('update', 0)):>4} "
        f"step={int(eval_metrics.get('step', 0)):>7} "
        f"score={_c(f'{score:.1%}', GREEN if score >= 0.5 else RED)} "
        + " ".join(parts)
    )


def _policy_opponent(model: OrbitPolicy, device, sample: bool = False):
    def opponent(obs):
        encoded = encode_obs(obs)
        batch = _stack_encoded([encoded], device)
        with torch.no_grad():
            logits, _value = model(**batch)
            if sample:
                action = Categorical(logits=logits).sample()
            else:
                action = torch.argmax(logits, dim=-1)
        return decode_move(obs, int(action.item()))

    return opponent


def _state_dict_cpu(model: OrbitPolicy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _model_from_state(state: dict[str, torch.Tensor], device, config: PPOConfig) -> OrbitPolicy:
    model = build_policy(
        config.model,
        config.hidden,
        config.transformer_layers,
        config.transformer_heads,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _checkpoint_opponent(path: str, device, sample: bool = False):
    ckpt = torch.load(path, map_location=device)
    ckpt_config = ckpt.get("config", {})
    model = build_policy(
        ckpt_config.get("model", "mlp"),
        ckpt_config.get("hidden", 128),
        ckpt_config.get("transformer_layers", 3),
        ckpt_config.get("transformer_heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return _policy_opponent(model, device, sample=sample)


def evaluate_policy(
    model: OrbitPolicy,
    device,
    opponent: str,
    games: int,
    seed: int,
    step: int,
    progress_label: str | None = None,
) -> dict:
    wins = losses = draws = 0
    margins = []
    if opponent in {"self", "self_sample", "snapshot", "snapshot_sample"}:
        opponent_agent = _policy_opponent(
            model,
            device,
            sample=opponent in {"self_sample", "snapshot_sample"},
        )
    elif opponent.startswith("checkpoint:") or opponent.startswith("checkpoint_sample:"):
        prefix, path = opponent.split(":", 1)
        opponent_agent = _checkpoint_opponent(path, device, sample=prefix == "checkpoint_sample")
    else:
        opponent_agent = opponent
    for i in range(games):
        env = OrbitWarsDuelEnv(seed=seed + i, opponent=opponent_agent)
        obs = env.reset(seed + i)
        done = False
        result = None
        while not done:
            action = _greedy_action(model, obs, device)
            result = env.step(action)
            obs = result.obs
            done = result.done
        assert result is not None
        stats = result.info["stats"]
        margin = float(stats["own_score"] - stats["enemy_score"])
        margins.append(margin)
        raw = result.info["raw_rewards"]
        if raw[0] > raw[1]:
            wins += 1
            outcome = _c("W", GREEN)
        elif raw[1] > raw[0]:
            losses += 1
            outcome = _c("L", RED)
        else:
            draws += 1
            outcome = _c("D", YELLOW)
        if progress_label:
            wr = (wins + 0.5 * draws) / (i + 1)
            print(
                f"{_c(progress_label, BOLD + MAGENTA)} "
                f"{_c(opponent, BLUE)} {i + 1:>3}/{games} {outcome} "
                f"wr={_c(f'{wr:.1%}', GREEN if wr >= 0.5 else RED)} "
                f"margin={margin:+.0f}",
                flush=True,
            )
    return {
        "step": step,
        "games": games,
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / max(1, games),
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
    }


def train(config: PPOConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    metrics_path = checkpoint_dir / "metrics.jsonl"
    eval_path = checkpoint_dir / "eval.jsonl"
    phase_path = checkpoint_dir / "phase_events.jsonl"
    is_resume = bool(config.resume_checkpoint)
    if not is_resume:
        metrics_path.write_text("")
        eval_path.write_text("")
        phase_path.write_text("")
    else:
        metrics_path.touch()
        eval_path.touch()
        phase_path.touch()

    model = build_policy(
        config.model,
        config.hidden,
        config.transformer_layers,
        config.transformer_heads,
    ).to(device)
    resume_ckpt = None
    if config.resume_checkpoint:
        resume_ckpt = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        append_jsonl(
            phase_path,
            {
                "kind": "resume",
                "step": int(resume_ckpt.get("global_step", _last_jsonl(metrics_path).get("step", 0) or 0)),
                "label": "resumed",
                "checkpoint": config.resume_checkpoint,
            },
        )
    elif config.init_checkpoint:
        ckpt = torch.load(config.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        ckpt_config = ckpt.get("config", {})
        append_jsonl(
            phase_path,
            {
                "kind": "pretrain_end",
                "step": 0,
                "label": "pretrain ended",
                "checkpoint": config.init_checkpoint,
                "teacher": ckpt_config.get("teacher"),
                "bc_samples": ckpt_config.get("bc_samples"),
                "bc_epochs": ckpt_config.get("bc_epochs"),
                "bc_final_accuracy": ckpt_config.get("bc_final_accuracy"),
                "bc_final_loss": ckpt_config.get("bc_final_loss"),
            },
        )
    else:
        append_jsonl(phase_path, {"kind": "ppo_start", "step": 0, "label": "PPO start"})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
    last_metrics = _last_jsonl(metrics_path) if is_resume else {}
    global_step = int(
        (resume_ckpt or {}).get("global_step", last_metrics.get("step", 0) or 0)
    )
    start_global_step = global_step
    target_global_step = start_global_step + config.total_steps if is_resume else config.total_steps
    episode = int((resume_ckpt or {}).get("episode", last_metrics.get("episode", 0) or 0))
    episode_return = 0.0
    best_mean_return = float(
        (resume_ckpt or {}).get(
            "best_mean_return",
            _max_jsonl_value(metrics_path, "mean_return_25", -1e9) if is_resume else -1e9,
        )
    )
    recent_returns: list[float] = list((resume_ckpt or {}).get("recent_returns", []))
    start_time = perf_counter()
    update_idx = int((resume_ckpt or {}).get("update_idx", last_metrics.get("update", 0) or 0))
    opponent_names = _names_csv(config.opponent)
    opponent_cursor = int((resume_ckpt or {}).get("opponent_cursor", episode))
    recent_opponent_returns: dict[str, list[float]] = {
        k: list(v) for k, v in (resume_ckpt or {}).get("recent_opponent_returns", {}).items()
    }
    recent_opponent_outcomes: dict[str, list[float]] = {
        k: list(v) for k, v in (resume_ckpt or {}).get("recent_opponent_outcomes", {}).items()
    }
    snapshot_states: list[dict[str, torch.Tensor]] = [
        {key: value.detach().cpu().clone() for key, value in state.items()}
        for state in (resume_ckpt or {}).get("opponent_snapshots", [])
    ]
    checkpoint_opponent_cache = {}

    def capture_snapshot() -> None:
        if config.snapshot_pool_size <= 0:
            return
        snapshot_states.append(_state_dict_cpu(model))
        del snapshot_states[: max(0, len(snapshot_states) - config.snapshot_pool_size)]

    if any(name in {"snapshot", "snapshot_sample"} for name in opponent_names) and not snapshot_states:
        capture_snapshot()

    def make_env(opponent_name: str, env_seed: int) -> OrbitWarsDuelEnv:
        if opponent_name in {"self", "self_sample"}:
            opponent = _policy_opponent(model, device, sample=opponent_name == "self_sample")
        elif opponent_name in {"snapshot", "snapshot_sample"}:
            if snapshot_states:
                snapshot_model = _model_from_state(random.choice(snapshot_states), device, config)
                opponent = _policy_opponent(snapshot_model, device, sample=opponent_name == "snapshot_sample")
            else:
                opponent = _policy_opponent(model, device, sample=True)
        elif opponent_name.startswith("checkpoint:") or opponent_name.startswith("checkpoint_sample:"):
            if opponent_name not in checkpoint_opponent_cache:
                prefix, path = opponent_name.split(":", 1)
                checkpoint_opponent_cache[opponent_name] = _checkpoint_opponent(
                    path,
                    device,
                    sample=prefix == "checkpoint_sample",
                )
            opponent = checkpoint_opponent_cache[opponent_name]
        else:
            opponent = opponent_name
        return OrbitWarsDuelEnv(
            seed=env_seed,
            opponent=opponent,
            reward_weights=RewardWeights(mode=config.reward_mode),
        )

    def choose_opponent() -> str:
        return opponent_names[opponent_cursor % len(opponent_names)]

    current_opponent = choose_opponent()
    env = make_env(current_opponent, config.seed + episode)
    obs = env.reset(config.seed + episode)

    while global_step < target_global_step:
        update_idx += 1
        encoded_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        component_sums: dict[str, float] = {}
        noop_count = 0
        launch_count = 0
        send_bin_sum = 0.0

        for _ in range(config.rollout_steps):
            encoded = encode_obs(obs)
            action, logprob, value = _sample_action(model, encoded, device)
            decoded_action = decode_action_index(action)
            if decoded_action is None:
                noop_count += 1
            else:
                launch_count += 1
                send_bin_sum += decoded_action[2]
            result = env.step(action)
            encoded_buf.append(encoded)
            action_buf.append(action)
            logprob_buf.append(logprob)
            reward_buf.append(result.reward)
            done_buf.append(result.done)
            value_buf.append(value)
            for name, component_value in result.info.get("reward_components", {}).items():
                component_sums[name] = component_sums.get(name, 0.0) + float(component_value)
            episode_return += result.reward
            global_step += 1

            obs = result.obs
            if result.done:
                recent_returns.append(episode_return)
                recent_returns = recent_returns[-25:]
                recent_opponent_returns.setdefault(current_opponent, []).append(episode_return)
                recent_opponent_returns[current_opponent] = recent_opponent_returns[current_opponent][-25:]
                raw = result.info.get("raw_rewards", [0.0, 0.0])
                outcome = 1.0 if raw[0] > raw[1] else (0.0 if raw[1] > raw[0] else 0.5)
                recent_opponent_outcomes.setdefault(current_opponent, []).append(outcome)
                recent_opponent_outcomes[current_opponent] = recent_opponent_outcomes[current_opponent][-25:]
                episode += 1
                opponent_cursor += 1
                current_opponent = choose_opponent()
                episode_return = 0.0
                env = make_env(current_opponent, config.seed + episode)
                obs = env.reset(config.seed + episode)
            if global_step >= target_global_step:
                break

        with torch.no_grad():
            next_encoded = encode_obs(obs)
            next_batch = _stack_encoded([next_encoded], device)
            _, next_value_t = model(**next_batch)
            next_value = float(next_value_t.item())

        rewards = np.asarray(reward_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        values = np.asarray(value_buf + [next_value], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + config.gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + config.gamma * config.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + np.asarray(value_buf, dtype=np.float32)
        value_targets = np.asarray(value_buf, dtype=np.float32)
        explained_denom = float(np.var(returns))
        explained_var = (
            1.0 - float(np.var(returns - value_targets)) / explained_denom
            if explained_denom > 1e-8
            else 0.0
        )

        batch = _stack_encoded(encoded_buf, device)
        actions = torch.as_tensor(action_buf, dtype=torch.long, device=device)
        old_logprobs = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))

        batch_size = len(action_buf)
        inds = np.arange(batch_size)
        progress_steps = global_step - start_global_step if is_resume else global_step
        progress_total = config.total_steps if is_resume else config.total_steps
        progress = min(1.0, progress_steps / max(1, progress_total))
        lr = _scheduled_lr(
            config.learning_rate,
            progress,
            progress_steps,
            config.lr_warmup_steps,
            config.lr_schedule,
        )
        ent_coef = _scheduled_entropy(config.ent_coef, config.ent_coef_final, progress)
        optimizer.param_groups[0]["lr"] = lr

        last_stats = {}
        for _epoch in range(config.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, config.minibatch_size):
                mb = inds[start : start + config.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
                logits, values_t = model(
                    batch["planets"][mb_t],
                    batch["planet_mask"][mb_t],
                    batch["globals_"][mb_t],
                    batch["action_mask"][mb_t],
                )
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(actions[mb_t])
                entropy = dist.entropy().mean()
                logratio = new_logprob - old_logprobs[mb_t]
                ratio = logratio.exp()
                pg_loss1 = -adv[mb_t] * ratio
                pg_loss2 = -adv[mb_t] * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = F.mse_loss(values_t, ret[mb_t])
                loss = policy_loss + config.vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                last_stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "clip_frac": clip_frac,
                    "approx_kl": approx_kl,
                    "lr": lr,
                    "ent_coef": ent_coef,
                }

        with torch.no_grad():
            logits, values_t = model(
                batch["planets"],
                batch["planet_mask"],
                batch["globals_"],
                batch["action_mask"],
            )
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(actions)
            logratio = new_logprob - old_logprobs
            ratio = logratio.exp()
            last_stats["entropy"] = float(dist.entropy().mean().item())
            last_stats["clip_frac"] = float(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
            last_stats["approx_kl"] = float(((ratio - 1.0) - logratio).mean().item())
            last_stats["value_loss"] = float(F.mse_loss(values_t, ret).item())
            last_stats.update(_component_entropies(logits))

        mean_return = float(np.mean(recent_returns)) if recent_returns else episode_return
        session_steps = global_step - start_global_step if is_resume else global_step
        sps = session_steps / max(1e-6, perf_counter() - start_time)
        reward_components = {
            f"reward_{name}": round(total / max(1, len(reward_buf)), 6)
            for name, total in sorted(component_sums.items())
        }
        train_opponent_metrics = {}
        for name in opponent_names:
            key = _metric_name(name)
            returns_for_name = recent_opponent_returns.get(name, [])
            outcomes_for_name = recent_opponent_outcomes.get(name, [])
            if returns_for_name:
                train_opponent_metrics[f"train_return_{key}"] = round(float(np.mean(returns_for_name)), 4)
                train_opponent_metrics[f"train_games_{key}"] = len(returns_for_name)
            if outcomes_for_name:
                train_opponent_metrics[f"train_win_rate_{key}"] = round(float(np.mean(outcomes_for_name)), 6)
        if (
            config.snapshot_every_updates > 0
            and update_idx % config.snapshot_every_updates == 0
            and any(name in {"snapshot", "snapshot_sample"} for name in opponent_names)
        ):
            capture_snapshot()
        metrics = {
            "step": global_step,
            "update": update_idx,
            "episode": episode,
            "current_opponent": current_opponent,
            "snapshot_pool_size": len(snapshot_states),
            "mean_return_25": round(mean_return, 4),
            "sps": round(sps, 1),
            "explained_var": round(explained_var, 6),
            "reward_mean": round(float(np.mean(reward_buf)), 6),
            "noop_rate": round(noop_count / max(1, len(action_buf)), 6),
            "launch_rate": round(launch_count / max(1, len(action_buf)), 6),
            "avg_send_bin": round(send_bin_sum / max(1, launch_count), 6),
            **{k: round(v, 6) for k, v in last_stats.items()},
            **reward_components,
            **train_opponent_metrics,
        }
        print(_format_train_metrics(metrics, opponent_names), flush=True)
        append_jsonl(metrics_path, metrics)

        improved_best = mean_return > best_mean_return and recent_returns
        if improved_best:
            best_mean_return = mean_return

        latest = checkpoint_dir / "latest.pt"
        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "action_dim": ACTION_DIM,
            "global_step": global_step,
            "episode": episode,
            "update_idx": update_idx,
            "recent_returns": recent_returns,
            "best_mean_return": best_mean_return,
            "opponent_cursor": opponent_cursor,
            "recent_opponent_returns": recent_opponent_returns,
            "recent_opponent_outcomes": recent_opponent_outcomes,
            "opponent_snapshots": snapshot_states,
        }
        torch.save(checkpoint_payload, latest)
        if improved_best:
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")

        if config.eval_every_updates > 0 and update_idx % config.eval_every_updates == 0:
            eval_metrics = {"step": global_step, "update": update_idx, "games": config.eval_games}
            eval_win_rates = []
            eval_margins = []
            for eval_idx, opponent in enumerate(_names_csv(config.eval_opponents)):
                result = evaluate_policy(
                    model,
                    device,
                    opponent,
                    config.eval_games,
                    config.seed + 100_000 + update_idx * 100 + eval_idx * 10_000,
                    global_step,
                    progress_label="eval",
                )
                eval_metrics[f"win_rate_{opponent}"] = result["win_rate"]
                eval_metrics[f"avg_margin_{opponent}"] = result["avg_margin"]
                eval_metrics[f"wins_{opponent}"] = result["wins"]
                eval_metrics[f"losses_{opponent}"] = result["losses"]
                eval_metrics[f"draws_{opponent}"] = result["draws"]
                eval_win_rates.append(result["win_rate"])
                eval_margins.append(result["avg_margin"])
            if eval_win_rates:
                eval_metrics["eval_score"] = float(np.mean(eval_win_rates))
                eval_metrics["eval_avg_margin"] = float(np.mean(eval_margins))
            append_jsonl(eval_path, eval_metrics)
            print(_format_eval_metrics(eval_metrics), flush=True)

        if config.report_every_updates > 0 and update_idx % config.report_every_updates == 0:
            write_training_report(checkpoint_dir)

    return checkpoint_dir / "latest.pt"
