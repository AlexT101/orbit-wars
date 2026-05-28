from __future__ import annotations

import resource
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn

from arch import GalaxyNet
from features import encode_features
from orbit_wars_engine import OrbitWarsEngine


DEVICE = "cpu"
BATCH_SIZES = (1, 4, 64)
SEED = 0


@dataclass
class ModuleStats:
    name: str
    class_name: str
    own_params: int
    total_params: int
    own_param_bytes: int
    total_param_bytes: int
    output_bytes: int = 0
    output_shape: str = "-"


def fmt_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    x = float(n)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:,.2f} {unit}"
        x /= 1024.0
    raise AssertionError("unreachable")


def tensor_bytes(x: Any) -> int:
    if isinstance(x, th.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return sum(tensor_bytes(v) for v in x)
    return 0


def tensor_shape(x: Any) -> str:
    if isinstance(x, th.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {tensor_shape(v)}" for k, v in x.items()) + "}"
    if isinstance(x, (tuple, list)):
        return "(" + ", ".join(tensor_shape(v) for v in x) + ")"
    return type(x).__name__


def param_count(module: nn.Module, recurse: bool) -> int:
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def param_bytes(module: nn.Module, recurse: bool) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=recurse))


def rss_bytes() -> int:
    # Linux ru_maxrss is KiB. This is max RSS for the process, not just the model.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def make_obs(batch_size: int, device: str) -> dict[str, th.Tensor]:
    engine = OrbitWarsEngine(num_players=2)
    raw_obs = engine.reset(seed=SEED)["observations"][0]
    obs, _feat = encode_features(raw_obs, player=0)
    out: dict[str, th.Tensor] = {}
    for key, value in obs.items():
        arr = np.expand_dims(value, axis=0)
        tensor = th.as_tensor(arr, device=device)
        repeats = (batch_size,) + (1,) * (tensor.ndim - 1)
        out[key] = tensor.repeat(repeats)
    return out


def collect_stats(model: GalaxyNet, obs: dict[str, th.Tensor]) -> tuple[list[ModuleStats], tuple[th.Tensor, th.Tensor]]:
    stats = {
        name: ModuleStats(
            name=name or "<root>",
            class_name=module.__class__.__name__,
            own_params=param_count(module, recurse=False),
            total_params=param_count(module, recurse=True),
            own_param_bytes=param_bytes(module, recurse=False),
            total_param_bytes=param_bytes(module, recurse=True),
        )
        for name, module in model.named_modules()
    }

    hooks = []

    def hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            stats[name].output_bytes += tensor_bytes(output)
            stats[name].output_shape = tensor_shape(output)

        return _hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook(name)))

    try:
        output = model(obs)
    finally:
        for h in hooks:
            h.remove()

    return list(stats.values()), output


def print_table(rows: list[ModuleStats]) -> None:
    print(
        f"{'module':46} {'type':22} {'own params':>12} {'total params':>12} "
        f"{'own mem':>12} {'total mem':>12} {'out mem':>12} output"
    )
    print("-" * 150)
    for r in rows:
        if r.total_params == 0 and r.output_bytes == 0:
            continue
        print(
            f"{r.name[:46]:46} {r.class_name[:22]:22} "
            f"{r.own_params:12,d} {r.total_params:12,d} "
            f"{fmt_bytes(r.own_param_bytes):>12} {fmt_bytes(r.total_param_bytes):>12} "
            f"{fmt_bytes(r.output_bytes):>12} {r.output_shape}"
        )


def main() -> int:
    th.set_grad_enabled(True)
    device = DEVICE
    model = GalaxyNet().to(device)
    total_params = param_count(model, recurse=True)
    total_param_bytes = param_bytes(model, recurse=True)

    print(f"device: {device}")
    print(f"parameters: {total_params:,} ({fmt_bytes(total_param_bytes)})")
    print(f"gradients:  {fmt_bytes(total_param_bytes)} if all params receive grads")
    print(f"Adam state: {fmt_bytes(total_param_bytes * 2)} for exp_avg + exp_avg_sq")
    print(f"train state rough total: {fmt_bytes(total_param_bytes * 4)} params + grads + Adam state")
    print(f"process max RSS before forwards: {fmt_bytes(rss_bytes())}")
    print()

    for batch_size in BATCH_SIZES:
        obs = make_obs(batch_size, device)
        input_bytes = tensor_bytes(obs)
        before = rss_bytes()
        stats, output = collect_stats(model, obs)
        value, logits = output
        after = rss_bytes()

        print(f"=== batch_size={batch_size} ===")
        print(f"input tensors: {fmt_bytes(input_bytes)}")
        print(f"value output:  {tuple(value.shape)} {fmt_bytes(tensor_bytes(value))}")
        print(f"logits output: {tuple(logits.shape)} {fmt_bytes(tensor_bytes(logits))}")
        print(f"module output bytes, summed: {fmt_bytes(sum(r.output_bytes for r in stats))}")
        print(f"process max RSS delta: {fmt_bytes(max(0, after - before))}")
        print_table(stats)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
