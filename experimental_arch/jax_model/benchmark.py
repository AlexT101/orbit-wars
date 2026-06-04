from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from .engine import Limits, actions_to_jax, jit_step, reset


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark compiled JAX Orbit Wars stepping.")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-fleets", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    limits = Limits(max_fleets=args.max_fleets)
    state = reset(args.seed, limits=limits)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (args.batch,) + x.shape), state)
    actions, mask = actions_to_jax([[], []], limits)
    actions = jnp.broadcast_to(actions, (args.batch,) + actions.shape)
    mask = jnp.broadcast_to(mask, (args.batch,) + mask.shape)
    step_fn = jax.jit(jax.vmap(jit_step))

    states = step_fn(states, actions, mask)
    jax.block_until_ready(states.step)
    start = time.perf_counter()
    for _ in range(args.steps):
        states = step_fn(states, actions, mask)
    jax.block_until_ready(states.step)
    elapsed = time.perf_counter() - start
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        f"batch={args.batch} steps={args.steps} max_fleets={args.max_fleets} "
        f"env_steps_s={args.batch * args.steps / elapsed:.0f} wall={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
