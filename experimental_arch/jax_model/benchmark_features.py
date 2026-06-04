from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from .engine import actions_to_jax, batch_reset
from .features import FeatureConfig, jit_encode, jit_step_and_encode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--cheap-actions", action="store_true")
    parser.add_argument("--aim-horizon", type=int, default=64)
    args = parser.parse_args()
    config = FeatureConfig(aim_horizon=args.aim_horizon, exact_actions=not args.cheap_actions)

    states = batch_reset(list(range(args.batch)), num_players=args.players)
    actions, mask = actions_to_jax([[] for _ in range(args.players)])
    batched_actions = jnp.broadcast_to(actions, (args.batch,) + actions.shape)
    batched_mask = jnp.broadcast_to(mask, (args.batch,) + mask.shape)

    encode_fn = jax.jit(
        jax.vmap(jit_encode, in_axes=(0, None, None)),
        static_argnames=("player", "config"),
    )
    step_encode_fn = jax.jit(
        jax.vmap(jit_step_and_encode, in_axes=(0, 0, 0, None, None)),
        static_argnames=("player", "config"),
    )

    features = encode_fn(states, 0, config)
    features.tokens.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        features = encode_fn(states, 0, config)
    features.tokens.block_until_ready()
    encode_wall = time.perf_counter() - t0

    next_states, features = step_encode_fn(states, batched_actions, batched_mask, 0, config)
    features.tokens.block_until_ready()
    t0 = time.perf_counter()
    cur = states
    for _ in range(args.steps):
        cur, features = step_encode_fn(cur, batched_actions, batched_mask, 0, config)
    features.tokens.block_until_ready()
    step_wall = time.perf_counter() - t0

    print(
        f"backend={jax.default_backend()} devices={jax.devices()} "
        f"exact_actions={not args.cheap_actions} aim_horizon={args.aim_horizon}"
    )
    encode_rate = args.batch * args.steps / encode_wall
    step_rate = args.batch * args.steps / step_wall
    print(
        f"encode batch={args.batch} steps={args.steps} "
        f"env_features_s={encode_rate:.3f} wall={encode_wall:.3f}s"
    )
    print(
        f"step_encode batch={args.batch} steps={args.steps} "
        f"env_steps_s={step_rate:.3f} wall={step_wall:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
