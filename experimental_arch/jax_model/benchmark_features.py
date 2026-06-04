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
    args = parser.parse_args()

    states = batch_reset(list(range(args.batch)), num_players=args.players)
    actions, mask = actions_to_jax([[] for _ in range(args.players)])
    batched_actions = jnp.broadcast_to(actions, (args.batch,) + actions.shape)
    batched_mask = jnp.broadcast_to(mask, (args.batch,) + mask.shape)

    encode_fn = jax.jit(jax.vmap(jit_encode))
    step_encode_fn = jax.jit(jax.vmap(jit_step_and_encode))

    features = encode_fn(states)
    features.tokens.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        features = encode_fn(states)
    features.tokens.block_until_ready()
    encode_wall = time.perf_counter() - t0

    next_states, features = step_encode_fn(states, batched_actions, batched_mask)
    features.tokens.block_until_ready()
    t0 = time.perf_counter()
    cur = states
    for _ in range(args.steps):
        cur, features = step_encode_fn(cur, batched_actions, batched_mask)
    features.tokens.block_until_ready()
    step_wall = time.perf_counter() - t0

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        f"encode batch={args.batch} steps={args.steps} "
        f"env_features_s={args.batch * args.steps / encode_wall:.0f} wall={encode_wall:.3f}s"
    )
    print(
        f"step_encode batch={args.batch} steps={args.steps} "
        f"env_steps_s={args.batch * args.steps / step_wall:.0f} wall={step_wall:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
