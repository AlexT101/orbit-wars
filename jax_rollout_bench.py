"""Minimal vectorized rollout demo for the JAX engine.

Drives N parallel games on GPU with a trivial JAX policy (random-launch-from-
owned-planet). Measures end-to-end game-steps/s. This is the upper bound for
what a JAX-native PPO loop could hit — once you swap in a real policy net the
ceiling drops a bit but the engine remains the same.

Run remotely with:
    XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u jax_rollout_bench.py
"""

import time
import contextlib
import io

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

with contextlib.redirect_stdout(io.StringIO()):
    from bots.mine.jax_engine.init import init_batch
    from bots.mine.jax_engine.action import ActionBatch
    from bots.mine.jax_engine.state import (
        BatchState, MAX_ACTIONS_PER_PLAYER, NUM_PLAYERS_PAD, MAX_PLANETS,
        P_ID, P_OWNER, P_SHIPS,
    )
    from bots.mine.jax_engine.step import step_batch


def random_policy(state: BatchState, key) -> ActionBatch:
    """For each player, pick up to one owned-planet launch per turn at a
    uniform random angle. Sends half the ships."""
    B = state.planets.shape[0]

    def per_game(p, mk, num_players, key):
        # p: (MAX_PLANETS, 7), mk: (MAX_PLANETS,)
        def per_player(player_id, key):
            owns = mk & (p[:, P_OWNER].astype(jnp.int32) == player_id) & (p[:, P_SHIPS] > 20)
            any_own = jnp.any(owns)
            # First-owned slot
            slot = jnp.argmax(owns.astype(jnp.int32))
            kid, kang, kshp = jax.random.split(key, 3)
            angle = jax.random.uniform(kang, (), minval=-3.141592, maxval=3.141592)
            from_id = p[slot, P_ID]
            ships_avail = p[slot, P_SHIPS]
            ships = jnp.maximum(jnp.int32(1), (ships_avail / 2).astype(jnp.int32))
            move = jnp.stack([
                from_id.astype(jnp.float64),
                angle.astype(jnp.float64),
                ships.astype(jnp.float64),
            ])
            # Pad to MAX_ACTIONS_PER_PLAYER with zeros; mask first slot only.
            moves = jnp.zeros((MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float64).at[0].set(move)
            mask = jnp.zeros((MAX_ACTIONS_PER_PLAYER,), dtype=jnp.bool_).at[0].set(
                any_own & (player_id < num_players)
            )
            return moves, mask

        # Build for all NUM_PLAYERS_PAD players (mask out invalid in step).
        keys = jax.random.split(key, NUM_PLAYERS_PAD)
        moves_per = []
        masks_per = []
        for pi in range(NUM_PLAYERS_PAD):
            m, msk = per_player(jnp.int32(pi), keys[pi])
            moves_per.append(m)
            masks_per.append(msk)
        moves = jnp.stack(moves_per, axis=0)
        masks = jnp.stack(masks_per, axis=0)
        return moves, masks

    keys = jax.random.split(key, B)
    moves, masks = jax.vmap(per_game)(state.planets, state.planet_mask, state.num_players, keys)
    return ActionBatch(moves=moves, mask=masks)


@jax.jit
def rollout_step(state, key):
    actions = random_policy(state, key)
    return step_batch(state, actions)


def main():
    print('device:', jax.devices(), flush=True)
    print('=== batched rollout (JAX engine + JAX random policy) ===', flush=True)
    rng = jax.random.PRNGKey(0)

    for BATCH in [16, 64, 256, 1024]:
        try:
            state = init_batch(list(range(BATCH)), num_players=2)
            state = jax.tree_util.tree_map(jax.device_put, state)

            # Warm
            t0 = time.time()
            key, sub = jax.random.split(rng)
            s = rollout_step(state, sub)
            jax.block_until_ready(s.planets)
            print(f"  batch={BATCH:>5d}  compile={int((time.time()-t0)*1000):>5d}ms", flush=True)

            # Steady-state (rollout 500 steps total).
            t0 = time.time()
            N = 500
            for i in range(N):
                key, sub = jax.random.split(key)
                s = rollout_step(s, sub)
            jax.block_until_ready(s.planets)
            elapsed = time.time() - t0
            per_step = elapsed / N * 1000
            gps = BATCH / (elapsed / N)
            print(f"    {N} steps in {elapsed:.2f}s  -> {per_step:.2f}ms/step  {gps:>8.0f} g-s/s", flush=True)
        except Exception as e:
            print(f"  batch={BATCH} ERROR: {type(e).__name__}: {str(e)[:100]}", flush=True)


if __name__ == "__main__":
    main()
