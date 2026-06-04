from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

LOCAL_KAGGLE = Path(__file__).resolve().parents[2] / "kaggle-environments"
if LOCAL_KAGGLE.exists():
    sys.path.insert(0, str(LOCAL_KAGGLE))

from jax_model import actions_to_jax, jit_step, reset
from jax_model.features import (
    ACTIONS_DIM,
    GLOBAL_DIM,
    NUM_FRAMES,
    PLANET_SLOTS,
    TOKEN_DIM,
    FeatureConfig,
    _fast_trajectory,
    _slow_trajectory,
    encode,
    jit_encode,
    jit_step_and_encode,
)


def test_encode_shapes_and_finiteness():
    state = reset(3, num_players=2)
    features = jit_encode(state)

    assert features.planet_ids.shape == (PLANET_SLOTS,)
    assert features.tokens.shape == (NUM_FRAMES, PLANET_SLOTS, TOKEN_DIM)
    assert features.globals.shape == (GLOBAL_DIM,)
    assert features.presence.shape == (NUM_FRAMES, PLANET_SLOTS)
    assert features.turns.shape == (NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
    assert features.angles.shape == (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
    assert features.mask.shape == (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
    assert features.ship_counts.shape == (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
    assert features.reachable_mask.shape == (NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
    assert features.frame_planets.shape == (NUM_FRAMES, PLANET_SLOTS, 5)

    assert bool(jnp.all(jnp.isfinite(features.tokens)))
    assert bool(jnp.all(jnp.isfinite(features.globals)))
    assert bool(jnp.all(jnp.isfinite(features.turns)))
    assert bool(jnp.all(jnp.isfinite(features.angles)))


def test_frame_offsets_for_no_fleets_are_canonical():
    state = reset(5, num_players=2)
    features = encode(state)
    assert features.frame_offsets.tolist() == [0, 1, 10, 0]
    assert int(features.num_planets) == int(state.planet_count)
    assert int(jnp.sum(features.presence[0])) == int(state.planet_count)


def test_t1_frame_matches_empty_step():
    state = reset(7, num_players=2)
    features = jit_encode(state)
    actions, mask = actions_to_jax([[], []])
    stepped = jit_step(state, actions, mask)
    stepped_features = jit_encode(stepped)

    assert jnp.allclose(features.frame_planets[1], stepped_features.frame_planets[0])
    assert jnp.allclose(features.tokens[1], stepped_features.tokens[0])


def test_policy_mask_only_exposes_canonical_noop_until_exact_projector_exists():
    state = reset(9, num_players=2)
    features = jit_encode(state)
    assert int(jnp.sum(features.mask[..., 0])) == PLANET_SLOTS
    assert int(jnp.sum(features.mask[..., 1])) == 0
    assert int(jnp.sum(features.reachable_mask)) == 0


def test_step_and_encode_matches_separate_calls():
    state = reset(11, num_players=2)
    actions, mask = actions_to_jax([[], []])
    next_state, features = jit_step_and_encode(state, actions, mask)
    expected_state = jit_step(state, actions, mask)
    expected_features = jit_encode(expected_state)

    assert int(next_state.step) == int(expected_state.step)
    assert jnp.allclose(features.tokens, expected_features.tokens)
    assert jnp.allclose(features.globals, expected_features.globals)


def test_vmap_encode_batch():
    states = jax.tree.map(lambda *xs: jnp.stack(xs), reset(13, 2), reset(14, 2), reset(15, 2))
    features = jax.jit(jax.vmap(jit_encode))(states)
    assert features.tokens.shape == (3, NUM_FRAMES, PLANET_SLOTS, TOKEN_DIM)
    assert features.globals.shape == (3, GLOBAL_DIM)


def test_fast_trajectory_matches_slow_engine_scan_with_fleets_and_comets():
    cfg = FeatureConfig()
    fast = jax.jit(_fast_trajectory, static_argnames=("config",))
    slow = jax.jit(_slow_trajectory, static_argnames=("config",))
    noop, noop_mask = actions_to_jax([[], []])

    cases = []
    for seed in (1, 3, 9):
        state = reset(seed, num_players=2)
        cases.append(state)
        owned = state.planets[(state.planets[:, 1] == 0.0)][:1]
        launch, launch_mask = actions_to_jax([[[int(owned[0, 0]), 0.0, 5]], []])
        cases.append(jit_step(state, launch, launch_mask))
        comet_state = state
        for _ in range(49):
            comet_state = jit_step(comet_state, noop, noop_mask)
        cases.append(comet_state)
        cases.append(jit_step(comet_state, noop, noop_mask))

    for state in cases:
        fast_frames, fast_offsets = fast(state, cfg)
        slow_frames, slow_offsets = slow(state, cfg)
        assert jnp.array_equal(fast_offsets, slow_offsets)
        assert jnp.array_equal(fast_frames.planet_count, slow_frames.planet_count)
        assert jnp.allclose(fast_frames.planets, slow_frames.planets, atol=1e-8)
        assert jnp.allclose(fast_frames.initial_planets, slow_frames.initial_planets, atol=1e-8)
