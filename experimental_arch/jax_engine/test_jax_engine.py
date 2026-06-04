from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

LOCAL_KAGGLE = Path(__file__).resolve().parents[2] / "kaggle-environments"
if LOCAL_KAGGLE.exists():
    sys.path.insert(0, str(LOCAL_KAGGLE))

from engine_parity_checker.agents import AGENTS
from engine_parity_checker.harness import run_parity
from engine_parity_checker.kaggle_engine import KaggleEngine
from jax_engine import JaxOrbitWarsEngine, actions_to_jax, batch_reset, jit_step, reset


def _assert_parity(agent_name: str, seed: int, players: int, steps: int, atol: float = 1e-9):
    result = run_parity(
        engine_a=KaggleEngine(),
        engine_b=JaxOrbitWarsEngine(use_jit=True),
        seed=seed,
        num_players=players,
        agents=[AGENTS[agent_name]] * players,
        max_steps=steps,
        atol=atol,
    )
    assert result.converged, result.summary()


def test_reset_parity_2p_and_4p():
    _assert_parity("noop", seed=1, players=2, steps=0)
    _assert_parity("noop", seed=2024, players=4, steps=0)


def test_noop_step_parity_crosses_first_comet():
    _assert_parity("noop", seed=42, players=2, steps=55)
    _assert_parity("noop", seed=123, players=4, steps=55)


def test_aggressive_step_parity_before_comets():
    _assert_parity("aggressive", seed=7, players=2, steps=30, atol=1e-8)


def test_nearest_sniper_step_parity_before_comets():
    _assert_parity("nearest_sniper", seed=2024, players=2, steps=35, atol=1e-8)


def test_random_4p_parity_crosses_first_comet():
    _assert_parity("random", seed=123, players=4, steps=80, atol=1e-8)


def test_jit_step_and_batch_reset_shapes():
    state = reset(5, num_players=2)
    actions, mask = actions_to_jax([[], []])
    next_state = jit_step(state, actions, mask)
    assert int(next_state.step) == 1
    assert next_state.planets.shape == state.planets.shape
    batch = batch_reset([1, 2, 3], num_players=2)
    batch_actions = jnp.broadcast_to(actions, (3,) + actions.shape)
    batch_mask = jnp.broadcast_to(mask, (3,) + mask.shape)
    stepped = jax.jit(jax.vmap(jit_step))(batch, batch_actions, batch_mask)
    assert stepped.planets.shape[0] == 3
    assert stepped.fleets.shape[1:] == state.fleets.shape


def test_simple_launch_creates_fleet():
    state = reset(9, num_players=2)
    host_planets = state.planets
    owned = host_planets[(host_planets[:, 1] == 0.0)][:1]
    pid = int(owned[0, 0])
    actions, mask = actions_to_jax([[[pid, 0.0, 5]], []])
    next_state = jit_step(state, actions, mask)
    assert int(next_state.fleet_count) == 1
    assert int(next_state.next_fleet_id) == 1
    assert math.isclose(float(next_state.fleets[0, 4]), 0.0)
