"""Combined parity test suite.

Bundles the four parity test groups that previously lived in separate
modules:

  * TestSelfParity       — KaggleEngine vs KaggleEngine (harness sanity).
  * TestRustResetParity  — reset-only path (max_steps=0).
  * TestRustStepParity   — turn pipeline before comet spawn at step 50.
  * TestRustCometParity  — crosses the first comet spawn window at step 50.

Run: python -m unittest parity.test_parity -v
"""

from __future__ import annotations

import unittest

from parity.agents import AGENTS
from parity.harness import run_parity
from parity.kaggle_engine import KaggleEngine

try:
    from parity.candidates.rust import RustEngine
except ImportError:
    RustEngine = None


class TestSelfParity(unittest.TestCase):
    """Self-parity sanity tests: KaggleEngine vs KaggleEngine.

    If this ever fails, the harness itself is broken (or Kaggle's interpreter
    is non-deterministic given a fixed seed — which would also be news).
    """

    def _run(self, agent_name: str, num_players: int, seed: int, steps: int) -> None:
        result = run_parity(
            engine_a=KaggleEngine(),
            engine_b=KaggleEngine(),
            seed=seed,
            num_players=num_players,
            agents=[AGENTS[agent_name]] * num_players,
            max_steps=steps,
            atol=0.0,
        )
        self.assertTrue(
            result.converged,
            f"Kaggle-vs-Kaggle diverged at step {result.first_divergence_step}:\n"
            f"{result.summary()}",
        )

    def test_noop_2p_seed42_50_steps(self):
        self._run("noop", 2, seed=42, steps=50)

    def test_noop_4p_seed42_50_steps(self):
        self._run("noop", 4, seed=42, steps=50)

    def test_random_2p_seed7_200_steps(self):
        self._run("random", 2, seed=7, steps=200)

    def test_random_4p_seed123_200_steps(self):
        self._run("random", 4, seed=123, steps=200)

    def test_aggressive_2p_past_first_comet_spawn(self):
        # Step 50 is the first comet spawn; make sure the harness handles it.
        self._run("aggressive", 2, seed=1, steps=80)

    def test_aggressive_2p_full_episode(self):
        self._run("aggressive", 2, seed=2024, steps=500)

    def test_nearest_sniper_2p_seed42_200_steps(self):
        self._run("nearest_sniper", 2, seed=42, steps=200)

    def test_nearest_sniper_4p_seed123_200_steps(self):
        self._run("nearest_sniper", 4, seed=123, steps=200)


@unittest.skipIf(RustEngine is None, "Rust extension is not available")
class TestRustResetParity(unittest.TestCase):
    """Reset-only parity checks for the Rust engine candidate.

    These tests intentionally stop at `max_steps=0`, which exercises the full
    initialization path without touching the not-yet-ported step pipeline.
    """

    def _run(self, seed: int, num_players: int) -> None:
        result = run_parity(
            engine_a=KaggleEngine(),
            engine_b=RustEngine(),
            seed=seed,
            num_players=num_players,
            agents=[AGENTS["noop"]] * num_players,
            max_steps=0,
            atol=0.0,
        )
        self.assertTrue(
            result.converged,
            f"Rust reset diverged at step {result.first_divergence_step}",
        )

    def test_reset_2p_seed_1(self):
        self._run(seed=1, num_players=2)

    def test_reset_2p_seed_2024(self):
        self._run(seed=2024, num_players=2)

    def test_reset_4p_seed_7(self):
        self._run(seed=7, num_players=4)

    def test_reset_4p_seed_2024(self):
        self._run(seed=2024, num_players=4)


@unittest.skipIf(RustEngine is None, "Rust extension is not available")
class TestRustStepParity(unittest.TestCase):
    """Pre-comet step parity checks for the Rust engine candidate.

    These cover the turn pipeline before comet spawn at step 50.
    """

    def _run(self, agent_name: str, seed: int, num_players: int, steps: int, atol: float = 0.0) -> None:
        result = run_parity(
            engine_a=KaggleEngine(),
            engine_b=RustEngine(),
            seed=seed,
            num_players=num_players,
            agents=[AGENTS[agent_name]] * num_players,
            max_steps=steps,
            atol=atol,
        )
        self.assertTrue(
            result.converged,
            f"Rust step diverged at step {result.first_divergence_step}",
        )

    def test_noop_2p_20_steps(self):
        self._run("noop", seed=42, num_players=2, steps=20)

    def test_noop_4p_20_steps(self):
        self._run("noop", seed=42, num_players=4, steps=20)

    def test_random_2p_20_steps(self):
        self._run("random", seed=7, num_players=2, steps=20)

    def test_random_4p_20_steps(self):
        self._run("random", seed=123, num_players=4, steps=20)

    def test_aggressive_2p_20_steps(self):
        self._run("aggressive", seed=1, num_players=2, steps=20)

    def test_nearest_sniper_2p_20_steps(self):
        self._run("nearest_sniper", seed=42, num_players=2, steps=20)


@unittest.skipIf(RustEngine is None, "Rust extension is not available")
class TestRustCometParity(unittest.TestCase):
    """Post-comet parity checks for the Rust engine candidate.

    These tests cross the first comet spawn window at step 50.
    """

    def _run(self, agent_name: str, seed: int, num_players: int, steps: int, atol: float = 0.0) -> None:
        result = run_parity(
            engine_a=KaggleEngine(),
            engine_b=RustEngine(),
            seed=seed,
            num_players=num_players,
            agents=[AGENTS[agent_name]] * num_players,
            max_steps=steps,
            atol=atol,
        )
        self.assertTrue(
            result.converged,
            f"Rust comet parity diverged at step {result.first_divergence_step}",
        )

    def test_noop_2p_55_steps(self):
        self._run("noop", seed=42, num_players=2, steps=55)

    def test_noop_4p_55_steps(self):
        self._run("noop", seed=42, num_players=4, steps=55)

    def test_random_2p_80_steps(self):
        self._run("random", seed=7, num_players=2, steps=80)

    def test_random_4p_80_steps(self):
        self._run("random", seed=123, num_players=4, steps=80)

    def test_aggressive_2p_80_steps(self):
        self._run("aggressive", seed=2024, num_players=2, steps=80)

    def test_nearest_sniper_2p_80_steps(self):
        self._run("nearest_sniper", seed=2024, num_players=2, steps=80)


if __name__ == "__main__":
    unittest.main()
