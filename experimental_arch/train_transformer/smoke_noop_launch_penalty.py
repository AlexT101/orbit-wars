from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arch import GalaxyMaskablePolicy
from env import OrbitWarsEnv
from features import decode_action, encode_features, flat_action_mask
from opponents import Opponent
from orbit_wars_engine import OrbitWarsEngine


SEED = 33
ENV_SEED = 999
TOTAL_TIMESTEPS = 1536
# Short-run PPO smoke: this should prove the policy strongly moves toward noop
# and that deterministic action selection stops launching. Full stochastic
# saturation takes longer with the rl_orbit_wars-style entity transformer.
MIN_NOOP_PROB = 0.50
MIN_NOOP_PROB_GAIN = 0.04
REWARD_WEIGHTS = {
    "terminal": 0.0,
    "terminal_time": 0.0,
    "production_income": 0.0,
    "launch_penalty": -9.0,
}


class PassOpponent(Opponent):
    name = "pass"

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> list:
        return []


def launch_gate_probs(model: MaskablePPO, seed: int) -> tuple[list[tuple[int, float, float]], int]:
    engine = OrbitWarsEngine(num_players=2, reward_weights=REWARD_WEIGHTS)
    obs = engine.reset(seed=seed)["observations"][0]
    model_obs, feat = encode_features(obs, player=0)

    owned_sources: list[int] = []
    for source_slot, planet_id in enumerate(feat["planet_ids"]):
        planet = next((p for p in obs["planets"] if int(p[0]) == int(planet_id)), None)
        if planet is not None and int(planet[1]) == 0:
            owned_sources.append(source_slot)

    obs_tensor, _ = model.policy.obs_to_tensor(model_obs)
    masks = th.as_tensor(flat_action_mask(feat).reshape(1, -1), dtype=th.bool)
    dist = model.policy.get_distribution(obs_tensor, action_masks=masks)

    probs: list[tuple[int, float, float]] = []
    for source_slot in owned_sources:
        gate = dist.distributions[2 * source_slot].probs.detach().cpu().numpy().reshape(-1)
        probs.append((source_slot, float(gate[0]), float(gate[1])))

    action, _ = model.predict(model_obs, deterministic=True, action_masks=flat_action_mask(feat))
    deterministic_launches = len(decode_action(feat, np.asarray(action)))
    return probs, deterministic_launches


def main() -> int:
    env = OrbitWarsEnv(
        opponent=PassOpponent(),
        seed=ENV_SEED,
        side_mode="fixed",
        reward_weights=REWARD_WEIGHTS,
    )
    model = MaskablePPO(
        GalaxyMaskablePolicy,
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        learning_rate=1e-3,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=0,
        seed=SEED,
        device="cpu",
    )

    initial_probs = {}
    for probe_seed in (1, ENV_SEED, ENV_SEED + 1):
        probs, deterministic_launches = launch_gate_probs(model, probe_seed)
        initial_probs[probe_seed] = {source_slot: noop_prob for source_slot, noop_prob, _send_prob in probs}
        print(f"initial seed={probe_seed}: {(probs, deterministic_launches)}")

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    failures = []
    for probe_seed in (1, ENV_SEED, ENV_SEED + 1):
        probs, deterministic_launches = launch_gate_probs(model, probe_seed)
        print(f"after seed={probe_seed}: probs={probs} deterministic_launches={deterministic_launches}")
        for source_slot, noop_prob, _send_prob in probs:
            if noop_prob < MIN_NOOP_PROB:
                failures.append((probe_seed, source_slot, noop_prob))
            initial_noop_prob = initial_probs[probe_seed][source_slot]
            if noop_prob - initial_noop_prob < MIN_NOOP_PROB_GAIN:
                failures.append((probe_seed, source_slot, "gain", noop_prob - initial_noop_prob))
        if deterministic_launches:
            failures.append((probe_seed, "deterministic_launches", float(deterministic_launches)))

    if failures:
        print(f"FAIL: noop smoke test failed: {failures}")
        return 1
    print("OK: launch penalty smoke test learned noop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
