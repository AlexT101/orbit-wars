"""Self-play wrapper around Kaggle's `orbit_wars` environment.

We drive the env step-by-step instead of `env.run([...])` so the trainer can
record (obs, action, logprob, value) tuples for the learning agent. Both
agents see real observations from Kaggle's interpreter."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable

with open(os.devnull, "w") as _devnull:
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        from kaggle_environments import make as kg_make  # type: ignore[import-not-found]


PLAYER_ME = 0
PLAYER_OPP = 1


def _observation_for(env_obs, player: int) -> dict:
    """Pull a per-player observation dict out of `env.state[i].observation`."""
    raw = env_obs.observation
    # `Observation` instances behave like dicts but also like namespaces.
    out = dict(raw)
    out["player"] = player
    return out


def run_episode(
    learner_agent: Callable[[dict], tuple[list, object]],
    opponent_agent: Callable[[dict], list],
    *,
    seed: int | None = None,
    swap_sides: bool = False,
) -> tuple[list, float, int]:
    """Play one 2-player game.

    `learner_agent(obs_dict)` must return `(moves, extras)` where `extras` is
    whatever the trainer wants to stash for this step (typically the
    `ActionRecord` plus value estimate). The wrapper threads `extras` back
    out in a list aligned with the steps the learner acted on.

    Returns `(extras_per_step, final_reward, num_steps)`. `final_reward` is
    +1 / 0 / -1 for the learner. `swap_sides` puts the learner on player 1
    instead of player 0 (used to balance training)."""

    config = {} if seed is None else {"seed": seed}
    env = kg_make("orbit_wars", configuration=config, debug=False)
    env.reset(2)

    learner_player = PLAYER_OPP if swap_sides else PLAYER_ME
    opp_player = 1 - learner_player

    extras_list: list = []
    steps = 0
    while not env.done:
        state = env.state
        obs_learner = _observation_for(state[learner_player], learner_player)
        obs_opp = _observation_for(state[opp_player], opp_player)

        moves_learner, extras = learner_agent(obs_learner)
        moves_opp = opponent_agent(obs_opp)

        actions: list[Any] = [[], []]
        actions[learner_player] = moves_learner
        actions[opp_player] = moves_opp
        env.step(actions)
        extras_list.append(extras)
        steps += 1

    final = env.state
    reward = final[learner_player].reward
    reward = 0.0 if reward is None else float(reward)
    return extras_list, reward, steps
