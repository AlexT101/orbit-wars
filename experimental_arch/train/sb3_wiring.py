from __future__ import annotations

from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from arch import GalaxyMaskablePolicy


def make_model(env, *, device: str, verbose: int, seed: int | None, **ppo_kwargs: Any) -> MaskablePPO:
    return MaskablePPO(
        GalaxyMaskablePolicy,
        env,
        **ppo_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
    )


def load_or_make_model(
    checkpoint_path: Path,
    env,
    *,
    device: str,
    verbose: int,
    seed: int | None,
    **ppo_kwargs: Any,
) -> tuple[MaskablePPO, bool]:
    if checkpoint_path.exists():
        load_kwargs = dict(ppo_kwargs)
        if load_kwargs.get("policy_kwargs") is None:
            load_kwargs.pop("policy_kwargs")
        model = MaskablePPO.load(
            checkpoint_path,
            env=env,
            device=device,
            seed=seed,
            verbose=verbose,
            **load_kwargs,
        )
        return model, True

    model = make_model(
        env,
        device=device,
        verbose=verbose,
        seed=seed,
        **ppo_kwargs,
    )
    return model, False
