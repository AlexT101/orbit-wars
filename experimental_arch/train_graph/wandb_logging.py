from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import wandb
from stable_baselines3.common.logger import KVWriter, configure


class WandbKVWriter(KVWriter):
    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        if wandb.run is None:
            return
        payload: dict[str, int | float] = {"train/total_timesteps": step}
        for key, value in key_values.items():
            if "wandb" in key_excluded.get(key, ()):
                continue
            scalar = self._as_scalar(value)
            if scalar is not None:
                payload[key] = scalar
        if len(payload) > 1:
            wandb.log(payload, step=step)

    def close(self) -> None:
        pass

    @staticmethod
    def _as_scalar(value: Any) -> int | float | None:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return None


def configure_wandb_sb3_logger(log_dir: Path):
    sb3_logger = configure(folder=str(log_dir), format_strings=["stdout", "csv"])
    sb3_logger.output_formats.append(WandbKVWriter())
    return sb3_logger
