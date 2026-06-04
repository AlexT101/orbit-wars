from __future__ import annotations

import re
import shutil
from pathlib import Path


SNAPSHOT_RE = re.compile(r"opponent_gen_(\d+)\.zip$")


def generation_path(self_play_dir: Path, generation: int) -> Path:
    return self_play_dir / f"opponent_gen_{generation:06d}.zip"


def snapshot_generation(path: Path) -> int | None:
    match = SNAPSHOT_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_snapshot(self_play_dir: Path) -> tuple[int, Path] | None:
    if not self_play_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in self_play_dir.glob("opponent_gen_*.zip"):
        generation = snapshot_generation(path)
        if generation is not None:
            candidates.append((generation, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def write_current_snapshot(pointer_path: Path, snapshot_path: Path) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(snapshot_path.resolve()) + "\n", encoding="utf-8")


def save_model_snapshot(model, self_play_dir: Path, pointer_path: Path, generation: int) -> tuple[int, Path]:
    path = generation_path(self_play_dir, generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    write_current_snapshot(pointer_path, path)
    return generation, path


def read_current_snapshot(pointer_path: Path, self_play_dir: Path) -> tuple[int, Path] | None:
    if pointer_path.exists():
        raw = pointer_path.read_text(encoding="utf-8").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = pointer_path.parent / path
            generation = snapshot_generation(path)
            if generation is not None and path.exists():
                return generation, path
    return latest_snapshot(self_play_dir)


def bootstrap_snapshot_from_legacy(
    *,
    legacy_checkpoint: Path,
    self_play_dir: Path,
    pointer_path: Path,
) -> tuple[int, Path] | None:
    current = read_current_snapshot(pointer_path, self_play_dir)
    if current is not None or not legacy_checkpoint.exists():
        return current

    path = generation_path(self_play_dir, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        shutil.copy2(legacy_checkpoint, path)
    write_current_snapshot(pointer_path, path)
    return 0, path
