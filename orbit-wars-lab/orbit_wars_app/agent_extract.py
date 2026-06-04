"""Extract Kaggle-style `submission.tar.gz` agents into a runnable folder.

Kaggle competitions accept submissions as a tarball containing `main.py` at the
root alongside any helpers / weights / native modules. Local users want to drop
that same artifact under `agents/mine/<name>/submission.tar.gz` and have it
work — same on-disk format they uploaded to Kaggle, no manual extract step.

This module handles the extraction with three guarantees:
- Safe: rejects tar members that would escape the destination dir (tar-slip).
- Cached: re-uses the extracted tree until the tarball's mtime changes.
- Bounded: refuses tarballs larger than `MAX_TARBALL_BYTES` (compressed) or
  expanding past `MAX_EXTRACT_BYTES` (uncompressed), to keep a malformed or
  malicious archive from filling the disk.
"""
from __future__ import annotations

import tarfile
from pathlib import Path


SUBMISSION_FILENAME = "submission.tar.gz"
EXTRACT_DIRNAME = ".extracted"
MTIME_MARKER = ".tar-mtime"

# Kaggle's own submission cap is 100 MB compressed. We allow a bit more to
# accommodate dev artifacts (e.g. uncompressed test tarballs) but still bound
# the worst case.
MAX_TARBALL_BYTES = 200 * 1024 * 1024
MAX_EXTRACT_BYTES = 1024 * 1024 * 1024  # 1 GB uncompressed


class TarballError(Exception):
    """Raised when a submission tarball is missing, oversized, or unsafe."""


def ensure_extracted(agent_dir: Path) -> Path:
    """Return the runnable agent dir, preferring loose source over tarballs.

    Returns `agent_dir` unchanged when it already has a loose `main.py`, even
    if a stale or alternate `submission.tar.gz` is present. If there is no
    loose `main.py`, a `submission.tar.gz` is extracted into
    `agent_dir/.extracted/`. A marker file records the source tarball's mtime
    so subsequent calls skip the extract if the tarball is unchanged.
    """
    if (agent_dir / "main.py").is_file():
        return agent_dir

    tarball = agent_dir / SUBMISSION_FILENAME
    if not tarball.is_file():
        return agent_dir

    extract_dir = agent_dir / EXTRACT_DIRNAME
    marker = extract_dir / MTIME_MARKER
    src_mtime = tarball.stat().st_mtime_ns

    if marker.is_file():
        try:
            if int(marker.read_text().strip()) == src_mtime:
                return extract_dir
        except (OSError, ValueError):
            pass  # corrupt marker — fall through and re-extract

    _extract_safely(tarball, extract_dir)
    marker.write_text(str(src_mtime))
    return extract_dir


def _extract_safely(tarball: Path, dest: Path) -> None:
    size = tarball.stat().st_size
    if size > MAX_TARBALL_BYTES:
        raise TarballError(
            f"{tarball.name} is {size} bytes; exceeds limit {MAX_TARBALL_BYTES}"
        )

    # Wipe any previous extract so removed files don't linger.
    if dest.exists():
        _rmtree(dest)
    dest.mkdir(parents=True)

    try:
        with tarfile.open(tarball, mode="r:gz") as tf:
            total = 0
            for member in tf.getmembers():
                _check_member(member, dest)
                total += max(member.size, 0)
                if total > MAX_EXTRACT_BYTES:
                    raise TarballError(
                        f"{tarball.name} expands past {MAX_EXTRACT_BYTES} bytes"
                    )
            # `filter='data'` (Python 3.12+) drops setuid/symlinks-outside/etc.
            # — equivalent to the per-member check above, kept for defence in
            # depth in case a future tarfile bug regresses one of them.
            tf.extractall(dest, filter="data")
    except tarfile.TarError as e:
        raise TarballError(f"{tarball.name}: {e}") from e


def _check_member(member: tarfile.TarInfo, dest: Path) -> None:
    """Reject members whose resolved path escapes `dest` (tar-slip)."""
    target = (dest / member.name).resolve()
    dest_resolved = dest.resolve()
    try:
        target.relative_to(dest_resolved)
    except ValueError:
        raise TarballError(f"unsafe path in tarball: {member.name!r}")
    if member.issym() or member.islnk():
        link_target = (dest / member.name).parent / member.linkname
        try:
            link_target.resolve().relative_to(dest_resolved)
        except ValueError:
            raise TarballError(
                f"symlink escapes archive: {member.name!r} -> {member.linkname!r}"
            )


def _rmtree(path: Path) -> None:
    """Recursive delete that tolerates read-only files on Windows."""
    import shutil
    import stat

    def _onerror(func, p, _exc):
        try:
            Path(p).chmod(stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerror)