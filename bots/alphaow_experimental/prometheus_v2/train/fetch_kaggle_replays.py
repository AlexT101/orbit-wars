"""Fetch Orbit Wars replay JSONs for one or more Kaggle submissions.

Raw replays are saved outside the repo by default:
  ~/.cache/orbit-wars/prometheus/replays/<tag>/raw/*.json

Requires the Kaggle CLI to be installed and authenticated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_OUT_ROOT = Path.home() / ".cache" / "orbit-wars" / "prometheus" / "replays"
KAGGLE_CMD: list[str] | None = None
KAGGLE_ENV: dict[str, str] | None = None


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def kaggle_cmd() -> list[str]:
    global KAGGLE_CMD
    if KAGGLE_CMD is not None:
        return KAGGLE_CMD
    exe = shutil.which("kaggle")
    if exe:
        KAGGLE_CMD = [exe]
        return KAGGLE_CMD
    try:
        run([sys.executable, "-m", "kaggle", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "Kaggle CLI was not found.\n"
            f"Python: {sys.executable}\n"
            "Install it in this environment with:\n"
            "  python3 -m pip install kaggle\n"
            "Then authenticate with one of:\n"
            "  kaggle auth login\n"
            "  or place your token at ~/.kaggle/kaggle.json / ~/.kaggle/access_token"
        ) from exc
    KAGGLE_CMD = [sys.executable, "-m", "kaggle"]
    return KAGGLE_CMD


def kaggle_env() -> dict[str, str] | None:
    """Return an env that bridges OAuth credentials to API-token auth.

    Newer Kaggle CLI `auth login` stores OAuth credentials in
    ~/.kaggle/credentials.json, while some competition endpoints still ask for
    KAGGLE_API_TOKEN / ~/.kaggle/access_token. We avoid printing or persisting
    the token; it is passed only to child processes.
    """
    global KAGGLE_ENV
    if KAGGLE_ENV is not None:
        return KAGGLE_ENV
    if os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "access_token").exists():
        KAGGLE_ENV = None
        return KAGGLE_ENV
    if not (Path.home() / ".kaggle" / "credentials.json").exists():
        KAGGLE_ENV = None
        return KAGGLE_ENV
    proc = run(kaggle_cmd() + ["auth", "print-access-token"])
    token = proc.stdout.strip()
    if not token:
        KAGGLE_ENV = None
        return KAGGLE_ENV
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = token
    KAGGLE_ENV = env
    return KAGGLE_ENV


def kaggle(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run(kaggle_cmd() + args, env=kaggle_env())


def format_kaggle_error(exc: subprocess.CalledProcessError) -> str:
    cmd = " ".join(str(x) for x in exc.cmd)
    stderr = (exc.stderr or "").strip()
    stdout = (exc.stdout or "").strip()
    parts = [f"command failed ({exc.returncode}): {cmd}"]
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    return "\n".join(parts)


def episode_ids_for_submission(submission_id: str) -> list[str]:
    proc = kaggle(["competitions", "episodes", submission_id, "-v"])
    text = proc.stdout.strip()
    if not text:
        return []

    ids: list[str] = []
    try:
        rows = list(csv.DictReader(text.splitlines()))
        for row in rows:
            for key in ("EpisodeId", "episodeId", "episode_id", "Id", "id"):
                val = (row.get(key) or "").strip()
                if val.isdigit():
                    ids.append(val)
                    break
    except csv.Error:
        pass

    if not ids:
        # Fallback for non-CSV CLI output. Episode IDs are usually 7+ digits.
        ids = re.findall(r"\b\d{7,}\b", text)

    seen = set()
    out = []
    for eid in ids:
        if eid not in seen and eid != str(submission_id):
            seen.add(eid)
            out.append(eid)
    return out


def existing_replay(raw_dir: Path, episode_id: str) -> Path | None:
    direct = raw_dir / f"{episode_id}.json"
    if direct.exists():
        return direct
    matches = sorted(raw_dir.glob(f"*{episode_id}*.json"))
    return matches[0] if matches else None


def fetch_replay(raw_dir: Path, episode_id: str, force: bool) -> Path | None:
    if not force:
        existing = existing_replay(raw_dir, episode_id)
        if existing is not None:
            return existing
    before = {p.resolve() for p in raw_dir.glob("*.json")}
    kaggle(["competitions", "replay", episode_id, "-p", str(raw_dir)])
    after = {p.resolve() for p in raw_dir.glob("*.json")}
    created = sorted(after - before)
    existing = existing_replay(raw_dir, episode_id)
    if existing is not None:
        return existing
    return Path(created[-1]) if created else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-id", action="append", required=True, help="Kaggle submission id; repeatable")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--tag", default=None, help="run label; default current timestamp")
    p.add_argument("--limit", type=int, default=None, help="max episodes per submission")
    p.add_argument("--force", action="store_true", help="redownload existing replay JSONs")
    args = p.parse_args()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root.expanduser() / tag
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "submission_ids": args.submission_id,
        "episodes": [],
    }

    for sid in args.submission_id:
        try:
            ids = episode_ids_for_submission(sid)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                format_kaggle_error(exc)
                + "\n\n"
                "Things to check:\n"
                "  1. You are authenticated: python3 -m kaggle auth login\n"
                "  2. The submission id is from: kaggle competitions submissions orbit-wars\n"
                "  3. The submission has completed and has episodes available."
            ) from exc
        if args.limit is not None:
            ids = ids[: args.limit]
        print(f"submission {sid}: {len(ids)} episode(s)")
        for i, eid in enumerate(ids, 1):
            try:
                path = fetch_replay(raw_dir, eid, args.force)
            except subprocess.CalledProcessError as exc:
                print(f"  [{i}/{len(ids)}] {eid}: fetch failed:\n{format_kaggle_error(exc)}", file=sys.stderr)
                continue
            if path is None:
                print(f"  [{i}/{len(ids)}] {eid}: no JSON written", file=sys.stderr)
                continue
            print(f"  [{i}/{len(ids)}] {eid}: {path}")
            manifest["episodes"].append({"submission_id": sid, "episode_id": eid, "path": str(path)})

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    print(f"raw replay dir: {raw_dir}")


if __name__ == "__main__":
    main()
