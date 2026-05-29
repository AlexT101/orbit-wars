"""Match runner — fast mode (in-process kaggle-environments), faithful mode
(subprocess+HTTP), and ultrafast mode (native Rust engine, no replay).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional


_DEBUG_PREFIXES = ("[LINE]", "[DOT]", "[TEXT]")

_KAGGLE_LOGGING_SILENCED = False


def silence_kaggle_environments_logging() -> None:
    """Mute kaggle-environments' chatty import-time INFO logging.

    On first import, ``kaggle_environments`` registers every bundled env, which
    loads ``.../envs/open_spiel_env/open_spiel_env.py``. At *module* scope that
    file does (see the installed package):

        _log = logging.getLogger(__name__)   # ...open_spiel_env.open_spiel_env
        _log.setLevel(logging.INFO)          # explicit level, ignores parent
        _log.addHandler(logging.StreamHandler(sys.stdout))  # its own handler

    then logs the full OpenSpiel game registry (~50 lines). Because that child
    logger has an *explicit* level and its *own* stdout handler, raising the
    parent ``kaggle_environments`` logger's level does nothing — the child's
    level wins and its handler bypasses parent propagation. (That's why the
    earlier parent-level approach never worked, and why each spawn-based pebble
    worker re-spammed on its first import.)

    A logger-level filter, unlike ``setLevel``, survives the module's own
    ``setLevel``/``addHandler`` calls and gates records *before* they reach the
    handler — including the one-shot logging that fires during import. So this
    MUST run before the first ``import kaggle_environments`` to kill the startup
    spam. We drop everything below WARNING, keeping genuine warnings/errors.

    Idempotent and per-process (the module flag does not cross the spawn
    boundary, which is exactly what we want — each worker silences itself).
    """
    global _KAGGLE_LOGGING_SILENCED
    if _KAGGLE_LOGGING_SILENCED:
        return
    logging.getLogger(
        "kaggle_environments.envs.open_spiel_env.open_spiel_env"
    ).addFilter(lambda record: record.levelno >= logging.WARNING)
    # Belt-and-suspenders for any other submodule that relies on propagation.
    logging.getLogger("kaggle_environments").setLevel(logging.WARNING)
    _KAGGLE_LOGGING_SILENCED = True


def _parse_debug_lines(text: str, *, player: int, step: int) -> list[dict]:
    """Parse a chunk of agent stdout into debug message dicts.

    Each non-empty line becomes one message tagged with `player` and `step`.
    Lines starting with `[LINE]`/`[DOT]`/`[TEXT]` are classified by `kind`;
    everything else is treated as a free-form log line (`kind="log"`).
    """
    out: list[dict] = []
    if not text:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip optional `[P<n>]` prefix some agents emit when manually
        # tagging their own output. We already know the player from the
        # env.logs structure, so this is just a courtesy strip.
        if line.startswith("[P") and "]" in line:
            close = line.index("]")
            inner = line[2:close]
            if inner.isdigit():
                line = line[close + 1:].strip()
                if not line:
                    continue
        if line.startswith(_DEBUG_PREFIXES):
            # `[LINE]` -> `line`, `[DOT]` -> `dot`, `[TEXT]` -> `text`.
            kind = line.split(maxsplit=1)[0][1:-1].lower()
            out.append({"kind": kind, "raw": line, "player": player, "step": step})
        else:
            out.append({"kind": "log", "raw": line, "player": player, "step": step})
    return out


def _attach_debug_output(
    replay: dict, env_logs: list, num_agents: int
) -> dict:
    """Walk `env.logs[step][agent_idx]` and attach parsed debug messages.

    kaggle-environments captures each agent's stdout chunk for the step in
    `entry["stdout"]`; we split those into lines and tag each with the player
    + step from the log structure. The viewer reads
    `replay["debug"]["messages"]` to overlay [LINE]/[DOT] indicators and
    per-team log panels.
    """
    if not env_logs:
        return replay
    messages: list[dict] = []
    for step_idx, step_entries in enumerate(env_logs):
        if not isinstance(step_entries, list):
            continue
        for agent_idx, entry in enumerate(step_entries):
            if agent_idx >= num_agents or not isinstance(entry, dict):
                continue
            chunk = entry.get("stdout")
            if not chunk:
                continue
            text = chunk if isinstance(chunk, str) else str(chunk)
            messages.extend(_parse_debug_lines(text, player=agent_idx, step=step_idx))
    if messages:
        replay["debug"] = {"messages": messages}
    return replay


def _attach_durations(replay: dict, env_logs: list, num_agents: int) -> dict:
    """Attach per-step per-agent wall-clock durations as `replay["durations"]`.

    Shape: `durations[step][agent]` of seconds (or `None` when missing).
    The viewer reads this to show how long each agent took per turn.
    """
    if not env_logs:
        return replay
    durations: list[list[float | None]] = []
    for step_entries in env_logs:
        row: list[float | None] = [None] * num_agents
        if isinstance(step_entries, list):
            for agent_idx, entry in enumerate(step_entries):
                if agent_idx >= num_agents:
                    continue
                if isinstance(entry, dict):
                    d = entry.get("duration")
                    if isinstance(d, (int, float)):
                        row[agent_idx] = float(d)
        durations.append(row)
    replay["durations"] = durations
    return replay


@dataclass
class MatchOutcome:
    agent_ids: list[str]
    winner: Optional[str]           # agent_id or None (draw / error)
    scores: list[int]               # final ship sum per player (planets + fleets)
    turns: int
    duration_s: float
    seed: int = 0                   # logged for audit; engine currently ignores it
    status: Literal["ok", "timeout", "crashed", "agent_failed_to_start", "invalid_action", "draw"] = "ok"
    # agent_failed_to_start is reserved for faithful mode (Task 9)
    replay: dict = field(default_factory=dict)
    # Per-agent per-turn wallclock samples (seconds). Index aligns with agent_ids.
    # Empty inner list = agent never took a turn (match crashed before its move).
    # Sourced from kaggle-environments' own Agent.act() timing (core.py logs),
    # so this is the same number Kaggle would deadline-check against.
    per_agent_turn_seconds: list[list[float]] = field(default_factory=list)


def _crashed_replay_skeleton(error: str) -> dict:
    """Produce a replay dict with the same top-level keys as env.toJSON()
    so downstream code (save_replay, viewer) doesn't KeyError on missing keys."""
    return {"error": error, "steps": [], "rewards": [], "statuses": []}


def _per_agent_durations_from_logs(env_logs: list, num_agents: int) -> list[list[float]]:
    """Extract per-agent per-turn durations from `env.logs`.

    Shape of env.logs: list[step][agent_idx] -> dict with at least one of
    {"duration", "stdout", "stderr"}. The initial reset step has no
    "duration" key (no agent has acted yet) — those are skipped.
    """
    out: list[list[float]] = [[] for _ in range(num_agents)]
    for step in env_logs or []:
        if not isinstance(step, list):
            continue
        for idx, entry in enumerate(step):
            if idx >= num_agents:
                continue
            if isinstance(entry, dict):
                d = entry.get("duration")
                if isinstance(d, (int, float)):
                    out[idx].append(float(d))
    return out


def _per_agent_streams_from_logs(
    env_logs: list, num_agents: int
) -> tuple[list[list[str]], list[list[str]]]:
    """Extract per-agent stdout/stderr chunks from `env.logs` (fast mode)."""
    outs: list[list[str]] = [[] for _ in range(num_agents)]
    errs: list[list[str]] = [[] for _ in range(num_agents)]
    for step in env_logs or []:
        if not isinstance(step, list):
            continue
        for idx, entry in enumerate(step):
            if idx >= num_agents or not isinstance(entry, dict):
                continue
            so = entry.get("stdout")
            if so:
                outs[idx].append(so if isinstance(so, str) else str(so))
            se = entry.get("stderr")
            if se:
                errs[idx].append(se if isinstance(se, str) else str(se))
    return outs, errs


def _write_agent_logs(
    log_dir: Optional[Path],
    log_prefix: str,
    agent_ids: list[str],
    stdouts: list[list[str]],
    stderrs: list[list[str]],
) -> None:
    """Persist per-agent stdout/stderr to <log_dir>/<prefix>-<safe_id>.{stdout,stderr}.log.

    Empty streams are skipped — no zero-byte files. Failures are swallowed so
    one bad disk write can't fail the match.
    """
    if log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    prefix = f"{log_prefix}-" if log_prefix else ""
    for aid, so_chunks, se_chunks in zip(agent_ids, stdouts, stderrs):
        safe = aid.replace("/", "_")
        for ext, chunks in (("stdout", so_chunks), ("stderr", se_chunks)):
            if not chunks:
                continue
            text = "".join(chunks)
            if not text.endswith("\n"):
                text += "\n"
            try:
                (log_dir / f"{prefix}{safe}.{ext}.log").write_text(
                    text, encoding="utf-8", errors="replace"
                )
            except OSError:
                pass


def run_match_fast(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Run a single match in fast mode (kaggle-envs in-process).

    `agent_ids` order = player order (index 0 = player 0 = Q1 home).
    `agent_paths` must correspond 1:1 to agent_ids.

    `seed` is stored in the outcome for audit; kaggle-environments engine
    currently ignores the seed internally (per postmortem 2026-04-20).
    """
    if len(agent_ids) != len(agent_paths):
        raise ValueError(
            f"agent_ids and agent_paths length mismatch: "
            f"{len(agent_ids)} vs {len(agent_paths)}"
        )
    import contextlib
    import io as _io

    silence_kaggle_environments_logging()
    from kaggle_environments import make

    from .agent_extract import ensure_extracted
    from .agent_serve import load_agent, _count_args

    env = make("orbit_wars", debug=False)

    # `submission.tar.gz` agents are extracted to their cached dir; loose
    # source agents are pass-through.
    resolved_paths = [ensure_extracted(p) for p in agent_paths]

    # Load each agent function and wrap it so we can capture its per-turn
    # stdout in a side channel. kaggle-envs does not redirect agent stdout
    # into `env.logs` when agents are loaded via file paths, so we do the
    # capture ourselves. Each agent's stdout is appended to
    # `per_step_stdout[agent_idx]` once per call; the index aligns with the
    # turn number from kaggle-envs' perspective.
    per_step_stdout: list[list[str]] = [[] for _ in agent_ids]
    wrapped_agents = []
    for idx, path in enumerate(resolved_paths):
        agent_fn = load_agent(str(path))
        if agent_fn is None:
            raise RuntimeError(f"No callable found in {path}/main.py")
        argcount = _count_args(agent_fn)

        def make_wrapped(fn, player_idx, count):
            def wrapped(obs, config=None):
                buf = _io.StringIO()
                with contextlib.redirect_stdout(buf):
                    args = [obs, config][:count] if count >= 1 else []
                    res = fn(*args)
                per_step_stdout[player_idx].append(buf.getvalue())
                return res
            return wrapped

        wrapped_agents.append(make_wrapped(agent_fn, idx, argcount))

    start = time.monotonic()
    try:
        env.run(wrapped_agents)
    except Exception as e:
        duration = time.monotonic() - start
        env_logs = getattr(env, "logs", []) or []
        partial_timings = _per_agent_durations_from_logs(env_logs, len(agent_ids))
        outs, errs = _per_agent_streams_from_logs(env_logs, len(agent_ids))
        _write_agent_logs(log_dir, log_prefix, agent_ids, outs, errs)
        return MatchOutcome(
            agent_ids=agent_ids,
            winner=None,
            scores=[],
            turns=0,
            duration_s=duration,
            seed=seed,
            status="crashed",
            replay=_crashed_replay_skeleton(str(e)),
            per_agent_turn_seconds=partial_timings,
        )
    duration = time.monotonic() - start
    replay = env.toJSON()
    env_logs = env.logs or []
    replay = _attach_debug_output_from_capture(
        replay, per_step_stdout, len(agent_ids), env_logs
    )
    replay = _attach_durations(replay, env_logs, len(agent_ids))
    winner, scores, turns, status = _extract_outcome(replay, agent_ids)
    per_agent_timings = _per_agent_durations_from_logs(env_logs, len(agent_ids))
    # Build outs from per_step_stdout (joined) for log-file persistence so the
    # log writer keeps working even without env.logs stdout.
    outs = [["".join(chunks)] for chunks in per_step_stdout]
    _, errs = _per_agent_streams_from_logs(env_logs, len(agent_ids))
    _write_agent_logs(log_dir, log_prefix, agent_ids, outs, errs)
    return MatchOutcome(
        agent_ids=agent_ids,
        winner=winner,
        scores=scores,
        turns=turns,
        duration_s=duration,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        replay=replay,
        per_agent_turn_seconds=per_agent_timings,
    )


def _attach_debug_output_from_capture(
    replay: dict,
    per_step_stdout: list[list[str]],
    num_agents: int,
    env_logs: list,
) -> dict:
    """Build `replay["debug"]["messages"]` from the per-call stdout captures.

    `per_step_stdout[agent_idx]` is a list of stdout chunks — one per
    invocation of that agent's function. The call index aligns with the
    *observed* step the agent saw — i.e. the kaggle replay step the
    playback bar is on when the agent makes that decision. So call 0
    corresponds to step 0 (initial state — the agent is choosing its
    first action). The action *result* shows up at step 1.

    This convention matches the lab-side reference renderer (and apollo2's
    own `self.current_turn`, which is incremented after each call). Viewer
    step S then aligns with debug messages whose step == S, which in turn
    aligns with the agent's `=== turn S ===` text marker.

    Falls back to scanning `env_logs` for any extra stdout the engine may
    have captured itself (e.g. in faithful mode).
    """
    messages: list[dict] = []
    for agent_idx, chunks in enumerate(per_step_stdout):
        if agent_idx >= num_agents:
            continue
        for call_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            messages.extend(
                _parse_debug_lines(chunk, player=agent_idx, step=call_idx)
            )
    # Also harvest any stdout kaggle did capture (faithful mode subprocess
    # entries, etc.). env_logs[N] is the log row for the transition into
    # state N — i.e. populated by the agents' (N-1)-th call. To keep this
    # aligned with the call-indexed messages above (call 0 -> step 0), we
    # subtract 1 here. env_logs[0] is the initial reset row with no agent
    # output, so the off-by-one never produces a negative step in practice.
    if env_logs:
        for log_idx, step_entries in enumerate(env_logs):
            if not isinstance(step_entries, list):
                continue
            step_idx = max(0, log_idx - 1)
            for agent_idx, entry in enumerate(step_entries):
                if agent_idx >= num_agents or not isinstance(entry, dict):
                    continue
                chunk = entry.get("stdout") or ""
                if not chunk:
                    continue
                text = chunk if isinstance(chunk, str) else str(chunk)
                messages.extend(
                    _parse_debug_lines(text, player=agent_idx, step=step_idx)
                )
    if messages:
        replay["debug"] = {"messages": messages}
    return replay


def _extract_outcome(
    replay: dict, agent_ids: list[str]
) -> tuple[Optional[str], list[int], int, str]:
    """Parse terminal state: winner, per-player scores, turn count, status."""
    steps = replay.get("steps") or []
    if not steps:
        return None, [], 0, "crashed"
    final_step = steps[-1]
    if not final_step:
        return None, [], 0, "crashed"

    num_players = len(agent_ids)
    rewards = [s.get("reward") for s in final_step]

    # Extract scores from last observation in state[0]
    state0 = final_step[0]
    obs = state0.get("observation", {})
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])

    scores = [0] * num_players
    for p in planets:
        owner = p[1] if len(p) > 1 else -1
        ships = p[5] if len(p) > 5 else 0
        if 0 <= owner < num_players:
            scores[owner] += int(ships)
    for f in fleets:
        owner = f[1] if len(f) > 1 else -1
        ships = f[6] if len(f) > 6 else 0
        if 0 <= owner < num_players:
            scores[owner] += int(ships)

    # Winner: exactly one reward == 1
    winners_idx = [i for i, r in enumerate(rewards) if r == 1]
    if len(winners_idx) == 1:
        winner = agent_ids[winners_idx[0]]
    else:
        winner = None

    # Status based on any agent's final status
    statuses = [s.get("status") for s in final_step]
    if "ERROR" in statuses:
        status = "crashed"
    elif "TIMEOUT" in statuses:
        status = "timeout"
    elif "INVALID" in statuses:
        status = "invalid_action"
    elif winner is None:
        status = "draw"
    else:
        status = "ok"

    turns = len(steps)
    return winner, scores, turns, status


def run_match(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    mode: Literal["fast", "faithful", "ultrafast"] = "fast",
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Dispatcher: fast (in-process kaggle-envs), faithful (subprocess+HTTP),
    or ultrafast (native Rust engine, no replay).

    If `log_dir` is set, per-agent stdout/stderr is written to
    `<log_dir>/<log_prefix>-<safe_agent_id>.{stdout,stderr}.log`. Ultrafast
    mode does not capture stdout/stderr.
    """
    if mode == "fast":
        return run_match_fast(
            agent_ids, agent_paths, seed=seed,
            log_dir=log_dir, log_prefix=log_prefix,
        )
    if mode == "ultrafast":
        return run_match_ultrafast(
            agent_ids, agent_paths, seed=seed,
            log_dir=log_dir, log_prefix=log_prefix,
        )
    return run_match_faithful(
        agent_ids, agent_paths, seed=seed,
        log_dir=log_dir, log_prefix=log_prefix,
    )


# --- ultrafast mode (native Rust engine, no replay) -----------------------


_RUST_CORE_CLS: Optional[type] = None


def _load_rust_engine_core() -> type:
    """Import `orbit_wars_rust.RustEngineCore`, falling back to the cargo
    `target/release/` build if the module isn't installed in the venv.

    Mirrors engine_parity_checker/candidates/rust.py — kept inline so this
    module has no dependency on the parity-checker package.
    """
    global _RUST_CORE_CLS
    if _RUST_CORE_CLS is not None:
        return _RUST_CORE_CLS

    module_name = "orbit_wars_rust"
    try:
        import importlib
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[2]
        release_dir = repo_root / "rust_engine" / "target" / "release"
        # Cargo's cdylib output name varies by platform:
        #   Windows: orbit_wars_rust.dll  (must be renamed to .pyd to import)
        #   Linux:   liborbit_wars_rust.so
        #   macOS:   liborbit_wars_rust.dylib
        # Pre-built .pyd from `maturin build` works as-is.
        dll = release_dir / "orbit_wars_rust.dll"
        pyd = release_dir / "orbit_wars_rust.pyd"
        so = release_dir / "liborbit_wars_rust.so"
        dylib = release_dir / "liborbit_wars_rust.dylib"
        load_path: Optional[Path] = None
        if pyd.exists():
            load_path = pyd
        elif dll.exists() or so.exists() or dylib.exists():
            src = dll if dll.exists() else (so if so.exists() else dylib)
            ext = ".pyd" if src.suffix == ".dll" else src.suffix
            tmp = Path(tempfile.gettempdir()) / "orbit_wars_rust"
            tmp.mkdir(parents=True, exist_ok=True)
            target = tmp / f"orbit_wars_rust_{os.getpid()}{ext}"
            shutil.copyfile(src, target)
            load_path = target
        if load_path is None:
            raise ImportError(
                "Could not import 'orbit_wars_rust'. Build with "
                "`maturin develop --release` in rust_engine/, or rebuild the "
                "Docker image (the rust-build stage produces the wheel)."
            ) from exc
        spec = importlib.util.spec_from_file_location(module_name, load_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

    _RUST_CORE_CLS = mod.RustEngineCore
    return _RUST_CORE_CLS


def _load_agent_callable(agent_path: Path) -> tuple[Callable, int]:
    """Import `main.py` under a unique module name and return (agent, arity).

    Arity is the number of positional args the agent accepts (1 or 2 —
    kaggle's convention is `agent(obs)` or `agent(obs, config)`).
    """
    from .agent_extract import ensure_extracted

    resolved = ensure_extracted(agent_path)
    main_py = resolved / "main.py"
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    mod_name = f"ow_ultrafast_agent_{digest}"

    spec = importlib.util.spec_from_file_location(
        mod_name, main_py, submodule_search_locations=[str(resolved)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {main_py}")
    mod = importlib.util.module_from_spec(spec)
    # Stash under the unique name so co-located helpers (`from foo import …`)
    # in the same agent dir don't collide across agents in this process.
    sys.modules[mod_name] = mod
    # Add agent dir to sys.path briefly so relative imports inside main.py
    # resolve. Removed after exec so different agents don't see each other.
    path_entry = str(resolved)
    sys.path.insert(0, path_entry)
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(path_entry)
        except ValueError:
            pass

    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise AttributeError(f"{main_py} does not define a callable `agent`")

    try:
        sig = inspect.signature(fn)
        # Count positional-capable params (skip *args/**kwargs).
        arity = sum(
            1
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (TypeError, ValueError):
        arity = 1
    return fn, max(1, min(2, arity))


def _scores_from_snapshot(snap: dict, num_players: int) -> list[int]:
    scores = [0] * num_players
    for p in snap.get("planets", []):
        owner = int(p[1]) if len(p) > 1 else -1
        ships = int(p[5]) if len(p) > 5 else 0
        if 0 <= owner < num_players:
            scores[owner] += ships
    for f in snap.get("fleets", []):
        owner = int(f[1]) if len(f) > 1 else -1
        ships = int(f[6]) if len(f) > 6 else 0
        if 0 <= owner < num_players:
            scores[owner] += ships
    return scores


def _winner_from_rewards(
    rewards: Optional[list], agent_ids: list[str]
) -> Optional[str]:
    if not rewards:
        return None
    winners = [i for i, r in enumerate(rewards) if r == 1 or r == 1.0]
    if len(winners) == 1:
        return agent_ids[winners[0]]
    return None


def _write_ultrafast_error(
    log_dir: Optional[Path], log_prefix: str, msg: str
) -> None:
    """Persist the crash reason for an ultrafast match. Ultrafast skips replay
    files, so without this the failure reason has no surfacing path and the
    UI just reports a bare `crashed` status. Failure to write is swallowed —
    a busted log shouldn't take down the tournament."""
    if log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{log_prefix}-" if log_prefix else ""
        (log_dir / f"{prefix}ultrafast-error.log").write_text(
            msg + ("\n" if not msg.endswith("\n") else ""), encoding="utf-8"
        )
    except OSError:
        pass


def run_match_ultrafast(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Run a match against the native Rust engine, in-process.

    No replay is produced (intended for tournament throughput). Per-turn
    timings are captured for each agent the same way fast/faithful modes
    report them, so deadline analysis still works.

    Crash reasons are written to `<log_dir>/<log_prefix>-ultrafast-error.log`
    when provided, since ultrafast can't surface them via the replay file.
    """
    import traceback
    if len(agent_ids) != len(agent_paths):
        raise ValueError(
            f"agent_ids and agent_paths length mismatch: "
            f"{len(agent_ids)} vs {len(agent_paths)}"
        )

    num_players = len(agent_ids)
    per_agent_timings: list[list[float]] = [[] for _ in range(num_players)]

    # Load engine + agents up front so we can report agent_failed_to_start
    # symmetrically with faithful mode.
    try:
        core_cls = _load_rust_engine_core()
    except Exception as e:
        msg = f"rust engine load failed: {e}\n{traceback.format_exc()}"
        _write_ultrafast_error(log_dir, log_prefix, msg)
        return MatchOutcome(
            agent_ids=agent_ids, winner=None, scores=[], turns=0,
            duration_s=0.0, seed=seed, status="crashed",
            replay=_crashed_replay_skeleton(msg),
            per_agent_turn_seconds=per_agent_timings,
        )

    agent_fns: list[Callable] = []
    agent_arities: list[int] = []
    for aid, apath in zip(agent_ids, agent_paths):
        try:
            fn, arity = _load_agent_callable(apath)
        except Exception as e:
            msg = f"agent {aid} failed to load: {e}\n{traceback.format_exc()}"
            _write_ultrafast_error(log_dir, log_prefix, msg)
            return MatchOutcome(
                agent_ids=agent_ids, winner=None, scores=[], turns=0,
                duration_s=0.0, seed=seed, status="agent_failed_to_start",
                replay=_crashed_replay_skeleton(msg),
                per_agent_turn_seconds=per_agent_timings,
            )
        agent_fns.append(fn)
        agent_arities.append(arity)

    core = core_cls()
    start = time.monotonic()
    try:
        payload = core.reset(int(seed), num_players, None)
    except Exception as e:
        msg = f"engine reset failed: {e}\n{traceback.format_exc()}"
        _write_ultrafast_error(log_dir, log_prefix, msg)
        return MatchOutcome(
            agent_ids=agent_ids, winner=None, scores=[], turns=0,
            duration_s=time.monotonic() - start, seed=seed, status="crashed",
            replay=_crashed_replay_skeleton(msg),
            per_agent_turn_seconds=per_agent_timings,
        )

    obs_list: list[dict] = list(payload["observations"])
    # Reset always returns a full snapshot — pull configuration once. The
    # per-step snapshot dict is intentionally skipped via step_observations_only
    # in the loop below; we re-fetch the final snapshot via core.snapshot()
    # after the game ends to compute scores/winner.
    init_snap: dict = payload["snapshot"]
    config: dict = init_snap.get("configuration", {}) or {}
    done = bool(init_snap.get("done", False))
    # Prefer the snapshot-skipping step variant when available; fall back to
    # full `step` if the Rust crate predates it.
    step_fn = getattr(core, "step_observations_only", None) or core.step

    turns = 0
    crash_status: Optional[str] = None
    crash_msg = ""

    import contextlib
    import io as _io
    # Captured stdout per agent per turn. Used both to silence ultrafast
    # (otherwise debug-printing agents like apollo2 spam the tournament
    # terminal) and to populate `replay["debug"]["messages"]` on the
    # otherwise-empty ultrafast replay so a viewer can still render the
    # overlay if the caller chooses to materialize the match.
    per_step_stdout: list[list[str]] = [[] for _ in range(num_players)]

    while not done:
        actions: list[Any] = []
        for i, (fn, arity) in enumerate(zip(agent_fns, agent_arities)):
            t0 = time.perf_counter()
            buf = _io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    if arity >= 2:
                        mv = fn(obs_list[i], config)
                    else:
                        mv = fn(obs_list[i])
            except Exception as e:
                per_agent_timings[i].append(time.perf_counter() - t0)
                per_step_stdout[i].append(buf.getvalue())
                crash_status = "crashed"
                crash_msg = (
                    f"{agent_ids[i]} raised at turn {turns}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                break
            per_agent_timings[i].append(time.perf_counter() - t0)
            per_step_stdout[i].append(buf.getvalue())
            actions.append(mv if mv is not None else [])
        if crash_status is not None:
            break

        try:
            payload = step_fn(actions)
        except Exception as e:
            crash_status = "invalid_action"
            crash_msg = (
                f"engine rejected actions at turn {turns}: {e}\n"
                f"{traceback.format_exc()}"
            )
            break

        obs_list = list(payload["observations"])
        done = bool(payload.get("done", False))
        turns += 1

    # Fetch final snapshot once for score/winner extraction. With the
    # snapshot-only step path this is the only snapshot dict built per game
    # after reset.
    snap: dict
    if crash_status is None:
        try:
            snap = core.snapshot()
        except Exception as e:
            crash_status = "crashed"
            crash_msg = f"snapshot fetch failed: {e}\n{traceback.format_exc()}"
            snap = {}
    else:
        snap = {}

    duration = time.monotonic() - start

    if crash_status is not None:
        _write_ultrafast_error(log_dir, log_prefix, crash_msg)
        return MatchOutcome(
            agent_ids=agent_ids, winner=None, scores=[], turns=turns,
            duration_s=duration, seed=seed, status=crash_status,  # type: ignore[arg-type]
            replay=_crashed_replay_skeleton(crash_msg),
            per_agent_turn_seconds=per_agent_timings,
        )

    scores = _scores_from_snapshot(snap, num_players)
    winner = _winner_from_rewards(snap.get("rewards"), agent_ids)
    status: str = "ok" if winner is not None else "draw"

    # Ultrafast skips the full kaggle-envs replay, but we still surface the
    # captured agent stdout as `replay["debug"]["messages"]` so anything that
    # *does* read this outcome (e.g. a per-match debug dump) gets the same
    # `[LINE]`/`[DOT]`/`[TEXT]` markers the fast-mode viewer overlay uses.
    ultrafast_replay: dict = {}
    _attach_debug_output_from_capture(
        ultrafast_replay, per_step_stdout, num_players, env_logs=[]
    )

    return MatchOutcome(
        agent_ids=agent_ids,
        winner=winner,
        scores=scores,
        turns=turns,
        duration_s=duration,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        replay=ultrafast_replay,
        per_agent_turn_seconds=per_agent_timings,
    )


def run_match_faithful(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Run match with each agent in its own subprocess + HTTP server.

    Uses kaggle-envs UrlAgent path — identical protocol to Kaggle production.
    """
    if len(agent_ids) != len(agent_paths):
        raise ValueError(
            f"agent_ids and agent_paths length mismatch: "
            f"{len(agent_ids)} vs {len(agent_paths)}"
        )
    silence_kaggle_environments_logging()
    from kaggle_environments import make

    from .agent_subprocess import spawn_agent, shutdown

    handles: list = []
    try:
        for aid, apath in zip(agent_ids, agent_paths):
            try:
                h = spawn_agent(apath, agent_id=aid)
                handles.append(h)
            except Exception as e:
                # One agent's spawn failed; report and abort this match
                return MatchOutcome(
                    agent_ids=agent_ids,
                    winner=None,
                    scores=[],
                    turns=0,
                    duration_s=0.0,
                    seed=seed,
                    status="agent_failed_to_start",
                    replay=_crashed_replay_skeleton(f"{aid}: {e}"),
                )

        urls = [h.url for h in handles]
        env = make("orbit_wars", debug=False)

        start = time.monotonic()
        try:
            env.run(urls)
        except Exception as e:
            duration = time.monotonic() - start
            partial_timings = _per_agent_durations_from_logs(
                getattr(env, "logs", []) or [], len(agent_ids)
            )
            return MatchOutcome(
                agent_ids=agent_ids,
                winner=None,
                scores=[],
                turns=0,
                duration_s=duration,
                seed=seed,
                status="crashed",
                replay=_crashed_replay_skeleton(str(e)),
                per_agent_turn_seconds=partial_timings,
            )
        duration = time.monotonic() - start
        replay = env.toJSON()
        env_logs = env.logs or []
        # Faithful mode runs each agent in a subprocess, so its stdout is not
        # in env.logs; we still try, but the agent_subprocess path captures
        # stdout in `h.stdout_lines` (handled below). Either way attach what
        # we have so the viewer can show timings.
        replay = _attach_debug_output(replay, env_logs, len(agent_ids))
        replay = _attach_durations(replay, env_logs, len(agent_ids))
        winner, scores, turns, status = _extract_outcome(replay, agent_ids)
        per_agent_timings = _per_agent_durations_from_logs(env_logs, len(agent_ids))
        return MatchOutcome(
            agent_ids=agent_ids,
            winner=winner,
            scores=scores,
            turns=turns,
            duration_s=duration,
            seed=seed,
            status=status,  # type: ignore[arg-type]
            replay=replay,
            per_agent_turn_seconds=per_agent_timings,
        )
    finally:
        for h in handles:
            shutdown(h)
        # Persist per-agent stdout/stderr captured by shutdown(). Aligns
        # `handles` with `agent_ids` by order; if a spawn failed earlier we
        # have fewer handles, so zip naturally truncates.
        if log_dir is not None and handles:
            stdouts = [h.stdout_lines for h in handles]
            stderrs = [h.stderr_lines for h in handles]
            _write_agent_logs(
                log_dir, log_prefix, agent_ids[:len(handles)], stdouts, stderrs,
            )
