"""Structured snapshot diffing.

Returns a list of `Diff` records pinpointing exactly where two engines
disagree, in dotted-path form. The harness uses the first record to fail
fast on divergence, but the full list is useful when porting and we want
to see the blast radius of a single bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from parity.engine import Snapshot


@dataclass
class Diff:
    path: str
    a: Any
    b: Any
    reason: str = ""


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _cmp_scalar(path: str, a: Any, b: Any, atol: float, out: list[Diff]) -> None:
    if _is_num(a) and _is_num(b):
        if abs(float(a) - float(b)) > atol:
            out.append(Diff(path, a, b, f"|Δ|={abs(float(a)-float(b)):.3e} > atol={atol:.0e}"))
        return
    if a != b:
        out.append(Diff(path, a, b, "!="))


def _cmp_list(path: str, a: list, b: list, atol: float, out: list[Diff]) -> None:
    if len(a) != len(b):
        out.append(Diff(f"{path}.len", len(a), len(b), "length mismatch"))
        return
    for i, (x, y) in enumerate(zip(a, b)):
        _cmp(f"{path}[{i}]", x, y, atol, out)


def _cmp_dict(path: str, a: dict, b: dict, atol: float, out: list[Diff]) -> None:
    keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in a:
            out.append(Diff(f"{path}.{k}", "<missing>", b[k], "key only in b"))
        elif k not in b:
            out.append(Diff(f"{path}.{k}", a[k], "<missing>", "key only in a"))
        else:
            _cmp(f"{path}.{k}", a[k], b[k], atol, out)


def _cmp(path: str, a: Any, b: Any, atol: float, out: list[Diff]) -> None:
    if isinstance(a, list) and isinstance(b, list):
        _cmp_list(path, a, b, atol, out)
    elif isinstance(a, dict) and isinstance(b, dict):
        _cmp_dict(path, a, b, atol, out)
    else:
        _cmp_scalar(path, a, b, atol, out)


def diff_snapshots(a: Snapshot, b: Snapshot, atol: float = 0.0) -> list[Diff]:
    """Compare two snapshots field-by-field. `info` is excluded by design."""
    out: list[Diff] = []
    _cmp_scalar("step", a.step, b.step, atol, out)
    _cmp_scalar("angular_velocity", a.angular_velocity, b.angular_velocity, atol, out)
    _cmp_scalar("next_fleet_id", a.next_fleet_id, b.next_fleet_id, atol, out)
    _cmp_scalar("done", a.done, b.done, atol, out)
    _cmp_list("planets", a.planets, b.planets, atol, out)
    _cmp_list("initial_planets", a.initial_planets, b.initial_planets, atol, out)
    _cmp_list("fleets", a.fleets, b.fleets, atol, out)
    _cmp_list("comet_planet_ids", a.comet_planet_ids, b.comet_planet_ids, atol, out)
    _cmp_list("comets", a.comets, b.comets, atol, out)
    if a.rewards is None and b.rewards is None:
        pass
    elif a.rewards is None or b.rewards is None:
        out.append(Diff("rewards", a.rewards, b.rewards, "one side None"))
    else:
        _cmp_list("rewards", a.rewards, b.rewards, atol, out)
    return out


def format_diff(diffs: list[Diff], max_lines: int = 20) -> str:
    if not diffs:
        return "(no diff)"
    lines = [f"{len(diffs)} divergence(s):"]
    for d in diffs[:max_lines]:
        lines.append(f"  {d.path}: a={d.a!r}  b={d.b!r}  ({d.reason})")
    if len(diffs) > max_lines:
        lines.append(f"  ... {len(diffs) - max_lines} more")
    return "\n".join(lines)
