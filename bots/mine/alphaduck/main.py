"""Lab wrapper: re-exports `agent` from bots/alphaduck/main.py.

The real bot lives at bots/alphaduck/. This shim exists so the lab's
discovery (which scans bots/{baselines,external,mine}/) can find it.
"""
import importlib.util
import inspect
import os
import sys


def _here() -> str:
    f = inspect.currentframe()
    if f is not None and f.f_code.co_filename and f.f_code.co_filename != "<string>":
        return os.path.dirname(os.path.abspath(f.f_code.co_filename))
    return os.path.dirname(os.path.abspath(__file__))


_HERE = _here()
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ALPHADUCK = os.path.join(_ROOT, "bots", "alphaduck", "main.py")

spec = importlib.util.spec_from_file_location("alphaduck_main", _ALPHADUCK)
_duck = importlib.util.module_from_spec(spec)
sys.modules["alphaduck_main"] = _duck
spec.loader.exec_module(_duck)

agent = _duck.agent
