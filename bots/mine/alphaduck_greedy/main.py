"""Lab wrapper: re-exports `agent` from bots/alphaduck/greedy/main.py.

Greedy variant: same model as alphaduck, no MCTS — argmax over policy head.
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
_GREEDY = os.path.join(_ROOT, "bots", "alphaduck", "greedy", "main.py")

spec = importlib.util.spec_from_file_location("alphaduck_greedy_main", _GREEDY)
_greedy = importlib.util.module_from_spec(spec)
sys.modules["alphaduck_greedy_main"] = _greedy
spec.loader.exec_module(_greedy)

agent = _greedy.agent
