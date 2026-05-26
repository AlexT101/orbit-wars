import os
import subprocess
import json
import atexit

rust_proc = None
rust_dead = False
rust_dead_reason = ""


def cleanup():
    global rust_proc
    if rust_proc is not None:
        try:
            rust_proc.terminate()
            rust_proc.wait(timeout=1.0)
        except Exception:
            pass
        rust_proc = None


def mark_dead(turn, reason):
    global rust_dead, rust_dead_reason
    rust_dead = True
    rust_dead_reason = reason
    cleanup()
    print(f"[TEXT] {turn} RUST DEAD: {reason}", flush=True)


atexit.register(cleanup)


def agent(obs):
    global rust_proc, rust_dead, rust_dead_reason

    obs_dict = obs
    turn = obs_dict["step"]

    if rust_dead:
        print(f"[TEXT] {turn} RUST DEAD: {rust_dead_reason}", flush=True)
        return []

    if rust_proc is None:
        # Kaggle's agent runner imports main.py without setting `__file__`,
        # so fall back to the worker's unpack path when running there.
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = "/kaggle_simulations/agent"
        exec_path = os.path.join(current_dir, "target_binary", "binary")
        if not os.path.exists(exec_path):
            raise FileNotFoundError("binary not found")
            
        rust_proc = subprocess.Popen(
            [exec_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    # 1. Send observation to Rust
    try:
        message = json.dumps(obs_dict) + "\n"
        rust_proc.stdin.write(message)
        rust_proc.stdin.flush()
    except Exception as e:
        mark_dead(turn, f"stdin write failed: {e}")
        return []

    # 2. Read the single expected JSON line from stdout
    response_line = rust_proc.stdout.readline()
    if not response_line:
        mark_dead(turn, "process exited unexpectedly (EOF)")
        return []

    # 3. Fail fast if it's not valid JSON
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as e:
        mark_dead(turn, f"bad JSON / fail fast triggered: {e}")
        return []

    # 4. Handle debug array inside the JSON payload if present
    debug = response.get("debug", [])
    if debug:
        print(*debug, sep='\n', flush=True)
    else:
        print(f"[TEXT] {turn} No logs on turn {turn}.", flush=True)

    return response.get("moves", [])
