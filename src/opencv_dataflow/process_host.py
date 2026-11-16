"""Fixed child launcher. Wait for ownership assignment before starting user code.

Standalone, standard-library only: launched with -I -S, also on Python 3.14.
"""

import json
import os
import subprocess
import sys


def main():
    # The parent puts this waiting process into its job/session before sending Go.
    line = sys.stdin.buffer.readline(1024 * 1024)
    if not line:
        return 2
    request = json.loads(line)
    child = subprocess.Popen(
        request["argv"],
        cwd=request["cwd"],
        env=request["environment"],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    print(f"DATAFLOW_CHILD_PID={child.pid}", flush=True)
    # A control reader waits here; the parent also checks early process exit.
    import threading

    def control():
        try:
            # Do not hold a buffered stdin lock in a daemon at interpreter exit.
            # The child may finish naturally while this reader is waiting.
            command = bytearray()
            while len(command) < 32768:
                chunk = os.read(sys.stdin.fileno(), 1)
                if not chunk or chunk == b"\n":
                    break
                command.extend(chunk)
            options = json.loads(command) if command else {}
            text = options.get("stop_input", "")
            if text:
                child.stdin.write(text.encode("utf-8"))
                child.stdin.flush()
            child.stdin.close()
            child.wait(timeout=options.get("timeout", 0.1))
        except OSError, ValueError, subprocess.TimeoutExpired:
            if child.poll() is None:
                child.terminate()

    threading.Thread(target=control, daemon=True).start()
    return child.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(2)
