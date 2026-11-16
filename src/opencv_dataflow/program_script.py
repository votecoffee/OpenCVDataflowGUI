"""Standalone bounded-hook entry point; trusted code, not a security sandbox."""

import contextlib
import json
import sys
import traceback
from pathlib import Path


def main():
    request = json.loads(Path(sys.argv[1]).read_bytes())
    context, state = request["context"], request["state"]
    result = {}
    try:
        namespace = {"__name__": "__program__", "context": context, "state": state}
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(request["source"], f"<program:{context['event']}>", "exec"), namespace)
        state = namespace["state"]
        if not isinstance(state, dict):
            raise ValueError("Program state must remain a JSON object")
        result = {"ok": True, "state": state}
        encoded = json.dumps(result, allow_nan=False).encode()
        if len(encoded) > 65536:
            raise ValueError("Program state exceeds 64 KiB")
    except BaseException as exc:
        result = {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc)[:4096],
            "traceback": traceback.format_exc()[-16384:],
        }
    Path(sys.argv[2]).write_bytes(json.dumps(result, allow_nan=False).encode())
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
