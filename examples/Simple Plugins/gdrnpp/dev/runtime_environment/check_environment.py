"""Read-only interpreter/dependency check; never import ML, load weights or inspect images."""

import argparse
import importlib.metadata
import json
import platform
import sys
import tomllib
from pathlib import Path


def check(lock):
    data = tomllib.loads(Path(lock).read_text(encoding="utf-8"))
    problems = []
    if (
        sys.platform,
        platform.machine(),
        platform.python_implementation(),
        platform.python_version(),
    ) != ("win32", "AMD64", "CPython", "3.13.9"):
        problems.append("This lock is qualified only for Windows x64 CPython 3.13.9")
    packages = []
    for spec in data["packages"]:
        name, expected = spec["name"], spec["version"]
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages.append(dict(name=name, expected=expected, installed=actual))
        if actual != expected:
            problems.append(f"{name}: expected {expected}, installed {actual or 'missing'}")
    return dict(
        compatible=not problems,
        python=platform.python_version(),
        platform=sys.platform,
        architecture=platform.machine(),
        packages=packages,
        problems=problems,
        scope="Interpreter and installed version metadata only; no model/pose qualification",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    mode = "validation" if args.validation else "runtime"
    result = check(Path(__file__).with_name(f"pylock.{mode}.toml"))
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["compatible"] else 1)
