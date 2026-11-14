"""Write a local ROS worker profile without changing either Python environment."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".local" / "ros2-runtime.json",
    )
    args = parser.parse_args(argv)
    prefix, python = args.prefix.resolve(strict=True), args.python.resolve(strict=True)
    if not prefix.is_dir() or not python.is_file():
        parser.error("Expected a ROS install prefix and Python executable")
    environment = python.parent
    profile = {
        "python": str(python),
        "prefix": str(prefix),
        "ros_distro": "lyrical",
        "path_prepend": [
            str(p)
            for p in (
                prefix / "bin",
                environment,
                environment / "Library/bin",
                environment / "Scripts",
            )
            if p.is_dir()
        ],
        "python_paths": [str(p) for p in (prefix / "Lib/site-packages",) if p.is_dir()],
        "ament_prefixes": [str(prefix)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8")) != profile:
            parser.error("Existing runtime profile differs; choose a new --output path")
    else:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(profile, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
