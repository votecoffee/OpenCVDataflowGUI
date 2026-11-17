"""Select two explicit trusted wheels without changing the host environment."""

import argparse
import sys
from pathlib import Path


def select(wheel, fringe_wheel):
    for path in (fringe_wheel, wheel):
        if path.suffix != ".whl" or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Select a bounded trusted tool wheel")
        sys.path.insert(0, str(path.resolve()))
    import opencv_simple_fringe_scan

    import opencv_simple_deflectometry

    for module, path in (
        (opencv_simple_deflectometry, wheel),
        (opencv_simple_fringe_scan, fringe_wheel),
    ):
        if not str(module.__file__).startswith(str(path.resolve())):
            raise RuntimeError("Another tool version is already loaded; start a fresh interpreter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--fringe-wheel", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    select(args.wheel, args.fringe_wheel)
    from opencv_simple_deflectometry.cli import main

    raise SystemExit(main(remaining))
