"""Run an explicitly selected, trusted surface-tool wheel without installation."""

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-graphics", action="store_true")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        raise ValueError("Select the trusted surface-tool wheel explicitly")
    sys.path.insert(0, str(wheel))
    from opencv_simple_surface_enhance.cli import main

    main(
        [str(args.capture), "--output", str(args.output)]
        + (["--no-graphics"] if args.no_graphics else [])
    )
