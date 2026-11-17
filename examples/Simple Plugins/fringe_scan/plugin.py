"""Explicit trusted loader for the bundled SimpleFringeScan wheel; no installation."""

import hashlib
import sys
from pathlib import Path

NAME = "opencv_simple_fringe_scan-0.1.0.dev2-py3-none-any.whl"
WHEEL = Path(__file__).resolve().parent / "releases" / NAME
EXPECTED = "5e75af7d9533d277d25001b2f0389094008159fae5e3fefadd0a61abfa2ce780"
if WHEEL.stat().st_size > 2 * 1024 * 1024 or (
    hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleFringeScan wheel checksum/size differs")
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

import opencv_simple_fringe_scan  # noqa: E402

if not str(opencv_simple_fringe_scan.__file__).startswith(str(WHEEL)):
    origin = str(opencv_simple_fringe_scan.__file__)
    end = origin.lower().rfind(".whl")
    loaded_wheel = Path(origin[: end + 4]) if end >= 0 else None
    if (
        loaded_wheel is None
        or not loaded_wheel.is_file()
        or (
            loaded_wheel.stat().st_size > 2 * 1024 * 1024
            or hashlib.sha256(loaded_wheel.read_bytes()).hexdigest() != EXPECTED
        )
    ):
        raise RuntimeError(
            "Another fringe-tool version is loaded; restart and load this plugin first"
        )

from opencv_simple_fringe_scan.dataflow import register  # noqa: E402

__all__ = ["register"]
