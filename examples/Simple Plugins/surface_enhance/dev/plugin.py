"""Explicit trusted loader for the bundled surface tool; no installation required."""

import hashlib
import sys
from pathlib import Path

NAME = "opencv_simple_surface_enhance-0.1.0.dev3-py3-none-any.whl"
WHEEL = Path(__file__).resolve().parent / "releases" / NAME
EXPECTED = "d08ace480ee7f86f1e9691d33566a0291053998a6f5029e490f79a8719201cb3"
if WHEEL.stat().st_size > 2 * 1024 * 1024 or (
    hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleSurfaceEnhance wheel checksum/size differs")
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

import opencv_simple_surface_enhance  # noqa: E402

if not str(opencv_simple_surface_enhance.__file__).startswith(str(WHEEL)):
    raise RuntimeError("Another surface-tool version is loaded; restart and load this plugin first")

from opencv_simple_surface_enhance.dataflow import register  # noqa: E402

__all__ = ["register"]
