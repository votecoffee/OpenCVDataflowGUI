"""Explicit trusted loader for the bundled SimpleLaserProfile wheel (no installation)."""

import hashlib
import sys
from pathlib import Path

NAME = "opencv_simple_laser_profile-0.1.0.dev2-py3-none-any.whl"
EXPECTED = "97433ebfe03f775fb8b7c43173b18682b1cf7bd2b16333d1c3d965b209719f06"
WHEEL = Path(__file__).resolve().parent / "releases" / NAME
if (
    WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleLaserProfile wheel checksum/size differs")
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

import opencv_simple_laser_profile  # noqa: E402

if not str(opencv_simple_laser_profile.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SimpleLaserProfile version is loaded; restart and load this plugin first"
    )

from opencv_simple_laser_profile.dataflow import register  # noqa: E402

__all__ = ["register"]
