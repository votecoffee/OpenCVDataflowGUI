"""Explicit checksum-pinned loader for shared numerical and deflectometry wheels."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "releases"
WHEEL = ROOT / "opencv_simple_deflectometry-0.1.0.dev2-py3-none-any.whl"
FRINGE_WHEEL = ROOT / "opencv_simple_fringe_scan-0.1.0.dev2-py3-none-any.whl"
EXPECTED = "84bc0ff026977cb47718621d9d54da557e2df62c807bfb5b5a75917a7f009056"
FRINGE_EXPECTED = "5e75af7d9533d277d25001b2f0389094008159fae5e3fefadd0a61abfa2ce780"


def checked(path, digest):
    return (
        path.is_file()
        and path.stat().st_size <= 2 * 1024 * 1024
        and (hashlib.sha256(path.read_bytes()).hexdigest() == digest)
    )


for path, digest in ((FRINGE_WHEEL, FRINGE_EXPECTED), (WHEEL, EXPECTED)):
    if not checked(path, digest):
        raise RuntimeError("Bundled numerical/tool wheel checksum or size differs")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opencv_simple_fringe_scan  # noqa: E402

import opencv_simple_deflectometry  # noqa: E402

for module, path, digest in (
    (opencv_simple_fringe_scan, FRINGE_WHEEL, FRINGE_EXPECTED),
    (opencv_simple_deflectometry, WHEEL, EXPECTED),
):
    if not str(module.__file__).startswith(str(path)):
        origin = str(module.__file__)
        end = origin.lower().rfind(".whl")
        loaded = Path(origin[: end + 4]) if end >= 0 else None
        if loaded is None or not checked(loaded, digest):
            raise RuntimeError(
                "Another tool version is already loaded; restart and load this plugin first"
            )

from opencv_simple_deflectometry.dataflow import register  # noqa: E402

__all__ = ["register"]
