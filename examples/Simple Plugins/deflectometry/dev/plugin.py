"""Explicit checksum-pinned loader for shared numerical and deflectometry wheels."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "releases"
WHEEL = ROOT / "opencv_simple_deflectometry-0.1.0.dev3-py3-none-any.whl"
FRINGE_WHEEL = ROOT / "opencv_simple_fringe_scan-0.1.0.dev3-py3-none-any.whl"
EXPECTED = "417c4699aed97edbba0f531038cbb8fb9a88d9d6d8b3e426fe70996340aec7f9"
FRINGE_EXPECTED = "0106d63746e36244358454432e9f707628989134e5af2288de3e436c18737f87"
FRINGE_COMPATIBLE = (
    FRINGE_EXPECTED,
)


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
        accepted = FRINGE_COMPATIBLE if module is opencv_simple_fringe_scan else (digest,)
        if loaded is None or not any(checked(loaded, value) for value in accepted):
            raise RuntimeError(
                "Another tool version is already loaded; restart and load this plugin first"
            )

from opencv_simple_deflectometry.dataflow import SimpleDeflectometry  # noqa: E402


def register(registry):
    if registry.types.get(SimpleDeflectometry.type_id) is not SimpleDeflectometry:
        registry.register(SimpleDeflectometry)

__all__ = ["register"]
