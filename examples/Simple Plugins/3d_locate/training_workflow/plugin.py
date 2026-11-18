"""Explicit checksum-pinned Simple3DLocate; no environment installation."""

import hashlib
import sys
from pathlib import Path

import cv2
from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_3d_locate-0.1.0.dev22-py3-none-any.whl"
)
EXPECTED = "c299e0d8433c10d245f5ec6849eed9843fb761eab53a12cdd3c99e1ee495275c"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for paired snapshots and independent Results review")
if not hasattr(cv2, "ppf_match_3d"):
    raise RuntimeError("Simple3DLocate requires an OpenCV contrib build with ppf_match_3d")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled Simple3DLocate wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_3d_locate")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another Simple3DLocate version is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_3d_locate.dataflow import Simple3DLocate  # noqa: E402


def register(registry):
    if registry.types.get(Simple3DLocate.type_id) is Simple3DLocate:
        return
    registry.register(Simple3DLocate)
