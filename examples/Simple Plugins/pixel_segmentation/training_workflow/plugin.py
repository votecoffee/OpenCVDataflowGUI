"""Explicit checksum-pinned SimpleSegmentation plugin; no environment installation."""

import hashlib
import sys
from pathlib import Path

import cv2
from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_defect_segmentation-0.1.0.dev3-py3-none-any.whl"
)
EXPECTED = "f703100682e6e0c1cc6a190d82419825e1e39cbaeef18b80a26b3ff981aafcfc"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError(
        "Update the host for paired execution snapshots and independent Results review"
    )
if not hasattr(cv2, "ml") or not hasattr(cv2.ml, "RTrees_create"):
    raise RuntimeError(
        "SimpleSegmentation requires a compatible OpenCV build with cv2.ml.RTrees"
    )
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleSegmentation wheel checksum or size differs")
loaded = sys.modules.get("opencv_defect_segmentation")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SimpleSegmentation package is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_defect_segmentation.dataflow import register  # noqa: E402,F401
