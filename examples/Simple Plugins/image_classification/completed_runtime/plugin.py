"""Explicit checksum-pinned SimpleClassify plugin; no environment installation."""

import hashlib
import sys
from pathlib import Path

import cv2
from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = Path(__file__).resolve().parent / "releases/opencv_simple_classifier-0.3.2-py3-none-any.whl"
EXPECTED = "994529369244ff6ae07e4c645f3f197cd73ea23cd22391899ebeb883b7619e5d"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError(
        "Update the host for paired execution snapshots and independent Results review"
    )
if not hasattr(cv2, "ml") or not hasattr(cv2.ml, "RTrees_create"):
    raise RuntimeError("SimpleClassify needs a compatible OpenCV build with cv2.ml.RTrees")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleClassify wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_classifier")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SimpleClassify package is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_classifier.dataflow import ClassifyImage  # noqa: E402


def register(registry):
    # This explicitly selected version replaces only the known bundled block.
    # The loader stages registration, leaving unrelated providers untouched.
    existing = registry.types.get(ClassifyImage.type_id)
    if existing is not None:
        if (
            existing is not ClassifyImage
            and existing.__module__ != "opencv_dataflow.simple_classifier.dataflow"
        ):
            raise ValueError("A different provider already owns the SimpleClassify type ID")
        registry.types.pop(ClassifyImage.type_id)
    registry.register(ClassifyImage)
