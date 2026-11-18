"""Explicit checksum-pinned Simple3DMeasure; no environment installation."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_3d_measure-0.1.0.dev7-py3-none-any.whl"
)
EXPECTED = "32fca513e9839ca7d78053bca447359289193950d8a3daec6eb53b00a05cfc6a"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for paired snapshots and independent Results review")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled Simple3DMeasure wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_3d_measure")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another Simple3DMeasure version is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_3d_measure.dataflow import Simple3DMeasure  # noqa: E402


def register(registry):
    if registry.types.get(Simple3DMeasure.type_id) is Simple3DMeasure:
        return
    registry.register(Simple3DMeasure)
