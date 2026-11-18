"""Explicit checksum-pinned PathWidth plugin; no environment installation or device I/O."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_path_width-0.1.0.dev14-py3-none-any.whl"
)
EXPECTED = "4276490dbf0d291418088367be7fa83801008e19edd5e99ba144ba3e62242840"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError(
        "Update the host for paired execution snapshots and read-only Results review"
    )
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled PathWidth wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_path_width")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another PathWidth package is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_path_width.dataflow import register  # noqa: E402,F401
