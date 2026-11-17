"""Explicit checksum-pinned ReferenceInspect plugin; no environment installation."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_reference_inspect-0.1.0.dev4-py3-none-any.whl"
)
EXPECTED = "099693e91b18a7441e5bf0fb7f21f02999a94d622ca327dcca3c4df8b80a5041"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError(
        "Update the host for paired execution snapshots and independent Results review"
    )
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled ReferenceInspect wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_reference_inspect")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another ReferenceInspect package is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_reference_inspect.dataflow import register  # noqa: E402,F401
