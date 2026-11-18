"""Explicit pinned completed-review plugin; no environment installation or device I/O."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_geometric_pattern_match-0.1.0.dev2-py3-none-any.whl"
)
EXPECTED = "a36824f45e5671e98906e932d860a52cc489f466839579275ca43d919d44f575"

if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError(
        "Update the host to include paired execution snapshots and read-only Results review"
    )
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled PatternMatch wheel checksum or size differs")
loaded = sys.modules.get("opencv_geometric_pattern_match")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another PatternMatch package is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_geometric_pattern_match.dataflow import register  # noqa: E402,F401
