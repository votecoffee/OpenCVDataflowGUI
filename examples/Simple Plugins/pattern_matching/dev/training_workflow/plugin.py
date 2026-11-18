"""Explicit pinned completed-review plugin; no environment installation or device I/O."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_geometric_pattern_match-0.1.0.dev3-py3-none-any.whl"
)
EXPECTED = "a3e4025d4100d7d0b7fd3a1c469cd2468de685c10ffdec29ba8fff98f62eb2eb"

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
