"""Explicit checksum-pinned SimpleLaserProfile; no environment installation."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_laser_profile-0.1.0.dev3-py3-none-any.whl"
)
EXPECTED = "6e0471de7f32e46f2138bc99194840a593c5bedc78549e4ab9b807cdb977d63b"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for paired snapshots and independent Results review")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleLaserProfile wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_laser_profile")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SimpleLaserProfile version is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_laser_profile.dataflow import SimpleLaserProfile  # noqa: E402


def register(registry):
    if registry.types.get(SimpleLaserProfile.type_id) is SimpleLaserProfile:
        return
    registry.register(SimpleLaserProfile)
