"""Pinned FringeScan and recorded normalized pattern-stack source; no installs."""

import hashlib
import re
import sys
from pathlib import Path

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = Path(__file__).resolve().parent / (
    "releases/opencv_simple_fringe_scan-0.1.0.dev3-py3-none-any.whl"
)
EXPECTED = "0106d63746e36244358454432e9f707628989134e5af2288de3e436c18737f87"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for independent Results and paired snapshots")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled FringeScan wheel checksum/size differs")
loaded = sys.modules.get("opencv_simple_fringe_scan")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    origin = str(loaded.__file__)
    end = origin.lower().rfind(".whl")
    loaded_wheel = Path(origin[: end + 4]) if end >= 0 else None
    if (
        loaded_wheel is None
        or not loaded_wheel.is_file()
        or loaded_wheel.stat().st_size > 2 * 1024 * 1024
        or hashlib.sha256(loaded_wheel.read_bytes()).hexdigest() != EXPECTED
    ):
        raise RuntimeError(
            "Another FringeScan version is loaded; restart and load this plugin first"
        )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_fringe_scan.dataflow import SimpleFringeScan, recipe_path  # noqa: E402
from opencv_simple_fringe_scan.execution import capture_from_recipe  # noqa: E402
from opencv_simple_fringe_scan.recipe import load_recipe  # noqa: E402


class RecordedFringeCapture(Block):
    type_id, title = "input.simple_fringe_capture", "Recorded pattern capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="fringe_capture_stack"),)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        recipe_path(self.project_dir, p["capture_path"])
        if not re.fullmatch("[0-9a-f]{64}", p["sha256"]):
            raise ValidationError("Choose the recorded capture archive and SHA-256", "sha256")

    def process(self, inputs):
        source = load_recipe(
            recipe_path(self.project_dir, self.parameters["capture_path"]),
            sha256=self.parameters["sha256"],
        )
        capture = capture_from_recipe(source)
        return Result(
            dict(capture=capture),
            payloads={
                "capture": {
                    "spatial": frame(
                        list(source.reference.shape[:0:-1]), source.definition["pixel_grid"]
                    ),
                    "source_archive_sha256": self.parameters["sha256"],
                    "scope": "Recorded simulated control; no acquisition or training substitution",
                }
            },
            message=f"Recorded actual normalized stack: {source.definition['capture_id']}",
        )


def register(registry):
    for block in (SimpleFringeScan, RecordedFringeCapture):
        if registry.types.get(block.type_id) is block:
            continue
        registry.register(block)
