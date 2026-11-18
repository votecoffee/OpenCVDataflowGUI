"""Explicit pinned SurfaceEnhance and a recorded normalized-stack source; no installs."""

import hashlib
import re
import sys
from pathlib import Path

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = Path(__file__).resolve().parent / (
    "releases/opencv_simple_surface_enhance-0.1.0.dev4-py3-none-any.whl"
)
EXPECTED = "26a1276124dbb25b50983a7bfad7ead7a61ce920a34d32aec97fc5e53990c6d8"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for independent Results and paired snapshots")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SurfaceEnhance wheel checksum/size differs")
loaded = sys.modules.get("opencv_simple_surface_enhance")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SurfaceEnhance version is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_surface_enhance.dataflow import SimpleSurfaceEnhance, recipe_path  # noqa: E402
from opencv_simple_surface_enhance.execution import capture_from_recipe  # noqa: E402
from opencv_simple_surface_enhance.recipe import load_recipe  # noqa: E402


class RecordedSurfaceCapture(Block):
    type_id, title = "input.simple_surface_capture", "Recorded lighting capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="surface_illumination_stack"),)

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
                    "spatial": frame(list(source.reference.shape[:0:-1]), source.pixel_grid),
                    "source_archive_sha256": self.parameters["sha256"],
                    "scope": "Recorded rendered control; no acquisition or training substitution",
                }
            },
            message=f"Recorded actual normalized stack: {source.capture_id}",
        )


def register(registry):
    for block in (SimpleSurfaceEnhance, RecordedSurfaceCapture):
        if registry.types.get(block.type_id) is block:
            continue
        registry.register(block)
