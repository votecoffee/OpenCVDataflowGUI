"""Explicit pinned reflected-pattern plugin and a recorded native capture source."""

import hashlib
import re
import sys
from pathlib import Path

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame
from opencv_dataflow.plugin_dialog import PluginDialogContext

ROOT = Path(__file__).resolve().parent / "releases"
WHEELS = {
    "opencv_simple_fringe_scan": (
        "opencv_simple_fringe_scan-0.1.0.dev3-py3-none-any.whl",
        "0106d63746e36244358454432e9f707628989134e5af2288de3e436c18737f87",
    ),
    "opencv_simple_deflectometry": (
        "opencv_simple_deflectometry-0.1.0.dev3-py3-none-any.whl",
        "417c4699aed97edbba0f531038cbb8fb9a88d9d6d8b3e426fe70996340aec7f9",
    ),
}
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for independent Results and paired snapshots")
for name, (filename, digest) in WHEELS.items():
    path = ROOT / filename
    if (
        not path.is_file()
        or path.stat().st_size > 2 * 1024 * 1024
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise RuntimeError("Bundled tool/shared wheel checksum or size differs")
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = str(loaded.__file__)
        end = origin.lower().rfind(".whl")
        old = Path(origin[: end + 4]) if end >= 0 else None
        if (
            old is None
            or not old.is_file()
            or old.stat().st_size > 2 * 1024 * 1024
            or hashlib.sha256(old.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("Another tool version is loaded; restart and load this plugin first")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opencv_simple_deflectometry.dataflow import SimpleDeflectometry, recipe_path  # noqa: E402
from opencv_simple_deflectometry.execution import capture_bundle  # noqa: E402
from opencv_simple_deflectometry.recipe import load_recipe  # noqa: E402


class RecordedDeflectCapture(Block):
    type_id, title = "input.simple_deflect_capture", "Recorded reflected-pattern capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="deflectometry_capture_stack"),)

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
        capture = capture_bundle(source.reference, source.notes.get("source_provenance", {}))
        return Result(
            dict(capture=capture),
            payloads={
                "capture": {
                    "spatial": frame(
                        list(source.reference.mask.shape[::-1]),
                        source.reference.sequence.pixel_grid,
                    ),
                    "source_archive_sha256": self.parameters["sha256"],
                    "scope": "Recorded inspection source; no acquisition or training substitution",
                }
            },
            message=f"Recorded actual X/Y capture: {source.reference.capture_id}",
        )


def register(registry):
    for block in (SimpleDeflectometry, RecordedDeflectCapture):
        if registry.types.get(block.type_id) is block:
            continue
        registry.register(block)
