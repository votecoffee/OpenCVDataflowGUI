"""Explicit checksum-pinned SimpleBinPicking; no environment installation or robot work."""

import hashlib
import sys
from pathlib import Path

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import frame
from opencv_dataflow.plugin_dialog import PluginDialogContext

WHEEL = Path(__file__).resolve().parent / (
    "releases/opencv_simple_bin_picking-0.1.0.dev13-py3-none-any.whl"
)
EXPECTED = "87aa7bd5775fd909a6a3dde446021150125d80422b875ac24147d69254502ea9"
if not hasattr(Block, "plugin_review_dialog_label") or not hasattr(
    PluginDialogContext, "snapshot_processed_execution"
):
    raise RuntimeError("Update the host for paired snapshots and independent Results review")
if (
    not WHEEL.is_file()
    or WHEEL.stat().st_size > 2 * 1024 * 1024
    or hashlib.sha256(WHEEL.read_bytes()).hexdigest() != EXPECTED
):
    raise RuntimeError("Bundled SimpleBinPicking wheel checksum or size differs")
loaded = sys.modules.get("opencv_simple_bin_picking")
if loaded is not None and not str(loaded.__file__).startswith(str(WHEEL)):
    raise RuntimeError(
        "Another SimpleBinPicking version is loaded; restart and load this plugin first"
    )
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

from opencv_simple_bin_picking.dataflow import SimpleBinPicking  # noqa: E402
from opencv_simple_bin_picking.jaw_cli import read_json, read_numeric  # noqa: E402
from opencv_simple_bin_picking.jaw_dataflow import SimpleBinPickingJaw  # noqa: E402


class RecordedJawExample(Block):
    """Recorded first returns and declared, explicitly assisted poses; never a robot source."""

    type_id, title = "input.simple_jaw_example", "Recorded depth + supplied part poses"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(depth_path="", records_path="", locator_scene_path="", locator_depth_path="")
    project_asset_parameters = ("depth_path", "records_path")
    outputs = (
        Port("depth", "depth"),
        Port("part_instances", "data"),
        Port("camera", "data"),
        Port("capture_id", "text"),
    )

    def _path(self, relative):
        if (
            not self.project_dir
            or not isinstance(relative, str)
            or not relative
            or any(c in relative for c in ("\\", ":"))
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise ValueError("Choose a project-contained relative recorded asset")
        root = Path(self.project_dir).resolve()
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Asset resolves outside this project")
        return path

    def process(self, inputs):
        if self.parameters["locator_scene_path"] or self.parameters["locator_depth_path"]:
            raise ValueError("This recorded replay source does not supply locator clouds")
        depth = read_numeric(self._path(self.parameters["depth_path"]))
        records = read_json(self._path(self.parameters["records_path"]))
        identity = records["capture_id"]
        return Result(
            dict(
                depth=DepthArray(depth, "m"),
                part_instances=records["part_instances"],
                camera=records["camera"],
                capture_id=identity,
            ),
            payloads={"depth": {"spatial": frame(depth.shape[1::-1], f"recorded:{identity}")}},
            message="Recorded first-return inputs with supplied pose assistance; no motion",
        )


def register(registry):
    for block in (SimpleBinPicking, SimpleBinPickingJaw, RecordedJawExample):
        if registry.types.get(block.type_id) is block:
            continue
        registry.register(block)
