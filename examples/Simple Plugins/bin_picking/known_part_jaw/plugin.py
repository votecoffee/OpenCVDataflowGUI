"""Explicit pinned jaw plugin and recorded, annotation-assisted example source."""

import hashlib
import sys
from pathlib import Path

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_bin_picking-0.1.0.dev5-py3-none-any.whl"
)
EXPECTED = "7cd2702f4775706dfbd7c7fefd40bd4ecf3a40c026c486fd731c81fa9a79a40f"


def checked(path):
    return (
        path.is_file()
        and path.stat().st_size <= 2 * 1024 * 1024
        and hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED
    )


if not checked(WHEEL):
    raise RuntimeError("Bundled jaw wheel checksum or size differs")
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

import opencv_simple_bin_picking  # noqa: E402

origin = str(opencv_simple_bin_picking.__file__)
if not origin.startswith(str(WHEEL)):
    end = origin.lower().rfind(".whl")
    loaded = Path(origin[: end + 4]) if end >= 0 else None
    if loaded is None or not checked(loaded):
        raise RuntimeError("Another BinPicking version is loaded; restart and load this plugin")

from opencv_dataflow.core import Block, Port, Result  # noqa: E402
from opencv_dataflow.images import DepthArray  # noqa: E402
from opencv_dataflow.payloads import frame  # noqa: E402

from opencv_simple_bin_picking.jaw_cli import read_json, read_numeric  # noqa: E402
from opencv_simple_bin_picking.jaw_dataflow import SimpleBinPickingJaw  # noqa: E402


class RecordedJawExample(Block):
    """A bounded numeric file replay; its supplied poses are explicit assistance."""

    type_id, title = "input.simple_jaw_example", "Recorded depth + supplied part poses"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(depth_path="", records_path="")
    project_asset_parameters = ("depth_path", "records_path")
    outputs = (
        Port("depth", "depth"),
        Port("part_instances", "data"),
        Port("camera", "data"),
        Port("capture_id", "text"),
    )
    parameter_help = {
        "depth_path": "Recorded axial depth in meters, bounded numeric NPY, never pickle.",
        "records_path": "Supplied pose/camera/capture records. Pose labels are assistance.",
    }

    def _path(self, relative):
        if not self.project_dir or not isinstance(relative, str) or not relative:
            raise ValueError("Save the project and choose its recorded assets")
        if any(c in relative for c in ("\\", ":")) or relative.startswith("/"):
            raise ValueError("Choose a project-contained relative asset")
        if ".." in Path(relative).parts:
            raise ValueError("Choose a project-contained relative asset")
        root = Path(self.project_dir).resolve()
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Asset resolves outside this project")
        return path

    def process(self, inputs):
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
            message="Recorded first-return depth and supplied pose assistance; no motion",
        )


def register(registry):
    registry.register(SimpleBinPickingJaw)
    registry.register(RecordedJawExample)
