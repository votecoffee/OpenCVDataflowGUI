"""Explicit pinned jaw plugin and recorded, annotation-assisted example source."""

import hashlib
import sys
from pathlib import Path

WHEEL = (
    Path(__file__).resolve().parent
    / "releases/opencv_simple_bin_picking-0.1.0.dev6-py3-none-any.whl"
)
EXPECTED = "c7c8dfff73fdc27ca2386904503c4d17ca696525bbf8d71344cf1a1bebbbcc30"


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

from opencv_dataflow.cloud_io import read_cloud  # noqa: E402
from opencv_dataflow.core import Block, Port, Result  # noqa: E402
from opencv_dataflow.images import DepthArray  # noqa: E402
from opencv_dataflow.payloads import NO_VALUE, frame  # noqa: E402
from opencv_dataflow.pointclouds import PointCloud  # noqa: E402

from opencv_simple_bin_picking.jaw_cli import read_json, read_numeric  # noqa: E402
from opencv_simple_bin_picking.jaw_dataflow import SimpleBinPickingJaw  # noqa: E402
from opencv_simple_bin_picking.pose_dataflow import SimplePartInstances  # noqa: E402


class RecordedJawExample(Block):
    """A bounded numeric file replay; its supplied poses are explicit assistance."""

    type_id, title = "input.simple_jaw_example", "Recorded depth + supplied part poses"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(depth_path="", records_path="", locator_scene_path="", locator_depth_path="")
    project_asset_parameters = (
        "depth_path",
        "records_path",
        "locator_scene_path",
        "locator_depth_path",
    )
    outputs = (
        Port("depth", "depth"),
        Port("part_instances", "data"),
        Port("camera", "data"),
        Port("capture_id", "text"),
        Port("locator_scene", "cloud", required=False),
        Port("locator_depth", "cloud", required=False),
    )
    parameter_help = {
        "depth_path": "Recorded axial depth in meters, bounded numeric NPY, never pickle.",
        "records_path": "Supplied pose/camera/capture records. Pose labels are assistance.",
        "locator_scene_path": "Optional paired PLY/PCD scene; preserve units/frame/normals.",
        "locator_depth_path": "Optional full measured first-return cloud for locator depth checks.",
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
        clouds = {}
        for name in ("locator_scene", "locator_depth"):
            relative = self.parameters[name + "_path"]
            cloud = NO_VALUE
            if relative:
                source = read_cloud(
                    self._path(relative), units="mm", frame=records["camera"]["frame"]
                )
                if source.frame != records["camera"]["frame"]:
                    raise ValueError("Recorded locator cloud differs from the depth camera frame")
                cloud = PointCloud(
                    source.points,
                    source.units,
                    source.frame,
                    colors=source.colors,
                    normals=source.normals,
                    metadata={
                        **source.metadata,
                        "capture_id": identity,
                        "source": "Explicit paired recorded file declaration",
                    },
                )
            clouds[name] = cloud
        return Result(
            dict(
                depth=DepthArray(depth, "m"),
                part_instances=records["part_instances"],
                camera=records["camera"],
                capture_id=identity,
                **clouds,
            ),
            payloads={"depth": {"spatial": frame(depth.shape[1::-1], f"recorded:{identity}")}},
            message=(
                "Recorded first-return inputs with supplied pose assistance; no motion"
                if records["part_instances"]
                else "Recorded first-return inputs; no supplied object poses or motion"
            ),
        )


def register(registry):
    registry.register(SimpleBinPickingJaw)
    registry.register(SimplePartInstances)
    registry.register(RecordedJawExample)
