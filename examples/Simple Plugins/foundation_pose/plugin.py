"""Explicit checksum-pinned FoundationPose loader and portable example capture source."""

import hashlib
import sys
from pathlib import Path

WHEEL = (
    Path(__file__).resolve().parent
    / "releases"
    / "opencv_simple_foundation_pose-0.1.0.dev2-py3-none-any.whl"
)
EXPECTED = "089b3be7b6344b90f1456485fffd31747b58756fcbfd135f15f3b34772625d0c"


def checked(path):
    return (
        path.is_file()
        and path.stat().st_size <= 2 * 1024 * 1024
        and hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED
    )


if not checked(WHEEL):
    raise RuntimeError("Bundled FoundationPose wheel checksum or size differs")
if str(WHEEL) not in sys.path:
    sys.path.insert(0, str(WHEEL))

import opencv_simple_foundation_pose  # noqa: E402

origin = str(opencv_simple_foundation_pose.__file__)
if not origin.startswith(str(WHEEL)):
    end = origin.lower().rfind(".whl")
    loaded = Path(origin[: end + 4]) if end >= 0 else None
    if loaded is None or not checked(loaded):
        raise RuntimeError("Another FoundationPose version is loaded; restart and load this plugin")

import numpy as np  # noqa: E402
from opencv_simple_foundation_pose.capture import read_capture  # noqa: E402
from opencv_simple_foundation_pose.dataflow import SimpleFoundationPose  # noqa: E402
from opencv_simple_foundation_pose.wire import strict_json  # noqa: E402

from opencv_dataflow.core import Block, Port, Result  # noqa: E402
from opencv_dataflow.images import DepthArray, ImageArray  # noqa: E402
from opencv_dataflow.payloads import NO_VALUE, frame  # noqa: E402


class FoundationPoseExampleCapture(Block):
    """File source for licensed, retained captures; no detection or pose truth input."""

    type_id = "input.foundation_pose_example_capture"
    title = "FoundationPose example capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = True
    process_inputs_readonly = True
    parallel_safe = True
    defaults = dict(capture_path="", coarse_path="", capture_id="")
    project_asset_parameters = ("capture_path", "coarse_path")
    outputs = (
        Port("image", "image"),
        Port("depth", "depth"),
        Port("mask", "image"),
        Port("camera", "data"),
        Port("capture_id", "text"),
        Port("coarse_poses", "data", required=False),
    )
    parameter_help = {
        "capture_path": "Retained NPZ RGB-D/mask/rectified camera; depth is meters.",
        "coarse_path": "Optional prior raw proposals for assisted replay, never pose labels.",
        "capture_id": "Name this recorded capture; a file replay is not a fresh acquisition.",
    }

    def _path(self, relative):
        if not self.project_dir:
            raise ValueError("Save the project before selecting a capture")
        if (
            not relative
            or "\\" in relative
            or ":" in relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise ValueError("Choose a project-contained relative asset")
        root = Path(self.project_dir).resolve()
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Asset resolves outside this project")
        return path

    def process(self, inputs):
        current = read_capture(self._path(self.parameters["capture_path"]))
        identity = self.parameters["capture_id"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Name the recorded capture")
        coarse = NO_VALUE
        if self.parameters["coarse_path"]:
            path = self._path(self.parameters["coarse_path"])
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Oversized prior-proposal file")
            coarse = strict_json(path.read_bytes())
        spatial = {"spatial": frame(current.rgb.shape[1::-1], f"recorded:{identity}")}
        return Result(
            dict(
                image=ImageArray(current.rgb, "rgb"),
                depth=DepthArray(current.depth, "m"),
                mask=ImageArray(current.mask.astype(np.uint8) * 255, "gray"),
                camera=dict(
                    matrix=current.camera.tolist(),
                    image_size=list(current.rgb.shape[1::-1]),
                    frame=current.frame,
                    lens="pinhole",
                    distortion=[],
                ),
                capture_id=identity,
                coarse_poses=coarse,
            ),
            payloads={name: spatial for name in ("image", "depth", "mask")},
            message="Recorded calibrated capture with supplied instance mask; no pose labels",
        )


def register(registry):
    registry.register(SimpleFoundationPose)
    registry.register(FoundationPoseExampleCapture)
