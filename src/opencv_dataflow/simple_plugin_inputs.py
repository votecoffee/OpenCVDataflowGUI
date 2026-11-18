"""Recorded-input blocks promoted from the public Simple Plugin examples.

Loading a graph constructs these sources; only explicit execution reads a capture.
The host fixes all imports. Project fields select data assets, never Python plugins.
"""

import re
from pathlib import Path

import numpy as np

from .cloud_io import read_cloud
from .core import Block, Port, Result
from .images import DepthArray, ImageArray
from .parameter_validation import ValidationError
from .payloads import NO_VALUE, frame
from .pointclouds import PointCloud


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
        from opencv_simple_bin_picking.jaw_cli import read_json, read_numeric

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
        from opencv_simple_foundation_pose.capture import read_capture
        from opencv_simple_foundation_pose.wire import strict_json

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


class RecordedSurfaceCapture(Block):
    type_id, title = "input.simple_surface_capture", "Recorded lighting capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="surface_illumination_stack"),)

    def validate_parameters(self, parameters=None):
        from opencv_simple_surface_enhance.dataflow import recipe_path

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        recipe_path(self.project_dir, p["capture_path"])
        if not re.fullmatch("[0-9a-f]{64}", p["sha256"]):
            raise ValidationError("Choose the recorded capture archive and SHA-256", "sha256")

    def process(self, inputs):
        from opencv_simple_surface_enhance.dataflow import recipe_path
        from opencv_simple_surface_enhance.execution import capture_from_recipe
        from opencv_simple_surface_enhance.recipe import load_recipe

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


class RecordedFringeCapture(Block):
    type_id, title = "input.simple_fringe_capture", "Recorded pattern capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="fringe_capture_stack"),)

    def validate_parameters(self, parameters=None):
        from opencv_simple_fringe_scan.dataflow import recipe_path

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        recipe_path(self.project_dir, p["capture_path"])
        if not re.fullmatch("[0-9a-f]{64}", p["sha256"]):
            raise ValidationError("Choose the recorded capture archive and SHA-256", "sha256")

    def process(self, inputs):
        from opencv_simple_fringe_scan.dataflow import recipe_path
        from opencv_simple_fringe_scan.execution import capture_from_recipe
        from opencv_simple_fringe_scan.recipe import load_recipe

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


class RecordedDeflectCapture(Block):
    type_id, title = "input.simple_deflect_capture", "Recorded reflected-pattern capture"
    category = block_type = "Input"
    subtype = "Examples"
    preview_safe = process_inputs_readonly = parallel_safe = True
    defaults = dict(capture_path="", sha256="")
    project_asset_parameters = ("capture_path",)
    outputs = (Port("capture", "any", visual_type="deflectometry_capture_stack"),)

    def validate_parameters(self, parameters=None):
        from opencv_simple_deflectometry.dataflow import recipe_path

        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        recipe_path(self.project_dir, p["capture_path"])
        if not re.fullmatch("[0-9a-f]{64}", p["sha256"]):
            raise ValidationError("Choose the recorded capture archive and SHA-256", "sha256")

    def process(self, inputs):
        from opencv_simple_deflectometry.dataflow import recipe_path
        from opencv_simple_deflectometry.execution import capture_bundle
        from opencv_simple_deflectometry.recipe import load_recipe

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

RECORDED_INPUTS = {
    cls.type_id: cls
    for cls in (
        RecordedJawExample,
        FoundationPoseExampleCapture,
        RecordedSurfaceCapture,
        RecordedFringeCapture,
        RecordedDeflectCapture,
    )
}
