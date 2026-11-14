"""One cached planning block; optional graphics never change numeric pick decisions."""

import hashlib
import re
from pathlib import Path, PurePosixPath

from opencv_dataflow.core import Port, Result
from opencv_dataflow.geometry3d_common import Tool3D, pose_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

from .graphics import result_cloud
from .persistence import load_recipe
from .planning import Picker
from .specs import Camera, rigid


def recipe_path(root, relative):
    if root is None:
        raise ValueError("Save the project before teaching bin picking")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Teach or import a bin-picking recipe")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
        raise ValueError("Recipe must stay inside the project and use forward slashes")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target.suffix != ".sbp":
        raise ValueError("Select a project-contained .sbp recipe")
    return target


def input_pose(value, recipe):
    if value is None:
        return recipe.bin_to_camera
    if (
        not isinstance(value, dict)
        or value.get("model") != "pose_3d_v1"
        or value.get("units") != recipe.units
        or value.get("source_frame") != recipe.frame
        or value.get("target_frame") != recipe.camera.frame
    ):
        raise ValueError("Bin pose must map the retained bin frame to camera in matching units")
    return rigid(value["source_to_target"])


def plan_inputs(picker, inputs):
    recipe = picker.recipe
    if inputs.get("camera") is not None and Camera.from_record(inputs["camera"]) != recipe.camera:
        raise ValueError("Camera calibration differs from training; retrain or restore it")
    depth = inputs["depth"]
    if hasattr(depth, "units") and (depth.units != recipe.units or recipe.depth_scale != 1.0):
        raise ValueError("Unit-bearing depth must match recipe units and have scale 1")
    return picker.plan(
        depth,
        bin_to_camera=input_pose(inputs.get("bin_pose"), recipe),
        instances=inputs.get("instances"),
    )


class SimpleBinPicking(Tool3D):
    type_id, title = "3d.simple_bin_picking", "SimpleBinPicking"
    subtype = "Bin picking"
    process_inputs_readonly = True
    parallel_safe = True
    plugin_dialog_label = "Train bin picking…"
    plugin_dialog_modeless = True
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data", required=False, required_when_connected=True),
        Port("bin_pose", "data", required=False, required_when_connected=True),
        Port("instances", required=False),
        Port("image", required=False),
    )
    outputs = (
        Port("picks", "data"),
        Port("candidates", "data"),
        Port("report", "data"),
        Port("count", "number"),
        Port("tcp", "data", required=False),
        Port("pregrasp", "data", required=False),
        Port("graphics", "cloud", required=False),
    )
    defaults = {"recipe_path": "", "sha256": "", "render_graphics": True}
    parameter_help = {
        "recipe_path": "Retained depth/RGB, calibration, bin and suction tool recipe.",
        "sha256": "Imported/staged recipe checksum.",
        "render_graphics": "Native cup/pose/sweep cloud; off skips composition.",
    }
    input_help = {
        "depth": "Rectified axial Z, uint16/float; scale and units come from the trained recipe.",
        "camera": "Optional camera_model_v1; when connected, must match retained intrinsics/frame.",
        "bin_pose": "Optional pose_3d_v1 mapping bin +Z-up frame into camera, in recipe units.",
        "instances": "Optional integer instance map: 0 excluded, positive IDs restrict contact.",
        "image": "Optional RGB/BGR appearance for reference capture; planning uses depth.",
    }
    output_help = {
        "picks": "Ranked eligible alternatives; reobserve after each physical pick.",
        "tcp": "Best tool pose; absent with no eligible pick. +Z retracts from the surface.",
        "pregrasp": "Best pregrasp tool pose; approach motion is -Z.",
        "candidates": "Includes rejected/unknown contact, collision and visibility evidence.",
        "graphics": "Native flyaround cups, swept tool envelopes, bin and witnesses.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_output = "graphics"
        self._digest = self._picker = None

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        try:
            recipe_path(self.project_dir, p["recipe_path"])
        except ValueError as error:
            raise ValidationError(str(error), "recipe_path") from error
        if not isinstance(p["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", p["sha256"]):
            raise ValidationError("Import/stage the intended recipe with its SHA-256", "sha256")
        if type(p["render_graphics"]) is not bool:
            raise ValidationError("render_graphics must be boolean", "render_graphics")

    def process(self, inputs):
        p = self.parameters
        path = recipe_path(self.project_dir, p["recipe_path"])
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Recipe exceeds 64 MiB")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != p["sha256"]:
            raise ValueError("Recipe SHA-256 changed; stage the intended revision")
        if digest != self._digest:
            self._picker, self._digest = Picker(load_recipe(path, sha256=digest)), digest
        recipe = self._picker.recipe
        result = plan_inputs(self._picker, inputs)
        graphics = NO_VALUE
        if p["render_graphics"] and self.generate_graphics:
            cloud = result_cloud(result)
            graphics = PointCloud(
                cloud.points, cloud.units, cloud.frame, colors=cloud.colors, metadata=cloud.metadata
            )
        picks = list(result.picks)
        tcp = pregrasp = NO_VALUE
        if picks:
            tcp = pose_record(picks[0]["tcp_pose"], "tcp", recipe.frame, recipe.units)
            pregrasp = pose_record(picks[0]["pregrasp_pose"], "tcp", recipe.frame, recipe.units)
        return Result(
            {
                "picks": picks,
                "candidates": list(result.candidates),
                "report": result.report(),
                "count": len(picks),
                "tcp": tcp,
                "pregrasp": pregrasp,
                "graphics": graphics,
            },
            passed=bool(picks),
            fail_reasons=[] if picks else ["No geometrically eligible suction alternative"],
            message=f"{len(picks)} eligible alternatives from {len(result.candidates)} candidates; "
            "contact / observed collision / unknown evidence retained",
        )

    def create_plugin_dialog(self, parent, parameters, context):
        from .qt_training import BinPickingDialog

        return BinPickingDialog(parent, parameters, context)

    def close(self):
        self._picker = self._digest = None


def register(registry):
    registry.register(SimpleBinPicking)
