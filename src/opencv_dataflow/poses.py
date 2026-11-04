"""Image-plane feature frames and inspections defined in local pose coordinates."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .objects import validate_collection
from .payloads import check_spatial, metadata, moved_payload, validate_payload
from .processing import finite, integer
from .regions import ShapeRegion, region_result
from .selection import rasterize, region_matrix, region_size

POSE_OFFSETS = {"pose_offset_x": 0.0, "pose_offset_y": 0.0, "pose_offset_angle": 0.0}
POSE_HELP = {
    "pose_offset_x": "Local x offset in feature/model pixels; follows rotation and scale.",
    "pose_offset_y": "Local y offset in feature/model pixels; follows rotation and scale.",
    "pose_offset_angle": "Local counterclockwise rotation in degrees (OpenCV convention).",
}
POSE_OUTPUTS = (Port("pose", "data"), Port("points", "data"), Port("angle_degrees", "number"))


def local_offset(x, y, angle):
    x, y, angle = (finite(v, "Pose offset") for v in (x, y, angle))
    if max(abs(x), abs(y)) > 1_000_000:
        raise ValueError("Pose offsets must be within one million pixels")
    a = math.radians(angle % 360)
    return np.array([[math.cos(a), math.sin(a), x], [-math.sin(a), math.cos(a), y], [0, 0, 1]])


def make_pose(matrix, size, payload, **details):
    # Reuse the region limits for all image-plane pose target grids.
    region_size({"model": "image_region_v1", "image_size": size})
    validate_payload(payload)
    check_spatial(payload, size)
    matrix = region_matrix(matrix)
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Pose origin maps to infinity")
    matrix = matrix / matrix[2, 2]
    origin = matrix[:2, 2]
    jacobian = matrix[:2, :2] - origin[:, None] * matrix[2, :2]
    if not np.isfinite(jacobian).all() or np.max(np.abs(origin)) > 1e7:
        raise ValueError("Pose origin or local axes are outside supported bounds")
    return {
        **details,
        "model": "image_pose_v1",
        "source_to_target": matrix.tolist(),
        "target_size": list(size),
        "target_payload": copy.deepcopy(payload),
        "origin": origin.tolist(),
        "angle_degrees": -math.degrees(math.atan2(jacobian[1, 0], jacobian[0, 0])),
        "scale_x": float(np.linalg.norm(jacobian[:, 0])),
        "scale_y": float(np.linalg.norm(jacobian[:, 1])),
    }


def validate_pose(pose):
    if not isinstance(pose, dict) or pose.get("model") != "image_pose_v1":
        raise ValueError("A valid image_pose_v1 pose is required; no pose is available on no match")
    return make_pose(
        pose.get("source_to_target"), pose.get("target_size"), pose.get("target_payload", {})
    )


def offset_pose(pose, x, y, angle):
    validated = validate_pose(pose)
    return make_pose(
        np.asarray(validated["source_to_target"]) @ local_offset(x, y, angle),
        validated["target_size"],
        validated["target_payload"],
    )


def pose_for_image(pose, image):
    pose = validate_pose(pose)
    size, payload = [image.shape[1], image.shape[0]], metadata(image)
    source = pose["target_payload"].get("spatial", {})
    target = payload.get("spatial", {})
    source_id, target_id = source.get("root_id"), target.get("root_id")
    matrix = np.asarray(pose["source_to_target"])
    if source_id is not None and target_id is not None:
        if source_id != target_id or source.get("root_size") != target.get("root_size"):
            raise ValueError("Pose and image have different coordinate roots")
        matrix = np.linalg.inv(target["to_root"]) @ source["to_root"] @ matrix
    elif pose["target_size"] != size:
        raise ValueError("Pose/image dimensions differ without a shared coordinate root")
    return make_pose(matrix, size, payload)


def draw_pose(overlay, pose):
    matrix = np.asarray(pose["source_to_target"], float)
    local = np.array([[[0, 0]], [[20, 0]], [[0, 20]]], np.float64)
    denominators = np.c_[local[:, 0], np.ones(3)] @ matrix[2]
    if np.any(np.abs(denominators) < 1e-12):
        return
    points = cv2.perspectiveTransform(local, matrix).reshape(-1, 2)
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e7:
        return
    origin, x, y = np.rint(points).astype(int)
    cv2.arrowedLine(overlay, tuple(origin), tuple(x), (40, 40, 240), 2)
    cv2.arrowedLine(overlay, tuple(origin), tuple(y), (40, 220, 40), 2)


def pose_result(pose, image=None):
    outputs = {"pose": pose, "points": [pose["origin"]], "angle_degrees": pose["angle_degrees"]}
    if image is not None:
        overlay = overlay_image(image)
        draw_pose(overlay, pose)
        outputs["overlay"] = overlay
    return Result(outputs, payloads={key: pose["target_payload"] for key in outputs})


class DefinePose(Block):
    type_id, title, category = "pose.define", "Define image pose", "Inspection"
    block_type, subtype = "Inspection", "Poses"
    inputs = (Port("image"),)
    outputs = (*POSE_OUTPUTS, Port("overlay"))
    defaults = {"x": 0.0, "y": 0.0, "angle_degrees": 0.0, "scale": 1.0}
    parameter_help = {"angle_degrees": "Counterclockwise degrees; local x right, local y down."}

    def process(self, inputs):
        image, p = inputs["image"], self.parameters
        scale = finite(p["scale"], "Pose scale", 0)
        if not 0 < scale <= 1000:
            raise ValueError("Pose scale must be in (0,1000]")
        matrix = local_offset(p["x"], p["y"], p["angle_degrees"]) @ np.diag([scale, scale, 1])
        return pose_result(
            make_pose(matrix, [image.shape[1], image.shape[0]], metadata(image)), image
        )


class OffsetPose(Block):
    type_id, title, category = "pose.offset", "Offset pose", "Inspection"
    block_type, subtype = "Inspection", "Poses"
    inputs, outputs = (Port("pose", "data"),), POSE_OUTPUTS
    defaults = POSE_OFFSETS
    parameter_help = POSE_HELP

    def process(self, inputs):
        p = self.parameters
        return pose_result(offset_pose(inputs["pose"], *(p[k] for k in POSE_OFFSETS)))


class PoseFromFeature(Block):
    type_id, title, category = "pose.from_feature", "Pose from feature", "Inspection"
    block_type, subtype = "Inspection", "Poses"
    inputs = (Port("image"), Port("features", "data"))
    outputs = (*POSE_OUTPUTS, Port("overlay"))
    defaults = {"index": 0, "angle_mode": "fixed", "angle_degrees": 0.0, **POSE_OFFSETS}
    parameter_choices = {"angle_mode": ("fixed", "feature")}
    parameter_help = {
        **POSE_HELP,
        "index": "Zero-based rank in a blob/line/contour collection or pattern matches.",
        "angle_mode": "Fixed uses angle_degrees. Feature uses the measured principal/line axis.",
    }

    def process(self, inputs):
        image, features, p = inputs["image"], inputs["features"], self.parameters
        index = integer(p["index"], "Feature index", 0, 99999)
        if not isinstance(features, dict):
            raise ValueError("Features must be a measured collection or pattern matches")
        if features.get("model") == "pattern_matches_v1":
            items = features.get("items", [])
            if not isinstance(items, list) or not index < len(items):
                raise ValueError("Feature index is outside available pattern matches")
            pose = pose_for_image(items[index].get("pose"), image)
        else:
            validate_collection(features)
            if not index < len(features["items"]):
                raise ValueError("Feature index is outside available collection")
            metrics = features["items"][index]["metrics"]
            angle = finite(p["angle_degrees"], "Fixed angle")
            if p["angle_mode"] == "feature":
                if metrics.get("angle") is None:
                    raise ValueError("Feature has no stable orientation; choose a fixed angle")
                # Collection axes use image-clockwise angles; poses use OpenCV CCW.
                angle = -finite(metrics["angle"], "Feature angle")
            origin = features.get("origin", [0, 0])
            matrix = local_offset(
                metrics["center_x"] - origin[0], metrics["center_y"] - origin[1], angle
            )
            context = {**features.get("context", {}), **self.input_payloads.get("features", {})}
            context = moved_payload(context, local_offset(*origin, 0), features["image_size"])
            pose = make_pose(matrix, features["image_size"], context)
            pose = pose_for_image(pose, image)
        pose = offset_pose(pose, *(p[k] for k in POSE_OFFSETS))
        return pose_result(pose, image)


class PoseRegion(Block):
    type_id, title, category = "region.pose", "Region relative to pose", "Image processing"
    block_type, subtype = "Image processing", "Regions"
    inputs = (Port("image"), Port("pose", "data"))
    outputs = ShapeRegion.outputs
    defaults = {
        **ShapeRegion.defaults,
        "rectangle": [-20, -20, 40, 40],
        "center": [0, 0],
        "vertices": [[-20, -20], [20, -20], [0, 20]],
    }
    parameter_choices = ShapeRegion.parameter_choices
    parameter_help = {
        "rectangle": "[x,y,width,height] in local pose pixels; negative positions are allowed.",
        "center": "Circle center in local pose pixels.",
        "vertices": "Polygon vertices in local pose pixels.",
        "invert": "Select the target-image area outside the positioned shape.",
    }

    supports_region_editor = True

    def process(self, inputs):
        image = inputs["image"]
        region = relative_region(image, inputs["pose"], self.parameters)
        result = region_result(image, region)
        draw_pose(result.outputs["overlay"], region["pose"])
        return result


def relative_region(image, pose, parameters):
    """Build the same pose-relative geometry for execution and interactive previews."""
    p = parameters
    pose = pose_for_image(pose, image)
    # Allocate a tight local raster, preserving shapes on either side of the origin.
    if p["shape"] == "rectangle":
        x, y, w, h = p["rectangle"]
        bounds = [x, y, x + w - 1, y + h - 1]
    elif p["shape"] == "circle":
        x, y = p["center"]
        r = p["radius"]
        bounds = [x - r, y - r, x + r, y + r]
    else:
        points = np.asarray(p["vertices"], float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Polygon needs at least three finite local vertices")
        bounds = [*points.min(axis=0), *points.max(axis=0)]
    bounds = np.asarray(bounds, float)
    if not np.isfinite(bounds).all() or np.max(np.abs(bounds)) > 1_000_000:
        raise ValueError("Local region bounds must be finite and within one million pixels")
    left, top = map(int, np.floor(bounds[:2]))
    width, height = (np.ceil(bounds[2:]).astype(int) - [left, top] + 1).tolist()
    source = {
        "model": "image_region_v1",
        "image_size": [width, height],
        **copy.deepcopy(p),
        "invert": False,
    }
    region_size(source)
    if p["shape"] == "rectangle":
        source["rectangle"] = [
            p["rectangle"][0] - left,
            p["rectangle"][1] - top,
            *p["rectangle"][2:],
        ]
    elif p["shape"] == "circle":
        source["center"] = [p["center"][0] - left, p["center"][1] - top]
    else:
        source["vertices"] = (np.asarray(p["vertices"]) - [left, top]).tolist()
    # Validate active geometry before constructing a lazy, serializable region tree.
    rasterize(source, (height, width))
    matrix = np.asarray(pose["source_to_target"]) @ local_offset(left, top, 0)
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
    )
    denominators = corners @ matrix[2]
    if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
        raise ValueError("Region crosses the projective pose horizon")
    region = {
        "model": "image_region_v1",
        "image_size": pose["target_size"],
        "shape": "transformed",
        "source": source,
        "source_to_target": matrix.tolist(),
        "interpolation": "nearest",
        "threshold": 0.5,
        "invert": p["invert"],
        "pose": pose,
    }
    return region


class ShapeMask(ShapeRegion):
    type_id, title = "mask.shape", "Shape mask"
    subtype = "Masks"
    inputs = (Port("image"), Port("pose", "data", required=False))
    editor_kind = "mask"
    parameter_help = {
        **ShapeRegion.parameter_help,
        "rectangle": "[x,y,width,height] in image pixels, or local pixels when pose is connected.",
        "center": "Image-space circle center, or local coordinates when pose is connected.",
        "vertices": "Image-space vertices, or local coordinates when pose is connected.",
    }

    def process(self, inputs):
        if "pose" not in inputs:
            return super().process(inputs)
        region = relative_region(inputs["image"], inputs["pose"], self.parameters)
        result = region_result(inputs["image"], region)
        draw_pose(result.outputs["overlay"], region["pose"])
        return result


POSE_BLOCKS = (DefinePose, OffsetPose, PoseFromFeature, PoseRegion, ShapeMask)
