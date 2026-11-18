"""Affine pose-local sampling with explicit measured support and crop coordinates."""


import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .poses import pose_for_image
from .processing import ImageTool, geometry_result, image_value
from .registration import BORDERS, INTERPOLATION
from .selection import selection_mask
from .tickets import ticket_key


def check_tickets(image_context, pose_context, target_context):
    current = image_context.get("ticket")
    parent = image_context.get("inspection", {}).get("parent_ticket")
    allowed = [ticket_key(t) for t in (current, parent) if t]
    if allowed:
        for context in (pose_context, target_context):
            ticket = context.get("ticket")
            if ticket and ticket_key(ticket) not in allowed:
                raise ValidationError("Pose belongs to a different image ticket", "pose")


def sampling_geometry(source, pose, p):
    validated = pose_for_image(pose, source)
    matrix = np.asarray(validated["source_to_target"], np.float64)
    if not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValidationError(
            "Pose crop requires affine axes; use Rectify quadrilateral for perspective", "pose"
        )
    width, height = p["width"], p["height"]
    sx, sy = p["local_width"] / width, p["local_height"] / height
    local = np.array(
        [
            [sx, 0, p["center_x"] - p["local_width"] / 2 + sx / 2],
            [0, sy, p["center_y"] - p["local_height"] / 2 + sy / 2],
            [0, 0, 1],
        ]
    )
    mapping = matrix @ local
    corners = (
        np.array(
            [
                [-0.5, -0.5, 1],
                [width - 0.5, -0.5, 1],
                [width - 0.5, height - 0.5, 1],
                [-0.5, height - 0.5, 1],
            ]
        )
        @ mapping.T
    )
    if not np.isfinite(mapping).all() or np.max(np.abs(corners[:, :2])) > 10_000_000:
        raise ValidationError(
            "Crop footprint exceeds supported coordinates", "pose", "local_width", "local_height"
        )
    return mapping, local, validated, corners[:, :2]


def measured_support(support, mapping, size, interpolation):
    # Positive interpolation weights provide exact support for nearest/linear.
    # Higher-order negative lobes need a conservative full-kernel halo instead.
    if interpolation in ("cubic", "lanczos4"):
        radius = 2 if interpolation == "cubic" else 4
        support = cv2.erode(
            support,
            np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        method = cv2.INTER_NEAREST
    else:
        method = INTERPOLATION[interpolation]
    sampled = cv2.warpAffine(
        support.astype(np.float32) / 255,
        mapping[:2],
        size,
        flags=method | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.uint8(sampled >= 1 - 1e-6) * 255


class CropPose(ImageTool):
    type_id, title, subtype = "cv.crop_pose", "Crop at image pose", "Geometry"
    inputs = (
        Port("image"),
        Port("pose", "data", visual_type="pose"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("geometry", "data"),
        Port("valid_mask", visual_type="region"),
        Port("coverage", "number"),
    )
    process_inputs_readonly = parallel_safe = preview_safe = True
    defaults = {
        "width": 64,
        "height": 32,
        "local_width": 64.0,
        "local_height": 32.0,
        "center_x": 0.0,
        "center_y": 0.0,
        "interpolation": "linear",
        "bounds_policy": "strict",
        "border": "constant",
        "border_value": 0.0,
        "minimum_coverage": 1.0,
        "maximum_image_pixels": 16_777_216,
        "maximum_crop_pixels": 1_048_576,
    }
    parameter_choices = {
        "interpolation": tuple(INTERPOLATION),
        "bounds_policy": ("strict", "pad"),
        "border": tuple(BORDERS),
    }
    parameter_ranges = {
        "width": NumericRange(1, 4096, whole=True),
        "height": NumericRange(1, 4096, whole=True),
        "local_width": NumericRange(0.0001, 100000),
        "local_height": NumericRange(0.0001, 100000),
        "center_x": NumericRange(-100000, 100000),
        "center_y": NumericRange(-100000, 100000),
        "border_value": NumericRange(-1e12, 1e12),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_crop_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        "width": "Output columns. Local width is a pixel-cell span; centers use a "
        "half-pixel resize convention.",
        "height": "Output rows, independent of the inspected local height.",
        "local_width": "Inspected width in pose-local units, including the pose scale/shear.",
        "local_height": "Inspected height in pose-local units. Pose origin is the crop "
        "center by default.",
        "center_x": "Crop center offset along the pose's local X axis.",
        "center_y": "Crop center offset along the pose's local Y axis.",
        "bounds_policy": "Strict rejects any unsupported sample. Pad emits a crop and "
        "measured coverage judgment.",
        "border": "Extrapolation for image-edge samples in Pad mode; it never makes "
        "them measured support.",
        "border_value": "Stored source units; also fills excluded mask/region/transparent "
        "samples before sampling.",
        "minimum_coverage": "Inclusive supported fraction required to pass in Pad mode.",
        "interpolation": "Higher-order interpolation uses a conservative support halo, not a "
        "ringing coverage estimate.",
    }
    input_help = {
        "pose": "Required affine image_pose_v1. Shared-root poses are mapped into this image grid.",
        "mask": "Selected measured pixels; intersected with Region and nonzero alpha.",
        "region": "Selection in this image frame; excluded pixels are replaced before "
        "interpolation.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (rule.whole and type(p[key]) is not int):
                raise ValidationError(
                    "Use an integer" if rule.whole else "Use a numeric value", key
                )
        if p["width"] * p["height"] > p["maximum_crop_pixels"]:
            raise ValidationError(
                "Crop exceeds its pixel allowance", "width", "height", "maximum_crop_pixels"
            )

    def process(self, inputs):
        source, pose, p = inputs["image"], inputs["pose"], self.parameters
        image = image_value(source)
        if (
            max(image.shape[:2]) > 16384
            or image.shape[0] * image.shape[1] > p["maximum_image_pixels"]
        ):
            raise ValidationError(
                "Source exceeds pose-crop image allowance", "maximum_image_pixels"
            )
        context = self.input_payloads.get("image", {})
        check_tickets(
            context,
            self.input_payloads.get("pose", {}),
            pose.get("target_payload", {}) if isinstance(pose, dict) else {},
        )
        mapping, local, validated, corners = sampling_geometry(source, pose, p)
        value = p["border_value"]
        if image.dtype.kind in "iu" and (
            value % 1 or not np.iinfo(image.dtype).min <= value <= np.iinfo(image.dtype).max
        ):
            raise ValidationError("Border value must fit source integer depth", "border_value")
        selected = selection_mask(inputs, source)
        selected = np.full(image.shape[:2], 255, np.uint8) if selected is None else selected.copy()
        if image.ndim == 3 and image.shape[2] == 4:
            selected[image[..., 3] == 0] = 0
        size = (p["width"], p["height"])
        valid = measured_support(selected, mapping, size, p["interpolation"])
        coverage = float(np.count_nonzero(valid) / valid.size)
        if p["bounds_policy"] == "strict" and not np.all(valid):
            raise ValidationError(
                "Pose crop has unsupported samples; reduce it or select Pad", "bounds_policy"
            )
        working = image
        if not np.all(selected):
            working = image.copy()
            working[selected == 0] = value
        fill = (value,) * image.shape[2] if image.ndim == 3 else value
        output = cv2.warpAffine(
            working,
            mapping[:2],
            size,
            flags=INTERPOLATION[p["interpolation"]] | cv2.WARP_INVERSE_MAP,
            borderMode=BORDERS[p["border"]],
            borderValue=fill,
        )
        if not np.isfinite(output).all():
            raise ValidationError("Pose sampling produced nonfinite image values")
        result = geometry_result(output, source, mapping)
        result.outputs.update(valid_mask=valid, coverage=coverage)
        result.outputs["geometry"].update(
            output_to_pose=local.tolist(),
            pose_to_input=validated["source_to_target"],
            source_corners=corners.tolist(),
            coverage=coverage,
            support_policy="native_positive_kernel"
            if p["interpolation"] in ("nearest", "linear")
            else "conservative_kernel_halo",
        )
        result.passed = coverage >= p["minimum_coverage"]
        result.score = coverage
        result.fail_reasons = (
            []
            if result.passed
            else [
                {
                    "code": "crop.coverage",
                    "message": f"Measured support {coverage:.3%}; "
                    f"minimum {p['minimum_coverage']:.3%}",
                }
            ]
        )
        result.message = f"{size[0]} x {size[1]} pose crop; {coverage:.1%} measured support"
        if self.generate_graphics:
            color = (70, 220, 130) if result.passed else (60, 60, 235)
            result.drawings["pose crop"] = GraphicDrawing(
                (graphic_object("polyline", corners, color, closed=True),)
            )
        return result
