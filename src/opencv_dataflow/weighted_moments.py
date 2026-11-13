"""Selected intensity/binary moments, stable centers and reusable image-plane poses."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .histogram import CHANNEL_CHOICES, HISTOGRAM_CHANNELS
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata
from .poses import local_offset, make_pose
from .processing import ImageTool, image_value
from .selection import selection_mask


def moment_plane(source, channel, units):
    encoding = image_format(source)
    if channel == "luminance":
        if source.ndim == 2:
            plane = np.asarray(source, np.float64)
        else:
            if encoding in ("hsv", "lab") and source.dtype == np.float64:
                raise ValidationError(
                    "Choose a channel or convert float64 HSV/Lab before luminance"
                )
            bgr = opencv_image(source).astype(np.float64)
            # Keep float64/HDR precision instead of rounding through an 8/32-bit gray image.
            plane = bgr[..., 0] * 0.114 + bgr[..., 1] * 0.587 + bgr[..., 2] * 0.299
    else:
        labels = HISTOGRAM_CHANNELS[encoding]
        if channel not in labels:
            raise ValidationError("Selected channel is unavailable in the image", "channel")
        plane = np.asarray(
            source if source.ndim == 2 else source[..., labels.index(channel)], np.float64
        )
    if units == "normalized" and source.dtype.kind == "u":
        plane = plane / np.iinfo(source.dtype).max
    if not np.isfinite(plane).all():
        raise ValidationError("Working luminance is nonfinite; rescale the input")
    return plane


def moment_analysis(weights, origin, scale, p, *, binary=False):
    local = cv2.moments(weights, binaryImage=binary)
    total = local["m00"]
    if not total or not all(math.isfinite(v) for v in local.values()):
        raise ValidationError("Moment accumulation is empty or nonfinite")
    if scale > np.finfo(np.float64).max / total:
        raise ValidationError("Moment mass exceeds the finite range; rescale the intensity weights")
    center = [local["m10"] / total + origin[0], local["m01"] / total + origin[1]]
    covariance = (
        np.array([[local["mu20"], local["mu11"]], [local["mu11"], local["mu02"]]], np.float64)
        / total
    )
    values, vectors = np.linalg.eigh(covariance)
    if values[0] < -1e-9 * max(float(values[1]), 1):
        raise ValidationError("Moment covariance lost numerical precision")
    values = np.maximum(values, 0)
    trace = float(values.sum())
    anisotropy = float((values[1] - values[0]) / trace) if trace else 0.0
    oriented = (
        values[1] > float(p["minimum_variance"])
        and values[1] > values[0]
        and anisotropy >= float(p["minimum_anisotropy"])
    )
    # Match image_pose_v1: positive angle is OpenCV counterclockwise, +y points down.
    angle = (-math.degrees(math.atan2(vectors[1, 1], vectors[0, 1]))) % 180 if oriented else None
    if angle is not None and angle >= 180:
        angle = 0.0
    moments = dict(local)
    # Translate only raw moments. Central/normalized moments retain the stable crop calculation.
    for key in local:
        if len(key) == 3 and key[0] == "m" and key[1:].isdigit():
            i, j = int(key[1]), int(key[2])
            moments[key] = sum(
                math.comb(i, a)
                * math.comb(j, b)
                * origin[0] ** (i - a)
                * origin[1] ** (j - b)
                * local[f"m{a}{b}"]
                for a in range(i + 1)
                for b in range(j + 1)
            )
    return {
        "center": center,
        "mass": total * scale,
        "covariance": covariance.tolist(),
        "major_variance": float(values[1]),
        "minor_variance": float(values[0]),
        "major_sigma": math.sqrt(values[1]),
        "minor_sigma": math.sqrt(values[0]),
        "anisotropy": anisotropy,
        "orientation_valid": bool(oriented),
        "angle_degrees": angle,
    }, moments


class IntensityMoments(ImageTool):
    type_id, title = "inspect.intensity_moments", "Intensity-weighted center and orientation"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("center", "data", required=False),
        Port("angle_degrees", "number", required=False),
        Port("pose", "data", required=False),
        Port("measurements", "data"),
        Port("mass", "number"),
        Port("x", "number", required=False),
        Port("y", "number", required=False),
        Port("moments", "data", required=False),
        Port("support", visual_type="region"),
        Port("weights", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "weighting": "intensity",
        "channel": "luminance",
        "intensity_units": "native",
        "polarity": "bright",
        "background": 0.0,
        "minimum_signal": 0.0,
        "minimum_support": 1,
        "minimum_mass": 0.0,
        "minimum_variance": 1e-9,
        "minimum_anisotropy": 0.05,
        "require_orientation": False,
        "pose_orientation": "principal_axis",
        "missing_center": "fail",
        "emit_weights": False,
        "show_geometric": True,
        "ellipse_sigma": 2.0,
    }
    parameter_choices = {
        "weighting": ("intensity", "uniform_foreground", "uniform_selection"),
        "channel": ("luminance", *CHANNEL_CHOICES),
        "intensity_units": ("native", "normalized"),
        "polarity": ("bright", "dark", "absolute"),
        "pose_orientation": ("principal_axis", "translation_only"),
        "missing_center": ("fail", "report"),
    }
    parameter_ranges = {
        "background": NumericRange(),
        "minimum_signal": NumericRange(0),
        "minimum_support": NumericRange(1, 4_194_304, whole=True),
        "minimum_mass": NumericRange(0),
        "minimum_variance": NumericRange(0),
        "minimum_anisotropy": NumericRange(0, 1),
        "ellipse_sigma": NumericRange(0.1, 10),
    }
    parameter_help = {
        "weighting": (
            "Intensity uses positive contrast weights. Uniform foreground weights qualifying "
            "pixels equally. Uniform selection measures the entire mask/region."
        ),
        "channel": "Use floating luminance (alpha ignored) or an explicit stored color channel.",
        "intensity_units": (
            "Native sample units, or integer full scale mapped to [0,1]; floats unchanged."
        ),
        "polarity": "Bright: value-background; dark: background-value; absolute: their magnitude.",
        "background": (
            "Constant baseline in the chosen intensity units; negative contrast is clipped."
        ),
        "minimum_signal": (
            "Keep contrast strictly above this value; surviving weights are not reduced."
        ),
        "minimum_support": "Minimum number of qualifying pixels for a valid center.",
        "minimum_mass": (
            "Require total weight strictly above this value; uniform modes use pixel count."
        ),
        "minimum_variance": (
            "Require major-axis variance above this pixel-squared value for orientation."
        ),
        "minimum_anisotropy": (
            "Inclusive minimum (major-minor)/(major+minor) for an unsigned principal axis."
        ),
        "require_orientation": (
            "Fail an otherwise valid center when its principal axis is ambiguous."
        ),
        "pose_orientation": (
            "Principal axis requires valid orientation; translation only defines zero angle."
        ),
        "missing_center": (
            "Fail missing/below-limit centers, or emit an informational report without a center."
        ),
        "emit_weights": "Emit the peak-normalized floating weight image and its scale metadata.",
        "show_geometric": "Draw a uniform-support center and covariance ellipse for comparison.",
        "ellipse_sigma": "Drawing radius in standard deviations; it is not a fitted part boundary.",
    }
    input_help = {
        "image": "Gray/color uint8, uint16 or float32/64, up to 4.2 MP and 16,384 per axis.",
        "mask": "Optional uint8 selection in the image's coordinate frame.",
        "region": "Optional circle, rectangle or polygon intersected with the mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "center", "angle_degrees", "pose", "measurements"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("require_orientation", "emit_weights", "show_geometric"):
            if type(p[key]) is not bool:
                raise ValidationError(f"{key} must be boolean", key)

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        h, w = source.shape[:2]
        if h * w > 4_194_304 or max(h, w) > 16384:
            raise ValidationError("Intensity moments are limited to 4.2 MP and 16,384 per axis")
        image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        mask = selection_mask(inputs, source)
        selected = np.ones((h, w), bool) if mask is None else mask != 0
        signal = None
        if p["weighting"] == "uniform_selection":
            support = selected
        else:
            plane = moment_plane(source, p["channel"], p["intensity_units"])
            with np.errstate(over="ignore", invalid="ignore"):
                contrast = plane - float(p["background"])
                signal = (
                    np.maximum(contrast, 0)
                    if p["polarity"] == "bright"
                    else np.maximum(-contrast, 0)
                    if p["polarity"] == "dark"
                    else np.abs(contrast)
                )
            support = selected & (signal > float(p["minimum_signal"]))
        count = int(support.sum())
        stats, geometric, moments, local_weights, origin = None, None, NO_VALUE, None, None
        scale = 1.0
        if count:
            x, y, width, height = cv2.boundingRect(support.astype(np.uint8))
            origin = [x, y]
            cropped = support[y : y + height, x : x + width]
            if p["weighting"] == "intensity":
                values = np.where(cropped, signal[y : y + height, x : x + width], 0)
                scale = float(values.max())
                if not math.isfinite(scale):
                    raise ValidationError(
                        "Intensity contrast exceeds the finite range; rescale the input"
                    )
                local_weights = values / scale
            else:
                local_weights = cropped.astype(np.float64)
            stats, moments = moment_analysis(
                local_weights, origin, scale, p, binary=p["weighting"] != "intensity"
            )
            geometric, _ = moment_analysis(cropped.astype(np.uint8), origin, 1, p, binary=True)
        mass = stats["mass"] if stats else 0.0
        center_valid = bool(
            stats and count >= int(p["minimum_support"]) and mass > float(p["minimum_mass"])
        )
        orientation_valid = center_valid and stats["orientation_valid"]
        reasons = []
        if not center_valid:
            reasons.append(
                "empty_moment_support"
                if not count
                else "insufficient_moment_support"
                if count < int(p["minimum_support"])
                else "insufficient_moment_mass"
            )
        elif p["require_orientation"] and not orientation_valid:
            reasons.append("ambiguous_moment_orientation")
        passed = not reasons or (not center_valid and p["missing_center"] == "report")
        report = {
            "model": "intensity_moments_v1",
            "weighting": p["weighting"],
            "channel": p["channel"],
            "intensity_units": p["intensity_units"],
            "polarity": p["polarity"],
            "background": float(p["background"]),
            "minimum_signal": float(p["minimum_signal"]),
            "selected_pixels": int(selected.sum()),
            "support_pixels": count,
            "mass": mass,
            "weight_scale": scale,
            "center_valid": center_valid,
            "orientation_valid": bool(orientation_valid),
            "weighted": stats,
            "geometric": geometric,
            "invalid_reasons": reasons,
            "center_offset": (
                [stats["center"][i] - geometric["center"][i] for i in (0, 1)] if stats else None
            ),
            "coordinates": "current input pixel centers; no automatic root/physical conversion",
            "angle_convention": "unsigned OpenCV counterclockwise degrees [0,180), +y down",
            "moments_basis": (
                "peak-normalized weights; raw moments global, central moments crop-stable"
            ),
        }
        values = {
            "image": source,
            "center": NO_VALUE,
            "angle_degrees": NO_VALUE,
            "pose": NO_VALUE,
            "measurements": report,
            "mass": mass,
            "x": NO_VALUE,
            "y": NO_VALUE,
            "moments": moments,
            "support": ImageArray(support.astype(np.uint8) * 255, "gray"),
            "weights": NO_VALUE,
            "overlay": NO_VALUE,
        }
        if center_valid:
            values["center"] = stats["center"]
            values["x"], values["y"] = stats["center"]
            if orientation_valid:
                values["angle_degrees"] = stats["angle_degrees"]
            if orientation_valid or p["pose_orientation"] == "translation_only":
                angle = stats["angle_degrees"] if p["pose_orientation"] == "principal_axis" else 0
                values["pose"] = make_pose(
                    local_offset(*stats["center"], angle),
                    [w, h],
                    metadata(source),
                    method="intensity_moments_v1",
                    orientation_mode=p["pose_orientation"],
                    angle_ambiguity_degrees=180 if p["pose_orientation"] == "principal_axis" else 0,
                    orientation_measured=p["pose_orientation"] == "principal_axis",
                )
        context = metadata(source)
        basis = {"weight_scale": scale, "coordinate_basis": "input_pixel_grid"}
        payloads = {"moments": {**context, "moments_basis": basis}}
        if p["emit_weights"]:
            weights = np.zeros((h, w), np.float64)
            if count:
                weights[y : y + height, x : x + width] = local_weights
            values["weights"] = ImageArray(weights, "gray", {**context, "moments_basis": basis})
        result = Result(
            values,
            payloads=payloads,
            passed=passed,
            score=stats["anisotropy"] if stats else 0.0,
            fail_reasons=[] if passed else reasons,
            message=f"{count} support pixels; "
            + (
                f"center ({stats['center'][0]:.3f}, {stats['center'][1]:.3f})"
                if center_valid
                else "no valid center"
            ),
        )
        if self.generate_graphics:
            objects = []
            if p["show_geometric"] and geometric:
                objects.extend(moment_graphics(geometric, p["ellipse_sigma"], (230, 170, 60)))
            if stats:
                objects.extend(
                    moment_graphics(
                        stats, p["ellipse_sigma"], (70, 220, 140) if passed else (60, 60, 240)
                    )
                )
            result.drawings["Intensity moments"] = GraphicDrawing(tuple(objects))
            overlay, identity = overlay_image(source), np.eye(3)
            for obj in objects:
                draw_object(overlay, obj, identity)
            values["overlay"] = overlay
            result.graphics["Intensity moments"] = overlay
        return result


def moment_graphics(stats, sigma, color):
    center = np.asarray(stats["center"])
    covariance = np.asarray(stats["covariance"])
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    circle = np.column_stack(
        (
            np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
        )
    )
    ring = (circle * (float(sigma) * np.sqrt(values))) @ vectors.T + center
    objects = [
        graphic_object("marker", [center], color),
        graphic_object("polyline", ring, color, closed=True, thickness=1),
    ]
    if stats["orientation_valid"]:
        extent = float(sigma) * math.sqrt(values[1]) * vectors[:, 1]
        objects.append(graphic_object("polyline", [center - extent, center + extent], color))
    return objects
