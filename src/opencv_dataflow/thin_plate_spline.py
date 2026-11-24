"""Register an image to a target pixel grid using paired thin-plate spline controls."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import scalar_value
from .payloads import ValuePacket, check_spatial, frame, require_same_frame
from .processing import ImageTool, image_value
from .registration import BORDERS, INTERPOLATION
from .result_graphics import ACCEPTED

MAX_CONTROLS = 64
MAX_PROBES = 256
MAX_PIXELS = 4_194_304
MAX_EVALUATIONS = 64_000_000


def pixel_points(value, size, field, minimum, maximum):
    message = f"{field} requires {minimum}–{maximum} finite [x,y] points in its image grid"
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(message, field) from exc
    if minimum == 0 and points.shape == (0,):
        return points.reshape(0, 2)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not minimum <= len(points) <= maximum
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points >= size)
    ):
        raise ValidationError(message, field)
    return points


def native_controls(points, origin, scale, field):
    local = ((points - origin) / scale).astype(np.float32)
    singular = np.linalg.svd(local - local.mean(axis=0), compute_uv=False)
    if len(np.unique(local, axis=0)) != len(local) or singular[1] <= singular[0] * 1e-5:
        raise ValidationError("Control points must be distinct and span both axes", field)
    return local.reshape(1, -1, 2)


def sample_positions(spline, points, scale, origin):
    _, transformed = spline.applyTransformation(points)
    pixels = transformed.reshape(-1, 2).astype(np.float64) * scale + origin
    if not np.isfinite(pixels).all():
        raise ValidationError("Thin-plate spline produced nonfinite coordinates")
    return pixels


def sample_grid(spline, target_size, target_origin, source_origin, scale):
    width, height = target_size
    field = np.empty((height, width, 2), np.float32)
    rows_per_batch = max(1, 16_384 // width)
    for first in range(0, height, rows_per_batch):
        stop = min(first + rows_per_batch, height)
        yy, xx = np.indices((stop - first, width), dtype=np.float32)
        local = np.stack((xx, yy + first), axis=-1)
        local = ((local - target_origin) / scale).astype(np.float32).reshape(1, -1, 2)
        pixels = sample_positions(spline, local, scale, source_origin)
        if np.max(np.abs(pixels)) > 32766:
            raise ValidationError(
                "Thin-plate spline samples exceed the OpenCV remap coordinate range"
            )
        field[first:stop] = pixels.reshape(stop - first, width, 2)
    return field


def point_error(actual, expected):
    distance = np.linalg.norm(actual - expected, axis=1)
    return {"rms_pixels": float(np.sqrt(np.mean(distance**2))), "max_pixels": float(distance.max())}


class RegisterThinPlateSpline(ImageTool):
    type_id, title, subtype = (
        "cv.thin_plate_spline",
        "Register image (thin-plate spline)",
        "Registration",
    )
    description = "Warp a source image to a target grid from ordered control point pairs."
    inputs = (
        Port("image"),
        Port("target", required=False, required_when_connected=True),
        Port("source_points", "data", required=False, required_when_connected=True),
        Port("target_points", "data", required=False, required_when_connected=True),
        Port("probe_points", "data", required=False, required_when_connected=True),
        Port("expected_source_points", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("mapped_points", "data"), Port("report", "data"))
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "source_points": [],
        "target_points": [],
        "probe_points": [],
        "expected_source_points": [],
        "regularization": 0.0,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0,
    }
    parameter_choices = {"interpolation": tuple(INTERPOLATION), "border": tuple(BORDERS)}
    parameter_ranges = {"regularization": NumericRange(0, 1)}
    parameter_help = {
        "source_points": "4–64 ordered points in source pixels. A wired list takes precedence.",
        "target_points": "Corresponding points in target pixels, in the same order as the source.",
        "probe_points": "Up to 256 held-out target locations to map to source pixels.",
        "expected_source_points": "Optional known source locations for measuring probe error.",
        "regularization": "Smoothing in normalized coordinates. Zero interpolates controls.",
        "border_value": "Constant fill as a scalar or stored-channel values.",
    }
    input_help = {
        "image": "Source grayscale or color image to warp.",
        "target": "Optional target image. Its size and spatial frame define the output grid.",
        "source_points": "4–64 ordered [x,y] control points in source pixels.",
        "target_points": "Same control pairs in target pixels.",
        "probe_points": "Optional [x,y] locations in target pixels to map with the image sampler.",
        "expected_source_points": "Optional exact source positions of the target probe points.",
    }
    output_help = {
        "image": "Source pixels on the target grid, retaining depth and color encoding.",
        "mapped_points": "The probes mapped to source pixel coordinates by the same warp.",
        "report": "Control fit error and optional probe error, measured in source pixels.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "target", "source_points", "target_points"]

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        source_size = [int(n) for n in source.shape[:2][::-1]]
        target = inputs.get("target")
        target_context = self.input_payloads.get("target", {})
        point_context = self.input_payloads.get("target_points", {})
        target_spatial = target_context.get("spatial") or point_context.get("spatial")
        target_size = (
            [int(n) for n in target.shape[:2][::-1]]
            if target is not None
            else target_spatial["image_size"]
            if target_spatial
            else source_size
        )
        if any(
            min(size) < 1 or max(size) > 8192 or size[0] * size[1] > MAX_PIXELS
            for size in (source_size, target_size)
        ):
            raise ValidationError(
                "Spline registration supports up to 4.19M pixels and 8192 per axis"
            )
        spatial = target_spatial or frame(target_size, f"{self.id}:tps-target")
        source_context = self.input_payloads["image"]
        check_spatial(source_context, source_size)
        output_context = {"spatial": spatial}
        check_spatial(output_context, target_size)
        for context, fields in (
            (source_context, ("source_points", "expected_source_points")),
            (output_context, ("target_points", "probe_points")),
        ):
            for field in fields:
                require_same_frame(
                    (
                        ValuePacket(None, context),
                        ValuePacket(None, self.input_payloads.get(field, {})),
                    ),
                    message=f"{field} must be in the corresponding image grid",
                )

        source_points, target_points, probes, truth = (
            pixel_points(inputs.get(field, p[field]), size, field, minimum, maximum)
            for field, size, minimum, maximum in (
                ("source_points", source_size, 4, MAX_CONTROLS),
                ("target_points", target_size, 4, MAX_CONTROLS),
                ("probe_points", target_size, 0, MAX_PROBES),
                ("expected_source_points", source_size, 0, MAX_PROBES),
            )
        )
        if len(source_points) != len(target_points):
            raise ValidationError("Spline registration needs one target point per source point")
        if len(truth) and len(truth) != len(probes):
            raise ValidationError("Provide one expected source point for each target probe")
        if target_size[0] * target_size[1] * len(source_points) > MAX_EVALUATIONS:
            raise ValidationError(
                "Spline work limit exceeded. Reduce the target size or control count."
            )

        image = image_value(source)
        if image.dtype == np.float64 and p["interpolation"] == "cubic":
            raise ValidationError("Use linear or float32 for cubic remap of float64 images")
        border_value = scalar_value(p["border_value"])
        limit = np.iinfo(image.dtype) if image.dtype.kind in "ui" else np.finfo(image.dtype)
        if any(not limit.min <= value <= limit.max for value in border_value):
            raise ValidationError("Border values must fit the source sample range", "border_value")
        create = getattr(cv2, "createThinPlateSplineShapeTransformer", None)
        if create is None:
            raise ValidationError(
                "Thin-plate spline registration requires the OpenCV contrib shape module"
            )
        scale = float(max(*source_size, *target_size))
        source_origin = (np.asarray(source_size) - 1) / 2
        target_origin = (np.asarray(target_size) - 1) / 2
        source_controls = native_controls(source_points, source_origin, scale, "source_points")
        target_controls = native_controls(target_points, target_origin, scale, "target_points")
        spline = create(float(p["regularization"]))
        matches = [cv2.DMatch(i, i, 0.0) for i in range(len(source_points))]
        # Image interpolation pulls source samples at each target pixel.
        spline.estimateTransformation(target_controls, source_controls, matches)
        fitted = sample_positions(spline, target_controls, scale, source_origin)
        fit = point_error(fitted, source_points)
        if np.linalg.matrix_rank(fitted - fitted.mean(axis=0), tol=1e-5) < 2 or (
            not float(p["regularization"]) and fit["max_pixels"] > 0.05
        ):
            raise ValidationError(
                "Spline fit is unstable. Spread the control points farther apart."
            )
        mapped = (
            sample_positions(
                spline,
                ((probes - target_origin) / scale).astype(np.float32).reshape(1, -1, 2),
                scale,
                source_origin,
            )
            if len(probes)
            else np.empty((0, 2))
        )
        field = sample_grid(spline, target_size, target_origin, source_origin, scale)
        warped = cv2.remap(
            image,
            field,
            None,
            INTERPOLATION[p["interpolation"]],
            borderMode=BORDERS[p["border"]],
            borderValue=border_value,
        )
        if not np.isfinite(warped).all():
            raise ValidationError("Spline warp produced nonfinite pixel values")

        report = {
            "model": "image_thin_plate_spline_v1",
            "direction": "target_to_source",
            "source_size": source_size,
            "target_size": target_size,
            "control_count": len(source_points),
            "probe_count": len(probes),
            "control_error": fit,
            "probe_error": point_error(mapped, truth) if len(truth) else None,
        }
        drawings = {}
        if self.generate_graphics:
            drawings["target controls"] = GraphicDrawing(
                tuple(graphic_circle(x, y, 3, ACCEPTED) for x, y in target_points), "image", True
            )
            if len(probes):
                drawings["target probes"] = GraphicDrawing(
                    tuple(graphic_circle(x, y, 5, (60, 220, 255)) for x, y in probes), "image", True
                )
        source_frame = source_context.get("spatial")
        alpha = (
            {"alpha_representation": source_context["alpha_representation"]}
            if ("alpha_representation" in source_context)
            else {}
        )
        return Result(
            {
                "image": ImageArray(warped, image_format(source)),
                "mapped_points": mapped.tolist(),
                "report": report,
            },
            drawings=drawings,
            payloads={
                "image": {**output_context, **alpha},
                "mapped_points": {"spatial": source_frame} if source_frame else {},
                "report": output_context,
            },
            message=f"{len(source_points)} control pairs · fit RMS {fit['rms_pixels']:.4g} px",
        )
