"""Prewarp an ideal image into a calibrated physical projector raster."""

import math

import cv2
import numpy as np

from .calibration_revisions import REVISION, digest
from .camera_projector import RIG_MODEL
from .core import Port, Result
from .flow_warp import HALO, sample_mask
from .geometry3d_common import Tool3D, require_pinhole
from .graphics_layers import GraphicDrawing
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import finite, image_value
from .registration import INTERPOLATION
from .remap_maps import COORDINATE_MAP_TAG, optical_remap_payload
from .selection_remap import boundaries

_MAX_PIXELS = 16_777_216
_BAND_PIXELS = 131_072
_UNSUPPORTED = -16.0


def _projector_parameters(record):
    if isinstance(record, dict) and record.get("model") == RIG_MODEL:
        record = record.get("projector")
    if not isinstance(record, dict) or not isinstance(record.get("frame"), str):
        raise ValidationError(
            "Connect a named pinhole projector model or calibrated rig", "projector"
        )
    return record, require_pinhole(record)


def _pattern_intrinsics(pattern_camera, pattern_frame, source_size, k, output_size):
    if pattern_camera is not None:
        matrix, distortion, size = require_pinhole(pattern_camera)
        if size != source_size or distortion.any():
            raise ValidationError(
                "The ideal pattern needs matching dimensions and zero distortion", "pattern_camera"
            )
        if pattern_frame and pattern_camera.get("frame") != pattern_frame:
            raise ValidationError(
                "Pattern optical frame differs from its ideal camera model", "pattern_camera"
            )
        return matrix
    scale = np.asarray(source_size) / output_size
    scaled = k.copy()
    scaled[0, 0] *= scale[0]
    scaled[1, 1] *= scale[1]
    scaled[:2, 2] = (scaled[:2, 2] + 0.5) * scale - 0.5
    return scaled


def _inverse_maps(k, distortion, rotation, ideal, size, cancelled):
    width, height = size
    mx = np.empty((height, width), np.float32)
    my = np.empty_like(mx)
    rows = max(1, _BAND_PIXELS // width)
    finite_pixels = front_pixels = 0
    for start in range(0, height, rows):
        if cancelled():
            return None
        end = min(start + rows, height)
        band_camera = k.copy()
        # The band's local row zero corresponds to a row in the physical raster.
        band_camera[1, 2] -= start
        x, y = cv2.initInverseRectificationMap(
            band_camera, distortion, rotation, ideal, (width, end - start), cv2.CV_32FC1
        )
        real = np.isfinite(x) & np.isfinite(y)
        # The homogeneous map can still be finite for a ray going behind the virtual camera.
        # Express [u, v, 1] in that camera's normalized axes and look back along raw Z.
        with np.errstate(over="ignore", invalid="ignore"):
            z = (
                (x - ideal[0, 2]) * (rotation[0, 2] / ideal[0, 0])
                + (y - ideal[1, 2]) * (rotation[1, 2] / ideal[1, 1])
                + rotation[2, 2]
            )
        front = real & (z > 1e-8)
        finite_pixels += int(real.sum())
        front_pixels += int(front.sum())
        # Bound coordinates before passing them to OpenCV's pixel sampling and packing.
        usable = front & (x >= -1) & (y >= -1) & (x <= 16_384) & (y <= 16_384)
        mx[start:end] = np.where(usable, x, _UNSUPPORTED)
        my[start:end] = np.where(usable, y, _UNSUPPORTED)
    return mx, my, finite_pixels, front_pixels


def _roundtrip(x, y, valid, raw, distortion, ideal, rotation_vector):
    rows, columns = (
        np.unique(np.linspace(0, length - 1, min(length, 17), dtype=int)) for length in valid.shape
    )
    yy, xx = np.meshgrid(rows, columns, indexing="ij")
    keep = valid[yy, xx]
    py, px = yy[keep], xx[keep]
    if not len(px):
        first_y, first_x = np.unravel_index(valid.argmax(), valid.shape)
        py, px = np.array([first_y]), np.array([first_x])
    ideal_rays = np.column_stack(
        (
            (x[py, px].astype(np.float64) - ideal[0, 2]) / ideal[0, 0],
            (y[py, px].astype(np.float64) - ideal[1, 2]) / ideal[1, 1],
            np.ones(len(px)),
        )
    )
    projected, _ = cv2.projectPoints(ideal_rays, -rotation_vector, np.zeros(3), raw, distortion)
    differences = projected[:, 0] - np.column_stack((px, py))
    error = np.linalg.norm(differences, axis=1)
    rms = float(np.sqrt(np.mean(error**2)))
    maximum = float(np.max(error))
    return {
        "samples": len(px),
        "rms_px": rms if math.isfinite(rms) else None,
        "maximum_px": maximum if math.isfinite(maximum) else None,
    }


class InverseRectifyProjectorPattern(Tool3D):
    type_id = "3d.projector_inverse_rectify"
    title, subtype = "Prewarp projector pattern", "Structured light"
    description = "Map an ideal pattern into a calibrated pinhole projector's native pixel grid."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "pattern"
    inputs = (
        Port("pattern"),
        Port("projector", "data"),
        Port("pattern_camera", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("valid_mask", required=False),
        Port("map_x", required=False),
        Port("map_y", required=False),
        Port("report", "data"),
    )
    defaults = {
        "rotation_vector": [0.0, 0.0, 0.0],
        "interpolation": "linear",
        "border_value": 0.0,
        "minimum_valid_fraction": 0.0,
        "maximum_roundtrip_error_px": 0.5,
        "maximum_pixels": 8_388_608,
    }
    parameter_choices = {"interpolation": tuple(INTERPOLATION)}
    parameter_ranges = {
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_roundtrip_error_px": NumericRange(0, 1000),
        "maximum_pixels": NumericRange(1, _MAX_PIXELS, whole=True),
    }
    parameter_help = {
        "rotation_vector": "Raw projector rays to ideal pattern axes. Rodrigues vector in radians.",
        "interpolation": "Sampling kernel. Validity requires its supported ideal source footprint.",
        "border_value": "Constant fill for unsupported projector pixels, in the pattern encoding.",
        "minimum_valid_fraction": "Required fraction of the projector raster with valid samples.",
        "maximum_roundtrip_error_px": "Sampled reprojection error limit. Zero disables it.",
        "maximum_pixels": "Independent pixel limit for the pattern and projector canvases.",
    }
    input_help = {
        "pattern": "Ideal pattern image with its original encoding.",
        "projector": "A named pinhole model or the output of Calibrate camera and projector.",
        "pattern_camera": "Optional distortion-free pinhole intrinsics on the pattern grid.",
    }
    output_help = {
        "image": "Prewarped raster on the physical projector grid.",
        "valid_mask": "Forward-visible projector pixels with complete pattern sampling support.",
        "map_x": "Ideal pattern X to sample at each projector pixel. Invalid entries are -16.",
        "map_y": "Ideal pattern Y to sample at each projector pixel. Invalid entries are -16.",
        "report": "Projection direction, coverage, grid dimensions and sampled round-trip error.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        vector = np.asarray(p["rotation_vector"], dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(abs(vector) > 2 * np.pi):
            raise ValidationError(
                "Enter three finite rotation components in radians within ±2π", "rotation_vector"
            )

    def process(self, inputs):
        p = self.parameters
        source = image_value(inputs["pattern"])
        source_size = source.shape[1], source.shape[0]
        model, (k, distortion, size) = _projector_parameters(inputs["projector"])
        if (
            max(source_size) > 16384
            or max(math.prod(size), math.prod(source_size)) > p["maximum_pixels"]
        ):
            raise ValidationError(
                "Pattern or projector exceeds the configured pixel limit", "maximum_pixels"
            )
        method = p["interpolation"]
        if source.dtype == np.float64 and method == "cubic":
            raise ValidationError("Use linear or float32 for Float64 cubic remap", "interpolation")
        source_payload = self.input_payloads["pattern"]
        pattern_camera = inputs.get("pattern_camera")
        ideal = _pattern_intrinsics(
            pattern_camera, source_payload.get("optical_frame"), source_size, k, size
        )
        rotation_vector = np.asarray(p["rotation_vector"], np.float64)
        rotation = cv2.Rodrigues(rotation_vector)[0]
        report = {
            "model": "projector_inverse_rectification_v1",
            "direction": "raw_projector_pixel_to_ideal_pattern_sample",
            "projector_frame": model["frame"],
            "projector_size": list(size),
            "pattern_size": list(source_size),
            "pattern_camera_matrix": ideal.tolist(),
            "projector_to_pattern_rotation": rotation.tolist(),
            "interpolation": method,
            "invalid_coordinate": _UNSUPPORTED,
        }
        outputs = dict.fromkeys((port.name for port in self.outputs), NO_VALUE)
        outputs["report"] = report
        payloads = {"report": {}}

        def failure(code, message):
            report["status"] = "failed"
            return Result(
                outputs,
                payloads=payloads,
                passed=False,
                score=report.get("valid_fraction", 0.0),
                fail_reasons=[code],
                message=message,
            )

        maps = _inverse_maps(k, distortion, rotation, ideal, size, self._cancelled)
        if maps is None or self._cancelled():
            return failure("projector_cancelled", "Projector warp cancelled")
        x, y, finite_count, front_count = maps
        mask = sample_mask(
            np.ones(source.shape[:2], np.uint8),
            x,
            y,
            "nearest" if method == "nearest" else "linear",
            HALO[method],
        )
        valid_count = int(mask.sum())
        fraction = valid_count / mask.size
        report.update(
            finite_map_pixels=finite_count,
            forward_pixels=front_count,
            valid_pixels=valid_count,
            valid_fraction=fraction,
        )
        optics = {"frame": model["frame"]}
        if REVISION in model:
            optics[REVISION] = model[REVISION]
        projector_identity = [optics, list(size), k.tolist(), distortion.tolist()]
        spatial = frame(size, f"projector-raster:{digest(projector_identity)}")
        context = {**source_payload, "spatial": spatial}
        context.pop("pyramid", None)
        context.pop(COORDINATE_MAP_TAG, None)
        optical_remap_payload(context, optics)
        outputs["valid_mask"] = np.uint8(mask) * 255
        payloads["valid_mask"] = {
            key: value for key, value in context.items() if key != "alpha_representation"
        }
        if not valid_count or fraction < float(p["minimum_valid_fraction"]):
            return failure(
                "projector_support", f"Only {fraction:.1%} of the projector is supported"
            )
        check = _roundtrip(x, y, mask, k, distortion, ideal, rotation_vector)
        report["roundtrip"] = check
        limit = float(p["maximum_roundtrip_error_px"])
        if check["maximum_px"] is None or (limit and check["maximum_px"] > limit):
            return failure(
                "projector_roundtrip", "Sampled raw-projector reprojection exceeded the limit"
            )
        if self._cancelled():
            return failure("projector_cancelled", "Projector warp cancelled")

        x[~mask] = y[~mask] = _UNSUPPORTED
        fill = finite(p["border_value"], "Border value")
        if source.ndim == 3:
            fill = (fill,) * source.shape[2]
        warped = cv2.remap(
            source, x, y, INTERPOLATION[method], borderMode=cv2.BORDER_CONSTANT, borderValue=fill
        )
        payloads["image"] = context
        outputs["image"] = ImageArray(warped, image_format(inputs["pattern"]))
        if {"map_x", "map_y"} & set(self.exposed_outputs):
            map_info = {
                "model": "image_coordinate_map_v1",
                "source_size": list(source_size),
                "output_size": list(size),
                "source_spatial": source_payload["spatial"],
                "output_spatial": spatial,
                "output_optics": optics,
                "sampling": {"interpolation": method, "border": "constant"},
            }
            map_info["id"] = digest([map_info, report])
            for axis, values in (("x", x), ("y", y)):
                name = f"map_{axis}"
                if name in self.exposed_outputs:
                    outputs[name] = values
                    payloads[name] = {
                        "spatial": spatial,
                        COORDINATE_MAP_TAG: {**map_info, "axis": axis},
                    }
        report["status"] = "ok"
        drawings = {}
        if self.generate_graphics:
            objects = boundaries(outputs["valid_mask"], 1024)
            if objects:
                drawings["Projector support"] = GraphicDrawing(objects, "image", from_output=True)
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            score=fraction,
            message=(
                f"{fraction:.1%} projector support, {check['maximum_px']:.4g} px sampled closure"
            ),
        )
