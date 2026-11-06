"""Build reusable destination-to-source maps for pinhole lens correction."""

import hashlib
import json
import math

import cv2
import numpy as np

from .calibration import camera_parameters
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import finite

_MAX_AXIS = 16_384
MAX_MAP_PIXELS = 64_000_000
_TAG = "lens_remap_map"
_COLORMAPS = {
    name.removeprefix("COLORMAP_").lower(): getattr(cv2, name)
    for name in dir(cv2)
    if name.startswith("COLORMAP_")
}


def valid_lens_size(size):
    return (
        isinstance(size, list)
        and len(size) == 2
        and all(type(v) is int and 1 <= v <= _MAX_AXIS for v in size)
    )


def _signature(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _map_frame(info):
    return frame(info["output_size"], f"lens-map:{info['id']}")


def lens_map_payloads(info):
    spatial = _map_frame(info)
    return {
        f"map_{axis}": {"spatial": spatial, _TAG: {**info, "axis": axis}} for axis in ("x", "y")
    }


def lens_map_pair(block, output_size):
    x, y = (block.input_payloads[name] for name in ("map_x", "map_y"))
    if _TAG not in x and _TAG not in y:
        return None
    info, other = x.get(_TAG), y.get(_TAG)
    if (
        not isinstance(info, dict)
        or info.get("axis") != "x"
        or not isinstance(info.get("id"), str)
        or {**info, "axis": "y"} != other
    ):
        raise ValidationError("Connect matching X and Y lens maps from the same correction")
    if (
        not valid_lens_size(info.get("source_size"))
        or not valid_lens_size(output_size)
        or info.get("output_size") != output_size
        or x.get("spatial") != _map_frame(info)
        or y.get("spatial") != _map_frame(info)
    ):
        raise ValidationError("Lens maps must retain their original pixel grid")
    return info


def lens_remap_payload(block, source_size, output_size):
    info = lens_map_pair(block, output_size)
    if info is None:
        return {}
    if info["source_size"] != source_size:
        raise ValidationError("Source image dimensions do not match the lens map calibration")
    payload = dict(block.input_payloads["image"])
    spatial = payload["spatial"]
    source_frame = spatial if spatial["root_id"] is not None else [block.id, spatial]
    identity = _signature([source_frame, info["id"]])
    payload["spatial"] = frame(output_size, f"lens-remap:{identity}")
    payload.pop("pyramid", None)
    payload.pop(_TAG, None)
    return {"image": payload}


class BuildLensMaps(Block):
    type_id, title = "calibration.lens_maps", "Build lens correction maps"
    category, block_type, subtype = "Calibration", "Calibration", "Lens distortion"
    inputs = (Port("calibration", "data"),)
    outputs = (
        Port("map_x"),
        Port("map_y"),
        Port("rectified_calibration", "data"),
        Port("valid_roi", "data", required=False),
    )
    defaults = {
        "width": 0,
        "height": 0,
        "camera_matrix_mode": "scaled",
        "new_camera_matrix": [],
        "alpha": 0.0,
        "center_principal_point": False,
        "rotation_vector": [0.0, 0.0, 0.0],
        "max_rms_pixels": 1.0,
    }
    parameter_choices = {"camera_matrix_mode": ("scaled", "optimal", "custom")}
    parameter_ranges = {
        "width": NumericRange(0, _MAX_AXIS, whole=True),
        "height": NumericRange(0, _MAX_AXIS, whole=True),
        "alpha": NumericRange(0, 1),
        "max_rms_pixels": NumericRange(1e-9),
    }
    parameter_help = {
        "width": "Output width in pixels. Zero keeps the calibrated width.",
        "height": "Output height in pixels. Zero keeps the calibrated height.",
        "camera_matrix_mode": "Scale the original intrinsics, optimize the crop, or use a matrix.",
        "new_camera_matrix": "Custom output intrinsics as a 3×3 pinhole camera matrix.",
        "alpha": "Optimal mode crop. Zero favors valid pixels and one retains more of the source.",
        "center_principal_point": "Center the principal point in optimal mode.",
        "rotation_vector": "Optional rectification rotation as a Rodrigues vector in radians.",
        "max_rms_pixels": "Maximum source calibration reprojection RMS in pixels.",
    }

    def process(self, inputs):
        p, profile = self.parameters, inputs["calibration"]
        matrix, distortion = camera_parameters(profile)
        source = profile["image_size"]
        if not valid_lens_size(source):
            raise ValidationError("Lens calibration needs valid [width, height] dimensions")
        size = [int(p["width"]) or source[0], int(p["height"]) or source[1]]
        if size[0] * size[1] > MAX_MAP_PIXELS:
            raise ValidationError("Lens maps are limited to 64 million output pixels")
        rms = finite(profile["rms_pixels"], "Lens calibration RMS", 0)
        if rms > float(p["max_rms_pixels"]):
            raise ValidationError("Lens calibration exceeds the allowed reprojection error")
        units = profile.get("units")
        if not isinstance(units, str) or not units.strip():
            raise ValidationError("Lens calibration must specify units")
        vector = np.asarray(p["rotation_vector"], dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValidationError(
                "Use three finite Rodrigues rotation components", "rotation_vector"
            )
        rotation = cv2.Rodrigues(vector)[0]

        roi = NO_VALUE
        mode = p["camera_matrix_mode"]
        if mode == "optimal":
            target, bounds = cv2.getOptimalNewCameraMatrix(
                matrix,
                distortion,
                tuple(source),
                float(p["alpha"]),
                tuple(size),
                centerPrincipalPoint=p["center_principal_point"],
            )
            if not np.any(vector):
                roi = list(bounds)
        elif mode == "custom":
            target = np.asarray(p["new_camera_matrix"], dtype=np.float64)
        else:
            target = matrix.copy()
            target[:2] *= (np.asarray(size) / source)[:, None]

        rectified = {
            "model": "opencv_pinhole_v1",
            "camera_matrix": target.tolist(),
            "distortion": [0.0] * 5,
            "image_size": size,
            "rms_pixels": rms,
            "units": units,
            "rectification_rotation": rotation.tolist(),
        }
        camera_parameters(rectified)
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, distortion, rotation, target, tuple(size), cv2.CV_32FC1
        )
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValidationError("Lens correction produced nonfinite map coordinates")
        identity = _signature(
            [
                matrix.tolist(),
                distortion.tolist(),
                source,
                rectified["camera_matrix"],
                size,
                rectified["rectification_rotation"],
            ]
        )
        info = {
            "id": identity,
            "source_size": source,
            "output_size": size,
            "rectified_calibration": rectified,
            "valid_roi": None if roi is NO_VALUE else roi,
        }
        return Result(
            {"map_x": map_x, "map_y": map_y, "rectified_calibration": rectified, "valid_roi": roi},
            payloads=lens_map_payloads(info),
            message=f"{size[0]} × {size[1]} lens correction maps",
        )


def _displacement_arrows(dx, dy, valid, ratio_x, ratio_y, spacing, scale):
    height, width = dx.shape
    step = max(spacing, math.ceil(max(width, height) / 64))
    arrows = []
    for row in range(min(step // 2, height - 1), height, step):
        for column in range(min(step // 2, width - 1), width, step):
            if not valid[row, column]:
                continue
            vx = float(dx[row, column]) * scale / ratio_x
            vy = float(dy[row, column]) * scale / ratio_y
            length = math.hypot(vx, vy)
            if length < 1:
                continue
            head = (column + vx, row + vy)
            tip = min(0.3, 5 / length)
            left = (head[0] - tip * (vx - vy / 2), head[1] - tip * (vy + vx / 2))
            right = (head[0] - tip * (vx + vy / 2), head[1] - tip * (vy - vx / 2))
            arrows.append(
                graphic_object(
                    "polyline",
                    ((column, row), head, left, head, right),
                    (255, 255, 255),
                    thickness=1,
                )
            )
    return tuple(arrows)


class LensMapDisplacement(Block):
    type_id, title = "calibration.lens_map_displacement", "Inspect lens map displacement"
    category, block_type, subtype = "Calibration", "Calibration", "Lens distortion"
    inputs = (Port("map_x"), Port("map_y"))
    outputs = (
        Port("heatmap"),
        Port("magnitude"),
        Port("valid_mask"),
        Port("summary", "data"),
        Port("maximum", "number"),
        Port("valid_fraction", "number"),
    )
    defaults = {"colormap": "turbo", "heatmap_max": 0.0, "arrow_spacing": 0, "arrow_scale": 1.0}
    parameter_choices = {"colormap": tuple(_COLORMAPS)}
    parameter_ranges = {
        "heatmap_max": NumericRange(1e-6, float(np.finfo(np.float32).max), zero_allowed=True),
        "arrow_spacing": NumericRange(0, _MAX_AXIS, whole=True),
        "arrow_scale": NumericRange(0.01, 100),
    }
    parameter_help = {
        "colormap": "Color palette used to display displacement magnitude.",
        "heatmap_max": "Source-pixel displacement at the top of the palette. Zero fits the data.",
        "arrow_spacing": "Minimum sample spacing in output pixels. Zero hides direction arrows.",
        "arrow_scale": "Display multiplier for arrow length. Measurements retain their scale.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        x, y = (np.asarray(inputs[name]) for name in ("map_x", "map_y"))
        if (
            x.ndim != 2
            or y.shape != x.shape
            or x.dtype != np.float32
            or y.dtype != np.float32
            or x.size > MAX_MAP_PIXELS
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValidationError("Use matching, finite float32 lens map planes")
        height, width = x.shape
        info = lens_map_pair(self, [width, height])
        if info is None:
            raise ValidationError("Connect X and Y maps from Build lens correction maps")
        source_width, source_height = info["source_size"]
        ratio_x, ratio_y = source_width / width, source_height / height
        dx = x - (np.arange(width) * ratio_x).astype(np.float32)
        dy = y - (np.arange(height) * ratio_y).astype(np.float32)[:, None]
        with np.errstate(over="ignore"):
            magnitude = np.hypot(dx, dy)
        if not np.isfinite(magnitude).all():
            raise ValidationError("Lens map displacement exceeds float32 range")

        # Tolerate roundoff at the outermost source pixel centers.
        eps = 1e-5
        valid = (
            (x >= -eps)
            & (x <= source_width - 1 + eps)
            & (y >= -eps)
            & (y <= source_height - 1 + eps)
        )
        valid_count = int(np.count_nonzero(valid))
        fraction = valid_count / x.size
        minimum, maximum, _, _ = cv2.minMaxLoc(magnitude)
        mean, stddev = cv2.meanStdDev(magnitude)
        mean, stddev = float(mean.item()), float(stddev.item())
        limit = float(self.parameters["heatmap_max"]) or max(maximum, 1e-5)
        levels = cv2.convertScaleAbs(np.minimum(magnitude, limit), alpha=255 / limit)
        heatmap = cv2.applyColorMap(levels, _COLORMAPS[self.parameters["colormap"]])
        outputs = {
            "heatmap": heatmap,
            "magnitude": magnitude,
            "valid_mask": valid.astype(np.uint8) * 255,
            "summary": {
                "minimum_source_pixels": minimum,
                "maximum_source_pixels": maximum,
                "mean_source_pixels": mean,
                "rms_source_pixels": math.hypot(mean, stddev),
                "heatmap_max_source_pixels": limit,
                "valid_count": valid_count,
                "pixel_count": x.size,
                "valid_fraction": fraction,
            },
            "maximum": maximum,
            "valid_fraction": fraction,
        }
        drawings = {}
        spacing = int(self.parameters["arrow_spacing"])
        if self.generate_graphics and spacing:
            arrows = _displacement_arrows(
                dx, dy, valid, ratio_x, ratio_y, spacing, float(self.parameters["arrow_scale"])
            )
            if arrows:
                drawings["Displacement vectors"] = GraphicDrawing(arrows, "map_x")
        payload = {key: value for key, value in self.input_payloads["map_x"].items() if key != _TAG}
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            message=f"Peak {maximum:.3g} source px, {fraction:.1%} inside source",
        )
