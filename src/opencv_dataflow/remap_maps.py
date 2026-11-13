"""Reusable destination-to-source maps and pinhole lens correction."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import camera_parameters
from .core import Block, Port, Result
from .geometry3d_common import matrix as finite_matrix
from .geometry3d_common import points as finite_points
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .parameters import COLORMAPS
from .payloads import NO_VALUE, check_spatial, frame
from .processing import finite, image_value

_MAX_AXIS = 16_384
_REMAP_MAX_AXIS = 32_766
MAX_MAP_PIXELS = 64_000_000
_TAG = "lens_remap_map"
_CORRECTION_ID = "lens_correction_id"
COORDINATE_MAP_TAG = "coordinate_remap"


def coordinate_map_info(block, output_size):
    """Optional general remap provenance, independent of a camera/lens model."""
    names = ("packed_map",) if "packed_map" in block.input_payloads else ("map_x", "map_y")
    payloads = [block.input_payloads[name] for name in names]
    tags = [payload.get(COORDINATE_MAP_TAG) for payload in payloads]
    if all(tag is None for tag in tags):
        return None
    first = tags[0]
    axes = ("packed",) if len(names) == 1 else ("x", "y")
    if (
        not isinstance(first, dict)
        or first.get("model") != "image_coordinate_map_v1"
        or not isinstance(first.get("id"), str)
        or first.get("output_size") != output_size
        or any(
            not isinstance(tag, dict) or tag != {**first, "axis": axis}
            for tag, axis in zip(tags, axes, strict=True)
        )
        or any(payload.get("spatial") != first.get("output_spatial") for payload in payloads)
    ):
        raise ValidationError("Connect matching coordinate maps on their original output grid")
    for name in ("source", "output"):
        size = first.get(f"{name}_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(axis) is not int or not 1 <= axis < 32767 for axis in size)
            or math.prod(size) > MAX_MAP_PIXELS
        ):
            raise ValidationError("Invalid coordinate map dimensions")
        check_spatial({"spatial": first.get(f"{name}_spatial")}, size)
        if first.get(f"{name}_spatial") is None:
            raise ValidationError("Coordinate maps require source and output frames")
    return first


@dataclass(frozen=True)
class PackedRemapMap:
    """Immutable runtime packet for OpenCV's fixed-point remap planes."""

    size: tuple[int, int]
    coordinates: bytes = field(repr=False)
    fractions: bytes | None = field(default=None, repr=False)

    def __post_init__(self):
        if (
            not isinstance(self.size, tuple)
            or len(self.size) != 2
            or any(type(axis) is not int or not 1 <= axis <= _REMAP_MAX_AXIS for axis in self.size)
            or math.prod(self.size) > MAX_MAP_PIXELS
        ):
            raise ValueError("Invalid packed remap dimensions")
        pixels = math.prod(self.size)
        if not isinstance(self.coordinates, bytes) or len(self.coordinates) != pixels * 4:
            raise ValueError("Packed remap requires two int16 coordinates per pixel")
        if self.fractions is not None and (
            not isinstance(self.fractions, bytes)
            or len(self.fractions) != pixels * 2
            or np.frombuffer(self.fractions, np.uint16).max() >= cv2.INTER_TAB_SIZE2
        ):
            raise ValueError("Invalid packed remap interpolation indices")

    def native_maps(self):
        width, height = self.size
        coordinates = np.frombuffer(self.coordinates, np.int16).reshape(height, width, 2)
        fractions = None
        if self.fractions is not None:
            fractions = np.frombuffer(self.fractions, np.uint16).reshape(height, width)
        return coordinates, fractions

    def describe(self):
        return {
            "model": "fixed_remap_map_v1",
            "size": list(self.size),
            "sampling": "nearest" if self.fractions is None else "subpixel",
            "bytes": len(self.coordinates) + (len(self.fractions) if self.fractions else 0),
        }


def split_remap_fields(inputs):
    x, y = (image_value(inputs[key], (np.float32,)) for key in ("map_x", "map_y"))
    if x.ndim != 2 or y.shape != x.shape or x.size > MAX_MAP_PIXELS:
        raise ValueError("Remap needs matching 2D float32 coordinate maps")
    return x, y


def valid_lens_size(size):
    return (
        isinstance(size, list)
        and len(size) == 2
        and all(type(v) is int and 1 <= v <= _MAX_AXIS for v in size)
    )


def _signature(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _lens_identity(matrix, distortion, source, target, size, rotation):
    return _signature(
        [matrix.tolist(), distortion.tolist(), source, target.tolist(), size, rotation.tolist()]
    )


def _corrected_frame(source_frame, identity, size):
    return frame(size, f"lens-remap:{_signature([source_frame, identity])}")


def _map_frame(info):
    return frame(info["output_size"], f"lens-map:{info['id']}")


def lens_map_payloads(info):
    spatial = _map_frame(info)
    payloads = {
        f"map_{axis}": {"spatial": spatial, _TAG: {**info, "axis": axis}}
        for axis in ("x", "y")
    }
    payloads["rectified_calibration"] = {_CORRECTION_ID: info["id"]}
    return payloads


def _check_lens_map_grid(info, output_size, *spatials):
    if (
        not valid_lens_size(info.get("source_size"))
        or not valid_lens_size(output_size)
        or info.get("output_size") != output_size
        or any(spatial != _map_frame(info) for spatial in spatials)
    ):
        raise ValidationError("Lens maps must retain their original pixel grid")


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
    _check_lens_map_grid(info, output_size, x.get("spatial"), y.get("spatial"))
    return info


def packed_lens_info(payload, output_size):
    info = payload.get(_TAG)
    if info is None:
        return None
    if (
        not isinstance(info, dict)
        or info.get("axis") != "packed"
        or not isinstance(info.get("id"), str)
    ):
        raise ValidationError("Invalid packed lens map metadata")
    _check_lens_map_grid(info, output_size, payload.get("spatial"))
    return info


def lens_remap_payload(block, source_size, output_size, info):
    if info is None:
        coordinate = coordinate_map_info(block, output_size)
        if coordinate is None:
            return {}
        source = block.input_payloads["image"]
        if (
            source_size != coordinate["source_size"]
            or source.get("spatial") != coordinate["source_spatial"]
        ):
            raise ValidationError("Source coordinate frame differs from the coordinate maps")
        payload = copy.deepcopy(source)
        payload["spatial"] = copy.deepcopy(coordinate["output_spatial"])
        payload.pop("pyramid", None)
        payload.pop(COORDINATE_MAP_TAG, None)
        return {"image": payload}
    if info["source_size"] != source_size:
        raise ValidationError("Source image dimensions do not match the lens map calibration")
    payload = dict(block.input_payloads["image"])
    spatial = payload["spatial"]
    source_frame = spatial if spatial["root_id"] is not None else [block.id, spatial]
    payload["spatial"] = _corrected_frame(source_frame, info["id"], output_size)
    payload.pop("pyramid", None)
    payload.pop(_TAG, None)
    return {"image": payload}


class PackRemapMap(Block):
    type_id, title = "cv.pack_remap_map", "Pack remap coordinates"
    category, block_type, subtype = "Image processing", "Image processing", "Geometry"
    inputs = (Port("map_x"), Port("map_y"))
    outputs = (Port("packed_map", "remap_map", visual_type="data"),)
    payload_reference = "map_x"
    process_inputs_readonly = True
    defaults = {"sampling": "subpixel"}
    parameter_choices = {"sampling": ("subpixel", "nearest")}
    parameter_help = {
        "sampling": "Subpixel retains fractional indices. Nearest stores rounded coordinates only."
    }

    def process(self, inputs):
        x, y = split_remap_fields(inputs)
        size = x.shape[::-1]
        if max(size) > _REMAP_MAX_AXIS:
            raise ValidationError(f"Remap axes are limited to {_REMAP_MAX_AXIS:,} pixels")
        nearest = self.parameters["sampling"] == "nearest"
        maximum = 32767 if nearest else 32768 - 1 / cv2.INTER_TAB_SIZE
        for plane in (x, y):
            low, high, _, _ = cv2.minMaxLoc(plane)
            if low < -32768 or high > maximum:
                raise ValidationError("Remap coordinates exceed the fixed-point range")
        info = lens_map_pair(self, list(size))
        coordinate = coordinate_map_info(self, list(size))
        coordinates, fractions = cv2.convertMaps(x, y, cv2.CV_16SC2, nninterpolation=nearest)
        packed = PackedRemapMap(
            size, coordinates.tobytes(), None if fractions is None else fractions.tobytes()
        )
        payload = dict(self.input_payloads["map_x"])
        if info is not None:
            payload[_TAG] = {**info, "axis": "packed"}
        if coordinate is not None:
            payload[COORDINATE_MAP_TAG] = {**coordinate, "axis": "packed"}
        return Result(
            {"packed_map": packed},
            payloads={"packed_map": payload},
            message=f"Packed {size[0]} × {size[1]} {self.parameters['sampling']} map",
        )


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
        identity = _lens_identity(matrix, distortion, source, target, size, rotation)
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


class UndistortLensPoints(Block):
    type_id, title = "calibration.undistort_lens_points", "Undistort points onto lens map"
    category, block_type, subtype = "Calibration", "Calibration", "Lens distortion"
    inputs = (
        Port("points", "data"),
        Port("calibration", "data"),
        Port("rectified_calibration", "data"),
    )
    outputs = (
        Port("points", "data", required=False),
        Port("inside", "data", required=False),
        Port("roundtrip_max_error", "number"),
    )
    defaults = {"max_iterations": 30, "epsilon": 1e-8, "max_roundtrip_error": 0.05}
    parameter_ranges = {
        "max_iterations": NumericRange(1, 100, whole=True),
        "epsilon": NumericRange(1e-12, 1),
        "max_roundtrip_error": NumericRange(1e-9),
    }
    parameter_help = {
        "epsilon": "OpenCV convergence tolerance for inverse distortion.",
        "max_roundtrip_error": "Maximum reprojection error back on the raw grid, in pixels.",
    }
    input_help = {
        "points": "Ordered [x, y] measurements in the original calibrated image grid.",
        "calibration": "Original pinhole profile used to build the maps.",
        "rectified_calibration": "Rectified calibration from Build or Load lens correction maps.",
    }
    process_inputs_readonly = True
    payload_reference = "points"

    def process(self, inputs):
        profile = inputs["calibration"]
        corrected = inputs["rectified_calibration"]
        matrix, distortion = camera_parameters(profile)
        target, zero = camera_parameters(corrected)
        source, size = profile.get("image_size"), corrected.get("image_size")
        if not valid_lens_size(source) or not valid_lens_size(size) or np.any(zero):
            raise ValidationError("Use valid source and rectified lens profiles")
        rotation = finite_matrix(corrected.get("rectification_rotation"), (3, 3), "Rotation")
        identity = _lens_identity(matrix, distortion, source, target, size, rotation)
        if self.input_payloads["rectified_calibration"].get(_CORRECTION_ID) != identity:
            raise ValidationError("The rectified profile must come from maps for this calibration")

        raw = finite_points(inputs["points"], 2)
        if len(raw) > 100_000 or (raw < 0).any() or (raw >= source).any():
            raise ValidationError("Use up to 100,000 points inside the original image")
        payload = self.input_payloads["points"].copy()
        spatial = payload.pop("spatial", None)
        pyramid = payload.pop("pyramid", None)
        if pyramid is not None:
            raise ValidationError("Map point coordinates to the original image first")
        if spatial is not None and spatial.get("root_id") is not None:
            check_spatial({"spatial": spatial}, source)
            if spatial["root_size"] != source or not np.allclose(spatial["to_root"], np.eye(3)):
                raise ValidationError("Points must be in the original calibrated pixel grid")
            payload["spatial"] = _corrected_frame(spatial, identity, size)

        p = self.parameters
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            int(p["max_iterations"]),
            float(p["epsilon"]),
        )
        normalized = cv2.undistortPoints(
            raw[:, None, :], matrix, distortion, criteria=criteria
        ).reshape(-1, 2)
        rays = np.column_stack((normalized, np.ones(len(raw))))
        if not np.isfinite(rays).all():
            raise ValidationError("Lens correction produced nonfinite points")
        origin = np.zeros(3)
        reprojected = cv2.projectPoints(rays, origin, origin, matrix, distortion)[0].reshape(-1, 2)
        error = float(np.linalg.norm(reprojected - raw, axis=1).max())
        if not math.isfinite(error):
            raise ValidationError("Lens correction produced nonfinite reprojection error")
        projected = rays @ rotation.T @ target.T
        in_front = projected[:, 2] > 1e-12
        if not in_front.all() or error > float(p["max_roundtrip_error"]):
            reason = (
                "lens_points_outside_rectification"
                if not in_front.all()
                else ("lens_points_roundtrip_above_limit")
            )
            return Result(
                {"points": NO_VALUE, "inside": NO_VALUE, "roundtrip_max_error": error},
                passed=False,
                fail_reasons=[reason],
                message="Some points do not map reliably onto the corrected grid",
            )

        xy = projected[:, :2] / projected[:, 2, None]
        if not np.isfinite(xy).all():
            raise ValidationError("Lens correction produced nonfinite output coordinates")
        inside = ((xy >= 0) & (xy < size)).all(axis=1)
        drawings = {}
        if self.generate_graphics and "spatial" in payload and inside.any():
            objects = tuple(
                graphic_object("marker", [point], (70, 220, 140)) for point in xy[inside][:256]
            )
            drawings["Corrected points"] = GraphicDrawing(objects, "points", from_output=True)
        return Result(
            {"points": xy.tolist(), "inside": inside.tolist(), "roundtrip_max_error": error},
            payloads={"points": payload, "inside": payload},
            drawings=drawings,
            message=f"Corrected {len(raw)} points · max round-trip error {error:.4g} px",
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
    parameter_choices = {"colormap": tuple(COLORMAPS)}
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
        heatmap = cv2.applyColorMap(levels, COLORMAPS[self.parameters["colormap"]])
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
