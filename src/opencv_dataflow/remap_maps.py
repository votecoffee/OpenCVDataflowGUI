"""Build reusable destination-to-source maps for pinhole lens correction."""

import hashlib
import json

import cv2
import numpy as np

from .calibration import camera_parameters
from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import finite

_MAX_AXIS = 16_384
_TAG = "lens_remap_map"


def _signature(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _map_frame(info):
    return frame(info["output_size"], f"lens-map:{info['id']}")


def lens_remap_payload(block, source_size, output_size):
    x, y = (block.input_payloads[name] for name in ("map_x", "map_y"))
    if _TAG not in x and _TAG not in y:
        return {}
    info, other = x.get(_TAG), y.get(_TAG)
    if (
        not isinstance(info, dict)
        or info.get("axis") != "x"
        or not isinstance(info.get("id"), str)
        or {**info, "axis": "y"} != other
    ):
        raise ValidationError("Connect matching X and Y lens maps from the same correction")
    if info.get("source_size") != source_size:
        raise ValidationError("Source image dimensions do not match the lens map calibration")
    if (
        info.get("output_size") != output_size
        or x.get("spatial") != _map_frame(info)
        or y.get("spatial") != _map_frame(info)
    ):
        raise ValidationError("Lens maps must retain their original pixel grid")

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
        if (
            not isinstance(source, list)
            or len(source) != 2
            or any(type(v) is not int or not 1 <= v <= _MAX_AXIS for v in source)
        ):
            raise ValidationError("Lens calibration needs valid [width, height] dimensions")
        size = [int(p["width"]) or source[0], int(p["height"]) or source[1]]
        if size[0] * size[1] > 64_000_000:
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
        info = {"id": identity, "source_size": source, "output_size": size}
        spatial = _map_frame(info)
        return Result(
            {"map_x": map_x, "map_y": map_y, "rectified_calibration": rectified, "valid_roi": roi},
            payloads={
                f"map_{axis}": {"spatial": spatial, _TAG: {**info, "axis": axis}}
                for axis in ("x", "y")
            },
            message=f"{size[0]} × {size[1]} lens correction maps",
        )
