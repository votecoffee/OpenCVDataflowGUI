"""Resample a planar calibration into a uniform physical pixel grid."""

import cv2
import numpy as np

from .calibration import validate_profile
from .core import Port
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata, moved_payload
from .processing import ImageTool, geometry_result, image_value
from .selection import selection_mask

LENGTH_UNITS = ("mm", "cm", "m", "in")
SCALE_MODEL = "rectified_workplane_v1"
_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_STEP_RANGE = NumericRange(0.000001, 1000)


def read_length_scale(scale):
    if not isinstance(scale, dict) or scale.get("units") not in LENGTH_UNITS:
        raise ValidationError("Workplane scale requires mm, cm, m or in")
    step = _STEP_RANGE.check(scale.get("units_per_pixel"), "Physical units per pixel")
    return scale["units"], step


class RectifyWorkplane(ImageTool):
    type_id, title = "calibration.rectify_workplane", "Rectify workplane"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (
        Port("image"), Port("mapping", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"), Port("valid_mask", visual_type="region"),
        Port("geometry", "data"), Port("scale", "data"),
    )
    defaults = {
        "origin": [0.0, 0.0], "width": 256, "height": 256, "units_per_pixel": 0.25,
        "maximum_rms_error": 1.0, "registration": "same_frame", "interpolation": "linear",
    }
    parameter_choices = {
        "registration": ("same_frame", "fixed_grid"),
        "interpolation": tuple(_INTERPOLATION),
    }
    parameter_ranges = {
        "width": NumericRange(1, 8192, whole=True),
        "height": NumericRange(1, 8192, whole=True),
        "units_per_pixel": _STEP_RANGE,
        "maximum_rms_error": NumericRange(0),
    }
    parameter_help = {
        "origin": "Physical [x,y] at output pixel center (0,0).",
        "units_per_pixel": "Spacing in the calibration's physical units, equal on both axes.",
        "maximum_rms_error": "Maximum allowed planar calibration error in its physical units.",
        "registration": "Fixed grid uses a saved calibration on the same camera pixel grid. "
        "Same frame requires the calibration's frame metadata to match.",
    }
    input_help = {
        "mapping": "Planar calibration mapping from Calibration grid or Load calibration profile.",
        "mask": "Only sample valid pixels selected by this source-grid mask.",
        "region": "Only sample valid pixels inside this source-grid region.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        origin = np.asarray(p["origin"], np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValidationError("Origin must be a finite physical [x,y] point", "origin")
        if int(p["width"]) * int(p["height"]) > 8_388_608:
            raise ValidationError("Output is limited to 8,388,608 pixels", "width", "height")

    def process(self, inputs):
        source, mapping = inputs["image"], inputs["mapping"]
        image = image_value(source, allow_labels=True)
        if not isinstance(mapping, dict) or mapping.get("model") != "planar_homography_v1":
            raise ValidationError("Use a calibration-grid planar mapping")
        validate_profile(mapping)
        p = self.parameters
        units, step = read_length_scale({
            "units": mapping["units"], "units_per_pixel": p["units_per_pixel"],
        })
        height, width = image.shape[:2]
        if not 0 < image.size <= 67_108_864 or max(height, width) >= 32767:
            raise ValidationError("Source exceeds the workplane image limits")
        if mapping["image_size"] != [width, height]:
            raise ValidationError("Calibration size must match the source camera grid")
        if float(mapping["rms_error"]) > float(p["maximum_rms_error"]):
            raise ValidationError("Planar calibration exceeds the maximum RMS error")
        payload = metadata(source)
        if (
            p["registration"] == "same_frame"
            and self.input_payloads["mapping"].get("spatial") != payload["spatial"]
        ):
            raise ValidationError("Calibration and source must share the same coordinate frame")
        mode = _INTERPOLATION[p["interpolation"]]
        if image.dtype == np.int32 and mode != cv2.INTER_NEAREST:
            raise ValidationError("Use nearest interpolation with signed labels", "interpolation")

        x, y = np.asarray(p["origin"], np.float64)
        plane_from_output = np.array([[step, 0, x], [0, step, y], [0, 0, 1]])
        transform = np.linalg.solve(np.asarray(mapping["pixel_to_physical"]), plane_from_output)
        size = (int(p["width"]), int(p["height"]))
        last_x, last_y = size[0] - 1, size[1] - 1
        corners = np.array([[0, 0, 1], [last_x, 0, 1], [0, last_y, 1], [last_x, last_y, 1]])
        denominators = corners @ transform[2]
        if not np.isfinite(transform).all() or not (
            np.all(denominators > 1e-12) or np.all(denominators < -1e-12)
        ):
            raise ValidationError("Workplane bounds cross the calibration horizon")
        transform /= transform[2, 2]
        flags = mode | cv2.WARP_INVERSE_MAP
        selection = selection_mask(inputs, source)
        if selection is None:
            selection = np.full(image.shape[:2], 255, np.uint8)
        valid = cv2.warpPerspective(selection.astype(np.float32), transform, size, flags=flags)
        valid = valid == 255
        warped = cv2.warpPerspective(image, transform, size, flags=flags)
        warped[~valid] = 0

        result = geometry_result(warped, source, transform)
        result.outputs.update(
            valid_mask=valid.astype(np.uint8) * 255,
            scale={
                "model": SCALE_MODEL, "spatial": moved_payload(payload, transform, size)["spatial"],
                "units": units, "units_per_pixel": step, "origin": [float(x), float(y)],
            },
        )
        result.passed = bool(valid.any())
        result.fail_reasons = [] if result.passed else ["workplane_empty_coverage"]
        result.message = (
            f"{size[0]} × {size[1]} workplane · {step:g} {units}/px · {valid.mean():.1%} valid"
        )
        return result
