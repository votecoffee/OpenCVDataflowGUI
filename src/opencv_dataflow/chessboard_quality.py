"""Measure checkerboard edge quality before accepting calibration captures."""

import cv2
import numpy as np

from .calibration import chessboard_sb_flags
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import gray_image
from .result_graphics import ACCEPTED, REJECTED
from .tickets import ticket_key


def _quality_reasons(width, black, white, parameters):
    reasons = []
    limit = float(parameters["max_transition_pixels"])
    if limit and width > limit:
        reasons.append("chessboard_transition_too_wide")
    if white <= black or white - black < float(parameters["min_contrast"]):
        reasons.append("chessboard_low_contrast")
    if black > float(parameters["max_black"]) or white < float(parameters["min_white"]):
        reasons.append("chessboard_brightness_outside_limits")
    return reasons


class CheckChessboardQuality(Block):
    type_id, title = "calibration.chessboard_quality", "Check chessboard quality"
    category, block_type, subtype = "Calibration", "Calibration", "Grid and coordinates"
    inputs = (
        Port("image"),
        Port("corners", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("corners", "data", required=False),
        Port("report", "data"),
        Port("max_width", "number", required=False),
        Port("min_contrast", "number", required=False),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    defaults = {
        "columns": 9,
        "rows": 6,
        "orientation": "both",
        "rise_distance": 0.8,
        "normalize": True,
        "exhaustive": True,
        "accuracy": False,
        "max_transition_pixels": 3.0,
        "min_contrast": 32.0,
        "max_black": 255.0,
        "min_white": 0.0,
    }
    parameter_choices = {"orientation": ("both", "horizontal", "vertical")}
    parameter_ranges = {
        "columns": NumericRange(3, 64, whole=True),
        "rows": NumericRange(3, 64, whole=True),
        "rise_distance": NumericRange(1e-6, 1),
        "max_transition_pixels": NumericRange(0, 32766),
        "min_contrast": NumericRange(0, 255),
        "max_black": NumericRange(0, 255),
        "min_white": NumericRange(0, 255),
    }
    parameter_help = {
        "columns": "Number of inner corner columns in the chessboard.",
        "rows": "Number of inner corner rows in the chessboard.",
        "orientation": "Chessboard edge directions in detected board coordinates.",
        "rise_distance": "Intensity transition fraction. 0.8 measures the 10% to 90% rise.",
        "normalize": "Normalize the detector image. Brightness is measured before normalization.",
        "exhaustive": "Spend more detector effort searching for the chessboard.",
        "accuracy": "Upsample during SB corner detection for improved corner localization.",
        "max_transition_pixels": "Maximum mean width per direction. Zero skips the limit.",
        "min_contrast": "Minimum mean white-minus-black contrast per direction, in 8-bit units.",
        "max_black": "Maximum mean black-cell intensity per direction, on an 8-bit scale.",
        "min_white": "Minimum mean white-cell intensity per direction, on an 8-bit scale.",
    }
    input_help = {
        "corners": "Optional SB inner corners in row order, from this image's frame and capture."
    }

    def _read_corners(self, inputs, gray, size):
        p = self.parameters
        if "corners" not in inputs:
            found, points = cv2.findChessboardCornersSB(gray, size, flags=chessboard_sb_flags(p))
            return points.reshape(-1, 2) if found else None
        payload = self.input_payloads["corners"]
        require_same_frame(
            (inputs["image"], ValuePacket(inputs["corners"], payload)),
            message="Chessboard corners must use the image coordinate frame",
        )
        spatial = payload.get("spatial")
        image_spatial = self.input_payloads["image"]["spatial"]
        if spatial and (
            spatial["image_size"] != image_spatial["image_size"]
            or not np.allclose(spatial["to_root"], image_spatial["to_root"], atol=1e-9, rtol=0)
        ):
            raise ValidationError("Chessboard corners must use the image coordinate frame")
        a, b = self.input_payloads["image"].get("ticket"), payload.get("ticket")
        if (
            a
            and b
            and (ticket_key(a), a["source"], a["sequence"])
            != (ticket_key(b), b["source"], b["sequence"])
        ):
            raise ValidationError("Chessboard corners must come from the same captured image")
        points = np.asarray(inputs["corners"], dtype=np.float64)
        if (
            points.shape != (size[0] * size[1], 2)
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points[:, 0] > gray.shape[1] - 1)
            or np.any(points[:, 1] > gray.shape[0] - 1)
        ):
            raise ValidationError(
                "Use the complete list of finite, in-image SB corners in row order"
            )
        return points.astype(np.float32)

    @staticmethod
    def _unmeasured(reason):
        return Result(
            {
                "image": NO_VALUE,
                "corners": NO_VALUE,
                "report": {"model": "chessboard_quality_v1", "status": reason},
                "max_width": NO_VALUE,
                "min_contrast": NO_VALUE,
            },
            passed=False,
            fail_reasons=[reason],
            message="Chessboard quality could not be measured",
        )

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        if gray.size > 8_388_608:
            raise ValidationError("Chessboard quality is limited to 8,388,608 input pixels")
        size = (int(p["columns"]), int(p["rows"]))
        points = self._read_corners(inputs, gray, size)
        if points is None:
            return self._unmeasured("chessboard_not_found")
        orientations = (
            ("horizontal", "vertical") if p["orientation"] == "both" else (p["orientation"],)
        )
        directions, reasons, markers = {}, [], []
        widths, contrasts = [], []
        for axis in orientations:
            average, profiles = cv2.estimateChessboardSharpness(
                gray,
                size,
                points,
                rise_distance=float(p["rise_distance"]),
                vertical=axis == "vertical",
            )
            if profiles is not None:
                profiles = profiles[np.any(profiles != 0, axis=1)]
            if (
                profiles is None
                or not len(profiles)
                or not np.isfinite(profiles).all()
                or not np.isfinite(average[:3]).all()
            ):
                reasons.append("chessboard_profiles_unavailable")
                directions[axis] = {"status": "unmeasured", "profiles": []}
                continue
            width, black, white = average[:3]
            widths.append(width)
            contrasts.append(white - black)
            reasons.extend(_quality_reasons(width, black, white, p))
            directions[axis] = {
                "status": "measured",
                "average_width_pixels": width,
                "average_black": black,
                "average_white": white,
                "profiles": profiles.tolist(),
            }
            if self.generate_graphics:
                markers.extend(
                    graphic_object(
                        "marker",
                        ((x, y),),
                        REJECTED if _quality_reasons(w, low, high, p) else ACCEPTED,
                    )
                    for x, y, w, low, high in profiles[:512]
                )
        reasons = list(dict.fromkeys(reasons))
        passed = not reasons
        complete = len(widths) == len(orientations)
        max_width = max(widths) if complete else NO_VALUE
        message = (
            f"Worst mean edge transition {max_width:.3g} px"
            if complete
            else "Some requested chessboard edge profiles are unavailable"
        )
        return Result(
            {
                "image": source if passed else NO_VALUE,
                "corners": points.tolist(),
                "report": {
                    "model": "chessboard_quality_v1",
                    "status": "ok" if passed else ("outside_limits" if complete else "incomplete"),
                    "pattern_size": list(size),
                    "rise_distance": float(p["rise_distance"]),
                    "profile_fields": ["x", "y", "width_pixels", "black", "white"],
                    "directions": directions,
                },
                "max_width": max_width,
                "min_contrast": min(contrasts) if complete else NO_VALUE,
            },
            passed=passed,
            fail_reasons=reasons,
            drawings={"checkerboard edge quality": GraphicDrawing(tuple(markers))}
            if markers
            else {},
            message=message,
        )
