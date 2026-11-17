"""Shared capture-property requests and honest backend/frame observations; no device opens."""

import copy
import math
from datetime import datetime, timezone
from typing import Protocol

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError


class CaptureHandle(Protocol):
    def set(self, property_id: int, value: float) -> bool: ...
    def get(self, property_id: int) -> float: ...
    def getBackendName(self) -> str: ...


PROPERTIES = {
    "fourcc": cv2.CAP_PROP_FOURCC,
    "width": cv2.CAP_PROP_FRAME_WIDTH,
    "height": cv2.CAP_PROP_FRAME_HEIGHT,
    "fps": cv2.CAP_PROP_FPS,
    "buffer_size": cv2.CAP_PROP_BUFFERSIZE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "auto_focus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
    "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gamma": cv2.CAP_PROP_GAMMA,
    "zoom": cv2.CAP_PROP_ZOOM,
    "backlight": cv2.CAP_PROP_BACKLIGHT,
    "pan": cv2.CAP_PROP_PAN,
    "tilt": cv2.CAP_PROP_TILT,
    "roll": cv2.CAP_PROP_ROLL,
    "iris": cv2.CAP_PROP_IRIS,
}
EXTRA_CONTROLS = tuple(PROPERTIES)[12:]
READ_ONLY = {
    "backend_id": cv2.CAP_PROP_BACKEND,
    "format_code": cv2.CAP_PROP_FORMAT,
    "codec_pixel_format": cv2.CAP_PROP_CODEC_PIXEL_FORMAT,
}
DEFAULTS = {
    "width": 0,
    "height": 0,
    "fps": 0.0,
    "buffer_size": 0,
    "fourcc": "",
    "auto_exposure_mode": "unchanged",
    "auto_exposure_value": 0.75,
    "exposure_mode": "unchanged",
    "exposure_value": -6.0,
    "gain_mode": "unchanged",
    "gain_value": 0.0,
    "focus_mode": "unchanged",
    "focus_value": 0.0,
    "white_balance_mode": "unchanged",
    "white_balance_temperature": 4500.0,
    "additional_controls": {},
    "negotiation_policy": "require_setter",
    "readback": "on_open",
    "absolute_tolerance": 0.01,
    "relative_tolerance": 0.005,
    "maximum_frame_pixels": 16_777_216,
    "maximum_frame_bytes": 128 * 1024**2,
}
CHOICES = {
    **{
        name: ("unchanged", "backend_value")
        for name in ("auto_exposure_mode", "exposure_mode", "gain_mode")
    },
    "focus_mode": ("unchanged", "auto", "manual"),
    "white_balance_mode": ("unchanged", "auto", "manual"),
    "negotiation_policy": ("require_setter", "require_readback", "report"),
    "readback": ("on_open", "every_frame"),
}
RANGES = {
    **{key: NumericRange(0, 16384, whole=True) for key in ("width", "height", "buffer_size")},
    "fps": NumericRange(0, 100_000),
    **{
        key: NumericRange(-1_000_000, 1_000_000)
        for key in (
            "auto_exposure_value",
            "exposure_value",
            "gain_value",
            "focus_value",
            "white_balance_temperature",
        )
    },
    "absolute_tolerance": NumericRange(0, 1_000_000),
    "relative_tolerance": NumericRange(0, 1),
    "maximum_frame_pixels": NumericRange(1, 67_108_864, whole=True),
    "maximum_frame_bytes": NumericRange(1, 1024**3, whole=True),
}
HELP = {
    "width": "Requested decoded width; zero leaves the backend unchanged.",
    "height": "Requested decoded height; zero leaves the backend unchanged.",
    "fps": "Requested backend frame rate; zero leaves it unchanged. Not measured cadence.",
    "buffer_size": "Requested backend buffer size; zero leaves it unchanged.",
    "fourcc": "Optional four-character transport/codec request, including meaningful spaces.",
    "auto_exposure_mode": "Apply a raw backend auto-exposure value, or leave its state unchanged.",
    "auto_exposure_value": "Backend-specific auto-exposure encoding; no portable 0/1 mapping.",
    "exposure_mode": "Apply Exposure value in native backend units, or leave it unchanged.",
    "exposure_value": "Backend units can be negative, steps or time; not universally milliseconds.",
    "gain_mode": "Apply Gain value in native backend units, or leave it unchanged.",
    "gain_value": "Native gain request; zero is a real request when its mode is enabled.",
    "focus_mode": "Auto writes autofocus=1; Manual writes autofocus=0 then Focus value.",
    "focus_value": "Manual focus in backend units; zero is permitted.",
    "white_balance_mode": "Auto writes auto-WB=1; Manual disables it then requests temperature.",
    "white_balance_temperature": "Manual white balance in backend units, usually kelvin.",
    "additional_controls": "Optional named raw controls: " + ", ".join(EXTRA_CONTROLS) + ".",
    "negotiation_policy": (
        "Require successful setters, matching readback/frame geometry, or report only."
    ),
    "readback": "Poll after opening/first frame, or every frame. Geometry is always observed.",
    "absolute_tolerance": "Absolute tolerance for scalar property readback; size/FourCC are exact.",
    "relative_tolerance": "Relative tolerance for scalar readback; not a hardware accuracy claim.",
    "maximum_frame_pixels": "Decoded frame guard; native read may already allocate its frame.",
    "maximum_frame_bytes": "Decoded array byte guard; excludes driver/native/graph copies.",
}


def validate_settings(p):
    for name, limits in RANGES.items():
        if type(p[name]) not in (int, float) or (limits.whole and type(p[name]) is not int):
            raise ValidationError("Use typed numeric capture settings", name)
    value = p["fourcc"]
    if not isinstance(value, str) or (
        value and (len(value) != 4 or any(not 32 <= ord(c) <= 126 for c in value))
    ):
        raise ValidationError(
            "FourCC must be empty or exactly four printable ASCII characters", "fourcc"
        )
    controls = p["additional_controls"]
    if not isinstance(controls, dict) or controls.keys() - set(EXTRA_CONTROLS):
        raise ValidationError(
            "Use supported named additional capture controls", "additional_controls"
        )
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1_000_000
        for v in controls.values()
    ):
        raise ValidationError(
            "Additional controls need finite numeric backend values within +/-1000000",
            "additional_controls",
        )


def requests(p):
    values = {}
    if p["fourcc"]:
        values["fourcc"] = cv2.VideoWriter_fourcc(*p["fourcc"])
    values.update((name, p[name]) for name in ("width", "height", "fps", "buffer_size") if p[name])
    for name in ("auto_exposure", "exposure", "gain"):
        if p[name + "_mode"] == "backend_value":
            values[name] = p[name + "_value"]
    if p["focus_mode"] != "unchanged":
        values["auto_focus"] = int(p["focus_mode"] == "auto")
        if p["focus_mode"] == "manual":
            values["focus"] = p["focus_value"]
    if p["white_balance_mode"] != "unchanged":
        values["auto_white_balance"] = int(p["white_balance_mode"] == "auto")
        if p["white_balance_mode"] == "manual":
            values["white_balance_temperature"] = p["white_balance_temperature"]
    values.update(p["additional_controls"])
    return values


def fourcc_text(value):
    if (
        value is None
        or not math.isfinite(value)
        or not 0 <= value <= 2**32 - 1
        or value != int(value)
    ):
        return None
    text = "".join(chr((int(value) >> (8 * i)) & 255) for i in range(4))
    return text if all(32 <= ord(c) <= 126 for c in text) else None


def frame_descriptor(image, p):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim not in (2, 3)
        or min(image.shape[:2]) < 1
        or max(image.shape[:2]) > 16384
        or image.dtype not in (np.uint8, np.uint16, np.float32, np.float64)
        or image.shape[2:] not in ((), (3,), (4,))
    ):
        raise ValidationError("Capture must decode a supported gray/BGR/BGRA intensity array")
    h, w = image.shape[:2]
    if h * w > p["maximum_frame_pixels"] or image.nbytes > p["maximum_frame_bytes"]:
        raise ValidationError(
            "Decoded capture exceeds frame pixel/byte limits",
            "maximum_frame_pixels",
            "maximum_frame_bytes",
        )
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValidationError("Decoded capture contains nonfinite samples")
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "bytes": image.nbytes,
        "color_format": {1: "gray", 3: "bgr", 4: "bgra"}[channels],
    }


class CaptureNegotiation:
    def __init__(self, capture: CaptureHandle, p, logger=None):
        self.capture, self.parameters, self.logger = capture, copy.deepcopy(p), logger
        self.requested = requests(p)
        self.setter_results, self.after_set, self.after_frame = {}, None, None
        self.readback_index, self.readback_timestamp = None, None

    def snapshot(self):
        values = {}
        for name, code in {**PROPERTIES, **READ_ONLY}.items():
            try:
                value = float(self.capture.get(code))
                values[name] = {
                    "value": value if math.isfinite(value) else None,
                    "state": "finite" if math.isfinite(value) else "nonfinite",
                    "possible_unsupported_sentinel": value in (-1, 0),
                }
            except Exception as exc:
                values[name] = {
                    "value": None,
                    "state": "exception",
                    "error": str(exc)[:512],
                    "possible_unsupported_sentinel": False,
                }
        return values

    def apply(self):
        for name, value in self.requested.items():
            try:
                self.setter_results[name] = {
                    "returned": bool(self.capture.set(PROPERTIES[name], value))
                }
            except Exception as exc:
                self.setter_results[name] = {"returned": False, "exception": str(exc)[:512]}
                if self.logger:
                    self.logger.warning("Capture property %s setter raised", name, exc_info=True)
        self.after_set = self.snapshot()

    def report(self, image=None, frame_index=0):
        p = self.parameters
        decoded = frame_descriptor(image, p) if image is not None else None
        fresh = image is not None and (self.after_frame is None or p["readback"] == "every_frame")
        if fresh:
            self.after_frame = self.snapshot()
            self.readback_index = frame_index
            self.readback_timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        observed = self.after_frame or self.after_set
        rows = []
        for name, expected in self.requested.items():
            getter = observed[name]
            actual = getter["value"]
            exact = name in (
                "width",
                "height",
                "fourcc",
                "buffer_size",
                "auto_focus",
                "auto_white_balance",
            )
            matches = actual is not None and (
                actual == expected
                if exact
                else math.isclose(
                    actual,
                    expected,
                    abs_tol=p["absolute_tolerance"],
                    rel_tol=p["relative_tolerance"],
                )
            )
            frame_matches = (
                decoded[name] == expected
                if decoded is not None and name in ("width", "height")
                else None
            )
            returned = self.setter_results[name]["returned"]
            state = (
                "setter_rejected"
                if not returned
                else "readback_unavailable"
                if actual is None
                else "frame_mismatch"
                if frame_matches is False
                else "reported_match"
                if matches
                else "readback_mismatch"
            )
            rows.append(
                {
                    "name": name,
                    "property_id": PROPERTIES[name],
                    "requested": expected,
                    "setter": self.setter_results[name],
                    "after_set": self.after_set[name],
                    "after_frame": self.after_frame[name] if self.after_frame else None,
                    "reported_match": bool(matches),
                    "frame_match": frame_matches,
                    "state": state,
                    "possible_unsupported_sentinel": getter["possible_unsupported_sentinel"],
                }
            )
        setter_ok = all(row["setter"]["returned"] for row in rows)
        matched = all(row["state"] == "reported_match" for row in rows)
        passed = p["negotiation_policy"] == "report" or (
            setter_ok and (p["negotiation_policy"] != "require_readback" or matched)
        )
        try:
            backend = str(self.capture.getBackendName())
        except Exception:
            backend = None
        return {
            "model": "capture_negotiation_v1",
            "policy": p["negotiation_policy"],
            "policy_satisfied": passed,
            "status": "rejected"
            if not passed
            else "unchanged"
            if not rows
            else "matched"
            if matched
            else "warning",
            "requested": copy.deepcopy(self.requested),
            "properties": copy.deepcopy(rows),
            "readback": copy.deepcopy(observed),
            "readback_policy": p["readback"],
            "readback_fresh": fresh,
            "readback_frame_index": self.readback_index,
            "readback_timestamp": self.readback_timestamp,
            "frame_index": frame_index,
            "backend_name": backend,
            "decoded_frame": decoded,
            "fourcc_text": fourcc_text(observed["fourcc"]["value"]),
            "codec_pixel_format_text": fourcc_text(observed["codec_pixel_format"]["value"]),
            "hardware_support_proven": False,
        }


def diagnostic(report):
    rows = report["properties"]
    image = np.full((max(110, 58 + 22 * len(rows)), 760, 3), 25, np.uint8)
    cv2.putText(
        image,
        "Capture negotiation: " + report["status"],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 220, 140) if report["policy_satisfied"] else (65, 85, 240),
        1,
    )
    for i, row in enumerate(rows):
        observation = row["after_frame"] or row["after_set"]
        text = f"{row['name']}: {row['requested']:g} -> {observation['value']} [{row['state']}]"
        cv2.putText(
            image,
            text[:100],
            (8, 48 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (70, 220, 140) if row["state"] == "reported_match" else (40, 180, 240),
            1,
        )
    if not rows:
        cv2.putText(
            image,
            "No property changes requested; readback is descriptive only.",
            (8, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
    return image
