"""Camera acquisition and recorded-source qualification with shared negotiation reports."""

import copy
import uuid

import cv2

from .capture_negotiation import (
    CHOICES,
    DEFAULTS,
    HELP,
    RANGES,
    CaptureNegotiation,
    diagnostic,
    validate_settings,
)
from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .video_files import VideoFile

CAMERA_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "ffmpeg": cv2.CAP_FFMPEG,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}
OUTPUTS = (Port("image"), Port("info", "data"), Port("diagnostic", required=False))


def attach_report(block, result, report, source_kind):
    """Keep negotiated backend values distinct from the final formatted output."""
    report = copy.deepcopy(report)
    report["source_kind"] = source_kind
    passed = report["policy_satisfied"]
    image = result.outputs["image"]
    if image is not NO_VALUE:
        report["output_frame"] = {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": 1 if image.ndim == 2 else image.shape[2],
            "dtype": str(image.dtype),
            "color_format": image_format(image),
        }
        result.payloads.setdefault("image", {})["capture"] = {
            "source_kind": source_kind,
            "generation": report["generation"],
            "frame_index": report["frame_index"],
            "status": report["status"],
            "backend_name": report["backend_name"],
        }
    if not passed:
        result.outputs["image"] = NO_VALUE
        result.payloads.pop("image", None)
    result.outputs["info"] = {**result.outputs.get("info", {}), "capture_negotiation": report}
    result.outputs["diagnostic"] = NO_VALUE
    if block.generate_graphics:
        chart = diagnostic(report)
        result.outputs["diagnostic"] = ImageArray(chart, "bgr")
        result.payloads["diagnostic"] = {
            "spatial": frame(list(chart.shape[1::-1]), f"{block.id}:capture_report")
        }
    result.passed = passed
    result.fail_reasons = (
        []
        if passed
        else [
            {
                "code": "capture.negotiation_rejected",
                "message": "Capture settings failed the selected negotiation policy: "
                + ", ".join(
                    r["name"] + " (" + r["state"] + ")"
                    for r in report["properties"]
                    if r["state"] != "reported_match"
                ),
            }
        ]
    )
    result.message = "Capture settings: " + report["status"]
    signature = (report["generation"], tuple((r["name"], r["state"]) for r in report["properties"]))
    if report["status"] in ("warning", "rejected") and signature != block._last_warning:
        block.logger.warning("Capture negotiation %s: %s", report["status"], signature[1])
        block._last_warning = signature
    return result


class Camera(Block):
    block_type, subtype = "Input", "Cameras"
    type_id, title, category = "input.camera", "Camera", "Input"
    preview_safe = parallel_safe = False
    outputs = OUTPUTS
    defaults = {"device": 0, "backend": "auto", **DEFAULTS}
    parameter_choices = {"backend": tuple(CAMERA_BACKENDS), **CHOICES}
    parameter_ranges = {"device": NumericRange(0, 65535, whole=True), **RANGES}
    parameter_help = {
        **HELP,
        "device": "Local capture-device index. Opening happens only on execution.",
        "backend": "Native capture backend; Auto lets OpenCV choose. Build-specific availability.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capture = self._settings = self._negotiation = None
        self._last_warning = None
        self._frame_index = 0
        self.exposed_outputs.remove("diagnostic")

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_settings(p)
        if type(p["device"]) is not int:
            raise ValidationError("Device must be an integer index", "device")

    def process(self, inputs):
        p = self.parameters
        try:
            if self._capture is None or self._settings != p:
                self.close()
                self._capture = (
                    cv2.VideoCapture(p["device"])
                    if p["backend"] == "auto"
                    else cv2.VideoCapture(p["device"], CAMERA_BACKENDS[p["backend"]])
                )
                if not self._capture.isOpened():
                    raise RuntimeError(f"Cannot open camera {p['device']}")
                self._settings = copy.deepcopy(p)
                self._generation = uuid.uuid4().hex
                self._frame_index = 0
                self._negotiation = CaptureNegotiation(self._capture, p, self.logger)
                self._negotiation.apply()
            session = self._negotiation
            rejected = any(not item["returned"] for item in session.setter_results.values())
            if rejected and p["negotiation_policy"] != "report":
                report = session.report(frame_index=self._frame_index)
                report["generation"] = self._generation
                result = attach_report(self, Result({"image": NO_VALUE}), report, "camera")
                self.close()
                return result
            ok, raw = self._capture.read()
            if not ok:
                raise RuntimeError("Camera did not return a frame")
            report = session.report(raw, self._frame_index)
            report["generation"] = self._generation
            image = ImageArray(raw, report["decoded_frame"]["color_format"])
            self._frame_index += 1
            size = list(image.shape[1::-1])
            root = f"{self.id}:capture:{self._generation}:{size[0]}x{size[1]}"
            result = attach_report(
                self,
                Result({"image": image}, payloads={"image": {"spatial": frame(size, root)}}),
                report,
                "camera",
            )
            if not result.passed:
                self.close()
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        capture, self._capture = self._capture, None
        self._settings = self._negotiation = None
        if capture is not None:
            capture.release()

    def reset_runtime(self):
        self.close()


class RecordedCapture(VideoFile):
    """Exercise the same negotiation logic against a file backend, without camera IO."""

    type_id, title = "input.capture_file", "Recorded capture with settings"
    preview_safe = parallel_safe = False
    outputs = OUTPUTS
    defaults = {**VideoFile.defaults, **DEFAULTS}
    parameter_choices = {**VideoFile.parameter_choices, **CHOICES}
    parameter_ranges = {**VideoFile.parameter_ranges, **RANGES}
    parameter_help = {
        **VideoFile.parameter_help,
        **HELP,
        "path": "Recorded video for offline property negotiation; does not qualify a camera.",
        "color_format": "Host conversion after decoding; independent of backend FourCC/format.",
        "pixel_depth": "Host conversion after decoding; raw decoded type is reported separately.",
    }

    def __init__(self, **kwargs):
        self._negotiation = self._frame_report = None
        self._last_warning = None
        super().__init__(**kwargs)
        self.exposed_outputs.remove("diagnostic")

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        validate_settings(self.parameters if parameters is None else parameters)

    def configure_capture(self):
        self._generation = uuid.uuid4().hex
        self._frame_report = None
        self._negotiation = CaptureNegotiation(self._capture, self.parameters, self.logger)
        self._negotiation.apply()

    def observe_capture_frame(self, image, index):
        self._frame_report = self._negotiation.report(image, index)
        self._frame_report["generation"] = self._generation

    def process(self, inputs):
        result = super().process(inputs)
        if result.outputs["image"] is NO_VALUE:
            result.outputs["diagnostic"] = NO_VALUE
            return result
        return attach_report(self, result, self._frame_report, "recorded_file")

    def close(self):
        self._negotiation = None
        super().close()
