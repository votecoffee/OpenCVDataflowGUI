"""Ticketed video file replay and video output using OpenCV Video I/O."""

import copy
import math
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import finite, gray_image, image_value, integer
from .replay import ReplaySource

BACKENDS = {
    "auto": cv2.CAP_ANY,
    "ffmpeg": cv2.CAP_FFMPEG,
    "gstreamer": cv2.CAP_GSTREAMER,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "opencv_mjpeg": cv2.CAP_OPENCV_MJPEG,
}
PATH_BASES = ("working_directory", "project")


def video_path(block):
    value = block.parameters["path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Select a video file path")
    path = Path(value)
    if block.parameters["path_base"] == "project" and not path.is_absolute():
        if block.project_dir is None:
            raise ValueError("Save or open the project before using a project-relative video path")
        path = block.project_dir / path
    return path.parent.resolve() / path.name


class VideoFile(ReplaySource):
    type_id, title, category = "input.video_file", "Video file", "Input"
    block_type, subtype = "Input", "Video"
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "backend": "auto",
        "start_frame": 0,
        "end_frame": -1,
        "stride": 1,
        "repeat": "stop",
        "frame_ticket": True,
        "color_format": "bgr",
        "pixel_depth": "uint8",
    }
    parameter_choices = {
        "path_base": PATH_BASES,
        "backend": tuple(BACKENDS),
        "repeat": ("stop", "loop"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
    }
    parameter_help = {
        "start_frame": "Zero-based first frame.",
        "end_frame": "Exclusive end frame. -1 reads until the decoder reaches end of file.",
        "stride": "Emit every Nth frame, counting from the start frame.",
        "frame_ticket": "Use filename and frame index as the default ticket value.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capture = None
        self.reset_runtime()

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def reset_runtime(self):
        self.close()
        self._sequence = {"settings": None, "next": 0, "selected": None, "finished": False}

    def snapshot_runtime(self):
        return copy.deepcopy(self._sequence)

    def restore_runtime(self, snapshot):
        if snapshot != self._sequence:
            self.close()
            self._sequence = copy.deepcopy(snapshot)

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        if "repeat" in self.recipe_bindings:
            raise ValueError("Video repeat is a run setting and cannot be recipe-bound")

    @property
    def exhausted(self):
        return (
            self.finite_source
            and self._sequence["finished"]
            and self._sequence["settings"] == {**self.parameters, **self.last_recipe_parameters}
        )

    def open_frame(self, path, index):
        backend = BACKENDS[self.parameters["backend"]]
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        self._capture = cv2.VideoCapture(str(path), backend)
        if not self._capture.isOpened():
            raise ValueError(f"OpenCV cannot open video with this backend: {path}")
        if index:
            positioned = self._capture.set(cv2.CAP_PROP_POS_FRAMES, index) and (
                self._capture.get(cv2.CAP_PROP_POS_FRAMES) == index
            )
            if not positioned:
                self.close()
                self._capture = cv2.VideoCapture(str(path), backend)
                for _ in range(index):
                    if self._cancelled() or not self._capture.grab():
                        return False
        return not self._cancelled() and self._capture.grab()

    def issue_source_ticket(self):
        state = self._sequence
        p = self.parameters
        if state["settings"] != p:
            self.reset_runtime()
            state = self._sequence
            state["settings"] = copy.deepcopy(p)
            state["next"] = p["start_frame"]
        state["selected"] = None
        if state["finished"]:
            return None
        try:
            start = integer(p["start_frame"], "Start frame", 0, 2**31 - 1)
            end = integer(p["end_frame"], "End frame", -1, 2**31 - 1)
            integer(p["stride"], "Stride", 1, 100_000)
            if end != -1 and end <= start:
                raise ValueError("End frame must be later than the start frame")
            path = video_path(self)
            state["next"] = int(state["next"])
            if self._capture is None and not self.open_frame(path, state["next"]):
                if self._cancelled():
                    self.close()
                    return None
                raise ValueError(f"Cannot decode frame {state['next']} in {path.name}")
            label = f"{path.name}#{state['next']:06d}"
            ticket = super().issue_source_ticket()
            state["selected"] = label
            if p["frame_ticket"] and self.ticket_settings["mode"] == "auto_number":
                ticket.update(
                    kind="string",
                    value=label,
                    session=f"{ticket['session']}:video:{ticket['sequence']}",
                )
            return ticket
        except Exception:
            self.close()
            state["finished"] = True
            raise

    def process(self, inputs):
        state = self._sequence
        if state["selected"] is None:
            return Result({"image": NO_VALUE, "info": NO_VALUE}, message="No video frame available")
        p = self.parameters
        index = state["next"]
        try:
            ok, raw = self._capture.retrieve()
            if not ok:
                raise ValueError(f"Decoder could not retrieve frame {index}")
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else None
            timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            origin = "decoder"
            if not math.isfinite(timestamp) or timestamp < 0 or (index and timestamp == 0):
                timestamp = 1000.0 * index / fps if fps else None
                origin = "frame_rate_estimate" if timestamp is not None else "unavailable"
            output = format_image(raw, p["color_format"], p["pixel_depth"])
            info = {
                "filename": video_path(self).name,
                "frame_index": index,
                "timestamp_ms": timestamp,
                "timestamp_source": origin,
                "fps": fps,
                "backend": self._capture.getBackendName(),
            }
            next_index = index + int(p["stride"])
            state["next"] = next_index
            if p["end_frame"] != -1 and next_index >= p["end_frame"]:
                self.finish_range()
            else:
                for _ in range(int(p["stride"])):
                    if self._cancelled():
                        self.close()
                        break
                    if not self._capture.grab():
                        self.finish_range()
                        break
        except Exception:
            self.close()
            state["finished"] = True
            raise
        return Result(
            {"image": output, "info": info},
            payloads={
                "image": {
                    "video_frame": info,
                    "spatial": frame([output.shape[1], output.shape[0]], f"{self.id}:image"),
                }
            },
            message=state["selected"],
        )

    def finish_range(self):
        self.close()
        if self.finite_source:
            self._sequence["finished"] = True
        else:
            self._sequence["next"] = int(self.parameters["start_frame"])


class WriteVideo(Block):
    type_id, title, category = "output.video", "Write video file", "Output"
    block_type, subtype = "Output", "Video"
    inputs = (Port("image"),)
    outputs = (Port("info", "data"),)
    defaults = {
        "path": "output.avi",
        "path_base": "working_directory",
        "backend": "auto",
        "codec": "MJPG",
        "custom_fourcc": "",
        "fps": 30.0,
        "color": "bgr",
    }
    parameter_choices = {
        "path_base": PATH_BASES,
        "backend": tuple(BACKENDS),
        "codec": ("MJPG", "mp4v", "XVID", "FFV1", "HFYU", "custom"),
        "color": ("bgr", "gray"),
    }
    parameter_help = {
        "path": "Use a new file. The container is finalized when execution closes.",
        "custom_fourcc": "Four ASCII characters when codec is set to custom.",
        "fps": "Constant playback FPS. The OpenCV MJPEG backend requires a whole number.",
        "color": (
            "Uint8 BGR or grayscale. Alpha is discarded. "
            "Grayscale requires a backend other than OpenCV MJPEG."
        ),
        "backend": (
            "FFmpeg is qualified for color/grayscale and fractional FPS. "
            "Codec availability depends on the build."
        ),
    }
    parameter_ranges = {"fps": NumericRange(0.001)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._writer = None
        self._settings = None
        self._count = 0

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._settings = None

    def reset_runtime(self):
        self.close()

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        path = video_path(self)
        if not path.suffix:
            raise ValueError("Video output needs a container extension, such as .avi")
        codec = p["custom_fourcc"] if p["codec"] == "custom" else p["codec"]
        if (
            not isinstance(codec, str)
            or len(codec) != 4
            or not codec.isascii()
            or not codec.isprintable()
        ):
            raise ValueError("Video FourCC must contain four printable ASCII characters")
        fps = finite(p["fps"], "FPS", 0.001)
        bgr = p["color"] == "bgr"
        if bgr:
            image = image_value(source, (np.uint8,), canonical=True)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image = gray_image(source, (np.uint8,))
        size = (image.shape[1], image.shape[0])
        if min(size) < 2 or any(axis % 2 for axis in size):
            raise ValueError("Video output requires even frame dimensions of at least two pixels")
        settings = (path, BACKENDS[p["backend"]], codec, fps, size, bgr)
        if self._writer is not None and settings != self._settings:
            raise ValueError("Video settings or frame dimensions changed during recording")
        if self._writer is None:
            self.open_writer(*settings)
            self._settings = settings
            self._count = 0
        self._writer.write(np.ascontiguousarray(image))
        self._count += 1
        info = {
            "path": str(path),
            "frames_submitted": self._count,
            "fps": fps,
            "codec": codec,
            "backend": self._writer.getBackendName(),
            "frame_size": list(size),
            "color_format": p["color"],
            "input_color_format": image_format(source),
            "alpha_discarded": source.ndim == 3 and source.shape[2] == 4,
            "submission_status": "submitted_unverified",
        }
        return Result({"info": info}, message=f"Submitted frame {self._count} to {path.name}")

    def open_writer(self, path, backend, codec, fps, size, bgr):
        # Reserve the name exclusively before handing it to the native backend.
        with path.open("xb"):
            pass
        try:
            self._writer = cv2.VideoWriter(
                str(path), backend, cv2.VideoWriter_fourcc(*codec), fps, size, bgr
            )
            if not self._writer.isOpened():
                raise ValueError(
                    f"OpenCV cannot open video codec {codec!r}, container {path.suffix!r}, "
                    f"backend {self.parameters['backend']!r}"
                )
            # This backend silently rounds fractional FPS and its gray stream did
            # not survive a decoder round trip on the supported OpenCV 5 build.
            # Inspect the actual backend as CAP_ANY can also choose CV_MJPEG.
            if self._writer.getBackendName() == "CV_MJPEG":
                if not bgr:
                    raise ValidationError(
                        "OpenCV MJPEG grayscale recording is not reliable; select FFmpeg or BGR"
                    )
                if fps != round(fps):
                    raise ValidationError(
                        "OpenCV MJPEG requires whole-number FPS; select FFmpeg for fractional FPS"
                    )
        except Exception:
            self.close()
            path.unlink(missing_ok=True)
            raise
