"""Replay a native OpenCV moving foreground with its exact reference mask."""

import copy
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .core import Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .replay import ReplaySource

MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 2_097_152
MAX_CLIP_PIXELS = 128_000_000


def load_asset(value, project_dir, field):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Choose a background and a foreground image", field)
    path = Path(value)
    if not path.is_absolute():
        if project_dir is None:
            raise ValidationError("Save or open the project to resolve relative paths", field)
        path = project_dir / path
    with path.open("rb") as stream:
        raw = stream.read(MAX_ASSET_BYTES + 1)
    if len(raw) > MAX_ASSET_BYTES:
        raise ValidationError("Source image exceeds 8 MiB", field)
    try:
        with Image.open(io.BytesIO(raw), formats=("PNG", "JPEG", "BMP")) as header:
            width, height = header.size
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValidationError("Choose a bounded PNG, JPEG or BMP image", field) from exc
    if min(width, height) < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValidationError(f"Source image is limited to {MAX_IMAGE_PIXELS:,} pixels", field)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        image is None
        or image.dtype != np.uint8
        or image.shape[:2] != (height, width)
        or not (image.ndim == 2 or image.shape[2] == 3)
    ):
        raise ValidationError("Use an 8-bit grayscale or BGR image without an alpha channel", field)
    return image, hashlib.sha256(raw).hexdigest()


class SyntheticBackgroundSequence(ReplaySource):
    type_id, title, category = (
        "input.synthetic_background_sequence",
        "Synthetic background sequence",
        "Input",
    )
    block_type, subtype = "Input", "Video"
    project_path_parameters = ("background_path", "object_path")
    outputs = (Port("image"), Port("foreground_mask", visual_type="mask"), Port("info", "data"))
    defaults = {
        "background_path": "",
        "object_path": "",
        "amplitude": 2.0,
        "wavelength": 20.0,
        "wave_speed": 0.2,
        "object_speed": 3.0,
        "frame_count": 24,
        "fps": 30.0,
        "repeat": "stop",
    }
    parameter_choices = {"repeat": ("stop", "loop")}
    parameter_ranges = {
        "amplitude": NumericRange(0, 4096),
        "wavelength": NumericRange(0.01, 1_000_000),
        "wave_speed": NumericRange(-4096, 4096),
        "object_speed": NumericRange(0, 1024),
        "fps": NumericRange(0.01, 1000),
        "frame_count": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "background_path": "8-bit grayscale or BGR PNG, JPEG or BMP. Relative to the project.",
        "object_path": "Opaque rectangular foreground image. Same formats as the background.",
        "amplitude": "Native background distortion amplitude in pixels.",
        "wavelength": "Native background distortion wavelength in pixels.",
        "wave_speed": "Native wave phase increment per frame. Negative reverses the phase advance.",
        "object_speed": "Foreground motion in pixels per frame. Limited by its available clearance.",
        "frame_count": "Frames per clip. Total decoded background pixels per clip cannot exceed 128 million.",
        "fps": "Nominal rate used for timestamps. Frame production is driven by the program.",
        "repeat": "Stop at the end or replay the same native trajectory with new tickets.",
    }
    output_help = {
        "image": "OpenCV-generated uint8 BGR frame.",
        "foreground_mask": "Native uint8 mask with 255 inside the opaque foreground rectangle.",
        "info": "Clip index, sequence index, nominal timestamp and exact foreground bounds.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def close(self):
        self._generator = None
        self._assets = None

    def reset_runtime(self):
        self.close()
        self._sequence = {
            "settings": None, "next": 0, "emitted": 0, "finished": False, "asset_ids": None,
        }

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
            raise ValidationError(
                "Sequence repeat is a run setting and cannot be recipe-bound", "repeat"
            )

    @property
    def exhausted(self):
        return (
            self.finite_source
            and self._sequence["finished"]
            and self._sequence["settings"] == {**self.parameters, **self.last_recipe_parameters}
        )

    def _load_clip(self):
        p = self.parameters
        fields = self.project_path_parameters
        background, background_id = load_asset(p["background_path"], self.project_dir, "background_path")
        target, target_id = load_asset(p["object_path"], self.project_dir, "object_path")
        asset_ids = background_id, target_id
        state = self._sequence
        if state["asset_ids"] is not None and state["asset_ids"] != asset_ids:
            raise ValidationError(
                "Source images changed since this clip started. Rewind to use them.", *fields
            )
        height, width = background.shape[:2]
        clearance = min(height - target.shape[0], width - target.shape[1])
        if clearance < 1:
            raise ValidationError("Foreground must be smaller than the background on both axes", *fields)
        if float(p["object_speed"]) > clearance / 2:
            raise ValidationError(
                f"Object speed must be at most {clearance / 2:g} pixels/frame for these images",
                "object_speed",
            )
        if height * width * p["frame_count"] > MAX_CLIP_PIXELS:
            raise ValidationError(
                f"Use at most {MAX_CLIP_PIXELS // (height * width)} frames for this background size",
                "frame_count",
            )
        state["asset_ids"] = asset_ids
        self._assets = background, target

    def _open_generator(self):
        factory = getattr(getattr(cv2, "bgsegm", None), "createSyntheticSequenceGenerator", None)
        if factory is None:
            raise RuntimeError("The OpenCV contrib bgsegm synthetic sequence generator is unavailable")
        if self._assets is None:
            self._load_clip()
        p = self.parameters
        generator = factory(
            *self._assets,
            float(p["amplitude"]), float(p["wavelength"]),
            float(p["wave_speed"]), float(p["object_speed"]),
        )
        # The native generator owns its RNG and trajectory. Rebuild the cursor only
        # when a snapshot is restored or the native handle was released.
        for _ in range(self._sequence["next"]):
            if self._cancelled():
                return
            generator.getNextFrame()
        self._generator = generator

    def issue_source_ticket(self):
        state = self._sequence
        if state["settings"] != self.parameters:
            self.reset_runtime()
            state = self._sequence
            state["settings"] = copy.deepcopy(self.parameters)
        if state["finished"] or self._cancelled():
            return None
        try:
            if self._generator is None:
                self._open_generator()
            if self._cancelled() or self._generator is None:
                return None
            return super().issue_source_ticket()
        except Exception:
            self.close()
            state["finished"] = True
            raise

    def process(self, inputs):
        if self._source_ticket is None:
            return Result(dict.fromkeys(("image", "foreground_mask", "info"), NO_VALUE))
        state = self._sequence
        p = self.parameters
        try:
            image, mask = self._generator.getNextFrame()
            height, width = self._assets[0].shape[:2]
            if (
                image is None or mask is None
                or image.shape != (height, width, 3) or mask.shape != (height, width)
                or image.dtype != np.uint8 or mask.dtype != np.uint8
            ):
                raise RuntimeError("OpenCV returned an unexpected synthetic frame or mask")
        except Exception:
            self.close()
            state["finished"] = True
            raise

        index = state["next"]
        emitted = state["emitted"]
        fps = float(p["fps"])
        x, y, box_width, box_height = cv2.boundingRect(mask)
        info = {
            "frame_index": index,
            "sequence_index": emitted,
            "loop_index": emitted // int(p["frame_count"]),
            "timestamp_ms": 1000.0 * emitted / fps,
            "timestamp_source": "synthetic_frame_rate",
            "fps": fps,
            "foreground_bounds_xywh": [x, y, box_width, box_height],
            "foreground_pixels": box_width * box_height,
        }
        state["next"] += 1
        state["emitted"] += 1
        if state["next"] == p["frame_count"]:
            self._generator = None
            if self.finite_source:
                state["finished"] = True
                self.close()
            else:
                state["next"] = 0
        context = {"video_frame": info, "spatial": frame([width, height], f"{self.id}:image")}
        return Result(
            {"image": image, "foreground_mask": mask, "info": info},
            payloads={"image": context, "foreground_mask": context},
            message=f"Synthetic frame {index + 1}/{p['frame_count']}",
        )
