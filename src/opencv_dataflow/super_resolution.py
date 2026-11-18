"""Pinned single-image neural upsampling with explicit nominal pixel-grid geometry."""

import hashlib
import re
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .dnn_blocks import BACKENDS, TARGETS
from .images import ImageArray, convert_depth, image_format, opencv_image
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .selection import selection_mask

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
SCALES = {"fsrcnn": (2, 3, 4), "espcn": (2, 3, 4), "edsr": (2, 3, 4), "lapsrn": (2, 4, 8)}
CREATE_LOCK = threading.Lock()


class SuperResolution(Block):
    type_id, title = "ml.super_resolution", "Learned image super-resolution"
    category = block_type = "ML"
    subtype = "Image restoration"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("baseline"),
        Port("processed_mask", visual_type="mask"),
        Port("geometry", "data"),
        Port("info", "data"),
    )
    project_asset_parameters = (*Block.project_asset_parameters, "model_path")
    defaults = {
        "model_path": "",
        "model_sha256": "",
        "algorithm": "fsrcnn",
        "scale": 2,
        "depth_mode": "uint8_only",
        "baseline_interpolation": "cubic",
        "backend": "opencv",
        "target": "cpu",
        "maximum_model_bytes": 64 * 1024 * 1024,
        "maximum_input_pixels": 1_000_000,
        "maximum_output_pixels": 16_000_000,
    }
    parameter_choices = {
        "algorithm": tuple(SCALES),
        "scale": (2, 3, 4, 8),
        "depth_mode": ("uint8_only", "convert_full_scale"),
        "baseline_interpolation": tuple(INTERPOLATIONS),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_ranges = {
        "maximum_model_bytes": NumericRange(1, 256 * 1024 * 1024, whole=True),
        "maximum_input_pixels": NumericRange(1, 16_000_000, whole=True),
        "maximum_output_pixels": NumericRange(1, 64_000_000, whole=True),
    }
    parameter_help = {
        "model_path": "Frozen .pb: relative assets copy on Save As; absolute paths stay linked.",
        "model_sha256": "Exact digest checked each run; imports a private snapshot on cache miss.",
        "algorithm": "Must match the export. Qualified here: pinned FSRCNN x2 on CPU.",
        "scale": "Export scale; FSRCNN/ESPCN/EDSR: 2/3/4; LapSRN: 2/4/8. Output size is checked.",
        "depth_mode": "Quantizes uint16/float32 [0,1] to 8 bits, then restores source depth.",
        "baseline_interpolation": "Full-source resize; also supplies unselected pixels and alpha.",
        "backend": "Unavailable targets fail. Layer placement is not measured.",
        "target": "CPU baseline; other available devices require their own qualification.",
        "maximum_model_bytes": "Bound the exact model snapshot; does not bound native workspace.",
        "maximum_input_pixels": "Full-source inference with unselected context; no ROI speedup.",
        "maximum_output_pixels": "Bound predicted output size before resize and native inference.",
    }
    input_help = {
        "image": "GRAY/BGR/RGB/BGRA/RGBA. Interpolated alpha; no physical depth.",
        "mask": "Optional source-grid nonzero selection; intersects Region and repeats to output.",
        "region": "Source-grid region. Full source remains model context; outside uses baseline.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True
    opencv_functions = ("dnn_superres.DnnSuperResImpl_create",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model, self._model_key = None, None

    close = reset_runtime

    def project_assets(self):
        assets = super().project_assets()
        model = self.parameters["model_path"]
        if Path(model).is_absolute():
            assets.remove(model)
        return assets

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["scale"] not in SCALES[p["algorithm"]] or type(p["scale"]) is not int:
            raise ValidationError(
                "Scale must match the selected model family", "algorithm", "scale"
            )
        if not isinstance(p["model_path"], str) or len(p["model_path"]) > 4096:
            raise ValidationError("Use a local model path", "model_path")
        path = Path(p["model_path"])
        if not path.is_absolute() and (".." in path.parts or path.anchor or ":" in p["model_path"]):
            raise ValidationError("Keep relative models inside the project", "model_path")
        if not isinstance(p["model_sha256"], str) or (
            p["model_sha256"] and not re.fullmatch(r"[0-9a-f]{64}", p["model_sha256"])
        ):
            raise ValidationError("Use a lowercase SHA256 digest", "model_sha256")
        if p["model_path"] and not p["model_sha256"]:
            raise ValidationError("Pin the chosen model's SHA256", "model_sha256")
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError(
                "CUDA backend and target must be selected together", "backend", "target"
            )

    def infer(self, pixels):
        try:
            return self._infer(pixels)
        except Exception:
            self.reset_runtime()
            raise

    def _infer(self, pixels):
        p = self.parameters
        if not p["model_sha256"]:
            raise ValidationError("Pin the chosen model's SHA256", "model_sha256")
        path = local_path(self, p["model_path"])
        with path.open("rb") as stream:
            raw = stream.read(int(p["maximum_model_bytes"]) + 1)
        if not raw or len(raw) > p["maximum_model_bytes"]:
            raise ValidationError(
                "Model snapshot is empty or exceeds its byte limit", "maximum_model_bytes"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != p["model_sha256"]:
            raise ValidationError("Super-resolution model digest differs", "model_sha256")
        backend, target = BACKENDS[p["backend"]], TARGETS[p["target"]]
        # CPU is the built-in classic backend. Enumerating all targets can create
        # an unused OpenCL context on a worker thread and break driver teardown.
        if p["target"] != "cpu" and target not in cv2.dnn.getAvailableTargets(backend):
            raise ValidationError("Requested DNN target is unavailable", "backend", "target")
        # The native API accepts a filename, not bytes. Import the hashed snapshot,
        # never reopen the caller's mutable original during native loading.
        key = (digest, p["algorithm"], p["scale"], backend, target)
        cached = self._model is not None and self._model_key == key
        if not cached:
            self.reset_runtime()
            with tempfile.TemporaryDirectory(prefix="dataflow-superres-") as directory:
                snapshot = Path(directory) / "model.pb"
                snapshot.write_bytes(raw)
                # Native construction registers DepthToSpace with a process-wide
                # non-atomic static flag. Serialize registration across instances.
                with CREATE_LOCK:
                    model = cv2.dnn_superres.DnnSuperResImpl_create()
                model.readModel(str(snapshot))
                model.setModel(p["algorithm"], p["scale"])
                model.setPreferableBackend(backend)
                model.setPreferableTarget(target)
                self._model, self._model_key = model, key
        output = self._model.upsample(pixels)
        return output, {"sha256": digest, "bytes": len(raw), "native_model_reused": cached}

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image_value(source, (np.uint8, np.uint16, np.float32))
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Convert the image to gray or RGB/BGR channels before upsampling")
        channels = 1 if source.ndim == 2 else source.shape[2]
        if channels != {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}[encoding]:
            raise ValidationError("Image encoding tag disagrees with its channel layout")
        sh, sw = source.shape[:2]
        scale = p["scale"]
        width, height = sw * scale, sh * scale
        if (
            min(sw, sh) < 1
            or sw * sh > p["maximum_input_pixels"]
            or width * height > p["maximum_output_pixels"]
            or max(width, height) > 16384
        ):
            raise ValidationError(
                "Input/output grid exceeds pixel or 16384-axis limits",
                "maximum_input_pixels",
                "maximum_output_pixels",
            )
        if source.dtype != np.uint8 and p["depth_mode"] == "uint8_only":
            raise ValidationError(
                "Use uint8 or explicitly select full-scale conversion", "depth_mode"
            )
        if source.dtype == np.float32 and (source.min() < 0 or source.max() > 1):
            raise ValidationError("Float source must be in [0,1]; HDR clipping is not implicit")
        selection = selection_mask(inputs, source)
        selected = np.ones((sh, sw), np.uint8) * 255 if selection is None else selection
        processed = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
        baseline = cv2.resize(
            np.asarray(source),
            (width, height),
            interpolation=INTERPOLATIONS[p["baseline_interpolation"]],
        )
        clipped = 0
        if source.dtype == np.float32:
            clipped = int(np.count_nonzero((baseline < 0) | (baseline > 1)))
            np.clip(baseline, 0, 1, out=baseline)
        output, identity = baseline.copy(), None
        calls = int(bool(np.any(processed)))
        if calls:
            canonical = opencv_image(source)
            pixels = convert_depth(canonical, "uint8")
            alpha = pixels.ndim == 3 and pixels.shape[2] == 4
            if alpha:
                pixels = pixels[..., :3]
            gray_as_bgr = pixels.ndim == 2 and p["algorithm"] == "edsr"
            if gray_as_bgr:
                pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            learned, identity = self.infer(np.ascontiguousarray(pixels))
            expected = (height, width) if pixels.ndim == 2 else (height, width, 3)
            if (
                not isinstance(learned, np.ndarray)
                or learned.shape != expected
                or learned.dtype != np.uint8
            ):
                self.reset_runtime()
                raise ValidationError(
                    "Native model output does not match declared scale/channels/depth",
                    "algorithm",
                    "scale",
                )
            if gray_as_bgr:
                learned = cv2.cvtColor(learned, cv2.COLOR_BGR2GRAY)
            if encoding in ("rgb", "rgba"):
                learned = cv2.cvtColor(learned, cv2.COLOR_BGR2RGB)
            learned = convert_depth(learned, source.dtype.name)
            channels = output[..., :3] if alpha else output
            channels[processed != 0] = learned[processed != 0]
        matrix = [
            [1 / scale, 0, (1 / scale - 1) / 2],
            [0, 1 / scale, (1 / scale - 1) / 2],
            [0, 0, 1],
        ]
        geometry = {
            "input_size": [sw, sh],
            "output_size": [width, height],
            "output_to_input": matrix,
            "coordinate_convention": "nominal half-pixel grid; learned alignment uncalibrated",
        }
        info = {
            "model": "single_image_super_resolution_v1",
            "algorithm": p["algorithm"],
            "scale": scale,
            "artifact": identity,
            "native_calls": calls,
            "engine": "classic",
            "requested_backend": p["backend"],
            "requested_target": p["target"],
            "opencv": cv2.__version__,
            "input_dtype": source.dtype.name,
            "working_dtype": "uint8",
            "quantized": source.dtype != np.uint8,
            "color_format": encoding,
            "alpha_inferred": False,
            "baseline_interpolation": p["baseline_interpolation"],
            "baseline_clipped_samples": clipped,
            "selected_source_pixels": int(np.count_nonzero(selected)),
            "processed_output_pixels": int(np.count_nonzero(processed)),
            "selection_semantics": "full context; nearest selection; baseline elsewhere",
            "measurement_validity": "reconstruction, not additional measured detail",
        }
        return Result(
            {
                "image": ImageArray(output, encoding, getattr(source, "metadata", {})),
                "baseline": ImageArray(baseline, encoding, getattr(source, "metadata", {})),
                "processed_mask": ImageArray(processed, "gray"),
                "geometry": geometry,
                "info": info,
            },
            message=f"{p['algorithm'].upper()} x{scale}; {calls} native call; reconstruction",
        )
