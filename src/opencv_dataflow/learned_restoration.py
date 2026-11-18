"""Same-grid ONNX restoration with bounded halo tiles and explicit sample contracts."""

import numpy as np

from .core import Port, Result
from .dnn_blocks import BACKENDS, ENGINES, TARGETS, DNNInference
from .images import ImageArray, image_format, opencv_image
from .ml_common import file_key
from .model_contracts import (
    CONTRACT_CHOICES,
    CONTRACT_DEFAULTS,
    CONTRACT_HELP,
    model_bytes,
    resolve_contract,
)
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .selection import selection_mask


def preprocessing(parameters):
    return {
        "model": "same_grid_restoration_preprocessing_v1",
        "color_format": parameters["color_format"],
        "layout": parameters["input_layout"],
        "source_units": "integer_full_scale_or_float32_unit",
        "mean": list(parameters["mean"]),
        "std": list(parameters["std"]),
    }


def windows(width, height, p):
    if p["processing_mode"] == "whole":
        yield (0, 0, width, height), (0, 0, width, height)
        return
    tw, th, halo = (int(p[key]) for key in ("tile_width", "tile_height", "halo"))
    for y in range(0, height, th):
        for x in range(0, width, tw):
            right, bottom = min(x + tw, width), min(y + th, height)
            yield (
                (x, y, right, bottom),
                (
                    max(0, x - halo),
                    max(0, y - halo),
                    min(width, right + halo),
                    min(height, bottom + halo),
                ),
            )


class RestoreImage(DNNInference):
    type_id, title = "ml.restore_image", "Restore image with ONNX"
    subtype = "Image restoration"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True, visual_type="mask"),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("processed_mask", visual_type="mask"),
        Port("info", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **CONTRACT_DEFAULTS,
        "model_path": "",
        "input_name": "input",
        "output_names": [],
        "color_format": "gray",
        "input_layout": "NCHW",
        "output_layout": "NCHW",
        "mean": [0.0],
        "std": [1.0],
        "output_mode": "restored",
        "blend": 1.0,
        "processing_mode": "whole",
        "tile_width": 256,
        "tile_height": 256,
        "halo": 20,
        "engine": "classic",
        "backend": "opencv",
        "target": "cpu",
        "fusion": True,
        "maximum_source_pixels": 4_000_000,
        "maximum_tiles": 256,
        "maximum_processed_pixels": 16_000_000,
        "max_output_elements": 8_000_000,
    }
    parameter_choices = {
        **CONTRACT_CHOICES,
        "color_format": ("gray", "rgb", "bgr"),
        "input_layout": ("NCHW", "NHWC"),
        "output_layout": ("NCHW", "NHWC"),
        "output_mode": ("restored", "noise_residual"),
        "processing_mode": ("whole", "halo_tiles"),
        "engine": tuple(ENGINES),
        "backend": tuple(BACKENDS),
        "target": tuple(TARGETS),
    }
    parameter_ranges = {
        "blend": NumericRange(0, 1),
        "tile_width": NumericRange(1, 4096, whole=True),
        "tile_height": NumericRange(1, 4096, whole=True),
        "halo": NumericRange(0, 256, whole=True),
        "maximum_source_pixels": NumericRange(1, 16_000_000, whole=True),
        "maximum_tiles": NumericRange(1, 1024, whole=True),
        "maximum_processed_pixels": NumericRange(1, 64_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **CONTRACT_HELP,
        "model_path": "Local same-grid ONNX; Contract path selects a sealed model/settings bundle.",
        "input_name": "Exact native input name; single image, batch one.",
        "output_names": "Empty requires one output; otherwise select one exact terminal head.",
        "color_format": "Match gray/color source channels. Alpha passes unchanged.",
        "mean": "One or C values in normalized source units, subtracted before dividing by Std.",
        "std": "One or C positive divisors. Output restoration reverses this normalization.",
        "output_mode": "Restored reverses normalization; Noise residual subtracts predicted noise.",
        "blend": "Fraction of restored values to write. Zero skips model I/O/inference.",
        "processing_mode": "No resize. Halo tiles trim context and write disjoint cores.",
        "halo": "Context around cores; cover the model's receptive field for parity.",
        "tile_width": "Core width, excluding Halo. Edge cores/context clip to actual image bounds.",
        "tile_height": "Core height. Native models must accept every resulting context shape.",
        "maximum_source_pixels": "Source-grid guard before normalization or native inference.",
        "maximum_tiles": "Bound selected native calls; unselected cores are skipped.",
        "maximum_processed_pixels": "Combined context pixels, including repeated Halo work.",
        "max_output_elements": "Retained elements per call; does not bound native workspace.",
        "engine": "Classic preserves explicit backend/fusion; non-CPU targets require Classic.",
    }
    input_help = {
        "image": "uint8/uint16 or float32 [0,1], GRAY/BGR/RGB/BGRA/RGBA; no physical depth.",
        "mask": "Write selection, intersected with Region. Excluded pixels remain context.",
        "region": "Source-grid region; outside remains bit-identical to the input.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        channels = 1 if p["color_format"] == "gray" else 3
        for key in ("mean", "std"):
            value = p[key]
            if not isinstance(value, list) or len(value) not in (1, channels):
                raise ValidationError("Use one or C normalization values", key)
            if any(
                type(v) not in (float, int)
                or not np.isfinite(v)
                or abs(v) > 1e6
                or (key == "std" and v < 1e-12)
                for v in value
            ):
                raise ValidationError(
                    "Use finite bounded means and positive standard deviations", key
                )
        if isinstance(p["blend"], (bool, str)):
            raise ValidationError("Use a numeric blend fraction", "blend")
        if type(p["fusion"]) is not bool:
            raise ValidationError("Fusion must be a checkbox", "fusion")
        if (p["backend"] == "cuda") != p["target"].startswith("cuda"):
            raise ValidationError("Select CUDA backend and target together", "backend", "target")
        if (p["target"] != "cpu" or not p["fusion"]) and p["engine"] != "classic":
            raise ValidationError("Select Classic for non-CPU targets or disabled fusion", "engine")
        if not isinstance(p["input_name"], str) or len(p["input_name"]) > 256:
            raise ValidationError("Use an exact bounded input name", "input_name")
        names = p["output_names"]
        if (
            not isinstance(names, list)
            or len(names) > 1
            or any(not isinstance(n, str) or not n or len(n) > 256 for n in names)
        ):
            raise ValidationError("Select at most one native terminal head", "output_names")

    def process(self, inputs):
        self.begin_inference()
        try:
            return self.restore(inputs)
        except Exception:
            self.reset_runtime()
            raise

    def restore(self, inputs):
        source, p = inputs["image"], self.parameters
        image_value(source, (np.uint8, np.uint16, np.float32))
        encoding = image_format(source)
        expected = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
        channels = 1 if source.ndim == 2 else source.shape[2]
        if encoding not in expected or channels != expected[encoding]:
            raise ValidationError("Use an image with matching gray/RGB/BGR channel tags")
        height, width = source.shape[:2]
        if (
            min(width, height) < 1
            or max(width, height) > 16384
            or width * height > p["maximum_source_pixels"]
        ):
            raise ValidationError("Source exceeds pixel or axis guard", "maximum_source_pixels")
        if source.dtype == np.float32 and (source.min() < 0 or source.max() > 1):
            raise ValidationError("Float image must use [0,1]; HDR clipping is not implicit")
        if (channels == 1) != (p["color_format"] == "gray"):
            raise ValidationError(
                "Match gray/color model channels; use channel tools for separate planes",
                "color_format",
            )
        selection = selection_mask(inputs, source)
        selected = np.ones((height, width), bool) if selection is None else selection != 0
        work = []
        total = 0
        if p["blend"]:
            for core, context in windows(width, height, p):
                x, y, r, b = core
                if not selected[y:b, x:r].any():
                    continue
                x0, y0, x1, y1 = context
                total += (x1 - x0) * (y1 - y0)
                if len(work) >= p["maximum_tiles"] or total > p["maximum_processed_pixels"]:
                    raise ValidationError(
                        "Restoration tile/context work exceeds its budget",
                        "maximum_tiles",
                        "maximum_processed_pixels",
                    )
                model_channels = 1 if channels == 1 else 3
                if (x1 - x0) * (y1 - y0) * model_channels > p["max_output_elements"]:
                    raise ValidationError(
                        "Predicted same-grid output exceeds its element budget",
                        "max_output_elements",
                    )
                work.append((core, context))
        output = np.asarray(source).copy()
        processed = np.zeros((height, width), np.uint8)
        clipped, maximum_change = 0, 0.0
        if work:
            # Freeze bounded model bytes once for the whole activation, including
            # raw paths; native tile calls cannot reopen a changed source file.
            path, contract = resolve_contract(self)
            raw, identity = model_bytes(path, bounded=True)
            if contract and identity != {k: contract["artifact"][k] for k in ("sha256", "bytes")}:
                raise ValidationError("Bundled ONNX model checksum differs", "contract_path")
            self._snapshot = path, contract, file_key(path), raw, identity
            canonical = opencv_image(source)
            if channels == 4:
                canonical = canonical[..., :3]
            if p["color_format"] == "rgb":
                canonical = canonical[..., ::-1]
            unit = canonical.astype(np.float32)
            if source.dtype != np.float32:
                unit /= np.iinfo(source.dtype).max
            if unit.ndim == 2:
                unit = unit[..., None]
            mean, std = (np.asarray(p[k], np.float32).reshape(1, 1, -1) for k in ("mean", "std"))
            for core, context in work:
                x, y, r, b = core
                x0, y0, x1, y1 = context
                patch = unit[y0:y1, x0:x1]
                normalized = (patch - mean) / std
                blob = (
                    normalized.transpose(2, 0, 1)[None]
                    if p["input_layout"] == "NCHW"
                    else normalized[None]
                )
                if not np.isfinite(blob).all():
                    raise ValidationError("Restoration normalization overflowed", "mean", "std")
                tensors, _ = self.infer_tensors(
                    ((p["input_name"], np.ascontiguousarray(blob, np.float32)),),
                    array_outputs=True,
                    preprocessing={p["input_name"]: preprocessing(p)},
                )
                if len(tensors) != 1:
                    raise ValidationError("Select exactly one restoration output", "output_names")
                tensor = next(iter(tensors.values()))
                expected_shape = (
                    (1, unit.shape[2], y1 - y0, x1 - x0)
                    if p["output_layout"] == "NCHW"
                    else (1, y1 - y0, x1 - x0, unit.shape[2])
                )
                if tensor.shape != expected_shape:
                    raise ValidationError(
                        "Model output must match its input patch grid and channels", "output_layout"
                    )
                values = tensor[0].transpose(1, 2, 0) if p["output_layout"] == "NCHW" else tensor[0]
                restored = (
                    values * std + mean if p["output_mode"] == "restored" else patch - values * std
                )
                restored = patch + float(p["blend"]) * (restored - patch)
                if not np.isfinite(restored).all():
                    raise ValidationError("Restoration output conversion overflowed", "mean", "std")
                restored = restored[y - y0 : b - y0, x - x0 : r - x0]
                choose = selected[y:b, x:r]
                samples = restored[choose]
                clipped += int(np.count_nonzero((samples < 0) | (samples > 1)))
                maximum_change = max(
                    maximum_change, float(np.max(np.abs(samples - unit[y:b, x:r][choose])))
                )
                restored = np.clip(restored, 0, 1)
                if source.dtype != np.float32:
                    restored = np.rint(restored * np.iinfo(source.dtype).max).astype(source.dtype)
                if channels == 1:
                    restored = restored[..., 0]
                elif (
                    encoding in ("rgb", "rgba")
                    and p["color_format"] == "bgr"
                    or encoding in ("bgr", "bgra")
                    and p["color_format"] == "rgb"
                ):
                    restored = restored[..., ::-1]
                target = output[y:b, x:r, :3] if channels == 4 else output[y:b, x:r]
                target[choose] = restored[choose]
                processed[y:b, x:r][choose] = 255
                if self._receipt_calls and p["receipt_mode"] != "off":
                    self._receipt_calls[-1]["restoration_window"] = {
                        "core": list(core),
                        "context": list(context),
                    }
        info = {
            "model": "same_grid_restoration_v1",
            "native_calls": len(work),
            "processing_mode": p["processing_mode"],
            "context_pixels": total,
            "written_pixels": int(np.count_nonzero(processed)),
            "input_dtype": source.dtype.name,
            "color_format": encoding,
            "output_mode": p["output_mode"],
            "blend": p["blend"],
            "clipped_samples": clipped,
            "maximum_unclipped_unit_change": maximum_change,
            "alpha_restored": False,
            "selection_semantics": "write selection; excluded pixels remain model context",
            "tiling_semantics": "disjoint cores; clipped source halo; no synthetic outer padding",
            "halo_guarantee": "model-specific receptive field and whole/tiled comparison required",
        }
        return Result(
            {
                "image": ImageArray(output, encoding, getattr(source, "metadata", {})),
                "processed_mask": ImageArray(processed, "gray"),
                "info": info,
            },
            message=f"{len(work)} native restoration calls; {info['written_pixels']} pixels",
        )
