"""One named native LapSRN forward, with independent scale grids and source provenance."""

import copy
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .dnn_blocks import BACKENDS, TARGETS
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, convert_depth, image_format, opencv_image
from .ml_common import local_path
from .parameter_validation import ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, image_with_payload, moved_payload
from .processing import image_value
from .selection import selection_mask
from .super_resolution import CREATE_LOCK, INTERPOLATIONS, SuperResolution


def prepare_luminance(pixels):
    """Match native uint8 Y/Cr/Cb preparation, including its float conversion rounding."""
    source = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) if pixels.ndim == 3 else pixels
    return cv2.addWeighted(source, 1 / 255, source, 0, 0, dtype=cv2.CV_32F)


def reconstruct_luminance(luminance, prepared, size):
    if prepared.ndim == 2:
        return cv2.addWeighted(luminance, 255, luminance, 0, 0, dtype=cv2.CV_8U)
    channels = [luminance, *(cv2.resize(prepared[..., k], size) for k in (1, 2))]
    joined = cv2.merge(channels)
    encoded = cv2.addWeighted(joined, 255, joined, 0, 0, dtype=cv2.CV_8U)
    return cv2.cvtColor(encoded, cv2.COLOR_YCrCb2BGR)


class MultiScaleSuperResolution(SuperResolution):
    type_id, title = "ml.super_resolution_multi", "Multi-scale learned super-resolution"
    outputs = (
        Port("image"),
        Port("images", "any"),
        *(Port(f"image_x{s}", required=False) for s in (2, 4, 8)),
        Port("baseline"),
        Port("processed_mask", visual_type="mask"),
        Port("geometry", "data"),
        Port("info", "data"),
    )
    defaults = {
        **{k: v for k, v in SuperResolution.defaults.items() if k not in ("algorithm", "scale")},
        "output_scales": [2, 4, 8],
        "output_nodes": ["NCHW_output_2x", "NCHW_output_4x", "NCHW_output_8x"],
        "preview_scale": "largest",
    }
    parameter_choices = {
        **{
            k: v
            for k, v in SuperResolution.parameter_choices.items()
            if k not in ("algorithm", "scale")
        },
        "preview_scale": ("largest", "2", "4", "8"),
    }
    parameter_help = {
        **{
            k: v
            for k, v in SuperResolution.parameter_help.items()
            if k not in ("algorithm", "scale")
        },
        "output_scales": "Two or three distinct LapSRN scales from [2,4,8], in output-group order.",
        "output_nodes": (
            "Exact native node names, aligned one-to-one with Output scales. No guessed heads."
        ),
        "preview_scale": (
            "Image/baseline/mask/geometry preview; Largest selects the largest requested scale."
        ),
        "maximum_output_pixels": (
            "Aggregate pixels across all requested scales, bounded before inference/allocation."
        ),
    }
    opencv_functions = ("dnn.readNetFromTensorflow", "dnn.Net.forward")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [p for p in self.exposed_outputs if not p.startswith("image_x")]

    def validate_parameters(self, parameters=None):
        p = self.parameters if parameters is None else parameters
        # Reuse the single-image model/path/backend checks on an immutable adapter.
        super().validate_parameters({**p, "algorithm": "lapsrn", "scale": 8})
        for key in self.parameter_ranges:
            if type(p[key]) is not int:
                raise ValidationError("Use integer super-resolution budgets", key)
        scales, nodes = p["output_scales"], p["output_nodes"]
        if (
            not isinstance(scales, list)
            or not 2 <= len(scales) <= 3
            or any(type(s) is not int or s not in (2, 4, 8) for s in scales)
            or len(set(scales)) != len(scales)
        ):
            raise ValidationError("Select two or three distinct scales from 2,4,8", "output_scales")
        if (
            not isinstance(nodes, list)
            or len(nodes) != len(scales)
            or any(not isinstance(n, str) or not n.strip() or len(n) > 256 for n in nodes)
            or len(set(nodes)) != len(nodes)
        ):
            raise ValidationError(
                "Use distinct node names aligned with the selected scales", "output_nodes"
            )
        if p["preview_scale"] != "largest" and int(p["preview_scale"]) not in scales:
            raise ValidationError(
                "Preview must be one of the requested scales", "preview_scale", "output_scales"
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
        with local_path(self, p["model_path"]).open("rb") as stream:
            raw = stream.read(p["maximum_model_bytes"] + 1)
        if not raw or len(raw) > p["maximum_model_bytes"]:
            raise ValidationError("Model snapshot exceeds the byte budget", "maximum_model_bytes")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != p["model_sha256"]:
            raise ValidationError("Multi-scale model digest differs", "model_sha256")
        backend, target = BACKENDS[p["backend"]], TARGETS[p["target"]]
        if p["target"] != "cpu" and target not in cv2.dnn.getAvailableTargets(backend):
            raise ValidationError("Requested DNN target is unavailable", "backend", "target")
        key = (digest, tuple(p["output_nodes"]), backend, target)
        reused = self._model is not None and self._model_key == key
        if not reused:
            self.reset_runtime()
            # Registration is process-wide and shared with the existing native wrapper.
            with CREATE_LOCK:
                cv2.dnn_superres.DnnSuperResImpl_create()
            with tempfile.TemporaryDirectory(prefix="dataflow-multiscale-superres-") as directory:
                snapshot = Path(directory) / "model.pb"
                snapshot.write_bytes(raw)
                network = cv2.dnn.readNetFromTensorflow(
                    str(snapshot), "", cv2.dnn.ENGINE_CLASSIC, p["output_nodes"]
                )
            if network.empty():
                raise ValidationError("Multi-scale model import produced an empty network")
            network.setPreferableBackend(backend)
            network.setPreferableTarget(target)
            self._model, self._model_key = network, key
        if any(self._model.getLayerId(name) < 0 for name in p["output_nodes"]):
            raise ValidationError("Requested native output node does not exist", "output_nodes")
        prepared = prepare_luminance(pixels)
        y = prepared[..., 0] if prepared.ndim == 3 else prepared
        self._model.setInput(cv2.dnn.blobFromImage(y))
        tensors = self._model.forward(p["output_nodes"])
        if not isinstance(tensors, (tuple, list)) or len(tensors) != len(p["output_scales"]):
            raise ValidationError("Native forward did not deliver every requested scale")
        images = []
        h, w = pixels.shape[:2]
        for tensor, scale in zip(tensors, p["output_scales"], strict=True):
            if (
                not isinstance(tensor, np.ndarray)
                or tensor.dtype != np.float32
                or tensor.shape != (1, 1, h * scale, w * scale)
                or not np.isfinite(tensor).all()
            ):
                raise ValidationError(
                    "Native output grid does not match its declared scale",
                    "output_nodes",
                    "output_scales",
                )
            images.append(reconstruct_luminance(tensor[0, 0], prepared, (w * scale, h * scale)))
        return images, {"sha256": digest, "bytes": len(raw), "native_model_reused": reused}

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image_value(source, (np.uint8, np.uint16, np.float32))
        encoding = image_format(source)
        channels = 1 if source.ndim == 2 else source.shape[2]
        if (
            encoding not in ("gray", "bgr", "rgb", "bgra", "rgba")
            or channels != {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}[encoding]
        ):
            raise ValidationError("Use gray or RGB/BGR channels with matching encoding")
        if source.dtype != np.uint8 and p["depth_mode"] == "uint8_only":
            raise ValidationError(
                "Choose explicit full-scale conversion for this depth", "depth_mode"
            )
        if source.dtype == np.float32 and (source.min() < 0 or source.max() > 1):
            raise ValidationError("Float source must be in [0,1]; HDR clipping is not implicit")
        h, w = source.shape[:2]
        scales = p["output_scales"]
        aggregate = h * w * sum(s * s for s in scales)
        if (
            h * w > p["maximum_input_pixels"]
            or aggregate > p["maximum_output_pixels"]
            or max(h, w) * max(scales) > 16384
        ):
            raise ValidationError(
                "Multi-scale grids exceed aggregate pixel/axis budgets",
                "maximum_output_pixels",
                "maximum_input_pixels",
            )
        selected = selection_mask(inputs, source)
        selected = np.full((h, w), 255, np.uint8) if selected is None else selected
        native_calls = int(bool(np.any(selected)))
        learned, identity = None, None
        if native_calls:
            pixels = convert_depth(opencv_image(source), "uint8")
            if channels == 4:
                pixels = pixels[..., :3]
            learned, identity = self.infer(np.ascontiguousarray(pixels))
        context = self.input_payloads.get("image", {})
        packets, geometries, products = [], [], {}
        for index, (scale, node) in enumerate(zip(scales, p["output_nodes"], strict=True)):
            size = (w * scale, h * scale)
            baseline = cv2.resize(
                np.asarray(source), size, interpolation=INTERPOLATIONS[p["baseline_interpolation"]]
            )
            clipped = 0
            if source.dtype == np.float32:
                clipped = int(np.count_nonzero((baseline < 0) | (baseline > 1)))
                np.clip(baseline, 0, 1, out=baseline)
            mask = cv2.resize(selected, size, interpolation=cv2.INTER_NEAREST)
            output = baseline.copy()
            if learned is not None:
                result = learned[index]
                if encoding in ("rgb", "rgba"):
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                result = convert_depth(result, source.dtype.name)
                writable = output[..., :3] if channels == 4 else output
                writable[mask != 0] = result[mask != 0]
            matrix = [
                [1 / scale, 0, (1 / scale - 1) / 2],
                [0, 1 / scale, (1 / scale - 1) / 2],
                [0, 0, 1],
            ]
            geometry = {
                "scale": scale,
                "node": node,
                "input_size": [w, h],
                "output_size": list(size),
                "output_to_input": matrix,
                "baseline_clipped_samples": clipped,
                "coordinate_convention": "nominal half-pixel grid; learned alignment uncalibrated",
            }
            payload = moved_payload(context, np.asarray(matrix), size)
            payload["super_resolution"] = {
                "scale": scale,
                "node": node,
                "source_size": [w, h],
                "measurement_validity": "reconstruction, not additional measured detail",
            }
            image = image_with_payload(ImageArray(output, encoding), payload)
            packets.append(ValuePacket(image, payload))
            geometries.append(geometry)
            products[scale] = (
                image,
                ImageArray(baseline, encoding),
                ImageArray(mask, "gray"),
                payload,
                geometry,
            )
        preview = max(scales) if p["preview_scale"] == "largest" else int(p["preview_scale"])
        image, baseline, mask, payload, geometry = products[preview]
        outputs = {
            "image": image,
            "images": FlowGroup(tuple(packets), tuple(f"x{s}" for s in scales)),
            **{f"image_x{s}": products[s][0] if s in products else NO_VALUE for s in (2, 4, 8)},
            "baseline": baseline,
            "processed_mask": mask,
            "geometry": geometry,
            "info": {
                "model": "multi_scale_super_resolution_v1",
                "algorithm": "lapsrn",
                "scales": scales.copy(),
                "nodes": p["output_nodes"].copy(),
                "preview_scale": preview,
                "outputs": geometries,
                "aggregate_output_pixels": aggregate,
                "artifact": identity,
                "native_calls": native_calls,
                "native_api": "named_dnn_forward",
                "python_multioutput_binding_qualified": False,
                "engine": "classic",
                "requested_backend": p["backend"],
                "requested_target": p["target"],
                "input_dtype": source.dtype.name,
                "working_dtype": "uint8",
                "color_format": encoding,
                "quantized": source.dtype != np.uint8,
                "alpha_inferred": False,
                "selected_source_pixels": int(np.count_nonzero(selected)),
                "selection_semantics": "full context; nearest selection; baseline elsewhere",
                "native_placement_verified": False,
            },
        }
        payloads = {
            name: copy.deepcopy(payload)
            for name in ("image", "baseline", "processed_mask", "geometry")
        }
        payloads.update({f"image_x{s}": copy.deepcopy(products[s][3]) for s in scales})
        payloads.update(images=copy.deepcopy(context), info=copy.deepcopy(context))
        drawings = (
            {
                "Nominal reconstruction": GraphicDrawing(
                    (
                        graphic_object(
                            "label",
                            [[8, 22]],
                            (60, 230, 120),
                            text=f"LapSRN x{preview}: nominal reconstruction",
                        ),
                    ),
                    from_output=True,
                )
            }
            if self.generate_graphics
            else {}
        )
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            message=f"LapSRN {scales}; {native_calls} native forward; nominal reconstruction",
        )
