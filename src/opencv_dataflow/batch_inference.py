"""Bounded native image batches with ordered per-image results and provenance."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .dnn_blocks import DNNInference, preprocess
from .ml_common import tensor_record
from .multi_inference import PREPROCESS_FIELDS, PrepareImageTensor, read_tensor, shape
from .parameter_validation import NumericRange, ValidationError
from .payloads import (
    NO_VALUE,
    FlowGroup,
    ValuePacket,
    check_spatial,
    frame,
    image_with_payload,
    metadata,
    require_same_frame,
)
from .recipes import unique_ref
from .tickets import ticket_key

ROW_MODEL = "image_inference_result_v1"


def image_group(value, maximum):
    if not isinstance(value, FlowGroup) or not 1 <= len(value.items) <= maximum:
        raise ValidationError("Connect a bounded Flow group of images", "maximum_images")
    return value


class BatchImageInference(DNNInference):
    type_id, title = "ml.dnn_batch", "Batched ONNX image inference"
    preserve_image_sets = process_inputs_readonly = True
    # Computing has no inspection judgment. Preserve each member's incoming status
    # independently, rather than marking good siblings failed by one failed input.
    outcome_passthrough = True
    preview_safe = parallel_safe = False
    payload_reference = "images"
    inputs = (
        Port("images", "any"),
        Port("masks", "any", required=False, required_when_connected=True),
        Port("regions", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("results", "any"),
        Port("info", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **copy.deepcopy(DNNInference.defaults),
        "input_name": "image",
        "input_shape": [-1, 3, -1, -1],
        "batch_size": 8,
        "tail_policy": "variable",
        "spatial_mode": "configured",
        "maximum_images": 64,
        "maximum_image_pixels": 16_777_216,
        "maximum_source_pixels": 32_000_000,
        "maximum_tensor_elements": 4_000_000,
    }
    parameter_choices = {
        **DNNInference.parameter_choices,
        "tail_policy": ("variable", "pad_repeat", "reject"),
        "spatial_mode": ("configured", "source"),
    }
    parameter_ranges = {
        **{k: v for k, v in PrepareImageTensor.parameter_ranges.items() if k in PREPROCESS_FIELDS},
        "batch_size": NumericRange(1, 64, whole=True),
        "maximum_images": NumericRange(1, 64, whole=True),
        "maximum_image_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_source_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_tensor_elements": NumericRange(1, 8_000_000, whole=True),
        "max_output_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        **DNNInference.parameter_help,
        **{k: v for k, v in PrepareImageTensor.parameter_help.items() if k in PREPROCESS_FIELDS},
        "input_name": "Exact native model image input name; this block binds one NCHW input.",
        "input_shape": "Declared [N,C,H,W]; -1 permits supported variation, never reshaping.",
        "batch_size": "Maximum rows per native call, preserving group order across chunks.",
        "tail_policy": (
            "Variable submits the real tail; Pad repeat copies its last row and discards padded "
            "outputs; Reject requires a complete multiple. Padding can affect batch-coupled models."
        ),
        "spatial_mode": (
            "Configured resizes each image; Source keeps dimensions, requiring one size per group."
        ),
        "width": "Configured mode: network width. Source mode uses each source width (up to 4096).",
        "height": "Configured mode: network height. Source mode uses source height (up to 4096).",
        "maximum_images": "Maximum group entries. No internal waiting or collection of images.",
        "maximum_image_pixels": "Per-source pixel guard; excludes runtime copies.",
        "maximum_source_pixels": "Combined source pixel budget checked before image conversion.",
        "maximum_tensor_elements": "Prepared N*C*H*W budget per native call, including padding.",
        "max_output_elements": "Combined retained outputs for the activation, including padding.",
        "backend": "Native backend; the selected target must be available on this installation.",
        "target": "Native compute target. Non-CPU execution requires the classic engine.",
        "fusion": "Native layer fusion; disabling requires the classic engine.",
    }
    input_help = {
        "images": "Ordered image Flow group, for example from Batch serial entries or TIFF pages.",
        "masks": "Optional same-order, same-label mask group; known tickets and grids must match.",
        "regions": "Optional same-order, same-label region group on the corresponding image grid.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in ((int,) if limits.whole else (int, float)):
                raise ValidationError("Use typed numeric counts, budgets and normalization", name)
        if type(p["fusion"]) is not bool:
            raise ValidationError("Fusion must be boolean", "fusion")
        name = p["input_name"]
        if not isinstance(name, str) or not name or name != name.strip() or len(name) > 256:
            raise ValidationError("Use the exact nonempty model input name", "input_name")
        declared = shape(p["input_shape"], "input_shape", wildcard=True)
        if len(declared) != 4:
            raise ValidationError("Declare four NCHW axes", "input_shape")
        preparation = {k: copy.deepcopy(p[k]) for k in PREPROCESS_FIELDS}
        # Validate shared normalization independently of this block's batch budget.
        preparation.update(width=1, height=1)
        PrepareImageTensor(parameters=preparation).validate_parameters()

    def canonical_outputs(self, arrays, submitted):
        """Validate/normalize native output batch axes before forming ordered rows."""
        if not arrays or any(a.ndim < 1 or a.shape[0] != submitted for a in arrays.values()):
            raise ValidationError("Every selected output must retain the submitted batch axis")
        return arrays

    def process(self, inputs):
        self.begin_inference()
        try:
            return self.infer_group(inputs)
        except Exception:
            self.reset_runtime()
            raise

    def infer_group(self, inputs):
        p = self.parameters
        group = image_group(inputs["images"], p["maximum_images"])
        if any(not isinstance(label, str) or len(label) > 1024 for label in group.labels):
            raise ValidationError("Image group labels must be strings of at most 1024 characters")
        # Routing permits mixed recipes; inference requires one execution context.
        unique_ref(
            [
                ValuePacket(value, self.input_payloads.get(name, {}))
                for name, value in inputs.items()
            ]
        )
        rows, source_pixels, sizes = [], 0, set()
        for entry in group.items:
            image = entry.value
            if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or not image.size:
                raise ValidationError("Every batch entry must be a nonempty intensity image")
            h, w = image.shape[:2]
            source_pixels += h * w
            if (
                max(h, w) > 16384
                or h * w > p["maximum_image_pixels"]
                or source_pixels > p["maximum_source_pixels"]
            ):
                raise ValidationError("Source batch exceeds its image/pixel budgets")
            context = metadata(entry)
            context.setdefault("spatial", frame([w, h]))
            check_spatial(context, [w, h])
            context.pop("tensor_preprocessing", None)
            selected = {"image": image_with_payload(image, context)}
            rows.append((selected, context))
            sizes.add((w, h) if p["spatial_mode"] == "source" else (p["width"], p["height"]))
        if len(sizes) != 1:
            raise ValidationError("Source mode requires one common source size in each group")
        w, h = next(iter(sizes))
        if not 1 <= w <= 4096 or not 1 <= h <= 4096:
            raise ValidationError("Network image axes must be 1..4096 pixels", "width", "height")
        for name, field in (("masks", "mask"), ("regions", "region")):
            if name not in inputs:
                continue
            selections = image_group(inputs[name], p["maximum_images"])
            if len(selections.items) != len(rows) or selections.labels != group.labels:
                raise ValidationError(f"{name} must match image count, labels and order")
            for (selected, context), entry in zip(rows, selections.items, strict=True):
                auxiliary = metadata(entry)
                if "ticket" in context and "ticket" in auxiliary:
                    if ticket_key(context["ticket"]) != ticket_key(auxiliary["ticket"]):
                        raise ValidationError(f"{name} entry belongs to a different ticket")
                require_same_frame([selected["image"], entry])
                selected[field] = (
                    image_with_payload(entry.value, auxiliary) if field == "mask" else entry.value
                )
        count, batch = len(rows), p["batch_size"]
        if p["tail_policy"] == "reject" and count % batch:
            raise ValidationError("Tail policy requires a complete multiple of Batch size")
        channels = 1 if p["color_format"] == "gray" else 3
        chunks = []
        for start in range(0, count, batch):
            real = min(batch, count - start)
            submitted = batch if p["tail_policy"] == "pad_repeat" else real
            actual = [submitted, channels, h, w]
            if any(a != -1 and a != b for a, b in zip(p["input_shape"], actual, strict=True)):
                raise ValidationError(
                    f"Submitted NCHW shape {actual} disagrees with declared Input shape",
                    "input_shape",
                )
            if submitted * channels * h * w > p["maximum_tensor_elements"]:
                raise ValidationError("Prepared batch exceeds its tensor element budget")
            chunks.append((start, real, submitted, actual))
        preparation = {**p, "width": w, "height": h}
        results, calls, retained = [], [], 0
        for start, real, submitted, actual in chunks:
            images, geometries = [], []
            for selected, _ in rows[start : start + real]:
                tensor, geometry = preprocess(selected, preparation)
                images.append(tensor[0].transpose(1, 2, 0))
                geometries.append(geometry)
            images.extend([images[-1]] * (submitted - real))
            blob = cv2.dnn.blobFromImages(images, ddepth=cv2.CV_32F)
            if list(blob.shape) != actual or not np.isfinite(blob).all():
                raise ValidationError("Native batch assembly returned an invalid NCHW tensor")
            tensors, filename = self.infer_tensors(
                ((p["input_name"], blob),),
                ((p["input_name"], tuple(actual)),),
                array_outputs=True,
                preprocessing={p["input_name"]: geometries[0]["preprocessing_contract"]},
            )
            arrays = self.canonical_outputs(tensors, submitted)
            retained += sum(a.size for a in arrays.values())
            if retained > p["max_output_elements"]:
                raise ValidationError(
                    "Combined native outputs exceed the activation element budget"
                )
            calls.append({"start_index": start, "real_rows": real, "submitted_shape": actual})
            for index in range(real):
                selected, context = rows[start + index]
                results.append(
                    ValuePacket(
                        {
                            "model": ROW_MODEL,
                            "index": start + index,
                            "label": group.labels[start + index],
                            "image": selected["image"],
                            "tensors": {
                                name: tensor_record(
                                    a[index : index + 1], p["max_output_elements"], allow_empty=True
                                )
                                for name, a in arrays.items()
                            },
                            "preprocessing": geometries[index],
                            "receipt": {
                                "model": "onnx_image_execution_receipt_v1",
                                "schema_version": 1,
                                "group_index": start + index,
                                "native_call_index": len(calls) - 1,
                                "native_row_index": index,
                                "call": copy.deepcopy(self._receipt_calls[-1]),
                            }
                            if self._receipt_calls
                            else NO_VALUE,
                        },
                        copy.deepcopy(context),
                    )
                )
        info = {
            "model": "image_batch_inference_v1",
            "image_count": count,
            "native_calls": calls,
            "padding_rows": sum(call[2] - call[1] for call in chunks),
            "tail_policy": p["tail_policy"],
            "spatial_mode": p["spatial_mode"],
            "retained_elements_including_padding": retained,
            "input_name": p["input_name"],
        }
        aggregate = copy.deepcopy(self.input_payloads.get("images", {}))
        for name in ("spatial", "pyramid", "tensor_preprocessing"):
            aggregate.pop(name, None)
        return Result(
            {"results": FlowGroup(tuple(results), group.labels), "info": info},
            payloads={"results": aggregate, "info": aggregate},
            message=f"{filename}: {count} images in {len(calls)} native calls",
        )


class UnpackImageInference(Block):
    type_id, title = "ml.unpack_inference", "Unpack image inference result"
    category = block_type = "ML"
    subtype = "DNN inference"
    outcome_passthrough = True
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "result"
    inputs = (Port("result", "any"),)
    outputs = (
        Port("image"),
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("index", "integer"),
        Port("label", "text"),
        Port("receipt", "data", required=False),
    )
    defaults = {}
    input_help = {"result": "One result from Select group entry or Unbatch serial entries."}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs.remove("receipt")

    def process(self, inputs):
        row = inputs["result"]
        if (
            not isinstance(row, dict)
            or row.get("model") != ROW_MODEL
            or type(row.get("index")) is not int
            or not 0 <= row["index"] < 64
            or not isinstance(row.get("label"), str)
            or not isinstance(row.get("image"), np.ndarray)
            or row["image"].ndim not in (2, 3)
            or not row["image"].size
            or not isinstance(row.get("preprocessing"), dict)
            or row["preprocessing"].get("input_size") != list(row["image"].shape[1::-1])
            or not isinstance(row.get("tensors"), dict)
            or not row["tensors"]
        ):
            raise ValidationError("Select one complete image_inference_result_v1 record")
        for name, value in row["tensors"].items():
            tensor = read_tensor(value, "", 8_000_000, name, allow_empty=True)
            if tensor.ndim < 1 or tensor.shape[0] != 1:
                raise ValidationError("Unpacked outputs must contain exactly one image row")
        context = copy.deepcopy(self.input_payloads.get("result", {}))
        check_spatial(context, list(row["image"].shape[1::-1]))
        return Result(
            {port.name: row.get(port.name, NO_VALUE) for port in self.outputs},
            payloads={port.name: copy.deepcopy(context) for port in self.outputs},
            message=f"Row {row['index']}: {row['label']}",
        )


BATCH_INFERENCE_BLOCKS = (BatchImageInference, UnpackImageInference)
