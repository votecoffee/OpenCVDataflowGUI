"""Bounded source-grid prompts and the pinned fused EfficientSAM ONNX interface."""

import copy
import hashlib

import cv2
import numpy as np

from .core import Block, Port, Result
from .dnn_blocks import DNNInference
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .model_contracts import configuration, revision
from .multi_inference import NETWORK_FIELDS
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, moved_payload, require_same_frame, validate_payload
from .processing import image_value
from .recipes import unique_ref
from .tickets import ticket_key

ABI = "opencv_zoo_efficientsam_ti_2025april"
SIDE, CANDIDATES, MAX_PROMPTS = 1024, 3, 6
NATIVE_ELEMENTS = CANDIDATES * SIDE * SIDE + CANDIDATES
CONTEXT_FIELDS = ("ticket", "spatial", "pyramid", "recipe_ref")


def image_limit(image, maximum):
    h, w = image.shape[:2]
    if max(w, h) > 16384 or w * h > maximum:
        raise ValidationError(
            "Prompt segmentation source exceeds pixel/axis limits", "maximum_image_pixels"
        )
    return w, h


def check_contexts(contexts):
    for value in contexts:
        validate_payload(value)
    require_same_frame(
        [ValuePacket(None, p) for p in contexts], message="Prompt segmentation frames differ"
    )
    known = [p["spatial"] for p in contexts if p.get("spatial", {}).get("root_id") is not None]
    if known and any(s.get("root_size") != known[0].get("root_size") for s in known[1:]):
        raise ValidationError("Prompt segmentation root dimensions differ")
    tickets = [ticket_key(p["ticket"]) for p in contexts if "ticket" in p]
    if tickets and any(t != tickets[0] for t in tickets[1:]):
        raise ValidationError("Prompt segmentation inputs belong to different acquisition tickets")
    pyramids = [p["pyramid"] for p in contexts if "pyramid" in p]
    if pyramids and any(p != pyramids[0] for p in pyramids[1:]):
        raise ValidationError("Prompt segmentation pyramid contexts differ")
    return unique_ref([ValuePacket(None, p) for p in contexts])


def prompt_values(points, labels, box, size=None, *, require_positive=True):
    if not isinstance(points, list) or len(points) > MAX_PROMPTS:
        raise ValidationError("Use at most six point prompts", "points")
    if (
        not isinstance(labels, list)
        or len(labels) != len(points)
        or any(type(v) is not int or v not in (0, 1) for v in labels)
    ):
        raise ValidationError(
            "Each point needs integer label 1 foreground or 0 background", "points", "point_labels"
        )
    if not isinstance(box, list) or len(box) not in (0, 4):
        raise ValidationError("Box is empty or [x,y,width,height]", "box")
    if len(points) + (2 if box else 0) > MAX_PROMPTS:
        raise ValidationError(
            "Points plus two box corners exceed six model prompt slots", "points", "box"
        )
    if require_positive and not (1 in labels or box):
        raise ValidationError("Supply a foreground point or a positive box prompt")
    seen, normalized = set(), []
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(type(v) not in (int, float) for v in point)
        ):
            raise ValidationError("Each point needs numeric [x,y]", "points")
        x, y = map(float, point)
        if not np.isfinite([x, y]).all() or max(abs(x), abs(y)) > 1e6 or (x, y) in seen:
            raise ValidationError("Point prompts must be finite, bounded and distinct", "points")
        if size and not (0 <= x <= size[0] - 1 and 0 <= y <= size[1] - 1):
            raise ValidationError(
                "Point prompts must lie on the current image's pixel-center grid", "points"
            )
        seen.add((x, y))
        normalized.append([x, y])
    rectangle = []
    if box:
        if any(type(v) not in (int, float) for v in box):
            raise ValidationError("Box needs numeric XYWH coordinates", "box")
        x, y, w, h = map(float, box)
        if (
            not np.isfinite([x, y, w, h]).all()
            or max(map(abs, (x, y, w, h))) > 1e6
            or min(w, h) <= 0
        ):
            raise ValidationError("Box must be finite, bounded and positive", "box")
        if size and not (x >= 0 and y >= 0 and x + w <= size[0] and y + h <= size[1]):
            raise ValidationError("Box pixel edges must stay inside the source image", "box")
        rectangle = [x, y, w, h]
    return normalized, list(labels), rectangle


def read_prompts(value, size):
    if not isinstance(value, dict) or set(value) != {
        "model",
        "image_size",
        "points",
        "labels",
        "box",
        "context",
    }:
        raise ValidationError("Connect a segmentation_prompts_v1 record")
    if (
        value["model"] != "segmentation_prompts_v1"
        or value["image_size"] != list(size)
        or not isinstance(value["context"], dict)
    ):
        raise ValidationError("Prompt record/image dimensions disagree")
    validate_payload(value["context"])
    points, labels, box = prompt_values(value["points"], value["labels"], value["box"], size)
    return {
        "model": value["model"],
        "image_size": list(size),
        "points": points,
        "labels": labels,
        "box": box,
        "context": copy.deepcopy(value["context"]),
    }


def native_prompts(record):
    w, h = record["image_size"]
    points, labels = copy.deepcopy(record["points"]), list(record["labels"])
    if record["box"]:
        x, y, width, height = record["box"]
        points.extend(([x, y], [x + width, y + height]))
        labels.extend((2, 3))
    coords = np.zeros((1, 1, MAX_PROMPTS, 2), np.float32)
    kinds = np.full((1, 1, MAX_PROMPTS), -1, np.float32)
    coords[0, 0, : len(points)] = np.asarray(points) * [SIDE / w, SIDE / h]
    kinds[0, 0, : len(points)] = labels
    return coords, kinds


def network_context(context, size):
    sx, sy = size[0] / SIDE, size[1] / SIDE
    return moved_payload(
        context, [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], [SIDE, SIDE]
    )


def preprocessing_contract():
    return {
        "profile": ABI,
        "image": {
            "size": [SIDE, SIDE],
            "resize": "uint8_stretch_linear_then_float32",
            "color": "rgb",
            "scale": 1 / 255,
            "layout": "nchw",
        },
        "prompts": {
            "slots": MAX_PROMPTS,
            "coordinates": "source_xy_times_network_over_source",
            "labels": {
                "background": 0,
                "foreground": 1,
                "box_start": 2,
                "box_end": 3,
                "padding": -1,
            },
        },
    }


def image_identity(image):
    values = np.ascontiguousarray(image)
    return {
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "color": image_format(image),
        "sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
    }


def inference_revision(info):
    return revision(
        {
            k: info[k]
            for k in ("request_revision", "source_image", "artifact", "configuration_revision")
        }
    )


class SegmentationPrompts(Block):
    type_id, title = "ml.segmentation_prompts", "Define segmentation prompts"
    category = block_type = "ML"
    subtype = "Prompted segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        *tuple(
            Port(n, "data", required=False, required_when_connected=True)
            for n in ("points", "point_labels", "box")
        ),
    )
    outputs = (Port("image"), Port("prompts", "data"))
    defaults = {"points": [], "point_labels": [], "box": [], "maximum_image_pixels": 4_194_304}
    parameter_ranges = {"maximum_image_pixels": NumericRange(1, 16_777_216, whole=True)}
    parameter_help = {
        "points": (
            "Source pixel-center [x,y] pairs; at most six total model slots including box corners."
        ),
        "point_labels": "One integer per point: 1 foreground, 0 background. Padding is internal.",
        "box": "Optional [x,y,width,height] source pixel-edge rectangle; costs two prompt slots.",
        "maximum_image_pixels": "Reject larger source grids; maximum axis is 16384.",
    }
    input_help = {
        "image": "Current source grid/ticket. Points and box do not crop the image.",
        **{
            n: (
                "Optional current-event override of the matching saved field; no last-value "
                "fallback."
            )
            for n in ("points", "point_labels", "box")
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Define current-image foreground/background points and an optional box."
        self.documentation = (
            "<p>Point labels are 1 foreground and 0 background. A box uses two of six "
            "native prompt slots. Points do not crop or move the image. Prompts retain "
            "their source ticket, coordinate frame and recipe. "
            "See docs/prompted-segmentation.md.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        prompt_values(p["points"], p["point_labels"], p["box"], require_positive=False)

    def process(self, inputs):
        image = image_value(inputs["image"])
        size = image_limit(image, self.parameters["maximum_image_pixels"])
        context = self.input_payloads.get("image", {})
        check_contexts([self.input_payloads.get(n, {}) for n in inputs])
        points, labels, box = prompt_values(
            *(inputs.get(n, self.parameters[n]) for n in ("points", "point_labels", "box")), size
        )
        record = {
            "model": "segmentation_prompts_v1",
            "image_size": list(size),
            "points": points,
            "labels": labels,
            "box": box,
            "context": {k: copy.deepcopy(context[k]) for k in CONTEXT_FIELDS if k in context},
        }
        graphics = []
        if self.generate_graphics:
            for point, label in zip(points, labels, strict=True):
                graphics.append(
                    graphic_object("marker", [point], (60, 230, 120) if label else (60, 80, 240))
                )
            if box:
                x, y, w, h = box
                graphics.append(
                    graphic_object(
                        "polyline",
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        (240, 180, 60),
                        closed=True,
                    )
                )
        return Result(
            {"image": inputs["image"], "prompts": record},
            payloads={n: copy.deepcopy(context) for n in ("image", "prompts")},
            drawings={"Segmentation prompts": GraphicDrawing(tuple(graphics))} if graphics else {},
            message=f"{len(points)} point prompts, {int(bool(box))} box",
        )


class EfficientSAMInference(DNNInference):
    type_id, title = "ml.efficientsam", "EfficientSAM prompted inference"
    subtype = "Prompted segmentation"
    preview_safe = parallel_safe = False
    process_inputs_readonly = True
    inputs = (Port("image"), Port("prompts", "data"))
    outputs = (
        Port("image"),
        Port("candidates", "any"),
        Port("qualities", "data"),
        Port("prompts", "data"),
        Port("info", "data"),
        Port("receipt", "data", required=False),
    )
    defaults = {
        **{k: copy.deepcopy(DNNInference.defaults[k]) for k in NETWORK_FIELDS},
        "engine": "new",
        "output_names": ["output_masks", "iou_predictions"],
        "max_output_elements": 4_000_000,
        "maximum_image_pixels": 4_194_304,
    }
    parameter_choices = {
        k: v for k, v in DNNInference.parameter_choices.items() if k in NETWORK_FIELDS
    }
    parameter_ranges = {
        "max_output_elements": NumericRange(NATIVE_ELEMENTS, 8_000_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        **{
            k: DNNInference.parameter_help.get(k, "Shared native inference control.")
            for k in NETWORK_FIELDS
        },
        "output_names": (
            "Exactly two terminal names in masks, quality order, for the fixed EfficientSAM ABI."
        ),
        "maximum_image_pixels": (
            "Source grid limit before 1024x1024 RGB preprocessing; axis limit 16384."
        ),
        "max_output_elements": (
            "Budget must fit three 1024x1024 logit planes plus three quality values."
        ),
    }
    input_help = {
        "image": "Original uint8 color/gray source. Full-image inference; no implicit ROI crop.",
        "prompts": (
            "Current segmentation_prompts_v1 record; points/background/box and source "
            "context are checked."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Run the fixed EfficientSAM-Ti image/point/box ONNX interface."
        self.documentation = (
            "<p>Uses full-image uint8 RGB preprocessing and all six padded prompt slots. "
            "Returns three network-grid logit candidates, predicted qualities and bound "
            "request/model/source identity. Decode masks separately. No download or "
            "training occurs during execution; no split image-embedding cache is claimed.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if (
            not isinstance(p["output_names"], list)
            or len(p["output_names"]) != 2
            or any(not isinstance(v, str) or not v or len(v) > 256 for v in p["output_names"])
            or len(set(p["output_names"])) != 2
        ):
            raise ValidationError(
                "Select two distinct terminal outputs in masks, quality order", "output_names"
            )
        if type(p["fusion"]) is not bool:
            raise ValidationError("Fusion must be a checkbox boolean", "fusion")
        for key in self.parameter_ranges:
            if type(p[key]) is not int:
                raise ValidationError("Use integer image/output budgets", key)

    def process(self, inputs):
        self.begin_inference()
        p = self.parameters
        image = image_value(inputs["image"], (np.uint8,), canonical=True)
        size = image_limit(image, p["maximum_image_pixels"])
        record = read_prompts(inputs["prompts"], size)
        contexts = [
            self.input_payloads.get("image", {}),
            self.input_payloads.get("prompts", {}),
            record["context"],
        ]
        recipe = check_contexts(contexts)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            raise ValidationError(
                "Convert alpha images to explicit gray/BGR/RGB before EfficientSAM"
            )
        resized = cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        coords, labels = native_prompts(record)
        tensors, _ = self.infer_tensors(
            (
                ("batched_images", blob),
                ("batched_point_coords", coords),
                ("batched_point_labels", labels),
            ),
            (ABI,),
            array_outputs=True,
            preprocessing=preprocessing_contract(),
        )
        masks, scores = (tensors[n] for n in p["output_names"])
        if masks.shape != (1, 1, CANDIDATES, SIDE, SIDE) or scores.shape != (1, 1, CANDIDATES):
            raise ValidationError("Model outputs differ from the fixed EfficientSAM 3-mask ABI")
        request = revision(record)
        context = copy.deepcopy(contexts[0])
        if recipe is not None:
            context["recipe_ref"] = copy.deepcopy(recipe)
        info = {
            "model": "efficient_sam_inference_v1",
            "profile": ABI,
            "source_size": list(size),
            "network_size": [SIDE, SIDE],
            "request_revision": request,
            "preprocessing": preprocessing_contract(),
            "artifact": copy.deepcopy(self._model_identity),
            "image_embedding_cache": False,
            "source_image": image_identity(inputs["image"]),
            "configuration_revision": revision(configuration(self)),
        }
        token = inference_revision(info)
        info["inference_revision"] = token
        payload = network_context(context, size)
        payload["prompt_request"], payload["prompt_inference"] = request, token
        candidates = FlowGroup(
            tuple(
                ValuePacket(ImageArray(plane.copy(), "gray", payload), copy.deepcopy(payload))
                for plane in masks[0, 0]
            ),
            tuple(f"candidate_{i}" for i in range(CANDIDATES)),
        )
        quality = {
            "model": "prompt_mask_qualities_v1",
            "request_revision": request,
            "inference_revision": token,
            "values": scores[0, 0].tolist(),
        }
        return Result(
            {
                "image": inputs["image"],
                "candidates": candidates,
                "qualities": quality,
                "prompts": record,
                "info": info,
            },
            payloads={
                n: copy.deepcopy(context)
                for n in ("image", "candidates", "qualities", "prompts", "info", "receipt")
            },
            message="Three prompted logit candidates; decode explicitly on the source grid",
        )
