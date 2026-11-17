"""Explicit source-grid candidate selection, prompt membership and mask judgments."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .model_contracts import revision
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket
from .processing import image_value
from .prompt_segmentation import (
    ABI,
    CANDIDATES,
    SIDE,
    check_contexts,
    image_identity,
    image_limit,
    inference_revision,
    network_context,
    preprocessing_contract,
    read_prompts,
)
from .raster_regions import region_from_mask
from .selection import SELECTION_INPUTS, selection_mask


class DecodePromptMasks(Block):
    type_id, title = "ml.decode_prompt_masks", "Decode prompted masks"
    category = block_type = "ML"
    subtype = "Prompted segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        *SELECTION_INPUTS,
        Port("candidates", "any"),
        Port("qualities", "data"),
        Port("prompts", "data"),
        Port("info", "data"),
    )
    outputs = (
        Port("image"),
        Port("mask"),
        Port("region", "data", required=False, visual_type="region"),
        Port("logits", required=False),
        Port("quality", "float"),
        Port("pixel_count", "integer"),
        Port("fraction", "float"),
        Port("report", "data"),
    )
    defaults = {
        "selection": "best_quality",
        "candidate_index": 0,
        "prompt_policy": "require_consistent",
        "minimum_quality": 0.0,
        "logit_threshold": 0.0,
        "resampling": "mask_nearest",
        "minimum_pixels": 1,
        "maximum_pixels": 0,
        "minimum_fraction": 0.0,
        "maximum_fraction": 1.0,
        "emit_region": True,
        "emit_logits": False,
        "maximum_region_runs": 100_000,
        "maximum_image_pixels": 4_194_304,
        "graphics_maximum_contours": 2000,
        "graphics_maximum_points": 100_000,
    }
    parameter_choices = {
        "selection": ("best_quality", "index"),
        "prompt_policy": ("report_only", "prefer_consistent", "require_consistent"),
        "resampling": ("mask_nearest", "logits_linear"),
    }
    parameter_ranges = {
        "candidate_index": NumericRange(0, CANDIDATES - 1, whole=True),
        "minimum_quality": NumericRange(-1e6, 1e6),
        "logit_threshold": NumericRange(-1e6, 1e6),
        "minimum_pixels": NumericRange(0, 16_777_216, whole=True),
        "maximum_pixels": NumericRange(0, 16_777_216, whole=True),
        "minimum_fraction": NumericRange(0, 1),
        "maximum_fraction": NumericRange(0, 1),
        "maximum_region_runs": NumericRange(1, 1_000_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_contours": NumericRange(0, 20000, whole=True),
        "graphics_maximum_points": NumericRange(0, 1_000_000, whole=True),
    }
    parameter_help = {
        "selection": (
            "Highest predicted quality, optionally preferring consistent point "
            "membership, or explicit candidate index."
        ),
        "candidate_index": "Zero-based native candidate index; used only by Index selection.",
        "prompt_policy": (
            "Report only; prefer candidates satisfying point labels; or also fail if no "
            "selected candidate satisfies them."
        ),
        "minimum_quality": (
            "Raw predicted IoU-quality floor; a heuristic, not measured IoU or a "
            "calibrated probability."
        ),
        "logit_threshold": (
            "Foreground is logit >= threshold. Default zero matches the pinned native demo."
        ),
        "resampling": (
            "Threshold on the native grid then nearest resize, or linearly resize logits "
            "then threshold."
        ),
        "minimum_pixels": "Minimum selected output-mask pixels to pass.",
        "maximum_pixels": "Maximum selected mask pixels; zero disables the upper limit.",
        "minimum_fraction": "Minimum foreground fraction within selected ROI/mask support.",
        "maximum_fraction": "Maximum foreground fraction within selected ROI/mask support.",
        "emit_region": (
            "Emit an exact bounded scanline-run region; off avoids region encoding work "
            "and emits no Region event."
        ),
        "emit_logits": (
            "Retain the selected source-grid float32 linearly resized logits. "
            "Mask-nearest can differ at boundaries."
        ),
        "maximum_region_runs": "Reject region encoding overflow, never silently truncate.",
        "maximum_image_pixels": "Source-grid allocation limit; maximum axis is 16384.",
        "graphics_maximum_contours": (
            "Bound optional drawn outlines; numerical masks stay complete."
        ),
        "graphics_maximum_points": "Bound optional outline vertices; zero suppresses outlines.",
    }
    input_help = {
        "image": (
            "Exact inference source image; pixels/encoding, current frame and ticket are verified."
        ),
        "candidates": (
            "Three float32 network-grid logit planes from EfficientSAM, preserving their "
            "child payloads."
        ),
        "qualities": "Bound quality record, not an arbitrary score list.",
        "prompts": "Exact source-grid prompt record used for this inference.",
        "info": "Matching model/request/preprocessing/source identity from inference.",
        "mask": (
            "Optional source-grid output support; full-image inference and raw prompt "
            "membership are unchanged."
        ),
        "region": (
            "Optional source-grid output support, intersected with Mask; the box prompt "
            "itself never clips outputs."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, bounds in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (bounds.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric mask settings", key)
        for key in ("emit_region", "emit_logits"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)
        if p["maximum_pixels"] and p["minimum_pixels"] > p["maximum_pixels"]:
            raise ValidationError(
                "Minimum exceeds maximum pixels", "minimum_pixels", "maximum_pixels"
            )
        if p["minimum_fraction"] > p["maximum_fraction"]:
            raise ValidationError(
                "Minimum exceeds maximum fraction", "minimum_fraction", "maximum_fraction"
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Select and inspect a prompted mask on its exact source image grid."
        self.documentation = (
            "<p>Checks model/request/source identity, chooses by predicted quality or index, "
            "and reports point membership separately. A box is a prompt, not a clip. "
            "Mask/region restrict output support after full-image inference. Inspection "
            "failures retain current geometry; malformed/mixed inputs emit no current result. "
            "See docs/prompted-segmentation.md.</p>"
        )

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = image_value(source)
        w, h = size = image_limit(image, p["maximum_image_pixels"])
        record = read_prompts(inputs["prompts"], size)
        info, quality = inputs["info"], inputs["qualities"]
        info_fields = {
            "model",
            "profile",
            "source_size",
            "network_size",
            "request_revision",
            "preprocessing",
            "artifact",
            "image_embedding_cache",
            "source_image",
            "configuration_revision",
            "inference_revision",
        }
        if not isinstance(info, dict) or set(info) != info_fields:
            raise ValidationError("Connect complete EfficientSAM inference info")
        if (
            not isinstance(info, dict)
            or info.get("model") != "efficient_sam_inference_v1"
            or info.get("profile") != ABI
            or info.get("source_size") != [w, h]
            or info.get("network_size") != [SIDE, SIDE]
            or info.get("preprocessing") != preprocessing_contract()
        ):
            raise ValidationError("Inference info does not match the source/EfficientSAM profile")
        request = revision(record)
        token = info.get("inference_revision")
        if (
            info.get("request_revision") != request
            or info.get("source_image") != image_identity(source)
            or token != inference_revision(info)
        ):
            raise ValidationError(
                "Prompt request, model settings or source image changed since inference"
            )
        if (
            not isinstance(quality, dict)
            or set(quality) != {"model", "request_revision", "inference_revision", "values"}
            or quality["model"] != "prompt_mask_qualities_v1"
            or quality["request_revision"] != request
            or quality["inference_revision"] != token
        ):
            raise ValidationError("Quality record belongs to a different prompted inference")
        scores = quality["values"]
        if (
            not isinstance(scores, list)
            or len(scores) != CANDIDATES
            or any(type(v) not in (int, float) for v in scores)
            or not np.isfinite(scores).all()
        ):
            raise ValidationError("EfficientSAM requires three finite predicted qualities")
        source_context = self.input_payloads.get("image", {})
        contexts = [
            self.input_payloads.get(n, {})
            for n in ("image", "candidates", "qualities", "prompts", "info")
        ]
        recipe = check_contexts([*contexts, record["context"]])
        group = inputs["candidates"]
        if (
            not isinstance(group, FlowGroup)
            or len(group.items) != CANDIDATES
            or group.labels != tuple(f"candidate_{i}" for i in range(CANDIDATES))
        ):
            raise ValidationError("Connect the ordered three-candidate logit group")
        expected = network_context(source_context, size)
        planes = []
        for packet in group.items:
            if (
                not isinstance(packet, ValuePacket)
                or packet.payload.get("prompt_request") != request
                or packet.payload.get("prompt_inference") != token
            ):
                raise ValidationError("Candidate logits belong to another prompted inference")
            check_contexts([expected, packet.payload])
            plane = packet.value
            if (
                not isinstance(plane, np.ndarray)
                or plane.dtype != np.float32
                or plane.shape != (SIDE, SIDE)
                or not np.isfinite(plane).all()
            ):
                raise ValidationError(
                    "Each candidate must be a finite 1024x1024 float32 logit image"
                )
            planes.append(plane)
        support = selection_mask(inputs, source)
        rows, masks = [], []
        for i, plane in enumerate(planes):
            if p["resampling"] == "mask_nearest":
                mask = cv2.resize(
                    (plane >= p["logit_threshold"]).astype(np.uint8) * 255,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask = (
                    cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR)
                    >= p["logit_threshold"]
                ).astype(np.uint8) * 255
            membership = [
                bool(mask[int(np.floor(y + 0.5)), int(np.floor(x + 0.5))]) == bool(label)
                for (x, y), label in zip(record["points"], record["labels"], strict=True)
            ]
            rows.append(
                {
                    "index": i,
                    "predicted_quality": float(scores[i]),
                    "source_pixels_before_selection": int(np.count_nonzero(mask)),
                    "point_membership": membership,
                    "prompts_consistent": all(membership),
                }
            )
            masks.append(mask)
        order = sorted(range(CANDIDATES), key=lambda i: (-scores[i], i))
        consistent = [i for i in order if rows[i]["prompts_consistent"]]
        index = (
            p["candidate_index"]
            if p["selection"] == "index"
            else (consistent[0] if p["prompt_policy"] != "report_only" and consistent else order[0])
        )
        mask = masks[index]
        if support is not None:
            mask = cv2.bitwise_and(mask, support)
        pixels = int(np.count_nonzero(mask))
        roi_pixels = w * h if support is None else int(np.count_nonzero(support))
        fraction = pixels / roi_pixels if roi_pixels else 0.0
        reasons = []
        if scores[index] < p["minimum_quality"]:
            reasons.append(
                {
                    "code": "prompt_segmentation.quality",
                    "message": "Predicted quality is below the configured floor",
                }
            )
        if p["prompt_policy"] == "require_consistent" and not rows[index]["prompts_consistent"]:
            reasons.append(
                {
                    "code": "prompt_segmentation.prompts",
                    "message": "Selected mask contradicts foreground/background points",
                }
            )
        if (
            pixels < p["minimum_pixels"]
            or p["maximum_pixels"]
            and pixels > p["maximum_pixels"]
            or not p["minimum_fraction"] <= fraction <= p["maximum_fraction"]
        ):
            reasons.append(
                {
                    "code": "prompt_segmentation.area",
                    "message": "Selected mask area/fraction is outside the configured limits",
                }
            )
        context = copy.deepcopy(source_context)
        if recipe is not None:
            context["recipe_ref"] = copy.deepcopy(recipe)
        region = NO_VALUE
        if p["emit_region"]:
            region = region_from_mask(mask, p["maximum_region_runs"])
            if "spatial" in context:
                region["coordinate_context"] = copy.deepcopy(context["spatial"])
        logits = (
            cv2.resize(planes[index], (w, h), interpolation=cv2.INTER_LINEAR)
            if p["emit_logits"]
            else NO_VALUE
        )
        report = {
            "model": "prompt_mask_decode_v1",
            "selected_index": int(index),
            "candidates": rows,
            "request_revision": request,
            "inference_revision": token,
            "inference": copy.deepcopy(info),
            "pixel_count": pixels,
            "roi_pixels": roi_pixels,
            "fraction": fraction,
            "resampling": p["resampling"],
            "logit_threshold": p["logit_threshold"],
            "box_is_output_clip": False,
            "quality_semantics": (
                "model-predicted IoU heuristic, not measured IoU or calibrated probability"
            ),
            "graphics_truncated": False,
        }
        drawings = []
        if self.generate_graphics:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            points = 0
            color = (60, 230, 120) if not reasons else (60, 80, 240)
            for contour in contours:
                if (
                    len(drawings) >= p["graphics_maximum_contours"]
                    or points + len(contour) > p["graphics_maximum_points"]
                ):
                    report["graphics_truncated"] = True
                    break
                drawings.append(
                    graphic_object("polyline", contour.reshape(-1, 2), color, closed=True)
                )
                points += len(contour)
        return Result(
            {
                "image": source,
                "mask": mask,
                "region": region,
                "logits": logits,
                "quality": float(scores[index]),
                "pixel_count": pixels,
                "fraction": fraction,
                "report": report,
            },
            payloads={
                n: copy.deepcopy(context)
                for n in (
                    "image",
                    "mask",
                    "region",
                    "logits",
                    "quality",
                    "pixel_count",
                    "fraction",
                    "report",
                )
            },
            passed=not reasons,
            score=float(np.clip(scores[index], 0, 1)),
            fail_reasons=reasons,
            drawings={"Prompted segmentation": GraphicDrawing(tuple(drawings))} if drawings else {},
            message=(
                f"Candidate {index}: {pixels} selected pixels, "
                f"predicted quality {scores[index]:.3f}"
            ),
        )
