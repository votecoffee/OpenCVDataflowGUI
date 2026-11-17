"""Ticket-checked, bounded semantic tile fusion on a reference image grid."""

import copy

import cv2
import numpy as np

from .core import Block, Port
from .detection_fusion import edge_mapping, spatial_context
from .outcomes import failed_input
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, require_same_frame
from .processing import image_value
from .recipes import unique_ref
from .selection import SELECTION_INPUTS, selection_mask
from .semantic_segmentation import DecodeSegmentation, segmentation_result
from .tickets import ticket_key

AREA_FIELDS = (
    "selected_class_ids",
    "ignored_class_ids",
    "minimum_confidence",
    "minimum_pixels",
    "maximum_pixels",
    "minimum_fraction",
    "maximum_fraction",
    "maximum_image_pixels",
    "graphics_maximum_contours",
    "graphics_maximum_points",
    "export_segmentation",
)


def tile_record(packet, context, reference, identity, probability_required):
    data = packet.value
    if not isinstance(data, dict) or data.get("model") != "segmentation_tile_v1":
        raise ValidationError("Use Decode semantic segmentation with Export segmentation enabled")
    classes, names = data.get("class_count"), data.get("class_names")
    if (
        type(classes) is not int
        or not 1 <= classes <= 256
        or not isinstance(names, list)
        or len(names) != classes
        or any(not isinstance(s, str) or not s or len(s) > 256 for s in names)
    ):
        raise ValidationError("Tile requires 1..256 ordered class names")
    labels, scores = data.get("labels"), data.get("probabilities")
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.int32
        or labels.ndim != 2
        or min(labels.shape) < 1
        or max(labels.shape) > 16384
    ):
        raise ValidationError("Tile labels require a nonempty bounded int32 image")
    size = list(labels.shape[::-1])
    for payload in (data.get("context", {}), packet.payload):
        spatial_context(payload, size)
        if "ticket" not in payload or ticket_key(payload["ticket"]) != identity:
            raise ValidationError("Tile segmentation belongs to another acquisition ticket")
    embedded = data["context"]
    require_same_frame(
        [ValuePacket(None, embedded), packet],
        message="Segmentation record and wire frames disagree",
    )
    if embedded["spatial"]["root_size"] != packet.payload["spatial"]["root_size"]:
        raise ValidationError("Segmentation record and wire root sizes disagree")
    if embedded.get("pyramid") != packet.payload.get("pyramid"):
        raise ValidationError("Segmentation record and wire pyramid contexts disagree")
    unique_ref([ValuePacket(None, context), ValuePacket(None, embedded), packet])
    # edge_mapping validates the supported relative map; restore pixel-center translation.
    matrix = edge_mapping(packet.payload["spatial"], reference)
    matrix[:2, 2] -= 0.5 - matrix[:2, :2] @ np.array([0.5, 0.5])
    if probability_required and scores is None:
        raise ValidationError(
            "Probability averaging requires full class planes; use label_vote for IDs"
        )
    return data, labels, scores, matrix


class FuseTileSegmentation(Block):
    type_id, title = "ml.fuse_tile_segmentation", "Fuse tile segmentation"
    category = block_type = "ML"
    subtype = "Semantic segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("tiles", "any", visual_type="collection"))
    outputs = (
        *DecodeSegmentation.outputs,
        Port("coverage", required=False),
        Port("weight", required=False),
    )
    defaults = {
        **{key: DecodeSegmentation.defaults[key] for key in AREA_FIELDS},
        "mode": "probability_mean",
        "weighting": "uniform",
        "feather_width": 16.0,
        "expected_tiles": 0,
        "minimum_coverage": 1.0,
        "maximum_input_elements": 32_000_000,
        "maximum_resample_elements": 8_000_000,
        "maximum_mapped_elements": 64_000_000,
    }
    parameter_choices = {
        "mode": ("probability_mean", "label_vote"),
        "weighting": ("uniform", "feather"),
    }
    parameter_ranges = {
        **{
            key: value
            for key, value in DecodeSegmentation.parameter_ranges.items()
            if key in AREA_FIELDS
        },
        "expected_tiles": NumericRange(0, 64, whole=True),
        "feather_width": NumericRange(0.5, 16384),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_input_elements": NumericRange(1, 64_000_000, whole=True),
        "maximum_resample_elements": NumericRange(1, 32_000_000, whole=True),
        "maximum_mapped_elements": NumericRange(1, 256_000_000, whole=True),
    }
    parameter_help = {
        **{key: DecodeSegmentation.parameter_help[key] for key in AREA_FIELDS},
        "mode": (
            "Average exclusive class probabilities with bilinear masked sampling, or "
            "vote on nearest decoded labels. Ties choose the lowest class ID."
        ),
        "weighting": "Uniform support or distance-to-tile-edge feathering, in tile pixels.",
        "feather_width": "Distance in tile pixels from its boundary to full feather weight.",
        "expected_tiles": "Zero accepts 1..64 distinct tiles; positive requires exact count.",
        "minimum_coverage": (
            "Required valid tile support / reference ROI pixels before the final "
            "class/confidence gate; empty ROI has zero coverage."
        ),
        "minimum_confidence": "Minimum winning probability, or vote fraction in label mode.",
        "maximum_input_elements": "Bound total tile label and class-plane samples.",
        "maximum_resample_elements": "Bound classes x reference pixels before accumulation.",
        "maximum_mapped_elements": "Conservative tiles x classes x reference pixels work limit.",
    }
    input_help = {
        "image": "Reference image; requires a known spatial root and acquisition ticket.",
        "mask": "Optional reference-grid ROI; intersects Region.",
        "region": "Reference-grid rectangle/circle/polygon/raster selection.",
        "tiles": (
            "Flat FlowGroup from Join of matching exported segmentation records. Shared "
            "ticket, root, class order and recipe required; crop/scale/flip/right-angle maps."
        ),
    }

    def validate_parameters(self, parameters=None):
        Block.validate_parameters(self, parameters)
        p = self.parameters if parameters is None else parameters
        # Reuse the decoder's typed class lists, limits and cross-field rules.
        DecodeSegmentation(
            parameters={**DecodeSegmentation.defaults, **{key: p[key] for key in AREA_FIELDS}}
        ).validate_parameters()
        for key, limits in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (limits.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric fusion settings", key)

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        h, w = image.shape[:2]
        if max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError("Reference exceeds the pixel/axis budget", "maximum_image_pixels")
        context = self.input_payloads.get("image", {})
        reference = spatial_context(context, [w, h])
        if "ticket" not in context:
            raise ValidationError("Reference image requires an acquisition ticket")
        identity = ticket_key(context["ticket"])
        group = inputs["tiles"]
        if not isinstance(group, FlowGroup):
            raise ValidationError("Connect a flat FlowGroup of exported tile segmentations")
        if (
            not 1 <= len(group.items) <= 64
            or len(group.items) != len(group.labels)
            or any(not isinstance(packet, ValuePacket) for packet in group.items)
        ):
            raise ValidationError("Tile groups require 1..64 labeled value packets")
        if any(
            not isinstance(label, str) or not label or len(label) > 256 for label in group.labels
        ) or len(set(group.labels)) != len(group.labels):
            raise ValidationError("Tile labels must be distinct bounded nonempty strings")
        if p["expected_tiles"] and len(group.items) != p["expected_tiles"]:
            raise ValidationError("Unexpected tile count", "expected_tiles")
        aggregate = self.input_payloads.get("tiles", {})
        if "ticket" in aggregate and ticket_key(aggregate["ticket"]) != identity:
            raise ValidationError("Tile group belongs to another acquisition ticket")
        unique_ref([ValuePacket(None, context), ValuePacket(None, aggregate), *group.items])
        records, elements, class_names = [], 0, None
        for packet in group.items:
            data, labels, scores, matrix = tile_record(
                packet, context, reference, identity, p["mode"] == "probability_mean"
            )
            classes = data["class_count"]
            if class_names is not None and data["class_names"] != class_names:
                raise ValidationError("Tile class names/order disagree")
            class_names = data["class_names"]
            if scores is not None and (
                not isinstance(scores, np.ndarray)
                or scores.dtype != np.float32
                or scores.shape != (classes, *labels.shape)
            ):
                raise ValidationError("Class planes must be float32 C,H,W matching labels")
            elements += labels.size + (scores.size if scores is not None else 0)
            if elements > p["maximum_input_elements"]:
                raise ValidationError(
                    "Tile inputs exceed the sample budget", "maximum_input_elements"
                )
            if np.any(labels < -1) or np.any(labels >= classes):
                raise ValidationError("Tile labels must be -1 or an in-range class ID")
            if scores is not None and (
                not np.isfinite(scores).all()
                or np.any(scores < 0)
                or np.any(scores > 1)
                or not np.allclose(scores.sum(axis=0), 1, atol=1e-4, rtol=0)
            ):
                raise ValidationError(
                    "Tile class probabilities must be finite, in [0,1] and sum to one"
                )
            records.append((labels, scores, matrix))
        if classes * h * w > p["maximum_resample_elements"]:
            raise ValidationError(
                "Class accumulation exceeds the sample budget", "maximum_resample_elements"
            )
        if classes * h * w * len(records) > p["maximum_mapped_elements"]:
            raise ValidationError("Tile mapping exceeds the work budget", "maximum_mapped_elements")
        if any(i >= classes for key in ("selected_class_ids", "ignored_class_ids") for i in p[key]):
            raise ValidationError("Class selection is outside the fused class range")
        total = np.zeros((classes, h, w), np.float64)
        weight = np.zeros((h, w), np.float64)
        coverage = np.zeros((h, w), np.uint16)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        details = []
        for label, packet, (local, scores, matrix) in zip(
            group.labels, group.items, records, strict=True
        ):
            reverse = np.linalg.inv(matrix)
            mx = (reverse[0, 0] * xx + reverse[0, 1] * yy + reverse[0, 2]).astype(np.float32)
            my = (reverse[1, 0] * xx + reverse[1, 1] * yy + reverse[1, 2]).astype(np.float32)
            th, tw = local.shape
            inside = (mx >= -0.5) & (mx < tw - 0.5) & (my >= -0.5) & (my < th - 0.5)
            xi = np.floor(mx + 0.5).clip(0, tw - 1).astype(np.int32)
            yi = np.floor(my + 0.5).clip(0, th - 1).astype(np.int32)
            nearest = local[yi, xi]
            if p["mode"] == "probability_mean":
                valid = (local >= 0).astype(np.float32)
                support = cv2.remap(
                    valid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                inside &= support > 1e-6
            else:
                inside &= nearest >= 0
            tile_weight = inside.astype(np.float64)
            if p["weighting"] == "feather":
                distance = np.minimum.reduce((mx + 0.5, tw - 0.5 - mx, my + 0.5, th - 0.5 - my))
                tile_weight *= np.clip(distance / p["feather_width"], 1e-6, 1)
            for index in range(classes):
                if p["mode"] == "probability_mean":
                    plane = cv2.remap(
                        scores[index] * valid,
                        mx,
                        my,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    plane = np.divide(
                        plane, support, out=np.zeros_like(plane), where=support > 1e-6
                    )
                else:
                    plane = nearest == index
                total[index] += plane * tile_weight
            weight += tile_weight
            coverage += inside.astype(np.uint16)
            details.append(
                {
                    "label": label,
                    "size": [tw, th],
                    "to_reference": matrix.tolist(),
                    "supported_pixels": int(inside.sum()),
                    "failed": failed_input(packet),
                }
            )
        np.divide(total, weight[None], out=total, where=weight[None] > 0)
        labels = total.argmax(axis=0).astype(np.int32)
        confidence = total.max(axis=0).astype(np.float32)
        labels[weight == 0] = -1
        roi = selection_mask(inputs, image)
        eligible = np.ones((h, w), bool) if roi is None else roi != 0
        roi_pixels = int(eligible.sum())
        covered = int(((coverage > 0) & eligible).sum())
        fraction = covered / roi_pixels if roi_pixels else 0.0
        reasons = []
        if fraction < p["minimum_coverage"]:
            reasons.append(
                {"code": "segmentation.coverage", "message": "Insufficient valid tile coverage"}
            )
        if any(row["failed"] for row in details):
            reasons.append(
                {"code": "segmentation.tile_failed", "message": "One or more supplied tiles failed"}
            )
        report = {
            "model": "semantic_segmentation_fusion_v1",
            "mode": p["mode"],
            "weighting": p["weighting"],
            "tiles": details,
            "covered_pixels": covered,
            "coverage_fraction": fraction,
            "minimum_coverage": p["minimum_coverage"],
            "uncovered_pixels_in_roi": roi_pixels - covered,
            "confidence_semantics": (
                "winning_probability"
                if p["mode"] == "probability_mean"
                else "weighted_vote_fraction"
            ),
            "input_elements": elements,
            "grid_convention": "reference_pixel_centers",
        }
        # Unknown pixels retain no evidence; the exported distribution is a dummy
        # background there because labels=-1 remains the validity mask.
        scores_out = None
        if p["export_segmentation"] and p["mode"] == "probability_mean":
            scores_out = total.astype(np.float32)
            scores_out[0, weight == 0] = 1
        result = segmentation_result(
            self, inputs, labels, confidence, classes, report, scores_out, class_names, reasons
        )
        for name, value in (("coverage", coverage), ("weight", weight.astype(np.float32))):
            result.outputs[name] = value
            result.payloads[name] = copy.deepcopy(context)
        if reasons:
            result.message += "; " + "; ".join(row["message"] for row in reasons)
        return result
