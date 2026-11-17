"""Explicit per-pixel model heads restored to the acquisition image's grid."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry, probability
from .graphics_layers import GraphicDrawing, graphic_object
from .images import label_color_map
from .ml_common import selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask


def head_tensor(inputs, p):
    tensors = inputs["tensors"]
    try:
        record = tensors[p["output_name"]] if p["output_name"] else next(iter(tensors.values()))
        if not isinstance(record, dict) or record.get("dtype") != "float32":
            raise ValidationError("Use dequantized float32 tensor_v1 segmentation outputs")
        flat = record.get("values")
        if not isinstance(flat, list) or not 1 <= len(flat) <= p["maximum_tensor_elements"]:
            raise ValidationError("Head exceeds Maximum tensor elements", "maximum_tensor_elements")
        if any(type(v) not in (int, float) for v in flat):
            raise ValidationError("Tensor values must be flat numeric float32 samples")
        values = selected_tensor(
            tensors, p["output_name"], max_elements=p["maximum_tensor_elements"]
        )
    except ValidationError:
        raise
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(str(exc)) from exc
    layout = p["layout"]
    expected = {"NCHW": 4, "NHWC": 4, "CHW": 3, "HWC": 3, "NHW": 3, "HW": 2}[layout]
    if values.ndim != expected:
        raise ValidationError("Tensor dimensions disagree with the selected layout", "layout")
    if layout.startswith("N"):
        if values.shape[0] != 1:
            raise ValidationError("Decode one image at a time; batch dimension must be one")
        values, layout = values[0], layout[1:]
    if layout == "HWC":
        values = values.transpose(2, 0, 1)
    elif layout == "HW":
        values = values[None]
    if max(values.shape[1:]) > 4096 or values.shape[0] > 256:
        raise ValidationError("Segmentation head exceeds 4096 pixels/axis or 256 channels")
    channels = values.shape[0]
    if p["format"] in ("binary", "class_ids") and channels != 1:
        raise ValidationError("Binary/class-ID heads require exactly one channel", "format")
    classes = 2 if p["format"] == "binary" else p["class_count"] or channels
    if p["format"] == "class_ids":
        if np.any(values != np.floor(values)) or np.any(values < -1) or np.any(values >= classes):
            raise ValidationError("Class-ID pixels must be -1 or integer IDs below Class count")
    elif p["class_count"] and p["class_count"] != classes:
        raise ValidationError("Class count disagrees with the score head", "class_count")
    if p["format"] == "multiclass" and classes != channels:
        raise ValidationError("Class count disagrees with the score channels", "class_count")
    if p["labels"] and len(p["labels"]) != classes:
        raise ValidationError("Provide exactly one label per class, including background", "labels")
    for name in ("selected_class_ids", "ignored_class_ids"):
        if any(i >= classes for i in p[name]):
            raise ValidationError("Class selection is outside the head's class range", name)
    return values, classes


def class_scores(values, p):
    """Exclusive class probabilities, preserving explicit binary score meaning."""
    if p["format"] == "class_ids":
        return None
    if p["format"] == "binary":
        foreground = probability(values[0], p["score_activation"])
        return np.stack((1 - foreground, foreground)).astype(np.float32)
    if p["score_activation"] == "softmax":
        shifted = values.astype(np.float64) - np.max(values, axis=0)
        np.exp(shifted, out=shifted)
        scores = shifted / shifted.sum(axis=0)
    else:
        scores = probability(values, "probability")
        if not np.allclose(scores.sum(axis=0), 1, atol=1e-4, rtol=0):
            raise ValidationError("Multiclass probabilities must sum to one at every pixel")
    return scores


def classify(values, p):
    scores = class_scores(values, p)
    if scores is None:
        return values[0].astype(np.int32), None
    labels = (
        (scores[1] >= p["binary_threshold"]).astype(np.int32)
        if p["format"] == "binary"
        else np.argmax(scores, axis=0).astype(np.int32)
    )
    return labels, np.take_along_axis(scores, labels[None], axis=0)[0].astype(np.float32)


def source_coordinates(size, reverse, shape, head_shape):
    h, w = shape[:2]
    oh, ow = head_shape
    # Preprocessing matrices describe continuous pixel EDGES. Sample source pixel
    # centers, then express the result in the output head's pixel-center frame.
    sx, sy = 1 / reverse[0, 0], 1 / reverse[1, 1]
    tx, ty = -reverse[0, 2] * sx, -reverse[1, 2] * sy
    edges = [tx, ty, sx * w + tx, sy * h + ty]
    if (
        edges[0] < -1e-6
        or edges[1] < -1e-6
        or edges[2] > size[0] + 1e-6
        or edges[3] > size[1] + 1e-6
    ):
        raise ValidationError("Preprocessing places source pixels outside the network field")
    x = ((np.arange(w, dtype=np.float64) + 0.5) * sx + tx) * ow / size[0]
    y = ((np.arange(h, dtype=np.float64) + 0.5) * sy + ty) * oh / size[1]
    return x, y


class DecodeSegmentation(Block):
    type_id, title = "ml.decode_segmentation", "Decode semantic segmentation"
    category, block_type, subtype = "ML", "ML", "Semantic segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("tensors", "data"), Port("preprocessing", "data"))
    outputs = (
        Port("image"),
        Port("labels"),
        Port("mask"),
        Port("confidence", required=False),
        Port("pixel_count", "integer"),
        Port("fraction", "float"),
        Port("report", "data"),
        Port("segmentation", "any", required=False, visual_type="collection"),
    )
    defaults = {
        "export_segmentation": False,
        "output_name": "",
        "format": "multiclass",
        "layout": "NCHW",
        "score_activation": "softmax",
        "resampling": "labels_nearest",
        "class_count": 0,
        "labels": [],
        "selected_class_ids": [1],
        "ignored_class_ids": [],
        "minimum_confidence": 0.0,
        "binary_threshold": 0.5,
        "minimum_pixels": 1,
        "maximum_pixels": 0,
        "minimum_fraction": 0.0,
        "maximum_fraction": 1.0,
        "maximum_tensor_elements": 8_000_000,
        "maximum_image_pixels": 4_194_304,
        "maximum_resample_elements": 32_000_000,
        "graphics_maximum_contours": 2000,
        "graphics_maximum_points": 100_000,
    }
    parameter_choices = {
        "format": ("multiclass", "binary", "class_ids"),
        "layout": ("NCHW", "NHWC", "CHW", "HWC", "NHW", "HW"),
        "score_activation": ("softmax", "probability", "sigmoid"),
        "resampling": ("labels_nearest", "scores_linear"),
    }
    parameter_ranges = {
        "class_count": NumericRange(0, 256, whole=True),
        **{
            k: NumericRange(0, 1)
            for k in (
                "minimum_confidence",
                "binary_threshold",
                "minimum_fraction",
                "maximum_fraction",
            )
        },
        **{
            k: NumericRange(0, 16_777_216, whole=True)
            for k in (
                "minimum_pixels",
                "maximum_pixels",
            )
        },
        "maximum_tensor_elements": NumericRange(1, 8_000_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_resample_elements": NumericRange(1, 64_000_000, whole=True),
        "graphics_maximum_contours": NumericRange(1, 20_000, whole=True),
        "graphics_maximum_points": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "export_segmentation": (
            "Emit bounded array-backed class probabilities or decoded labels for tile fusion; "
            "off avoids retaining class planes. Class filtering/ROI gates apply."
        ),
        "output_name": "Named tensor_v1 output; empty selects the first output.",
        "format": "Multiclass exclusive scores, one binary foreground score, or decoded class IDs.",
        "layout": "Explicit axes; N must equal one. HW/NHW denote a single-channel plane.",
        "score_activation": (
            "Exclusive logits: softmax. Binary logits: sigmoid. Probability validates input."
        ),
        "resampling": "Nearest winner labels, or linear raw scores before activation and decision.",
        "class_count": "Zero infers channels; binary has two classes. IDs require 1..256.",
        "labels": "Class names including background; empty uses numeric IDs.",
        "selected_class_ids": "Integer IDs for Mask/area judgment; empty selects all nonignored.",
        "ignored_class_ids": (
            "Integer IDs converted to -1 after winner selection; no runner-up reassignment."
        ),
        "minimum_confidence": (
            "Winning confidence >= threshold; lower pixels become -1. IDs require zero."
        ),
        "binary_threshold": (
            "Foreground probability >= threshold selects class 1; otherwise class 0."
        ),
        "minimum_pixels": "Fewest selected pixels to pass; zero allows an empty mask.",
        "maximum_pixels": "Most selected pixels permitted; zero disables this limit.",
        "minimum_fraction": "Minimum selected pixels divided by the full ROI/mask area.",
        "maximum_fraction": (
            "Maximum selected/ROI fraction; ignored/low confidence stays in denominator."
        ),
        "maximum_tensor_elements": (
            "Reject larger heads before conversion; inference has its own output limit."
        ),
        "maximum_image_pixels": "Source-grid pixel budget; maximum axis is 16384 pixels.",
        "maximum_resample_elements": (
            "Bound channels x source pixels before allocating mapped scores."
        ),
        "graphics_maximum_contours": (
            "Limit optional outlines, including holes; numerical masks remain complete."
        ),
        "graphics_maximum_points": (
            "Limit outline vertices; report truncation without changing the mask."
        ),
    }
    input_help = {
        "image": "Original acquisition image, returned with its pixel format and depth.",
        "mask": "Source-grid output selection intersected with Region after inference.",
        "region": "Optional circle/rectangle/polygon/raster selection in the source image frame.",
        "tensors": "One image's dequantized outputs with explicit axes/score meaning.",
        "preprocessing": "Matching inference sizes and inverse resize/padding maps.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["export_segmentation"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "export_segmentation")
        if not isinstance(p["output_name"], str) or len(p["output_name"]) > 1024:
            raise ValidationError("Output name must be bounded text", "output_name")
        if (
            not isinstance(p["labels"], list)
            or len(p["labels"]) > 256
            or any(not isinstance(s, str) or not s or len(s) > 256 for s in p["labels"])
        ):
            raise ValidationError("Labels must contain up to 256 nonempty names", "labels")
        for name in ("selected_class_ids", "ignored_class_ids"):
            values = p[name]
            if (
                not isinstance(values, list)
                or len(values) > 256
                or any(type(v) is not int or not 0 <= v < 256 for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValidationError("Class IDs must be unique integers in 0..255", name)
        if p["format"] == "multiclass" and p["score_activation"] == "sigmoid":
            raise ValidationError(
                "Exclusive multiclass scores require softmax/probability", "score_activation"
            )
        if p["format"] == "binary" and p["score_activation"] == "softmax":
            raise ValidationError("Binary scores require sigmoid/probability", "score_activation")
        if p["format"] == "class_ids":
            if not p["class_count"]:
                raise ValidationError("Set Class count for decoded IDs", "class_count")
            if p["minimum_confidence"] != 0 or p["resampling"] != "labels_nearest":
                raise ValidationError(
                    "IDs have no confidence and require nearest resampling",
                    "minimum_confidence",
                    "resampling",
                )
        if p["maximum_pixels"] and p["maximum_pixels"] < p["minimum_pixels"]:
            raise ValidationError(
                "Maximum pixels must cover Minimum pixels", "maximum_pixels", "minimum_pixels"
            )
        if p["minimum_fraction"] > p["maximum_fraction"]:
            raise ValidationError(
                "Fraction bounds are reversed", "minimum_fraction", "maximum_fraction"
            )

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        h, w = source.shape[:2]
        if min(h, w) < 1 or max(h, w) > 16384 or h * w > p["maximum_image_pixels"]:
            raise ValidationError(
                "Source exceeds segmentation pixel/axis budget", "maximum_image_pixels"
            )
        image = image_value(source)
        checked = check_context(self, kind="Segmentation")
        size, reverse = preprocessing_geometry(inputs["preprocessing"], image)
        values, classes = head_tensor(inputs, p)
        x, y = source_coordinates(size, reverse, image.shape, values.shape[1:])
        if p["export_segmentation"] and classes * w * h > p["maximum_resample_elements"]:
            raise ValidationError(
                "Exported class planes exceed the resampling budget", "maximum_resample_elements"
            )
        scores = None
        if p["resampling"] == "scores_linear":
            if values.shape[0] * w * h > p["maximum_resample_elements"]:
                raise ValidationError(
                    "Mapped scores exceed the resampling budget", "maximum_resample_elements"
                )
            # Validate raw probabilities/IDs even if sampling would hide invalid cells.
            classify(values, p)
            mx, my = np.meshgrid((x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32))
            mapped = np.empty((values.shape[0], h, w), np.float32)
            for i, plane in enumerate(values):
                mapped[i] = cv2.remap(
                    plane, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
            if not np.isfinite(mapped).all():
                raise ValidationError("Interpolating scores produced nonfinite values")
            labels, confidence = classify(mapped, p)
            if p["export_segmentation"]:
                scores = class_scores(mapped, p)
                if scores is not None:
                    scores = scores.astype(np.float32)
        else:
            labels, confidence = classify(values, p)
            xi = np.floor(x).astype(np.int32).clip(0, labels.shape[1] - 1)
            yi = np.floor(y).astype(np.int32).clip(0, labels.shape[0] - 1)
            labels = labels[yi[:, None], xi]
            if confidence is not None:
                confidence = confidence[yi[:, None], xi]
        if p["export_segmentation"] and p["resampling"] == "labels_nearest":
            raw_scores = class_scores(values, p)
            if raw_scores is not None:
                scores = raw_scores[:, yi[:, None], xi].astype(np.float32)
        details = {
            "model": "semantic_segmentation_decode_v1",
            "format": p["format"],
            "layout": p["layout"],
            "head_shape_chw": list(values.shape),
            "resampling": p["resampling"],
            "grid_convention": "full_network_pixel_edges_source_centers_half_pixel_head",
            "context_checks": checked,
            "preprocessing": copy.deepcopy(inputs["preprocessing"]),
        }
        return segmentation_result(self, inputs, labels, confidence, classes, details, scores)


def segmentation_result(
    block,
    inputs,
    labels,
    confidence,
    classes,
    details,
    scores=None,
    class_names=None,
    extra_fail_reasons=(),
):
    """Common selected-mask judgments, graphics and optional fusion transport."""
    p, source = block.parameters, inputs["image"]
    if class_names is not None:
        p = {**p, "labels": class_names}
    h, w = source.shape[:2]
    image = image_value(source)
    selection = selection_mask(inputs, image)
    eligible = np.ones((h, w), bool) if selection is None else selection != 0
    roi_pixels = int(np.count_nonzero(eligible))
    valid = eligible & (labels >= 0) & ~np.isin(labels, p["ignored_class_ids"])
    if confidence is not None:
        valid &= confidence >= p["minimum_confidence"]
    labels[~valid] = -1
    if confidence is not None:
        confidence[~valid] = 0
    included = valid & (
        np.isin(labels, p["selected_class_ids"]) if p["selected_class_ids"] else True
    )
    mask = included.astype(np.uint8) * 255
    pixels = int(np.count_nonzero(included))
    fraction = pixels / roi_pixels if roi_pixels else 0.0
    passed = (
        pixels >= p["minimum_pixels"]
        and (not p["maximum_pixels"] or pixels <= p["maximum_pixels"])
        and p["minimum_fraction"] <= fraction <= p["maximum_fraction"]
    )
    area_passed = passed
    passed = passed and not extra_fail_reasons
    counts = np.bincount(labels[valid], minlength=classes)
    sums = (
        np.bincount(labels[valid], weights=confidence[valid], minlength=classes)
        if confidence is not None
        else None
    )
    statistics = [
        {
            "class_id": i,
            "label": p["labels"][i] if p["labels"] else str(i),
            "pixels": int(counts[i]),
            "fraction": int(counts[i]) / roi_pixels if roi_pixels else 0.0,
            "mean_confidence": float(sums[i] / counts[i])
            if sums is not None and counts[i]
            else None,
            "selected": (not p["selected_class_ids"] or i in p["selected_class_ids"])
            and i not in p["ignored_class_ids"],
            "ignored": i in p["ignored_class_ids"],
        }
        for i in range(classes)
    ]
    report = {
        **details,
        "class_count": classes,
        "classes": statistics,
        "source_size": [w, h],
        "roi_pixels": roi_pixels,
        "selected_pixels": pixels,
        "ignored_pixels_in_roi": roi_pixels - int(counts.sum()),
        "selected_fraction": fraction,
        "confidence_available": confidence is not None,
        "minimum_confidence": p["minimum_confidence"],
        "graphics_truncated": False,
    }
    drawings, graphics = {}, {}
    if block.generate_graphics:
        graphics["Semantic labels"] = label_color_map(labels + 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        objects, points = [], 0
        color = (60, 230, 120) if passed else (60, 80, 240)
        for contour in contours:
            n = len(contour)
            if (
                len(objects) >= p["graphics_maximum_contours"]
                or points + n > p["graphics_maximum_points"]
            ):
                report["graphics_truncated"] = True
                break
            points += n
            objects.append(graphic_object("polyline", contour.reshape(-1, 2), color, closed=True))
        objects.append(
            graphic_object(
                "label",
                [[2, 16]],
                color,
                text=f"{pixels} selected pixels ({fraction:.1%})"
                + ("; outlines limited" if report["graphics_truncated"] else ""),
            )
        )
        drawings["Semantic selection"] = GraphicDrawing(tuple(objects))
    outputs = {
        "image": source,
        "labels": labels,
        "mask": mask,
        "confidence": confidence if confidence is not None else NO_VALUE,
        "pixel_count": pixels,
        "fraction": fraction,
        "report": report,
    }
    context = block.input_payloads.get("image", {})
    if p["export_segmentation"]:
        outputs["segmentation"] = {
            "model": "segmentation_tile_v1",
            "class_count": classes,
            "class_names": [p["labels"][i] if p["labels"] else str(i) for i in range(classes)],
            "labels": labels.copy(),
            "probabilities": scores,
            "context": copy.deepcopy(context),
        }
    else:
        outputs["segmentation"] = NO_VALUE
    return Result(
        outputs,
        payloads={
            name: copy.deepcopy(context) for name, value in outputs.items() if value is not NO_VALUE
        },
        passed=passed,
        score=float(confidence[included].mean()) if confidence is not None and pixels else None,
        message=f"{pixels} selected pixels ({fraction:.1%} of ROI)",
        fail_reasons=list(extra_fail_reasons)
        + (
            []
            if area_passed
            else [
                {
                    "code": "segmentation.area",
                    "message": "Selected pixel count or ROI fraction outside configured limits",
                }
            ]
        ),
        drawings=drawings,
        graphics=graphics,
    )
