"""Explicit box-relative and coefficient/prototype instance masks with overlap intact."""

import copy
import math

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .detection_decode import DecodeDetections, preprocessing_geometry, probability, tensor_rows
from .graphics_layers import GraphicDrawing, graphic_object
from .images import label_color_map
from .ml_common import selected_tensor, tensor_record
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import image_value
from .raster_regions import region_from_mask
from .selection import rasterize, selection_mask
from .semantic_segmentation import source_coordinates
from .tickets import ticket_key

MASK_MODEL = "instance_masks_v1"
COLORS = ((80, 230, 120), (240, 180, 70), (200, 100, 240), (50, 220, 240), (230, 130, 100))


def bounded_tensor(tensors, name, limit):
    try:
        key = name or next(iter(tensors))
        record = tensors[key]
        if not isinstance(record, dict) or record.get("dtype") != "float32":
            raise ValidationError("Use dequantized float32 tensor_v1 instance heads")
        values = record.get("values")
        if not isinstance(values, list) or not 0 <= len(values) <= limit:
            raise ValidationError("Instance tensor exceeds its element budget")
        if any(type(v) not in (int, float) for v in values):
            raise ValidationError("Instance tensor values must be flat numeric samples")
        with np.errstate(over="ignore", invalid="ignore"):
            return key, selected_tensor(tensors, key, max_elements=limit, allow_empty=True)
    except ValidationError:
        raise
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Invalid instance tensor: " + str(exc)) from exc


def mask_head(inputs, p):
    name, values = bounded_tensor(
        inputs["tensors"], p["mask_output_name"], p["maximum_mask_elements"]
    )
    layout = p["mask_layout"]
    dimensions = {"NCHW": 4, "NHWC": 4, "NHW": 3, "CHW": 3, "HWC": 3}[layout]
    if p["mask_mode"] != "prototypes" and values.ndim == dimensions + 1 and values.shape[0] == 1:
        values = values[0]  # One image selected from a batch retains its leading B=1 axis.
    if values.ndim != dimensions:
        raise ValidationError("Mask dimensions disagree with Mask layout", "mask_layout")
    if p["mask_mode"] == "prototypes":
        if layout.startswith("N"):
            if values.shape[0] != 1:
                raise ValidationError("Prototype masks require one image, not an image batch")
            values, layout = values[0], layout[1:]
        if layout == "HWC":
            values = values.transpose(2, 0, 1)
    else:
        if layout == "NHWC":
            values = values.transpose(0, 3, 1, 2)
        elif layout == "NHW":
            values = values[:, None]
        if p["mask_mode"] == "roi" and values.shape[1] != 1:
            raise ValidationError("ROI mode requires one mask channel per detection", "mask_mode")
        if p["mask_activation"] == "probability":
            probability(values, "probability")
    if min(values.shape[-3:]) < 1 or max(values.shape[-2:]) > 4096 or values.shape[-3] > 256:
        raise ValidationError("Mask head exceeds 4096 pixels/axis or 256 channels")
    return name, values


def detection_inputs(inputs, p, masks, mask_name):
    key, raw = bounded_tensor(inputs["tensors"], p["output_name"], 2_000_000)
    if key == mask_name:
        raise ValidationError("Detection and mask outputs must be different named tensors")
    adapted = dict(inputs)
    coefficients = None
    if p["mask_mode"] == "prototypes":
        k = masks.shape[0]
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 2:
            raise ValidationError("Packed prototype rows require [N,C], [C,N] or batch one")
        if p["layout"] == "channels_first":
            raw = raw.T
        base = 6 if p["format"] == "xywh_objectness" else 5
        if raw.shape[1] < k + base:
            raise ValidationError(
                "Packed rows need box/scores followed by one coefficient per prototype"
            )
        coefficients = raw[:, -k:].astype(np.float64)
        rows = raw[:, :-k]
        tensors = dict(inputs["tensors"])
        tensors[key] = tensor_record(
            rows.T if p["layout"] == "channels_first" else rows, 2_000_000, allow_empty=True
        )
        adapted["tensors"] = tensors
    rows = tensor_rows(adapted, p)
    if p["mask_mode"] != "prototypes" and masks.shape[0] != len(rows):
        raise ValidationError(
            "ROI mask rows must match ALL raw detection rows before filtering/NMS"
        )
    if p["selection_mode"] == "pixels":
        adapted = {k: v for k, v in adapted.items() if k not in ("mask", "region")}
    return adapted, coefficients


def pixel_span(box, width, height):
    left, top, right, bottom = box
    # Pixel centers inside a half-open continuous box; do not stretch a clipped ROI.
    x0, y0 = max(0, math.ceil(left - 0.5)), max(0, math.ceil(top - 0.5))
    x1, y1 = min(width, math.ceil(right - 0.5)), min(height, math.ceil(bottom - 0.5))
    return x0, y0, max(x0, x1), max(y0, y1)


def sample_mask(plane, x, y, p):
    if p["activation_order"] == "before_resampling":
        plane = probability(plane, p["mask_activation"])
    if p["mask_interpolation"] == "nearest":
        xi = np.floor(x).astype(np.int32).clip(0, plane.shape[1] - 1)
        yi = np.floor(y).astype(np.int32).clip(0, plane.shape[0] - 1)
        sampled = plane[yi[:, None], xi]
    else:
        mx, my = np.meshgrid((x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32))
        sampled = cv2.remap(plane, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if not np.isfinite(sampled).all():
        raise ValidationError("Mask resampling returned nonfinite values")
    if p["activation_order"] == "after_resampling":
        sampled = probability(sampled, p["mask_activation"])
    return sampled


class DecodeInstances(DecodeDetections):
    type_id, title, subtype = (
        "ml.decode_instances",
        "Decode instance segmentation",
        "Instance segmentation",
    )
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("tensors", "data"),
        Port("preprocessing", "data"),
    )
    outputs = (
        Port("image"),
        Port("collection", "data", visual_type="collection"),
        Port("count", "integer"),
        Port("mask"),
        Port("labels"),
        Port("overlap_count"),
        Port("report", "data"),
    )
    defaults = {
        **DecodeDetections.defaults,
        "maximum_results": 256,
        "maximum_image_pixels": 4_194_304,
        "mask_mode": "roi",
        "mask_output_name": "masks",
        "mask_layout": "NCHW",
        "mask_class_offset": 0,
        "mask_activation": "probability",
        "activation_order": "before_resampling",
        "mask_interpolation": "linear",
        "mask_threshold": 0.5,
        "crop_to_box": True,
        "selection_mode": "pixels",
        "minimum_mask_pixels": 1,
        "maximum_mask_pixels": 0,
        "label_overlap": "detection_score",
        "maximum_mask_elements": 8_000_000,
        "maximum_decoded_pixels": 16_777_216,
        "maximum_prototype_operations": 128_000_000,
        "maximum_region_runs": 200_000,
        "graphics_maximum_contours": 2000,
        "graphics_maximum_points": 100_000,
    }
    parameter_choices = {
        **DecodeDetections.parameter_choices,
        "mask_mode": ("roi", "roi_per_class", "prototypes"),
        "mask_layout": ("NCHW", "NHWC", "NHW", "CHW", "HWC"),
        "mask_activation": ("probability", "sigmoid"),
        "activation_order": ("before_resampling", "after_resampling"),
        "mask_interpolation": ("nearest", "linear"),
        "selection_mode": ("pixels", "centers", "centers_and_pixels"),
        "label_overlap": ("detection_score", "mask_confidence", "score_times_mask"),
    }
    parameter_ranges = {
        **DecodeDetections.parameter_ranges,
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "mask_class_offset": NumericRange(-4096, 4096, whole=True),
        "mask_threshold": NumericRange(0, 1),
        **{
            k: NumericRange(0, 16_777_216, whole=True)
            for k in ("minimum_mask_pixels", "maximum_mask_pixels")
        },
        "maximum_mask_elements": NumericRange(1, 8_000_000, whole=True),
        "maximum_decoded_pixels": NumericRange(1, 128_000_000, whole=True),
        "maximum_prototype_operations": NumericRange(1, 512_000_000, whole=True),
        "maximum_region_runs": NumericRange(1, 1_000_000, whole=True),
        "graphics_maximum_contours": NumericRange(1, 20_000, whole=True),
        "graphics_maximum_points": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        **DecodeDetections.parameter_help,
        "mask_mode": (
            "One ROI mask, class-indexed ROI masks, or packed coefficients plus full- "
            "field prototypes."
        ),
        "mask_output_name": (
            "Exact ROI mask/prototype tensor name; distinct from the detection head."
        ),
        "mask_layout": (
            "ROI: NCHW/NHWC/NHW, N=raw rows. Prototypes: CHW/HWC or batch-one NCHW/NHWC."
        ),
        "mask_class_offset": (
            "roi_per_class channel = final detection class ID + offset; unused in other modes."
        ),
        "mask_activation": (
            "Declared mask probabilities, or sigmoid logits; independent of detection scores."
        ),
        "activation_order": (
            "Activate the raw mask grid before interpolation, or activate sampled logits afterward."
        ),
        "mask_interpolation": (
            "Nearest head cells or native floating linear interpolation with replicated head edges."
        ),
        "mask_threshold": (
            "Mask probability >= threshold is foreground, within source/box/selection support."
        ),
        "crop_to_box": (
            "Prototypes only: restrict masks to original predicted boxes. ROI masks "
            "always use their boxes."
        ),
        "selection_mode": (
            "Gate mask pixels, detection centers, or both with the optional mask/region."
        ),
        "minimum_mask_pixels": (
            "Drop instances below this selected source-pixel area; zero keeps valid empty masks."
        ),
        "maximum_mask_pixels": "Drop larger masks; zero disables this area filter.",
        "label_overlap": (
            "Resolve only the summary Labels image. Separate instance masks retain ALL "
            "overlapping pixels."
        ),
        "maximum_mask_elements": "Reject larger serialized mask/prototype heads before conversion.",
        "maximum_decoded_pixels": (
            "Bound summed source tiles before resampling, rather than allocate N full- size masks."
        ),
        "maximum_prototype_operations": (
            "Bound retained instances x prototype channels x prototype pixels before mixing."
        ),
        "maximum_region_runs": (
            "Total scanline-run budget across all retained instance masks; never truncate masks."
        ),
        "graphics_maximum_contours": (
            "Bound optional outline objects; report truncation without changing masks."
        ),
        "graphics_maximum_points": (
            "Bound optional outline vertices; report truncation without changing masks."
        ),
        "minimum_count": "Fewest instances remaining AFTER mask area filtering required to pass.",
        "maximum_count": (
            "Most instances after mask filtering; zero disables the upper count judgment."
        ),
        "format": (
            "Detection row schema. Prototype mode appends K coefficients after XYWH class "
            "scores/objectness."
        ),
    }
    input_help = {
        **DecodeDetections.input_help,
        "tensors": (
            "Matching detection AND mask/prototype heads; row IDs preserve mask identity "
            "through NMS."
        ),
        "mask": "Source-grid selection of mask pixels and/or centers according to Selection mode.",
        "region": "Source-grid circle/rectangle/polygon/raster selection; intersected with Mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["mask_output_name"], str) or not 1 <= len(p["mask_output_name"]) <= 256:
            raise ValidationError(
                "Mask output name must contain 1..256 characters", "mask_output_name"
            )
        if p["mask_output_name"] == p["output_name"]:
            raise ValidationError(
                "Detection and mask output names must differ", "mask_output_name", "output_name"
            )
        if type(p["crop_to_box"]) is not bool:
            raise ValidationError("Crop to box is a checkbox", "crop_to_box")
        if p["mask_mode"] == "prototypes":
            if p["mask_layout"] == "NHW" or p["format"] not in ("xywh_scores", "xywh_objectness"):
                raise ValidationError(
                    "Prototypes need CHW/HWC and packed XYWH score/objectness rows",
                    "mask_layout",
                    "format",
                )
        elif p["mask_layout"] not in ("NCHW", "NHWC", "NHW"):
            raise ValidationError("ROI masks require NCHW, NHWC or NHW", "mask_layout")
        if p["mask_class_offset"] and p["mask_mode"] != "roi_per_class":
            raise ValidationError(
                "Mask class offset only applies to class-indexed ROI masks", "mask_class_offset"
            )
        if p["maximum_mask_pixels"] and p["maximum_mask_pixels"] < p["minimum_mask_pixels"]:
            raise ValidationError(
                "Mask area bounds are reversed", "minimum_mask_pixels", "maximum_mask_pixels"
            )

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        # Reuse the ordinary decoder algorithm, without nesting its executor:
        # timing, outcomes and exceptions publish exactly once for this invocation.
        mask_name, masks = mask_head(inputs, p)
        adapted, coefficients = detection_inputs(inputs, p, masks, mask_name)
        detected = super().process(adapted)
        h, w = source.shape[:2]
        selection = selection_mask(inputs, source)
        size, reverse = preprocessing_geometry(inputs["preprocessing"], source)
        proposals = detected.outputs["collection"]["items"]
        operations = len(proposals) * masks.size if coefficients is not None else 0
        if operations > p["maximum_prototype_operations"]:
            raise ValidationError(
                "Prototype mixing exceeds its operation budget", "maximum_prototype_operations"
            )
        if coefficients is not None:
            masks = masks.astype(np.float64)
            full_x, full_y = source_coordinates(size, reverse, source.shape, masks.shape[1:])
        tiles = []
        for item in proposals:
            box = item["geometry"]["unclipped_xyxy"]
            tiles.append(
                pixel_span(box, w, h) if coefficients is None or p["crop_to_box"] else (0, 0, w, h)
            )
        decoded_pixels = sum((b - a) * (d - c) for a, c, b, d in tiles)
        if decoded_pixels > p["maximum_decoded_pixels"]:
            raise ValidationError(
                "Instance source tiles exceed the decoded pixel budget", "maximum_decoded_pixels"
            )
        union, labels, overlaps = (
            np.zeros((h, w), np.uint8),
            np.zeros((h, w), np.int32),
            np.zeros((h, w), np.int32),
        )
        priority = np.full((h, w), -1, np.float64)
        items, filtered, objects, runs, points = [], [], [], 0, 0
        graphics_truncated = False
        for item, (x0, y0, x1, y1) in zip(proposals, tiles, strict=True):
            identity, class_id = item["id"], item["class_id"]
            if coefficients is not None:
                plane = (coefficients[identity] @ masks.reshape(masks.shape[0], -1)).reshape(
                    masks.shape[1:]
                )
                x, y = full_x[x0:x1], full_y[y0:y1]
            else:
                channel = (
                    class_id + p["mask_class_offset"] if p["mask_mode"] == "roi_per_class" else 0
                )
                if not 0 <= channel < masks.shape[1]:
                    raise ValidationError(
                        "Detection class maps outside ROI mask channels", "mask_class_offset"
                    )
                plane = masks[identity, channel].astype(np.float64)
                left, top, right, bottom = item["geometry"]["unclipped_xyxy"]
                x = (np.arange(x0, x1) + 0.5 - left) * plane.shape[1] / (right - left)
                y = (np.arange(y0, y1) + 0.5 - top) * plane.shape[0] / (bottom - top)
            if x1 > x0 and y1 > y0:
                confidence = sample_mask(plane, x, y, p)
                binary = confidence >= p["mask_threshold"]
                if selection is not None and p["selection_mode"] != "centers":
                    binary &= selection[y0:y1, x0:x1] != 0
            else:
                confidence, binary = np.zeros((0, 0)), np.zeros((0, 0), bool)
            area = int(np.count_nonzero(binary))
            if (
                area < p["minimum_mask_pixels"]
                or p["maximum_mask_pixels"]
                and area > p["maximum_mask_pixels"]
            ):
                filtered.append({"id": identity, "mask_pixels": area, "reason": "mask_area"})
                continue
            region = region_from_mask(binary, p["maximum_region_runs"] - runs)
            region["image_size"] = [w, h]
            for run in region["runs"]:
                run[0] += y0
                run[1] += x0
            runs += len(region["runs"])
            item = copy.deepcopy(item)
            item["mask_region"], item["label_id"] = region, identity + 1
            yy, xx = np.nonzero(binary)
            item["metrics"].update(
                mask_area=area,
                mask_mean_confidence=float(confidence[binary].mean()) if area else None,
                mask_center_x=float(xx.mean() + x0) if area else None,
                mask_center_y=float(yy.mean() + y0) if area else None,
                mask_coverage=area / max(1, binary.size),
            )
            items.append(item)
            if area:
                union[y0:y1, x0:x1][binary] = 255
                overlaps[y0:y1, x0:x1] += binary
                strength = (
                    confidence
                    if p["label_overlap"] == "mask_confidence"
                    else confidence * item["score"]
                    if p["label_overlap"] == "score_times_mask"
                    else np.full(binary.shape, item["score"])
                )
                best = priority[y0:y1, x0:x1]
                wins = binary & (strength > best)
                labels[y0:y1, x0:x1][wins] = identity + 1
                best[wins] = strength[wins]
            if self.generate_graphics:
                color = COLORS[identity % len(COLORS)]
                if not graphics_truncated:
                    contours, _ = (
                        cv2.findContours(
                            binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if area
                        else ([], None)
                    )
                    for contour in contours:
                        if (
                            len(objects) >= p["graphics_maximum_contours"]
                            or points + len(contour) > p["graphics_maximum_points"]
                        ):
                            graphics_truncated = True
                            break
                        points += len(contour)
                        objects.append(
                            graphic_object(
                                "polyline", contour.reshape(-1, 2) + [x0, y0], color, closed=True
                            )
                        )
                    if len(objects) < p["graphics_maximum_contours"]:
                        objects.append(
                            graphic_object(
                                "label",
                                [[x0, max(14, y0)]],
                                color,
                                text=f"#{identity} {item['label']} {item['score']:.3f}; {area}px",
                            )
                        )
        count = len(items)
        passed = count >= p["minimum_count"] and (
            not p["maximum_count"] or count <= p["maximum_count"]
        )
        context = self.input_payloads.get("image", {})
        data = collection("boxes", items, source, context)
        data["instance_masks"] = {
            "model": MASK_MODEL,
            "encoding": "image_region_v1",
            "overlap_preserved": True,
        }
        report = {
            "model": "instance_segmentation_decode_v1",
            "detection": detected.outputs["report"],
            "mask_mode": p["mask_mode"],
            "mask_head_shape": list(masks.shape),
            "activation_order": p["activation_order"],
            "mask_activation": p["mask_activation"],
            "mask_interpolation": p["mask_interpolation"],
            "mask_threshold": p["mask_threshold"],
            "selection_mode": p["selection_mode"],
            "retained_boxes": len(proposals),
            "retained_instances": count,
            "area_filtered": filtered,
            "decoded_source_pixels": decoded_pixels,
            "prototype_operations": operations,
            "region_runs": runs,
            "overlap_pixels": int(np.count_nonzero(overlaps > 1)),
            "label_overlap": p["label_overlap"],
            "label_ids": [{"instance_id": i["id"], "label_id": i["label_id"]} for i in items],
            "graphics_truncated": graphics_truncated,
        }
        outputs = {
            "image": source,
            "collection": data,
            "count": count,
            "mask": union,
            "labels": labels,
            "overlap_count": overlaps,
            "report": report,
        }
        return Result(
            outputs,
            payloads={name: copy.deepcopy(context) for name in outputs},
            passed=passed,
            score=max((i["score"] for i in items), default=0),
            message=f"{count} instances; {report['overlap_pixels']} overlapping pixels",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "instance.count",
                    "message": "Instance count after mask filtering outside configured limits",
                }
            ],
            drawings={"Instance masks": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
            graphics={"Instance labels": label_color_map(labels)} if self.generate_graphics else {},
        )


class SelectInstanceMask(Block):
    type_id, title = "ml.select_instance_mask", "Select instance mask"
    category, block_type, subtype = "ML", "ML", "Instance segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("image"), Port("collection", "data", visual_type="collection"))
    outputs = (
        Port("image"),
        Port("mask"),
        Port("region", "data", visual_type="region"),
        Port("item", "data", required=False),
        Port("found", "boolean"),
    )
    defaults = {
        "selection": "index",
        "index": 0,
        "instance_id": 0,
        "missing": "empty",
        "maximum_region_runs": 200_000,
        "maximum_image_pixels": 4_194_304,
    }
    parameter_choices = {"selection": ("index", "id"), "missing": ("empty", "error")}
    parameter_ranges = {
        "index": NumericRange(0, 9999, whole=True),
        "instance_id": NumericRange(0, 99999, whole=True),
        "maximum_region_runs": NumericRange(1, 1_000_000, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        "selection": ("Choose collection position after sorting/filtering, or original row ID."),
        "index": "Zero-based position in the current collection.",
        "instance_id": (
            "Original detection-row ID, preserved through suppression/sorting/filtering."
        ),
        "missing": "Emit a normal failed empty mask/region, or reject without current output.",
        "maximum_region_runs": "Reject larger selected mask encodings before rasterizing.",
        "maximum_image_pixels": "Bound the full source mask allocation.",
    }
    input_help = {
        "image": "Original source-grid image, with the matching ticket/frame/recipe.",
        "collection": (
            "Decode instance segmentation collection, optionally sorted/filtered; "
            "overlaps remain intact."
        ),
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        h, w = image.shape[:2]
        if h * w > self.parameters["maximum_image_pixels"]:
            raise ValidationError(
                "Source exceeds selected-mask pixel budget", "maximum_image_pixels"
            )
        data, other = read_collection(self, inputs)
        if (
            data.get("instance_masks", {}).get("model") != MASK_MODEL
            or data["image_size"] != [w, h]
            or data.get("origin", [0, 0]) != [0, 0]
        ):
            raise ValidationError("Use a source-aligned instance-mask collection")
        context = self.input_payloads.get("image", {})
        require_same_frame(
            [ValuePacket(None, context), ValuePacket(None, other)],
            message="Instance collection uses another source frame",
        )
        if (
            "ticket" in context
            and "ticket" in other
            and ticket_key(context["ticket"]) != ticket_key(other["ticket"])
        ):
            raise ValidationError("Instance collection belongs to another acquisition ticket")
        if (
            "recipe_ref" in context
            and "recipe_ref" in other
            and context["recipe_ref"] != other["recipe_ref"]
        ):
            raise ValidationError("Instance collection belongs to another recipe revision")
        p = self.parameters
        item = (
            next((i for i in data["items"] if i["id"] == p["instance_id"]), None)
            if p["selection"] == "id"
            else data["items"][p["index"]]
            if p["index"] < len(data["items"])
            else None
        )
        if item is None and p["missing"] == "error":
            raise ValidationError(
                "Requested instance is absent", "instance_id" if p["selection"] == "id" else "index"
            )
        region = item.get("mask_region") if item else region_from_mask(np.zeros((h, w), np.uint8))
        if (
            not isinstance(region, dict)
            or region.get("shape") != "raster"
            or not isinstance(region.get("runs"), list)
            or len(region["runs"]) > p["maximum_region_runs"]
        ):
            raise ValidationError(
                "Invalid or over-budget instance mask region", "maximum_region_runs"
            )
        mask = rasterize(region, (h, w))
        if item is not None and (
            type(item.get("score")) not in (int, float)
            or not math.isfinite(item["score"])
            or not 0 <= item["score"] <= 1
        ):
            raise ValidationError("Instance score must be a finite confidence in [0,1]")
        outputs = {
            "image": source,
            "mask": mask,
            "region": region,
            "item": item if item is not None else NO_VALUE,
            "found": item is not None,
        }
        return Result(
            outputs,
            payloads={name: copy.deepcopy(context) for name in outputs},
            passed=item is not None,
            score=item["score"] if item else None,
            message=f"Selected instance {item['id']}" if item else "Requested instance absent",
            fail_reasons=[]
            if item
            else [
                {
                    "code": "instance.missing",
                    "message": "Requested instance absent from current collection",
                }
            ],
        )


class PackDetectionHeads(Block):
    type_id, title = "ml.pack_detection_heads", "Pack separate detection heads"
    category, block_type, subtype = "ML", "ML", "Instance segmentation"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("tensors", "data"),)
    outputs = (Port("tensors", "data"), Port("count", "integer"), Port("report", "data"))
    defaults = {
        "boxes_output": "boxes",
        "scores_output": "scores",
        "classes_output": "labels",
        "output_name": "packed_boxes",
        "maximum_rows": 10_000,
        "maximum_total_elements": 8_000_000,
    }
    parameter_ranges = {
        "maximum_rows": NumericRange(1, 100_000, whole=True),
        "maximum_total_elements": NumericRange(1, 8_000_000, whole=True),
    }
    parameter_help = {
        "boxes_output": "Named XYXY [N,4] or [1,N,4] head, in the export's coordinate units.",
        "scores_output": "Named [N] scores with optional singleton axes; probabilities in [0,1].",
        "classes_output": "Named [N] nonnegative integer class IDs, with optional singleton axes.",
        "output_name": "New [1,N,6] XYXY/score/class head; retain original heads.",
        "maximum_rows": "Reject more boxes before packing.",
        "maximum_total_elements": "Bound original heads plus the packed head before copying.",
    }
    input_help = {
        "tensors": "One inference activation's boxes/scores/classes and optional mask heads."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        fields = ("boxes_output", "scores_output", "classes_output", "output_name")
        for field in fields:
            if not isinstance(p[field], str) or not 1 <= len(p[field]) <= 256:
                raise ValidationError("Tensor names require 1..256 characters", field)
        if len({p[f] for f in fields}) != len(fields):
            raise ValidationError("Use four distinct tensor names", *fields)

    def process(self, inputs):
        p, tensors = self.parameters, inputs["tensors"]
        if not isinstance(tensors, dict) or not 1 <= len(tensors) <= 64:
            raise ValidationError("Connect a bounded dictionary of inference tensors")
        if p["output_name"] in tensors:
            raise ValidationError(
                "New output name already exists; existing tensors are preserved", "output_name"
            )
        total = 0
        for record in tensors.values():
            if not isinstance(record, dict) or not isinstance(record.get("values"), list):
                raise ValidationError("Malformed tensor record")
            total += len(record["values"])
        if total > p["maximum_total_elements"]:
            raise ValidationError(
                "Original tensors exceed Total elements", "maximum_total_elements"
            )
        _, boxes = bounded_tensor(tensors, p["boxes_output"], p["maximum_rows"] * 4)
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValidationError("Boxes require [N,4] or [1,N,4] XYXY rows")
        n = len(boxes)
        vectors = []
        for field in ("scores_output", "classes_output"):
            _, values = bounded_tensor(tensors, p[field], p["maximum_rows"])
            if values.size != n or sum(s != 1 for s in values.shape) > 1:
                raise ValidationError(
                    "Scores/classes must have one value per box; batch one only", field
                )
            vectors.append(values.reshape(-1))
        scores, classes = vectors
        probability(scores, "probability")
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValidationError("XYXY boxes require positive width and height")
        if (
            np.any(classes < 0)
            or np.any(classes > 1_000_000)
            or np.any(classes != np.floor(classes))
        ):
            raise ValidationError("Class IDs must be integers in 0..1000000", "classes_output")
        if total + n * 6 > p["maximum_total_elements"]:
            raise ValidationError("Packed output exceeds Total elements", "maximum_total_elements")
        output = copy.deepcopy(tensors)
        output[p["output_name"]] = tensor_record(
            np.column_stack((boxes, scores, classes))[None], max(1, n * 6), allow_empty=True
        )
        context = self.input_payloads.get("tensors", {})
        return Result(
            {
                "tensors": output,
                "count": n,
                "report": {
                    "model": "packed_detection_heads_v1",
                    "rows": n,
                    "format": "xyxy_score_class",
                    "output_name": p["output_name"],
                    "original_tensor_names": list(tensors),
                },
            },
            payloads={name: copy.deepcopy(context) for name in ("tensors", "count", "report")},
            message=f"Packed {n} XYXY/score/class rows; original heads retained",
        )


INSTANCE_BLOCKS = (DecodeInstances, SelectInstanceMask, PackDetectionHeads)
