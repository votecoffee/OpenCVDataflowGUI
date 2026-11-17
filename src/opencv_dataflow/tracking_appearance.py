"""Bounded image-box appearance descriptors and explicit feature-space contracts."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import image_value
from .recipes import unique_ref
from .selection import SELECTION_INPUTS, selection_mask
from .tickets import ticket_key


def unit_appearance(value, maximum):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"space", "vector"}
        or not isinstance(value["space"], str)
        or not value["space"]
        or len(value["space"]) > 256
        or not isinstance(value["vector"], list)
        or not 1 <= len(value["vector"]) <= maximum
        or any(type(v) not in (int, float) for v in value["vector"])
    ):
        raise ValidationError("Appearance needs a named feature space and a bounded numeric vector")
    vector = np.asarray(value["vector"], float)
    magnitude = np.abs(vector).max()
    if not np.isfinite(vector).all() or magnitude == 0:
        raise ValidationError("Appearance vectors must be finite and nonzero")
    vector = vector / magnitude
    vector /= np.linalg.norm(vector)
    return {"space": value["space"], "vector": vector}


def appearance_distance(first, second):
    if first is None or second is None:
        return None
    if first["space"] != second["space"] or first["vector"].shape != second["vector"].shape:
        raise ValidationError("Appearance feature space/dimension changed; reset or reacquire")
    return float(np.clip(1 - np.dot(first["vector"], second["vector"]), 0, 2))


def updated_appearance(previous, current, rate):
    if current is None:
        return copy.deepcopy(previous)
    if previous is None:
        return copy.deepcopy(current)
    appearance_distance(previous, current)
    vector = (1 - rate) * previous["vector"] + rate * current["vector"]
    norm = np.linalg.norm(vector)
    return {
        "space": current["space"],
        "vector": vector / norm if norm > 1e-12 else current["vector"].copy(),
    }


class DescribeBoxAppearance(Block):
    type_id, title = "ml.box_appearance", "Describe box appearance"
    category = block_type = "ML"
    subtype = "Object tracking"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (*SELECTION_INPUTS, Port("detections", "data", visual_type="collection"))
    outputs = (
        Port("image"),
        Port("collection", "data", visual_type="collection"),
        Port("report", "data"),
    )
    defaults = {
        "color_model": "hsv_hs",
        "hue_bins": 16,
        "saturation_bins": 8,
        "gray_bins": 32,
        "roi_inset": 0.1,
        "minimum_pixels": 16,
        "insufficient": "omit",
        "maximum_boxes": 128,
        "maximum_sample_pixels": 2_000_000,
    }
    parameter_choices = {"color_model": ("hsv_hs", "gray"), "insufficient": ("omit", "error")}
    parameter_ranges = {
        "hue_bins": NumericRange(2, 64, whole=True),
        "saturation_bins": NumericRange(2, 32, whole=True),
        "gray_bins": NumericRange(2, 256, whole=True),
        "roi_inset": NumericRange(0, 0.49),
        "minimum_pixels": NumericRange(1, 1_000_000, whole=True),
        "maximum_boxes": NumericRange(1, 128, whole=True),
        "maximum_sample_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        "color_model": (
            "HSV hue/saturation joint histogram or grayscale intensity; uint8 input only."
        ),
        "hue_bins": "HSV hue bins over [0,180); used only by hsv_hs.",
        "saturation_bins": "HSV saturation bins over [0,256); product with hue bins at most 512.",
        "gray_bins": "Intensity bins over [0,256); used only by gray.",
        "roi_inset": "Fraction removed from each box edge before raster sampling; no crop output.",
        "minimum_pixels": "Minimum selected nontransparent pixels for a descriptor.",
        "insufficient": "Omit appearance (retain box), or reject when ROI support is insufficient.",
        "maximum_boxes": "Reject excess boxes before histogram work; never truncate.",
        "maximum_sample_pixels": (
            "Reject excess summed raster ROI pixels, including overlapping boxes."
        ),
    }
    input_help = {
        "image": "Current uint8 gray/BGR/RGB/alpha image. Primary pixels remain unchanged.",
        "detections": (
            "Current-grid boxes, blobs or contours. Uses axis-aligned bounds; retains records."
        ),
        "mask": "Intersect ROI support with this current-grid mask and region.",
        "region": "Optional selected support within each box, rather than a center-only gate.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, bounds in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (bounds.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric histogram settings", key)
        if p["color_model"] == "hsv_hs" and p["hue_bins"] * p["saturation_bins"] > 512:
            raise ValidationError(
                "Joint histogram is limited to 512 bins", "hue_bins", "saturation_bins"
            )

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"], (np.uint8,), canonical=True)
        h, w = image.shape[:2]
        if p["color_model"] == "hsv_hs" and image.ndim != 3:
            raise ValidationError("HSV appearance requires a color image; select gray explicitly")
        if max(h, w) > 16384 or h * w > 16_777_216:
            raise ValidationError("Appearance images are limited to 16M pixels and 16384 per axis")
        data = inputs["detections"]
        validate_collection(data)
        if (
            data["kind"] not in ("boxes", "blobs", "contours")
            or data["image_size"] != [w, h]
            or data.get("origin", [0, 0]) != [0, 0]
        ):
            raise ValidationError("Appearance needs zero-origin boxes on the current image grid")
        if len(data["items"]) > p["maximum_boxes"]:
            raise ValidationError("Appearance box count exceeds limit", "maximum_boxes")
        context = self.input_payloads.get("image", {})
        contexts = [context, self.input_payloads.get("detections", {}), data.get("context", {})]
        require_same_frame([ValuePacket(None, value) for value in contexts])
        recipe = unique_ref([ValuePacket(None, value) for value in contexts])
        tickets = [ticket_key(value["ticket"]) for value in contexts if "ticket" in value]
        if tickets and any(t != tickets[0] for t in tickets[1:]):
            raise ValidationError("Appearance boxes belong to another ticket")
        rois, work = [], 0
        for item in data["items"]:
            raw = item["geometry"].get("bbox")
            if raw is None and data["kind"] in ("blobs", "contours"):
                raw = [item["metrics"].get(key) for key in ("x", "y", "width", "height")]
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 4
                or any(type(v) not in (int, float) for v in raw)
            ):
                raise ValidationError("Every appearance box needs numeric XYWH geometry")
            bounds = np.asarray(raw, float)
            if (
                not np.isfinite(bounds).all()
                or np.any(np.abs(bounds) > 1e6)
                or np.any(bounds[2:] <= 0)
            ):
                raise ValidationError("Appearance boxes need bounded positive rectangles")
            low = np.floor(bounds[:2] + bounds[2:] * p["roi_inset"])
            high = np.ceil(bounds[:2] + bounds[2:] * (1 - p["roi_inset"]))
            x, y = np.clip(low, [0, 0], [w, h]).astype(int)
            right, bottom = np.clip(high, [0, 0], [w, h]).astype(int)
            work += max(0, right - x) * max(0, bottom - y)
            rois.append((x, y, right, bottom))
        if work > p["maximum_sample_pixels"]:
            raise ValidationError(
                "Summed appearance ROI pixels exceed limit", "maximum_sample_pixels"
            )
        mask = selection_mask(inputs, image)
        definition = {
            "version": 1,
            "method": p["color_model"],
            "inset": float(p["roi_inset"]),
            "alpha": "positive",
        }
        definition["bins"] = (
            [p["hue_bins"], p["saturation_bins"]]
            if p["color_model"] == "hsv_hs"
            else [p["gray_bins"]]
        )
        space = (
            "opencv_box_histogram:"
            + hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        )
        output, rows, drawings = copy.deepcopy(data), [], []
        for item, (x, y, right, bottom) in zip(output["items"], rois, strict=True):
            item.pop("appearance", None)
            roi = image[y:bottom, x:right]
            selected = (
                np.full(roi.shape[:2], 255, np.uint8)
                if mask is None
                else mask[y:bottom, x:right].copy()
            )
            if image.ndim == 3 and image.shape[2] == 4:
                selected[roi[..., 3] == 0] = 0
                roi = roi[..., :3]
            count = int(np.count_nonzero(selected))
            rows.append(
                {
                    "id": item["id"],
                    "selected_pixels": count,
                    "available": count >= p["minimum_pixels"],
                }
            )
            if self.generate_graphics and right > x and bottom > y:
                drawings.append(
                    graphic_object(
                        "polyline",
                        [[x, y], [right, y], [right, bottom], [x, bottom]],
                        (60, 210, 100) if count >= p["minimum_pixels"] else (40, 190, 240),
                        closed=True,
                    )
                )
            if count < p["minimum_pixels"]:
                if p["insufficient"] == "error":
                    raise ValidationError(
                        "A box has insufficient selected pixels", "minimum_pixels"
                    )
                continue
            if p["color_model"] == "hsv_hs":
                if roi.ndim != 3:
                    raise ValidationError(
                        "HSV appearance requires a color image; select gray explicitly"
                    )
                native = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                vector = cv2.calcHist(
                    [native], [0, 1], selected, definition["bins"], [0, 180, 0, 256]
                )
            else:
                native = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                vector = cv2.calcHist([native], [0], selected, definition["bins"], [0, 256])
            vector = vector.reshape(-1).astype(float)
            vector /= np.linalg.norm(vector)
            item["appearance"] = {"space": space, "vector": vector.tolist()}
        context = copy.deepcopy(context)
        if recipe is not None:
            context["recipe_ref"] = copy.deepcopy(recipe)
        return Result(
            {
                "image": inputs["image"],
                "collection": output,
                "report": {
                    "model": "box_appearance_v1",
                    "space": space,
                    "definition": definition,
                    "sampled_roi_pixels": int(work),
                    "boxes": rows,
                    "available": sum(row["available"] for row in rows),
                    "semantics": (
                        "color/intensity histogram, not learned re-identification or a probability"
                    ),
                },
            },
            payloads={name: copy.deepcopy(context) for name in ("image", "collection", "report")},
            drawings={"Appearance sampling ROIs": GraphicDrawing(tuple(drawings))}
            if drawings
            else {},
            message=f"Appearance for {sum(row['available'] for row in rows)}/{len(rows)} boxes",
        )
