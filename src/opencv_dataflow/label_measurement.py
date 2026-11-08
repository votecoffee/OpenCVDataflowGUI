"""Measure raster label IDs and reconstruct masks from filtered collections."""

from hashlib import sha256

import numpy as np

from .collection_tools import collection_result, read_collection
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import gray_image
from .region_proposals import box_item
from .result_graphics import ACCEPTED
from .selection import selection_mask
from .tickets import ticket_key

_MAX_PIXELS = 2_100_000
_MAX_LABELS = 10_000
_LABEL_DTYPES = (np.uint8, np.uint16, np.int16, np.int32)
_SOURCE_MODEL = "measured_label_source_v1"
_LABEL_INPUT = Port("labels")
_SELECTION_INPUTS = (
    Port("mask", required=False, required_when_connected=True),
    Port("region", "data", required=False, required_when_connected=True),
)


def check_label_plane(labels):
    if labels.ndim != 2 or labels.dtype not in _LABEL_DTYPES:
        raise ValidationError("Use a single-channel uint8, uint16, int16 or int32 label plane")
    if not 0 < labels.size <= _MAX_PIXELS or max(labels.shape) > 16384:
        raise ValidationError("Label plane must fit the 2.1 million pixel and 16,384 axis limits")


def raster_digest(image):
    # Dimensions are checked separately against the collection's image_size.
    digest = sha256(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image))
    return digest.hexdigest()


def scope_digest(mask):
    return None if mask is None or mask.all() else raster_digest(mask)


def measure_regions(labels, selected, intensity):
    height, width = labels.shape
    indices = np.flatnonzero(selected)
    ids, inverse, areas = np.unique(
        labels.ravel()[indices], return_inverse=True, return_counts=True
    )
    if len(ids) > _MAX_LABELS:
        raise ValidationError("Selection contains more than 10,000 label IDs")

    y, x = np.divmod(indices, width)
    minimum_x = np.full(len(ids), width, dtype=np.int64)
    minimum_y = np.full(len(ids), height, dtype=np.int64)
    maximum_x = np.zeros(len(ids), dtype=np.int64)
    maximum_y = np.zeros(len(ids), dtype=np.int64)
    np.minimum.at(minimum_x, inverse, x)
    np.minimum.at(minimum_y, inverse, y)
    np.maximum.at(maximum_x, inverse, x)
    np.maximum.at(maximum_y, inverse, y)
    center_x = np.bincount(inverse, weights=x) / areas
    center_y = np.bincount(inverse, weights=y) / areas
    means = (
        np.bincount(inverse, weights=intensity.ravel()[indices]) / areas
        if intensity is not None else [None] * len(ids)
    )

    items = []
    for index, label in enumerate(ids):
        item = box_item(
            int(label), int(minimum_x[index]), int(minimum_y[index]),
            int(maximum_x[index]) + 1, int(maximum_y[index]) + 1,
        )
        metrics = item["metrics"]
        metrics.update(
            area=int(areas[index]),
            center_x=float(center_x[index]),
            center_y=float(center_y[index]),
            extent=int(areas[index]) / (metrics["width"] * metrics["height"]),
            intensity=float(means[index]) if means[index] is not None else None,
        )
        items.append(item)
    return items


class MeasureLabelRegions(Block):
    type_id, title = "inspect.measure_label_regions", "Measure label regions"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (
        _LABEL_INPUT,
        Port("intensity_image", required=False, required_when_connected=True),
        *_SELECTION_INPUTS,
    )
    outputs = (Port("collection", "data"), Port("count", "integer"), Port("points", "data"))
    defaults = {"ignored_labels": [-1, 0], "preview_count": 25}
    parameter_ranges = {"preview_count": NumericRange(0, 250, whole=True)}
    parameter_help = {
        "ignored_labels": "Skip these IDs. For K-means, remove 0 to measure the first cluster.",
        "preview_count": "Maximum regions to draw. Zero disables drawing.",
    }
    input_help = {
        "labels": "Integer label plane with up to 2.1 million pixels. Each ID yields one record.",
        "intensity_image": "Optional aligned grayscale or color image for mean luminance.",
        "mask": "Measure only selected pixels. Area and centroid reflect the clipped region.",
        "region": "Optional region, intersected with a wired mask.",
    }
    payload_reference = "labels"
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        ignored = values["ignored_labels"]
        if not isinstance(ignored, list) or len(ignored) > 256 or any(
            type(label) is not int or not -(2**31) <= label < 2**31 for label in ignored
        ):
            raise ValidationError(
                "Ignored labels must be a list of up to 256 int32 IDs", "ignored_labels"
            )

    def process(self, inputs):
        labels = inputs["labels"]
        check_label_plane(labels)

        intensity = inputs.get("intensity_image")
        if intensity is not None:
            if intensity.shape[:2] != labels.shape:
                raise ValidationError("Intensity image must match the label dimensions")
            require_same_frame((labels, intensity), message="Intensity image and labels must align")
            intensity = gray_image(intensity)
            if not np.isfinite(intensity).all():
                raise ValidationError("Intensity luminance must be finite")

        mask = selection_mask(inputs, labels)
        selected = ~np.isin(np.asarray(labels), self.parameters["ignored_labels"])
        if mask is not None:
            selected &= mask != 0
        items = measure_regions(np.asarray(labels), selected, intensity)
        payload = self.input_payloads["labels"]
        data = collection("boxes", items, labels, payload)
        data["label_source"] = {
            "model": _SOURCE_MODEL,
            "labels_sha256": raster_digest(labels),
            "scope_sha256": scope_digest(mask),
        }
        result = collection_result(data, payload)
        result.message = f"{len(items)} measured label IDs"
        if self.generate_graphics:
            objects = []
            for item in items[:int(self.parameters["preview_count"])]:
                x, y, width, height = item["geometry"]["bounds"]
                metrics = item["metrics"]
                center = (metrics["center_x"], metrics["center_y"])
                objects.extend((
                    graphic_box(x, y, width, height, ACCEPTED),
                    graphic_object("marker", (center,), ACCEPTED),
                ))
            if objects:
                result.drawings["measured labels"] = GraphicDrawing(tuple(objects), source="labels")
        return result


class MaskFromLabelCollection(Block):
    type_id, title = "inspect.mask_from_labels", "Mask from filtered labels"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (_LABEL_INPUT, Port("collection", "data"), *_SELECTION_INPUTS)
    outputs = (Port("mask", visual_type="region"), Port("pixel_count", "integer"))
    input_help = {
        "labels": "The same label plane supplied to Measure label regions.",
        "collection": "Measured labels, optionally passed through collection filters or sorting.",
        "mask": "Use the same measurement mask here when one was supplied.",
        "region": "Use the same measurement region here when one was supplied.",
    }
    payload_reference = "labels"
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        labels = inputs["labels"]
        check_label_plane(labels)
        data, context = read_collection(self, inputs)
        source = data.get("label_source")
        if not isinstance(source, dict) or source.get("model") != _SOURCE_MODEL:
            raise ValidationError("Connect a collection produced by Measure label regions")
        if data["image_size"] != [labels.shape[1], labels.shape[0]]:
            raise ValidationError("Collection and label plane dimensions differ")
        require_same_frame(
            (labels, ValuePacket(labels, context)),
            message="Collection and label plane coordinate frames differ",
        )
        current_ticket = self.input_payloads["labels"].get("ticket")
        collection_ticket = context.get("ticket")
        if current_ticket is not None and collection_ticket is not None:
            if ticket_key(current_ticket) != ticket_key(collection_ticket):
                raise ValidationError("Collection and label plane must belong to the same ticket")
        if source.get("labels_sha256") != raster_digest(labels):
            raise ValidationError("Wire the original label plane used to measure the collection")
        scope = selection_mask(inputs, labels)
        if "scope_sha256" not in source or source["scope_sha256"] != scope_digest(scope):
            raise ValidationError("Wire the same mask or region used to measure the collection")

        ids = [item["id"] for item in data["items"]]
        if len(ids) > _MAX_LABELS or any(
            type(label) is not int or not -(2**31) <= label < 2**31 for label in ids
        ):
            raise ValidationError("Measured label collections must contain up to 10,000 int32 IDs")
        selected = np.isin(np.asarray(labels), ids)
        if scope is not None:
            selected &= scope != 0
        count = int(np.count_nonzero(selected))
        return Result(
            {"mask": selected.astype(np.uint8) * 255, "pixel_count": count},
            message=f"{count} pixels from {len(ids)} retained label IDs",
        )
