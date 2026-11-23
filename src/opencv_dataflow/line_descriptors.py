"""Binary Line Band Descriptors and native Hamming matches with endpoint geometry."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .payloads import check_spatial
from .processing import ImageTool, gray_image
from .registration import paired_feature_result, ratio_matches
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, contains, selection_mask

FEATURE_MODEL = "binary_line_features_v1"
MATCH_MODEL = "binary_line_matches_v1"
DESCRIPTOR_BYTES = 32
MAX_FEATURES = 1024
MAX_PIXELS = 2_097_152
BAND_WIDTH = NumericRange(3, 31, whole=True, odd=True)


class LineDescriptors(ImageTool):
    type_id, title, subtype = "inspect.line_descriptors", "Binary line descriptors", "Features"
    category = block_type = "Inspection"
    description = "Detect line segments and compute one binary Line Band Descriptor per segment."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("features", "data"),
        Port("collection", "data"),
        Port("count", "number"),
    )
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {"band_width": 7, "min_length": 16.0, "max_features": 256, "preview_count": 80}
    parameter_ranges = {
        "band_width": BAND_WIDTH,
        "min_length": NumericRange(1, 4096),
        "max_features": NumericRange(1, MAX_FEATURES, whole=True),
        "preview_count": NumericRange(0, 256, whole=True),
    }
    parameter_help = {
        "band_width": "Pixel width of each native descriptor band. Match sets with the same width.",
        "min_length": "Minimum detected segment length in pixels at the selected level.",
        "max_features": "Describe at most this many selected segments, longest first.",
        "preview_count": "Draw at most this many described segments. Zero hides the drawings.",
    }
    input_help = {
        "image": "Gray or color source converted to 8-bit grayscale. Describes the selected "
        "application pyramid level at one native octave.",
        "mask": "Retain segments whose two endpoints lie inside the mask.",
        "region": "Retain segments whose two endpoints lie inside the region and any wired mask.",
    }
    output_help = {
        "image": "Original image or selected pyramid level.",
        "features": "One 32-byte binary descriptor per ordered pair of segment endpoints.",
        "collection": "Source-frame line geometry with length, direction and native response.",
    }

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 24 or max(height, width) > 4096 or height * width > MAX_PIXELS:
            raise ValidationError(
                "Line descriptors need 24–4096 pixels per axis and at most 2,097,152 pixels"
            )
        create = getattr(getattr(cv2, "line_descriptor", None), "BinaryDescriptor", None)
        if create is None:
            raise ValidationError("Binary line descriptors require OpenCV contrib line_descriptor")
        p = self.parameters
        selection = selection_mask(inputs, source)
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        detector = create.createBinaryDescriptor()
        band = int(p["band_width"])
        detector.setNumOfOctaves(1)
        detector.setWidthOfBand(band)
        keys = []
        if selection is None or cv2.countNonZero(selection):
            for key in detector.detect(gray):
                if key.lineLength < float(p["min_length"]):
                    continue
                points = (key.getStartPoint(), key.getEndPoint())
                if all(
                    0 <= x < width and 0 <= y < height and contains(selection, (x, y))
                    for x, y in points
                ):
                    keys.append(key)
        keys.sort(key=lambda key: -key.lineLength)
        keys = keys[: int(p["max_features"])]
        descriptors = []
        if keys:
            # Single-octave classes are independent. Dense IDs keep native allocations bounded.
            for index, key in enumerate(keys):
                key.class_id = index
            keys, rows = detector.compute(gray, keys)
            if (
                rows is None
                or rows.dtype != np.uint8
                or rows.shape != (len(keys), DESCRIPTOR_BYTES)
            ):
                raise ValidationError("OpenCV returned inconsistent binary line descriptors")
            descriptors = rows.tolist()

        items, segments = [], []
        for index, key in enumerate(keys):
            start, end = key.getStartPoint(), key.getEndPoint()
            item = line_item((*start, *end), index)
            item["metrics"]["strength"] = float(key.response)
            items.append(item)
            segments.append(item["geometry"]["points"])
        result = Result(
            {
                "image": source,
                "features": {
                    "model": FEATURE_MODEL,
                    "band_width": band,
                    "image_size": [width, height],
                    "segments": segments,
                    "descriptors": descriptors,
                },
                "collection": collection("lines", items, source, self.input_payloads["image"]),
                "count": len(items),
            },
            message=f"{len(items)} binary line descriptors",
        )
        if self.generate_graphics and segments and p["preview_count"]:
            shown = segments[: int(p["preview_count"])]
            result.drawings["Described lines"] = GraphicDrawing(
                tuple(graphic_object("polyline", points, ACCEPTED, thickness=1) for points in shown)
            )
        return result


def read_line_features(record, payload):
    if not isinstance(record, dict) or record.get("model") != FEATURE_MODEL:
        raise ValidationError("Connect a binary line descriptor feature set")
    size = record.get("image_size")
    band = record.get("band_width")
    segments, descriptors = (record.get(name) for name in ("segments", "descriptors"))
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(axis) is not int or not 1 <= axis <= 4096 for axis in size)
        or type(band) is not int
        or not isinstance(segments, list)
        or not isinstance(descriptors, list)
        or len(segments) != len(descriptors)
        or len(segments) > MAX_FEATURES
        or any(
            not isinstance(row, list)
            or len(row) != DESCRIPTOR_BYTES
            or any(type(value) is not int or not 0 <= value <= 255 for value in row)
            for row in descriptors
        )
    ):
        raise ValidationError("Invalid source size or aligned 32-byte line descriptor rows")
    BAND_WIDTH.check(band, "Descriptor band width")
    if "spatial" in payload:
        check_spatial(payload, size)
    try:
        points = np.asarray(segments, np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Line segments must contain finite endpoint pairs") from exc
    if not segments:
        points = points.reshape(0, 2, 2)
    if (
        points.shape != (len(segments), 2, 2)
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points >= size)
        or np.any(np.all(points[:, 0] == points[:, 1], axis=1))
    ):
        raise ValidationError("Line segments need two distinct endpoints inside their source image")
    return points, np.asarray(descriptors, np.uint8).reshape(-1, DESCRIPTOR_BYTES)


class MatchLineDescriptors(Block):
    type_id, title, subtype = (
        "inspect.match_line_descriptors",
        "Match binary line descriptors",
        "Matching",
    )
    category = block_type = "Inspection"
    description = "Match binary line descriptors and retain endpoints and native Hamming distances."
    inputs = (Port("query", "data"), Port("train", "data"))
    outputs = (Port("matches", "data"), Port("count", "number"))
    payload_reference = "query"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {"mode": "ratio_cross_check", "ratio": 0.82, "max_distance": 48, "min_matches": 0}
    parameter_choices = {"mode": ("ratio", "cross_check", "ratio_cross_check")}
    parameter_ranges = {
        "ratio": NumericRange(0.01, 0.99),
        "max_distance": NumericRange(0, 256, whole=True),
        "min_matches": NumericRange(0, MAX_FEATURES, whole=True),
    }
    parameter_help = {
        "mode": "Use the two-neighbor distance ratio, reciprocal nearest matches, or both.",
        "ratio": "Best distance must be strictly below this fraction of the second-best distance.",
        "max_distance": "Largest accepted Hamming distance, in differing bits.",
        "min_matches": "Fewest accepted matches needed to pass. Zero permits empty results.",
    }
    output_help = {
        "matches": "Paired midpoints, endpoints, descriptor indexes, distances and image frames.",
    }

    def process(self, inputs):
        qpoints, query = read_line_features(inputs["query"], self.input_payloads["query"])
        tpoints, train = read_line_features(inputs["train"], self.input_payloads["train"])
        if inputs["query"]["band_width"] != inputs["train"]["band_width"]:
            raise ValidationError("Query and train line descriptors need the same band width")
        factory = getattr(getattr(cv2, "line_descriptor", None), "BinaryDescriptorMatcher", None)
        if factory is None:
            raise ValidationError("Binary line matching requires OpenCV contrib line_descriptor")
        p = self.parameters
        mode, maximum = p["mode"], int(p["max_distance"])
        selected = []
        if len(query) and len(train) >= (1 if mode == "cross_check" else 2):
            matcher = factory()
            if mode == "cross_check":
                selected = [m for m in matcher.match(query, train) if m.distance <= maximum]
            else:
                pairs = matcher.knnMatch(query, train, 2)
                selected = ratio_matches(pairs, float(p["ratio"]), maximum)
            if mode != "ratio":
                reverse = {m.queryIdx: m.trainIdx for m in factory().match(train, query)}
                selected = [m for m in selected if reverse.get(m.trainIdx) == m.queryIdx]

        result = paired_feature_result(inputs, qpoints.mean(axis=1), tpoints.mean(axis=1), selected)
        result.outputs["matches"].update(
            {
                "model": MATCH_MODEL,
                "query_segments": [qpoints[m.queryIdx].tolist() for m in selected],
                "train_segments": [tpoints[m.trainIdx].tolist() for m in selected],
                **{
                    f"{name}_context": {
                        key: self.input_payloads[name][key]
                        for key in ("spatial", "pyramid", "ticket")
                        if key in self.input_payloads[name]
                    }
                    for name in ("query", "train")
                },
            }
        )
        result.passed = len(selected) >= int(p["min_matches"])
        if not result.passed:
            result.fail_reasons = ["too_few_line_matches"]
        result.message = f"{len(selected)} binary line matches"
        return result
