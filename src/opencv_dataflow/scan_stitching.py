"""Automatically register and stitch an ordered batch of overlapping views."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .image_stitching import mosaic_canvas, mosaic_frame, warp_mosaic_view
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, check_spatial, image_with_payload, metadata
from .processing import ImageTool, gray_image, image_value
from .registration import ratio_matches
from .selection import selection_mask

_MAX_VIEWS = 8
_MAX_VIEW_PIXELS = 8_388_608
_MAX_INPUT_BYTES = 128 * 1024**2
_INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}
_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)


def _unpack(group, mask_group):
    if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= _MAX_VIEWS:
        raise ValidationError("Views requires an ordered group of two to eight images")
    if mask_group is not None and (
        not isinstance(mask_group, FlowGroup) or len(mask_group.items) != len(group.items)
    ):
        raise ValidationError("Masks must contain one mask per view in the same order")

    values = [entry.value for entry in group.items]
    if any(not isinstance(value, np.ndarray) for value in values):
        raise ValidationError("Every view must be an image")
    if sum(value.nbytes for value in values) > _MAX_INPUT_BYTES or any(
        value.ndim < 2 or value.shape[0] * value.shape[1] > _MAX_VIEW_PIXELS for value in values
    ):
        raise ValidationError("Scan inputs exceed the 128 MiB or per-view pixel limit")

    contexts = [metadata(entry) for entry in group.items]
    images = [
        image_with_payload(value, context) for value, context in zip(values, contexts, strict=True)
    ]
    pixels = [image_value(image, _DEPTHS) for image in images]
    for image, context in zip(pixels, contexts, strict=True):
        if min(image.shape[:2]) < 8:
            raise ValidationError("Every scan view must be at least 8 × 8 pixels")
        check_spatial(context, list(image.shape[1::-1]))
    encoding = image_format(images[0])
    if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or any(
        image_format(image) != encoding
        or image.dtype != pixels[0].dtype
        or image.shape[2:] != pixels[0].shape[2:]
        for image in images
    ):
        raise ValidationError("Use matching grayscale or RGB/BGR views with the same depth")

    masks = [None] * len(images)
    if mask_group is not None:
        for index, (image, entry) in enumerate(zip(images, mask_group.items, strict=True)):
            if not isinstance(entry.value, np.ndarray):
                raise ValidationError("Every mask group entry must be a uint8 image")
            mask = image_with_payload(entry.value, metadata(entry))
            masks[index] = selection_mask({"image": image, "mask": mask}, image)
    return images, pixels, masks, contexts


def _features(detector, image, mask, maximum):
    gray = gray_image(image, (np.uint8,), auto_depth=True)
    keys, descriptors = detector.detectAndCompute(gray, mask)
    return keys[:maximum], descriptors[:maximum] if descriptors is not None else None


def _pairs(matcher, source, target, ratio, distance):
    source_keys, a = source
    target_keys, b = target
    if a is None or b is None or min(len(a), len(b)) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    forward = ratio_matches(matcher.knnMatch(a, b, k=2), ratio, distance)
    reverse = {
        m.queryIdx: m.trainIdx for m in ratio_matches(matcher.knnMatch(b, a, k=2), ratio, distance)
    }
    source_points, target_points = [], []
    used_source, used_target = set(), set()
    for match in sorted(forward, key=lambda item: item.distance):
        if reverse.get(match.trainIdx) != match.queryIdx:
            continue
        p, q = source_keys[match.queryIdx].pt, target_keys[match.trainIdx].pt
        # SIFT may emit several orientations at the same position.
        if p not in used_source and q not in used_target:
            source_points.append(p)
            target_points.append(q)
            used_source.add(p)
            used_target.add(q)
    return np.asarray(source_points, np.float32).reshape(-1, 2), np.asarray(
        target_points, np.float32
    ).reshape(-1, 2)


def _estimate(source, target, settings, report):
    minimum = int(settings["min_inliers"])
    if len(source) < minimum:
        report["status"] = "insufficient_matches"
        return None
    threshold = float(settings["inlier_pixels"])
    matrix = np.eye(3)
    if settings["motion_model"] == "translation":
        shift, mask = cv2.estimateTranslation2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold
        )
        matrix[:2, 2] = shift
    else:
        affine, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold
        )
        if affine is None:
            report["status"] = "no_consensus"
            return None
        matrix[:2] = affine
    if mask is None or not np.isfinite(matrix).all():
        report["status"] = "no_consensus"
        return None
    inliers = mask.ravel() != 0
    count = int(inliers.sum())
    report.update(inliers=count, inlier_ratio=count / len(source))
    if count < minimum or report["inlier_ratio"] < float(settings["min_inlier_ratio"]):
        report["status"] = "insufficient_inliers"
        return None
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    scale_limit = float(settings["max_scale_ratio"])
    if not 1 / scale_limit <= scale <= scale_limit or abs(rotation) > float(
        settings["max_rotation_degrees"]
    ):
        report["status"] = "motion_out_of_range"
        return None
    residuals = source[inliers] @ matrix[:2, :2].T + matrix[:2, 2] - target[inliers]
    report.update(
        status="registered",
        rms_pixels=float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
        source_to_previous=matrix.tolist(),
    )
    return matrix


class StitchScanSequence(ImageTool):
    type_id, title, subtype = "cv.stitch_scan_sequence", "Stitch scan sequence", "Geometry"
    process_inputs_readonly = True
    payload_reference = "views"
    inputs = (
        Port("views", "any"),
        Port("masks", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("coverage", visual_type="region"),
        Port("overlap", visual_type="region"),
        Port("layout", "data"),
    )
    defaults = {
        "motion_model": "translation",
        "max_features": 900,
        "match_ratio": 0.75,
        "max_descriptor_distance": 250.0,
        "inlier_pixels": 2.0,
        "min_inliers": 12,
        "min_inlier_ratio": 0.5,
        "max_scale_ratio": 1.15,
        "max_rotation_degrees": 15.0,
        "min_overlap_pixels": 64,
        "interpolation": "linear",
        "blend": "prefer_first",
    }
    parameter_choices = {
        "motion_model": ("translation", "similarity"),
        "interpolation": tuple(_INTERPOLATION),
        "blend": ("prefer_first", "prefer_last"),
    }
    parameter_ranges = {
        "max_features": NumericRange(12, 2500, whole=True),
        "match_ratio": NumericRange(1e-6, 1 - 1e-6),
        "max_descriptor_distance": NumericRange(0, 8192),
        "inlier_pixels": NumericRange(1e-6, 100),
        "min_inliers": NumericRange(3, 2500, whole=True),
        "min_inlier_ratio": NumericRange(0, 1),
        "max_scale_ratio": NumericRange(1, 2),
        "max_rotation_degrees": NumericRange(0, 180),
        "min_overlap_pixels": NumericRange(1, 16_777_216, whole=True),
    }
    parameter_help = {
        "motion_model": "Similarity additionally fits rotation and uniform scale.",
        "match_ratio": "Require a distinctive nearest descriptor in both matching directions.",
        "min_inliers": "Minimum distinct correspondences supporting each adjacent transform.",
        "max_scale_ratio": "Similarity scale range is the reciprocal of this ratio to the ratio.",
        "min_overlap_pixels": "Minimum number of valid overlapping pixels for each adjacent pair.",
        "blend": "Choose the earliest or latest view in batch order for covered overlap pixels.",
    }
    input_help = {
        "views": "Two to eight ordered overlapping images from Batch serial entries.",
        "masks": "Optional ordered uint8 masks for feature detection and valid mosaic pixels.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["min_inliers"]) > int(p["max_features"]):
            raise ValidationError(
                "Minimum inliers exceeds maximum features", "min_inliers", "max_features"
            )

    @staticmethod
    def _reject(layout, pair, reason):
        return Result(
            {"image": NO_VALUE, "coverage": NO_VALUE, "overlap": NO_VALUE, "layout": layout},
            passed=False,
            fail_reasons=[reason],
            message=f"Views {pair['previous_index'] + 1} and {pair['view_index'] + 1}: "
            f"{pair['status'].replace('_', ' ')}",
        )

    def process(self, inputs):
        group = inputs["views"]
        images, pixels, masks, contexts = _unpack(group, inputs.get("masks"))
        p = self.parameters
        maximum = int(p["max_features"])
        detector = cv2.SIFT_create(maximum)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        layout = {"model": "scan_mosaic_v1", "view_count": len(images), "pairs": []}
        transforms = [np.eye(3)]
        previous = _features(detector, images[0], masks[0], maximum)
        for index in range(1, len(images)):
            current = _features(detector, images[index], masks[index], maximum)
            source, target = _pairs(
                matcher,
                current,
                previous,
                float(p["match_ratio"]),
                float(p["max_descriptor_distance"]),
            )
            report = {"view_index": index, "previous_index": index - 1, "matches": len(source)}
            layout["pairs"].append(report)
            transform = _estimate(source, target, p, report)
            if transform is None:
                return self._reject(layout, report, "scan_registration_failed")
            transforms.append(transforms[-1] @ transform)
            previous = current

        size, anchor_to_mosaic = mosaic_canvas(pixels, transforms)
        width, height = size
        mosaic = np.zeros((height, width, *pixels[0].shape[2:]), dtype=pixels[0].dtype)
        counts = np.zeros((height, width), dtype=np.uint8)
        layout["views"] = []
        interpolation = _INTERPOLATION[p["interpolation"]]
        previous_valid = None
        for index, (image, mask, to_anchor, context, label) in enumerate(
            zip(pixels, masks, transforms, contexts, group.labels, strict=True)
        ):
            to_mosaic = anchor_to_mosaic @ to_anchor
            support = (
                np.ones(image.shape[:2], dtype=np.float32)
                if mask is None
                else (mask != 0).astype(np.float32)
            )
            valid = warp_mosaic_view(support, to_mosaic, size, interpolation) == 1
            if previous_valid is not None:
                report = layout["pairs"][index - 1]
                report["overlap_pixels"] = int(np.count_nonzero(previous_valid & valid))
                if report["overlap_pixels"] < int(p["min_overlap_pixels"]):
                    report["status"] = "insufficient_overlap"
                    return self._reject(layout, report, "scan_insufficient_overlap")
            use = valid if p["blend"] == "prefer_last" else valid & (counts == 0)
            warped = warp_mosaic_view(image, to_mosaic, size, interpolation)
            np.copyto(mosaic, warped, where=use if mosaic.ndim == 2 else use[..., None])
            counts += valid
            previous_valid = valid
            layout["views"].append(
                {
                    "index": index,
                    "label": label,
                    "source_to_mosaic": to_mosaic.tolist(),
                    "source_spatial": context.get("spatial"),
                    "source_ticket": context.get("ticket"),
                }
            )
        if mosaic.dtype.kind == "f" and not np.isfinite(mosaic).all():
            raise ValidationError("Stitching produced nonfinite samples")
        layout.update(
            image_size=list(size),
            covered_pixels=int(np.count_nonzero(counts)),
            overlap_pixels=int(np.count_nonzero(counts >= 2)),
        )
        payload = {
            **self.input_payloads["views"],
            "spatial": mosaic_frame(
                contexts[0].get("spatial", {}), anchor_to_mosaic, size, self.id
            ),
        }
        payload.pop("pyramid", None)
        return Result(
            {
                "image": ImageArray(mosaic, image_format(images[0])),
                "coverage": (counts != 0).astype(np.uint8) * 255,
                "overlap": (counts >= 2).astype(np.uint8) * 255,
                "layout": layout,
            },
            payloads=dict.fromkeys(("image", "coverage", "overlap", "layout"), payload),
            message=f"Stitched {len(images)} views into a {width} × {height} mosaic",
        )
