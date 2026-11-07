"""Automatically register and stitch an ordered batch of overlapping views."""

import math

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .core import Port, Result
from .image_stitching import mosaic_canvas, mosaic_frame, warp_mosaic_view
from .images import ImageArray, image_format
from .mosaic_photometry import (
    IDENTITY_EXPOSURE,
    MAX_EXPOSURE_SAMPLES,
    FeatherMosaic,
    corrected_view,
    fit_overlap_exposure,
)
from .mosaic_seams import SEAM_CHOICES, SEAM_DEFAULTS, SEAM_HELP, SEAM_RANGES, graph_cut_selection
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


def _reprojection_rms(matrix, source, target):
    residuals = source @ matrix[:2, :2].T + matrix[:2, 2] - target
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def _register(matcher, current, reference, settings, report):
    source, target = _pairs(
        matcher,
        current,
        reference,
        float(settings["match_ratio"]),
        float(settings["max_descriptor_distance"]),
    )
    report["matches"] = len(source)
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
    points = source[inliers], target[inliers]
    report.update(
        status="registered",
        rms_pixels=_reprojection_rms(matrix, *points),
    )
    return matrix, points


def _max_corner_shift(image, before, after):
    h, w = image.shape[:2]
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
    return float(np.linalg.norm(corners @ (after - before)[:2].T, axis=1).max())


def _distribute_loop_drift(pixels, transforms, closing, adjacent, settings, layout):
    report = layout["closure"]
    report["source_to_anchor"] = closing.tolist()
    report["uncorrected_drift_pixels"] = _max_corner_shift(pixels[-1], transforms[-1], closing)
    correction = closing @ np.linalg.inv(transforms[-1])
    # Blending two similarities preserves their form. Reject a large rotation
    # disagreement before interpolating, which could otherwise collapse a view.
    if correction[0, 0] <= 0:
        report["status"] = "rotation_disagreement"
        return None
    last = len(transforms) - 1
    adjusted = [transforms[0]]
    for i, matrix in enumerate(transforms[1:-1], start=1):
        adjusted.append(matrix + (i / last) * (correction @ matrix - matrix))
    adjusted.append(closing)
    report["max_adjustment_pixels"] = max(
        _max_corner_shift(image, before, after)
        for image, before, after in zip(pixels, transforms, adjusted, strict=True)
    )
    if report["max_adjustment_pixels"] > float(settings["max_loop_drift_pixels"]):
        report["status"] = "drift_out_of_range"
        return None
    for index, (pair, points) in enumerate(zip(layout["pairs"], adjacent, strict=True)):
        relative = np.linalg.solve(adjusted[index], adjusted[index + 1])
        pair["adjusted_rms_pixels"] = _reprojection_rms(relative, *points)
    if max(pair["adjusted_rms_pixels"] for pair in layout["pairs"]) > float(
        settings["inlier_pixels"]
    ):
        report["status"] = "adjusted_fit_out_of_range"
        return None
    return adjusted


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
        "loop_closure": False,
        "max_loop_drift_pixels": 8.0,
        "interpolation": "linear",
        "blend": "prefer_first",
        "feather_width": 32,
        **SEAM_DEFAULTS,
        "exposure": "none",
        "min_exposure_samples": 64,
        "max_exposure_gain": 2.0,
        "max_exposure_offset_fraction": 0.25,
    }
    parameter_choices = {
        "motion_model": ("translation", "similarity"),
        "interpolation": tuple(_INTERPOLATION),
        "blend": ("prefer_first", "prefer_last", "feather", "graph_cut"),
        **SEAM_CHOICES,
        "exposure": ("none", "gain", "offset"),
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
        "max_loop_drift_pixels": NumericRange(0, 512),
        "feather_width": NumericRange(1, 512, whole=True),
        **SEAM_RANGES,
        "min_exposure_samples": NumericRange(1, MAX_EXPOSURE_SAMPLES, whole=True),
        "max_exposure_gain": NumericRange(1, 8),
        "max_exposure_offset_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "motion_model": "Similarity additionally fits rotation and uniform scale.",
        "match_ratio": "Require a distinctive nearest descriptor in both matching directions.",
        "inlier_pixels": "Fit threshold and maximum adjacent RMS error after loop adjustment.",
        "min_inliers": "Minimum distinct correspondences supporting each adjacent transform.",
        "max_scale_ratio": "Similarity scale range is the reciprocal of this ratio to the ratio.",
        "min_overlap_pixels": "Minimum valid overlap for each adjacent pair and the closing pair.",
        "loop_closure": "Require an independent match from the last view back to the first.",
        "max_loop_drift_pixels": "Maximum allowed corner adjustment in first-view pixels.",
        "blend": "Choose priority, feather valid edges, or select a seam from image content.",
        "feather_width": "Distance from a valid edge, in mosaic pixels, to reach full weight.",
        **SEAM_HELP,
        "exposure": "Match the first view using a median gain or offset from valid overlaps.",
        "min_exposure_samples": "Minimum usable samples per adjacent overlap for exposure fitting.",
        "max_exposure_gain": "Limit cumulative gain to this ratio and its reciprocal.",
        "max_exposure_offset_fraction": "Limit cumulative offset as a fraction of full scale.",
    }
    input_help = {
        "views": "Two to eight ordered overlapping images from Batch serial entries.",
        "masks": "Optional ordered uint8 masks for feature detection and valid mosaic pixels.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["loop_closure"], bool):
            raise ValidationError("Loop closure must be a boolean", "loop_closure")
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
        close_loop = p["loop_closure"]
        if close_loop and len(images) < 3:
            raise ValidationError("Loop closure requires at least three views", "loop_closure")
        maximum = int(p["max_features"])
        detector = cv2.SIFT_create(maximum)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        layout = {"model": "scan_mosaic_v1", "view_count": len(images), "pairs": []}
        transforms = [np.eye(3)]
        adjacent = []
        previous = _features(detector, images[0], masks[0], maximum)
        first = previous if close_loop else None
        for index in range(1, len(images)):
            current = _features(detector, images[index], masks[index], maximum)
            report = {"view_index": index, "previous_index": index - 1}
            layout["pairs"].append(report)
            fit = _register(matcher, current, previous, p, report)
            if fit is None:
                return self._reject(layout, report, "scan_registration_failed")
            transform, points = fit
            report["source_to_previous"] = transform.tolist()
            transforms.append(transforms[-1] @ transform)
            if close_loop:
                adjacent.append(points)
            previous = current
        closure = None
        if close_loop:
            closure = {"view_index": len(images) - 1, "previous_index": 0}
            layout["closure"] = closure
            fit = _register(matcher, previous, first, p, closure)
            if fit is None:
                return self._reject(layout, closure, "scan_loop_closure_failed")
            closing, _ = fit
            transforms = _distribute_loop_drift(pixels, transforms, closing, adjacent, p, layout)
            if transforms is None:
                return self._reject(layout, closure, "scan_loop_closure_failed")

        dtype = pixels[0].dtype
        channels = 1 if pixels[0].ndim == 2 else pixels[0].shape[2]
        fit_exposure = p["exposure"] != "none"
        feather = p["blend"] == "feather"
        graph_cut = p["blend"] == "graph_cut"
        extra_bytes = 2 if close_loop else 0
        # Include retained warped views, sample indices, and blend/cast buffers.
        if fit_exposure:
            extra_bytes += 9 + channels * (dtype.itemsize + 16)
        if feather:
            extra_bytes += 20 + 24 * channels
        if graph_cut:
            extra_bytes += 4
            layout["seams"] = []
        size, anchor_to_mosaic = mosaic_canvas(
            pixels, transforms, extra_bytes_per_pixel=extra_bytes
        )
        width, height = size
        shape = (height, width, *pixels[0].shape[2:])
        blender = FeatherMosaic(shape, int(p["feather_width"])) if feather else None
        mosaic = np.zeros(shape, dtype=dtype) if blender is None else None
        counts = np.zeros((height, width), dtype=np.uint8)
        layout["views"] = []
        interpolation = _INTERPOLATION[p["interpolation"]]
        first_valid = None
        previous_valid = None
        previous_warped = None
        correction = IDENTITY_EXPOSURE
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
                overlap = previous_valid & valid
                report["overlap_pixels"] = int(np.count_nonzero(overlap))
                if report["overlap_pixels"] < int(p["min_overlap_pixels"]):
                    report["status"] = "insufficient_overlap"
                    return self._reject(layout, report, "scan_insufficient_overlap")
            if closure is not None:
                if index == 0:
                    first_valid = valid
                elif index == closure["view_index"]:
                    closure["overlap_pixels"] = int(np.count_nonzero(first_valid & valid))
                    if closure["overlap_pixels"] < int(p["min_overlap_pixels"]):
                        closure["status"] = "insufficient_overlap"
                        return self._reject(layout, closure, "scan_loop_closure_failed")
                    closure["status"] = "closed"
            warped = warp_mosaic_view(image, to_mosaic, size, interpolation)
            if previous_warped is not None:
                correction, exposure_report = fit_overlap_exposure(
                    previous_warped, warped, overlap, correction, p
                )
                report["exposure"] = exposure_report
                if correction is None:
                    report["status"] = exposure_report["status"]
                    return self._reject(layout, report, "scan_exposure_failed")
            if fit_exposure:
                previous_warped = warped
                warped = corrected_view(warped, valid, correction)
            if blender is not None:
                blender.add(warped, valid)
            else:
                warped = cast_intensity(warped, dtype) if warped.dtype != dtype else warped
                if graph_cut and index:
                    use, seam = graph_cut_selection(mosaic, warped, counts != 0, valid, p)
                    layout["seams"].append({"view_index": index, **seam})
                else:
                    use = valid if p["blend"] == "prefer_last" else valid & (counts == 0)
                np.copyto(mosaic, warped, where=use if channels == 1 else use[..., None])
            counts += valid
            previous_valid = valid
            view = {
                "index": index,
                "label": label,
                "source_to_mosaic": to_mosaic.tolist(),
                "source_spatial": context.get("spatial"),
                "source_ticket": context.get("ticket"),
            }
            if fit_exposure:
                gain, offset = correction
                view["exposure"] = {"gain": gain, "offset": offset}
            layout["views"].append(view)
        if blender is not None:
            mosaic = blender.finish(dtype)
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
