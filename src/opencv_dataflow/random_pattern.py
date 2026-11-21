"""Generate planar random textures and identify metric calibration correspondences."""

import hashlib

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .registration import ratio_matches
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, selection_mask

_TARGET = "random_pattern_target_v1"
_MAX_PIXELS = 4_194_304
_MAX_AXIS = 4096
_REF_FEATURES = 1800
_UNITS = ("mm", "m")


def _raster_size(p):
    width = int(p["raster_width"])
    height = round(width * float(p["height"]) / float(p["width"]))
    if not 256 <= height <= _MAX_AXIS or width * height > _MAX_PIXELS:
        raise ValidationError(
            "Use a raster with 256 to 4096 pixels per axis and at most 4,194,304 pixels",
            "raster_width",
            "width",
            "height",
        )
    return width, height


def _texture(width, height, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    # Introduce texture at several scales so both distant views and closeups have landmarks.
    base = np.full((height, width), 160, np.float32)
    for grid, weight in ((12, 0.28), (28, 0.24), (64, 0.16)):
        shape = (max(2, round(grid * height / width)), grid)
        cells = rng.integers(-127, 128, shape, dtype=np.int16).astype(np.float32)
        base += weight * cv2.resize(cells, (width, height), interpolation=cv2.INTER_LINEAR)
    image = np.clip(base, 0, 255).astype(np.uint8)
    scale = min(width, height)
    for _ in range(round(420 * max(width, height) / scale)):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        radius = max(2, round(scale * rng.uniform(0.004, 0.028)))
        ink = int(rng.choice((12, 48, 98, 208, 245)))
        if rng.integers(2):
            cv2.circle(image, (x, y), radius, ink, -1, cv2.LINE_AA)
        else:
            triangle = np.column_stack(
                (
                    x + rng.integers(-radius, radius + 1, 3),
                    y + rng.integers(-radius, radius + 1, 3),
                )
            ).astype(np.int32)
            cv2.fillConvexPoly(image, triangle, ink, cv2.LINE_AA)
    return image


def _sha256(gray):
    return hashlib.sha256(np.ascontiguousarray(gray)).hexdigest()


class GenerateRandomPatternTarget(Block):
    type_id, title = "calibration.generate_random_pattern", "Generate random-pattern target"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    outputs = (Port("image"), Port("target", "data"))
    defaults = {"width": 240.0, "height": 160.0, "units": "mm", "raster_width": 900, "seed": 13}
    parameter_choices = {"units": _UNITS}
    parameter_ranges = {
        "width": NumericRange(1e-6, 10_000),
        "height": NumericRange(1e-6, 10_000),
        "raster_width": NumericRange(256, _MAX_AXIS, whole=True),
        "seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "width": "Physical width of the textured rectangle, measured edge to edge.",
        "height": "Physical height of the textured rectangle, measured edge to edge.",
        "units": "Units of the physical dimensions and detected XYZ coordinates.",
        "raster_width": "Width in pixels. Raster height follows the physical aspect ratio.",
        "seed": "Keep the seed, size and a lossless copy of the reference for repeat use.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        _raster_size(self.parameters if parameters is None else parameters)

    def process(self, inputs):
        p = self.parameters
        width, height = _raster_size(p)
        pixels = _texture(width, height, int(p["seed"]))
        size = [float(p["width"]), float(p["height"])]
        units = p["units"]
        inches_per_unit = (1_000 if units == "m" else 1) / 25.4
        return Result(
            {
                "image": ImageArray(pixels, "gray"),
                "target": {
                    "model": _TARGET,
                    "image_size": [width, height],
                    "target_size": size,
                    "units": units,
                    "seed": int(p["seed"]),
                    "reference_sha256": _sha256(pixels),
                    "print_dpi_xy": (
                        np.array([width, height]) / (np.asarray(size) * inches_per_unit)
                    ).tolist(),
                },
            },
            message=f"Target {size[0]:g} × {size[1]:g} {units} · {width} × {height} px",
        )


def _gray(image):
    height, width = image.shape[:2]
    if min(width, height) < 16 or max(width, height) > _MAX_AXIS or width * height > _MAX_PIXELS:
        raise ValidationError(
            "Match images with 16 to 4096 pixels per axis and at most 4,194,304 pixels"
        )
    return np.ascontiguousarray(gray_image(image, (np.uint8,), auto_depth=True))


def _target_dimensions(target, reference):
    if not isinstance(target, dict) or target.get("model") != _TARGET:
        raise ValidationError("Connect a generated random-pattern target record")
    units, size = target.get("units"), target.get("target_size")
    if (
        units not in _UNITS
        or not isinstance(size, list)
        or len(size) != 2
        or any(type(value) not in (int, float) or not 1e-6 <= value <= 10_000 for value in size)
    ):
        raise ValidationError("Target dimensions must be positive and bounded in mm or m")
    if target.get("image_size") != list(reference.shape[::-1]) or target.get(
        "reference_sha256"
    ) != _sha256(reference):
        raise ValidationError("Reference raster does not match the random-pattern target record")
    return np.asarray(size, dtype=np.float64), units


def _pixel_ids(points, shape):
    height, width = shape
    x, y = np.clip(np.floor(points + 0.5).astype(int), [0, 0], [width - 1, height - 1]).T
    return y * width + x


def _matches(reference, gray, mask, limit, ratio):
    ref_keys, ref_desc = cv2.SIFT_create(
        _REF_FEATURES, enable_precise_upscale=True
    ).detectAndCompute(reference, None)
    view_keys, view_desc = cv2.SIFT_create(limit, enable_precise_upscale=True).detectAndCompute(
        gray, mask
    )
    if len(ref_keys) < 2 or len(view_keys) < 2:
        return np.empty(0, dtype=int), np.empty((0, 2)), np.empty((0, 2))

    ref_xy = np.asarray([key.pt for key in ref_keys], dtype=np.float64)
    view_xy = np.asarray([key.pt for key in view_keys], dtype=np.float64)
    ref_ids = _pixel_ids(ref_xy, reference.shape)
    view_ids = _pixel_ids(view_xy, gray.shape)
    # Anchor an ID to a single subpixel reference center even if multiple
    # keypoints rounded to that pixel match in different views.
    center_order = np.lexsort((ref_xy[:, 1], ref_xy[:, 0], ref_ids))
    _, first, inverse = np.unique(ref_ids[center_order], return_index=True, return_inverse=True)
    centers = np.empty_like(ref_xy)
    centers[center_order] = ref_xy[center_order[first[inverse]]]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = ratio_matches(matcher.knnMatch(view_desc, ref_desc, k=2), ratio, float("inf"))
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(ref_desc, view_desc)}
    # A SIFT landmark can have several orientations. Count one pair per location in each image.
    chosen, used_ref, used_view = [], set(), set()
    for match in sorted(forward, key=lambda m: (m.distance, m.trainIdx, m.queryIdx)):
        i, j = match.trainIdx, match.queryIdx
        if reverse[i] == j and ref_ids[i] not in used_ref and view_ids[j] not in used_view:
            used_ref.add(ref_ids[i])
            used_view.add(view_ids[j])
            chosen.append((i, j))
    i, j = np.asarray(sorted(chosen, key=lambda pair: ref_ids[pair[0]]), dtype=int).reshape(-1, 2).T
    return ref_ids[i], centers[i], view_xy[j]


def _project(matrix, points):
    uvw = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return uvw[:, :2] / uvw[:, 2:3]


def _fit(reference, observed, size, p):
    rejected = np.zeros(len(reference), bool)
    if len(reference) < 4:
        return None, rejected, None, None
    limit = float(p["ransac_threshold_px"])
    matrix, mask = cv2.findHomography(
        reference,
        observed,
        cv2.RANSAC,
        limit,
        maxIters=int(p["max_iterations"]),
        confidence=float(p["confidence"]),
    )
    if (
        matrix is None
        or mask is None
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
    ):
        return None, rejected, None, None
    width, height = size
    outline = np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]]
    )
    w = outline @ matrix[2, :2] + matrix[2, 2]
    if not (np.all(w > 1e-9) or np.all(w < -1e-9)):
        return None, rejected, None, None
    errors = np.linalg.norm(_project(matrix, reference) - observed, axis=1)
    inliers = mask.ravel().astype(bool) & (errors <= limit)
    return matrix, inliers, errors[inliers], _project(matrix, outline)


def _coverage(points, size):
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (size[0] * size[1]))


def _draw(observed, inliers, outline, passed, shape):
    drawings = {}
    if outline is not None and np.max(np.abs(outline)) < 16 * max(shape):
        drawings["Target bounds"] = GraphicDrawing(
            (graphic_object("polyline", outline, ACCEPTED if passed else REJECTED, closed=True),)
        )
    for key, indices, color in (
        ("Accepted points", np.flatnonzero(inliers), ACCEPTED if passed else REJECTED),
        ("Rejected matches", np.flatnonzero(~inliers), REJECTED),
    ):
        if len(indices):
            stride = max(1, (len(indices) + 95) // 96)
            drawings[key] = GraphicDrawing(
                tuple(graphic_object("marker", [observed[i]], color) for i in indices[::stride])
            )
    return drawings


class DetectRandomPatternTarget(Block):
    type_id, title = "calibration.detect_random_pattern", "Detect random-pattern target"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    inputs = (*SELECTION_INPUTS, Port("reference"), Port("target", "data"))
    outputs = (
        Port("image"),
        Port("observation", "data"),
        Port("feature_ids", "data", required=False),
        Port("image_points", "data", required=False),
        Port("object_points", "data", required=False),
        Port("reference_to_image", "data", required=False),
    )
    defaults = {
        "max_view_features": 1800,
        "ratio": 0.78,
        "ransac_threshold_px": 2.5,
        "max_iterations": 3000,
        "confidence": 0.995,
        "minimum_inliers": 16,
        "minimum_inlier_fraction": 0.6,
        "minimum_target_coverage": 0.02,
    }
    parameter_ranges = {
        "max_view_features": NumericRange(32, 4096, whole=True),
        "ratio": NumericRange(0.1, 0.99),
        "ransac_threshold_px": NumericRange(0.1, 20),
        "max_iterations": NumericRange(1, 20_000, whole=True),
        "confidence": NumericRange(0.5, 0.9999),
        "minimum_inliers": NumericRange(4, _REF_FEATURES, whole=True),
        "minimum_inlier_fraction": NumericRange(0, 1),
        "minimum_target_coverage": NumericRange(0.001, 1),
    }
    parameter_help = {
        "max_view_features": "Requested SIFT feature budget in the observed view.",
        "ratio": "Maximum best-to-second descriptor distance ratio.",
        "ransac_threshold_px": "Largest allowed match residual in view pixels.",
        "max_iterations": "Maximum robust homography trials.",
        "confidence": "RANSAC confidence used to estimate the number of trials.",
        "minimum_inliers": "Minimum distinct supported point pairs to emit calibration outputs.",
        "minimum_inlier_fraction": "Minimum fraction of matches that support the homography.",
        "minimum_target_coverage": "Minimum inlier hull area as a fraction of the target area.",
    }
    input_help = {
        "image": "Photograph of the target. Rectify strong lens distortion first.",
        "reference": "Lossless grayscale target raster, with no scaling or padding.",
        "target": "Target record containing print geometry and the reference raster digest.",
        "mask": "Optional mask restricting SIFT keypoint centers in the observed image.",
        "region": "Optional region restricting SIFT keypoint centers in the observed image.",
    }
    payload_reference = "image"
    composes_result_image = True
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["minimum_inliers"]) > int(p["max_view_features"]):
            raise ValidationError(
                "Minimum inliers cannot exceed the view feature budget",
                "minimum_inliers",
                "max_view_features",
            )

    def process(self, inputs):
        p, image, target = self.parameters, inputs["image"], inputs["target"]
        gray, reference = _gray(image), _gray(inputs["reference"])
        physical, units = _target_dimensions(target, reference)
        size = reference.shape[::-1]
        ids, ref_points, observed = _matches(
            reference,
            gray,
            selection_mask(inputs, image),
            int(p["max_view_features"]),
            float(p["ratio"]),
        )
        matrix, inliers, errors, outline = _fit(ref_points, observed, size, p)
        count = int(inliers.sum())
        fraction = count / max(1, len(ids))
        coverage = _coverage(ref_points[inliers], size)
        checks = (
            (count < int(p["minimum_inliers"]), "count", "Too few geometric inliers"),
            (
                fraction < float(p["minimum_inlier_fraction"]),
                "fraction",
                "Too few matches support the homography",
            ),
            (
                coverage < float(p["minimum_target_coverage"]),
                "coverage",
                "Inliers cover too little of the target",
            ),
        )
        reasons = [
            {"code": f"random_pattern_{code}", "message": message}
            for failed, code, message in checks
            if failed
        ]
        passed = not reasons
        # Use the outer top-left edge as the physical origin. Pixel centers are half a pixel inside.
        coordinates = (ref_points[inliers] + 0.5) * physical / size
        xyz = np.column_stack((coordinates, np.zeros(count))).tolist()
        selected, xy = ids[inliers].tolist(), observed[inliers].tolist()
        rms = float(np.sqrt(np.mean(errors**2))) if count else None
        observation = {
            "model": "random_pattern_observation_v1",
            "target": target,
            "image_size": list(gray.shape[::-1]),
            "feature_ids": selected,
            "image_points": xy,
            "object_points": xyz,
            "reference_points": ref_points[inliers].tolist(),
            "match_count": len(ids),
            "inlier_count": count,
            "inlier_fraction": fraction,
            "target_coverage": coverage,
            "rms_px": rms,
            "passed": passed,
            "inlier_residuals_px": [] if errors is None else errors.tolist(),
        }
        return Result(
            {
                "image": image,
                "observation": observation,
                "feature_ids": selected if passed else NO_VALUE,
                "image_points": xy if passed else NO_VALUE,
                "object_points": xyz if passed else NO_VALUE,
                "reference_to_image": matrix.tolist() if passed else NO_VALUE,
            },
            payloads={"object_points": {"object_point_units": units}} if passed else {},
            passed=passed,
            score=fraction,
            fail_reasons=reasons,
            message=f"{count}/{len(ids)} inlier matches · {coverage:.1%} target coverage",
            drawings=_draw(observed, inliers, outline, passed, gray.shape)
            if self.generate_graphics
            else {},
        )
