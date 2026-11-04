"""Trainable geometric patterns, shared model persistence and pose results."""

import copy
import hashlib
import json
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .payloads import frame, metadata
from .poses import POSE_HELP, POSE_OFFSETS, draw_pose, local_offset, make_pose, offset_pose
from .processing import finite, gray_image, integer
from .selection import SELECTION_INPUTS, contains, selection_mask

SHAPE_DEFAULTS = {
    "feature_mode": "edges",
    "blur_kernel": 3,
    "blur_sigma": 1.0,
    "edge_low": 50.0,
    "edge_high": 150.0,
    "edge_aperture": "3",
    "edge_l2": True,
    "max_points": 512,
    "min_points": 8,
    "corner_quality": 0.01,
    "corner_distance": 3.0,
    "corner_block": 3,
    "corner_gradient": "3",
    "corner_harris": False,
    "corner_k": 0.04,
    "line_votes": 20,
    "line_min_length": 10.0,
    "line_max_gap": 3.0,
    "line_rho": 1.0,
    "line_theta_degrees": 1.0,
    "circle_dp": 1.2,
    "circle_min_distance": 10.0,
    "circle_threshold": 20.0,
    "circle_min_radius": 0,
    "circle_max_radius": 0,
}
SHAPE_CHOICES = {
    "feature_mode": ("edges", "corners", "lines", "circles", "combined"),
    "edge_aperture": ("3", "5", "7"),
    "corner_gradient": ("3", "5", "7"),
}
PATTERN_OUTPUTS = (
    Port("matches", "data"),
    Port("count", "number"),
    Port("score", "number"),
    Port("transform", "data"),
    Port("pose", "data"),
    Port("points", "data"),
    Port("overlay"),
)
MATCH_INPUTS = (
    *SELECTION_INPUTS,
    Port("model", "data", required=False),
    Port("training_mask", required=False),
    Port("training_region", "data", required=False),
)


def seal_model(model):
    content = {k: v for k, v in model.items() if k not in ("id", "training_context")}
    model["id"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    return model


def validate_model(model, kind):
    if not isinstance(model, dict) or model.get("model") != f"{kind}_pattern_v1":
        raise ValueError(f"Expected a trained {kind} pattern model")
    size = model.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or not 2 <= n <= 4096 for n in size)
    ):
        raise ValueError("Pattern size must be 2..4096 pixels per dimension")
    if not isinstance(model.get("settings"), dict):
        raise ValueError("Trained pattern has no extraction settings")
    if seal_model(copy.deepcopy(model))["id"] != model.get("id"):
        raise ValueError("Trained model checksum does not match its content")
    points = np.asarray(model.get("points"), float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 4 <= len(points) <= 10000
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points[:, 0] >= size[0])
        or np.any(points[:, 1] >= size[1])
    ):
        raise ValueError("Pattern needs 4..10000 finite points inside its template")
    if (
        kind == "shape"
        and model["settings"].get("feature_mode") not in SHAPE_CHOICES["feature_mode"]
    ):
        raise ValueError("Unknown trained geometric feature mode")
    return model


def training_image(inputs):
    source = inputs["image"]
    gray = gray_image(source, (np.uint8,))
    mask = selection_mask(inputs, gray)
    if mask is None:
        mask = np.full_like(gray, 255)
    if not mask.any():
        raise ValueError("Training requires a nonempty mask/region")
    x, y, w, h = cv2.boundingRect(mask)
    if min(w, h) < 2 or max(w, h) > 4096:
        raise ValueError("Training region must be 2..4096 pixels per dimension")
    return (
        gray[y : y + h, x : x + w].copy(),
        mask[y : y + h, x : x + w].copy(),
        [x, y],
        metadata(source),
    )


def shape_features(gray, p, mask=None):
    blur = integer(p["blur_kernel"], "Blur kernel", 1, 63, odd=True)
    sigma = finite(p["blur_sigma"], "Blur sigma", 0)
    image = cv2.GaussianBlur(gray, (blur, blur), sigma)
    low, high = finite(p["edge_low"], "Edge low", 0), finite(p["edge_high"], "Edge high", 0)
    if high < low:
        raise ValueError("Edge high must be at least edge low")
    mode = p["feature_mode"]
    output = np.zeros_like(gray)
    edges = cv2.Canny(
        image, low, high, apertureSize=int(p["edge_aperture"]), L2gradient=p["edge_l2"]
    )
    if mode in ("edges", "combined"):
        output |= edges
    if mode in ("corners", "combined"):
        quality = finite(p["corner_quality"], "Corner quality", 0)
        if not 0 < quality <= 1:
            raise ValueError("Corner quality must be in (0,1]")
        found = cv2.goodFeaturesToTrack(
            image,
            integer(p["max_points"], "Maximum points", 4, 4096),
            quality,
            finite(p["corner_distance"], "Corner distance", 0),
            blockSize=integer(p["corner_block"], "Corner block", 3, 31, odd=True),
            gradientSize=int(p["corner_gradient"]),
            useHarrisDetector=p["corner_harris"],
            k=finite(p["corner_k"], "Harris k", 0),
            mask=mask,
        )
        for x, y in found.reshape(-1, 2) if found is not None else []:
            output[round(y), round(x)] = 255
    if mode in ("lines", "combined"):
        rho, theta = (
            finite(p["line_rho"], "Line rho", 0),
            finite(p["line_theta_degrees"], "Line theta", 0),
        )
        if rho <= 0 or not 0 < theta <= 180:
            raise ValueError("Line rho must be positive and theta in (0,180]")
        found = cv2.HoughLinesP(
            edges,
            rho,
            np.deg2rad(theta),
            integer(p["line_votes"], "Line votes", 1),
            minLineLength=finite(p["line_min_length"], "Line minimum length", 0),
            maxLineGap=finite(p["line_max_gap"], "Line gap", 0),
        )
        for x1, y1, x2, y2 in found.reshape(-1, 4) if found is not None else []:
            cv2.line(output, (x1, y1), (x2, y2), 255, 1)
    if mode in ("circles", "combined"):
        dp = finite(p["circle_dp"], "Circle DP", 1)
        distance = finite(p["circle_min_distance"], "Circle distance", 0)
        threshold = finite(p["circle_threshold"], "Circle center threshold", 0)
        minimum = integer(p["circle_min_radius"], "Minimum radius", 0, 4096)
        maximum = integer(p["circle_max_radius"], "Maximum radius", 0, 4096)
        if min(distance, threshold, high) <= 0 or (maximum and maximum < minimum):
            raise ValueError("Invalid circle distances/thresholds/radius range")
        found = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp,
            distance,
            param1=high,
            param2=threshold,
            minRadius=minimum,
            maxRadius=maximum,
        )
        for x, y, radius in found.reshape(-1, 3) if found is not None else []:
            cv2.circle(output, (round(x), round(y)), round(radius), 255, 1)
    return output


def train_shape(inputs, parameters):
    p = {key: parameters[key] for key in SHAPE_DEFAULTS}
    gray, mask, origin, context = training_image(inputs)
    full = gray_image(inputs["image"], (np.uint8,))
    full_mask = selection_mask(inputs, full)
    x, y = origin
    features = shape_features(full, p, full_mask)[
        y : y + gray.shape[0], x : x + gray.shape[1]
    ].copy()
    features[mask == 0] = 0
    points = np.argwhere(features != 0)[:, ::-1]
    minimum = integer(p["min_points"], "Minimum points", 4, 4096)
    maximum = integer(p["max_points"], "Maximum points", 4, 4096)
    if minimum > maximum or len(points) < minimum:
        raise ValueError("Not enough geometric features to train this pattern")
    if len(points) > maximum:
        points = points[np.linspace(0, len(points) - 1, maximum).astype(int)]
    return seal_model(
        {
            "model": "shape_pattern_v1",
            "size": [gray.shape[1], gray.shape[0]],
            "points": points.tolist(),
            "settings": copy.deepcopy(p),
            "training_origin": origin,
            "training_context": context,
        }
    )


def model_payload(model):
    return {"spatial": frame(model["size"], f"pattern:{model['id']}")}


def training_result(model, image):
    preview = overlay_image(image)
    origin = np.asarray(model["training_origin"])
    for point in np.asarray(model["points"]) + origin:
        cv2.circle(preview, tuple(np.rint(point).astype(int)), 1, (60, 240, 100), -1)
    return Result(
        {"model": model, "preview": preview},
        graphics={"trained features": preview},
        payloads={"model": model_payload(model)},
        message=f"Trained {len(model['points'])} features",
    )


class TrainableMatcher:
    supports_training = True
    model_kind = "shape"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trained_model = None

    def train(self, inputs):
        self.validate_parameters()
        training = dict(inputs)
        if "training_mask" in inputs:
            training["mask"] = inputs["training_mask"]
        if "training_region" in inputs:
            training["region"] = inputs["training_region"]
        model = self.build_model(training)
        self.check_model(model)
        self.trained_model = copy.deepcopy(model)
        self.clear_result()
        return copy.deepcopy(model)

    def check_model(self, model):
        return validate_model(model, self.model_kind)

    def serialize_extra(self):
        return (
            {"trained_model": copy.deepcopy(self.trained_model)}
            if self.trained_model is not None
            else {}
        )

    def restore_extra(self, data):
        if "trained_model" in data:
            self.check_model(data["trained_model"])
            self.trained_model = copy.deepcopy(data["trained_model"])

    def get_model(self, inputs):
        model = inputs.get("model", self.trained_model)
        if model is None:
            raise ValueError("Train this pattern from its input or connect a trained model")
        return self.check_model(model)

    def finalize_result(self, result, inputs):
        p = self.parameters
        # Validate even for negative inspections, so invalid controls are never hidden.
        local_offset(*(p[k] for k in POSE_OFFSETS))
        matches = result.outputs["matches"]["items"]
        for match in matches:
            transform = match["transform"]
            w, h = transform["source_size"]
            matrix = np.asarray(transform["source_to_target"]) @ local_offset(
                (w - 1) / 2, (h - 1) / 2, 0
            )
            pose = make_pose(matrix, transform["target_size"], transform["target_payload"])
            match["pose"] = offset_pose(pose, *(p[k] for k in POSE_OFFSETS))
            draw_pose(result.outputs["overlay"], match["pose"])
        result.outputs["pose"] = matches[0]["pose"] if matches else {}
        return result


class TrainShapePattern(Block):
    type_id, title, category = "pattern.train_shape", "Train shape pattern", "Inspection"
    block_type, subtype = "Inspection", "Pattern training"
    inputs = SELECTION_INPUTS
    outputs = (Port("model", "data"), Port("preview"))
    defaults, parameter_choices = SHAPE_DEFAULTS, SHAPE_CHOICES

    def train(self, inputs):
        self.validate_parameters()
        return train_shape(inputs, self.parameters)

    def process(self, inputs):
        return training_result(self.train(inputs), inputs["image"])


def pattern_corners(size):
    w, h = size
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def pose_record(model, matrix, target, score, **details):
    corners = cv2.perspectiveTransform(pattern_corners(model["size"])[:, None, :], matrix).reshape(
        -1, 2
    )
    center = cv2.perspectiveTransform(
        np.array([[[(model["size"][0] - 1) / 2, (model["size"][1] - 1) / 2]]], np.float64), matrix
    )[0, 0]
    if not np.isfinite(corners).all() or not np.isfinite(center).all():
        raise ValueError("Pattern pose projects to nonfinite coordinates")
    return {
        "score": float(np.clip(score, 0, 1)),
        "center": center.tolist(),
        "corners": corners.tolist(),
        "transform": {
            "model": "image_pattern_pose_v1",
            "source_to_target": matrix.tolist(),
            "source_size": model["size"],
            "target_size": [target.shape[1], target.shape[0]],
            "source_payload": model_payload(model),
            "target_payload": metadata(target),
        },
        **details,
    }


def pattern_result(image, model, matches):
    overlay = overlay_image(image)
    for index, match in enumerate(matches, 1):
        corners = np.rint(match["corners"]).astype(np.int32)
        cv2.polylines(overlay, [corners], True, (60, 230, 120), 2)
        center = tuple(np.rint(match["center"]).astype(int))
        cv2.drawMarker(overlay, center, (60, 230, 120), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(
            overlay,
            f"{index}: {match['score']:.2f}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 230, 120),
            1,
        )
    score = matches[0]["score"] if matches else 0.0
    return Result(
        {
            "matches": {
                "model": "pattern_matches_v1",
                "pattern_id": model["id"],
                "items": matches,
                "count": len(matches),
            },
            "count": len(matches),
            "score": score,
            "transform": matches[0]["transform"] if matches else {},
            "pose": {},  # TrainableMatcher adds the offset pose after matching.
            "points": [m["center"] for m in matches],
            "overlay": overlay,
        },
        passed=bool(matches),
        score=score,
        graphics={"pattern matches": overlay},
        message=f"{len(matches)} pattern matches, best score {score:.3f}",
    )


def bounded_samples(low, high, step, name):
    low, high, step = finite(low, name), finite(high, name), finite(step, name, 0)
    if high < low or step <= 0 or (high - low) / step > 127:
        raise ValueError(f"Invalid or excessive {name} search range")
    return np.arange(low, high + step * 1e-6, step)


def overlap(a, b):
    x, y, w, h = cv2.boundingRect(np.asarray(a["corners"], np.float32))
    u, v, s, t = cv2.boundingRect(np.asarray(b["corners"], np.float32))
    intersection = max(0, min(x + w, u + s) - max(x, u)) * max(0, min(y + h, v + t) - max(y, v))
    return intersection / (w * h + s * t - intersection)


class MatchShapePattern(TrainableMatcher, Block):
    type_id, title, category = "pattern.match_shape", "Match shape pattern", "Inspection"
    block_type, subtype = "Inspection", "Pattern matching"
    inputs = MATCH_INPUTS
    outputs = PATTERN_OUTPUTS
    defaults = {
        **SHAPE_DEFAULTS,
        **POSE_OFFSETS,
        "rotation_min": 0.0,
        "rotation_max": 0.0,
        "rotation_step": 5.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "scale_step": 0.1,
        "distance_tolerance": 3.0,
        "min_score": 0.8,
        "min_coverage": 0.8,
        "max_matches": 1,
        "max_overlap": 0.3,
    }
    parameter_choices = SHAPE_CHOICES
    parameter_help = {
        **POSE_HELP,
        "feature_mode": "Training setting; stored models retain their extraction settings.",
        "distance_tolerance": "Pixel distance for coverage and chamfer score decay.",
        "rotation_min": "OpenCV rotation degrees; search is sampled, not unrestricted.",
        "min_score": "exp(-mean feature distance / tolerance), from 0 to 1.",
    }

    def build_model(self, inputs):
        return train_shape(inputs, self.parameters)

    def process(self, inputs):
        model = self.get_model(inputs)
        image = gray_image(inputs["image"], (np.uint8,))
        features = shape_features(image, model["settings"])
        selection = selection_mask(inputs, image)
        p = self.parameters
        angles = bounded_samples(
            p["rotation_min"], p["rotation_max"], p["rotation_step"], "rotation"
        )
        scales = bounded_samples(p["scale_min"], p["scale_max"], p["scale_step"], "scale")
        if (
            min(scales) <= 0
            or max(scales) > 8
            or len(angles) * len(scales) > 128
            or image.size * len(angles) * len(scales) > 100_000_000
        ):
            raise ValueError(
                "Shape search exceeds scale/variant/work limits; reduce image or search range"
            )
        tolerance = finite(p["distance_tolerance"], "Distance tolerance", 0)
        minimum, coverage_min, maximum_overlap = (
            finite(p[k], k, 0) for k in ("min_score", "min_coverage", "max_overlap")
        )
        if tolerance <= 0 or max(minimum, coverage_min, maximum_overlap) > 1:
            raise ValueError("Invalid shape score, coverage, overlap or distance tolerance")
        required = integer(p["max_matches"], "Maximum matches", 1, 256)
        if not features.any():
            return pattern_result(inputs["image"], model, [])
        distance = cv2.distanceTransform(255 - features, cv2.DIST_L2, 3)
        points = np.asarray(model["points"], np.float64)
        center = ((model["size"][0] - 1) / 2, (model["size"][1] - 1) / 2)
        candidates = []
        for angle in angles:
            for scale in scales:
                if self._cancelled():
                    raise RuntimeError("Shape search cancelled")
                affine = cv2.getRotationMatrix2D(center, float(angle), float(scale))
                matrix = np.vstack((affine, [0, 0, 1]))
                corners = cv2.perspectiveTransform(
                    pattern_corners(model["size"])[:, None], matrix
                ).reshape(-1, 2)
                low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
                w, h = (high - low + 1).astype(int)
                if w > image.shape[1] or h > image.shape[0]:
                    continue
                matrix[:2, 2] -= low
                transformed = cv2.perspectiveTransform(points[:, None], matrix).reshape(-1, 2)
                indices = np.unique(np.rint(transformed).astype(int), axis=0)
                kernel = np.zeros((h, w), np.float32)
                kernel[indices[:, 1], indices[:, 0]] = 1 / len(indices)
                cost = cv2.matchTemplate(distance, kernel, cv2.TM_CCORR)
                anchor = cv2.perspectiveTransform(np.array([[center]], np.float64), matrix)[0, 0]
                if selection is not None:
                    rows = np.rint(np.arange(cost.shape[0]) + anchor[1]).astype(int)
                    columns = np.rint(np.arange(cost.shape[1]) + anchor[0]).astype(int)
                    cost[selection[rows[:, None], columns[None, :]] == 0] = np.inf
                for _ in range(required * 8):
                    mean, _, location, _ = cv2.minMaxLoc(cost)
                    if (
                        not np.isfinite(mean)
                        or math.exp(-max(0, mean) / tolerance) + 1e-5 < minimum
                    ):
                        break
                    x, y = location
                    radius = max(2, min(w, h) // 8)
                    cost[
                        max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
                    ] = np.inf
                    if not contains(selection, anchor + [x, y]):
                        continue
                    values = distance[indices[:, 1] + y, indices[:, 0] + x]
                    score, coverage = (
                        math.exp(-float(values.mean()) / tolerance),
                        float(np.mean(values <= tolerance)),
                    )
                    if score < minimum or coverage < coverage_min:
                        continue
                    pose = matrix.copy()
                    pose[:2, 2] += [x, y]
                    candidates.append(
                        pose_record(
                            model,
                            pose,
                            inputs["image"],
                            score,
                            angle_degrees=float(angle),
                            scale=float(scale),
                            edge_coverage=coverage,
                            mean_distance=float(values.mean()),
                        )
                    )
        matches = []
        for candidate in sorted(
            candidates, key=lambda m: (-m["score"], abs(m["angle_degrees"]), abs(m["scale"] - 1))
        ):
            if all(overlap(candidate, previous) <= maximum_overlap for previous in matches):
                matches.append(candidate)
                if len(matches) >= required:
                    break
        return pattern_result(inputs["image"], model, matches)


class PatternDemoImages(Block):
    type_id, title, category = "input.pattern_demo", "Pattern training demo images", "Input"
    block_type, subtype = "Input", "Synthetic"
    outputs = (Port("image"), Port("shape_reference"), Port("texture_reference"))

    def process(self, inputs):
        shape = np.zeros((96, 112), np.uint8)
        cv2.line(shape, (18, 16), (18, 70), 255, 3)
        cv2.line(shape, (18, 70), (60, 70), 255, 3)
        cv2.circle(shape, (76, 30), 13, 255, 2)
        cv2.rectangle(shape, (47, 13), (57, 23), 255, 2)
        texture = np.random.default_rng(37).integers(0, 256, (192, 192), np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        image = np.full((400, 640), 20, np.uint8)
        matrix = cv2.getRotationMatrix2D((55.5, 47.5), 10, 1)
        matrix[:, 2] += [70, 80]
        rotated = cv2.warpAffine(shape, matrix, (640, 400))
        image[rotated != 0] = rotated[rotated != 0]
        image[120:312, 330:522] = texture
        image[20:380, 590:600] = 240
        outputs = {"image": image, "shape_reference": shape, "texture_reference": texture}
        return Result(
            outputs,
            payloads={
                key: {"spatial": frame([value.shape[1], value.shape[0]], f"{self.id}:{key}")}
                for key, value in outputs.items()
            },
        )
