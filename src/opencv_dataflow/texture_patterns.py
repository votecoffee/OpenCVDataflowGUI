"""Trainable ORB/SIFT planar texture-pattern localization."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .patterns import (
    MATCH_INPUTS,
    PATTERN_OUTPUTS,
    TrainableMatcher,
    pattern_result,
    pose_record,
    seal_model,
    training_image,
    training_result,
    validate_model,
)
from .poses import POSE_HELP, POSE_OFFSETS
from .processing import finite, gray_image, integer
from .registration import ORBFeatures
from .result_graphics import ACCEPTED, REJECTED, review_legend
from .selection import SELECTION_INPUTS, contains, selection_mask

TEXTURE_DEFAULTS = {
    "algorithm": "orb",
    **{f"orb_{k}": v for k, v in ORBFeatures.defaults.items()},
    "sift_max_features": 1000,
    "sift_octave_layers": 3,
    "sift_contrast": 0.04,
    "sift_edge_threshold": 10.0,
    "sift_sigma": 1.6,
    "sift_precise_upscale": False,
    "min_training_features": 8,
}
TEXTURE_CHOICES = {
    "algorithm": ("orb", "sift"),
    **{f"orb_{k}": v for k, v in ORBFeatures.parameter_choices.items()},
}


def texture_features(gray, mask, p):
    if p["algorithm"] == "orb":
        parameters = {key: p[f"orb_{key}"] for key in ORBFeatures.defaults}
        tool = ORBFeatures(parameters=parameters)
        tool.execute({"image": gray, **({"mask": mask} if mask is not None else {})})
        if tool.last_result is None:
            raise ValueError(tool.error)
        features = tool.last_result.outputs["features"]
        return np.asarray([k["point"] for k in features["keypoints"]], float).reshape(
            -1, 2
        ), np.asarray(features["descriptors"], np.uint8).reshape(-1, 32)
    if p["algorithm"] != "sift":
        raise ValueError("Texture algorithm must be orb or sift")
    count = integer(p["sift_max_features"], "SIFT maximum features", 0, 5000)
    layers = integer(p["sift_octave_layers"], "SIFT octave layers", 1, 8)
    contrast, edge, sigma = (
        finite(p[k], k, 0) for k in ("sift_contrast", "sift_edge_threshold", "sift_sigma")
    )
    if min(edge, sigma) <= 0:
        raise ValueError("SIFT edge threshold and sigma must be positive")
    detector = cv2.SIFT_create(
        count, layers, contrast, edge, sigma, enable_precise_upscale=p["sift_precise_upscale"]
    )
    keys, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    keep = [i for i, k in enumerate(keys) if contains(mask, k.pt)]
    return np.asarray([keys[i].pt for i in keep], float).reshape(-1, 2), descriptors[keep]


def train_texture(inputs, parameters):
    p = {key: parameters[key] for key in TEXTURE_DEFAULTS}
    gray, mask, origin, context = training_image(inputs)
    points, descriptors = texture_features(gray, mask, p)
    if len(points) < integer(p["min_training_features"], "Minimum training features", 4, 10000):
        raise ValueError(
            "Not enough texture features to train; enlarge the ROI or adjust detector settings"
        )
    if len(points) > 10000:
        raise ValueError("Texture model exceeds 10000 features; limit the detector count")
    return seal_model(
        {
            "model": "texture_pattern_v1",
            "size": [gray.shape[1], gray.shape[0]],
            "points": points.tolist(),
            "descriptors": descriptors.tolist(),
            "settings": copy.deepcopy(p),
            "training_origin": origin,
            "training_context": context,
        }
    )


def check_texture_model(model):
    validate_model(model, "texture")
    p = model["settings"]
    algorithm = p.get("algorithm")
    if algorithm not in ("orb", "sift"):
        raise ValueError("Unknown trained texture algorithm")
    descriptors = np.asarray(model.get("descriptors"), float)
    width = 32 if algorithm == "orb" else 128
    if (
        descriptors.shape != (len(model["points"]), width)
        or not np.isfinite(descriptors).all()
        or np.any(descriptors < 0)
        or np.any(descriptors > (255 if algorithm == "orb" else 1e6))
    ):
        raise ValueError("Invalid trained texture descriptor matrix")
    if algorithm == "orb" and (
        np.any(descriptors != np.rint(descriptors)) or p.get("orb_wta_k") not in ("2", "3", "4")
    ):
        raise ValueError("Invalid ORB descriptor bytes or WTA_K")
    return model


class TrainTexturePattern(Block):
    type_id, title, category = "pattern.train_texture", "Train texture pattern", "Inspection"
    block_type, subtype = "Inspection", "Pattern training"
    inputs = SELECTION_INPUTS
    outputs = (Port("model", "data"), Port("preview"))
    defaults, parameter_choices = TEXTURE_DEFAULTS, TEXTURE_CHOICES

    def train(self, inputs):
        self.validate_parameters()
        return train_texture(inputs, self.parameters)

    def process(self, inputs):
        return training_result(self.train(inputs), inputs["image"])


class MatchTexturePattern(TrainableMatcher, Block):
    type_id, title, category = "pattern.match_texture", "Match texture pattern", "Inspection"
    block_type, subtype = "Inspection", "Pattern matching"
    model_kind = "texture"
    inputs = MATCH_INPUTS
    outputs = PATTERN_OUTPUTS
    defaults = {
        **TEXTURE_DEFAULTS,
        **POSE_OFFSETS,
        "ratio": 0.75,
        "mutual": True,
        "orb_max_distance": 64.0,
        "sift_max_distance": 250.0,
        "pose_model": "homography",
        "robust_method": "ransac",
        "reprojection_pixels": 3.0,
        "max_iterations": 2000,
        "confidence": 0.995,
        "min_matches": 8,
        "min_inliers": 6,
        "min_inlier_ratio": 0.5,
        "min_model_coverage": 0.02,
        "min_scale": 0.2,
        "max_scale": 5.0,
        "max_rms_pixels": 3.0,
        "refine_iterations": 10,
    }
    parameter_choices = {
        **TEXTURE_CHOICES,
        "pose_model": ("similarity", "affine", "homography"),
        "robust_method": ("ransac", "lmeds"),
    }
    parameter_help = {
        **POSE_HELP,
        "algorithm": "Training control; a supplied/stored model retains its detector settings.",
        "min_inlier_ratio": "Pose score = inliers / accepted descriptor correspondences.",
        "min_model_coverage": "Inliers / trained feature count, from 0 to 1.",
    }

    def build_model(self, inputs):
        return train_texture(inputs, self.parameters)

    def check_model(self, model):
        return check_texture_model(model)

    def process(self, inputs):
        model = self.get_model(inputs)
        gray = gray_image(inputs["image"], (np.uint8,))
        mask = selection_mask(inputs, gray)
        settings, p = model["settings"], self.parameters
        target_points, target_desc = texture_features(gray, mask, settings)
        binary = settings["algorithm"] == "orb"
        dtype = np.uint8 if binary else np.float32
        descriptors = np.asarray(model["descriptors"], dtype)
        points = np.asarray(model["points"], np.float64)
        norm = (
            (cv2.NORM_HAMMING if settings.get("orb_wta_k") == "2" else cv2.NORM_HAMMING2)
            if binary
            else cv2.NORM_L2
        )
        ratio = finite(p["ratio"], "Ratio", 0)
        distance = finite(
            p["orb_max_distance"] if binary else p["sift_max_distance"], "Maximum distance", 0
        )
        confidence = finite(p["confidence"], "Confidence", 0)
        tolerance = finite(p["reprojection_pixels"], "Reprojection threshold", 0)
        fraction, coverage = (
            finite(p[k], k, 0) for k in ("min_inlier_ratio", "min_model_coverage")
        )
        low, high = (
            finite(p["min_scale"], "Minimum scale", 0),
            finite(p["max_scale"], "Maximum scale", 0),
        )
        rms_limit = finite(p["max_rms_pixels"], "Maximum RMS", 0)
        iterations = integer(p["max_iterations"], "Maximum iterations", 1, 100000)
        refinement = integer(p["refine_iterations"], "Refine iterations", 0, 10000)
        required = integer(p["min_matches"], "Minimum matches", 4, 10000)
        inlier_min = integer(p["min_inliers"], "Minimum inliers", 4, 10000)
        if (
            not 0 < ratio < 1
            or not 0 < confidence < 1
            or tolerance <= 0
            or max(fraction, coverage) > 1
            or not 0 < low <= high
            or rms_limit <= 0
        ):
            raise ValueError(
                "Invalid texture matching ratio, confidence, thresholds or scale range"
            )
        if len(target_desc) < 2:
            return pattern_result(inputs["image"], model, [])
        matcher = cv2.BFMatcher(norm)
        pairs = matcher.knnMatch(descriptors, target_desc, k=2)
        matches = [
            a[0]
            for a in pairs
            if len(a) == 2 and a[0].distance < ratio * a[1].distance and a[0].distance <= distance
        ]
        if p["mutual"]:
            reverse = {m.queryIdx: m.trainIdx for m in matcher.match(target_desc, descriptors)}
            matches = [m for m in matches if reverse.get(m.trainIdx) == m.queryIdx]
        unique = {}
        for m in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(m.trainIdx, m)
        matches = list(unique.values())

        def finish(accepted=(), rejected=(), inliers=()):
            result = pattern_result(inputs["image"], model, list(accepted), rejected)
            if self.generate_graphics:
                good = {m.trainIdx for m, keep in zip(matches, inliers) if keep}
                proposed = {pair[0].trainIdx for pair in pairs if pair}
                graphic = overlay_image(inputs["image"])
                objects = []
                for indices, color in ((proposed - good, REJECTED), (good, ACCEPTED)):
                    for index in sorted(indices):
                        center = tuple(np.rint(target_points[index]).astype(int))
                        cv2.circle(graphic, center, 3, color, 1)
                        objects.append(graphic_object("marker", [target_points[index]], color))
                review_legend(graphic, len(good), len(proposed - good))
                result.graphics["texture correspondences"] = graphic
                result.drawings["texture correspondences"] = GraphicDrawing(tuple(objects))
            return result

        if len(matches) < required:
            return finish()
        a = np.asarray([points[m.queryIdx] for m in matches], np.float64)
        b = np.asarray([target_points[m.trainIdx] for m in matches], np.float64)
        method = cv2.RANSAC if p["robust_method"] == "ransac" else cv2.LMEDS
        if p["pose_model"] == "homography":
            matrix, inliers = cv2.findHomography(
                a, b, method, tolerance, maxIters=iterations, confidence=confidence
            )
        else:
            solver = (
                cv2.estimateAffinePartial2D
                if p["pose_model"] == "similarity"
                else cv2.estimateAffine2D
            )
            matrix, inliers = solver(
                a,
                b,
                method=method,
                ransacReprojThreshold=tolerance,
                maxIters=iterations,
                confidence=confidence,
                refineIters=refinement,
            )
            if matrix is not None:
                matrix = np.vstack((matrix, [0, 0, 1]))
        if (
            matrix is None
            or inliers is None
            or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) != 3
        ):
            return finish()
        valid = inliers.reshape(-1).astype(bool)
        count = int(valid.sum())
        score, model_coverage = count / len(matches), count / len(points)
        reason = ""
        if count < inlier_min or score < fraction:
            reason = "inliers"
        elif model_coverage < coverage:
            reason = "coverage"
        if reason and (not self.generate_graphics or not count):
            return finish(inliers=valid)
        projection = cv2.perspectiveTransform(a[valid, None], matrix).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projection - b[valid]) ** 2, axis=1))))
        try:
            record = pose_record(
                model,
                matrix,
                inputs["image"],
                score,
                inliers=count,
                correspondences=len(matches),
                rms_pixels=rms,
                model_coverage=model_coverage,
            )
        except ValueError:
            if reason:
                return finish(inliers=valid)
            raise
        corners = np.asarray(record["corners"], np.float32)
        if (
            not np.isfinite(corners).all()
            or np.max(np.abs(corners)) > 10_000_000
            or max(map(abs, record["center"])) > 10_000_000
        ):
            return finish(inliers=valid)
        area = cv2.contourArea(corners, oriented=True)
        scale = math.sqrt(max(0, area) / ((model["size"][0] - 1) * (model["size"][1] - 1)))
        if not cv2.isContourConvex(corners) or not contains(mask, record["center"]):
            return finish(inliers=valid)
        if not reason and not low <= scale <= high:
            reason = "scale"
        if not reason and rms > rms_limit:
            reason = "RMS"
        if reason:
            rejected = [(record, reason)] if self.generate_graphics else []
            return finish(rejected=rejected, inliers=valid)
        delta = corners[1] - corners[0]
        record.update(scale=scale, angle_degrees=-math.degrees(math.atan2(delta[1], delta[0])))
        return finish(accepted=[record], inliers=valid)
