"""SIFT/RootSIFT feature extraction and local-weight LightGlue planar matching."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .ml_common import file_key, local_path
from .payloads import NO_VALUE, metadata
from .poses import make_pose
from .processing import finite, gray_image, integer
from .selection import SELECTION_INPUTS, contains, selection_mask

SIFT_DEFAULTS = {
    "max_features": 1024,
    "octave_layers": 3,
    "contrast_threshold": 0.0066667,
    "edge_threshold": 10.0,
    "sigma": 1.6,
}


def sift_features(inputs, p):
    gray = gray_image(inputs["image"], (np.uint8,))
    detector = cv2.SIFT_create(
        nfeatures=integer(p["max_features"], "Maximum features", 4, 4096),
        nOctaveLayers=integer(p["octave_layers"], "Octave layers", 1, 8),
        contrastThreshold=finite(p["contrast_threshold"], "Contrast threshold", 0),
        edgeThreshold=finite(p["edge_threshold"], "Edge threshold", 0.0001),
        sigma=finite(p["sigma"], "Sigma", 0.0001),
    )
    mask = selection_mask(inputs, gray)
    keys, descriptors = detector.detectAndCompute(gray, mask)
    keep = [i for i, key in enumerate(keys) if mask is None or contains(mask, key.pt)]
    keys = [keys[i] for i in keep]
    descriptors = descriptors[keep] if descriptors is not None else np.empty((0, 128), np.float32)
    # The pretrained SIFT LightGlue checkpoint expects RootSIFT descriptors.
    if len(keys):
        descriptors /= np.maximum(np.abs(descriptors).sum(axis=1, keepdims=True), 1e-6)
        descriptors = np.sqrt(np.maximum(descriptors, 1e-6))
        descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-6)
    return {
        "model": "lightglue_sift_v1",
        "image_size": [gray.shape[1], gray.shape[0]],
        "points": [list(k.pt) for k in keys],
        "scales": [k.size for k in keys],
        "oris": [float(np.deg2rad(k.angle)) for k in keys],
        "descriptors": descriptors.tolist(),
        "coordinate_context": copy.deepcopy(metadata(inputs["image"])),
    }


def validate_features(value):
    if not isinstance(value, dict) or value.get("model") != "lightglue_sift_v1":
        raise ValueError("Reference features must come from LightGlue SIFT features")
    points = np.asarray(value.get("points"), np.float32)
    if not value.get("points"):
        points = points.reshape(0, 2)
    count = len(points)
    descriptors = np.asarray(value.get("descriptors"), np.float32).reshape(count, 128)
    scales, oris = (np.asarray(value.get(k), np.float32) for k in ("scales", "oris"))
    size = np.asarray(value.get("image_size"))
    if (
        count > 4096
        or points.shape != (count, 2)
        or scales.shape != (count,)
        or oris.shape != (count,)
        or size.shape != (2,)
        or np.any(size < 1)
        or np.any(size > 16384)
        or np.any(size != np.round(size))
        or any(not np.isfinite(v).all() for v in (points, descriptors, scales, oris))
        or np.any(scales <= 0)
    ):
        raise ValueError("Malformed LightGlue feature arrays")
    return points, descriptors, scales, oris


class LightGlueFeatures(Block):
    type_id, title = "ml.lightglue_features", "LightGlue SIFT features"
    category, block_type, subtype = "ML", "ML", "Feature matching"
    inputs = SELECTION_INPUTS
    outputs = (Port("features", "data"), Port("count", "number"), Port("overlay"))
    defaults = SIFT_DEFAULTS

    def train(self, inputs):
        """Capture portable reference descriptors; this does not fine-tune learned weights."""
        self.validate_parameters()
        return sift_features(inputs, self.parameters)

    def process(self, inputs):
        features = self.train(inputs)
        overlay = overlay_image(inputs["image"])
        for point in features["points"]:
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (60, 210, 250), 1)
        return Result(
            {"features": features, "count": len(features["points"]), "overlay": overlay},
            message="RootSIFT reference/query features (no learned weights needed)",
        )


class LightGlueMatch(Block):
    type_id, title = "ml.lightglue", "LightGlue feature matcher"
    category, block_type, subtype = "ML", "ML", "Feature matching"
    inputs = (
        *SELECTION_INPUTS,
        Port("reference", required=False),
        Port("reference_features", "data", required=False),
        Port("reference_mask", required=False),
        Port("reference_region", "data", required=False),
    )
    outputs = (
        Port("matches", "data"),
        Port("transform", "data"),
        Port("pose", "data"),
        Port("count", "number"),
        Port("inliers", "number"),
        Port("overlay"),
    )
    defaults = {
        **SIFT_DEFAULTS,
        "weights_path": "",
        "device": "cpu",
        "layers": 9,
        "filter_threshold": 0.1,
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "flash": True,
        "mixed_precision": False,
        "minimum_matches": 8,
        "minimum_inliers": 6,
        "minimum_inlier_ratio": 0.5,
        "geometry": "ransac",
        "ransac_threshold": 3.0,
        "ransac_confidence": 0.995,
        "ransac_iterations": 2000,
    }
    parameter_choices = {"device": ("cpu", "cuda", "auto"), "geometry": ("ransac", "lmeds", "rho")}
    parameter_help = {
        "weights_path": "Local official sift_lightglue.pth; no automatic download.",
        "layers": "1..9 pretrained layers; fewer layers trade accuracy for latency.",
        "depth_confidence": "Early stopping confidence, or -1 to disable.",
        "width_confidence": "Point pruning confidence, or -1 to disable.",
        "reference_features": "Use cached SIFT reference features instead of a reference image.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._matcher, self._key, self._torch, self._device = None, None, None, None

    close = reset_runtime

    def load_matcher(self):
        p = self.parameters
        path = local_path(self, p["weights_path"])
        for key in ("depth_confidence", "width_confidence"):
            value = finite(p[key], key)
            if value != -1 and not 0 <= value <= 1:
                raise ValueError(f"{key} must be -1 or in [0,1]")
        threshold = finite(p["filter_threshold"], "Filter threshold", 0)
        if threshold > 1:
            raise ValueError("Filter threshold must be in [0,1]")
        layers = integer(p["layers"], "Layers", 1, 9)
        key = (
            *file_key(path),
            p["device"],
            layers,
            threshold,
            p["depth_confidence"],
            p["width_confidence"],
            p["flash"],
            p["mixed_precision"],
        )
        if key == self._key:
            return
        try:
            import torch
            from lightglue import LightGlue
        except ImportError as exc:
            raise RuntimeError(
                "Install PyTorch and the official LightGlue package; see docs/ml.md"
            ) from exc
        device = "cuda" if p["device"] == "auto" and torch.cuda.is_available() else p["device"]
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; select CPU or install a compatible PyTorch build"
            )
        if p["mixed_precision"] and device != "cuda":
            raise ValueError("Mixed precision requires CUDA")
        matcher = LightGlue(
            features=None,
            weights=None,
            input_dim=128,
            add_scale_ori=True,
            n_layers=layers,
            filter_threshold=threshold,
            depth_confidence=p["depth_confidence"],
            width_confidence=p["width_confidence"],
            flash=p["flash"],
            mp=p["mixed_precision"],
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Invalid LightGlue state dictionary")
        state = state.get("state_dict", state)
        renamed = {}
        for name, value in state.items():
            for i in range(9):
                name = name.replace(f"self_attn.{i}.", f"transformers.{i}.self_attn.")
                name = name.replace(f"cross_attn.{i}.", f"transformers.{i}.cross_attn.")
            renamed[name] = value
        expected = matcher.state_dict()
        missing = set(expected) - renamed.keys() - {"confidence_thresholds"}
        if missing:
            raise ValueError("Checkpoint is incompatible with SIFT LightGlue weights")
        matcher.load_state_dict({k: renamed[k] for k in expected if k in renamed}, strict=False)
        self._matcher = matcher.eval().to(device)
        self._torch, self._device, self._key = torch, device, key

    def match_features(self, reference, query):
        self.load_matcher()
        torch = self._torch

        def to_torch(features):
            points, descriptors, scales, oris = validate_features(features)
            data = {
                "keypoints": points,
                "descriptors": descriptors,
                "scales": scales,
                "oris": oris,
                "image_size": np.asarray(features["image_size"], np.float32),
            }
            return {
                k: torch.from_numpy(np.ascontiguousarray(v))[None].to(self._device)
                for k, v in data.items()
            }

        with torch.inference_mode():
            matches = self._matcher({"image0": to_torch(reference), "image1": to_torch(query)})
        return (
            matches["matches"][0].detach().cpu().numpy(),
            matches["scores"][0].detach().cpu().numpy(),
        )

    def process(self, inputs):
        p = self.parameters
        query = sift_features(inputs, p)
        if "reference_features" in inputs:
            reference = inputs["reference_features"]
        elif "reference" in inputs:
            reference = sift_features(
                {
                    "image": inputs["reference"],
                    **{
                        k: inputs[f"reference_{k}"]
                        for k in ("mask", "region")
                        if f"reference_{k}" in inputs
                    },
                },
                p,
            )
        else:
            raise ValueError("Connect a reference image or cached reference features")
        points0, *_ = validate_features(reference)
        points1, *_ = validate_features(query)
        min_matches = integer(p["minimum_matches"], "Minimum matches", 4, 4096)
        min_inliers = integer(p["minimum_inliers"], "Minimum inliers", 4, 4096)
        min_ratio = finite(p["minimum_inlier_ratio"], "Minimum inlier ratio", 0)
        if min_ratio > 1:
            raise ValueError("Minimum inlier ratio must be in [0,1]")
        threshold = finite(p["ransac_threshold"], "RANSAC threshold", 0.0001)
        confidence = finite(p["ransac_confidence"], "RANSAC confidence", 0.0001)
        if confidence >= 1:
            raise ValueError("RANSAC confidence must be below 1")
        iterations = integer(p["ransac_iterations"], "RANSAC iterations", 1, 100000)
        indices, scores = (
            self.match_features(reference, query)
            if len(points0) and len(points1)
            else (np.empty((0, 2), int), np.empty(0))
        )
        source, target = points0[indices[:, 0]], points1[indices[:, 1]]
        matrix, inliers = None, np.zeros(len(indices), bool)
        if len(indices) >= 4:
            matrix, flags = cv2.findHomography(
                source,
                target,
                {"ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "rho": cv2.RHO}[p["geometry"]],
                threshold,
                maxIters=iterations,
                confidence=confidence,
            )
            if flags is not None:
                inliers = flags.reshape(-1).astype(bool)
        count, good = len(indices), int(inliers.sum())
        ratio = good / count if count else 0.0
        passed = (
            matrix is not None
            and count >= min_matches
            and good >= min_inliers
            and ratio >= min_ratio
        )
        overlay = overlay_image(inputs["image"])
        for point, accepted in zip(target, inliers):
            cv2.circle(
                overlay,
                tuple(np.rint(point).astype(int)),
                3,
                (70, 220, 100) if accepted else (40, 70, 230),
                1,
            )
        pose, transform = NO_VALUE, NO_VALUE
        if passed:
            try:
                pose = make_pose(
                    matrix, [overlay.shape[1], overlay.shape[0]], metadata(inputs["image"])
                )
                transform = {
                    "source_to_target": pose["source_to_target"],
                    "source_size": reference["image_size"],
                    "target_size": query["image_size"],
                    "source_context": reference.get("coordinate_context", {}),
                    "target_context": query["coordinate_context"],
                }
            except ValueError:
                passed, pose, transform = False, NO_VALUE, NO_VALUE
        matches = {
            "model": "keypoint_matches_v1",
            "reference_points": source.tolist(),
            "query_points": target.tolist(),
            "scores": scores.tolist(),
            "inliers": inliers.tolist(),
            "reference_context": reference.get("coordinate_context", {}),
            "query_context": query["coordinate_context"],
        }
        return Result(
            {
                "matches": matches,
                "transform": transform,
                "pose": pose,
                "count": count,
                "inliers": good,
                "overlay": overlay,
            },
            passed=passed,
            score=ratio,
            message=f"LightGlue: {count} matches / {good} geometric inliers",
            fail_reasons=[] if passed else ["lightglue_match"],
        )


LIGHTGLUE_BLOCKS = (LightGlueFeatures, LightGlueMatch)
