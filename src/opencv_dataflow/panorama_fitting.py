"""Explicit native rotation fitting with disjoint fit/validation keypoint observations."""

import hashlib
from itertools import combinations

import cv2
import numpy as np

from .panorama_geometry import intrinsics
from .parameter_validation import NumericRange, ValidationError

DEFAULTS = {
    "camera_solver": "automatic",
    "bundle_adjustment": "reprojection",
    "initial_intrinsics": [],
    "refine_focal": True,
    "refine_aspect": False,
    "refine_principal_point": False,
    "camera_features": 1500,
    "native_match_confidence": 0.65,
    "adjustment_iterations": 100,
    "adjustment_epsilon": 1e-6,
    "holdout_fraction": 0.35,
    "holdout_seed": 104,
}
CHOICES = {
    "camera_solver": ("automatic", "explicit"),
    "bundle_adjustment": ("none", "ray", "reprojection"),
}
RANGES = {
    "camera_features": NumericRange(32, 4000, whole=True),
    "native_match_confidence": NumericRange(0.01, 0.99),
    "adjustment_iterations": NumericRange(1, 1000, whole=True),
    "adjustment_epsilon": NumericRange(1e-10, 0.1),
    "holdout_fraction": NumericRange(0.1, 0.6),
    "holdout_seed": NumericRange(0, 2**31 - 1, whole=True),
}
HELP = {
    "camera_solver": "Automatic Stitcher or explicit native fit with reserved keypoints.",
    "bundle_adjustment": "Explicit mode only: no adjustment, ray or reprojection cost.",
    "initial_intrinsics": (
        "Explicit mode: one original-source 3x3 pinhole K per input, or [] for estimated focal."
    ),
    "refine_focal": "Explicit reprojection: optimize focal. Ray requires this on.",
    "refine_aspect": "Explicit reprojection: optimize fy/fx. Ray requires this off.",
    "refine_principal_point": "Explicit reprojection: optimize cx/cy. Ray requires this off.",
    "camera_features": "Explicit mode: cap detected SIFT observations before partitioning.",
    "native_match_confidence": "Explicit native matching uses distance ratio < 1 - this value.",
    "adjustment_iterations": "Explicit native adjustment iteration limit.",
    "adjustment_epsilon": "Explicit native adjustment termination epsilon.",
    "holdout_fraction": "Explicit mode: reserve this fraction of rounded keypoint centers.",
    "holdout_seed": "Explicit mode: stable spatial partition seed; no automatic retries.",
}


def validate_settings(p):
    for name in ("refine_focal", "refine_aspect", "refine_principal_point"):
        if type(p[name]) is not bool:
            raise ValidationError("Use boolean camera refinement settings", name)
    priors = p["initial_intrinsics"]
    if not isinstance(priors, list) or len(priors) > 8:
        raise ValidationError(
            "Initial intrinsics must be a list of up to eight matrices", "initial_intrinsics"
        )
    for value in priors:
        if not isinstance(value, list):
            raise ValidationError(
                "Use JSON matrix lists for initial intrinsics", "initial_intrinsics"
            )
        intrinsics(value)
    if priors and p["camera_solver"] != "explicit":
        raise ValidationError(
            "Initial intrinsics require explicit camera fitting",
            "initial_intrinsics",
            "camera_solver",
        )
    if (
        p["camera_solver"] == "explicit"
        and p["bundle_adjustment"] == "ray"
        and (not p["refine_focal"] or p["refine_aspect"] or p["refine_principal_point"])
    ):
        raise ValidationError(
            "Ray adjustment refines focal with fixed centered square intrinsics",
            "bundle_adjustment",
            "refine_focal",
            "refine_aspect",
            "refine_principal_point",
        )


def partition(keys, fraction, seed):
    """Keep all orientations/scales at a rounded pixel center on the same side."""
    reserved = []
    for key in keys:
        center = tuple(int(np.floor(v + 0.5)) for v in key.pt)
        digest = hashlib.sha256(f"{seed}:{center[0]}:{center[1]}".encode()).digest()
        reserved.append(int.from_bytes(digest[:8], "big") / 2**64 < fraction)
    return np.asarray(reserved, bool)


def subset_matches(original, source, target, threshold):
    result = cv2.detail.MatchesInfo()
    result.src_img_idx, result.dst_img_idx = source, target
    if source == target or original.confidence <= threshold or original.num_inliers < 4:
        return result
    result.matches = original.matches
    result.inliers_mask = original.inliers_mask
    result.num_inliers, result.confidence = original.num_inliers, original.confidence
    result.H = original.H
    return result


def explicit_fit(work, masks, scales, active, anchor, p):
    """Return status, original indices, native cameras, held-out features and audit."""
    audit = {
        "solver": "explicit",
        "bundle_adjustment": p["bundle_adjustment"],
        "validation_basis": "reserved_keypoints",
        "validation_excluded_from_camera_fit": True,
        "partition": "rounded_pixel_center_sha256_v1",
        "holdout_seed": p["holdout_seed"],
        "holdout_fraction": p["holdout_fraction"],
        "features": [],
        "training_pairs": [],
    }
    priors = p["initial_intrinsics"]
    if priors and len(priors) != len(work):
        raise ValidationError("Supply one initial K for each original view", "initial_intrinsics")
    work_priors = []
    for i, prior in enumerate(priors):
        K = np.diag([*scales[i], 1]) @ intrinsics(prior)
        width, height = work[i].shape[::-1]
        if not 0 <= K[0, 2] < width or not 0 <= K[1, 2] < height:
            raise ValidationError(
                "Initial principal points must be inside source images", "initial_intrinsics"
            )
        if p["bundle_adjustment"] == "ray" and (
            not np.isclose(K[0, 0], K[1, 1], rtol=1e-6)
            or not np.allclose(K[:2, 2], [width / 2, height / 2], atol=1e-6)
        ):
            raise ValidationError(
                "Ray priors must be centered and square on the work grid",
                "initial_intrinsics",
                "bundle_adjustment",
            )
        work_priors.append(K)
    held, training, usable = {}, {}, []
    detector = cv2.SIFT_create(p["camera_features"])
    for i in active:
        native = cv2.detail.computeImageFeatures2(detector, work[i], masks[i])
        keys = list(native.keypoints)[: p["camera_features"]]
        descriptors = native.descriptors.get()
        if descriptors is None or len(keys) < 4:
            audit["features"].append({"index": i, "fit": 0, "validation": 0})
            continue
        descriptors = descriptors[: len(keys)]
        reserved = partition(keys, p["holdout_fraction"], p["holdout_seed"])
        fit_indices, validation_indices = np.flatnonzero(~reserved), np.flatnonzero(reserved)
        audit["features"].append(
            {"index": i, "fit": len(fit_indices), "validation": len(validation_indices)}
        )
        held[i] = ([keys[j] for j in validation_indices], descriptors[reserved].copy())
        if len(fit_indices) < 4:
            continue
        native.keypoints = tuple(keys[j] for j in fit_indices)
        native.descriptors = cv2.UMat(descriptors[~reserved].copy())
        native.img_idx = len(usable)
        training[i] = native
        usable.append(i)

    def finish(status, component=(), cameras=()):
        return status, list(component), list(cameras), held, audit

    if len(usable) < 2 or anchor not in usable:
        return finish("insufficient_training_features")
    comparisons = sum(
        len(training[a].keypoints) * len(training[b].keypoints) for a, b in combinations(usable, 2)
    )
    if comparisons > p["maximum_verification_pairs"]:
        raise ValidationError(
            "Native fitting descriptor budget exceeded", "maximum_verification_pairs"
        )
    audit["training_descriptor_pairs"] = comparisons
    matcher = cv2.detail.BestOf2NearestMatcher(False, p["native_match_confidence"])
    try:
        matches = matcher.apply2([training[i] for i in usable])
    finally:
        matcher.collectGarbage()
    n = len(usable)
    if len(matches) != n * n:
        raise ValidationError("Native fitting returned an invalid pairwise matrix")
    edges = []
    for a, b in combinations(range(n), 2):
        match = matches[a * n + b]
        audit["training_pairs"].append(
            {
                "source_index": usable[a],
                "target_index": usable[b],
                "matches": len(match.matches),
                "inliers": match.num_inliers,
                "confidence": float(match.confidence),
            }
        )
        if match.confidence > p["native_confidence"] and match.num_inliers >= 4:
            edges.append((usable[a], usable[b]))
    component = {anchor}
    while True:
        before = len(component)
        for a, b in edges:
            if a in component or b in component:
                component.update((a, b))
        if len(component) == before:
            break
    component = sorted(component)
    if len(component) < 2:
        return finish("training_views_disconnected")
    indices = [usable.index(i) for i in component]
    features = [training[i] for i in component]
    for i, features_i in enumerate(features):
        features_i.img_idx = i
    selected = [
        subset_matches(
            matches[a * n + b], i if i != j else -1, j if i != j else -1, p["native_confidence"]
        )
        for i, a in enumerate(indices)
        for j, b in enumerate(indices)
    ]
    initial = None
    if priors:
        initial = []
        for i in component:
            K = work_priors[i]
            camera = cv2.detail.CameraParams()
            camera.focal, camera.aspect = K[0, 0], K[1, 1] / K[0, 0]
            camera.ppx, camera.ppy = K[:2, 2]
            camera.R = np.eye(3, dtype=np.float64)
            initial.append(camera)
    ok, cameras = cv2.detail.HomographyBasedEstimator(bool(priors)).apply(
        features, selected, initial
    )
    if not ok:
        return finish("homography_estimation_failed")
    for camera in cameras:
        intrinsics(camera.K())
        R = np.asarray(camera.R, np.float64)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            raise ValidationError("Native initial camera rotation is invalid")
        u, _, v = np.linalg.svd(R)
        proper = u @ np.diag([1, 1, np.linalg.det(u @ v)]) @ v
        camera.R = proper.astype(np.float32)
    audit["initial_intrinsics_work"] = [c.K().tolist() for c in cameras]
    kind = p["bundle_adjustment"]
    if kind != "none":
        adjuster = (
            cv2.detail.BundleAdjusterRay() if kind == "ray" else cv2.detail.BundleAdjusterReproj()
        )
        mask = np.zeros((3, 3), np.uint8)
        mask[0, 0], mask[1, 1] = p["refine_focal"], p["refine_aspect"]
        mask[0, 2] = mask[1, 2] = p["refine_principal_point"]
        adjuster.setRefinementMask(mask)
        adjuster.setConfThresh(p["native_confidence"])
        adjuster.setTermCriteria(
            (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                p["adjustment_iterations"],
                p["adjustment_epsilon"],
            )
        )
        audit["refinement_mask"] = mask.tolist()
        ok, cameras = adjuster.apply(features, selected, cameras)
        if not ok:
            return finish("camera_adjustment_failed")
    audit["fitted_intrinsics_work"] = [c.K().tolist() for c in cameras]
    return finish("ok", component, cameras)
