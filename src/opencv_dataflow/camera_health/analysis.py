"""Fixed-reference motion, foreground rejection and visible-region health metrics."""

import math

import cv2
import numpy as np

from ..selection import rasterize
from .model import focus, gray, union


def _tracking_image(image):
    # Local contrast avoids a moving dark/bright part changing the normalization
    # of every stationary pixel. Health photometry still uses original samples.
    pixels = np.rint(image.clip(0, 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pixels)


def _fit(a, b, minimum, p):
    if len(a) < minimum:
        return None, None
    matrix, inliers = cv2.estimateAffinePartial2D(
        a,
        b,
        method=cv2.RANSAC,
        ransacReprojThreshold=p["ransac_error"],
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or not np.isfinite(matrix).all():
        return None, None
    keep = inliers.ravel() != 0
    if keep.sum() < minimum or keep.mean() < p["minimum_inlier_fraction"]:
        return None, keep
    return matrix, keep


def foreground(reference, current, matrix, stable, external, p):
    h, w = reference.shape[:2]
    aligned = cv2.warpAffine(current, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    if reference.ndim == 3 and aligned.ndim == 2:
        aligned = aligned[..., None]
    visible = (
        cv2.warpAffine(
            np.ones((h, w), np.uint8),
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        != 0
    )
    known = (
        cv2.warpAffine(
            external.astype(np.uint8),
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        != 0
    )
    changed = np.zeros((h, w), bool)
    selected = stable & visible & ~known
    if p["auto_foreground"] and selected.sum() >= 16:
        a, b = aligned[selected], reference[selected]
        qa, qb = np.percentile(a, [10, 50, 90], axis=0), np.percentile(b, [10, 50, 90], axis=0)
        gain = np.clip((qb[2] - qb[0]) / np.maximum(qa[2] - qa[0], 1e-6), 0.25, 4)
        adjusted = gain * (aligned - qa[1]) + qb[1]
        difference = np.abs(adjusted - reference)
        if difference.ndim == 3:
            difference = difference.max(axis=2)
        raw = (difference > p["foreground_threshold"]).astype(np.uint8)
        raw &= selected.astype(np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        sizes = stats[:, cv2.CC_STAT_AREA]
        accepted = sizes >= p["foreground_min_area"]
        accepted[0] = False
        changed = accepted[labels] if count > 1 else changed
        radius = int(p["foreground_dilation"])
        if radius:
            changed = (
                cv2.dilate(changed.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) != 0
            )
    return changed | known, visible


def background_position(reference, current, stable, external, p):
    a, b = _tracking_image(reference), _tracking_image(current)
    points = cv2.goodFeaturesToTrack(
        a,
        maxCorners=int(p["max_features"]),
        qualityLevel=p["feature_quality"],
        minDistance=5,
        mask=stable.astype(np.uint8) * 255,
    )
    if points is None or len(points) < p["minimum_tracks"]:
        return None, {"reason": "too_few_reference_features", "tracks": 0}
    options = {"winSize": (int(p["tracking_window"]),) * 2, "maxLevel": int(p["pyramid_levels"])}
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **options)
    if forward is None:
        return None, {"reason": "tracking_failed", "tracks": 0}
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, forward, None, **options)
    if back is None:
        return None, {"reason": "reverse_tracking_failed", "tracks": 0}
    left, right = points.reshape(-1, 2), forward.reshape(-1, 2)
    good = (ok.ravel() != 0) & (back_ok.ravel() != 0)
    good &= np.isfinite(right).all(axis=1) & (
        np.linalg.norm(back.reshape(-1, 2) - left, axis=1) <= p["forward_backward_error"]
    )
    h, w = reference.shape
    rounded = np.rint(
        np.clip(np.nan_to_num(right, nan=-1, posinf=-1, neginf=-1), [-1, -1], [w, h])
    ).astype(int)
    inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < w) & (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
    good &= inside
    sample = np.clip(rounded, [0, 0], [w - 1, h - 1])
    good &= ~external[sample[:, 1], sample[:, 0]]
    left, right = left[good], right[good]
    first, _ = _fit(left, right, int(p["minimum_tracks"]), p)
    if first is None:
        return None, {"reason": "initial_background_fit_failed", "tracks": len(left)}
    changed, visible = foreground(reference, current, first, stable, external, p)
    sample = np.rint(left).astype(int)
    clean = ~changed[sample[:, 1], sample[:, 0]] & visible[sample[:, 1], sample[:, 0]]
    left, right = left[clean], right[clean]
    final, keep = _fit(left, right, int(p["minimum_tracks"]), p)
    report = {
        "tracks": len(left),
        "foreground_removed_tracks": int((~clean).sum()),
        "reason": "background_fit_failed" if final is None else "measured",
    }
    if final is not None:
        support = float(cv2.contourArea(cv2.convexHull(left[keep]))) / max(
            1, np.count_nonzero(stable)
        )
        report.update(
            inliers=int(keep.sum()), inlier_fraction=float(keep.mean()), spatial_support=support
        )
        if support < p["minimum_track_coverage"]:
            return None, {**report, "reason": "tracks_are_too_concentrated"}
    return final, report


def _subpixel(response, x, y):
    result = [float(x), float(y)]
    for axis in (0, 1):
        i = x if axis == 0 else y
        size = response.shape[1] if axis == 0 else response.shape[0]
        if 0 < i < size - 1:
            values = response[y, x - 1 : x + 2] if axis == 0 else response[y - 1 : y + 2, x]
            divisor = values[0] - 2 * values[1] + values[2]
            if np.isfinite(values).all() and divisor < -1e-8:
                result[axis] += float(np.clip(0.5 * (values[0] - values[2]) / divisor, -1, 1))
    return result


def fiducial_position(reference, current, rows, external, p):
    h, w = reference.shape
    matches, source, target = [], [], []
    margin = int(p["fiducial_search_margin"])
    for row in rows:
        mask = rasterize(row["region"], reference.shape)
        x, y, width, height = cv2.boundingRect(mask)
        template, template_mask = (
            reference[y : y + height, x : x + width],
            mask[y : y + height, x : x + width],
        )
        if float(template[template_mask != 0].std()) < 1e-4:
            matches.append({"name": row["name"], "found": False, "reason": "flat_template"})
            continue
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(w, x + width + margin), min(h, y + height + margin)
        response = cv2.matchTemplate(
            current[top:bottom, left:right].astype(np.float32),
            template.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
            mask=template_mask,
        )
        response = np.where(np.isfinite(response), response, -1).clip(-1, 1)
        _, score, _, (px, py) = cv2.minMaxLoc(response)
        other = response.copy()
        other[
            max(0, py - height // 2) : py + height // 2 + 1,
            max(0, px - width // 2) : px + width // 2 + 1,
        ] = -1
        separation = float(score - other.max())
        location = _subpixel(response, px, py)
        known = external[top + py : top + py + height, left + px : left + px + width]
        found = (
            score >= p["fiducial_min_score"]
            and separation >= p["fiducial_peak_margin"]
            and not np.any(known & (template_mask != 0))
        )
        ys, xs = np.nonzero(template_mask)
        anchor = [x + float(xs.mean()), y + float(ys.mean())]
        observed = [left + location[0] + float(xs.mean()), top + location[1] + float(ys.mean())]
        matches.append(
            {
                "name": row["name"],
                "found": bool(found),
                "score": float(score),
                "peak_margin": separation,
                "reference": anchor,
                "current": observed,
            }
        )
        if found:
            source.append(anchor)
            target.append(observed)
    report = {"fiducials": matches, "tracks": len(source), "reason": "measured"}
    if not rows or len(source) != len(rows):
        return None, {**report, "reason": "fiducial_missing_or_ambiguous"}
    if len(source) == 1:
        delta = np.asarray(target[0]) - source[0]
        report["translation_only"] = True
        return np.array([[1, 0, delta[0]], [0, 1, delta[1]]], float), report
    matrix, keep = _fit(np.asarray(source, np.float32), np.asarray(target, np.float32), 2, p)
    if matrix is None or not keep.all():
        return None, {**report, "reason": "inconsistent_fiducials"}
    return matrix, report


def level(value, warning, alarm):
    return "alarm" if value >= alarm else "warning" if value >= warning else "healthy"


def assess(model, image, external, p):
    info, regions = model.info, model.info["regions"]
    reference_gray, current_gray = (
        gray(model.image, info["encoding"]),
        gray(image, info["encoding"]),
    )
    stable = model.stable != 0
    position_mask = stable & union(regions, "position", stable.shape, whole_if_empty=True)
    matrix, position = np.array([[1, 0, 0], [0, 1, 0]], float), {"state": "disabled"}
    checks = []
    if p["check_position"]:
        fn = fiducial_position if p["position_method"] == "fiducials" else background_position
        matrix, position = fn(
            reference_gray,
            current_gray,
            regions["fiducials"] if p["position_method"] == "fiducials" else position_mask,
            external,
            p,
        )
        if matrix is None:
            position["state"] = "unavailable"
            checks.append(
                {
                    "check": "position",
                    "name": "Position",
                    "state": "alarm",
                    "reason": position["reason"],
                    "value": None,
                }
            )
        else:
            center = np.array([(image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2])
            displacement = float(np.linalg.norm(matrix @ np.r_[center, 1] - center))
            rotation = (
                None
                if position.get("translation_only")
                else math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            )
            scale = None if rotation is None else (math.hypot(matrix[0, 0], matrix[1, 0]) - 1) * 100
            for name, value, key in (
                ("Position", displacement, "position"),
                ("Rotation", rotation, "rotation"),
                ("Scale", scale, "scale"),
            ):
                if value is not None:
                    checks.append(
                        {
                            "check": "position",
                            "name": name,
                            "value": value,
                            "state": level(abs(value), p[f"{key}_warning"], p[f"{key}_alarm"]),
                        }
                    )
            position.update(
                state="measured",
                displacement_px=displacement,
                rotation_deg=rotation,
                scale_percent=scale,
            )
    usable_matrix = matrix if matrix is not None else np.array([[1, 0, 0], [0, 1, 0]], float)
    changed, visible = foreground(model.image, image, usable_matrix, stable, external, p)
    h, w = stable.shape
    changed_current = (
        cv2.warpAffine(changed.astype(np.uint8), usable_matrix, (w, h), flags=cv2.INTER_NEAREST)
        != 0
    )
    current_to_reference = cv2.invertAffineTransform(usable_matrix)
    for role in ("lighting", "focus"):
        if not p[f"check_{role}"]:
            continue
        if not regions[role]:
            checks.append(
                {
                    "check": role,
                    "name": role.title(),
                    "state": "alarm",
                    "reason": "no_trained_regions",
                    "value": None,
                }
            )
        for row in regions[role]:
            reference_mask = rasterize(row["region"], stable.shape) != 0
            common = reference_mask & stable & visible & ~changed
            coverage = int(common.sum()) / max(1, int(reference_mask.sum()))
            current_mask = (
                cv2.warpAffine(
                    common.astype(np.uint8), usable_matrix, (w, h), flags=cv2.INTER_NEAREST
                )
                != 0
            )
            # Reintersect to account for nearest-neighbor holes at rotated/scaled mask boundaries.
            current_mask &= ~external
            common &= (
                cv2.warpAffine(
                    current_mask.astype(np.uint8),
                    current_to_reference,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                )
                != 0
            )
            coverage = int(common.sum()) / max(1, int(reference_mask.sum()))
            baseline = next(r for r in info["baselines"][role] if r["name"] == row["name"])
            result = {
                "check": role,
                "name": row["name"],
                "coverage": coverage,
                "historical": baseline["statistics"],
                "position_compensated": matrix is not None and p["check_position"],
                "value": None,
                "state": "alarm",
            }
            if coverage < p["minimum_region_coverage"] or min(common.sum(), current_mask.sum()) < 8:
                result["reason"] = "insufficient_visible_region"
            elif role == "lighting":
                old, now = model.image[common].mean(axis=0), image[current_mask].mean(axis=0)
                change = (now - old) / np.maximum(np.abs(old), p["lighting_floor"]) * 100
                value = float(np.abs(change).max())
                result.update(
                    value=value,
                    channel_change_percent=change.tolist(),
                    reference_mean=old.tolist(),
                    current_mean=now.tolist(),
                    state=level(value, p["lighting_warning"], p["lighting_alarm"]),
                )
            else:
                old = focus(reference_gray, common, info["settings"])
                now = focus(current_gray, current_mask, info["settings"])
                if old is None or old <= 1e-10 or now is None:
                    result["reason"] = "insufficient_reference_or_current_texture"
                else:
                    loss = max(0.0, (1 - now / old) * 100)
                    result.update(
                        value=loss,
                        reference_sharpness=old,
                        current_sharpness=now,
                        state=level(loss, p["focus_warning"], p["focus_alarm"]),
                    )
            checks.append(result)
    state = max(
        (r["state"] for r in checks),
        key={"healthy": 0, "warning": 1, "alarm": 2}.get,
        default="healthy",
    )
    return {
        "model": "camera_health_report_v1",
        "state": state,
        "checks": checks,
        "position": position,
        "transform": matrix.tolist() if matrix is not None else None,
        "reference_to_current": True,
        "capture_count": info["capture_count"],
        "intensity_channels": ["Gray"]
        if image.shape[2] == 1
        else list(info["encoding"][:3].upper()),
        "mean_units": "full_scale_fraction",
        "focus_method": info["settings"]["focus_method"],
        "focus_is_contrast_normalized": info["settings"]["normalize_focus"],
        "foreground_pixels": int(changed_current.sum()),
        "reference_is_frozen": True,
    }, changed_current
