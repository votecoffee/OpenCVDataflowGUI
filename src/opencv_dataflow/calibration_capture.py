"""Qualify pinhole calibration coverage, repeatability and held-out accuracy."""

import hashlib
import re

import cv2
import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import require_pinhole
from .parameter_validation import NumericRange, ValidationError

_GRID = (8, 6)
_DIGEST = re.compile(r"[0-9a-f]{64}")


def observation_id(observed):
    """Identify exact pixel observations independently of point enumeration."""
    points = np.asarray(observed, dtype="<f4").reshape(-1, 2)
    order = np.lexsort((points[:, 1], points[:, 0]))
    return hashlib.sha256(points[order]).hexdigest()


def optics_id(matrix, distortion, size):
    values = np.concatenate((size, np.ravel(matrix), np.ravel(distortion))).astype("<f8")
    return hashlib.sha256(values).hexdigest()


def capture_geometry(image_views, object_views, image_size, views):
    grid = np.asarray(_GRID)
    seen = np.zeros(_GRID[::-1], dtype=int)
    outside = 0
    # Use one common board point even when the corner subset differs by view.
    lower = np.min(
        [np.asarray(points).reshape(-1, 3).min(axis=0) for points in object_views], axis=0
    )
    upper = np.max(
        [np.asarray(points).reshape(-1, 3).max(axis=0) for points in object_views], axis=0
    )
    center = (lower.astype(np.float64) + upper) / 2
    depths, tilts = [], []
    for observed, view in zip(image_views, views, strict=True):
        points = np.asarray(observed).reshape(-1, 2)
        uv = (points + 0.5) / image_size
        inside = ((uv >= 0) & (uv < 1)).all(axis=1)
        outside += int((~inside).sum())
        x, y = np.unique((uv[inside] * grid).astype(int), axis=0).T
        seen[y, x] += 1
        r = cv2.Rodrigues(np.asarray(view["rotation_vector_radians"]))[0]
        normal = r[:, 2] if r[2, 2] >= 0 else -r[:, 2]
        angles = np.degrees(np.arctan2(normal[:2], normal[2]))
        depth = float(r[2] @ center + view["translation_vector"][2])
        view.update(
            observation_sha256=observation_id(points),
            image_center_px=points.mean(axis=0).tolist(),
            normal_tilt_xy_degrees=angles.tolist(),
            board_center_depth=depth,
        )
        tilts.append(angles)
        depths.append(depth)
    return {
        "grid_size": list(_GRID),
        "views_per_cell": seen.tolist(),
        "tilt_span_xy_degrees": np.ptp(tilts, axis=0).tolist(),
        "depth_span_fraction": float(np.ptp(depths) / np.median(depths))
        if min(depths) > 0
        else None,
        "outside_image_points": outside,
    }


def _fit_report(record):
    if not isinstance(record, dict) or record.get("model") != "pinhole_fit_report_v1":
        raise ValidationError("Connect a pinhole lens fit report")
    try:
        optics = record["capture_optics"]
        matrix, distortion, size = require_pinhole(optics)
        capture = record["capture_geometry"]
        counts = np.asarray(capture["views_per_cell"])
        tilts = np.asarray(capture["tilt_span_xy_degrees"], dtype=np.float64)
        depth = capture["depth_span_fraction"]
        outside = capture["outside_image_points"]
        ids = [view["observation_sha256"] for view in record["views"]]
        if (
            len(ids) < 3
            or any(not isinstance(key, str) or not _DIGEST.fullmatch(key) for key in ids)
            or counts.dtype.kind not in "ui"
            or counts.shape != _GRID[::-1]
            or (counts < 0).any()
            or (counts > len(ids)).any()
            or capture["grid_size"] != list(_GRID)
            or tilts.shape != (2,)
            or not np.isfinite(tilts).all()
            or np.any((tilts < 0) | (tilts > 180))
            or (depth is not None and (not np.isfinite(depth) or depth < 0))
            or type(outside) is not int
            or outside < 0
        ):
            raise ValueError
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise ValidationError("Use a current lens fit report with valid capture geometry") from exc
    covered = counts > 0
    edge = np.ones_like(covered)
    edge[1:-1, 1:-1] = False
    return (
        {
            "views": len(ids),
            "fit_passed": record.get("passed") is True,
            "occupied_cell_fraction": float(covered.mean()),
            "occupied_edge_fraction": float(covered[edge].mean()),
            "grid_size": list(_GRID),
            "views_per_cell": counts.tolist(),
            "tilt_span_xy_degrees": tilts.tolist(),
            "depth_span_fraction": depth,
            "outside_image_points": outside,
        },
        set(ids),
        (matrix, distortion, size),
    )


def _coverage_checks(capture, p, label, reasons):
    for failed, code, message in (
        (not capture["fit_passed"], "fit_failed", "failed the lens fit checks"),
        (capture["views"] < p["minimum_views"], "view_count", "too few views"),
        (
            capture["occupied_cell_fraction"] < p["minimum_grid_fraction"],
            "image_coverage",
            "too few image cells contain observed points",
        ),
        (
            capture["occupied_edge_fraction"] < p["minimum_edge_fraction"],
            "edge_coverage",
            "too few border cells contain observed points",
        ),
        (
            min(capture["tilt_span_xy_degrees"]) < p["minimum_tilt_span_degrees"],
            "tilt_span",
            "insufficient target-normal variation in X or Y",
        ),
        (
            capture["depth_span_fraction"] is None
            or capture["depth_span_fraction"] < p["minimum_depth_span_fraction"],
            "depth_span",
            "insufficient positive board depth variation",
        ),
        (
            capture["outside_image_points"] > 0,
            "outside_points",
            "observations outside the image",
        ),
    ):
        if failed:
            reasons.append(
                {
                    "code": f"calibration_capture_{label}_{code}",
                    "message": f"{label.title()} set: {message}",
                }
            )


def _repeat_comparison(primary, repeat):
    k, d, size = primary
    rk, rd, rsize = repeat
    if size != rsize:
        raise ValidationError("Repeat fit must use the same image dimensions")
    if d.shape != rd.shape:
        raise ValidationError("Repeat fit must use the same distortion coefficient layout")
    # Use identical nominal camera rays. Sampling their forward projection includes
    # both the intrinsic matrix and distortion differences without an inverse fit.
    u, v = np.meshgrid(np.linspace(0, size[0] - 1, 9), np.linspace(0, size[1] - 1, 7))
    rays = np.column_stack((u.ravel(), v.ravel(), np.ones(u.size))) @ np.linalg.inv(k).T
    zero = np.zeros(3)
    baseline = cv2.projectPoints(rays, zero, zero, k, d)[0].reshape(-1, 2)
    revisit = cv2.projectPoints(rays, zero, zero, rk, rd)[0].reshape(-1, 2)
    error = np.linalg.norm(baseline - revisit, axis=1)
    if not np.isfinite(error).all():
        raise ValidationError("Repeat-fit projection produced nonfinite pixel drift")
    return {
        "samples": len(error),
        "rms_drift_pixels": float(np.sqrt(np.mean(error**2))),
        "max_drift_pixels": float(error.max()),
        "focal_shift_xy_percent": (100 * (rk.diagonal()[:2] / k.diagonal()[:2] - 1)).tolist(),
        "principal_point_shift_xy_pixels": (rk[:2, 2] - k[:2, 2]).tolist(),
        "distortion_coefficient_deltas": (rd - d).tolist(),
    }


def _check_holdout(record, optics, fit_ids, limit, label, reasons):
    def fail(code, message):
        reasons.append({"code": f"calibration_capture_{label}_{code}", "message": message})

    if not isinstance(record, dict) or record.get("model") != "pinhole_validation_v1":
        raise ValidationError("Connect a pinhole lens validation report")
    if record.get("status") != "measured":
        fail("holdout_unmeasured", "Held-out target validation did not measure residuals")
        return {"status": record.get("status"), "rms_pixels": None}
    if record.get("optics_sha256") != optics_id(*optics):
        raise ValidationError("Held-out validation must evaluate its connected fit optics")
    key, rms = record.get("observation_sha256"), record.get("rms_pixels")
    if (
        not isinstance(key, str)
        or not _DIGEST.fullmatch(key)
        or (type(rms) not in (int, float) or not np.isfinite(rms) or rms < 0)
    ):
        raise ValidationError(
            "Use a current lens validation report with measured pixel observations"
        )
    if key in fit_ids:
        fail("holdout_reused", "Held-out target points duplicate a fitted view")
    if rms > limit:
        fail("holdout_rms", f"Held-out RMS {rms:.4g} px exceeds {limit:g} px")
    elif record.get("passed") is not True:
        fail("holdout_validation_failed", "Held-out target failed the validation block checks")
    return {"status": "measured", "observation_sha256": key, "rms_pixels": rms}


def _coverage_plot(counts):
    cells = np.asarray(counts, dtype=np.float64)
    levels = np.rint(255 * cells / max(1, cells.max())).astype(np.uint8)
    plot = cv2.applyColorMap(
        cv2.resize(levels, None, fx=52, fy=52, interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_VIRIDIS,
    )
    for (j, i), value in np.ndenumerate(cells):
        origin = (i * 52 + 17, j * 52 + 32)
        cv2.putText(plot, str(int(value)), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(
            plot, str(int(value)), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return plot


class AuditCalibrationCapture(Block):
    type_id, title = "calibration.audit_capture", "Audit calibration capture set"
    block_type = category = "Calibration"
    subtype = "Lens distortion"
    inputs = (
        Port("fit_report", "data"),
        Port("repeat_fit_report", "data", required=False, required_when_connected=True),
        Port("holdout", "data", required=False, required_when_connected=True),
        Port("repeat_holdout", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("report", "data"),)
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "minimum_views": 6,
        "minimum_grid_fraction": 0.50,
        "minimum_edge_fraction": 0.40,
        "minimum_tilt_span_degrees": 15.0,
        "minimum_depth_span_fraction": 0.15,
        "maximum_repeat_drift_pixels": 2.0,
        "maximum_holdout_rms_pixels": 0.5,
    }
    parameter_ranges = {
        "minimum_views": NumericRange(3, whole=True),
        "minimum_grid_fraction": NumericRange(0, 1),
        "minimum_edge_fraction": NumericRange(0, 1),
        "minimum_tilt_span_degrees": NumericRange(0, 180),
        "minimum_depth_span_fraction": NumericRange(0),
        "maximum_repeat_drift_pixels": NumericRange(0),
        "maximum_holdout_rms_pixels": NumericRange(0),
    }
    parameter_help = {
        "minimum_views": "Minimum number of poses in each connected capture set.",
        "minimum_grid_fraction": "Minimum occupancy of the 8 × 6 image grid.",
        "minimum_edge_fraction": "Minimum occupancy of the outer ring of the image grid.",
        "minimum_tilt_span_degrees": "Minimum span in each of the camera X/Y normal tilt angles.",
        "minimum_depth_span_fraction": "Board-center depth range divided by its median.",
        "maximum_repeat_drift_pixels": "Maximum projection drift over 63 sampled camera rays.",
        "maximum_holdout_rms_pixels": "Maximum lens validation RMS on the held-out image.",
    }
    input_help = {
        "fit_report": "Fit report from Calibrate lens.",
        "repeat_fit_report": "Optional second fit from independently acquired views.",
        "holdout": "Optional Validate lens calibration report for the primary fit.",
        "repeat_holdout": "Optional validation of the same held-out view with the repeat fit.",
    }

    def process(self, inputs):
        p = {key: float(self.parameters[key]) for key in self.defaults}
        primary, fit_ids, optics = _fit_report(inputs["fit_report"])
        report = {"model": "pinhole_capture_audit_v1", "primary": primary, "thresholds": p}
        reasons, collected, repeat_optics = [], set(fit_ids), None
        _coverage_checks(primary, p, "primary", reasons)
        if "repeat_fit_report" in inputs:
            repeat, repeat_ids, repeat_optics = _fit_report(inputs["repeat_fit_report"])
            comparison = _repeat_comparison(optics, repeat_optics)
            report.update(repeat=repeat, comparison=comparison)
            _coverage_checks(repeat, p, "repeat", reasons)
            comparison["reused_views"] = len(fit_ids & repeat_ids)
            collected.update(repeat_ids)
            if comparison["reused_views"]:
                reasons.append(
                    {
                        "code": "calibration_capture_repeat_reused",
                        "message": "The repeat fit contains views from the primary fit",
                    }
                )
            drift = comparison["max_drift_pixels"]
            if drift > p["maximum_repeat_drift_pixels"]:
                reasons.append(
                    {
                        "code": "calibration_capture_repeat_drift",
                        "message": f"Repeat projection drift is {drift:.4g} px",
                    }
                )
        elif "repeat_holdout" in inputs:
            raise ValidationError("Repeat holdout requires a repeat fit report")
        for key, source in (("holdout", optics), ("repeat_holdout", repeat_optics)):
            if key in inputs:
                report[key] = _check_holdout(
                    inputs[key],
                    source,
                    collected,
                    p["maximum_holdout_rms_pixels"],
                    "primary" if key == "holdout" else "repeat",
                    reasons,
                )
        if "holdout" in report and "repeat_holdout" in report:
            first = report["holdout"].get("observation_sha256")
            second = report["repeat_holdout"].get("observation_sha256")
            if first and second and first != second:
                reasons.append(
                    {
                        "code": "calibration_capture_holdout_mismatch",
                        "message": "The two validation reports describe different held-out points",
                    }
                )
        report["passed"] = not reasons
        graphics = {}
        if self.generate_graphics:
            for key in ("primary", "repeat"):
                if key in report:
                    graphics[f"{key.title()} capture coverage"] = _coverage_plot(
                        report[key]["views_per_cell"]
                    )
        return Result(
            {"report": report},
            graphics=graphics,
            passed=not reasons,
            fail_reasons=reasons,
            message=f"{primary['views']} views · {primary['occupied_cell_fraction']:.0%} coverage"
            + (f" · {len(reasons)} failed checks" if reasons else ""),
        )
