"""Detector cadence and bounded local image evidence between full-frame refreshes."""

import math
from dataclasses import replace

import cv2
import numpy as np

from .core import Port
from .image_tracking import TrackImageParts, compatible_frame
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket
from .poses import local_offset
from .processing import gray_image
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key


class DetectionCadence(SequenceBlock):
    type_id, title = "flow.detection_cadence", "Detector refresh cadence"
    category, block_type, subtype = "Control flow", "Control flow", "Scheduling"
    outcome_passthrough = True
    process_inputs_readonly = True
    inputs = (
        SequenceBlock.inputs[0],
        Port("reset", "data", required=False),
        Port("stream", "data", required=False),
        Port("force", "boolean", required=False),
    )
    data_outputs = (Port("image"), Port("refresh", "boolean"), Port("report", "data"))
    defaults = {**SEQUENCE_DEFAULTS, "warmup": 1, "period": 5}
    parameter_choices = SEQUENCE_CHOICES
    parameter_ranges = {"period": NumericRange(1, 10000, whole=True)}
    parameter_help = {
        **SEQUENCE_HELP,
        "period": (
            "Emit detector image on the first acquisition and every N acquisitions after it."
        ),
        "warmup": "Readiness only; leave one so the first refresh is delivered.",
    }
    input_help = {"force": "Force an extra refresh and restart cadence from this acquisition."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["warmup"] != 1:
            raise ValidationError("Cadence warmup must be one", "warmup")

    def step(self, inputs, state, mask):
        force = inputs.get("force", False)
        if type(force) is not bool:
            raise ValidationError("Force refresh must be boolean")
        index = state["activations"]
        refresh = force or index >= state.get("next_refresh", 0)
        if refresh:
            state["next_refresh"] = index + int(self.parameters["period"])
        state["calls"] = state.get("calls", 0) + int(refresh)
        return SequenceStep(
            {
                "image": inputs["image"] if refresh else NO_VALUE,
                "refresh": refresh,
                "report": {
                    "frame_index": index,
                    "refresh": refresh,
                    "detector_calls": state["calls"],
                    "next_refresh": state["next_refresh"],
                },
            }
        )


def track_corners(track):
    mean = track["mean"]
    angle = mean[4] if track["angle_known"] else 0
    scale = np.exp(mean[2:4]) / track["reference_size"] if track["size_known"] else np.ones(2)
    matrix = local_offset(*mean[:2], angle) @ np.diag([*scale, 1])
    return (np.c_[track["model_box"], np.ones(4)] @ matrix.T)[:, :2]


def local_motion(previous, current, points, p, mask):
    """Forward/backward LK and a supported similarity fit; no native model state survives."""
    if len(points) < p["local_min_points"]:
        return None, {"reason": "insufficient_points", "points": len(points)}
    if len(points) * p["local_window"] ** 2 * (p["local_levels"] + 1) > p["local_work_limit"]:
        raise ValidationError(
            "Local point/window/level work exceeds configured limit", "local_work_limit"
        )
    options = {
        "winSize": (p["local_window"], p["local_window"]),
        "maxLevel": p["local_levels"],
        "criteria": (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            p["local_iterations"],
            p["local_epsilon"],
        ),
        "minEigThreshold": p["local_min_eigenvalue"],
        "flags": cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
        if p["local_error_metric"] == "min_eigenvalue"
        else 0,
    }
    found, valid, errors = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if found is None or valid is None or errors is None:
        return None, {"reason": "flow_failed"}
    original, target = points.reshape(-1, 2), found.reshape(-1, 2)
    good = valid.ravel().astype(bool) & np.isfinite(target).all(axis=1)
    good &= np.isfinite(errors.ravel())
    good &= (
        errors.ravel() >= p["local_min_eigenvalue"]
        if p["local_error_metric"] == "min_eigenvalue"
        else errors.ravel() <= p["local_max_error"]
    )
    # Never feed nonfinite or out-of-image coordinates into backward native work.
    good &= (target[:, 0] >= 0) & (target[:, 0] < current.shape[1])
    good &= (target[:, 1] >= 0) & (target[:, 1] < current.shape[0])
    indices = np.flatnonzero(good)
    if p["local_fb_tolerance"] and len(indices):
        back, back_valid, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, found[indices].copy(), None, **options
        )
        accepted = np.zeros(len(indices), bool)
        if back is not None and back_valid is not None:
            accepted = back_valid.ravel().astype(bool) & np.isfinite(back.reshape(-1, 2)).all(
                axis=1
            )
            accepted &= (
                np.linalg.norm(back.reshape(-1, 2) - original[indices], axis=1)
                <= p["local_fb_tolerance"]
            )
        good[indices] &= accepted
    if mask is not None:
        for index in np.flatnonzero(good):
            x, y = np.rint(target[index]).astype(int)
            good[index] &= 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])
    a, b = original[good], target[good]
    if len(a) < p["local_min_points"] or min(np.ptp(a, axis=0)) < p["local_min_span"]:
        return None, {"reason": "insufficient_support", "points": len(a)}
    matrix, inliers = cv2.estimateAffinePartial2D(
        a,
        b,
        method=cv2.RANSAC if p["fit_method"] == "ransac" else cv2.LMEDS,
        ransacReprojThreshold=p["fit_threshold"],
        maxIters=p["fit_iterations"],
        confidence=p["fit_confidence"],
        refineIters=p["fit_refine_iterations"],
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None, {"reason": "fit_failed"}
    inliers = inliers.ravel().astype(bool)
    ratio = float(np.count_nonzero(inliers) / len(points))
    scale = float(np.hypot(*matrix[:, 0]))
    angle = -math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    residual = np.linalg.norm(np.c_[a, np.ones(len(a))] @ matrix.T - b, axis=1)
    if (
        np.count_nonzero(inliers) < p["local_min_points"]
        or ratio < p["local_min_inliers"]
        or min(np.ptp(a[inliers], axis=0)) < p["local_min_span"]
        or not 1 / p["local_max_scale"] <= scale <= p["local_max_scale"]
        or abs(angle) > p["local_max_angle"]
        or np.max(residual[inliers]) > p["fit_threshold"]
    ):
        return None, {"reason": "fit_gate", "inlier_fraction": ratio}
    return (matrix, b[inliers].astype(np.float32).reshape(-1, 1, 2)), {
        "reason": "accepted",
        "inlier_fraction": ratio,
        "scale": scale,
        "angle_degrees": angle,
        "rms": float(np.sqrt(np.mean(residual[inliers] ** 2))),
    }


class RefreshedImageTracks(TrackImageParts):
    type_id, title = "sequence.track_refreshed", "Track with periodic detections"
    inputs = tuple(
        Port("detections", "data", required=False) if p.name == "detections" else p
        for p in TrackImageParts.inputs
    ) + (Port("refresh", "boolean"),)
    defaults = {
        **TrackImageParts.defaults,
        "maximum_detector_age_frames": 15,
        "maximum_detector_age_ms": 2000.0,
        "local_max_points": 48,
        "local_quality": 0.01,
        "local_min_distance": 3.0,
        "local_block_size": 5,
        "local_gradient_size": "3",
        "local_harris": False,
        "local_harris_k": 0.04,
        "local_window": 21,
        "local_levels": 3,
        "local_iterations": 30,
        "local_epsilon": 0.01,
        "local_min_eigenvalue": 0.0001,
        "local_max_error": 20.0,
        "local_error_metric": "patch",
        "fit_method": "ransac",
        "local_fb_tolerance": 1.0,
        "local_min_points": 4,
        "local_min_inliers": 0.6,
        "local_min_span": 3.0,
        "local_max_shift": 30.0,
        "local_max_scale": 1.2,
        "local_max_angle": 20.0,
        "local_work_limit": 32_000_000,
        "fit_threshold": 2.0,
        "fit_iterations": 200,
        "fit_confidence": 0.99,
        "fit_refine_iterations": 10,
    }
    parameter_choices = {
        **TrackImageParts.parameter_choices,
        "local_gradient_size": ("3", "5", "7"),
        "local_error_metric": ("patch", "min_eigenvalue"),
        "fit_method": ("ransac", "lmeds"),
    }
    parameter_ranges = {
        **TrackImageParts.parameter_ranges,
        "maximum_detector_age_frames": NumericRange(1, 10000, whole=True),
        "maximum_detector_age_ms": NumericRange(0.001, 1e9),
        "local_max_points": NumericRange(4, 256, whole=True),
        "local_min_points": NumericRange(4, 256, whole=True),
        "local_quality": NumericRange(1e-6, 1),
        "local_min_distance": NumericRange(0, 100),
        "local_block_size": NumericRange(3, 63, whole=True, odd=True),
        "local_harris_k": NumericRange(0, 1),
        "local_window": NumericRange(3, 63, whole=True, odd=True),
        "local_levels": NumericRange(0, 5, whole=True),
        "local_iterations": NumericRange(1, 200, whole=True),
        "local_epsilon": NumericRange(1e-6, 1),
        "local_min_eigenvalue": NumericRange(0, 1),
        "local_max_error": NumericRange(0, 255),
        "local_fb_tolerance": NumericRange(0, 100),
        "local_min_inliers": NumericRange(0.01, 1),
        "local_min_span": NumericRange(0.1, 100),
        "local_max_shift": NumericRange(0.1, 1000),
        "local_max_scale": NumericRange(1, 4),
        "local_max_angle": NumericRange(0, 180),
        "local_work_limit": NumericRange(1, 128_000_000, whole=True),
        "fit_threshold": NumericRange(0.001, 100),
        "fit_iterations": NumericRange(1, 10000, whole=True),
        "fit_confidence": NumericRange(0.001, 0.999999),
        "fit_refine_iterations": NumericRange(0, 100, whole=True),
    }
    parameter_help = {
        **TrackImageParts.parameter_help,
        "maximum_detector_age_frames": (
            "Expire beyond this many acquisitions since a full detector match, "
            "even if local motion keeps succeeding."
        ),
        "maximum_detector_age_ms": (
            "Expire beyond this acquisition/media age since a full detector match, "
            "regardless of local success."
        ),
        "local_max_points": "Maximum ROI corners per track, at most 4096 across all tracks.",
        "local_quality": "goodFeaturesToTrack relative corner strength.",
        "local_min_distance": "Minimum separation of ROI corners in pixels.",
        "local_block_size": "Odd corner covariance neighborhood.",
        "local_gradient_size": "Corner gradient aperture (dropdown).",
        "local_harris": "Use Harris corner scoring instead of minimum eigenvalues.",
        "local_harris_k": "Harris free parameter when enabled.",
        "local_window": "Odd Lucas-Kanade patch width and height.",
        "local_levels": "Maximum LK pyramid level; zero uses original resolution only.",
        "local_iterations": "LK termination iteration ceiling.",
        "local_epsilon": "LK termination displacement epsilon in pixels.",
        "local_min_eigenvalue": "LK minimum normalized corner eigenvalue.",
        "local_max_error": "Maximum forward native patch error; used only in patch error mode.",
        "local_error_metric": (
            "Native LK patch error or minimum eigenvalue; neither is confidence."
        ),
        "fit_method": "Robust similarity estimation: RANSAC or LMedS (dropdown).",
        "local_fb_tolerance": "Maximum round-trip pixel error; zero disables backward checking.",
        "local_min_points": "Minimum supported points for local similarity fitting.",
        "local_min_inliers": "Required retained-inlier fraction of the original ROI points.",
        "local_min_span": "Minimum point span on both axes before and after robust fitting.",
        "local_max_shift": "Maximum local center displacement per acquisition in pixels.",
        "local_max_scale": "Maximum per-frame scale ratio (reciprocal sets the minimum).",
        "local_max_angle": "Maximum per-frame local rotation in degrees.",
        "local_work_limit": (
            "Conservative point times window area times level bound; not elapsed "
            "time or native RSS."
        ),
        "fit_threshold": "RANSAC reprojection and retained residual ceiling in pixels.",
        "fit_iterations": "Maximum RANSAC iterations.",
        "fit_confidence": "RANSAC confidence control, not track probability.",
        "fit_refine_iterations": "Native similarity-fit refinement iterations.",
    }
    input_help = {
        **TrackImageParts.input_help,
        "image": (
            "Current uint8 image/grid/ticket; convert other depths explicitly "
            "before local tracking."
        ),
        "refresh": (
            "Wire cadence Refresh on every image. True requires current detector "
            "output, including explicit empty results."
        ),
        "detections": (
            "Wire a detector downstream of the cadence Image output; absent on local-only frames."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = (
            "Periodic full-frame detection with bounded local image tracking between refreshes."
        )
        self.documentation = (
            "<p>Wire Detector refresh cadence Image to any full-frame detector. "
            "Wire original Image, cadence Refresh and detector results here. "
            "Local LK similarity measurements refine existing identities only; only full detection "
            "discovers new objects. Blue means local evidence, green detector evidence, "
            "amber prediction. "
            "See docs/refresh-tracking.md for clock, budget, freshness and loss contracts.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["local_min_points"] > p["local_max_points"]:
            raise ValidationError(
                "Minimum local points exceeds maximum", "local_min_points", "local_max_points"
            )
        if p["maximum_tracks"] * p["local_max_points"] > 4096:
            raise ValidationError(
                "Total corner ceiling is 4096", "maximum_tracks", "local_max_points"
            )
        if type(p["local_harris"]) is not bool:
            raise ValidationError("Harris choice must be boolean", "local_harris")

    def estimate_state_bytes(self, image):
        return super().estimate_state_bytes(image) + image.shape[0] * image.shape[1] * 2 + 4096 * 32

    def extra_context(self, inputs):
        super().extra_context(inputs)
        if type(inputs.get("refresh")) is not bool:
            raise ValidationError("Refresh must be a current boolean event")
        if inputs["refresh"] != ("detections" in inputs):
            raise ValidationError(
                "Refresh requires detector results; local frames require no detector event"
            )
        a, b = self.input_payloads["image"], self.input_payloads.get("refresh", {})
        compatible_frame((ValuePacket(None, a), ValuePacket(None, b)))
        if (
            a.get("ticket")
            and b.get("ticket")
            and ticket_key(a["ticket"]) != ticket_key(b["ticket"])
        ):
            raise ValidationError("Refresh ticket differs from current image")
        return None

    def _allow_association(self, track, observation):
        return self._refresh or self._local_ids.get(observation["index"]) == track["id"]

    def _allow_birth(self, observation):
        return self._refresh

    def _detection_parameters(self, parameters):
        # Local optical flow is identity-bound geometry, not fresh appearance evidence.
        if not self._refresh and parameters["appearance_policy"] == "required":
            return {**parameters, "appearance_policy": "if_both"}
        return parameters

    def step(self, inputs, state, mask):
        p = dict(self.parameters)
        for name, bounds in self.parameter_ranges.items():
            p[name] = bounds.check(p[name], name, name)
        try:
            gray = np.ascontiguousarray(gray_image(inputs["image"], (np.uint8,)))
        except ValueError as exc:
            raise ValidationError("Local tracking needs a uint8 image: " + str(exc)) from exc
        if gray.size > 4_194_304:
            raise ValidationError("Local refresh tracking is limited to 4M image pixels")
        time_ms, dt, _ = self._time(inputs, state, p)
        index = state["activations"]
        self._refresh, self._local_ids = inputs["refresh"], {}
        stale = [
            {"id": t["id"], "reason": "detector_age"}
            for t in state.get("tracks", [])
            if index - t["detector_frame"] > p["maximum_detector_age_frames"]
            or time_ms - t["detector_timestamp_ms"] > p["maximum_detector_age_ms"]
        ]
        state["tracks"] = [
            t for t in state.get("tracks", []) if t["id"] not in {t["id"] for t in stale}
        ]
        local, diagnostics, next_points = [], [], {}
        previous = state.get("previous_gray")
        if not self._refresh and previous is not None and dt * 1000 <= p["maximum_interval_ms"]:
            total = sum(
                len(state.get("local_points", {}).get(t["id"], [])) for t in state["tracks"]
            )
            if total * p["local_window"] ** 2 * (p["local_levels"] + 1) > p["local_work_limit"]:
                raise ValidationError("Total local-flow work exceeds limit", "local_work_limit")
            for track in state["tracks"]:
                points = state.get("local_points", {}).get(
                    track["id"], np.empty((0, 1, 2), np.float32)
                )
                fitted, detail = local_motion(previous, gray, points, p, mask)
                if fitted is not None:
                    matrix, retained = fitted
                    center = matrix @ np.r_[track["mean"][:2], 1]
                    if np.linalg.norm(center - track["mean"][:2]) > p["local_max_shift"]:
                        fitted, detail = None, {"reason": "center_shift"}
                    else:
                        detection = {
                            "center": center.tolist(),
                            "label": track["label"],
                            "class_id": track.get("class_id"),
                            "id": f"local_flow:{track['id']}",
                            "score": track["detector_score"] * detail["inlier_fraction"],
                        }
                        if track["size_known"]:
                            detection["size"] = (
                                np.exp(track["mean"][2:4]) * detail["scale"]
                            ).tolist()
                        if track["angle_known"]:
                            detection["angle_degrees"] = float(
                                track["mean"][4] + detail["angle_degrees"]
                            )
                        self._local_ids[len(local)] = track["id"]
                        local.append(detection)
                        next_points[track["id"]] = retained
                diagnostics.append({"id": track["id"], **detail})
        result = super().step(
            {**inputs, "detections": inputs["detections"] if self._refresh else local}, state, mask
        )
        report = result.outputs["report"]
        report.update(
            refresh=self._refresh,
            detector_calls=state.get("detector_calls", 0) + int(self._refresh),
            local_motion=diagnostics,
            frame_index=index,
        )
        report["expired"] = stale + report["expired"]
        state["detector_calls"] = report["detector_calls"]
        details = {}
        for track in state["tracks"]:
            observed = track["misses"] == 0
            if self._refresh and observed:
                track["detector_frame"], track["detector_score"] = index, track["score"]
                track["detector_timestamp_ms"] = time_ms
                corners = track_corners(track)
                if not np.isfinite(corners).all() or np.max(np.abs(corners)) > 1e7:
                    raise ValidationError("Track ROI exceeds local image coordinate limits")
                x0, y0 = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
                x1, y1 = np.minimum(gray.shape[::-1], np.ceil(corners.max(axis=0)) + 1).astype(int)
                if x1 > x0 and y1 > y0:
                    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillConvexPoly(roi, np.rint(corners - [x0, y0]).astype(np.int32), 255)
                    if mask is not None:
                        roi &= mask[y0:y1, x0:x1]
                    points = cv2.goodFeaturesToTrack(
                        gray[y0:y1, x0:x1],
                        p["local_max_points"],
                        p["local_quality"],
                        p["local_min_distance"],
                        mask=roi,
                        blockSize=p["local_block_size"],
                        gradientSize=int(p["local_gradient_size"]),
                        useHarrisDetector=p["local_harris"],
                        k=p["local_harris_k"],
                    )
                    if points is not None:
                        next_points[track["id"]] = points + np.array([x0, y0], np.float32)
            source = (
                "detector"
                if self._refresh and observed
                else "local_flow"
                if observed
                else "prediction"
            )
            details[track["id"]] = {
                "measurement_source": source,
                "detector_age_frames": index - track["detector_frame"],
                "last_detector_frame": track["detector_frame"],
                "detector_age_ms": time_ms - track["detector_timestamp_ms"],
                "detector_observed": source == "detector",
            }
        for item in report["active"]:
            item.update(details[item["id"]])
        for item in result.outputs["tracks"]["items"]:
            item.update(details[item["id"]])
            item["pose"].update(details[item["id"]])
        if not self._refresh:
            self._track_drawings = tuple(
                replace(obj, color=(230, 180, 60)) if obj.color == (60, 210, 100) else obj
                for obj in self._track_drawings
            )
        state.update(
            previous_gray=gray.copy(),
            local_points={
                t["id"]: next_points[t["id"]]
                for t in state["tracks"]
                if t["id"] in next_points and t["misses"] == 0
            },
        )
        result.details.update(refresh=self._refresh, detector_calls=report["detector_calls"])
        return result


REFRESH_TRACKING_BLOCKS = (DetectionCadence, RefreshedImageTracks)
