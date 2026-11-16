"""Copyable constant-velocity image tracks with size/angle uncertainty and gated assignment."""

import copy
import math

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .objects import collection, validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .poses import local_offset, make_pose, pose_for_image
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key
from .tracking_assignment import associate_costs

_H = np.eye(5, 10)
_FIELDS = ("x", "y", "log_width", "log_height", "angle_degrees")


def compatible_frame(values):
    try:
        require_same_frame(values)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def angle_delta(angle, anchor, period):
    return (angle - anchor + period / 2) % period - period / 2


def matrices(dt, p):
    transition = np.eye(10)
    transition[:5, 5:] = np.eye(5) * dt
    acceleration = np.array(
        [p["position_acceleration"]] * 2
        + [p["scale_acceleration"]] * 2
        + [p["angular_acceleration"]]
    )
    noise = np.zeros((10, 10))
    for i, variance in enumerate(acceleration**2):
        noise[i, i] = variance * dt**4 / 4
        noise[i, i + 5] = noise[i + 5, i] = variance * dt**3 / 2
        noise[i + 5, i + 5] = variance * dt**2
    return transition, noise


def measurement_noise(p):
    return np.diag(
        np.array([p["position_noise"]] * 2 + [p["scale_noise"]] * 2 + [p["angle_noise"]]) ** 2
    )


def predict_track(track, dt, p):
    native = cv2.KalmanFilter(10, 5, 0, cv2.CV_64F)
    native.transitionMatrix, native.processNoiseCov = matrices(dt, p)
    native.measurementMatrix = _H.copy()
    native.statePost = track["mean"].reshape(10, 1).copy()
    native.errorCovPost = track["covariance"].copy()
    native.predict()
    return native


def correct_track(native, observation, track, p):
    indices = np.flatnonzero(observation["available"])
    query = cv2.KalmanFilter(10, len(indices), 0, cv2.CV_64F)
    query.statePre = native.statePre.copy()
    query.errorCovPre = native.errorCovPre.copy()
    query.measurementMatrix = _H[indices].copy()
    query.measurementNoiseCov = measurement_noise(p)[np.ix_(indices, indices)].copy()
    for dim in range(2, 5):
        known = track["angle_known"] if dim == 4 else track["size_known"]
        if observation["available"][dim] and not known:
            # An unmeasured fallback is not a prior size/orientation observation.
            query.statePre[dim, 0] = observation["value"][dim]
            query.statePre[dim + 5, 0] = 0
            query.errorCovPre[dim, :] = query.errorCovPre[:, dim] = 0
            query.errorCovPre[dim + 5, :] = query.errorCovPre[:, dim + 5] = 0
            query.errorCovPre[dim, dim] = (
                p["initial_angle_sigma" if dim == 4 else "initial_scale_sigma"] ** 2
            )
            query.errorCovPre[dim + 5, dim + 5] = (
                p["initial_angular_velocity_sigma" if dim == 4 else "initial_scale_velocity_sigma"]
                ** 2
            )
    measured = observation["value"].copy()
    measured[4] = query.statePre[4, 0] + angle_delta(
        measured[4], query.statePre[4, 0], float(p["angle_period"])
    )
    query.correct(measured[indices, None])
    # Joseph form preserves positive semidefiniteness under mixed state units.
    remainder = np.eye(10) - query.gain @ query.measurementMatrix
    covariance = (
        remainder @ query.errorCovPre @ remainder.T
        + query.gain @ query.measurementNoiseCov @ query.gain.T
    )
    mean = query.statePost[:, 0].copy()
    return mean, (covariance + covariance.T) / 2


def observations(value, image, context, p, mask):
    is_collection = isinstance(value, dict) and value.get("model") == "vision_collection_v1"
    is_pattern = isinstance(value, dict) and value.get("model") == "pattern_matches_v1"
    if is_collection:
        validate_collection(value)
        if value.get("origin", [0, 0]) != [0, 0]:
            raise ValidationError("Rebase detection collection origin to the current image first")
        if value["image_size"] != list(image.shape[1::-1]):
            raise ValidationError("Reproject detection collections to the current image size first")
        compatible_frame((image, ValuePacket(None, value.get("context", {}))))
    if isinstance(value, dict):
        if not (is_collection or is_pattern or value.get("model") == "image_detections_v1"):
            raise ValidationError(
                "Use image_detections_v1, a measured collection or pattern matches"
            )
        items = value.get("items")
        if not isinstance(items, list) or value.get("count") != len(items):
            raise ValidationError("Detection count must match its items")
    else:
        items = value
    if not isinstance(items, list) or len(items) > 128:
        raise ValidationError("Use an explicit list of 0-128 image detections")
    result = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValidationError("Every image detection must be an object")
        score = item.get("score", 1.0)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError("Detection score must be finite in [0,1]")
        if score < p["minimum_detection_score"]:
            continue
        label = item.get("label")
        if label is not None and (type(label) not in (int, str) or len(str(label)) > 128):
            raise ValidationError(
                "Optional detection labels need text or integer values up to 128 characters"
            )
        if p["label_policy"] == "required" and label is None:
            raise ValidationError("Label policy requires a label on every detection")
        reference_size = model_box = None
        if is_collection:
            metrics = item["metrics"]
            center = [metrics.get("center_x"), metrics.get("center_y")]
            size = [metrics.get("width"), metrics.get("height")]
            if metrics.get("major_size") and metrics.get("minor_size"):
                size = [metrics["major_size"], metrics["minor_size"]]
            if size == [None, None]:
                size = None
            angle = -metrics["angle"] if metrics.get("angle") is not None else None
        elif is_pattern:
            original_pose = item.get("pose", {})
            target = (
                original_pose.get("target_payload", {}) if isinstance(original_pose, dict) else {}
            )
            compatible_frame((image, ValuePacket(None, target)))
            a, b = image.metadata.get("ticket"), target.get("ticket")
            if a and b and ticket_key(a) != ticket_key(b):
                raise ValidationError("Pattern pose ticket differs from current image")
            pose = pose_for_image(original_pose, image)
            center, angle = pose["origin"], pose["angle_degrees"]
            transform = np.asarray(pose["source_to_target"], float)
            gram = transform[:2, :2].T @ transform[:2, :2]
            if (
                not np.allclose(transform[2], [0, 0, 1], atol=1e-9)
                or np.linalg.det(transform[:2, :2]) <= 0
                or not np.allclose(gram, np.eye(2) * gram[0, 0], rtol=1e-6, atol=1e-9)
            ):
                raise ValidationError(
                    "Pattern tracking requires a uniform affine pose; rectify first"
                )
            corners = np.asarray(item.get("corners"), float)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValidationError("Pattern tracking needs four finite corners per match")
            size = [
                float(np.linalg.norm(corners[1] - corners[0])),
                float(np.linalg.norm(corners[3] - corners[0])),
            ]
            reference_size = (np.array(size) / [pose["scale_x"], pose["scale_y"]]).tolist()
            model_box = (np.c_[corners, np.ones(4)] @ np.linalg.inv(transform).T)[:, :2].tolist()
        else:
            center, size, angle = item.get("center"), item.get("size"), item.get("angle_degrees")
        center = np.asarray(center, float)
        if center.shape != (2,) or not np.isfinite(center).all() or np.max(np.abs(center)) > 1e6:
            raise ValidationError(
                "Detection centers must be finite [x,y] within one million pixels"
            )
        margin = p["offscreen_margin"]
        if not (
            -margin <= center[0] <= image.shape[1] - 1 + margin
            and -margin <= center[1] <= image.shape[0] - 1 + margin
        ):
            continue
        if mask is not None:
            x, y = np.rint(center).astype(int)
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]):
                continue
        available = np.array([True, True, size is not None, size is not None, angle is not None])
        if size is None:
            size = np.array([p["initial_width"], p["initial_height"]])
        else:
            size = np.asarray(size, float)
            if (
                size.shape != (2,)
                or not np.isfinite(size).all()
                or np.any(size < 1e-6)
                or np.any(size > 1e6)
            ):
                raise ValidationError(
                    "Measured sizes must be positive [width,height] within [1e-6,1e6]"
                )
        if angle is None:
            angle = 0.0
        elif type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) > 1e9:
            raise ValidationError(
                "Measured angle must be finite counterclockwise degrees within +/-1e9"
            )
        if reference_size is None:
            reference_size = np.asarray(size).tolist()
            w, h = size
            model_box = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
        result.append(
            {
                "reference_size": reference_size,
                "model_box": model_box,
                "value": np.r_[
                    center, np.log(size), angle_delta(angle, 0, float(p["angle_period"]))
                ],
                "available": available,
                "score": float(score),
                "label": label,
                "index": index,
            }
        )
    return result


class TrackImageParts(SequenceBlock):
    type_id, title, subtype = "sequence.track_image", "Track image detections", "Object tracking"
    process_inputs_readonly = True
    inputs = (
        *SequenceBlock.inputs,
        Port("detections", "data", required_when_connected=True),
        Port("timestamp_ms", "number", required=False, required_when_connected=True),
    )
    data_outputs = (
        Port("tracks", "data"),
        Port("poses", "data"),
        Port("count", "number"),
        Port("report", "data"),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 1,
        "clock": "auto",
        "nominal_interval_ms": 100.0,
        "minimum_interval_ms": 0.001,
        "maximum_interval_ms": 1000.0,
        "angle_period": "360",
        "maximum_tracks": 64,
        "minimum_observations": 2,
        "emit_tentative": False,
        "emit_predictions": True,
        "maximum_missed_frames": 10,
        "maximum_unobserved_ms": 1000.0,
        "offscreen_margin": 100.0,
        "minimum_detection_score": 0.0,
        "minimum_confidence": 0.05,
        "confidence_decay_ms": 1000.0,
        "label_policy": "if_both",
        "association_distance": 50.0,
        "innovation_gate": 18.5,
        "maximum_scale_ratio": 2.0,
        "maximum_angle_difference": 60.0,
        "ambiguity_margin": 0.05,
        "position_noise": 1.0,
        "scale_noise": 0.05,
        "angle_noise": 5.0,
        "position_acceleration": 10.0,
        "scale_acceleration": 0.1,
        "angular_acceleration": 30.0,
        "initial_position_sigma": 2.0,
        "initial_velocity_sigma": 100.0,
        "initial_scale_sigma": 0.2,
        "initial_scale_velocity_sigma": 0.5,
        "initial_angle_sigma": 20.0,
        "initial_angular_velocity_sigma": 90.0,
        "initial_width": 20.0,
        "initial_height": 20.0,
        "maximum_position_sigma": 100.0,
        "maximum_scale_sigma": 1.0,
        "maximum_angle_sigma": 90.0,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "clock": ("auto", "explicit", "media", "nominal"),
        "angle_period": ("360", "180"),
        "label_policy": ("if_both", "ignore", "required"),
    }
    parameter_ranges = {
        "nominal_interval_ms": NumericRange(0.001, 1e6),
        "minimum_interval_ms": NumericRange(0.000001, 1000),
        "maximum_interval_ms": NumericRange(0.001, 1e6),
        "maximum_tracks": NumericRange(1, 128, whole=True),
        "minimum_observations": NumericRange(1, 1000, whole=True),
        "maximum_missed_frames": NumericRange(0, 10000, whole=True),
        "maximum_unobserved_ms": NumericRange(0.001, 1e9),
        "offscreen_margin": NumericRange(0, 1e5),
        "minimum_detection_score": NumericRange(0, 1),
        "minimum_confidence": NumericRange(0, 1),
        "confidence_decay_ms": NumericRange(0.001, 1e9),
        "association_distance": NumericRange(0.001, 1e6),
        "innovation_gate": NumericRange(0.001, 1e6),
        "maximum_scale_ratio": NumericRange(1, 1000),
        "maximum_angle_difference": NumericRange(0, 180),
        "ambiguity_margin": NumericRange(0, 1),
        **{
            key: NumericRange(1e-6, 1e6)
            for key in (
                "position_noise",
                "angle_noise",
                "initial_position_sigma",
                "initial_velocity_sigma",
                "initial_angle_sigma",
                "initial_angular_velocity_sigma",
                "maximum_position_sigma",
                "maximum_angle_sigma",
                "initial_width",
                "initial_height",
            )
        },
        **{
            key: NumericRange(1e-6, 10)
            for key in (
                "scale_noise",
                "initial_scale_sigma",
                "initial_scale_velocity_sigma",
                "maximum_scale_sigma",
            )
        },
        "position_acceleration": NumericRange(0, 1e6),
        "scale_acceleration": NumericRange(0, 10),
        "angular_acceleration": NumericRange(0, 1e6),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "clock": (
            "Auto uses explicit timestamp, then video media time, then nominal "
            "intervals; switching clock type retires state."
        ),
        "nominal_interval_ms": (
            "Deterministic per-activation interval when no acquisition/media clock is used."
        ),
        "minimum_interval_ms": "Reject duplicate/decreasing or smaller time intervals.",
        "maximum_interval_ms": (
            "Larger gaps expire all old tracks before consuming current detections."
        ),
        "angle_period": (
            "360 for directed poses; 180 for unoriented principal axes. Output "
            "angles use [-period/2,period/2)."
        ),
        "maximum_tracks": (
            "Bounded active-track capacity; overflow detections are reported, not silently merged."
        ),
        "minimum_observations": "Cumulative matched detections needed to confirm an identity.",
        "emit_tentative": "Include unconfirmed tracks in Tracks/Poses.",
        "emit_predictions": (
            "Include confirmed missing tracks; every item still identifies observed "
            "versus predicted."
        ),
        "maximum_missed_frames": (
            "Expire after more consecutive missing detections than this count."
        ),
        "maximum_unobserved_ms": (
            "Expire after this many media/nominal milliseconds since last observation."
        ),
        "offscreen_margin": (
            "Allowed predicted-center distance beyond image bounds before expiry."
        ),
        "minimum_detection_score": (
            "Ignore detections below this score; unscored detections use one."
        ),
        "minimum_confidence": (
            "Output floor for last score times confirmation and elapsed-time decay; "
            "not a probability."
        ),
        "confidence_decay_ms": "Exponential prediction confidence time constant.",
        "label_policy": (
            "Require equal known labels, ignore them, or require a label on every detection."
        ),
        "association_distance": (
            "Absolute center-distance gate in current pixels, independent of covariance growth."
        ),
        "innovation_gate": (
            "Maximum squared Mahalanobis innovation over available measurement "
            "dimensions; not a probability."
        ),
        "maximum_scale_ratio": (
            "Per-axis measured/predicted size ratio gate, applied only when size is present."
        ),
        "maximum_angle_difference": (
            "Wrapped angular association gate in degrees, applied when orientation is present."
        ),
        "ambiguity_margin": (
            "Normalized cost gap identifying near-tied associations; flagged and confidence halved."
        ),
        "position_noise": "Measurement standard deviation in current pixels.",
        "scale_noise": "Measurement standard deviation in log width/height.",
        "angle_noise": "Measurement standard deviation in degrees.",
        "position_acceleration": (
            "Process acceleration standard deviation in pixels/second squared."
        ),
        "scale_acceleration": (
            "Process acceleration standard deviation in log size/second squared."
        ),
        "angular_acceleration": (
            "Process acceleration standard deviation in degrees/second squared."
        ),
        "initial_position_sigma": "Initial center uncertainty in pixels.",
        "initial_velocity_sigma": "Initial center-velocity uncertainty in pixels/second.",
        "initial_scale_sigma": "Initial log-size uncertainty.",
        "initial_scale_velocity_sigma": "Initial log-size velocity uncertainty per second.",
        "initial_angle_sigma": "Initial angle uncertainty in degrees when measured.",
        "initial_angular_velocity_sigma": (
            "Initial angular-velocity uncertainty in degrees/second."
        ),
        "initial_width": (
            "Fallback pixels for center-only measurements; size stays explicitly unavailable."
        ),
        "initial_height": (
            "Fallback pixels for center-only measurements; size stays explicitly unavailable."
        ),
        "maximum_position_sigma": (
            "Expire when either predicted center standard deviation exceeds this pixel bound."
        ),
        "maximum_scale_sigma": "Expire measured-size tracks above this log-size uncertainty.",
        "maximum_angle_sigma": (
            "Expire measured-orientation tracks above this angular uncertainty in degrees."
        ),
    }
    input_help = {
        "image": "Current image/grid/ticket; its pixels are not modified.",
        "detections": (
            "An explicit empty list means no detections. Lists of center/optional "
            "size/angle records, measured collections or pattern matches."
        ),
        "timestamp_ms": (
            "Optional synchronized acquisition time in milliseconds; strict "
            "monotonicity within each stream."
        ),
        "mask": "Select detection and output-prediction centers; intersected with region.",
        "region": (
            "Current-grid selection. Excluded detections act as missed observations "
            "for existing tracks."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = self.description or (
            "Associate image detections and predict centers, size and angle through bounded gaps."
        )
        self.documentation = self.documentation or (
            "<p>OpenCV Kalman prediction with elapsed acquisition time "
            "and gated global assignment. "
            "Connect current detections, including an explicit empty list for a missed frame. "
            "Observed, predicted, tentative and ambiguous states remain explicit. "
            "Confidence is a heuristic, not a probability or a safety judgment.</p>"
            "<p>Use 180 degree angles for undirected shape axes, 360 for directed pattern poses. "
            "Clock, size, uncertainty and expiry controls have parameter tooltips. "
            "See docs/image-tracking.md for measurement, coordinate and lifecycle contracts.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["minimum_interval_ms"]) > float(p["maximum_interval_ms"]):
            raise ValidationError(
                "Minimum interval exceeds maximum", "minimum_interval_ms", "maximum_interval_ms"
            )
        for key in ("emit_tentative", "emit_predictions"):
            if type(p[key]) is not bool:
                raise ValidationError("Choose a boolean value", key)
        for kind in ("position", "scale", "angle"):
            if float(p[f"maximum_{kind}_sigma"]) < float(p[f"initial_{kind}_sigma"]):
                raise ValidationError(
                    "Maximum uncertainty is below initial uncertainty",
                    f"maximum_{kind}_sigma",
                    f"initial_{kind}_sigma",
                )

    def estimate_state_bytes(self, image):
        return image.nbytes + int(self.parameters["maximum_tracks"]) * 16384

    def extra_context(self, inputs):
        image_payload = self.input_payloads["image"]
        for port in ("detections", "timestamp_ms"):
            if port not in inputs:
                continue
            other = self.input_payloads.get(port, {})
            a, b = image_payload.get("ticket"), other.get("ticket")
            if a and b and ticket_key(a) != ticket_key(b):
                raise ValidationError(port + " ticket differs from current image")
            if port == "detections":
                embedded = inputs[port].get("context", {}) if isinstance(inputs[port], dict) else {}
                a, b = image_payload.get("ticket"), embedded.get("ticket")
                if a and b and ticket_key(a) != ticket_key(b):
                    raise ValidationError("Embedded detection ticket differs from current image")
                compatible_frame(
                    (
                        ValuePacket(inputs["image"], image_payload),
                        ValuePacket(None, other),
                        ValuePacket(None, embedded),
                    )
                )
        return None

    def _time(self, inputs, state, p):
        mode = p["clock"]
        if mode == "auto":
            mode = (
                "explicit"
                if "timestamp_ms" in inputs
                else "media"
                if "video_frame" in self.input_payloads["image"]
                else "nominal"
            )
        if state.get("clock", mode) != mode:
            raise ValidationError("Tracking clock changed; reset/reacquire on one consistent clock")
        if mode == "nominal":
            value = state.get("time_ms", -p["nominal_interval_ms"]) + p["nominal_interval_ms"]
        elif mode == "explicit":
            value = inputs.get("timestamp_ms")
        else:
            value = self.input_payloads["image"].get("video_frame", {}).get("timestamp_ms")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e15:
            raise ValidationError("Use a finite acquisition timestamp in [0,1e15] milliseconds")
        dt = value - state.get("time_ms", value)
        if "time_ms" in state and dt < p["minimum_interval_ms"]:
            raise ValidationError("Tracking timestamps must strictly increase by Minimum interval")
        return float(value), dt / 1000, mode

    def _allow_association(self, track, observation):
        return True

    def _allow_birth(self, observation):
        return True

    def step(self, inputs, state, mask):
        p = dict(self.parameters)
        for key, limits in self.parameter_ranges.items():
            p[key] = limits.check(p[key], key, key)
        image = inputs["image"]
        if max(image.shape[:2]) > 16384 or image.shape[0] * image.shape[1] > 16_777_216:
            raise ValidationError("Image tracking is limited to 16M pixels and 16384 per axis")
        found = observations(
            inputs["detections"],
            ImageArray(image, image_format(image), self.input_payloads["image"]),
            self.input_payloads.get("detections", {}),
            p,
            mask,
        )
        time_ms, dt, clock = self._time(inputs, state, p)
        tracks = state.get("tracks", [])
        expired = []
        predicted = []
        for track in tracks:
            if dt * 1000 > p["maximum_interval_ms"]:
                expired.append({"id": track["id"], "reason": "interval_gap"})
                continue
            native = predict_track(track, dt, p)
            mean, covariance = native.statePre[:, 0], native.errorCovPre
            reason = self._expiry(track, mean, covariance, time_ms, image, p)
            if reason:
                expired.append({"id": track["id"], "reason": reason})
            else:
                predicted.append((track, native))
        costs = np.zeros((len(predicted), len(found)))
        allowed = np.zeros(costs.shape, bool)
        r = measurement_noise(p)
        for row, (track, native) in enumerate(predicted):
            mean = native.statePre[:, 0]
            for column, item in enumerate(found):
                residual = item["value"] - mean[:5]
                residual[4] = angle_delta(item["value"][4], mean[4], float(p["angle_period"]))
                label_ok = (
                    p["label_policy"] == "ignore"
                    or track["label"] is None
                    or item["label"] is None
                    or track["label"] == item["label"]
                )
                available = item["available"].copy()
                if not track["size_known"]:
                    available[2:4] = False
                if not track["angle_known"]:
                    available[4] = False
                indices = np.flatnonzero(available)
                innovation = (
                    native.errorCovPre[np.ix_(indices, indices)] + r[np.ix_(indices, indices)]
                )
                squared = float(residual[indices] @ np.linalg.solve(innovation, residual[indices]))
                valid = (
                    label_ok
                    and self._allow_association(track, item)
                    and np.linalg.norm(residual[:2]) <= p["association_distance"]
                    and squared <= p["innovation_gate"]
                )
                if item["available"][2] and track["size_known"]:
                    valid &= max(abs(residual[2]), abs(residual[3])) <= math.log(
                        p["maximum_scale_ratio"]
                    )
                if item["available"][4] and track["angle_known"]:
                    valid &= abs(residual[4]) <= p["maximum_angle_difference"]
                allowed[row, column] = valid
                costs[row, column] = max(0, min(1, squared / p["innovation_gate"]))
        matches = associate_costs(costs, allowed)
        retained = []
        associations = []
        for row, (track, native) in enumerate(predicted):
            track = copy.deepcopy(track)
            if row in matches:
                column = matches[row]
                item = found[column]
                mean, covariance = correct_track(native, item, track, p)
                alternatives = costs[row, allowed[row]]
                competitors = costs[allowed[:, column], column]
                ambiguous = (
                    len(alternatives) > 1
                    and np.sort(alternatives)[1] - np.sort(alternatives)[0] <= p["ambiguity_margin"]
                ) or (
                    len(competitors) > 1
                    and np.sort(competitors)[1] - np.sort(competitors)[0] <= p["ambiguity_margin"]
                )
                if item["available"][2] and not track["size_known"]:
                    track["reference_size"] = item["reference_size"]
                    track["model_box"] = item["model_box"]
                track.update(
                    mean=mean,
                    covariance=covariance,
                    hits=track["hits"] + 1,
                    misses=0,
                    last_seen_ms=time_ms,
                    score=item["score"],
                    label=item["label"] if item["label"] is not None else track["label"],
                    size_known=track["size_known"] or bool(item["available"][2]),
                    angle_known=track["angle_known"] or bool(item["available"][4]),
                    ambiguous=bool(ambiguous),
                )
                associations.append(
                    {
                        "id": track["id"],
                        "detection_index": item["index"],
                        "normalized_cost": float(costs[row, column]),
                        "ambiguous": bool(ambiguous),
                    }
                )
            else:
                track.update(
                    mean=native.statePre[:, 0].copy(),
                    covariance=(native.errorCovPre + native.errorCovPre.T) / 2,
                    misses=track["misses"] + 1,
                )
            track["mean"][4] = angle_delta(track["mean"][4], 0, float(p["angle_period"]))
            reason = self._expiry(track, track["mean"], track["covariance"], time_ms, image, p)
            if reason:
                expired.append({"id": track["id"], "reason": reason})
            else:
                retained.append(track)
        used = set(matches.values())
        next_id = state.get("next_id", 1)
        dropped = 0
        for column, item in enumerate(found):
            if column in used or not self._allow_birth(item):
                continue
            if len(retained) >= p["maximum_tracks"]:
                dropped += 1
                continue
            sigmas = (
                [p["initial_position_sigma"]] * 2
                + [p["initial_scale_sigma"]] * 2
                + [
                    p["initial_angle_sigma"]
                    if item["available"][4]
                    else float(p["angle_period"]) / 2
                ]
                + [p["initial_velocity_sigma"]] * 2
                + [p["initial_scale_velocity_sigma"]] * 2
                + [p["initial_angular_velocity_sigma"]]
            )
            retained.append(
                {
                    "id": next_id,
                    "mean": np.r_[item["value"], np.zeros(5)],
                    "covariance": np.diag(np.square(sigmas)),
                    "reference_size": item["reference_size"],
                    "model_box": item["model_box"],
                    "hits": 1,
                    "misses": 0,
                    "last_seen_ms": time_ms,
                    "score": item["score"],
                    "label": item["label"],
                    "size_known": bool(item["available"][2]),
                    "angle_known": bool(item["available"][4]),
                    "ambiguous": False,
                }
            )
            next_id += 1
        state.update(tracks=retained, next_id=next_id, time_ms=time_ms, clock=clock)
        items, poses, diagnostics, graphics = [], [], [], []
        for track in retained:
            mean, covariance = track["mean"], track["covariance"]
            confirmed = track["hits"] >= p["minimum_observations"]
            confidence = (
                track["score"]
                * min(1, track["hits"] / p["minimum_observations"])
                * math.exp(-(time_ms - track["last_seen_ms"]) / p["confidence_decay_ms"])
                * (0.5 if track["ambiguous"] else 1)
            )
            status = "observed" if track["misses"] == 0 else "predicted"
            angle = angle_delta(mean[4], 0, float(p["angle_period"])) if track["angle_known"] else 0
            width, height = np.exp(mean[2:4])
            center = mean[:2].tolist()
            identity = f"{state['generation']}:{track['id']}"
            details = {
                "id": track["id"],
                "track_id": identity,
                "state": status if confirmed else "tentative",
                "observed": track["misses"] == 0,
                "confirmed": confirmed,
                "confidence": confidence,
                "misses": track["misses"],
                "hits": track["hits"],
                "angle_available": track["angle_known"],
                "size_available": track["size_known"],
                "association_ambiguous": track["ambiguous"],
                "label": track["label"],
                "center": center,
                "angle_degrees": angle if track["angle_known"] else None,
                "size": [float(width), float(height)] if track["size_known"] else None,
                "velocity": mean[5:].tolist(),
                "covariance": covariance.tolist(),
                "standard_deviation": dict(
                    zip(
                        _FIELDS,
                        np.sqrt(np.maximum(np.diag(covariance)[:5], 0)).tolist(),
                        strict=True,
                    )
                ),
            }
            diagnostics.append(details)
            selected = mask is None or (
                0 <= round(center[0]) < image.shape[1]
                and 0 <= round(center[1]) < image.shape[0]
                and mask[round(center[1]), round(center[0])] != 0
            )
            emit = (
                selected
                and (confirmed or p["emit_tentative"])
                and (status == "observed" or p["emit_predictions"])
                and confidence >= p["minimum_confidence"]
            )
            if not emit:
                continue
            scale = (
                np.array([width, height]) / track["reference_size"]
                if track["size_known"]
                else np.ones(2)
            )
            pose = make_pose(
                local_offset(*center, angle) @ np.diag([*scale, 1]),
                list(image.shape[1::-1]),
                self.input_payloads["image"],
                track_id=identity,
                observed=track["misses"] == 0,
                orientation_available=track["angle_known"],
            )
            corners = (
                np.c_[track["model_box"], np.ones(4)] @ np.asarray(pose["source_to_target"]).T
            )[:, :2]
            items.append(
                {
                    **details,
                    "pose": pose,
                    "metrics": {
                        "x": float(corners[:, 0].min()) if track["size_known"] else center[0],
                        "y": float(corners[:, 1].min()) if track["size_known"] else center[1],
                        "center_x": center[0],
                        "center_y": center[1],
                        "width": float(width) if track["size_known"] else None,
                        "height": float(height) if track["size_known"] else None,
                        "area": float(width * height) if track["size_known"] else None,
                        "angle": -angle if track["angle_known"] else None,
                        "confidence": confidence,
                    },
                    "geometry": {"points": corners.tolist() if track["size_known"] else [center]},
                }
            )
            poses.append(pose)
            if self.generate_graphics:
                color = (60, 210, 100) if status == "observed" else (40, 190, 240)
                graphics.append(
                    graphic_object("polyline", corners.tolist(), color, closed=True)
                    if track["size_known"]
                    else graphic_object("marker", [center], color)
                )
                graphics.append(
                    graphic_object(
                        "text", [center], color, text=f"{track['id']} {status} {confidence:.2f}"
                    )
                )
        self._track_drawings = tuple(graphics)
        return SequenceStep(
            {
                "tracks": collection("boxes", items, image, self.input_payloads["image"]),
                "poses": poses,
                "count": len(items),
                "report": {
                    "model": "image_tracking_v1",
                    "clock": clock,
                    "timestamp_ms": time_ms,
                    "dt_seconds": dt,
                    "active": diagnostics,
                    "associations": associations,
                    "expired": expired,
                    "capacity_dropped": dropped,
                    "measurement_count": len(found),
                    "covariance_order": [*_FIELDS, *["velocity_" + f for f in _FIELDS]],
                    "confidence_semantics": (
                        "score times confirmation and time decay; not probability"
                    ),
                },
            },
            details={
                "active_tracks": len(retained),
                "output_tracks": len(items),
                "expired_tracks": len(expired),
            },
        )

    def _expiry(self, track, mean, covariance, time_ms, image, p):
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            return "nonfinite_state"
        if np.min(np.linalg.eigvalsh((covariance + covariance.T) / 2)) < -1e-6:
            return "invalid_covariance"
        if np.any(np.abs(mean[:2]) > 1e6) or np.any(np.abs(mean[2:4]) > math.log(1e6)):
            return "state_bounds"
        sigma = np.sqrt(np.maximum(np.diag(covariance), 0))
        if max(sigma[:2]) > p["maximum_position_sigma"]:
            return "position_uncertainty"
        if track["size_known"] and max(sigma[2:4]) > p["maximum_scale_sigma"]:
            return "scale_uncertainty"
        if track["angle_known"] and sigma[4] > p["maximum_angle_sigma"]:
            return "angle_uncertainty"
        if track["misses"] > p["maximum_missed_frames"]:
            return "missed_frames"
        if time_ms - track["last_seen_ms"] > p["maximum_unobserved_ms"]:
            return "unobserved_time"
        margin = p["offscreen_margin"]
        if not (
            -margin <= mean[0] <= image.shape[1] - 1 + margin
            and -margin <= mean[1] <= image.shape[0] - 1 + margin
        ):
            return "offscreen"
        return None

    def finalize_result(self, result, inputs):
        result = super().finalize_result(result, inputs)
        if self.generate_graphics:
            result.drawings["image tracks"] = GraphicDrawing(self._track_drawings)
        return result

    def reset_runtime(self):
        super().reset_runtime()
        self._track_drawings = ()
