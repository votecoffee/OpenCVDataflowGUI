"""Associate detections and project part positions with a frame-latched encoder."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .poses import PoseFromFeature, local_offset, make_pose, pose_for_image
from .processing import finite, integer
from .selection import contains
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key


def associate_positions(predicted, detected, gate):
    """Maximize gated associations, then minimize their total distance."""
    if not len(predicted) or not len(detected):
        return {}
    offsets = predicted[:, None, :] - detected[None, :, :]
    distances = np.hypot(offsets[..., 0], offsets[..., 1])
    rows, detections = distances.shape
    # Dummy columns let every track remain unmatched. Their penalty makes
    # cardinality dominate the sum of normalized, in-gate distances.
    penalty = rows + 1
    costs = np.full((rows, detections + rows), float(penalty))
    costs[:, :detections] = np.where(
        distances <= gate, np.minimum(distances, gate) / gate, penalty**2
    )
    columns = costs.shape[1]
    row_potential, col_potential = np.zeros(rows + 1), np.zeros(columns + 1)
    owner = np.zeros(columns + 1, dtype=int)
    previous = np.zeros_like(owner)
    # One-based shortest augmenting paths. Column zero is the active row's root.
    for row in range(1, rows + 1):
        owner[0] = row
        current = 0
        used = np.zeros(columns + 1, dtype=bool)
        best = np.full(columns + 1, np.inf)
        while True:
            used[current] = True
            active = owner[current]
            available = np.flatnonzero(~used)
            reduced = costs[active - 1, available - 1] - row_potential[active]
            reduced -= col_potential[available]
            improved = reduced < best[available]
            changed = available[improved]
            best[changed] = reduced[improved]
            previous[changed] = current
            following = available[np.argmin(best[available])]
            delta = best[following]
            row_potential[owner[used]] += delta
            col_potential[used] -= delta
            best[available] -= delta
            current = following
            if not owner[current]:
                break
        while current:
            prior = previous[current]
            owner[current] = owner[prior]
            current = prior
    return {
        int(row - 1): col - 1
        for col, row in enumerate(owner[1 : detections + 1], 1)
        if row and distances[row - 1, col - 1] <= gate
    }


class TrackConveyorParts(SequenceBlock):
    type_id, title, subtype = "sequence.track_conveyor", "Track conveyor parts", "Object tracking"
    inputs = (
        *SequenceBlock.inputs,
        Port("detections", "data", required=False),
        Port("encoder", "any", required=False, visual_type="number"),
    )
    data_outputs = (
        Port("tracks", "data"),
        Port("poses", "data"),
        Port("count", "number"),
        Port("overlay", required=False),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 1,
        "motion": "stationary",
        "pixels_per_count": [1.0, 0.0],
        "encoder_modulus": 0,
        "maximum_step_pixels": 2000.0,
        "association_distance": 20.0,
        "maximum_missed_frames": 10,
        "offscreen_margin": 1000.0,
        "maximum_tracks": 64,
        "minimum_score": 0.0,
        "feature_angle": "fixed",
        "draw_overlay": True,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "motion": ("stationary", "encoder"),
        "feature_angle": ("fixed", "feature"),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "pixels_per_count": "Signed [dx, dy] displacement in image pixels per encoder count.",
        "encoder_modulus": "Counter rollover value. Zero means an unwrapped absolute position.",
        "maximum_step_pixels": "Reject encoder displacement above this size. Zero disables.",
        "association_distance": "Maximum predicted-to-detected center distance in image pixels.",
        "maximum_missed_frames": "Consecutive missed frames for which predictions are retained.",
        "offscreen_margin": "Retain predictions up to this distance beyond the image edges.",
        "minimum_score": "Reject scored detections below this value. Unscored detections pass.",
    }

    def synchronized(self, port):
        image_ticket = self.input_payloads["image"].get("ticket")
        other = self.input_payloads.get(port, {}).get("ticket")
        return not (image_ticket and other and ticket_key(image_ticket) != ticket_key(other))

    def encoder_motion(self, inputs, state):
        p = self.parameters
        if p["motion"] == "stationary":
            return None, np.zeros(2), None
        if not self.synchronized("encoder"):
            return None, None, "encoder_ticket_mismatch"
        position = inputs.get("encoder", self.input_payloads["image"].get("encoder"))
        if isinstance(position, (np.integer, np.floating)):
            position = position.item()
        if (
            type(position) not in (int, float)
            or abs(position) > 2**63 - 1
            or not np.isfinite(float(position))
        ):
            return None, None, "encoder_position_missing_or_invalid"
        vector = np.asarray(p["pixels_per_count"], dtype=np.float64)
        if vector.shape != (2,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Encoder displacement requires a finite nonzero [dx,dy] vector")
        modulus = integer(p["encoder_modulus"], "Encoder modulus", 0, 2**53)
        limit = finite(p["maximum_step_pixels"], "Maximum encoder step", 0)
        if modulus and (modulus < 2 or not 0 <= position < modulus):
            return None, None, "encoder_position_outside_modulus"
        delta = position - state.get("encoder", position)
        if modulus:
            delta %= modulus
            if 2 * delta == modulus:
                return None, None, "encoder_rollover_ambiguous"
            if 2 * delta > modulus:
                delta -= modulus
        with np.errstate(over="ignore", invalid="ignore"):
            shift = vector * delta
        distance = float(np.hypot(*shift))
        if not np.isfinite(distance) or (limit and distance > limit):
            return None, None, "encoder_step_above_maximum"
        return position, shift, None

    def observations(self, image, detections, mask, minimum):
        use_collection = isinstance(detections, dict) and detections.get("model") == (
            "vision_collection_v1"
        )
        if isinstance(detections, dict):
            items = detections.get("items")
            if (
                not isinstance(items, list)
                or detections.get("count") != len(items)
                or detections.get("model") not in ("vision_collection_v1", "pattern_matches_v1")
            ):
                raise ValueError("Invalid conveyor detection collection")
        else:
            items = detections
        if not isinstance(items, list) or len(items) > 128:
            raise ValueError("Conveyor tracking accepts up to 128 detections")
        found = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError("Each conveyor detection must supply a pose")
            score = item.get("score")
            if score is not None:
                score = finite(score, "Detection score", 0)
                if score > 1:
                    raise ValueError("Detection score must be at most 1")
                if score < minimum:
                    continue
            if use_collection:
                converter = PoseFromFeature(
                    parameters={
                        "index": index,
                        "angle_mode": self.parameters["feature_angle"],
                    }
                )
                converter.input_payloads = {"features": self.input_payloads["detections"]}
                pose = converter.feature_pose(image, detections)
            else:
                pose = pose_for_image(item.get("pose", item), image)
            if contains(mask, pose["origin"]):
                found.append((pose, score))
        return found

    def step(self, inputs, state, mask):
        image, p = inputs["image"], self.parameters
        position, shift, reason = self.encoder_motion(inputs, state)
        if not self.synchronized("detections"):
            reason = "detection_ticket_mismatch"
        if reason is None:
            try:
                require_same_frame(
                    (image, ValuePacket(None, self.input_payloads.get("detections", {})))
                )
            except ValueError:
                reason = "detection_frame_mismatch"
        if reason:
            return SequenceStep(
                dict.fromkeys(self.flow_primary),
                ready=False,
                learned=False,
                details={"reason": reason},
            )
        height, width = image.shape[:2]
        size = [width, height]
        gate = finite(p["association_distance"], "Association distance", 1e-6)
        max_misses = integer(p["maximum_missed_frames"], "Maximum missed frames", 0, 10_000)
        margin = finite(p["offscreen_margin"], "Offscreen margin", 0)
        capacity = integer(p["maximum_tracks"], "Maximum tracks", 1, 128)
        minimum = finite(p["minimum_score"], "Minimum score", 0)
        if margin > 100_000 or minimum > 1:
            raise ValueError("Offscreen margin must be at most 100000 and minimum score at most 1")
        found = self.observations(image, inputs.get("detections", []), mask, minimum)
        tracks = state.get("tracks", [])
        predicted = np.asarray([track["center"] for track in tracks], dtype=float).reshape(-1, 2)
        predicted += shift
        centers = np.asarray([pose["origin"] for pose, _ in found], dtype=float)
        association = associate_positions(predicted, centers, gate)
        used = set(association.values())
        retained = []
        for index, track in enumerate(tracks):
            if index in association:
                pose, score = found[association[index]]
                retained.append(
                    {
                        **track,
                        "matrix": pose["source_to_target"],
                        "center": pose["origin"],
                        "score": score,
                        "misses": 0,
                    }
                )
            elif track["misses"] < max_misses:
                x, y = predicted[index]
                if -margin <= x <= width - 1 + margin and -margin <= y <= height - 1 + margin:
                    matrix = local_offset(*shift, 0) @ np.asarray(track["matrix"])
                    retained.append(
                        {
                            **track,
                            "matrix": matrix.tolist(),
                            "center": [float(x), float(y)],
                            "misses": track["misses"] + 1,
                        }
                    )
        expired = len(tracks) - len(retained)
        next_id = state.get("next_track", 1)
        dropped = 0
        for index, (pose, score) in enumerate(found):
            if index in used:
                continue
            if len(retained) == capacity:
                dropped += 1
                continue
            retained.append(
                {
                    "id": next_id,
                    "matrix": pose["source_to_target"],
                    "center": pose["origin"],
                    "score": score,
                    "misses": 0,
                }
            )
            next_id += 1
        state.update(tracks=retained, next_track=next_id)
        if position is not None:
            state["encoder"] = position
        items, poses = [], []
        overlay = overlay_image(image) if p["draw_overlay"] else NO_VALUE
        for track in retained:
            x, y = track["center"]
            visible = bool(0 <= x <= width - 1 and 0 <= y <= height - 1)
            observed = track["misses"] == 0
            identity = f"{state['generation']}:{track['id']}"
            pose = make_pose(
                track["matrix"],
                size,
                self.input_payloads["image"],
                track_id=identity,
                predicted=not observed,
            )
            items.append(
                {
                    "track_id": identity,
                    "pose": pose,
                    "observed": observed,
                    "in_frame": visible,
                    "missed_frames": track["misses"],
                    "last_score": track["score"],
                }
            )
            if observed and visible:
                poses.append(pose)
            if visible and overlay is not NO_VALUE:
                point = (round(x), round(y))
                color = (70, 220, 140) if observed else (50, 190, 255)
                cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 11, 1)
                cv2.putText(
                    overlay, str(track["id"]), point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                )
        return SequenceStep(
            {
                "tracks": {"model": "conveyor_tracks_v1", "items": items, "count": len(items)},
                "poses": poses,
                "count": len(items),
                "overlay": overlay,
            },
            details={
                "encoder": position,
                "displacement": shift.tolist(),
                "observed": len(poses),
                "predicted": sum(not item["observed"] for item in items),
                "capacity_dropped": dropped,
                "expired": expired,
            },
            graphics={"conveyor tracks": overlay} if overlay is not NO_VALUE else {},
        )


class ConveyorDemo(Block):
    type_id, title, category = "input.conveyor_demo", "Conveyor demo frames", "Input"
    block_type, subtype = "Input", "Synthetic"
    outputs = (Port("image"), Port("reference"), Port("encoder", "data"))

    def reset_runtime(self):
        self.index = 0

    def copy_runtime_from(self, other):
        self.index = getattr(other, "index", 0)

    def process(self, inputs):
        index = getattr(self, "index", 0)
        self.index = index + 1
        reference = np.random.default_rng(72).integers(40, 240, (16, 20), dtype=np.uint8)
        image = np.full((90, 140), 20, np.uint8)
        for lane, x in enumerate((18 + index * 6, 58 + index * 6)):
            y = 22 + lane * 33
            if x + reference.shape[1] <= image.shape[1] and not (lane == 1 and index == 4):
                image[y : y + 16, x : x + 20] = reference
        return Result({"image": image, "reference": reference, "encoder": index * 6})
