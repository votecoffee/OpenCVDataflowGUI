"""Bounded mask association; only observed masks are emitted on the current image."""

import copy

import cv2
import numpy as np

from .classifier_common import canonical_json
from .core import Port
from .graphics_layers import GraphicDrawing, graphic_object
from .image_tracking import TrackImageParts, compatible_frame
from .images import convert_depth, label_color_map
from .instance_segmentation import COLORS, MASK_MODEL
from .local_anomaly import is_hash
from .objects import collection, validate_collection
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket
from .processing import image_value
from .raster_regions import region_from_mask
from .selection import rasterize
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .tickets import ticket_key
from .tracking_assignment import associate_costs


def cropped(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    x, y = int(xx.min()), int(yy.min())
    right, bottom = int(xx.max()) + 1, int(yy.max()) + 1
    return {
        "mask": np.ascontiguousarray(mask[y:bottom, x:right], np.uint8),
        "origin": (x, y),
        "area": len(xx),
        "center": np.array([xx.mean(), yy.mean()]),
    }


def predicted_mask(track, center, shape):
    previous = track["mask"]
    dx, dy = map(int, np.rint(center - previous["center"]))
    x, y = previous["origin"][0] + dx, previous["origin"][1] + dy
    mask = previous["mask"]
    left, top = max(0, x), max(0, y)
    right, bottom = min(shape[1], x + mask.shape[1]), min(shape[0], y + mask.shape[0])
    if right <= left or bottom <= top:
        return {"mask": np.zeros((0, 0), np.uint8), "origin": (left, top), "area": 0}
    result = mask[top - y : bottom - y, left - x : right - x]
    return {"mask": result, "origin": (left, top), "area": int(np.count_nonzero(result))}


def mask_iou(first, second, budget):
    ax, ay = first["origin"]
    bx, by = second["origin"]
    a, b = first["mask"], second["mask"]
    x, y = max(ax, bx), max(ay, by)
    right, bottom = min(ax + a.shape[1], bx + b.shape[1]), min(ay + a.shape[0], by + b.shape[0])
    if right <= x or bottom <= y:
        return 0.0
    budget[0] -= int((right - x) * (bottom - y))
    if budget[0] < 0:
        raise ValidationError("Mask association exceeds pixel-work budget", "maximum_iou_pixels")
    overlap = int(
        np.count_nonzero(
            a[y - ay : bottom - ay, x - ax : right - ax]
            & b[y - by : bottom - by, x - bx : right - bx]
        )
    )
    union = first["area"] + second["area"] - overlap
    return overlap / union if union else 0.0


def appearance(image, observed, p):
    if p["appearance"] == "off":
        return None
    x, y = observed["origin"]
    mask = observed["mask"] * 255
    roi = image[y : y + mask.shape[0], x : x + mask.shape[1]]
    if p["appearance"] == "gray":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        hist = cv2.calcHist([gray], [0], mask, [p["gray_bins"]], [0, 256])
    else:
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValidationError("HSV appearance needs color; choose gray or off", "appearance")
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], mask, [p["hue_bins"], p["saturation_bins"]], [0, 180, 0, 256]
        )
    hist = hist.reshape(-1)
    return np.ascontiguousarray(hist / hist.sum(), np.float32)


class TrackInstanceMasks(SequenceBlock):
    type_id, title, subtype = "sequence.track_instances", "Track instance masks", "Object tracking"
    process_inputs_readonly = True
    inputs = (
        *SequenceBlock.inputs,
        Port("instances", "data", visual_type="collection"),
        Port("receipt", "data", required=False, required_when_connected=True),
        Port("timestamp_ms", "number", required=False, required_when_connected=True),
    )
    data_outputs = (
        Port("image"),
        Port("tracks", "data", visual_type="collection"),
        Port("labels"),
        Port("overlap_count"),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 1,
        "clock": "auto",
        "nominal_interval_ms": 100.0,
        "minimum_interval_ms": 0.001,
        "maximum_interval_ms": 1000.0,
        "identity_policy": "receipt",
        "configured_model_revision": "",
        "appearance": "hsv_hs",
        "hue_bins": 12,
        "saturation_bins": 8,
        "gray_bins": 32,
        "appearance_weight": 0.5,
        "maximum_appearance_distance": 0.7,
        "appearance_update_rate": 0.1,
        "mask_weight": 0.5,
        "minimum_mask_iou": 0.0,
        "maximum_center_distance": 30.0,
        "velocity_update_rate": 1.0,
        "class_policy": "same",
        "ambiguity_policy": "withhold",
        "minimum_cost_margin": 0.03,
        "maximum_missed_frames": 3,
        "maximum_unobserved_ms": 1000.0,
        "minimum_observations": 1,
        "maximum_instances": 64,
        "maximum_tracks": 128,
        "minimum_mask_area": 1,
        "maximum_image_pixels": 4_194_304,
        "maximum_decoded_pixels": 16_777_216,
        "maximum_iou_pixels": 16_777_216,
        "maximum_region_runs": 200000,
        "graphics_maximum_points": 100000,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "clock": ("auto", "nominal", "explicit", "video"),
        "identity_policy": ("receipt", "configured"),
        "appearance": ("off", "gray", "hsv_hs"),
        "class_policy": ("same", "ignore"),
        "ambiguity_policy": ("withhold", "best"),
    }
    parameter_ranges = {
        **{
            k: NumericRange(0, 1)
            for k in (
                "appearance_weight",
                "maximum_appearance_distance",
                "appearance_update_rate",
                "mask_weight",
                "minimum_mask_iou",
                "velocity_update_rate",
                "minimum_cost_margin",
            )
        },
        "maximum_center_distance": NumericRange(0.001, 16384),
        "nominal_interval_ms": NumericRange(0.001, 1e6),
        "minimum_interval_ms": NumericRange(0.001, 1e6),
        "maximum_interval_ms": NumericRange(0.001, 1e9),
        "maximum_unobserved_ms": NumericRange(0, 1e9),
        "maximum_missed_frames": NumericRange(0, 10000, whole=True),
        "minimum_observations": NumericRange(1, 10000, whole=True),
        "maximum_instances": NumericRange(1, 128, whole=True),
        "maximum_tracks": NumericRange(1, 128, whole=True),
        "minimum_mask_area": NumericRange(1, 16_777_216, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_decoded_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_iou_pixels": NumericRange(1, 128_000_000, whole=True),
        "maximum_region_runs": NumericRange(1, 1_000_000, whole=True),
        "graphics_maximum_points": NumericRange(1, 100000, whole=True),
        "hue_bins": NumericRange(2, 32, whole=True),
        "saturation_bins": NumericRange(2, 32, whole=True),
        "gray_bins": NumericRange(2, 256, whole=True),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "identity_policy": "ONNX model/preprocessing receipt, or an explicit custom revision.",
        "appearance": "Masked uint8 histogram working copy; original depth is retained.",
        "mask_weight": "Blend mask IoU and residual center distance before appearance.",
        "minimum_cost_margin": "Withhold local near-ties; no calibrated identity confidence.",
        "maximum_missed_frames": "Retain missing tracks in reports; emit current masks only.",
        "maximum_unobserved_ms": "Expire on acquisition time, separately from wall-clock gaps.",
        "maximum_decoded_pixels": "Bound instance count times source pixels before rasterization.",
    }
    input_help = {
        "instances": "Current zero-origin instance_masks_v1 collection; row IDs are not track IDs.",
        "mask": "Intersect observed masks with selection; excluded objects become unobserved.",
        "receipt": "Default required identity; association does not authenticate mask inference.",
    }
    _time = TrackImageParts._time

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or rule.whole and type(p[key]) is not int:
                raise ValidationError("Use typed numeric tracking controls", key)
        if (
            not isinstance(p["configured_model_revision"], str)
            or len(p["configured_model_revision"]) > 128
        ):
            raise ValidationError(
                "Use a model revision up to 128 characters", "configured_model_revision"
            )
        if p["identity_policy"] == "configured" and not p["configured_model_revision"].strip():
            raise ValidationError("Configure custom model revision", "configured_model_revision")
        if p["hue_bins"] * p["saturation_bins"] > 256:
            raise ValidationError("At most 256 joint HSV bins", "hue_bins", "saturation_bins")
        if p["maximum_interval_ms"] < p["minimum_interval_ms"]:
            raise ValidationError(
                "Maximum interval must cover Minimum", "maximum_interval_ms", "minimum_interval_ms"
            )

    def extra_context(self, inputs):
        contexts = [self.input_payloads["image"], inputs["instances"].get("context", {})]
        contexts += [
            self.input_payloads[k] for k in ("instances", "receipt", "timestamp_ms") if k in inputs
        ]
        compatible_frame(tuple(ValuePacket(None, value) for value in contexts))
        for key in ("ticket", "recipe_ref"):
            known = [value[key] for value in contexts if key in value]
            if known and any(
                (ticket_key(v) != ticket_key(known[0]) if key == "ticket" else v != known[0])
                for v in known[1:]
            ):
                raise ValidationError("Instance masks/receipt have another " + key)
        if self.parameters["identity_policy"] == "configured":
            return {"configured_revision": self.parameters["configured_model_revision"]}
        receipt = inputs.get("receipt", {})
        if not isinstance(receipt, dict):
            raise ValidationError("Connect an ONNX execution receipt")
        calls = receipt.get("calls")
        if (
            receipt.get("model") != "onnx_inference_receipt_v1"
            or receipt.get("schema_version") != 1
            or not isinstance(calls, list)
            or len(calls) != 1
            or not isinstance(calls[0], dict)
        ):
            raise ValidationError("Connect one current ONNX receipt", "identity_policy")
        call = calls[0]
        artifact, revision = call.get("artifact", {}), call.get("contract_revision")
        if not is_hash(artifact.get("sha256")) or revision is not None and not is_hash(revision):
            raise ValidationError("Invalid model/contract receipt identity")
        preprocessing = call.get("preprocessing")
        if not isinstance(preprocessing, dict) or len(canonical_json(preprocessing)) > 16384:
            raise ValidationError("Invalid preprocessing receipt identity")
        return {
            "artifact_sha256": artifact["sha256"],
            "contract_revision": revision,
            "preprocessing": copy.deepcopy(preprocessing),
        }

    def step(self, inputs, state, mask):
        p, image = self.parameters, inputs["image"]
        h, w = image.shape[:2]
        if h * w > p["maximum_image_pixels"] or max(h, w) > 16384:
            raise ValidationError("Instance tracking image exceeds pixel/axis limits")
        data = inputs["instances"]
        validate_collection(data)
        if (
            data["image_size"] != [w, h]
            or data.get("origin", [0, 0]) != [0, 0]
            or data.get("instance_masks", {}).get("model") != MASK_MODEL
        ):
            raise ValidationError("Use current zero-origin decoded instance masks")
        if (
            len(data["items"]) > p["maximum_instances"]
            or len(data["items"]) * h * w > p["maximum_decoded_pixels"]
        ):
            raise ValidationError("Instance count/rasterization exceeds limits")
        working = convert_depth(image_value(image, canonical=True), "uint8")
        observations, runs = [], 0
        for item in data["items"]:
            region = item.get("mask_region")
            if not isinstance(region, dict):
                raise ValidationError("Every instance needs a mask region")
            runs += len(region.get("runs", []))
            if runs > p["maximum_region_runs"]:
                raise ValidationError("Instance region-run budget exceeded")
            binary = rasterize(region, image.shape) != 0
            if mask is not None:
                binary &= mask != 0
            observed = cropped(binary)
            if observed is None or observed["area"] < p["minimum_mask_area"]:
                continue
            score = item.get("score")
            if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValidationError("Instance scores must be finite in [0,1]")
            observed.update(
                item=item,
                appearance=appearance(working, observed, p),
                label=item.get("class_id", item.get("label")),
            )
            observations.append(observed)
        time_ms, dt, clock = self._time(inputs, state, p)
        tracks = copy.deepcopy(state.get("tracks", []))
        expired = [
            t["id"]
            for t in tracks
            if dt * 1000 > p["maximum_interval_ms"]
            or time_ms - t["last_seen_ms"] > p["maximum_unobserved_ms"]
        ]
        tracks = [t for t in tracks if t["id"] not in expired]
        predicted = [t["center"] + t["velocity"] * dt for t in tracks]
        masks = [predicted_mask(t, c, (h, w)) for t, c in zip(tracks, predicted, strict=True)]
        costs = np.zeros((len(tracks), len(observations)))
        allowed = np.zeros(costs.shape, bool)
        overlaps, distances = np.zeros_like(costs), np.full_like(costs, np.nan)
        budget = [p["maximum_iou_pixels"]]
        for row, track in enumerate(tracks):
            for column, observed in enumerate(observations):
                distance = float(np.linalg.norm(predicted[row] - observed["center"]))
                if distance > p["maximum_center_distance"] or (
                    p["class_policy"] == "same" and track["label"] != observed["label"]
                ):
                    continue
                overlap = mask_iou(masks[row], observed, budget)
                if overlap < p["minimum_mask_iou"]:
                    continue
                cost = (1 - p["mask_weight"]) * distance / p["maximum_center_distance"] + p[
                    "mask_weight"
                ] * (1 - overlap)
                if p["appearance"] != "off":
                    d = float(
                        cv2.compareHist(
                            track["appearance"], observed["appearance"], cv2.HISTCMP_BHATTACHARYYA
                        )
                    )
                    if d > p["maximum_appearance_distance"]:
                        continue
                    distances[row, column] = d
                    cost = (1 - p["appearance_weight"]) * cost + p["appearance_weight"] * d
                overlaps[row, column], costs[row, column], allowed[row, column] = (
                    overlap,
                    cost,
                    True,
                )
        ambiguous_rows, ambiguous_columns = set(), set()
        if p["ambiguity_policy"] == "withhold":
            for row in range(len(tracks)):
                candidates = np.flatnonzero(allowed[row])
                if (
                    len(candidates) > 1
                    and np.diff(np.sort(costs[row, candidates]))[0] < p["minimum_cost_margin"]
                ):
                    ambiguous_rows.add(row)
                    ambiguous_columns.update(candidates.tolist())
            for column in range(len(observations)):
                candidates = np.flatnonzero(allowed[:, column])
                if (
                    len(candidates) > 1
                    and np.diff(np.sort(costs[candidates, column]))[0] < p["minimum_cost_margin"]
                ):
                    ambiguous_columns.add(column)
                    ambiguous_rows.update(candidates.tolist())
            for row in ambiguous_rows:
                allowed[row] = False
            for column in ambiguous_columns:
                allowed[:, column] = False
        matched = associate_costs(costs, allowed)
        associations, current = [], []
        for row, track in enumerate(tracks):
            track["age"] += 1
            track["center"] = predicted[row]
            if row not in matched:
                track["misses"] += 1
                continue
            column = matched[row]
            observed = observations[column]
            elapsed = (time_ms - track["last_seen_ms"]) / 1000
            velocity = (
                (observed["center"] - track["mask"]["center"]) / elapsed
                if elapsed
                else track["velocity"]
            )
            track["velocity"] = (1 - p["velocity_update_rate"]) * track["velocity"] + p[
                "velocity_update_rate"
            ] * velocity
            track.update(
                center=observed["center"].copy(),
                mask={k: copy.deepcopy(observed[k]) for k in ("mask", "origin", "area", "center")},
                misses=0,
                hits=track["hits"] + 1,
                last_seen_ms=time_ms,
                last_ticket=copy.deepcopy(self.input_payloads["image"].get("ticket")),
            )
            if p["appearance"] != "off":
                rate = p["appearance_update_rate"]
                track["appearance"] = (1 - rate) * track["appearance"] + rate * observed[
                    "appearance"
                ]
            current.append((track, observed))
            associations.append(
                {
                    "track_id": track["id"],
                    "instance_id": observed["item"]["id"],
                    "mask_iou": float(overlaps[row, column]),
                    "cost": float(costs[row, column]),
                    "appearance_distance": float(distances[row, column])
                    if np.isfinite(distances[row, column])
                    else None,
                }
            )
        for track in tracks:
            if track["misses"] > p["maximum_missed_frames"]:
                expired.append(track["id"])
        tracks = [t for t in tracks if t["id"] not in expired]
        next_id = state.get("next_id", 1)
        for column, observed in enumerate(observations):
            if column in matched.values() or column in ambiguous_columns:
                continue
            if len(tracks) >= p["maximum_tracks"]:
                raise ValidationError(
                    "Track capacity exceeded; observations not silently dropped", "maximum_tracks"
                )
            track = {
                "id": next_id,
                "center": observed["center"].copy(),
                "velocity": np.zeros(2),
                "mask": {
                    k: copy.deepcopy(observed[k]) for k in ("mask", "origin", "area", "center")
                },
                "appearance": copy.deepcopy(observed["appearance"]),
                "label": observed["label"],
                "age": 1,
                "hits": 1,
                "misses": 0,
                "last_seen_ms": time_ms,
                "last_ticket": copy.deepcopy(self.input_payloads["image"].get("ticket")),
            }
            next_id += 1
            tracks.append(track)
            current.append((track, observed))
        labels, counts = np.full((h, w), -1, np.int32), np.zeros((h, w), np.uint16)
        priority = np.full((h, w), -1.0)
        items, drawings, points, output_runs = [], [], 0, 0
        for track, observed in sorted(current, key=lambda pair: pair[0]["id"]):
            if track["hits"] < p["minimum_observations"]:
                continue
            item = copy.deepcopy(observed["item"])
            x, y = observed["origin"]
            binary = observed["mask"] != 0
            height, width = binary.shape
            region = region_from_mask(binary, p["maximum_region_runs"] - output_runs)
            region["image_size"] = [w, h]
            for run in region["runs"]:
                run[0] += y
                run[1] += x
            output_runs += len(region["runs"])
            item.update(
                id=track["id"],
                track_id=track["id"],
                track_key=state["generation"] + ":" + str(track["id"]),
                source_instance_id=item["id"],
                observed=True,
                mask_region=region,
                source_geometry=copy.deepcopy(item["geometry"]),
                geometry={"bbox": [x, y, width, height]},
                label_id=track["id"],
                track_hits=track["hits"],
            )
            item["metrics"] = {
                **item.get("metrics", {}),
                "center_x": float(observed["center"][0]),
                "center_y": float(observed["center"][1]),
                "mask_area": observed["area"],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
            items.append(item)
            counts[y : y + height, x : x + width] += binary
            wins = binary & (item["score"] > priority[y : y + height, x : x + width])
            labels[y : y + height, x : x + width][wins] = track["id"]
            priority[y : y + height, x : x + width][wins] = item["score"]
            if self.generate_graphics:
                color = COLORS[(track["id"] - 1) % len(COLORS)]
                contours, _ = cv2.findContours(
                    binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if points + len(contour) > p["graphics_maximum_points"]:
                        break
                    points += len(contour)
                    drawings.append(
                        graphic_object(
                            "polyline", contour.reshape(-1, 2) + [x, y], color, closed=True
                        )
                    )
                drawings.append(
                    graphic_object("label", [[x, max(14, y)]], color, text=f"Track {track['id']}")
                )
        report = {
            "model": "instance_mask_tracking_v1",
            "clock": clock,
            "timestamp_ms": time_ms,
            "model_identity": self.extra_context(inputs),
            "associations": associations,
            "expired_ids": expired,
            "ambiguous_instance_ids": [
                observations[c]["item"]["id"] for c in sorted(ambiguous_columns)
            ],
            "active": [
                {
                    "id": t["id"],
                    "status": "observed" if t["misses"] == 0 else "predicted",
                    "center": t["center"].tolist(),
                    "velocity_px_per_second": t["velocity"].tolist(),
                    "hits": t["hits"],
                    "missed_frames": t["misses"],
                    "last_observed_ticket": copy.deepcopy(t["last_ticket"]),
                }
                for t in tracks
            ],
            "observed_masks_only": True,
            "prediction_model": "constant_velocity_nearest_pixel_translation",
            "iou_pixels": p["maximum_iou_pixels"] - budget[0],
            "overlap_pixels": int(np.count_nonzero(counts > 1)),
        }
        output = collection("boxes", items, image, self.input_payloads["image"])
        output["instance_masks"] = {
            "model": MASK_MODEL,
            "encoding": "image_region_v1",
            "overlap_preserved": True,
        }
        state.update(tracks=tracks, next_id=next_id, time_ms=time_ms, clock=clock)
        self._track_drawings = tuple(drawings)
        return SequenceStep(
            {
                "image": image,
                "tracks": output,
                "labels": labels,
                "overlap_count": counts,
                "count": len(items),
                "report": report,
            },
            details={"active_tracks": len(tracks)},
            graphics={"Instance track IDs": label_color_map(labels)}
            if self.generate_graphics
            else {},
        )

    def finalize_result(self, result, inputs):
        result = super().finalize_result(result, inputs)
        if self.generate_graphics and result.outputs["tracks"] is not NO_VALUE:
            result.drawings["instance tracks"] = GraphicDrawing(self._track_drawings)
        return result

    def reset_runtime(self):
        super().reset_runtime()
        self._track_drawings = ()
