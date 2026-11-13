"""Copyable single-target MeanShift/CamShift state with explicit teaching and loss gates."""

import copy
import math

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .images import image_format, opencv_image
from .native_trackers import box_center, box_in_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .poses import local_offset, make_pose
from .processing import image_value
from .selection import contains, selection_mask
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)

CRITERIA = {
    "count_and_epsilon": cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
    "count": cv2.TERM_CRITERIA_COUNT,
    "epsilon": cv2.TERM_CRITERIA_EPS,
}


class ColorDistributionTracker(SequenceBlock):
    type_id, title, subtype = "sequence.color_tracker", "Track region by color", "Object tracking"
    inputs = (
        *SequenceBlock.inputs,
        Port("seed", "data", required=False),
        Port("reseed", "data", required=False),
    )
    data_outputs = (
        Port("image"),
        Port("bounds", "data"),
        Port("points", "data"),
        Port("pose", "data", required=False),
        Port("tracked_region", "data", visual_type="region"),
        Port("confidence", "number"),
        Port("probability"),
        Port("geometry", "data"),
        Port("histogram", "data", required=False),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        "method": "cam_shift",
        "initial_bounds": [],
        "histogram_channels": "hue_saturation",
        "hue_bins": 32,
        "saturation_bins": 32,
        "minimum_saturation": 40,
        "minimum_value": 32,
        "maximum_value": 255,
        "minimum_alpha": 1,
        "minimum_seed_pixels": 16,
        "termination": "count_and_epsilon",
        "max_iterations": 20,
        "epsilon": 1.0,
        "minimum_confidence": 0.1,
        "support_threshold": 32,
        "minimum_support_fraction": 0.05,
        "minimum_area_ratio": 0.25,
        "maximum_area_ratio": 4.0,
        "maximum_center_jump": 0.0,
        "max_misses": 3,
        "learning_rate": 0.0,
        "pose_mode": "center_only",
        "minimum_axis_ratio": 1.1,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "method": ("mean_shift", "cam_shift"),
        "histogram_channels": ("hue", "hue_saturation"),
        "termination": tuple(CRITERIA),
        "pose_mode": ("center_only", "principal_axis"),
    }
    parameter_ranges = {
        "hue_bins": NumericRange(2, 180, whole=True),
        "saturation_bins": NumericRange(2, 256, whole=True),
        **{
            k: NumericRange(0, 255, whole=True)
            for k in ("minimum_saturation", "minimum_value", "maximum_value", "minimum_alpha")
        },
        "minimum_seed_pixels": NumericRange(1, 2_100_000, whole=True),
        "max_iterations": NumericRange(1, 100, whole=True),
        "epsilon": NumericRange(0, 10),
        "minimum_confidence": NumericRange(0, 1),
        "support_threshold": NumericRange(1, 255, whole=True),
        "minimum_support_fraction": NumericRange(0, 1),
        "minimum_area_ratio": NumericRange(0.001, 1000),
        "maximum_area_ratio": NumericRange(0.001, 1000),
        "maximum_center_jump": NumericRange(0, 32766),
        "max_misses": NumericRange(1, 1000, whole=True),
        "learning_rate": NumericRange(0, 1),
        "minimum_axis_ratio": NumericRange(1, 100),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "method": "MeanShift keeps window size; CamShift estimates a color-mass ellipse and scale.",
        "initial_bounds": "Initial [x,y,w,h]; read before teaching or after reset/reseed.",
        "histogram_channels": "Hue or joint hue/saturation, peak-normalized to 255.",
        "hue_bins": "Hue bin count over [0,180), in native 8-bit HSV units.",
        "saturation_bins": "Saturation bins over [0,256), ignored by hue-only models.",
        "minimum_saturation": "Discard lower-saturation pixels in teaching and backprojection.",
        "minimum_value": "Inclusive minimum native HSV value; reject dark unreliable hue.",
        "maximum_value": "Inclusive maximum HSV value; optionally exclude clipped highlights.",
        "minimum_alpha": "For BGRA/RGBA, reject lower stored opacity; ignored without alpha.",
        "minimum_seed_pixels": "Minimum eligible pixels in the seed selection before teaching.",
        "termination": "Native TermCriteria flags; inactive controls retain their values.",
        "max_iterations": "Native count limit when enabled; OpenCV caps MeanShift at 100.",
        "epsilon": "Native convergence displacement in pixels when enabled.",
        "minimum_confidence": "Mean backprojection/255; not an accuracy probability.",
        "support_threshold": "Count box pixels meeting this backprojection threshold.",
        "minimum_support_fraction": "Minimum fraction of the box meeting Support threshold.",
        "minimum_area_ratio": "Minimum area relative to the original seed box.",
        "maximum_area_ratio": "Maximum relative area; CamShift uses equivalent-ellipse area.",
        "maximum_center_jump": "Maximum center motion from last accepted update; zero disables.",
        "max_misses": "Misses before expiry; reset/reseed is then required to teach again.",
        "learning_rate": "Blend accepted updates; zero freezes. Lost frames never learn.",
        "pose_mode": "Zero-angle center pose, or elongated CamShift principal axis.",
        "minimum_axis_ratio": "Minimum axial pose ratio; orientation has 180° ambiguity.",
    }
    input_help = {
        "image": "uint8 BGR/RGB/BGRA/RGBA or native HSV, up to 2.1M pixels.",
        "seed": "Initial [x,y,w,h] or image_region_v1 shape on this grid; read when teaching.",
        "reseed": "Rising edge teaches from current seed; held true does not repeat.",
        "mask": "Current image work area, intersected with region for teaching and probability.",
        "region": "Current work area; use a separate shape on Seed for the taught target.",
    }
    process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "seed", "reset"]
        self.exposed_outputs = ["image", "bounds", "pose", "confidence", "ready", "status"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_value"] > p["maximum_value"]:
            raise ValidationError(
                "Minimum value must not exceed maximum", "minimum_value", "maximum_value"
            )
        if p["minimum_area_ratio"] > p["maximum_area_ratio"]:
            raise ValidationError(
                "Minimum area ratio must not exceed maximum",
                "minimum_area_ratio",
                "maximum_area_ratio",
            )
        if p["pose_mode"] == "principal_axis" and p["method"] != "cam_shift":
            raise ValidationError("Principal-axis poses require CamShift", "pose_mode", "method")

    def appearance(self, source, work_mask):
        pixels = image_value(source, (np.uint8,))
        if (
            min(pixels.shape[:2]) < 2
            or max(pixels.shape[:2]) > 16384
            or pixels.shape[0] * pixels.shape[1] > 2_100_000
        ):
            raise ValidationError("Use 2×2..16384 per axis and at most 2.1M color pixels")
        encoding = image_format(source)
        if encoding == "hsv" and pixels.ndim == 3 and pixels.shape[2] == 3:
            if np.any(pixels[..., 0] >= 180):
                raise ValidationError("Native uint8 HSV requires hue in [0,179]")
            hsv = np.ascontiguousarray(pixels)
        elif encoding in ("bgr", "rgb", "bgra", "rgba") and pixels.ndim == 3:
            if pixels.shape[2] != (4 if encoding.endswith("a") else 3):
                raise ValidationError("Color encoding does not match the channel layout")
            hsv = cv2.cvtColor(opencv_image(source)[..., :3], cv2.COLOR_BGR2HSV)
        else:
            raise ValidationError("Use uint8 color or native HSV for color-distribution tracking")
        p = self.parameters
        eligible = (hsv[..., 1] >= p["minimum_saturation"]) & (hsv[..., 2] >= p["minimum_value"])
        eligible &= hsv[..., 2] <= p["maximum_value"]
        if pixels.shape[2] == 4:
            eligible &= pixels[..., 3] >= p["minimum_alpha"]
        if work_mask is not None:
            eligible &= work_mask != 0
        return hsv, eligible

    def histogram_options(self):
        p = self.parameters
        return (
            ([0], [int(p["hue_bins"])], [0, 180])
            if p["histogram_channels"] == "hue"
            else ([0, 1], [int(p["hue_bins"]), int(p["saturation_bins"])], [0, 180, 0, 256])
        )

    def teach_histogram(self, hsv, mask):
        channels, bins, ranges = self.histogram_options()
        hist = cv2.calcHist([hsv], channels, mask.astype(np.uint8) * 255, bins, ranges)
        return hist * (255.0 / float(hist.max()))

    def pending(self, state, phase, reason, **details):
        return SequenceStep(
            dict.fromkeys(self.flow_primary, NO_VALUE),
            ready=False,
            learned=False,
            details={
                "phase": phase,
                "reason": reason,
                "misses": state.get("misses", 0),
                "track_id": state.get("track_id"),
                "confidence": 0.0,
                **details,
            },
        )

    def seed_selection(self, inputs, source, eligible):
        p = self.parameters
        if "seed" in inputs:
            seed = inputs["seed"]
            try:
                require_same_frame([source, ValuePacket(None, self.input_payloads["seed"])])
            except ValueError as exc:
                raise ValidationError("Seed coordinate frame differs from image") from exc
        else:
            seed = p["initial_bounds"]
        if isinstance(seed, dict) and seed.get("model") == "image_region_v1":
            selected = selection_mask({"image": source, "region": seed}, source) != 0
            if not selected.any():
                return None, None
            bounds = cv2.boundingRect(selected.astype(np.uint8))
            try:
                box_in_image(bounds, source.shape, minimum=2)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        else:
            if isinstance(seed, dict):
                seed = seed.get("bounds")
            if seed is None or (isinstance(seed, (list, tuple)) and not seed):
                return None, None
            try:
                bounds = box_in_image(seed, source.shape, minimum=2)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            selected = np.zeros(source.shape[:2], bool)
            x, y, w, h = bounds
            selected[y : y + h, x : x + w] = True
        return bounds, selected & eligible

    def step(self, inputs, state, mask):
        source, p = inputs["image"], self.parameters
        hsv, eligible = self.appearance(source, mask)
        reseed = inputs.get("reseed", False)
        if type(reseed) is not bool:
            raise ValidationError("Reseed must be boolean")
        rising = reseed and not state.get("last_reseed", False)
        state["last_reseed"] = reseed
        if rising:
            state.pop("histogram", None)
            state["expired"] = False
        if "histogram" not in state:
            if state.get("expired"):
                return self.pending(state, "expired", "explicit_reseed_required")
            bounds, selected = self.seed_selection(inputs, source, eligible)
            if bounds is None:
                return self.pending(state, "seed_required", "no_seed")
            count = int(np.count_nonzero(selected))
            if count < p["minimum_seed_pixels"]:
                return self.pending(
                    state, "seed_rejected", "insufficient_seed_pixels", seed_pixels=count
                )
            revision = state.get("track_revision", 0) + 1
            state.update(
                histogram=self.teach_histogram(hsv, selected),
                bounds=tuple(bounds),
                initial_area=bounds[2] * bounds[3],
                center=box_center(bounds).tolist(),
                misses=0,
                learning=0,
                track_revision=revision,
                track_id=f"{state['generation']}:{revision}",
                trained_ticket=copy.deepcopy(self.input_payloads["image"].get("ticket")),
                seed_pixels=count,
                model_updates=0,
            )
            result = self.pending(state, "initialized", "awaiting_update", seed_pixels=count)
            result.learned = True
            return result
        channels, _, ranges = self.histogram_options()
        probability = cv2.calcBackProject([hsv], channels, state["histogram"], ranges, 1)
        probability[~eligible] = 0
        criteria = (CRITERIA[p["termination"]], int(p["max_iterations"]), float(p["epsilon"]))
        axes, angle, native_box = None, None, None
        if p["method"] == "mean_shift":
            iterations, bounds = cv2.meanShift(probability, tuple(state["bounds"]), criteria)
            center = box_center(bounds)
            area = bounds[2] * bounds[3]
        else:
            native_box, bounds = cv2.CamShift(probability, tuple(state["bounds"]), criteria)
            center = np.asarray(native_box[0], float)
            axes, angle = list(native_box[1]), float(native_box[2])
            area = math.pi * axes[0] * axes[1] / 4
            iterations = None
        x, y, w, h = map(int, bounds)
        fits = (
            x >= 0
            and y >= 0
            and w > 0
            and h > 0
            and x + w <= source.shape[1]
            and y + h <= source.shape[0]
        )
        confidence = float(probability[y : y + h, x : x + w].mean() / 255) if fits else 0.0
        support = (
            float(np.mean(probability[y : y + h, x : x + w] >= p["support_threshold"]))
            if fits
            else 0.0
        )
        ratio = area / state["initial_area"]
        jump = float(np.linalg.norm(center - np.asarray(state["center"])))
        reason = None
        if w <= 0 or h <= 0 or area <= 0:
            reason = "no_probability_mass"
        elif not fits:
            reason = "bounds_outside_image"
        elif confidence <= 0:
            reason = "no_probability_mass"
        elif confidence < p["minimum_confidence"]:
            reason = "low_probability"
        elif support < p["minimum_support_fraction"]:
            reason = "insufficient_color_support"
        elif not p["minimum_area_ratio"] <= ratio <= p["maximum_area_ratio"]:
            reason = "scale_limit"
        elif p["maximum_center_jump"] and jump > p["maximum_center_jump"]:
            reason = "motion_jump"
        elif not contains(mask, center):
            reason = "outside_selection"
        if reason:
            state["misses"] += 1
            if state["misses"] >= p["max_misses"]:
                state.pop("histogram")
                state["expired"] = True
            return self.pending(
                state,
                "lost",
                reason,
                confidence=confidence,
                support_fraction=support,
                area_ratio=ratio,
                center_jump=jump,
            )
        bounds = box_in_image(list(map(int, bounds)), source.shape, minimum=1)
        state.update(bounds=bounds, center=center.tolist(), misses=0)
        size = [source.shape[1], source.shape[0]]
        orientation_valid = (
            axes is not None and min(axes) > 0 and max(axes) / min(axes) >= p["minimum_axis_ratio"]
        )
        axial_angle = ((angle + (90 if axes[1] > axes[0] else 0) + 90) % 180 - 90) if axes else None
        pose_angle = (
            -axial_angle if p["pose_mode"] == "principal_axis" and orientation_valid else 0.0
        )
        pose = make_pose(
            local_offset(*center, pose_angle),
            size,
            self.input_payloads["image"],
            track_id=state["track_id"],
            orientation=p["pose_mode"],
            ambiguous_180=p["pose_mode"] == "principal_axis",
        )
        if p["pose_mode"] == "principal_axis" and not orientation_valid:
            pose = NO_VALUE
        if axes:
            region = {
                "model": "image_region_v1",
                "image_size": size,
                "shape": "ellipse",
                "center": center.tolist(),
                "axis_lengths": axes,
                "angle_degrees": angle,
            }
        else:
            region = {
                "model": "image_region_v1",
                "image_size": size,
                "shape": "rectangle",
                "rectangle": list(bounds),
            }
        if p["learning_rate"]:
            learning_mask = selection_mask({"image": source, "region": region}, source) != 0
            learning_mask &= eligible
            if np.count_nonzero(learning_mask) >= p["minimum_seed_pixels"]:
                updated = self.teach_histogram(hsv, learning_mask)
                rate = float(p["learning_rate"])
                state["histogram"] = (1 - rate) * state["histogram"] + rate * updated
                state["histogram"] *= 255 / float(state["histogram"].max())
                state["model_updates"] += 1
        geometry = {
            "model": "color_track_geometry_v1",
            "search_bounds": list(bounds),
            "center": center.tolist(),
            "axes": axes,
            "angle_clockwise": angle,
            "orientation_valid": bool(orientation_valid),
            "area": float(area),
            "area_semantics": "equivalent_color_mass_ellipse" if axes else "search_rectangle",
            "track_id": state["track_id"],
        }
        histogram = NO_VALUE
        if "histogram" in self.exposed_outputs:
            histogram = {
                "model": "color_tracker_histogram_v1",
                "channels": p["histogram_channels"],
                "bins": list(state["histogram"].shape),
                "normalization": "peak_255",
                "weights": state["histogram"].tolist(),
                "trained_ticket": state["trained_ticket"],
            }
        return SequenceStep(
            {
                "image": source,
                "bounds": list(bounds),
                "points": [center.tolist()],
                "pose": pose,
                "tracked_region": region,
                "confidence": confidence,
                "probability": probability,
                "geometry": geometry,
                "histogram": histogram,
            },
            details={
                "phase": "tracking",
                "reason": None,
                "misses": 0,
                "confidence": confidence,
                "support_fraction": support,
                "area_ratio": ratio,
                "center_jump": jump,
                "track_id": state["track_id"],
                "native_iterations": iterations,
                "geometry": geometry,
                "model_updates": state["model_updates"],
            },
        )

    def process(self, inputs):
        result = super().process(inputs)
        status = result.outputs["status"]
        failed = status["phase"] in ("lost", "expired", "seed_rejected")
        result.passed = not failed
        result.score = float(status.get("confidence", 0))
        if failed:
            result.fail_reasons = [
                {"code": "color_track_" + status["reason"], "message": status["reason"]}
            ]
        result.message += " · " + status["phase"]
        for name, payload in result.payloads.items():
            if name != "image":
                payload.pop("alpha_representation", None)
                payload.pop("raw_sensor", None)
        if self.generate_graphics:
            color = (60, 200, 100) if not failed else (40, 70, 240)
            objects = [
                graphic_object(
                    "text",
                    [(5, min(18, inputs["image"].shape[0] - 1))],
                    color,
                    text=f"Color track: {status['phase']}",
                )
            ]
            geometry = status.get("geometry")
            if result.outputs["ready"] and geometry:
                if geometry["axes"]:
                    native = (
                        tuple(geometry["center"]),
                        tuple(geometry["axes"]),
                        geometry["angle_clockwise"],
                    )
                    objects.append(
                        graphic_object("polyline", cv2.boxPoints(native), color, closed=True)
                    )
                else:
                    objects.append(graphic_box(*geometry["search_bounds"], color))
            result.drawings = {"color tracker": GraphicDrawing(tuple(objects))}
        return result
