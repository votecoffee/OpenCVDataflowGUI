"""Seeded native MIL tracking with explicit readiness, loss guards and bounded work."""

import math

import cv2
import numpy as np

from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .native_trackers import TRACKER_DEFAULTS, TRACKER_HELP, NativeBoxTracker, box_center
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .selection import contains
from .sequence import SEQUENCE_CHOICES

# Every member of TrackerMIL.Params in the installed OpenCV 5 binding.
NATIVE_FIELDS = {
    "init_positive_radius": ("samplerInitInRadius", 3.0, NumericRange(1, 32)),
    "init_negative_samples": ("samplerInitMaxNegNum", 65, NumericRange(1, 4096, whole=True)),
    "search_radius": ("samplerSearchWinSize", 25.0, NumericRange(2, 128)),
    "track_positive_radius": ("samplerTrackInRadius", 4.0, NumericRange(1, 32)),
    "track_positive_samples": (
        "samplerTrackMaxPosNum",
        100000,
        NumericRange(1, 100000, whole=True),
    ),
    "track_negative_samples": ("samplerTrackMaxNegNum", 65, NumericRange(1, 4096, whole=True)),
    "features": ("featureSetNumFeatures", 250, NumericRange(1, 1024, whole=True)),
}


class MILSequence(NativeBoxTracker):
    type_id, title = "sequence.track_mil", "Track object box (MIL)"
    preview_safe = False
    defaults = {
        **TRACKER_DEFAULTS,
        **{name: field[1] for name, field in NATIVE_FIELDS.items()},
        "selection_mode": "center",
        "maximum_work": 4_000_000,
    }
    parameter_choices = {**SEQUENCE_CHOICES, "selection_mode": ("center", "seed_only")}
    parameter_ranges = {
        **{name: field[2] for name, field in NATIVE_FIELDS.items()},
        "maximum_work": NumericRange(1, 64_000_000, whole=True),
        "warmup": NumericRange(1, 10000, whole=True),
        "max_streams": NumericRange(1, 64, whole=True),
        "max_bytes_mb": NumericRange(0.01, 4096),
        "max_gap_ms": NumericRange(0, 3_600_000),
        "max_misses": NumericRange(1, 1000, whole=True),
        "minimum_contrast": NumericRange(0, 255),
        "maximum_center_jump": NumericRange(0, 65536),
    }
    parameter_help = {
        **TRACKER_HELP,
        "initial_bounds": "Initial [x,y,width,height], at least 8×8. Wired Seed takes precedence.",
        "selection_mode": "Mask/region bounds seed the model; Center also gates accepted centers.",
        "init_positive_radius": "Native positive-sampling radius, in local pixels.",
        "init_negative_samples": "Native cap on negative initialization samples.",
        "search_radius": "Native local search window radius; larger windows increase work.",
        "track_positive_radius": (
            "Native tracking sampler control; SDK versions may use init sampling for updates."
        ),
        "track_positive_samples": (
            "Native tracking positive-sample cap; SDK policy determines actual use."
        ),
        "track_negative_samples": (
            "Native tracking negative-sample cap; SDK policy determines actual use."
        ),
        "features": (
            "Native Haar feature count. MIL returns no probability/confidence through this API."
        ),
        "maximum_work": (
            "Conservative feature/candidate product bound before native initialization."
        ),
        "draw_overlay": (
            "Enable drawable results and optional raster Overlay; "
            "Lean/off skips unrequested graphics."
        ),
        "max_bytes_mb": (
            "Conservative model/workspace estimate, not a native allocator or hard RSS limit."
        ),
    }
    input_help = {
        "image": (
            "uint8 gray/RGB/BGR/RGBA/BGRA, up to 2.1M pixels. MIL receives canonical grayscale."
        ),
        "seed": (
            "Optional [x,y,w,h] or {bounds:[...]}, in this image's coordinates, "
            "when no model exists."
        ),
        "mask": "Seed support and optional accepted-center gate; not a native MIL training mask.",
        "region": "Rectangle/circle/polygon selection, intersected with Mask; same semantics.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs.remove("overlay")

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in self.parameter_ranges:
            if type(p[name]) not in (int, float):
                raise ValidationError(f"{name} must be a JSON number", name)
        if p["search_radius"] <= max(p["init_positive_radius"], p["track_positive_radius"]):
            raise ValidationError(
                "Search radius must exceed both positive-sampling radii",
                "search_radius",
                "init_positive_radius",
                "track_positive_radius",
            )
        if type(p["draw_overlay"]) is not bool:
            raise ValidationError("Draw overlay must be boolean", "draw_overlay")
        bounds = p["initial_bounds"]
        if not isinstance(bounds, list) or len(bounds) not in (0, 4):
            raise ValidationError(
                "Initial bounds must be empty or [x,y,width,height]", "initial_bounds"
            )
        for index, value in enumerate(bounds):
            NumericRange(8 if index > 1 else 0, 32766, whole=True).check(
                value, "Initial bound", "initial_bounds"
            )

    def prepare_image(self, source):
        if source.shape[0] * source.shape[1] > 2_100_000 or max(source.shape[:2]) > 32766:
            raise ValidationError("MIL images are limited to 2.1M pixels and 32766 per axis")
        return np.ascontiguousarray(gray_image(source, (np.uint8,)))

    def location_allowed(self, bounds, mask):
        return self.parameters["selection_mode"] == "seed_only" or contains(
            mask, box_center(bounds)
        )

    def tracked_overlay(self, source, bounds):
        if "overlay" not in self.exposed_outputs:
            return None
        return super().tracked_overlay(source, bounds)

    def new_tracker(self, image, bounds, mask, state):
        p = self.parameters
        side = 2 * math.ceil(p["search_radius"]) + 1
        positives = max(
            (2 * math.ceil(p[key]) + 1) ** 2
            for key in ("init_positive_radius", "track_positive_radius")
        )
        samples = side**2 + positives + p["init_negative_samples"] + p["track_negative_samples"]
        work = samples * p["features"]
        if work > p["maximum_work"]:
            raise ValidationError("MIL feature/candidate work exceeds Maximum work", "maximum_work")
        # Includes a conservative integral-image, response and model allowance.
        budget = (
            (image.shape[0] + 1) * (image.shape[1] + 1) * 16 + work * 16 + p["features"] * 65536
        )
        self.reserve_native_state(state, budget)
        options = cv2.TrackerMIL.Params()
        for name, (native, _, _) in NATIVE_FIELDS.items():
            value = p[name]
            setattr(
                options, native, int(value) if self.parameter_ranges[name].whole else float(value)
            )
        state["mil_work"] = work
        return cv2.TrackerMIL.create(options)

    def process(self, inputs):
        result = super().process(inputs)
        status = result.outputs["status"]
        failed = status["phase"] == "lost" or status.get("reason") == "tracker_expired"
        result.passed = not failed
        if failed:
            result.fail_reasons = [{"code": "mil_" + status["reason"], "message": status["reason"]}]
        status["native_confidence_available"] = False
        status["fixed_size_box"] = True
        with self._memory_lock:
            state = self.memory["streams"].get(status["context"], {})
            status["last_accepted_bounds"] = list(state["bounds"]) if "bounds" in state else None
        for payload in result.payloads.values():
            if "sequence" in payload:
                payload["sequence"].update(
                    native_confidence_available=False,
                    fixed_size_box=True,
                    last_accepted_bounds=status["last_accepted_bounds"],
                )
        if self.generate_graphics and self.parameters["draw_overlay"]:
            color = (40, 70, 240) if failed else (60, 200, 100)
            objects = [
                graphic_object(
                    "text",
                    [(5, min(18, inputs["image"].shape[0] - 1))],
                    color,
                    text=f"MIL: {status['phase']}",
                )
            ]
            if result.outputs["ready"]:
                objects.append(graphic_box(*result.outputs["bounds"], color))
            result.drawings = {"MIL tracking": GraphicDrawing(tuple(objects))}
        result.message += " · " + status["phase"]
        return result
