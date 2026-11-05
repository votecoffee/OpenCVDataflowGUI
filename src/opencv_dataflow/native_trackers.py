"""Single-target CSRT and KCF box tracking in ticketed image sequences."""

import cv2
import numpy as np

from .core import Port
from .features import overlay_image
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .poses import local_offset, make_pose
from .processing import finite, image_value, integer
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    NativeSequenceBlock,
    SequenceStep,
)

TRACKER_DEFAULTS = {
    **SEQUENCE_DEFAULTS,
    "warmup": 1,
    "initial_bounds": [],
    "max_misses": 3,
    "minimum_contrast": 2.0,
    "maximum_center_jump": 0.0,
    "draw_overlay": True,
}
TRACKER_HELP = {
    **SEQUENCE_HELP,
    "initial_bounds": "First [x,y,width,height] box. A wired seed overrides it.",
    "max_misses": "Consecutive missed updates before discarding the model and waiting for a seed.",
    "minimum_contrast": "Minimum grayscale standard deviation inside the box. Zero disables.",
    "maximum_center_jump": "Maximum motion since the last accepted update. Zero disables.",
    "max_bytes_mb": "Budget for estimated tracker state. Temporary buffers are extra.",
}


def box_in_image(bounds, shape, minimum):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("Tracker box must be [x, y, width, height]")
    height, width = shape[:2]
    x, y, w, h = (integer(v, "Box coordinate", 0, max(width, height)) for v in bounds)
    if w < minimum or h < minimum or x + w > width or y + h > height:
        raise ValueError("Tracker box must fit inside the image and meet the minimum dimensions")
    return x, y, w, h


def box_center(bounds):
    x, y, w, h = bounds
    return np.array([x + (w - 1) / 2, y + (h - 1) / 2])


def box_contrast(image, bounds):
    x, y, w, h = bounds
    gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    return float(cv2.meanStdDev(gray)[1][0, 0])


class NativeBoxTracker(NativeSequenceBlock):
    subtype = "Object tracking"
    inputs = (*NativeSequenceBlock.inputs, Port("seed", "data", required=False))
    data_outputs = (
        Port("bounds", "data"),
        Port("points", "data"),
        Port("pose", "data"),
        Port("tracked_region", "data", visual_type="region"),
        Port("overlay", required=False),
    )
    defaults, parameter_help = TRACKER_DEFAULTS, TRACKER_HELP

    def new_tracker(self, image, bounds, mask, state):
        raise NotImplementedError

    def pending(self, state, phase, reason, learned=False):
        return SequenceStep(
            dict.fromkeys(self.flow_primary),
            ready=False,
            learned=learned,
            details={
                "phase": phase,
                "reason": reason,
                "misses": state.get("misses", 0),
                "track_id": state.get("track_id"),
            },
        )

    def step(self, inputs, state, mask):
        source, p = inputs["image"], self.parameters
        image = image_value(source, (np.uint8,), canonical=True)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        contrast_minimum = finite(p["minimum_contrast"], "Minimum contrast", 0)
        jump_limit = finite(p["maximum_center_jump"], "Maximum center jump", 0)
        misses_limit = integer(p["max_misses"], "Maximum misses", 1, 1000)
        if "native" not in state:
            if "seed" in inputs:
                seed = inputs["seed"]
                require_same_frame((source, ValuePacket(None, self.input_payloads["seed"])))
                if isinstance(seed, dict):
                    seed = seed.get("bounds")
            elif state.get("misses", 0) >= misses_limit:
                return self.pending(state, "seed_required", "tracker_expired")
            elif p["initial_bounds"]:
                seed = p["initial_bounds"]
            elif mask is not None and mask.any():
                seed = cv2.boundingRect(mask)
            else:
                return self.pending(state, "seed_required", "no_seed")
            bounds = box_in_image(seed, image.shape, minimum=8)
            if box_contrast(image, bounds) < contrast_minimum:
                return self.pending(state, "seed_required", "seed_low_contrast")
            tracker = self.new_tracker(image, bounds, mask, state)
            tracker.init(image, bounds)
            revision = state.get("track_revision", 0) + 1
            state.update(
                native=tracker,
                bounds=bounds,
                misses=0,
                learning=0,
                track_revision=revision,
                track_id=f"{state['generation']}:{revision}",
            )
            return self.pending(state, "initialized", "awaiting_update", learned=True)

        located, bounds = state["native"].update(image)
        reason = "target_missing"
        if located:
            try:
                bounds = box_in_image(bounds, image.shape, minimum=1)
            except ValueError:
                located, reason = False, "bounds_outside_image"
        if located:
            contrast = box_contrast(image, bounds)
            jump = float(np.linalg.norm(box_center(bounds) - box_center(state["bounds"])))
            if contrast < contrast_minimum:
                located, reason = False, "target_low_contrast"
            elif jump_limit and jump > jump_limit:
                located, reason = False, "center_jump_above_maximum"
        if not located:
            state["misses"] += 1
            if state["misses"] >= misses_limit:
                state.pop("native")
                state.pop("native_bytes")
            return self.pending(state, "lost", reason)

        state.update(bounds=bounds, misses=0)
        center = box_center(bounds)
        size = [image.shape[1], image.shape[0]]
        pose = make_pose(
            local_offset(*center, 0), size, self.input_payloads["image"], track_id=state["track_id"]
        )
        x, y, w, h = bounds
        overlay = overlay_image(source) if p["draw_overlay"] else None
        if overlay is not None:
            cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (70, 220, 140), 2)
        return SequenceStep(
            {
                "bounds": list(bounds),
                "points": [center.tolist()],
                "pose": pose,
                "tracked_region": {
                    "model": "image_region_v1",
                    "image_size": size,
                    "shape": "rectangle",
                    "rectangle": list(bounds),
                },
                "overlay": overlay if overlay is not None else NO_VALUE,
            },
            details={
                "phase": "tracking",
                "misses": 0,
                "box_contrast": contrast,
                "track_id": state["track_id"],
            },
            graphics={"tracked box": overlay} if overlay is not None else {},
        )


CSRT_INTEGER_PARAMS = {
    "admm_iterations": (4, 1, 64),
    "background_ratio": (2, 1, 16),
    "histogram_bins": (16, 2, 64),
    "num_hog_channels_used": (18, 1, 31),
    "number_of_scales": (33, 3, 63),
}
CSRT_FLOAT_PARAMS = {
    "cheb_attenuation": (45.0, 1.0, 200.0),
    "filter_lr": (0.02, 0.0, 1.0),
    "gsl_sigma": (1.0, 0.001, 128.0),
    "histogram_lr": (0.04, 0.0, 1.0),
    "hog_clip": (0.2, 0.001, 1.0),
    "hog_orientations": (9.0, 1.0, 32.0),
    "kaiser_alpha": (3.75, 0.0, 20.0),
    "padding": (3.0, 0.0, 10.0),
    "psr_threshold": (0.035, 0.0, 1.0),
    "scale_lr": (0.025, 0.0, 1.0),
    "scale_model_max_area": (512.0, 16.0, 4096.0),
    "scale_sigma_factor": (0.25, 0.001, 2.0),
    "scale_step": (1.02, 1.001, 1.1),
    "template_size": (200.0, 16.0, 512.0),
    "weights_lr": (0.02, 0.0, 1.0),
}
CSRT_FLAGS = {
    "use_channel_weights": True,
    "use_color_names": True,
    "use_gray": True,
    "use_hog": True,
    "use_rgb": False,
    "use_segmentation": True,
}
KCF_DESCRIPTORS = {"none": 0, "gray": 1, "color_names": 2, "gray_and_color_names": 3}
KCF_FLAGS = {"compress_feature": True, "resize": True, "split_coeff": True, "wrap_kernel": False}
KCF_INTEGER_PARAMS = {"compressed_size": (2, 1, 11), "max_patch_size": (6400, 64, 262144)}
KCF_FLOAT_PARAMS = {
    "detect_thresh": (0.5, 0.0, 1.0),
    "interp_factor": (0.075, 0.0, 1.0),
    "lambda_": (0.0001, 1e-9, 10.0),
    "output_sigma_factor": (0.0625, 0.001, 2.0),
    "pca_learning_rate": (0.15, 0.0, 1.0),
    "sigma": (0.2, 0.001, 2.0),
}


def tracker_parameters(p, native, integers, floats, flags):
    for name, (_, low, high) in integers.items():
        setattr(native, name, integer(p[name], name, low, high))
    for name, (_, low, high) in floats.items():
        value = finite(p[name], name, low)
        if value > high:
            raise ValueError(f"{name} must be at most {high}")
        setattr(native, name, value)
    for name in flags:
        if type(p[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
        setattr(native, name, p[name])


class CSRTSequence(NativeBoxTracker):
    type_id, title = "sequence.track_csrt", "Track object box (CSRT)"
    defaults = {
        **TRACKER_DEFAULTS,
        **CSRT_FLAGS,
        "window_function": "hann",
        **{name: item[0] for name, item in (CSRT_INTEGER_PARAMS | CSRT_FLOAT_PARAMS).items()},
    }
    parameter_choices = {**SEQUENCE_CHOICES, "window_function": ("hann", "cheb", "kaiser")}

    def new_tracker(self, image, bounds, mask, state):
        p = self.parameters
        options = cv2.TrackerCSRT.Params()
        tracker_parameters(p, options, CSRT_INTEGER_PARAMS, CSRT_FLOAT_PARAMS, CSRT_FLAGS)
        options.window_function = p["window_function"]
        if not any((options.use_hog, options.use_color_names, options.use_gray, options.use_rgb)):
            raise ValueError("CSRT requires at least one enabled feature family")
        _, _, w, h = bounds
        padded = int(max(w, h) + options.padding * np.sqrt(w * h) + 1)
        if padded > 4096:
            raise ValueError("CSRT target and padding exceed the patch dimension limit")
        budget = int(options.template_size**2 * 384 + padded**2 * 8)
        budget += int(options.scale_model_max_area * options.number_of_scales * 128)
        self.reserve_native_state(state, budget)
        tracker = cv2.TrackerCSRT.create(options)
        if mask is not None and options.use_segmentation:
            x, y, w, h = bounds
            initial = mask[y : y + h, x : x + w].astype(np.float32) / 255
            if not initial.any():
                raise ValueError("CSRT initial mask has no selected pixels inside the seed")
            tracker.setInitialMask(initial)
        return tracker


class KCFSequence(NativeBoxTracker):
    type_id, title = "sequence.track_kcf", "Track object box (KCF)"
    defaults = {
        **TRACKER_DEFAULTS,
        **KCF_FLAGS,
        "desc_npca": "gray",
        "desc_pca": "color_names",
        **{name: item[0] for name, item in (KCF_INTEGER_PARAMS | KCF_FLOAT_PARAMS).items()},
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "desc_npca": tuple(KCF_DESCRIPTORS),
        "desc_pca": tuple(KCF_DESCRIPTORS),
    }

    def new_tracker(self, image, bounds, mask, state):
        p = self.parameters
        options = cv2.TrackerKCF.Params()
        tracker_parameters(p, options, KCF_INTEGER_PARAMS, KCF_FLOAT_PARAMS, KCF_FLAGS)
        options.desc_npca = KCF_DESCRIPTORS[p["desc_npca"]]
        options.desc_pca = KCF_DESCRIPTORS[p["desc_pca"]]
        if not (options.desc_npca or options.desc_pca):
            raise ValueError("KCF requires at least one descriptor")
        dimensions = (0, 1, 10, 11)[options.desc_pca]
        if options.compress_feature and dimensions and options.compressed_size > dimensions:
            raise ValueError("KCF compressed size exceeds its PCA descriptor dimensions")
        _, _, w, h = bounds
        self.reserve_native_state(state, 4 * w * h * 512 + 4096)
        return cv2.TrackerKCF.create(options)


NATIVE_TRACKER_BLOCKS = (CSRTSequence, KCFSequence)
