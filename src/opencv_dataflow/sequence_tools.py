"""Copyable background, optical-flow, tracking and reference-image sequences."""

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np

from .core import Block, Port, Result
from .image_assets import decode_image, encode_image
from .images import ImageArray, image_format, opencv_image
from .processing import finite, gray_image, integer
from .selection import contains
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)


def learning(inputs):
    value = inputs.get("learn", True)
    if type(value) is not bool:
        raise ValueError("Learn input must be boolean")
    return value


def same_encoding(image, source):
    return ImageArray(image, image_format(source))


class BackgroundSequence(SequenceBlock):
    type_id, title = "sequence.background", "Running background subtraction"
    data_outputs = (Port("mask"), Port("background"), Port("foreground"))
    inputs = (*SequenceBlock.inputs, Port("learn", "data", required=False))
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 5,
        "method": "absolute",
        "learning_rate": 0.05,
        "threshold": 20.0,
        "deviations": 3.0,
        "minimum_variance": 4.0,
        "learn_foreground": False,
        "stable_activations": 0,
        "max_foreground_fraction": 0.05,
    }
    parameter_choices = {**SEQUENCE_CHOICES, "method": ("absolute", "variance")}
    parameter_help = {
        **SEQUENCE_HELP,
        "learning_rate": "(0,1]; full-frame learning during warmup.",
        "stable_activations": "Optional consecutive low-foreground frames before readiness.",
        "learn_foreground": "After warmup, allow foreground pixels into the background model.",
    }

    def estimate_state_bytes(self, image):
        return image.size * 8 + 4096

    def step(self, inputs, state, mask):
        original = inputs["image"]
        image = opencv_image(original)
        if image.dtype not in (np.uint8, np.uint16, np.float32):
            raise ValueError("Background subtraction supports uint8, uint16 and float32")
        image = image.astype(np.float32)
        p = self.parameters
        rate = finite(p["learning_rate"], "Learning rate", 0)
        threshold = finite(p["threshold"], "Threshold", 0)
        deviations = finite(p["deviations"], "Deviations", 0)
        minimum = finite(p["minimum_variance"], "Minimum variance", 0)
        stable_required = integer(p["stable_activations"], "Stable activations", 0, 10000)
        fraction = finite(p["max_foreground_fraction"], "Foreground fraction", 0)
        if not 0 < rate <= 1 or minimum <= 0 or fraction > 1:
            raise ValueError("Invalid background learning, variance or stability settings")
        learn = learning(inputs)
        if "mean" not in state:
            if not learn:
                return SequenceStep(
                    dict.fromkeys(self.flow_primary, None),
                    False,
                    False,
                    {"reason": "learning_disabled"},
                )
            state["mean"] = image.copy()
            state["variance"] = np.full(image.shape, minimum, np.float32)
        delta = image - state["mean"]
        limit = (
            threshold
            if p["method"] == "absolute"
            else np.maximum(threshold, deviations * np.sqrt(state["variance"]))
        )
        changed = np.abs(delta) > limit
        changed = np.any(changed, axis=2) if changed.ndim == 3 else changed
        raw_mask = changed.astype(np.uint8) * 255
        observed = raw_mask != 0 if mask is None else (raw_mask != 0) & (mask != 0)
        selected_count = raw_mask.size if mask is None else np.count_nonzero(mask)
        foreground_fraction = np.count_nonzero(observed) / max(1, selected_count)
        state["stable"] = state.get("stable", 0) + 1 if foreground_fraction <= fraction else 0
        if learn:
            update_mask = (
                None
                if state["learning"] < int(p["warmup"]) or p["learn_foreground"]
                else cv2.bitwise_not(raw_mask)
            )
            cv2.accumulateWeighted(delta * delta, state["variance"], rate, mask=update_mask)
            np.maximum(state["variance"], minimum, out=state["variance"])
            cv2.accumulateWeighted(image, state["mean"], rate, mask=update_mask)
        if not np.isfinite(state["mean"]).all() or not np.isfinite(state["variance"]).all():
            raise ValueError("Background statistics overflowed; rescale input samples")
        visible_mask = raw_mask.copy()
        if mask is not None:
            visible_mask[mask == 0] = 0
        background = state["mean"]
        if np.issubdtype(original.dtype, np.integer):
            background = np.clip(
                np.rint(background), np.iinfo(original.dtype).min, np.iinfo(original.dtype).max
            )
        background = background.astype(original.dtype)
        # The model is canonical BGR/BGRA; retain correct encoding on output.
        encoding = "gray" if background.ndim == 2 else "bgra" if background.shape[2] == 4 else "bgr"
        foreground = np.asarray(original).copy()
        foreground[visible_mask == 0] = 0
        return SequenceStep(
            {
                "mask": ImageArray(visible_mask, "gray"),
                "background": ImageArray(background, encoding),
                "foreground": same_encoding(foreground, original),
            },
            state.get("stable", 0) >= stable_required,
            learn,
            {
                "foreground_fraction": foreground_fraction,
                "stable_activations": state.get("stable", 0),
            },
            {"learning foreground": ImageArray(visible_mask, "gray")},
        )


@dataclass
class DenseFlow:
    vectors: np.ndarray
    metadata: dict = field(default_factory=dict)

    def describe(self):
        magnitude = np.linalg.norm(self.vectors, axis=2)
        return {
            "model": "dense_flow_v1",
            "shape": list(self.vectors.shape),
            "dtype": str(self.vectors.dtype),
            "mean_magnitude": float(magnitude.mean()),
            "max_magnitude": float(magnitude.max()),
            "coordinates": "previous image grid; displacement previous-to-current, local pixels",
        }


class FarnebackSequence(SequenceBlock):
    type_id, title = "sequence.farneback", "Farneback optical flow"
    data_outputs = (Port("flow", "any"), Port("magnitude"), Port("angle"), Port("visualization"))
    defaults = {
        **SEQUENCE_DEFAULTS,
        "max_gap_ms": 1000.0,
        "pyramid_scale": 0.5,
        "levels": 3,
        "window": 15,
        "iterations": 3,
        "poly_n": "5",
        "poly_sigma": 1.2,
        "gaussian_window": False,
        "initial_flow": "none",
        "visual_max_motion": 10.0,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "poly_n": ("5", "7"),
        "initial_flow": ("none", "previous"),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "visual_max_motion": "Pixels mapped to full brightness; raw vectors are unchanged.",
    }

    def step(self, inputs, state, mask):
        gray = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        scale = finite(p["pyramid_scale"], "Pyramid scale", 0)
        sigma = finite(p["poly_sigma"], "Polynomial sigma", 0)
        visual = finite(p["visual_max_motion"], "Visual motion scale", 0)
        levels = integer(p["levels"], "Pyramid levels", 1, 8)
        window = integer(p["window"], "Window", 3, 255, odd=True)
        iterations = integer(p["iterations"], "Iterations", 1, 100)
        if not 0 < scale < 1 or sigma <= 0 or visual <= 0:
            raise ValueError("Invalid Farneback scale, sigma or visualization range")
        previous = state.get("previous")
        state["previous"] = gray.copy()
        if previous is None:
            return SequenceStep(
                dict.fromkeys(self.flow_primary, None),
                False,
                True,
                {"reason": "needs_previous_image"},
            )
        initial = state.get("flow") if p["initial_flow"] == "previous" else None
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if p["gaussian_window"] else 0
        if initial is not None:
            flags |= cv2.OPTFLOW_USE_INITIAL_FLOW
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            gray,
            None if initial is None else initial.copy(),
            scale,
            levels,
            window,
            iterations,
            int(p["poly_n"]),
            sigma,
            flags,
        )
        if not np.isfinite(flow).all():
            raise ValueError("Optical flow produced nonfinite vectors")
        state["flow"] = flow.copy() if p["initial_flow"] == "previous" else None
        if mask is not None:
            flow[mask == 0] = 0
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv = np.zeros((*gray.shape, 3), np.uint8)
        hsv[..., 0] = np.rint(angle / 2).astype(np.uint8) % 180
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(magnitude * 255 / visual, 0, 255).astype(np.uint8)
        view = ImageArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "bgr")
        field = DenseFlow(flow, copy.deepcopy(self.input_payloads["image"]))
        return SequenceStep(
            {
                "flow": field,
                "magnitude": ImageArray(magnitude, "gray"),
                "angle": ImageArray(angle, "gray"),
                "visualization": view,
            },
            graphics={"flow preview": view},
            details={"coordinates": field.describe()["coordinates"]},
        )


class TrackPointsSequence(SequenceBlock):
    type_id, title = "sequence.track_points", "Track points (Lucas-Kanade)"
    data_outputs = (Port("tracks", "data"), Port("overlay"))
    defaults = {
        **SEQUENCE_DEFAULTS,
        "max_gap_ms": 1000.0,
        "max_points": 200,
        "quality": 0.01,
        "min_distance": 7.0,
        "block_size": 7,
        "gradient_size": "3",
        "harris": False,
        "harris_k": 0.04,
        "window": 21,
        "max_level": 3,
        "iterations": 30,
        "epsilon": 0.01,
        "min_eigenvalue": 0.0001,
        "error_metric": "patch",
        "max_error": 30.0,
        "fb_tolerance": 1.0,
        "reseed_below": 50,
        "min_tracks": 3,
        "min_track_age": 2,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "gradient_size": ("3", "5", "7"),
        "error_metric": ("patch", "min_eigenvalue"),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "fb_tolerance": "Forward/backward pixel tolerance; 0 disables the backward check.",
        "max_error": "Patch-error limit; not applied when requesting minimum eigenvalues.",
    }

    def step(self, inputs, state, mask):
        gray = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        maximum = integer(p["max_points"], "Maximum points", 1, 5000)
        minimum = integer(p["min_tracks"], "Minimum tracks", 1, maximum)
        age_min = integer(p["min_track_age"], "Minimum track age", 2, 10000)
        reseed = integer(p["reseed_below"], "Reseed threshold", 0, maximum)
        quality = finite(p["quality"], "Quality", 0)
        distance = finite(p["min_distance"], "Minimum distance", 0)
        block = integer(p["block_size"], "Corner block size", 3, 63, odd=True)
        harris_k = finite(p["harris_k"], "Harris k", 0)
        window = integer(p["window"], "Tracking window", 3, 255, odd=True)
        level = integer(p["max_level"], "Maximum pyramid level", 0, 8)
        iterations = integer(p["iterations"], "Iterations", 1, 1000)
        epsilon = finite(p["epsilon"], "Epsilon", 0)
        eigen = finite(p["min_eigenvalue"], "Minimum eigenvalue", 0)
        error_max = finite(p["max_error"], "Maximum error", 0)
        fb = finite(p["fb_tolerance"], "Forward/backward tolerance", 0)
        if not 0 < quality <= 1 or epsilon <= 0:
            raise ValueError("Invalid tracking quality or epsilon")
        records, lost, positions, ids, ages = [], [], [], [], []
        previous = state.get("previous")
        points = state.get("points")
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, epsilon)
        flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS if p["error_metric"] == "min_eigenvalue" else 0
        if previous is not None and points is not None and len(points):
            found, valid, errors = cv2.calcOpticalFlowPyrLK(
                previous,
                gray,
                points,
                None,
                winSize=(window, window),
                maxLevel=level,
                criteria=criteria,
                flags=flags,
                minEigThreshold=eigen,
            )
            if found is None:
                lost.extend(state["ids"])
            else:
                backward = backward_valid = None
                if fb:
                    backward, backward_valid, _ = cv2.calcOpticalFlowPyrLK(
                        gray,
                        previous,
                        found,
                        None,
                        winSize=(window, window),
                        maxLevel=level,
                        criteria=criteria,
                        minEigThreshold=eigen,
                    )
                for index, point in enumerate(found.reshape(-1, 2)):
                    old = points[index, 0]
                    error = float(errors[index, 0]) if errors is not None else float("inf")
                    fb_error = (
                        float(np.linalg.norm(backward[index, 0] - old))
                        if backward is not None
                        else 0.0
                    )
                    ok = bool(valid[index, 0]) and np.isfinite(point).all() and np.isfinite(error)
                    ok = ok and (error <= error_max if not flags else error >= eigen)
                    ok = ok and (
                        not fb
                        or (
                            backward is not None
                            and bool(backward_valid[index, 0])
                            and fb_error <= fb
                        )
                    )
                    ok = (
                        ok
                        and 0 <= point[0] < gray.shape[1]
                        and 0 <= point[1] < gray.shape[0]
                        and contains(mask, point)
                    )
                    if not ok:
                        lost.append(state["ids"][index])
                        continue
                    age = state["ages"][index] + 1
                    ident = state["ids"][index]
                    positions.append(point.tolist())
                    ids.append(ident)
                    ages.append(age)
                    records.append(
                        {
                            "id": ident,
                            "point": point.tolist(),
                            "previous_point": old.tolist(),
                            "displacement": (point - old).tolist(),
                            "age": age,
                            "error": error,
                            "fb_error": fb_error,
                        }
                    )
        if len(positions) < reseed or not positions:
            seed_mask = np.full(gray.shape, 255, np.uint8) if mask is None else mask.copy()
            for point in positions:
                cv2.circle(
                    seed_mask,
                    tuple(np.rint(point).astype(int)),
                    max(1, int(np.ceil(distance))),
                    0,
                    -1,
                )
            if maximum > len(positions):
                new = cv2.goodFeaturesToTrack(
                    gray,
                    maximum - len(positions),
                    quality,
                    distance,
                    mask=seed_mask,
                    blockSize=block,
                    gradientSize=int(p["gradient_size"]),
                    useHarrisDetector=p["harris"],
                    k=harris_k,
                )
                for point in [] if new is None else new.reshape(-1, 2):
                    state["next_id"] = state.get("next_id", 0) + 1
                    ident = f"{state['generation']}:{state['next_id']}"
                    positions.append(point.tolist())
                    ids.append(ident)
                    ages.append(1)
        state.update(
            previous=gray.copy(),
            points=np.asarray(positions, np.float32).reshape(-1, 1, 2),
            ids=ids,
            ages=ages,
        )
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for point, age in zip(positions, ages):
            cv2.circle(
                overlay,
                tuple(np.rint(point).astype(int)),
                3,
                (40, 220, 70) if age >= age_min else (40, 180, 240),
                1,
            )
        mature = [record for record in records if record["age"] >= age_min]
        output = {
            "model": "point_tracks_v1",
            "items": mature,
            "lost_ids": lost,
            "generation": state["generation"],
            "coordinates": "current image, local pixels",
        }
        return SequenceStep(
            {"tracks": output, "overlay": ImageArray(overlay, "bgr")},
            len(mature) >= minimum,
            True,
            {
                "tracked": len(mature),
                "active_points": len(positions),
                "reason": "ready" if len(mature) >= minimum else "insufficient_tracks",
            },
            {"tracking preview": ImageArray(overlay, "bgr")},
        )


class ImageMemory(SequenceBlock):
    type_id, title, subtype = (
        "sequence.image_memory",
        "Image memory / trained reference",
        "Image memory",
    )
    supports_reference_training = True
    inputs = (
        *SequenceBlock.inputs,
        Port("learn", "data", required=False),
        Port("reference", "data", required=False),
    )
    data_outputs = (Port("reference"), Port("difference"), Port("image"))
    defaults = {
        **SEQUENCE_DEFAULTS,
        "warmup": 3,
        "mode": "mean",
        "freeze_after_warmup": True,
        "reference_context": "strict",
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "mode": ("first", "latest", "mean"),
        "reference_context": ("strict", "shape_only"),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "reference_context": "Strict checks grid/recipe; shape_only permits reuse.",
        "freeze_after_warmup": "Keep learned image after readiness; disable to keep learning.",
    }

    def __init__(self, **kwargs):
        self.trained_image = None
        super().__init__(**kwargs)

    def estimate_state_bytes(self, image):
        return (image.size * 8 if self.parameters["mode"] == "mean" else image.nbytes) + 4096

    def train_reference(self, image):
        asset = encode_image(image)
        decode_image(asset)
        self.trained_image = asset
        self.reset_runtime()
        self.clear_result()
        return copy.deepcopy(asset)

    def serialize_extra(self):
        return {"trained_image": copy.deepcopy(self.trained_image)} if self.trained_image else {}

    def restore_extra(self, data):
        if data.get("trained_image") is not None:
            decode_image(data["trained_image"])
            self.trained_image = copy.deepcopy(data["trained_image"])

    def extra_context(self, inputs):
        asset = inputs.get("reference", self.trained_image)
        if asset is not None:
            decode_image(asset, load=False)
        return asset["id"] if asset else None

    def step(self, inputs, state, mask):
        source = inputs["image"]
        image = np.asarray(source)
        asset = inputs.get("reference", self.trained_image)
        if "memory" not in state and asset is not None:
            reference_image = decode_image(asset)
            if (
                reference_image.shape != image.shape
                or reference_image.dtype != image.dtype
                or image_format(reference_image) != image_format(source)
            ):
                raise ValueError("Trained reference shape/depth/encoding differs from input")
            if self.parameters["reference_context"] == "strict":
                context = copy.deepcopy(asset["context"])
                bound = self.input_payloads.get("reference", {}).get("recipe_ref")
                if bound:
                    context["recipe_ref"] = bound
                if any(self.input_payloads["image"].get(k) != v for k, v in context.items()):
                    raise ValueError(
                        "Trained reference grid/recipe differs from input; "
                        "explicitly choose shape_only to reuse"
                    )
            state.update(
                memory=np.asarray(reference_image).copy(),
                samples=1,
                seeded=True,
                reference_id=asset["id"],
            )
        learn = learning(inputs)
        frozen = self.parameters["freeze_after_warmup"] and (
            state.get("seeded") or state["learning"] >= int(self.parameters["warmup"])
        )
        update = learn and not frozen
        if update:
            mode = self.parameters["mode"]
            if "memory" not in state or mode == "latest":
                state["memory"] = image.astype(np.float64) if mode == "mean" else image.copy()
                state["samples"] = 1
            elif mode == "mean":
                state["samples"] += 1
                state["memory"] = state["memory"].astype(np.float64)
                state["memory"] += (image.astype(np.float64) - state["memory"]) / state["samples"]
            if mode != "first" or "reference_ticket" not in state:
                state["reference_ticket"] = copy.deepcopy(
                    self.input_payloads["image"].get("ticket")
                )
        if "memory" not in state:
            return SequenceStep(
                dict.fromkeys(self.flow_primary, None),
                False,
                False,
                {"reason": "learning_disabled"},
            )
        if not np.isfinite(state["memory"]).all():
            raise ValueError("Reference accumulation overflowed; rescale input samples")
        reference_image = state["memory"]
        if np.issubdtype(image.dtype, np.integer):
            reference_image = np.clip(
                np.rint(reference_image), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max
            )
        reference_image = reference_image.astype(image.dtype)
        difference = cv2.absdiff(image, reference_image)
        if mask is not None:
            difference[mask == 0] = 0
        return SequenceStep(
            {
                "reference": same_encoding(reference_image, source),
                "difference": same_encoding(difference, source),
                "image": source,
            },
            True,
            update,
            {
                "samples": state.get("samples", 0),
                "reference_id": state.get("reference_id"),
                "reference_ticket": state.get("reference_ticket"),
            },
            {"reference preview": same_encoding(reference_image, source)},
        )


class SequenceDemo(Block):
    type_id, title, category = "input.sequence_demo", "Sequence demo frames", "Input"
    block_type, subtype = "Input", "Synthetic"
    outputs = (Port("image"),)
    defaults = {"stationary_frames": 5, "step_pixels": 2}

    def reset_runtime(self):
        self.index = 0

    def copy_runtime_from(self, other):
        self.index = getattr(other, "index", 0)

    def process(self, inputs):
        stationary = integer(self.parameters["stationary_frames"], "Stationary frames", 0, 10000)
        step = integer(self.parameters["step_pixels"], "Step pixels", 1, 20)
        self.index = getattr(self, "index", 0) + 1
        image = np.random.default_rng(17).integers(20, 80, (128, 176), dtype=np.uint8)
        x = 20 + (max(0, self.index - stationary) * step) % 110
        patch = np.random.default_rng(34).integers(120, 256, (32, 32), dtype=np.uint8)
        image[45:77, x : x + 32] = patch
        return Result({"image": image}, message=f"Synthetic frame {self.index}")


SEQUENCE_BLOCKS = (
    BackgroundSequence,
    FarnebackSequence,
    TrackPointsSequence,
    ImageMemory,
    SequenceDemo,
)
