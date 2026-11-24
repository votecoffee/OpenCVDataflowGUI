"""Sparse native RLOF across an explicit source and target image pair."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .edge_aware_flow import flow_guide, validate_match_context
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import check_spatial
from .result_graphics import ACCEPTED, REJECTED
from .rlof_parameters import (
    RLOF_CHOICES, RLOF_DEFAULTS, RLOF_HELP, RLOF_RANGES, rlof_settings, validate_rlof,
)
from .selection import contains, selection_mask
from .thin_plate_spline import pixel_points

MAX_POINTS = 10_000
CONTEXT_FIELDS = ("spatial", "ticket", "pyramid", "video_frame", "recipe_ref")


def guide_image(image, support):
    guide = flow_guide(image)
    if support == "fixed" and guide.ndim == 3:
        return cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    if support == "cross" and guide.ndim == 2:
        return cv2.cvtColor(guide, cv2.COLOR_GRAY2BGR)
    return guide


def point_list(raw, size, field):
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raise ValidationError("Use an ordered list of [x,y] points", field)
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) > MAX_POINTS:
        raise ValidationError(f"Use up to {MAX_POINTS:,} ordered [x,y] points", field)
    points = pixel_points(raw, size, field, 0, MAX_POINTS).astype(np.float32)
    if np.any(points >= size):
        raise ValidationError("A point rounds outside the native float32 pixel grid", field)
    return points


def inside_points(points, size):
    return np.all(np.isfinite(points) & (points >= 0) & (points < size), axis=1)


def point_context(payload):
    return {key: payload[key] for key in CONTEXT_FIELDS if key in payload}


def solve_points(calc, source, target, points, settings, initial=None):
    # OpenCV 5's final native copy uses a 1×N CV_32FC2 row. Passing an N×1
    # column can leave the Python return array unwritten even if statuses pass.
    points = np.ascontiguousarray(points).reshape(1, -1, 2)
    guess = None if initial is None else np.array(initial, dtype=np.float32, copy=True).reshape(1, -1, 2)
    found, _, _ = calc(
        source, target, points, guess, rlofParam=settings, forwardBackwardThreshold=0.0
    )
    if found is None or found.shape != points.shape:
        raise ValidationError("Native RLOF did not return one estimate per input point")
    return found.reshape(-1, 2)


class SparseRLOF(Block):
    type_id, title = "cv.sparse_rlof", "Track wired points (Sparse RLOF)"
    category = block_type = "Image processing"
    subtype = "Video processing"
    description = "Track supplied points from a source image to a target image with robust local flow."
    inputs = (
        Port("image"),
        Port("target"),
        Port("points", "data"),
        Port("initial_points", "data", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("target_points", "data"),
        Port("matches", "data"),
        Port("tracks", "data"),
        Port("report", "data"),
        Port("image"),
    )
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {
        **RLOF_DEFAULTS,
        "global_motion_prior": False,
        "fb_tolerance": 1.5,
        "max_displacement": 0.0,
        "minimum_tracks": 1,
        "max_graphics": 250,
    }
    parameter_choices = RLOF_CHOICES
    parameter_ranges = {
        **RLOF_RANGES,
        "fb_tolerance": NumericRange(0, 100),
        "max_displacement": NumericRange(0),
        "minimum_tracks": NumericRange(0, MAX_POINTS, whole=True),
        "max_graphics": NumericRange(0, 1000, whole=True),
    }
    parameter_help = {
        **RLOF_HELP,
        "global_motion_prior": "Homography initialization for images at least 40 × 30. "
                               "Choose this or the initial point input.",
        "fb_tolerance": "Maximum forward/backward round-trip error in source pixels. Zero disables it.",
        "max_displacement": "Maximum source-to-target displacement in pixels. Zero disables it.",
        "minimum_tracks": "Minimum number of accepted correspondences for a pass.",
        "max_graphics": "Maximum number of candidate tracks drawn on the source image.",
    }
    input_help = {
        "image": "Source image where the supplied points were found.",
        "target": "Target image with the same raster dimensions. May have another ticket and frame.",
        "points": "Ordered [x,y] coordinates on the source image, such as the corners block output.",
        "initial_points": "Optional target-grid guesses in the same order as the source points.",
        "mask": "Select source points before tracking using a mask on the source grid.",
        "region": "Select source points inside a region on the source grid.",
    }
    output_help = {
        "target_points": "Accepted target coordinates in original order, carrying target metadata.",
        "matches": "Accepted query/source and train/target pairs for registration or flow interpolation.",
        "tracks": "One diagnostic per input index, with status, rejection reason and round-trip error.",
        "report": "Track counts and median accepted displacement in pixels.",
        "image": "Source image for the result overlay.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "target", "points"]
        self.preview_output = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("global_motion_prior", "illumination_model", "m_estimator"):
            if type(p[field]) is not bool:
                raise ValidationError(f"{field.replace('_', ' ').capitalize()} must be boolean", field)
        validate_rlof(p)

    def process(self, inputs):
        source, target = inputs["image"], inputs["target"]
        height, width = source.shape[:2]
        size = [width, height]
        p = self.parameters
        if target.shape[:2] != source.shape[:2]:
            raise ValidationError("Source and target raster sizes must match", "target")
        if min(size) < int(p["large_window"]) or max(size) > 8192 or width * height > 2_097_152:
            raise ValidationError("Images must fit the large window and be at most 2,097,152 pixels")
        if p["global_motion_prior"] and (width < 40 or height < 30):
            raise ValidationError("Global motion prior requires at least 40 × 30 pixels", "global_motion_prior")
        source_context, target_context = (self.input_payloads[name] for name in ("image", "target"))
        for context in (source_context, target_context):
            check_spatial(context, size)
        validate_match_context(self.input_payloads["points"], source_context, size, "source", "points")
        points = point_list(inputs["points"], size, "points")
        initial = None
        if "initial_points" in inputs:
            if p["global_motion_prior"]:
                raise ValidationError("Choose initial points or the global motion prior", "global_motion_prior")
            validate_match_context(
                self.input_payloads["initial_points"], target_context, size, "initial", "initial_points"
            )
            initial = point_list(inputs["initial_points"], size, "initial_points")
            if len(initial) != len(points):
                raise ValidationError("Initial and source point counts must match", "initial_points")
        mask = selection_mask(inputs, source)
        chosen = np.asarray([contains(mask, point) for point in points], dtype=bool)
        estimates, errors, reasons = self.track(source, target, size, points, chosen, initial)
        return self.track_result(source, size, points, estimates, errors, reasons)

    def track(self, source, target, size, points, chosen, initial):
        estimates = np.full_like(points, np.nan)
        errors = np.full(len(points), np.nan)
        reasons = np.full(len(points), "outside_selection", dtype=object)
        if not chosen.any():
            return estimates, errors, reasons
        p = self.parameters
        optflow = getattr(cv2, "optflow", None)
        calc = getattr(optflow, "calcOpticalFlowSparseRLOF", None)
        create = getattr(optflow, "RLOFOpticalFlowParameter_create", None)
        if calc is None or create is None:
            raise ValidationError("Sparse RLOF requires OpenCV contrib optflow")

        a, b = (guide_image(image, p["support"]) for image in (source, target))
        settings = rlof_settings(optflow, create, p, initial=initial is not None)
        guess = initial[chosen] if initial is not None else None
        found = solve_points(calc, a, b, points[chosen], settings, guess)
        estimates[chosen] = found
        finite = np.isfinite(estimates).all(axis=1)
        bound = inside_points(estimates, size)
        reasons[chosen] = "nonfinite_estimate"
        reasons[chosen & finite] = "outside_target"
        reasons[bound] = ""
        maximum = float(p["max_displacement"])
        if maximum:
            large_shift = np.linalg.norm(estimates[bound] - points[bound], axis=1) > maximum
            reasons[np.flatnonzero(bound)[large_shift]] = "displacement_limit"
        eligible = reasons == ""
        tolerance = float(p["fb_tolerance"])
        if tolerance and eligible.any():
            # Reverse only valid target estimates. The upstream combined call
            # reverses every estimate, including points outside the target.
            settings.setUseInitialFlow(False)
            returned = solve_points(calc, b, a, estimates[eligible], settings)
            indices = np.flatnonzero(eligible)
            valid_return = np.isfinite(returned).all(axis=1)
            selected = indices[valid_return]
            errors[selected] = np.linalg.norm(
                returned[valid_return].astype(np.float64) - points[selected], axis=1
            )
            reasons[indices[~valid_return]] = "nonfinite_round_trip"
            reasons[selected[errors[selected] > tolerance]] = "round_trip_limit"
        return estimates, errors, reasons

    def track_result(self, source, size, points, estimates, errors, reasons):
        source_context, target_context = (self.input_payloads[name] for name in ("image", "target"))
        valid = reasons == ""
        finite = np.isfinite(estimates).all(axis=1)
        displacements = estimates.astype(np.float64) - points
        records = [
            {
                "index": index,
                "source": a,
                "target": b if finite[index] else None,
                "displacement": displacements[index].tolist() if finite[index] else None,
                "status": not bool(reason),
                "reason": reason or None,
                "fb_error": float(errors[index]) if np.isfinite(errors[index]) else None,
            }
            for index, (a, b, reason) in enumerate(zip(points.tolist(), estimates.tolist(), reasons, strict=True))
        ]
        accepted = int(valid.sum())
        measured = int(np.isfinite(errors).sum())
        passed = accepted >= int(self.parameters["minimum_tracks"])
        query_context, train_context = point_context(source_context), point_context(target_context)
        result = {
            "target_points": estimates[valid].tolist(),
            "matches": {
                "model": "rlof_matches_v1",
                "query_size": size,
                "train_size": size,
                "query_context": query_context,
                "train_context": train_context,
                "query_points": points[valid].tolist(),
                "train_points": estimates[valid].tolist(),
                "source_indices": np.flatnonzero(valid).tolist(),
            },
            "tracks": {
                "model": "sparse_rlof_tracks_v1",
                "source_context": query_context,
                "target_context": train_context,
                "items": records,
            },
            "report": {
                "input_count": len(points),
                "accepted_count": accepted,
                "round_trip_count": measured,
                "median_displacement": np.median(displacements[valid], axis=0).tolist()
                if accepted else None,
                "mean_fb_error": float(np.mean(errors[valid])) if accepted and np.isfinite(errors[valid]).all() else None,
            },
            "image": source,
        }
        drawings = {}
        limit = int(self.parameters["max_graphics"])
        if self.generate_graphics and limit and records:
            stride = (len(records) + limit - 1) // limit
            graphics = tuple(
                graphic_object("arrow", (row["source"], row["target"]), ACCEPTED, thickness=1)
                if row["status"] else graphic_object("marker", [row["source"]], REJECTED, thickness=1)
                for row in records[::stride]
            )
            drawings["sparse tracks"] = GraphicDrawing(graphics, source="image")
        return Result(
            result,
            payloads={"target_points": target_context},
            drawings=drawings,
            passed=passed,
            message=f"{accepted}/{len(points)} accepted tracks",
            fail_reasons=[] if passed else ["sparse_rlof_insufficient_tracks"],
        )
