"""Edge-aware dense displacement from externally matched image points."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .flow_warp import inside, same_grid
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, convert_depth, image_format
from .match_review import paired_points
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, check_spatial
from .processing import image_value
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask
from .sequence_tools import DenseFlow
from .thin_plate_spline import pixel_points
from .tickets import ticket_key

_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_MAX_MATCHES = 32766
_MAX_WORKSPACE_BYTES = 512 * 1024**2
_PAIR_FIELDS = ("spatial", "ticket", "pyramid", "video_frame", "recipe_ref")


def flow_guide(image):
    if image.dtype == np.float64 and image_format(image) in ("hsv", "lab"):
        image = ImageArray(image.astype(np.float32), image_format(image))
    native = image_value(image, _DEPTHS, canonical=True)
    if native.ndim == 3:
        native = native[..., :3]
    if native.dtype != np.uint8:
        native = convert_depth(native, "uint8")
    return np.ascontiguousarray(native)


def edge_aware_factory():
    create = getattr(getattr(cv2, "ximgproc", None), "createEdgeAwareInterpolator", None)
    if create is None:
        raise ValidationError("Edge-aware flow requires OpenCV contrib ximgproc")
    return create


def configure_interpolator(solver, p):
    solver.setK(int(p["neighbors"]))
    solver.setSigma(float(p["sigma"]))
    solver.setUsePostProcessing(p["post_smoothing"])
    solver.setFGSLambda(float(p["fgs_lambda"]))
    solver.setFGSSigma(float(p["fgs_sigma"]))


def unique_seeds(source, target, size, offset=0.5):
    source, target = (np.ascontiguousarray(points, dtype=np.float32) for points in (source, target))
    seeds = np.floor(source + np.float32(offset)).astype(np.int32)
    if offset == 0.5:
        # EdgeAwareInterpolator rounds and clamps. RIC truncates without a clamp.
        seeds = np.minimum(seeds, np.array(size) - 1)
    elif np.any(seeds >= size):
        raise ValidationError("Source seed rounds outside the float32 image grid", "matches")
    _, first, inverse = np.unique(seeds, axis=0, return_index=True, return_inverse=True)
    if len(first) != len(source):
        representatives = first[inverse]
        if not (
            np.array_equal(source, source[representatives])
            and np.array_equal(target, target[representatives])
        ):
            raise ValidationError("Different pairs share a source seed pixel", "matches")
        first.sort()
        source, target, seeds = source[first], target[first], seeds[first]
    if len(seeds) >= 4 and np.linalg.matrix_rank(seeds - seeds[0]) != 2:
        raise ValidationError("Source seed pixels must span both image axes", "matches")
    return source, target


def validate_match_context(supplied, expected, size, label, field="matches"):
    if not isinstance(supplied, dict):
        raise ValidationError(f"{label} match context must be an object", field)
    check_spatial(supplied, size)
    same_grid(supplied, expected, label)
    if supplied.get("ticket") is not None and expected.get("ticket") is not None:
        if ticket_key(supplied["ticket"]) != ticket_key(expected["ticket"]):
            raise ValidationError(f"{label} points belong to a different ticket", field)


class MatchedPointFlow(Block):
    """Shared correspondence inputs, spatial validation and dense flow outputs."""

    category = block_type = "Image processing"
    subtype = "Video processing"
    inputs = (
        *SELECTION_INPUTS,
        Port("target"),
        Port("matches", "data", required=False, required_when_connected=True),
        Port("source_points", "data", required=False, required_when_connected=True),
        Port("target_points", "data", required=False, required_when_connected=True),
        Port("cost_map", required=False, required_when_connected=True),
    )
    outputs = (
        Port("flow", "any"),
        Port("support", visual_type="mask"),
        Port("report", "data"),
        Port("magnitude", required=False),
        Port("angle", required=False),
        Port("visualization", required=False),
    )
    payload_reference = "image"
    process_inputs_readonly = preview_safe = True
    seed_offset = 0.5
    defaults = {
        "visual_max_motion": 10.0,
        "source_points": [],
        "target_points": [],
    }
    parameter_ranges = {
        "visual_max_motion": NumericRange(1e-6),
    }
    parameter_help = {
        "visual_max_motion": "Motion in pixels mapped to full brightness in the flow view.",
        "source_points": "Optional ordered [x,y] source coordinates for offline or fixed pairs.",
        "target_points": "Target coordinates in the same order as the configured source points.",
    }
    input_help = {
        "image": "Query/source image. Its edges guide interpolation onto its own pixel grid.",
        "target": "Train/target image on an equal-size grid.",
        "matches": "Descriptor match record with query and train points in corresponding order.",
        "source_points": "An ordered list of source pixel coordinates.",
        "target_points": "The matching list of target pixel coordinates.",
        "cost_map": "Optional source-grid float32 edge costs in [0, 1].",
        "mask": "Limit the published vectors and support on the source grid.",
        "region": "Intersect this source-grid region with the optional mask.",
    }
    output_help = {
        "flow": "Float32 source-to-target displacement in pixels on the source grid.",
        "support": "Selected pixels whose displaced coordinates are inside the target image.",
        "report": "Seed count, source and target bindings, and fit error at supplied matches.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "target", "matches"]
        self.exposed_outputs = ["flow", "support", "report", "visualization"]
        self.preview_output = "visualization"

    def points(self, inputs, size, source_context, target_context):
        role_contexts = (source_context, target_context)
        ports = ("source_points", "target_points")
        if "matches" in inputs:
            if any(name in inputs for name in ports):
                raise ValidationError("Choose a match record or the two point inputs", "matches")
            record = inputs["matches"]
            source, target = paired_points(record, size, size)
            for name, expected in zip(("query", "train"), role_contexts, strict=True):
                if f"{name}_context" in record:
                    validate_match_context(record[f"{name}_context"], expected, size, name)
        else:
            if sum(name in inputs for name in ports) == 1:
                raise ValidationError("Connect both point lists", *ports)
            parsed = []
            for name, expected in zip(ports, role_contexts, strict=True):
                validate_match_context(self.input_payloads.get(name, {}), expected, size, name)
                raw = inputs.get(name, self.parameters[name])
                if not isinstance(raw, (list, tuple)) or len(raw) > _MAX_MATCHES:
                    raise ValidationError(f"Use up to {_MAX_MATCHES:,} ordered points", name)
                parsed.append(pixel_points(raw, size, name, 0, _MAX_MATCHES))
            source, target = parsed
            if len(source) != len(target):
                raise ValidationError("Source and target point counts must match", *ports)
        if len(source) > _MAX_MATCHES:
            raise ValidationError(f"Use up to {_MAX_MATCHES:,} matches", "matches")
        return source, target

    def validate_workload(self, size, count):
        raise NotImplementedError

    def interpolate(self, guide, source_points, target_guide, target_points, cost):
        raise NotImplementedError

    def limit_support(self, valid, source_points):
        """Optionally narrow geometric support on the source grid in place."""

    def process(self, inputs):
        p = self.parameters
        source, target = inputs["image"], inputs["target"]
        height, width = source.shape[:2]
        size = [width, height]
        if min(size) < 2 or max(size) > 8192:
            raise ValidationError("Match interpolation requires 2–8192 pixels per axis")
        if target.shape[:2] != source.shape[:2]:
            raise ValidationError("Source and target raster sizes must match", "target")
        source_context, target_context = (self.input_payloads[name] for name in ("image", "target"))
        for context in (source_context, target_context):
            check_spatial(context, size)
        raw_source, raw_target = self.points(inputs, size, source_context, target_context)
        source_points, target_points = unique_seeds(raw_source, raw_target, size, self.seed_offset)
        count = len(source_points)
        self.validate_workload(size, count)
        guide, target_guide = flow_guide(source), flow_guide(target)
        selection = selection_mask(inputs, source)

        cost = inputs.get("cost_map")
        if cost is not None:
            context = self.input_payloads["cost_map"]
            check_spatial(context, size)
            same_grid(context, source_context, "cost map")
            if (
                cost.dtype != np.float32
                or cost.shape != (height, width)
                or not np.isfinite(cost).all()
                or cost.min() < 0
                or cost.max() > 1
            ):
                raise ValidationError("Cost map must be source-grid float32 in [0, 1]", "cost_map")

        shift = target_points - source_points
        flow, method = self.interpolate(guide, source_points, target_guide, target_points, cost)
        if (
            flow is None
            or flow.shape != (height, width, 2)
            or flow.dtype != np.float32
            or not np.isfinite(flow).all()
        ):
            raise ValidationError("Interpolation produced invalid dense vectors")

        fit = cv2.remap(
            flow,
            source_points[:, :1],
            source_points[:, 1:],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1, 2)
        fit_error = np.linalg.norm(fit - shift, axis=1)
        x = np.arange(width, dtype=np.float32) + flow[..., 0]
        y = np.arange(height, dtype=np.float32)[:, None] + flow[..., 1]
        valid = inside(x, y, (height, width))
        self.limit_support(valid, source_points)
        if selection is not None:
            valid &= selection != 0
            flow[selection == 0] = 0
        pair = {
            role: {name: context[name] for name in _PAIR_FIELDS if name in context}
            for role, context in (("source", source_context), ("target", target_context))
        }
        context = copy.deepcopy(source_context)
        for name in ("alpha_representation", "raw_sensor"):
            context.pop(name, None)
        context["flow_matches"] = copy.deepcopy(pair)
        outputs = {
            "flow": DenseFlow(flow, context),
            "support": ImageArray(valid.astype(np.uint8) * 255, "gray"),
            "report": {
                "method": method,
                "units": "local_pixels",
                "direction": "source_to_target",
                "input_matches": len(raw_source),
                "seeds": count,
                "supported_pixels": int(valid.sum()),
                "control_mean_error_pixels": float(fit_error.mean()),
                "control_max_error_pixels": float(fit_error.max()),
                "pair": pair,
            },
            "magnitude": NO_VALUE,
            "angle": NO_VALUE,
            "visualization": NO_VALUE,
        }
        if {"magnitude", "angle", "visualization"} & set(self.exposed_outputs):
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            outputs["magnitude"] = ImageArray(magnitude, "gray")
            outputs["angle"] = ImageArray(angle, "gray")
            if "visualization" in self.exposed_outputs:
                hsv = np.zeros((height, width, 3), np.uint8)
                hsv[..., 0] = np.rint(angle / 2).astype(np.uint8) % 180
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(magnitude * (255 / p["visual_max_motion"]), 0, 255)
                hsv[~valid] = 0
                outputs["visualization"] = ImageArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "bgr")
        drawings = {}
        if self.generate_graphics:
            stride = (count + 511) // 512
            arrows = tuple(
                graphic_object("arrow", (a, b), ACCEPTED, thickness=1)
                for a, b in zip(source_points[::stride], target_points[::stride], strict=True)
            )
            drawings["supplied matches"] = GraphicDrawing(arrows, source="image")
        return Result(
            outputs,
            payloads={name: context for name, value in outputs.items() if value is not NO_VALUE},
            drawings=drawings,
            message=f"{count} source seeds · {int(valid.sum())} supported pixels",
        )


class EdgeAwareFlow(MatchedPointFlow):
    type_id, title = "cv.edge_aware_flow", "Interpolate matched image motion"
    description = "Use paired image points and an edge guide to estimate dense pixel motion."
    process_inputs_readonly = True
    defaults = {
        "neighbors": 128,
        "sigma": 0.05,
        "geodesic_lambda": 999.0,
        "post_smoothing": True,
        "fgs_lambda": 500.0,
        "fgs_sigma": 1.5,
        **MatchedPointFlow.defaults,
    }
    parameter_ranges = {
        "neighbors": NumericRange(4, 512, whole=True),
        "sigma": NumericRange(1e-6, 1),
        "geodesic_lambda": NumericRange(0, 1000),
        "fgs_lambda": NumericRange(0, 100000),
        "fgs_sigma": NumericRange(1e-6, 255),
        **MatchedPointFlow.parameter_ranges,
    }
    parameter_help = {
        "neighbors": "Number of distinct source seed pixels in each local affine fit.",
        "sigma": "Distance weight falloff for the local affine fits.",
        "geodesic_lambda": "Edge weight in geodesic distance. Zero ignores edges during fitting.",
        "post_smoothing": "Apply the native fast global smoother after interpolation.",
        "fgs_lambda": "Regularization weight for post-smoothing.",
        "fgs_sigma": "Sensitivity to guide color or intensity during post-smoothing.",
        **MatchedPointFlow.parameter_help,
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["post_smoothing"]) is not bool:
            raise ValidationError("Post smoothing must be boolean", "post_smoothing")

    def validate_workload(self, size, count):
        width, height = size
        neighbors = int(self.parameters["neighbors"])
        if count < neighbors:
            raise ValidationError(
                f"Only {count} distinct source seeds for {neighbors} neighbors", "neighbors"
            )
        if height * width * 112 + count * (64 + 8 * neighbors) > _MAX_WORKSPACE_BYTES:
            raise ValidationError(
                "Edge-aware workspace exceeds 512 MiB. Crop or reduce matches or neighbors.",
                "neighbors",
            )

    def interpolate(self, guide, source_points, target_guide, target_points, cost):
        p = self.parameters
        shift = target_points - source_points
        if np.all(shift == shift[0]):
            # The native robust fit uses a zero inlier radius on exact translations.
            flow = np.empty((*guide.shape[:2], 2), np.float32)
            flow[:] = shift[0]
            return flow, "uniform_translation"

        solver = edge_aware_factory()()
        configure_interpolator(solver, p)
        solver.setLambda(float(p["geodesic_lambda"]))
        if cost is not None:
            solver.setCostMap(cost)
        return solver.interpolate(guide, source_points, target_guide, target_points), "edge_aware"
