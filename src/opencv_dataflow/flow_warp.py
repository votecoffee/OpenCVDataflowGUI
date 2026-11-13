"""Dense displacement sampling with explicit grids, coverage and consistency evidence."""

import copy
import struct
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .parameters import scalar_value
from .payloads import NO_VALUE, check_spatial, frame
from .processing import image_value
from .registration import BORDERS, INTERPOLATION
from .remap_maps import COORDINATE_MAP_TAG
from .selection import SELECTION_INPUTS, selection_mask
from .sequence_tools import DenseFlow

MAX_PIXELS = 2_100_000
HALO = {"nearest": 0, "linear": 0, "cubic": 2, "lanczos4": 4}


def same_grid(actual, expected, name):
    a, b = actual.get("spatial"), expected.get("spatial")
    if a and b and a.get("root_id") is not None and b.get("root_id") is not None:
        if (
            a["root_id"] != b["root_id"]
            or a["image_size"] != b["image_size"]
            or not np.allclose(a["to_root"], b["to_root"], atol=1e-9, rtol=0)
        ):
            raise ValidationError(f"{name} coordinate frame differs from its declared flow grid")


def flow_field(value, shape, maximum, name, context):
    raw = value.vectors if isinstance(value, DenseFlow) else value
    if (
        not isinstance(raw, np.ndarray)
        or raw.shape != (*shape, 2)
        or raw.dtype not in (np.float32, np.float64)
    ):
        raise ValidationError(f"{name} needs float32/64 H×W×2 displacement vectors")
    try:
        check_spatial(context, [shape[1], shape[0]])
    except ValueError as exc:
        raise ValidationError(f"{name} metadata: {exc}") from exc
    valid = np.isfinite(raw).all(axis=2) & (np.abs(raw) <= maximum).all(axis=2)
    safe = np.where(valid[..., None], raw, 0).astype(np.float32)
    return safe, valid


def sample_mask(mask, x, y, method="linear", halo=0):
    if halo:
        mask = cv2.erode(
            mask.astype(np.uint8),
            np.ones((2 * halo + 1, 2 * halo + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    values = cv2.remap(
        mask.astype(np.float32),
        x,
        y,
        INTERPOLATION[method],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return values >= 1 - 1e-6


def inside(x, y, shape, halo=0):
    h, w = shape
    return (x >= halo) & (x <= w - 1 - halo) & (y >= halo) & (y <= h - 1 - halo)


class StoredFlow(DenseFlow):
    def describe(self):
        return {
            "model": "dense_flow_v1",
            "shape": list(self.vectors.shape),
            "dtype": str(self.vectors.dtype),
            "coordinates": "File-defined displacement; consumer declares direction and grids",
            "nonfinite_vectors": self.metadata["flow_file"]["nonfinite_vectors"],
        }


class LoadFlowFile(Block):
    type_id, title = "input.flow_file", "Load dense flow file"
    category = block_type = "Input"
    subtype = "Motion fields"
    outputs = (Port("flow", "any"), Port("info", "data"))
    defaults = {"path": "", "path_base": "project"}
    parameter_choices = {"path_base": ("project", "working_directory")}
    parameter_help = {
        "path": "Middlebury .flo: float32 dx/dy; source/target identity is not stored.",
        "path_base": "Resolve relative file paths from this saved project or working directory.",
    }
    process_inputs_readonly = parallel_safe = True

    def process(self, inputs):
        value = self.parameters["path"]
        if not isinstance(value, str) or not value.strip() or Path(value).suffix.lower() != ".flo":
            raise ValidationError("Choose a .flo displacement file", "path")
        path = Path(value)
        if self.parameters["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save or open the project before using a relative flow path")
            path = self.project_dir / path
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise ValidationError("Truncated .flo header")
            magic, width, height = struct.unpack("<fii", header)
            if (
                magic != 202021.25
                or not 1 <= width < 32767
                or not 1 <= height < 32767
                or width * height > MAX_PIXELS
            ):
                raise ValidationError("Invalid .flo header or dimensions; at most 2.1M pixels")
            expected = width * height * 8
            raw = stream.read(expected + 1)
            if len(raw) != expected:
                raise ValidationError("Flow bytes do not match the declared dimensions")
        vectors = np.frombuffer(raw, "<f4").reshape(height, width, 2).copy()
        info = {
            "filename": path.name,
            "size": [width, height],
            "nonfinite_vectors": int(np.count_nonzero(~np.isfinite(vectors).all(axis=2))),
            "association": "unspecified; pair with the intended image and flow direction",
        }
        return Result(
            {"flow": StoredFlow(vectors, {"flow_file": copy.deepcopy(info)}), "info": info}
        )


class WarpDenseFlow(Block):
    type_id, title = "cv.warp_flow", "Warp image with dense flow"
    category = block_type = "Image processing"
    subtype = "Video processing"
    inputs = (
        *SELECTION_INPUTS,
        Port("flow", "any"),
        Port("target", required=False, required_when_connected=True),
        Port("reverse_flow", "any", required=False, required_when_connected=True),
        Port("flow_support", required=False, required_when_connected=True),
        Port("reverse_support", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("coverage", visual_type="region"),
        Port("invalid", visual_type="region"),
        Port("uncovered", visual_type="region"),
        Port("occluded", required=False, visual_type="region"),
        Port("roundtrip_error", required=False),
        Port("inverse_error", required=False),
        Port("map_x"),
        Port("map_y"),
        Port("coverage_preview", required=False),
        Port("report", "data"),
    )
    defaults = {
        "direction": "target_to_source",
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0,
        "invalid_fill": "constant",
        "consistency": "if_connected",
        "consistency_tolerance": 1.0,
        "maximum_motion": 4096.0,
        "inverse_iterations": 12,
        "inverse_tolerance": 0.1,
        "maximum_work": 64_000_000,
        "minimum_coverage": 0.0,
    }
    parameter_choices = {
        "direction": ("target_to_source", "source_to_target"),
        "interpolation": tuple(INTERPOLATION),
        "border": tuple(BORDERS),
        "invalid_fill": ("constant", "native_border"),
        "consistency": ("off", "if_connected", "required"),
    }
    parameter_ranges = {
        "consistency_tolerance": NumericRange(0, 100),
        "maximum_motion": NumericRange(0, 16384),
        "inverse_iterations": NumericRange(1, 50, whole=True),
        "inverse_tolerance": NumericRange(0, 10),
        "maximum_work": NumericRange(1, 256_000_000, whole=True),
        "minimum_coverage": NumericRange(0, 1),
    }
    parameter_help = {
        "direction": "Pull samples target + flow; forward flow is inverted before sampling.",
        "interpolation": "Cubic/Lanczos coverage has a conservative 2/4-pixel halo.",
        "border": "Native extrapolation is for display and never adds valid coverage.",
        "border_value": "Finite scalar or stored-channel list within the source sample range.",
        "invalid_fill": "Constant fills unsupported pixels; native_border keeps native sampling.",
        "consistency": "Reverse-field agreement is an occlusion proxy, not visibility proof.",
        "consistency_tolerance": "Maximum reverse/forward return error in local pixels.",
        "maximum_motion": "Per-axis limit; nonfinite or larger vectors are invalid.",
        "inverse_iterations": "Fixed-point source-to-target iterations; ignored in pull mode.",
        "inverse_tolerance": "Maximum residual of source + flow(source) = target, in pixels.",
        "maximum_work": "Pixel/iteration budget for inversion and field checks.",
        "minimum_coverage": "Minimum valid fraction. Zero valid donors always withholds the image.",
    }
    input_help = {
        "image": "Finite gray/color u8/u16/i16/f32/f64; same-size grids, up to 2.1M pixels.",
        "flow": "DenseFlow or floating H×W×2 array; indexed by the grid declared in Direction.",
        "target": "Optional output-grid anchor; pixels unused, source ticket stays primary.",
        "mask": "Source donor support, intersected with region; source coordinates only.",
        "region": "Source-grid selection; unsupported donors lose coverage.",
        "reverse_flow": "Opposite displacement on the opposite grid; used for round-trip checks.",
        "flow_support": "Optional uint8 support in the primary field's index grid.",
        "reverse_support": "Optional uint8 support in the reverse field's index grid.",
    }
    process_inputs_readonly = parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "flow", "mask", "region"]
        self.exposed_outputs = ["image", "coverage", "report"]

    def support(self, inputs, key, shape, context):
        if key not in inputs:
            return np.ones(shape, bool)
        value = inputs[key]
        if value.dtype != np.uint8 or value.shape != shape:
            raise ValidationError(f"{key} must be matching uint8 monochrome support")
        same_grid(self.input_payloads[key], context, key)
        return np.asarray(value) != 0

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        shape = image.shape[:2]
        height, width = shape
        p = self.parameters
        if min(shape) < 1 or height * width > MAX_PIXELS or max(shape) >= 32767:
            raise ValidationError("Dense-flow warping is limited to 2.1M pixels and 32766 per axis")
        if image.dtype == np.float64 and p["interpolation"] == "cubic":
            raise ValidationError(
                "Float64 cubic remap corrupts native boundaries; use linear or float32"
            )
        iterations = int(p["inverse_iterations"]) if p["direction"] == "source_to_target" else 0
        if height * width * (iterations + 3) > p["maximum_work"]:
            raise ValidationError("Dense-flow work exceeds maximum_work", "maximum_work")
        border_value = scalar_value(p["border_value"])
        limit = np.iinfo(image.dtype) if image.dtype.kind in "ui" else np.finfo(image.dtype)
        if any(not limit.min <= value <= limit.max for value in border_value):
            raise ValidationError(
                "Border values must be within the source sample range", "border_value"
            )
        incoming = copy.deepcopy(self.input_payloads["image"])
        flow_context = self.input_payloads["flow"]
        target = self.input_payloads.get("target", {})
        if "target" in inputs and inputs["target"].shape[:2] != shape:
            raise ValidationError("Target anchor must match the flow/output dimensions")
        identity = uuid4().hex
        target_frame = target.get("spatial")
        grid_context = target
        if target_frame is None and p["direction"] == "target_to_source":
            target_frame = flow_context.get("spatial")
            grid_context = flow_context
        if target_frame is None or (target_frame.get("root_id") is None and "target" not in inputs):
            target_frame = frame([width, height], f"flow-warp:{identity}")
            grid_context = {}
        output_context = {"spatial": target_frame}
        field_context = incoming if iterations else output_context
        reverse_context = output_context if iterations else incoming
        same_grid(flow_context, field_context, "flow")
        vectors, supported = flow_field(
            inputs["flow"], shape, p["maximum_motion"], "flow", flow_context
        )
        supported &= self.support(inputs, "flow_support", shape, field_context)
        selected = selection_mask(inputs, source)
        donor_support = np.ones(shape, bool) if selected is None else selected != 0
        yy, xx = np.indices(shape, np.float32)
        inverse_error = NO_VALUE
        if iterations:
            mx, my = xx.copy(), yy.copy()
            for _ in range(iterations):
                sampled = cv2.remap(
                    vectors, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                mx, my = xx - sampled[..., 0], yy - sampled[..., 1]
            sampled = cv2.remap(vectors, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            inverse_error = cv2.magnitude(mx + sampled[..., 0] - xx, my + sampled[..., 1] - yy)
            field_inside = inside(mx, my, shape)
            invalid = field_inside & (
                ~sample_mask(supported, mx, my) | (inverse_error > p["inverse_tolerance"])
            )
        else:
            mx, my = xx + vectors[..., 0], yy + vectors[..., 1]
            invalid = ~supported
        halo = HALO[p["interpolation"]]
        method = "nearest" if p["interpolation"] == "nearest" else "linear"
        available = inside(mx, my, shape, halo) & sample_mask(donor_support, mx, my, method, halo)
        uncovered = ~invalid & ~available
        candidate = ~invalid & available
        checked = p["consistency"] != "off" and "reverse_flow" in inputs
        if p["consistency"] == "required" and not checked:
            raise ValidationError("Reverse flow is required for the selected consistency mode")
        if "reverse_support" in inputs and p["consistency"] != "off" and not checked:
            raise ValidationError("Reverse support requires a reverse flow field")
        occluded, roundtrip = NO_VALUE, NO_VALUE
        if checked:
            same_grid(self.input_payloads["reverse_flow"], reverse_context, "reverse_flow")
            reverse, reverse_valid = flow_field(
                inputs["reverse_flow"],
                shape,
                p["maximum_motion"],
                "reverse_flow",
                self.input_payloads["reverse_flow"],
            )
            reverse_valid &= self.support(inputs, "reverse_support", shape, reverse_context)
            if iterations:
                evaluable = candidate & reverse_valid
                roundtrip = cv2.magnitude(xx + reverse[..., 0] - mx, yy + reverse[..., 1] - my)
            else:
                evaluable = candidate & sample_mask(reverse_valid, mx, my)
                returned = cv2.remap(
                    reverse, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                roundtrip = cv2.magnitude(mx + returned[..., 0] - xx, my + returned[..., 1] - yy)
            invalid |= candidate & ~evaluable
            roundtrip[~evaluable] = np.nan
            occluded = evaluable & (roundtrip > p["consistency_tolerance"])
            candidate = evaluable & ~occluded
        valid = candidate
        count = int(np.count_nonzero(valid))
        fraction = count / (height * width)
        passed = count > 0 and fraction >= p["minimum_coverage"]
        output = NO_VALUE
        if passed:
            output = cv2.remap(
                image,
                mx,
                my,
                INTERPOLATION[p["interpolation"]],
                borderMode=BORDERS[p["border"]],
                borderValue=border_value,
            )
            if p["invalid_fill"] == "constant":
                fill = cv2.copyMakeBorder(
                    image[:1, :1], 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=border_value
                )[0, 0]
                output[~valid] = fill
            output = ImageArray(output, getattr(source, "color_format", None))
        context = copy.deepcopy(incoming)
        context["spatial"] = copy.deepcopy(target_frame)
        context.pop("pyramid", None)
        context.pop("raw_sensor", None)
        if "pyramid" in grid_context:
            context["pyramid"] = copy.deepcopy(grid_context["pyramid"])
        lineage = {
            "direction": p["direction"],
            "source_spatial": incoming.get("spatial"),
            "source_ticket": incoming.get("ticket"),
            "target_ticket": target.get("ticket"),
            "target_spatial": copy.deepcopy(target_frame),
            "target_video_frame": copy.deepcopy(target.get("video_frame")),
            "map_id": identity,
            "global_invertibility": "not_established",
        }
        context["flow_warp"] = copy.deepcopy(lineage)
        preview = NO_VALUE
        if self.generate_graphics or "coverage_preview" in self.exposed_outputs:
            preview = np.zeros((*shape, 3), np.uint8)
            preview[valid] = (65, 190, 85)
            preview[uncovered] = (80, 80, 80)
            preview[invalid] = (20, 70, 240)
            if checked:
                preview[occluded] = (20, 180, 240)
            preview = ImageArray(preview, "bgr")

        def mask(value):
            return value.astype(np.uint8) * 255

        report = {
            **lineage,
            "passed": passed,
            "valid_pixels": count,
            "coverage_fraction": fraction,
            "invalid_pixels": int(np.count_nonzero(invalid)),
            "uncovered_pixels": int(np.count_nonzero(uncovered)),
            "occlusion_proxy_pixels": int(np.count_nonzero(occluded)) if checked else None,
            "occlusion_test": "roundtrip_inconsistency_proxy" if checked else "not_checked",
            "support": "source selection and conservative image kernel footprint",
            "interpolation": p["interpolation"],
            "border": p["border"],
            "invalid_fill": p["invalid_fill"],
            "image_size": [width, height],
            "inverse_iterations": iterations,
            "maximum_motion": p["maximum_motion"],
            "flow_grid_association": (
                "known_frame"
                if flow_context.get("spatial", {}).get("root_id") is not None
                else "size_only"
            ),
        }
        outputs = {
            "image": output,
            "coverage": mask(valid),
            "invalid": mask(invalid),
            "uncovered": mask(uncovered),
            "occluded": mask(occluded) if checked else NO_VALUE,
            "roundtrip_error": roundtrip,
            "inverse_error": inverse_error,
            "map_x": mx,
            "map_y": my,
            "coverage_preview": preview,
            "report": report,
        }
        payloads = {key: copy.deepcopy(context) for key in outputs}
        for key in outputs.keys() - {"image"}:
            payloads[key].pop("alpha_representation", None)
        map_info = {
            "model": "image_coordinate_map_v1",
            "id": identity,
            "source_size": [width, height],
            "output_size": [width, height],
            "source_spatial": incoming["spatial"],
            "output_spatial": target_frame,
        }
        for axis in ("x", "y"):
            payloads[f"map_{axis}"][COORDINATE_MAP_TAG] = {**map_info, "axis": axis}
        drawings = {}
        if self.generate_graphics:
            drawings["flow coverage"] = GraphicDrawing(
                (
                    graphic_object(
                        "text",
                        [(5, min(18, height - 1))],
                        (65, 190, 85) if passed else (20, 70, 240),
                        text=f"Flow coverage {fraction:.1%}",
                    ),
                ),
                "image",
                True,
            )
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            passed=passed,
            score=fraction,
            message=f"{count}/{height * width} valid donors; "
            + ("coverage accepted" if passed else "image withheld"),
        )
