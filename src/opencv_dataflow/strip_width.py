"""Inspect the width of center-connected foreground in a straightened strip."""

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .projection import profile_plot
from .result_graphics import REJECTED
from .selection import selection_mask
from .workplane import read_length_scale


def _strip_scale(image, sampling):
    if (
        not isinstance(sampling, dict)
        or sampling.get("model") != "path_strip_v1"
        or sampling.get("strip_spatial") != metadata(image).get("spatial")
    ):
        raise ValidationError("Sampling must describe the input strip's current pixel grid")
    center = NumericRange(0, image.shape[0] - 1, whole=True).check(
        sampling.get("center_row"), "Sampling center row"
    )
    step = NumericRange(0.0001, 1024).check(sampling.get("cross_step"), "Sampling cross step")
    if "physical_scale" in sampling:
        units, scale = read_length_scale(sampling["physical_scale"])
        return center, step * scale, units
    return center, step, "source px"


def _column_marks(columns, top, bottom, color):
    if len(columns) > 128:
        columns = columns[np.linspace(0, len(columns) - 1, 128, dtype=np.intp)]
    objects = []
    for x in columns:
        # Draw the run or a dot at the nominal center when no width was available.
        points = ((x, top[x]),) if top[x] == bottom[x] else ((x, top[x]), (x, bottom[x]))
        objects.append(graphic_object("polyline", points, color, closed=len(points) == 1))
    return GraphicDrawing(tuple(objects))


class InspectStripWidth(Block):
    type_id, title = "inspect.strip_width", "Inspect strip width"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True, visual_type="region"),
        Port("sampling", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("profile", "data"), Port("measurements", "data"), Port("pass_fraction", "number"),
    )
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    defaults = {
        "center_row": -1, "minimum_width": 1.0, "maximum_width": 0.0,
        "enforce_limits": True,
        "minimum_valid_fraction": 1.0, "minimum_coverage": 1.0,
        "minimum_in_range_fraction": 1.0,
    }
    parameter_ranges = {
        "center_row": NumericRange(-1, 2_100_000, whole=True),
        "minimum_width": NumericRange(0), "maximum_width": NumericRange(0),
        "minimum_valid_fraction": NumericRange(0, 1), "minimum_coverage": NumericRange(0, 1),
        "minimum_in_range_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "center_row": "Row the foreground run must cross. -1 uses the sampling center row "
        "or the middle row of an untagged mask.",
        "minimum_width": "Inclusive minimum. Measured as sample count times cross-strip step.",
        "maximum_width": "Inclusive maximum. Zero disables the upper limit.",
        "minimum_valid_fraction": "Minimum fraction of columns that can be classified. "
        "A clipped run or a masked-out center is unmeasurable.",
        "minimum_coverage": "Minimum fraction of classifiable columns with a measurable "
        "foreground run. At least one run must be measurable.",
        "minimum_in_range_fraction": "Minimum fraction of measured widths within limits.",
        "enforce_limits": "Turn off to report measurements and defect classifications "
        "without assigning a width verdict.",
    }
    input_help = {
        "image": "8-bit monochrome foreground mask. Nonzero pixels are foreground.",
        "mask": "Validity mask from the same strip grid. Zero means the sample is unavailable.",
        "sampling": "Optional Unwrap path strip sampling data for source or physical widths. "
        "Also connect its validity mask.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["enforce_limits"]) is not bool:
            raise ValidationError("Enforce limits must be a checkbox value", "enforce_limits")
        high = float(p["maximum_width"])
        if high and high < float(p["minimum_width"]):
            raise ValidationError(
                "Maximum width must be at least the minimum or zero", "maximum_width"
            )

    def process(self, inputs):
        image = inputs["image"]
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValidationError("Use an 8-bit monochrome foreground mask")
        height, width = image.shape
        if not 0 < image.size <= 2_100_000 or width > 8192:
            raise ValidationError("Use a nonempty strip up to 8192 columns and 2.1 million pixels")
        center, step, units = height // 2, 1.0, "strip px"
        if "sampling" in inputs:
            if "mask" not in inputs:
                raise ValidationError("Connect the strip validity mask when using sampling data")
            center, step, units = _strip_scale(image, inputs["sampling"])
        p = self.parameters
        if int(p["center_row"]) >= 0:
            center = int(p["center_row"])
        if center >= height:
            raise ValidationError("Center row must be inside the strip", "center_row")

        selected = selection_mask(inputs, image)
        valid = selected != 0 if selected is not None else np.ones(image.shape, dtype=bool)
        foreground = (image != 0) & valid
        top = center + 1 - np.logical_and.accumulate(foreground[center::-1], axis=0).sum(axis=0)
        bottom = center - 1 + np.logical_and.accumulate(foreground[center:], axis=0).sum(axis=0)
        columns = np.arange(width)
        # Both ends need an observed background sample. An image edge or invalid sample
        # next to the run leaves the true width unknown.
        measured = (
            foreground[center] & (top > 0) & (bottom < height - 1)
            & valid[np.maximum(top - 1, 0), columns]
            & valid[np.minimum(bottom + 1, height - 1), columns]
        )
        missing = valid[center] & ~foreground[center]
        unknown = ~(measured | missing)
        widths = (bottom - top + 1) * step
        low = float(p["minimum_width"])
        passing = measured & ((widths >= low) | np.isclose(widths, low, rtol=1e-12, atol=0))
        high = float(p["maximum_width"])
        if high:
            passing &= (widths <= high) | np.isclose(widths, high, rtol=1e-12, atol=0)
        outside = measured & ~passing
        measured_count = int(measured.sum())
        missing_count = int(missing.sum())
        passing_count = int(passing.sum())
        classifiable = measured_count + missing_count
        valid_fraction = classifiable / width
        coverage = measured_count / classifiable if classifiable else 0.0
        in_range = passing_count / measured_count if measured_count else 0.0
        fraction = passing_count / width

        reasons = []
        if not classifiable or valid_fraction < float(p["minimum_valid_fraction"]):
            reasons.append({
                "code": "strip_width_unmeasurable",
                "message": f"{width - classifiable} columns lack an observed center or bead edge",
            })
        if classifiable and (not measured_count or coverage < float(p["minimum_coverage"])):
            reasons.append({
                "code": "strip_foreground_missing",
                "message": f"{missing_count} classifiable columns lack foreground at the center",
            })
        if measured_count and in_range < float(p["minimum_in_range_fraction"]):
            reasons.append({
                "code": "strip_width_out_of_range",
                "message": f"{measured_count - passing_count} measured widths are outside limits",
            })

        status = np.full(width, "unmeasurable", dtype="<U12")
        status[missing] = "missing"
        status[outside] = "out_of_range"
        status[passing] = "pass"
        profile = {
            "units": units, "center_row": center,
            "width_limits": [low, high or None],
            "widths": [float(value) if keep else None
                       for value, keep in zip(widths, measured, strict=True)],
            "status": status.tolist(),
        }
        observed = widths[measured]
        measurements = {
            "units": units, "columns": width, "measured_count": measured_count,
            "missing_count": missing_count, "unmeasurable_count": width - classifiable,
            "out_of_range_count": measured_count - passing_count,
            "valid_fraction": valid_fraction, "coverage": coverage, "in_range_fraction": in_range,
            "minimum": float(observed.min()) if measured_count else None,
            "mean": float(observed.mean()) if measured_count else None,
            "maximum": float(observed.max()) if measured_count else None,
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            graphics["Width profile"] = profile_plot(
                widths[:, None], measured, ["Width"], "strip column", f"width ({units})"
            )
            centers = np.full(width, center)
            for name, mask, a, b, color in (
                ("Width out of range", outside, top, bottom, REJECTED),
                ("Missing foreground", missing, centers, centers, (30, 210, 250)),
                ("Width unmeasurable", unknown, centers, centers, (220, 70, 210)),
            ):
                if mask.any():
                    drawings[name] = _column_marks(columns[mask], a, b, color)
        if not p["enforce_limits"]:
            reasons = []
        return Result(
            {"profile": profile, "measurements": measurements, "pass_fraction": fraction},
            graphics=graphics, drawings=drawings, passed=not reasons, score=fraction,
            fail_reasons=reasons, message=f"{passing_count} of {width} strip columns pass",
        )
