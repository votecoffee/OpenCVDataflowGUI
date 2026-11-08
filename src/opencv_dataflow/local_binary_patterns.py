"""Circular local binary patterns and comparable categorical histograms."""

from math import ceil

import numpy as np

from .core import Block, Port, Result
from .histogram import histogram_plot
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .selection import SELECTION_INPUTS, selection_mask
from .skeleton import SKIMAGE_INSTALL

_METHODS = ("default", "ror", "uniform", "nri_uniform")
_MAX_PIXELS = 2_100_000


def code_bins(method, neighbors):
    if method == "uniform":
        return neighbors + 2
    if method == "nri_uniform":
        return neighbors * (neighbors - 1) + 3
    return 2**neighbors


class LocalBinaryPatterns(Block):
    type_id, title = "inspect.local_binary_patterns", "Local binary patterns"
    category, block_type, subtype = "Inspection", "Inspection", "Texture"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("codes"),
        Port("valid_mask", visual_type="region"),
        Port("histogram", "data"),
        Port("sample_count", "integer"),
    )
    defaults = {"method": "uniform", "neighbors": 8, "radius": 1.0}
    parameter_choices = {"method": _METHODS}
    parameter_ranges = {
        "neighbors": NumericRange(4, 24, whole=True),
        "radius": NumericRange(0.5, 32),
    }
    parameter_help = {
        "method": "Uniform and ror group rotations. Default and nri_uniform retain direction.",
        "neighbors": "Samples per circle. Default and ror support up to 12 to bound bin counts.",
        "radius": "Sampling radius in pixels. The border is excluded from the histogram.",
    }
    input_help = {
        "image": "Grayscale or color uint8/uint16, limited to 2.1 million pixels.",
        "mask": "Restrict code centers. Neighbor samples may come from outside the mask.",
        "region": "Restrict code centers. Intersected with a wired mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["method"] in ("default", "ror") and int(p["neighbors"]) > 12:
            raise ValidationError("Default and ror support at most 12 neighbors", "neighbors")

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        p = self.parameters
        method, neighbors, radius = p["method"], int(p["neighbors"]), float(p["radius"])
        margin = ceil(radius)
        if min(height, width) <= 2 * margin or height * width > _MAX_PIXELS:
            raise ValidationError(
                "Image must fit the sampling radius and the 2.1 million pixel limit"
            )
        gray = gray_image(source, (np.uint8, np.uint16))
        selection = selection_mask(inputs, source)
        valid = np.zeros(gray.shape, dtype=bool)
        valid[margin:-margin, margin:-margin] = True
        if selection is not None:
            valid &= selection != 0
        count = int(np.count_nonzero(valid))
        if not count:
            raise ValidationError("No selected pixels have a complete LBP neighborhood")

        try:
            from skimage.feature import local_binary_pattern
        except ImportError as exc:
            raise RuntimeError(SKIMAGE_INSTALL) from exc

        values = local_binary_pattern(gray, neighbors, radius, method=method)[valid]
        bins = code_bins(method, neighbors)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() >= bins:
            raise ValidationError("scikit-image returned invalid LBP codes")
        values = values.astype(np.int32)
        codes = np.full(gray.shape, -1, dtype=np.int32)
        codes[valid] = values
        counts = np.bincount(values, minlength=bins)
        channel = {
            "channel": f"LBP {method} P={neighbors} R={radius}",
            "counts": counts.tolist(),
            "edges": (np.arange(bins + 1) - 0.5).tolist(),
        }
        histogram = {
            "model": "local_binary_pattern_histogram_v1",
            "method": method,
            "neighbors": neighbors,
            "radius": radius,
            "sample_count": count,
            "channels": [channel],
        }
        graphics = {}
        if self.generate_graphics:
            graphics["LBP histogram"] = histogram_plot([channel], channel, 0, 100, False)
        return Result(
            {
                "codes": codes,
                "valid_mask": valid.astype(np.uint8) * 255,
                "histogram": histogram,
                "sample_count": count,
            },
            graphics=graphics,
            message=f"{count} valid LBP codes · {bins} bins",
        )
