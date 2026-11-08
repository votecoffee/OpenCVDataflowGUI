"""Directional gray-level co-occurrence measurements with mask-aware pairs."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .processing import gray_image
from .selection import selection_mask
from .skeleton import SKIMAGE_INSTALL

_PROPERTIES = {
    "contrast": "contrast",
    "dissimilarity": "dissimilarity",
    "homogeneity": "homogeneity",
    "energy": "energy",
    "correlation": "correlation",
    "asm": "ASM",
    "mean": "mean",
    "variance": "variance",
    "standard_deviation": "std",
    "entropy": "entropy",
}


def pair_views(image, dx, dy):
    height, width = image.shape
    origin = image[max(-dy, 0):height - max(dy, 0), max(-dx, 0):width - max(dx, 0)]
    neighbor = image[max(dy, 0):height - max(-dy, 0), max(dx, 0):width - max(-dx, 0)]
    return origin, neighbor


class GrayLevelCooccurrence(Block):
    type_id, title = "inspect.gray_level_cooccurrence", "Gray-level co-occurrence"
    category, block_type, subtype = "Inspection", "Inspection", "Texture"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("value", "number"), Port("measurements", "data"), Port("matrix"))
    defaults = {
        "offsets": [[1, 0], [0, 1]],
        "levels": 32,
        "range_mode": "dtype",
        "range_min": 0,
        "range_max": 255,
        "symmetric": True,
        "offset_index": 0,
        "metric": "contrast",
        "check_threshold": False,
        "threshold_mode": "maximum",
        "threshold": 1.0,
    }
    parameter_choices = {
        "range_mode": ("dtype", "manual"),
        "metric": tuple(_PROPERTIES),
        "threshold_mode": ("minimum", "maximum"),
    }
    parameter_ranges = {
        "levels": NumericRange(2, 256, whole=True),
        "range_min": NumericRange(0, 65535, whole=True),
        "range_max": NumericRange(0, 65535, whole=True),
        "offset_index": NumericRange(0, 15, whole=True),
        "threshold": NumericRange(),
    }
    parameter_help = {
        "offsets": "Up to 16 distinct nonzero [dx, dy] pixel offsets. Right and down are positive.",
        "levels": "Quantization bins. Keep levels and intensity range fixed across captures.",
        "range_mode": "Use the full integer depth or clamp to the manual endpoints before binning.",
        "symmetric": "Count each pair in both directions. Pair counts stay undoubled.",
        "offset_index": "Zero-based offset for the scalar output and probability matrix.",
        "metric": "Property to send to the Value output. Measurements contains all properties.",
        "threshold": "Inclusive lower or upper limit for the selected value.",
    }
    input_help = {
        "image": "Grayscale or color uint8/uint16, limited to 2.1 million pixels.",
        "mask": "Both pixels of every counted pair must be selected.",
        "region": "Both pixels must lie in the region and in any wired mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        offsets = p["offsets"]
        if not isinstance(offsets, list) or not 1 <= len(offsets) <= 16 or any(
            not isinstance(pair, list) or len(pair) != 2
            or any(type(value) is not int or abs(value) > 4096 for value in pair)
            or pair == [0, 0] for pair in offsets
        ):
            raise ValidationError(
                "Offsets need 1–16 nonzero [dx, dy] pairs within ±4096", "offsets"
            )
        if len(set(map(tuple, offsets))) != len(offsets):
            raise ValidationError("Offsets must be distinct", "offsets")
        if int(p["offset_index"]) >= len(offsets):
            raise ValidationError("Offset index must name an existing offset", "offset_index")
        if p["range_mode"] == "manual" and int(p["range_min"]) >= int(p["range_max"]):
            raise ValidationError("Manual minimum must be below maximum", "range_min", "range_max")

    def process(self, inputs):
        source = inputs["image"]
        if not 0 < source.shape[0] * source.shape[1] <= 2_100_000:
            raise ValidationError("Co-occurrence needs a nonempty image up to 2.1 million pixels")
        gray = gray_image(source, (np.uint8, np.uint16))
        mask = selection_mask(inputs, source)
        p = self.parameters
        levels = int(p["levels"])
        if p["range_mode"] == "manual":
            low, high = int(p["range_min"]), int(p["range_max"])
        else:
            low, high = 0, int(np.iinfo(gray.dtype).max)
        quantized = np.clip(gray.astype(np.int32), low, high)
        quantized = np.minimum((quantized - low) * levels // (high - low), levels - 1)

        matrices, rows = [], []
        height, width = gray.shape
        for dx, dy in p["offsets"]:
            if abs(dx) >= width or abs(dy) >= height:
                raise ValidationError(
                    f"Offset [{dx}, {dy}] exceeds the image dimensions", "offsets"
                )
            first, second = pair_views(quantized, dx, dy)
            codes = first * levels + second
            if mask is not None:
                source_mask, neighbor_mask = pair_views(mask, dx, dy)
                codes = codes[(source_mask != 0) & (neighbor_mask != 0)]
            if not codes.size:
                raise ValidationError(f"Offset [{dx}, {dy}] has no selected pixel pairs", "offsets")
            matrix = np.bincount(codes.ravel(), minlength=levels**2).reshape(levels, levels)
            matrices.append(matrix + matrix.T if p["symmetric"] else matrix)
            rows.append({"offset": [dx, dy], "pair_count": int(codes.size)})

        try:
            from skimage.feature import graycoprops
        except ImportError as exc:
            raise RuntimeError(SKIMAGE_INSTALL) from exc
        stack = np.stack(matrices, axis=-1)[..., None]
        for key, native_property in _PROPERTIES.items():
            values = graycoprops(stack, native_property).ravel()
            for row, value in zip(rows, values, strict=True):
                row[key] = float(value)

        index, metric = int(p["offset_index"]), p["metric"]
        value = rows[index][metric]
        passed = True
        if p["check_threshold"]:
            limit = float(p["threshold"])
            passed = value >= limit if p["threshold_mode"] == "minimum" else value <= limit
        probability = (matrices[index] / matrices[index].sum()).astype(np.float32)
        graphics = {}
        if self.generate_graphics:
            preview = np.rint(probability / probability.max() * 255).astype(np.uint8)
            graphics["Co-occurrence matrix"] = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        return Result(
            {
                "value": value,
                "measurements": {
                    "levels": levels, "range": [low, high],
                    "symmetric": p["symmetric"], "offsets": rows,
                },
                "matrix": probability,
            },
            payloads={"matrix": {}},
            graphics=graphics,
            passed=passed,
            fail_reasons=[] if passed else ["texture_cooccurrence_out_of_tolerance"],
            message=f"{metric.replace('_', ' ').title()} {value:.6g} at {rows[index]['offset']}",
        )
