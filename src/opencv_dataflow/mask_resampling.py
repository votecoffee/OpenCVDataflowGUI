"""Area-weighted mask resizing with explicit validity and pixel-grid provenance."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, moved_payload, require_same_frame

MAX_PIXELS = 16_000_000
MAX_AXIS = 16384
ROW_CHUNK = 64


def _size(width, height):
    if min(width, height) < 1 or max(width, height) > MAX_AXIS or width * height > MAX_PIXELS:
        raise ValidationError("Mask grids are limited to 16384 per axis and 16 million pixels")
    return width, height


def _overlaps(selected, size):
    """Any selected source cell with strictly positive overlap, using rational bounds."""
    sh, sw = selected.shape
    width, height = size
    integral = cv2.integral(selected.astype(np.uint8), sdepth=cv2.CV_32S)
    x0 = np.arange(width, dtype=np.int64) * sw // width
    x1 = (np.arange(1, width + 1, dtype=np.int64) * sw + width - 1) // width
    output = np.empty((height, width), bool)
    for start in range(0, height, ROW_CHUNK):
        rows = np.arange(start, min(height, start + ROW_CHUNK), dtype=np.int64)
        y0, y1 = rows * sh // height, ((rows + 1) * sh + height - 1) // height
        counts = (
            integral[y1[:, None], x1]
            - integral[y0[:, None], x1]
            - integral[y1[:, None], x0]
            + integral[y0[:, None], x0]
        )
        output[start : start + len(rows)] = counts > 0
    return output


def area_coverage(values, size, *, binary=False):
    """Integrate piecewise-constant source cells over each destination footprint."""
    sh, sw = values.shape
    width, height = size
    integral = cv2.integral(values, sdepth=cv2.CV_64F)
    numerator = np.arange(width + 1, dtype=np.int64) * sw
    x0, rx = numerator // width, numerator % width
    x1 = np.minimum(x0 + 1, sw)
    output = np.empty((height, width), np.float64)
    for start in range(0, height, ROW_CHUNK):
        stop = min(height, start + ROW_CHUNK)
        numerator = np.arange(start, stop + 1, dtype=np.int64) * sh
        y0, ry = numerator // height, numerator % height
        y1 = np.minimum(y0 + 1, sh)
        # Keep rational footprint weights as integers until the final division.
        # For binary input the bounded weighted sums stay below 2**53, so even
        # noninteger resize ratios have exact integer area numerators in float64.
        low = (width - rx) * integral[y0[:, None], x0] + rx * integral[y0[:, None], x1]
        high = (width - rx) * integral[y1[:, None], x0] + rx * integral[y1[:, None], x1]
        prefix = (height - ry[:, None]) * low + ry[:, None] * high
        bands = prefix[1:] - prefix[:-1]
        output[start:stop] = (bands[:, 1:] - bands[:, :-1]) / (sw * sh)
    del integral
    if binary:
        return output, output == 1
    complete = ~_overlaps(values < 1, size)
    positive = _overlaps(values > 0, size)
    # Integer support counts prevent cancellation/roundoff from admitting an
    # invalid footprint at threshold 1, or an empty footprint at threshold 0.
    np.clip(output, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), out=output)
    output[~positive], output[complete] = 0, 1
    return output, complete


class ResampleMask(Block):
    type_id, title = "cv.resample_mask", "Resample coverage mask"
    category = block_type = "Image processing"
    subtype = "Geometry"
    inputs = (
        Port("mask", visual_type="mask"),
        Port("target", required=False, required_when_connected=True),
    )
    outputs = (
        Port("mask", visual_type="mask"),
        Port("coverage"),
        Port("geometry", "data"),
        Port("info", "data"),
    )
    defaults = {"width": 640, "height": 480, "input_mode": "binary", "minimum_coverage": 1.0}
    parameter_ranges = {
        "width": NumericRange(1, MAX_AXIS, whole=True),
        "height": NumericRange(1, MAX_AXIS, whole=True),
        "minimum_coverage": NumericRange(0, 1),
    }
    parameter_choices = {"input_mode": ("binary", "coverage")}
    parameter_help = {
        "width": "Output width; a connected target supplies both dimensions instead.",
        "height": "Output height. Input and output each have a 16 million pixel limit.",
        "input_mode": "Binary: uint8 nonzero is valid. Coverage: float32/64 fractions in [0,1].",
        "minimum_coverage": (
            "Inclusive accepted area fraction. One requires every overlapping source cell "
            "to be fully valid. Zero accepts any positive coverage; empty cells stay invalid."
        ),
    }
    input_help = {
        "mask": "Single-channel validity mask or fractional coverage plane; no physical depth.",
        "target": (
            "Optional output-size reference for a whole-grid resize. Known coordinate frames "
            "must match that resize; this does not register unrelated images or follow a crop."
        ),
    }
    payload_reference = "mask"
    parallel_safe = preview_safe = process_inputs_readonly = True
    opencv_functions = ("integral",)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if isinstance(p["minimum_coverage"], (bool, str)):
            raise ValidationError("Use a numeric coverage fraction", "minimum_coverage")

    def process(self, inputs):
        source, p = inputs["mask"], self.parameters
        if isinstance(source, DepthArray) or source.ndim != 2:
            raise ValidationError("Supply a single-channel mask, not color or physical depth")
        sh, sw = source.shape
        _size(sw, sh)
        if p["input_mode"] == "binary":
            if source.dtype != np.uint8:
                raise ValidationError("Binary mode needs uint8; use Coverage for float fractions")
            values = (np.asarray(source) != 0).astype(np.float64)
        else:
            if source.dtype not in (np.float32, np.float64):
                raise ValidationError("Coverage mode needs float32/64 fractions in [0,1]")
            values = np.asarray(source, dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValidationError("Coverage values must be finite fractions in [0,1]")
        size = (int(p["width"]), int(p["height"]))
        target = inputs.get("target")
        if target is not None:
            size = (target.shape[1], target.shape[0])
        width, height = _size(*size)
        sx, sy = sw / width, sh / height
        matrix = [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]
        source_payload = self.input_payloads["mask"]
        payload = moved_payload(source_payload, np.asarray(matrix), list(size))
        verified = False
        if target is not None:
            try:
                require_same_frame(
                    [ValuePacket(None, payload), target],
                    message="Target must share the whole-grid resize coordinate frame",
                )
            except ValueError as exc:
                raise ValidationError(str(exc), "target") from exc
            verified = all(
                q.get("spatial", {}).get("root_id") is not None
                for q in (source_payload, self.input_payloads["target"])
            )
            if (
                verified
                and payload["spatial"]["root_size"]
                != self.input_payloads["target"]["spatial"]["root_size"]
            ):
                raise ValidationError("Target and mask declare different root grid sizes", "target")
        coverage, complete = area_coverage(values, size, binary=p["input_mode"] == "binary")
        threshold = float(p["minimum_coverage"])
        accepted = complete if threshold == 1 else (coverage > 0) & (coverage >= threshold)
        mask = accepted.astype(np.uint8) * 255
        geometry = {
            "input_size": [sw, sh],
            "output_size": list(size),
            "output_to_input": matrix,
            "coordinate_convention": (
                "continuous pixel centers; area footprints use half-pixel edges"
            ),
        }
        info = {
            "model": "mask_area_resampling_v1",
            "input_mode": p["input_mode"],
            "source_size": [sw, sh],
            "output_size": list(size),
            "minimum_coverage": threshold,
            "source_spatial": copy.deepcopy(source_payload["spatial"]),
            "output_spatial": copy.deepcopy(payload["spatial"]),
            "target_frame_verified": verified,
            "accepted_pixels": int(accepted.sum()),
            "fully_covered_pixels": int(complete.sum()),
            "mean_coverage": float(coverage.mean()),
            "source_mean_coverage": float(values.mean()),
            "sampling": "piecewise_constant_cell_area_float64",
        }
        payload["mask_resampling"] = info
        graphics = {}
        if self.generate_graphics:
            graphics["coverage"] = cv2.applyColorMap(
                np.rint(coverage * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )
        return Result(
            {
                "mask": ImageArray(mask, "gray"),
                "coverage": ImageArray(coverage, "gray"),
                "geometry": geometry,
                "info": info,
            },
            payloads={
                name: copy.deepcopy(payload) for name in ("mask", "coverage", "geometry", "info")
            },
            graphics=graphics,
            score=info["mean_coverage"],
            message=f"{int(accepted.sum())}/{mask.size} pixels accepted at coverage {threshold:g}",
        )
