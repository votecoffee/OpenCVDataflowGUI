"""Image-grid structure tensors, unsigned orientation and explicit confidence masks."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS
from .payloads import NO_VALUE, metadata
from .processing import ImageTool, gray_image
from .selection import SELECTION_INPUTS, selection_mask

MAX_PIXELS = 2_100_000
APERTURES = {"1": 1, "3": 3, "5": 5, "7": 7, "scharr": -1}


def tensor_maps(native, selected, p):
    """Sort native eigenpairs; direction is undefined for zero or equal eigenvalues."""
    if not np.isfinite(native).all():
        raise ValidationError("Nonfinite structure tensor; rescale the floating-point input")
    first = native[..., 0] >= native[..., 1]
    high = np.maximum(np.maximum(native[..., 0], native[..., 1]), 0)
    low = np.maximum(np.minimum(native[..., 0], native[..., 1]), 0)
    x = np.where(first, native[..., 2], native[..., 4])
    y = np.where(first, native[..., 3], native[..., 5])
    # Native float arithmetic can zero one eigenvector on a weak axis-aligned
    # gradient. The other eigenvector still defines its perpendicular direction.
    missing = (x == 0) & (y == 0)
    other_x = np.where(first, native[..., 4], native[..., 2])
    other_y = np.where(first, native[..., 5], native[..., 3])
    x, y = np.where(missing, -other_y, x), np.where(missing, other_x, y)
    energy = high.astype(np.float64) + low
    if np.any(energy > np.finfo(np.float32).max):
        raise ValidationError("Structure tensor energy exceeds float32; rescale the input")
    coherence = np.divide(
        high.astype(np.float64) - low, energy, out=np.zeros_like(energy), where=energy > 0
    )
    np.clip(coherence, 0, 1, out=coherence)
    valid = (
        selected
        & (energy > float(p["min_energy"]))
        & (high > low)
        & (coherence >= float(p["min_coherence"]))
        & ((x != 0) | (y != 0))
    )
    angle = np.degrees(np.arctan2(y.astype(np.float64), x))
    if p["orientation_basis"] == "tangent":
        angle += 90
    angle %= 180
    # Round-to-float32 can otherwise turn a value just below 180 into exactly 180.
    orientation = angle.astype(np.float32) % np.float32(180)
    radians = np.deg2rad(orientation.astype(np.float64))
    return {
        "eigenvalue_max": np.where(selected, high, 0).astype(np.float32),
        "eigenvalue_min": np.where(selected, low, 0).astype(np.float32),
        "energy": np.where(selected, energy, 0).astype(np.float32),
        "coherence": np.where(selected, coherence, 0).astype(np.float32),
        "orientation": np.where(valid, orientation, 0).astype(np.float32),
        "vector_x": np.where(valid, np.cos(radians), 0).astype(np.float32),
        "vector_y": np.where(valid, np.sin(radians), 0).astype(np.float32),
        "valid": valid.astype(np.uint8) * 255,
    }


def orientation_summary(maps, selected, p):
    valid = maps["valid"] != 0
    count, total = int(valid.sum()), int(selected.sum())
    angle, concentration = None, 0.0
    if count:
        radians = np.deg2rad(maps["orientation"][valid].astype(np.float64)) * 2
        weights = maps["energy"][valid].astype(np.float64) * maps["coherence"][valid]
        x, y = float(np.sum(weights * np.cos(radians))), float(np.sum(weights * np.sin(radians)))
        concentration = min(1.0, math.hypot(x, y) / float(weights.sum()))
        if concentration > 1e-6:
            angle = math.degrees(math.atan2(y, x)) / 2 % 180
            if angle >= 180:
                angle = 0.0
    return {
        "model": "structure_tensor_v1",
        "selected_pixels": total,
        "valid_pixels": count,
        "valid_fraction": count / total if total else 0.0,
        "dominant_angle_degrees": angle,
        "orientation_concentration": concentration,
        "mean_selected_energy": float(np.mean(maps["energy"][selected], dtype=np.float64))
        if total
        else 0.0,
        "mean_valid_coherence": float(np.mean(maps["coherence"][valid], dtype=np.float64))
        if count
        else 0.0,
        "orientation_basis": p["orientation_basis"],
        "angle_convention": "unsigned degrees [0,180), +x right, +y down; current input grid",
        "summary_weighting": "energy times coherence; axial circular mean",
        "intensity_basis": "uint8/uint16 full-scale normalized; floats retain supplied units",
        "selection_policy": "select output centers; neighborhoods use the full input image",
    }


class StructureTensor(ImageTool):
    type_id, title = "inspect.structure_tensor", "Local structure tensor and orientation"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("orientation"),
        Port("coherence"),
        Port("valid"),
        Port("dominant_angle", "number", required=False),
        Port("measurements", "data"),
        Port("eigenvalue_max"),
        Port("eigenvalue_min"),
        Port("energy"),
        Port("vector_x"),
        Port("vector_y"),
        Port("orientation_preview", required=False),
        Port("overlay", required=False),
    )
    defaults = {
        "block_size": 9,
        "gradient_size": "3",
        "border": "reflect_101",
        "orientation_basis": "tangent",
        "min_energy": 1e-6,
        "min_coherence": 0.25,
        "glyph_step": 16,
        "glyph_length": 12.0,
        "max_glyphs": 1000,
    }
    parameter_choices = {
        "gradient_size": tuple(APERTURES),
        "border": ("constant", "replicate", "reflect_101", "reflect_edge"),
        "orientation_basis": ("tangent", "normal"),
    }
    parameter_ranges = {
        "block_size": NumericRange(1, 63, whole=True),
        "min_energy": NumericRange(0),
        "min_coherence": NumericRange(0, 1),
        "glyph_step": NumericRange(1, 256, whole=True),
        "glyph_length": NumericRange(1, 256),
        "max_glyphs": NumericRange(1, 2000, whole=True),
    }
    parameter_help = {
        "block_size": "Neighborhood width/height for derivative products; odd sizes are centered.",
        "gradient_size": "Sobel derivative aperture, or Scharr. Output tensors use float32.",
        "border": "Extrapolation at the full image boundary; masks do not create new borders.",
        "orientation_basis": "Tangent follows an edge; normal follows the strongest gradient axis.",
        "min_energy": (
            "Require eigenvalue sum strictly above this limit. Integers use full-scale "
            "luminance; floats retain their supplied intensity units."
        ),
        "min_coherence": (
            "Inclusive minimum (largest-smallest)/(largest+smallest). Flat/equal-eigenvalue "
            "pixels have no direction even when this limit is zero."
        ),
        "glyph_step": "Spacing of direction marks in input pixels; graphics only.",
        "glyph_length": "Full length of each unsigned direction mark in input pixels.",
        "max_glyphs": "Bounded, evenly subsampled marks across the selected sampling grid.",
    }
    input_help = {
        "image": "Gray/color uint8, uint16, float32/64 up to 2.1 MP; passes through unchanged.",
        "mask": "Optional uint8 output-center selection in the same pixel frame.",
        "region": "Optional circle, rectangle or polygon intersected with the mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [
            "image",
            "orientation",
            "coherence",
            "valid",
            "dominant_angle",
            "measurements",
        ]

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        if source.shape[0] * source.shape[1] > MAX_PIXELS:
            raise ValidationError("Structure tensors are limited to 2.1 million input pixels")
        with np.errstate(over="ignore", invalid="ignore"):
            gray = gray_image(source, (np.float32,), auto_depth=True)
        if not np.isfinite(gray).all():
            raise ValidationError("Luminance exceeds float32; rescale the floating-point input")
        mask = selection_mask(inputs, gray)
        selected = np.ones(gray.shape, bool) if mask is None else mask != 0
        native = cv2.cornerEigenValsAndVecs(
            gray,
            int(p["block_size"]),
            APERTURES[p["gradient_size"]],
            borderType=BORDERS[p["border"]],
        )
        maps = tensor_maps(native, selected, p)
        report = orientation_summary(maps, selected, p)
        context = metadata(source)
        outputs = {
            "image": source,
            "dominant_angle": NO_VALUE,
            "measurements": report,
            "orientation_preview": NO_VALUE,
            "overlay": NO_VALUE,
        }
        for name, values in maps.items():
            units = (
                "degrees_mod_180"
                if name == "orientation"
                else "binary_0_255"
                if name == "valid"
                else "unitless"
                if name in ("coherence", "vector_x", "vector_y")
                else "working_luminance_squared"
            )
            outputs[name] = ImageArray(
                values,
                "gray",
                {
                    **context,
                    "structure_tensor_map": {
                        "quantity": name,
                        "units": units,
                        "validity_output": "valid",
                        "orientation_basis": p["orientation_basis"],
                    },
                },
            )
        angle = report["dominant_angle_degrees"]
        if angle is not None:
            outputs["dominant_angle"] = angle
        result = Result(
            outputs,
            message=(
                f"{report['valid_pixels']}/{report['selected_pixels']} reliable directions; "
                + (f"dominant {angle:.2f} deg" if angle is not None else "no dominant direction")
            ),
        )
        if self.generate_graphics:
            self.add_graphics(result, source, maps)
        return result

    def add_graphics(self, result, source, maps):
        p = self.parameters
        valid = maps["valid"] != 0
        hue = np.minimum(maps["orientation"], 179).astype(np.uint8)
        saturation = np.rint(maps["coherence"] * 255).astype(np.uint8)
        hsv = np.dstack((hue, saturation, valid.astype(np.uint8) * 255))
        colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result.outputs["orientation_preview"] = colors
        step = int(p["glyph_step"])
        ys, xs = np.nonzero(valid[step // 2 :: step, step // 2 :: step])
        ys, xs = ys * step + step // 2, xs * step + step // 2
        indices = np.linspace(0, len(xs) - 1, min(len(xs), int(p["max_glyphs"])), dtype=int)
        objects = []
        half = float(p["glyph_length"]) / 2
        for index in indices:
            x, y = xs[index], ys[index]
            point = np.array([x, y], float)
            delta = half * np.array([maps["vector_x"][y, x], maps["vector_y"][y, x]])
            objects.append(
                graphic_object(
                    "polyline",
                    [point - delta, point + delta],
                    tuple(int(c) for c in colors[y, x]),
                    thickness=1,
                )
            )
        result.drawings["Local orientation"] = GraphicDrawing(tuple(objects))
        overlay, identity = overlay_image(source), np.eye(3)
        for obj in objects:
            draw_object(overlay, obj, identity)
        result.outputs["overlay"] = overlay
        result.graphics["Local orientation"] = overlay
        result.graphics["Orientation hue / coherence saturation"] = colors
