"""Ternary binary-neighborhood matching with explicit image/selection support."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import sampled_nonzero_points
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, anchor_value
from .processing import Morphology, image_value
from .selection import SELECTION_INPUTS, selection_mask


class HitOrMiss(Block):
    type_id, title, category = "cv.hit_or_miss", "Hit-or-miss binary pattern", "Image processing"
    block_type, subtype = "Image processing", "Morphology"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("points", "data"),
        Port("count", "integer"),
        Port("measurements", "data"),
    )
    supports_kernel_editor = True
    kernel_cell_values = (0, 1, -1)
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "shape": "custom",
        "kernel": 3,
        "kernel_y": 0,
        "shape_anchor": [-1, -1],
        "custom_kernel": [[-1, 1, -1], [1, 1, 1], [-1, 1, -1]],
        "anchor": [-1, -1],
        "iterations": 1,
        "border": "constant",
        "border_value": [],
        "support": "full_kernel",
        "input_mode": "strict_binary",
        "polarity": "light",
        "max_points": 10000,
        "max_graphics": 200,
        "check_count": False,
        "minimum_count": 0,
        "maximum_count": -1,
    }
    parameter_choices = {
        "shape": Morphology.parameter_choices["shape"],
        "border": tuple(BORDERS),
        "support": ("full_kernel", "native_border"),
        "input_mode": ("strict_binary", "nonzero"),
        "polarity": ("light", "dark"),
    }
    parameter_ranges = {
        "kernel": NumericRange(1, 255, whole=True),
        "kernel_y": NumericRange(0, 255, whole=True),
        "iterations": NumericRange(1, 100, whole=True),
        "max_points": NumericRange(0, 100000, whole=True),
        "max_graphics": NumericRange(0, 2000, whole=True),
        "minimum_count": NumericRange(0, 64_000_000, whole=True),
        "maximum_count": NumericRange(-1, 64_000_000, whole=True),
    }
    parameter_help = {
        "shape": "Standard shapes require foreground at active cells. Custom allows ternary cells.",
        "custom_kernel": (
            "1 requires foreground, -1 requires background, 0 ignores a cell. "
            "Set at least one constraint."
        ),
        "kernel": "Standard shape width, 1..255. Custom uses its actual grid dimensions.",
        "kernel_y": "Standard shape height; zero uses the width.",
        "shape_anchor": "Construction origin for Cross; separate from the match anchor.",
        "anchor": "[x,y] reported match position within the kernel; -1 means centered per axis.",
        "iterations": (
            "Native erosions of foreground and background constraints; "
            "does not repeat the whole hit-or-miss operation."
        ),
        "border": "Native extrapolation. Full kernel still excludes out-of-image footprints.",
        "border_value": (
            "Empty uses native neutral borders; otherwise [0] or [255], "
            "applied independently to the image and its complement."
        ),
        "support": (
            "Full kernel requires the entire repeated footprint within the image. "
            "Native border keeps padded matches."
        ),
        "input_mode": (
            "Strict accepts uint8 gray 0/1 or 0/255. Nonzero explicitly treats any "
            "nonzero uint8 sample as foreground."
        ),
        "polarity": "Light uses nonzero as foreground; Dark reverses foreground and background.",
        "max_points": (
            "Coordinate cap; excess matches are evenly sampled in row-major order. "
            "Zero omits coordinates. Mask and count stay complete."
        ),
        "max_graphics": "Maximum markers from returned coordinates; also respects graphics-off.",
        "check_count": "Enable an inspection judgment using the total selected match-pixel count.",
        "minimum_count": "Inclusive minimum when Check count is enabled.",
        "maximum_count": "Inclusive maximum; -1 means no upper limit.",
    }
    input_help = {
        "image": "Nonempty uint8 monochrome binary image. Threshold grayscale/color upstream.",
        "mask": "Optional uint8 selection of reported match centers; neighbors use the full image.",
        "region": (
            "Optional region intersected with Mask; filters match centers "
            "without cropping neighborhoods."
        ),
    }

    @staticmethod
    def kernel_matrix(parameters):
        if parameters["shape"] != "custom":
            return Morphology.kernel_matrix(parameters).astype(np.int8)
        try:
            kernel = np.asarray(parameters["custom_kernel"])
            valid = (
                kernel.ndim == 2
                and all(1 <= n <= 255 for n in kernel.shape)
                and kernel.dtype.kind in "iuf"
                and np.isin(kernel, (-1, 0, 1)).all()
                and np.any(kernel)
            )
        except TypeError, ValueError:
            valid = False
        if not valid:
            raise ValidationError(
                "Custom kernel needs a rectangular grid of -1, 0, 1, up to 255 x 255, "
                "with at least one constraint",
                "custom_kernel",
            )
        return kernel.astype(np.int8)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        kernel = self.kernel_matrix(p)
        try:
            anchor_value(p["anchor"], kernel.shape[1], kernel.shape[0])
        except (ValueError, TypeError) as exc:
            raise ValidationError(str(exc), "anchor") from exc
        border = p["border_value"]
        if (
            not isinstance(border, list)
            or len(border) > 1
            or any(isinstance(v, bool) or v not in (0, 255) for v in border)
        ):
            raise ValidationError("Use an empty border value or [0] or [255]", "border_value")
        if not isinstance(p["check_count"], bool):
            raise ValidationError("Check count must be a checkbox value", "check_count")
        if p["maximum_count"] >= 0 and p["maximum_count"] < p["minimum_count"]:
            raise ValidationError(
                "Maximum count must be -1 or at least Minimum count",
                "minimum_count",
                "maximum_count",
            )

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image = image_value(source, (np.uint8,))
        if image.ndim != 2 or not image.size or image_format(source) != "gray":
            raise ValidationError("Hit-or-miss needs a nonempty uint8 monochrome image")
        if max(image.shape) > 16384 or image.size > 64_000_000:
            raise ValidationError("Hit-or-miss input exceeds 16384 per axis or 64 million pixels")
        if p["input_mode"] == "strict_binary":
            if not (np.isin(image, (0, 1)).all() or np.isin(image, (0, 255)).all()):
                raise ValidationError(
                    "Strict input requires 0/1 or 0/255; threshold upstream or select Nonzero",
                    "input_mode",
                )
        binary = np.where(image != 0, 255, 0).astype(np.uint8)
        if p["polarity"] == "dark":
            binary = cv2.bitwise_not(binary)
        selected = selection_mask(inputs, source)
        kernel = self.kernel_matrix(p)
        height, width = kernel.shape
        anchor = anchor_value(p["anchor"], width, height)
        anchor = tuple(
            size // 2 if v == -1 else v for v, size in zip(anchor, (width, height), strict=True)
        )
        iterations = int(p["iterations"])
        options = {"anchor": anchor, "iterations": iterations, "borderType": BORDERS[p["border"]]}
        if p["border_value"]:
            options["borderValue"] = p["border_value"][0]
        full_pixels = image.size
        if p["support"] == "full_kernel":
            left, top = anchor[0] * iterations, anchor[1] * iterations
            right, bottom = (
                (width - 1 - anchor[0]) * iterations,
                (height - 1 - anchor[1]) * iterations,
            )
            h, w = image.shape
            full_pixels = max(0, w - left - right) * max(0, h - top - bottom)
        matches = (
            cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel, **options)
            if full_pixels
            else np.zeros_like(binary)
        )
        if p["support"] == "full_kernel":
            matches[:top] = 0
            matches[:, :left] = 0
            if bottom:
                matches[max(0, h - bottom) :] = 0
            if right:
                matches[:, max(0, w - right) :] = 0
        if selected is not None:
            matches[selected == 0] = 0
        count = cv2.countNonZero(matches)
        points = (
            sampled_nonzero_points(matches, count, int(p["max_points"]))
            if p["max_points"]
            else np.empty((0, 2), np.int32)
        )
        passed = not p["check_count"] or (
            count >= p["minimum_count"] and (p["maximum_count"] < 0 or count <= p["maximum_count"])
        )
        graphics_count = min(len(points), int(p["max_graphics"])) if self.generate_graphics else 0
        color = (70, 220, 140) if passed else (70, 70, 240)
        drawings = (
            {
                "binary matches": GraphicDrawing(
                    tuple(
                        graphic_object("marker", [point], color)
                        for point in points[:graphics_count]
                    )
                )
            }
            if graphics_count
            else {}
        )
        measurements = {
            "count": count,
            "returned_count": len(points),
            "points_sampled": len(points) < count,
            "drawn_count": graphics_count,
            "anchor": list(anchor),
            "kernel_size": [width, height],
            "foreground_cells": int(np.count_nonzero(kernel == 1)),
            "background_cells": int(np.count_nonzero(kernel == -1)),
            "iterations": iterations,
            "support": p["support"],
            "support_centers": full_pixels,
            "input_mode": p["input_mode"],
            "polarity": p["polarity"],
            "coordinate_convention": "kernel anchor, input image pixels, x/y",
        }
        message = f"{count} binary pattern matches; {len(points)} coordinates returned"
        if not passed:
            expected = (
                f"{p['minimum_count']}..{p['maximum_count']} inclusive"
                if p["maximum_count"] >= 0
                else f"at least {p['minimum_count']}"
            )
            message += f"; expected {expected} matched anchor pixels"
        return Result(
            {
                "image": ImageArray(matches, "gray"),
                "points": points.tolist(),
                "count": count,
                "measurements": measurements,
            },
            passed=passed,
            drawings=drawings,
            fail_reasons=[]
            if passed
            else [{"code": "hit_or_miss_count_out_of_range", "message": message}],
            message=message,
        )
