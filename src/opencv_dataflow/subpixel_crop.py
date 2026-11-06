"""Axis-aligned bilinear patches with fractional pixel-center coordinates."""

import math

import cv2
import numpy as np

from .core import Port
from .graphics_layers import GraphicDrawing, graphic_object
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .poses import pose_for_image
from .processing import ImageTool, geometry_result, image_value
from .result_graphics import ACCEPTED

_SOURCE_DEPTHS = (np.uint8, np.uint16, np.int16, np.float32)
_MAX_PATCH_PIXELS = 16_777_216


def _replicated_support(image, left, top, size, half):
    x0, y0 = math.floor(left), math.floor(top)
    # Two extra samples cover the bilinear neighbor and a fraction rounded up to one.
    x1, y1 = x0 + size[0] + 2, y0 + size[1] + 2
    height, width = image.shape[:2]
    support = cv2.copyMakeBorder(
        image[max(y0, 0) : y1, max(x0, 0) : x1],
        max(-y0, 0),
        max(y1 - height, 0),
        max(-x0, 0),
        max(x1 - width, 0),
        cv2.BORDER_REPLICATE,
    )
    center = half + np.array([left - x0, top - y0], dtype=np.float32)
    dx, dy = (center - half).tolist()
    return support, center, (x0 + dx, y0 + dy)


def _sample_patch(image, size, center, output_depth):
    depth = image.dtype if output_depth == "same" else np.dtype(np.float32)
    # Sampling on a float32 working copy preserves stored units. It also avoids the
    # native uint8-to-float path's special treatment of very small X fractions.
    working = image if depth == np.uint8 else image.astype(np.float32, copy=False)
    if working.ndim == 3 and working.shape[2] == 4:
        colors = cv2.getRectSubPix(working[..., :3], size, center)
        alpha = cv2.getRectSubPix(working[..., 3], size, center)
        output = np.dstack((colors, alpha))
    else:
        output = cv2.getRectSubPix(working, size, center)
    if output.dtype == np.float32 and not np.isfinite(output).all():
        raise ValidationError("Subpixel sampling produced nonfinite pixels")
    if output.dtype != depth:
        limits = np.iinfo(depth)
        output = np.rint(output).clip(limits.min, limits.max).astype(depth)
    return output


class CropSubpixel(ImageTool):
    type_id, title, subtype = "cv.crop_subpixel", "Crop subpixel patch", "Geometry"
    inputs = (
        Port("image"),
        Port("center", "data", required=False, required_when_connected=True),
        Port("pose", "data", required=False, required_when_connected=True, visual_type="pose"),
    )
    outputs = (
        Port("image"),
        Port("geometry", "data"),
        Port("valid_mask", visual_type="region"),
    )
    process_inputs_readonly = True
    defaults = {
        "width": 32,
        "height": 32,
        "center_mode": "image_center",
        "center_x": 0.0,
        "center_y": 0.0,
        "bounds_policy": "strict",
        "output_depth": "same",
    }
    parameter_choices = {
        "center_mode": ("image_center", "custom"),
        "bounds_policy": ("strict", "replicate"),
        "output_depth": ("same", "float32"),
    }
    parameter_ranges = {
        "width": NumericRange(1, 16384, whole=True),
        "height": NumericRange(1, 16384, whole=True),
        "center_x": NumericRange(0),
        "center_y": NumericRange(0),
    }
    parameter_help = {
        "width": "Number of patch columns.",
        "height": "Number of patch rows.",
        "center_mode": "Use the image center or custom pixel coordinates when unwired.",
        "center_x": "Custom center column, including fractional pixels.",
        "center_y": "Custom center row, including fractional pixels.",
        "bounds_policy": "Strict requires full coverage. Replicate extends the nearest edge pixel.",
        "output_depth": "Preserve the source depth or emit float32 in stored sample units.",
    }
    input_help = {
        "center": "A single [x, y] point or one-point list in input image coordinates.",
        "pose": "Use the image pose origin as the center. Patch axes stay aligned with the image.",
    }

    def _center(self, inputs, source):
        p = self.parameters
        if "pose" in inputs:
            if "center" in inputs:
                raise ValidationError("Connect either a center or a pose")
            return pose_for_image(inputs["pose"], source)["origin"]
        if "center" in inputs:
            require_same_frame(
                (source, ValuePacket(inputs["center"], self.input_payloads["center"])),
                message="Patch center must use the input image coordinate frame",
            )
            return inputs["center"]
        if p["center_mode"] == "custom":
            return p["center_x"], p["center_y"]
        return (source.shape[1] - 1) / 2, (source.shape[0] - 1) / 2

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, _SOURCE_DEPTHS)
        p = self.parameters
        if (
            p["output_depth"] == "float32"
            and image.dtype != np.float32
            and image_format(source) in ("hsv", "lab")
        ):
            raise ValidationError("Convert HSV or Lab before changing patch depth", "output_depth")
        width, height = int(p["width"]), int(p["height"])
        if width * height > _MAX_PATCH_PIXELS:
            raise ValidationError(f"Subpixel patch exceeds {_MAX_PATCH_PIXELS:,} pixels")
        source_width, source_height = image.shape[1::-1]
        center = np.asarray(self._center(inputs, source))
        if center.shape == (1, 2):
            center = center[0]
        if center.shape != (2,) or center.dtype.kind not in "iuf" or not np.isfinite(center).all():
            raise ValidationError("Patch center must be a finite [x, y] point")
        if not 0 <= center[0] <= source_width - 1 or not 0 <= center[1] <= source_height - 1:
            raise ValidationError("Patch center must lie between the outermost input pixel centers")

        size = (width, height)
        center = center.astype(np.float32)
        half = (np.array(size, dtype=np.float32) - 1) / 2
        left, top = (center - half).tolist()
        right, bottom = left + width - 1, top + height - 1
        inside = (
            left >= 0 and top >= 0 and right <= source_width - 1 and bottom <= source_height - 1
        )
        if not inside and p["bounds_policy"] == "strict":
            raise ValidationError("Patch extends beyond the image in Strict mode", "bounds_policy")
        if not inside:
            # The native border path can leave incorrect samples on very small source images.
            image, center, (left, top) = _replicated_support(image, left, top, size, half)
            right, bottom = left + width - 1, top + height - 1

        output = _sample_patch(image, size, tuple(center.tolist()), p["output_depth"])
        result = geometry_result(output, source, [[1, 0, left], [0, 1, top], [0, 0, 1]])
        columns, rows = left + np.arange(width), top + np.arange(height)
        valid_columns = (columns >= 0) & (columns <= source_width - 1)
        valid_rows = (rows >= 0) & (rows <= source_height - 1)
        result.outputs["valid_mask"] = (valid_rows[:, None] & valid_columns).astype(np.uint8) * 255
        result.message = f"{width} × {height} patch from ({left:.5g}, {top:.5g}) px"
        if self.generate_graphics:
            outline = graphic_object(
                "polyline",
                ((left, top), (right, top), (right, bottom), (left, bottom)),
                ACCEPTED,
                closed=True,
            )
            cross = graphic_object("marker", (((left + right) / 2, (top + bottom) / 2),), ACCEPTED)
            result.drawings["subpixel patch"] = GraphicDrawing((outline, cross))
        return result
