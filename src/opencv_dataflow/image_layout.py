"""Pixel-preserving composition of images with independent source frames."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame, metadata
from .processing import ImageTool, image_value
from .tickets import ticket_key

_MAX_PIXELS = 16_777_216


class JoinImages(ImageTool):
    type_id, title, subtype = "cv.join_images", "Join images", "Geometry"
    inputs = (Port("image_a"), Port("image_b"))
    outputs = (Port("image"), Port("layout", "data"))
    process_inputs_readonly = True
    defaults = {
        "direction": "horizontal",
        "size_policy": "strict",
        "alignment": "center",
        "gap": 0,
        "fill_value": 0.0,
    }
    parameter_choices = {
        "direction": ("horizontal", "vertical"),
        "size_policy": ("strict", "pad"),
        "alignment": ("start", "center", "end"),
    }
    parameter_ranges = {"gap": NumericRange(0, 16384, whole=True), "fill_value": NumericRange()}
    parameter_help = {
        "direction": "Join left to right or top to bottom, with Image A first.",
        "size_policy": "Require a common height or width, or pad the smaller image to match.",
        "alignment": "Where to place the shorter image across the join in Pad mode.",
        "gap": "Number of fill pixels between images along the join.",
        "fill_value": "Use this value for all channels in the gap and padding, in stored units.",
    }
    input_help = {
        "image_a": "Left or top image. Use the same dtype and color encoding on both inputs.",
        "image_b": "Right or bottom image. When both inputs have tickets, they must match.",
    }

    def process(self, inputs):
        a, b = inputs["image_a"], inputs["image_b"]
        first, second = image_value(a), image_value(b)
        encoding = image_format(a)
        if (
            first.dtype != second.dtype
            or first.shape[2:] != second.shape[2:]
            or encoding != image_format(b)
        ):
            raise ValidationError("Images must use the same pixel depth and color encoding")
        a_payload, b_payload = metadata(a), metadata(b)
        if (
            "ticket" in a_payload
            and "ticket" in b_payload
            and ticket_key(a_payload["ticket"]) != ticket_key(b_payload["ticket"])
        ):
            raise ValidationError("Join images from the same ticket")

        p = self.parameters
        horizontal = p["direction"] == "horizontal"
        cross_axis = 0 if horizontal else 1
        a_cross, b_cross = first.shape[cross_axis], second.shape[cross_axis]
        cross_size = max(a_cross, b_cross)
        if p["size_policy"] == "strict" and a_cross != b_cross:
            dimension = "height" if horizontal else "width"
            raise ValidationError(f"Strict mode requires the same image {dimension}", "size_policy")
        gap = int(p["gap"])
        b_start = first.shape[1 - cross_axis] + gap
        span = b_start + second.shape[1 - cross_axis]
        size = [span, cross_size] if horizontal else [cross_size, span]
        if size[0] * size[1] > _MAX_PIXELS:
            raise ValidationError("Joined image exceeds 16,777,216 pixels")

        fill = float(p["fill_value"])
        if a_cross != b_cross or gap:
            limits = np.iinfo(first.dtype) if first.dtype.kind in "iu" else np.finfo(first.dtype)
            if not float(limits.min) <= fill <= float(limits.max):
                raise ValidationError(f"Fill value exceeds the {first.dtype} range", "fill_value")
        channels = first.shape[2] if first.ndim == 3 else 1
        value = (fill,) * channels
        padded, panels = [], []
        for name, image, context, start, trailing in (
            ("image_a", first, a_payload, 0, gap),
            ("image_b", second, b_payload, b_start, 0),
        ):
            extra = cross_size - image.shape[cross_axis]
            before = {"start": 0, "center": extra // 2, "end": extra}[p["alignment"]]
            after = extra - before
            borders = (before, after, 0, trailing) if horizontal else (0, trailing, before, after)
            padded.append(
                cv2.copyMakeBorder(image, *borders, cv2.BORDER_CONSTANT, value=value)
                if any(borders)
                else image
            )
            x, y = (start, before) if horizontal else (before, start)
            panels.append(
                {
                    "input": name,
                    "rect": [x, y, image.shape[1], image.shape[0]],
                    "source_spatial": context.get("spatial"),
                    "source_pyramid": context.get("pyramid"),
                }
            )
        joined = cv2.hconcat(padded) if horizontal else cv2.vconcat(padded)
        primary = a_payload if "ticket" in a_payload else b_payload
        payload = {key: primary[key] for key in ("ticket", "processing_time") if key in primary}
        payload["spatial"] = frame(size, f"{self.id}:image")
        layout = {"direction": p["direction"], "image_size": size, "panels": panels}
        return Result(
            {"image": ImageArray(joined, encoding), "layout": layout},
            payloads={"image": payload, "layout": payload},
            message=f"Joined two images into {size[0]} × {size[1]} pixels",
        )
