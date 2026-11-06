"""Pixel-preserving composition of images with independent source frames."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame, metadata
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask
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
            raise ValidationError(f"Joined image exceeds {_MAX_PIXELS:,} pixels")

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


class TileImage(ImageTool):
    type_id, title, subtype = "cv.tile_image", "Tile image", "Geometry"
    outputs = (Port("image"), Port("layout", "data"))
    process_inputs_readonly = True
    defaults = {"rows": 2, "columns": 2}
    parameter_ranges = {
        "rows": NumericRange(1, 4096, whole=True),
        "columns": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "rows": "Number of full copies vertically.",
        "columns": "Number of full copies horizontally.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        rows, columns = int(self.parameters["rows"]), int(self.parameters["columns"])
        height, width = image.shape[:2]
        size = [width * columns, height * rows]
        if size[0] * size[1] > _MAX_PIXELS:
            raise ValidationError(f"Tiled image exceeds {_MAX_PIXELS:,} pixels", "rows", "columns")

        context = metadata(source)
        payload = context
        if rows != 1 or columns != 1:
            # Repeated pixels cannot share the source's single spatial transform.
            payload = {key: context[key] for key in ("ticket", "processing_time") if key in context}
            payload["spatial"] = frame(size, f"{self.id}:image")
        layout = {
            "rows": rows,
            "columns": columns,
            "image_size": size,
            "tile_size": [width, height],
            "source_spatial": context.get("spatial"),
            "source_pyramid": context.get("pyramid"),
        }
        return Result(
            {
                "image": ImageArray(cv2.repeat(image, rows, columns), image_format(source)),
                "layout": layout,
            },
            payloads={"image": payload, "layout": payload},
            message=f"Repeated the image in {rows} rows and {columns} columns",
        )


class PasteImagePatch(ImageTool):
    type_id, title, subtype = "cv.paste_patch", "Paste image patch", "Geometry"
    inputs = (
        Port("canvas"),
        Port("patch"),
        Port("x", "integer", required=False, required_when_connected=True),
        Port("y", "integer", required=False, required_when_connected=True),
        *SELECTION_INPUTS[1:],
    )
    outputs = (
        Port("image"),
        Port("copied_mask", visual_type="region"),
        Port("placement", "data"),
        Port("copied_pixel_count", "number"),
    )
    payload_reference = "canvas"
    process_inputs_readonly = True
    defaults = {"x": 0, "y": 0, "bounds_policy": "clip"}
    parameter_choices = {"bounds_policy": ("clip", "strict")}
    parameter_ranges = {
        axis: NumericRange(-1_000_000, 1_000_000, whole=True) for axis in ("x", "y")
    }
    parameter_help = {
        "x": "Canvas X coordinate of the patch's top-left pixel. A wired value takes precedence.",
        "y": "Canvas Y coordinate of the patch's top-left pixel. A wired value takes precedence.",
        "bounds_policy": "Clip uses the visible overlap. Strict requires the whole patch to fit.",
    }
    input_help = {
        "canvas": "Image to copy and update. Its coordinates are used for the result.",
        "patch": "Samples to copy. Pixel depth and channel encoding must match the canvas.",
        "mask": "Optional monochrome mask in patch coordinates.",
        "region": "Optional region in patch coordinates.",
    }

    def process(self, inputs):
        canvas, patch = inputs["canvas"], inputs["patch"]
        base = image_value(canvas, allow_labels=True)
        pixels = image_value(patch, allow_labels=True)
        encoding = image_format(canvas)
        if (
            base.dtype != pixels.dtype
            or base.shape[2:] != pixels.shape[2:]
            or encoding != image_format(patch)
        ):
            raise ValidationError(
                "Canvas and patch must have the same pixel depth and color encoding"
            )
        if base.shape[0] * base.shape[1] > _MAX_PIXELS:
            raise ValidationError(f"Canvas exceeds {_MAX_PIXELS:,} pixels")
        payload, patch_payload = metadata(canvas), metadata(patch)
        if "ticket" in patch_payload:
            if "ticket" in payload and ticket_key(payload["ticket"]) != ticket_key(
                patch_payload["ticket"]
            ):
                raise ValidationError("Paste images from the same ticket")
            if "ticket" not in payload:
                payload.update(
                    {
                        key: patch_payload[key]
                        for key in ("ticket", "processing_time")
                        if key in patch_payload
                    }
                )

        x, y = (
            self.parameter_ranges[axis].check(inputs.get(axis, self.parameters[axis]), axis, axis)
            for axis in ("x", "y")
        )
        height, width = base.shape[:2]
        patch_h, patch_w = pixels.shape[:2]
        if self.parameters["bounds_policy"] == "strict" and (
            x < 0 or y < 0 or x + patch_w > width or y + patch_h > height
        ):
            raise ValidationError(
                "Patch must fit inside the canvas in Strict mode", "bounds_policy"
            )
        selection = selection_mask(
            {"image": patch, "mask": inputs.get("mask"), "region": inputs.get("region")}, patch
        )
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + patch_w), min(height, y + patch_h)
        output = base.copy()
        copied_mask = np.zeros((height, width), dtype=np.uint8)
        source_rect = destination_rect = None
        count = 0
        reason = "paste_no_overlap"
        if x0 < x1 and y0 < y1:
            destination = np.s_[y0:y1, x0:x1]
            origin = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
            patch_mask = None if selection is None else selection[origin]
            cv2.copyTo(pixels[origin], patch_mask, dst=output[destination])
            copied_mask[destination] = 255 if patch_mask is None else patch_mask
            count = int(cv2.countNonZero(copied_mask[destination]))
            source_rect = [x0 - x, y0 - y, x1 - x0, y1 - y0]
            destination_rect = [x0, y0, x1 - x0, y1 - y0]
            reason = "paste_empty_selection"

        placement = {
            "requested_rect": [x, y, patch_w, patch_h],
            "source_rect": source_rect,
            "destination_rect": destination_rect,
            "source_to_target": [[1, 0, x], [0, 1, y], [0, 0, 1]],
            "source_size": [patch_w, patch_h],
            "target_size": [width, height],
            "source_spatial": patch_payload.get("spatial"),
        }
        return Result(
            {
                "image": ImageArray(output, encoding),
                "copied_mask": copied_mask,
                "placement": placement,
                "copied_pixel_count": count,
            },
            payloads={
                name: payload
                for name in ("image", "copied_mask", "placement", "copied_pixel_count")
            },
            passed=bool(count),
            fail_reasons=[] if count else [reason],
            message=f"Copied {count} patch pixels",
        )
