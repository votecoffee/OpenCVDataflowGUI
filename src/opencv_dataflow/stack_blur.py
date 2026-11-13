"""Large-radius native stack smoothing with explicit size and arithmetic contracts."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask

MAX_PIXELS = 16_777_216
MAX_SCRATCH_BYTES = 128 * 1024 * 1024
SUPPORTED_DEPTHS = (np.uint8, np.uint16, np.int16, np.float32)


def effective_kernel(requested, shape, policy):
    height, width = shape[:2]
    effective = tuple(
        min(size, axis if axis % 2 else max(1, axis - 1))
        for size, axis in zip(requested, (width, height), strict=True)
    )
    if effective != requested and policy == "reject":
        raise ValidationError(
            "Kernel exceeds the image; reduce it or choose clamp", "kernel", "kernel_y"
        )
    return effective


class StackBlur(ImageTool):
    type_id, title, subtype = "cv.stack_blur", "Stack blur", "Filtering"
    inputs = SELECTION_INPUTS
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "kernel": 31,
        "kernel_y": 0,
        "oversize_policy": "clamp",
        "arithmetic": "float32",
        "alpha_mode": "preserve",
        "maximum_pixels": MAX_PIXELS,
    }
    parameter_choices = {
        "oversize_policy": ("clamp", "reject"),
        "arithmetic": ("float32", "native"),
        "alpha_mode": ("preserve", "filter"),
    }
    parameter_ranges = {
        "kernel": NumericRange(1, 2047, whole=True, odd=True),
        "kernel_y": NumericRange(1, 2047, whole=True, odd=True, zero_allowed=True),
        "maximum_pixels": NumericRange(1, MAX_PIXELS, whole=True),
    }
    parameter_help = {
        "kernel": "Positive odd X size, 1..2047. Stack blur uses replicated image borders.",
        "kernel_y": "Zero uses X; otherwise an independent positive odd Y size.",
        "oversize_policy": "Clamp follows native odd-size limits; Info records the effective size. "
        "Reject reports a kernel larger than its image axis.",
        "arithmetic": "Float32 avoids observed native integer artifacts and restores input depth "
        "by rounding/clipping. Native permits uint8 kernels up to 255 or float32.",
        "alpha_mode": "Preserve keeps alpha exactly; filter smooths all channels independently. "
        "No alpha association conversion is performed.",
        "maximum_pixels": "Whole-image work limit, including when only a region is written. "
        "Native scratch storage also has a fixed 128 MiB limit.",
    }
    input_help = {
        "image": "uint8, uint16, int16 or float32 gray/RGB/BGR/RGBA/BGRA. "
        "A mask and region intersect the write selection; neighborhoods use the full image."
    }
    process_inputs_readonly = parallel_safe = True

    def process(self, inputs):
        source = inputs["image"]
        if source.shape[0] * source.shape[1] > self.parameters["maximum_pixels"]:
            raise ValidationError("Image exceeds the stack blur pixel budget", "maximum_pixels")
        image = image_value(source, SUPPORTED_DEPTHS)
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Convert HSV/Lab to grayscale or RGB before stack smoothing")
        p = self.parameters
        requested = (int(p["kernel"]), int(p["kernel_y"]) or int(p["kernel"]))
        kernel = effective_kernel(requested, image.shape, p["oversize_policy"])
        mask = selection_mask(inputs, source)
        count = int(np.count_nonzero(mask)) if mask is not None else image.shape[0] * image.shape[1]
        alpha = image.ndim == 3 and image.shape[2] == 4
        if (
            alpha
            and p["alpha_mode"] == "preserve"
            and metadata(source).get("alpha_representation", "straight") != "straight"
        ):
            raise ValidationError(
                "Unassociate premultiplied alpha or choose filter for all channels", "alpha_mode"
            )
        if p["arithmetic"] == "native" and (
            image.dtype not in (np.uint8, np.float32)
            or image.dtype == np.uint8
            and max(kernel) > 255
        ):
            raise ValidationError(
                "Use float32 arithmetic for high-bit-depth or large integer kernels", "arithmetic"
            )
        channels = (
            1
            if image.ndim == 2
            else 3
            if alpha and p["alpha_mode"] == "preserve"
            else image.shape[2]
        )
        width, height = image.shape[1], image.shape[0]
        itemsize = 4 if p["arithmetic"] == "float32" else image.dtype.itemsize
        threads = max(1, min(cv2.getNumThreads(), max(1, height // 3)))
        row_scratch = threads * channels * (width + kernel[0] + 2) * 4
        column_scratch = width * channels * (kernel[1] * itemsize + 12)
        scratch = max(row_scratch, column_scratch)
        if scratch > MAX_SCRATCH_BYTES:
            raise ValidationError(
                "Stack blur native scratch estimate exceeds 128 MiB", "kernel", "kernel_y"
            )
        info = {
            "model": "stack_blur_v1",
            "requested_kernel": list(requested),
            "effective_kernel": list(kernel),
            "kernel_clamped": kernel != requested,
            "border": "replicate",
            "arithmetic": p["arithmetic"],
            "working_depth": "float32" if p["arithmetic"] == "float32" else image.dtype.name,
            "output_depth": image.dtype.name,
            "alpha_mode": p["alpha_mode"],
            "selected_pixels": count,
            "scratch_estimate_bytes": scratch,
            "status": "processed",
        }
        drawings = {}
        if count == 0 or kernel == (1, 1):
            output = image.copy()
            info["status"] = "empty_selection" if count == 0 else "identity"
        else:
            work = image[..., :3] if alpha and p["alpha_mode"] == "preserve" else image
            work = np.ascontiguousarray(
                work, dtype=np.float32 if p["arithmetic"] == "float32" else image.dtype
            )
            largest = max(abs(float(work.min())), abs(float(work.max())))
            weight = max((size // 2 + 1) ** 2 for size in kernel)
            if work.dtype == np.float32 and largest > np.finfo(np.float32).max / (4 * weight):
                raise ValidationError("Float image values exceed the stack accumulator range")
            smoothed = cv2.stackBlur(work, kernel)
            if not np.isfinite(smoothed).all():
                raise ValidationError("Native stack blur produced nonfinite samples")
            if smoothed.dtype != image.dtype:
                limits = np.iinfo(image.dtype)
                smoothed = np.clip(np.rint(smoothed), limits.min, limits.max).astype(image.dtype)
            if alpha and p["alpha_mode"] == "preserve":
                output = image.copy()
                output[..., :3] = smoothed
            else:
                output = smoothed
            if mask is not None:
                output[mask == 0] = image[mask == 0]
        if mask is not None and self.generate_graphics and 0 < count < mask.size:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            objects, remaining = [], 8192
            truncated = len(contours) > 128
            for contour in contours[:128]:
                if len(contour) < 2:
                    continue
                if remaining < 2:
                    truncated = True
                    break
                points = contour.reshape(-1, 2)
                if len(points) > remaining:
                    points = points[:: int(np.ceil(len(points) / remaining))]
                    truncated = True
                remaining -= len(points)
                objects.append(graphic_object("polyline", points, (180, 100, 240), closed=True))
            info["selection_outline_truncated"] = truncated
            drawings = {
                "write selection": GraphicDrawing(tuple(objects), source="image", from_output=True)
            }
        return Result(
            {"image": ImageArray(output, encoding), "info": info},
            drawings=drawings,
            message=f"Stack blur {kernel[0]} x {kernel[1]} / {info['working_depth']}",
        )
