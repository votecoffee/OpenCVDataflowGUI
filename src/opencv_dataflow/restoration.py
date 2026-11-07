"""Native denoising and repair operations with image-space selection."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .core import Port, Result
from .deblur import WienerDeblur
from .images import ImageArray, convert_depth, format_image, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import (
    FlowGroup,
    ValuePacket,
    check_spatial,
    image_with_payload,
    metadata,
    require_same_frame,
)
from .processing import ImageTool, encoded_result, finite, image_value, integer
from .selection import SELECTION_INPUTS, RegionalOperation, selection_mask


class NonLocalMeans(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.non_local_means", "Non-local means denoise", "Filters"
    defaults = {
        "mode": "channels",
        "strength": 3.0,
        "color_strength": 10.0,
        "norm": "l2",
        "template_window": 7,
        "search_window": 21,
    }
    parameter_choices = {"mode": ("channels", "luminance_chroma"), "norm": ("l2", "l1")}
    parameter_help = {
        "strength": "One value or a list per channel. Use one luminance value in color mode.",
        "norm": "Channel mode only. L1 supports uint16.",
        "color_strength": "Chroma denoising strength in luminance/chroma mode.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16))
        patch = integer(self.parameters["template_window"], "Template window", 1, 63, odd=True)
        window = integer(self.parameters["search_window"], "Search window", patch, 127, odd=True)
        channels = image.shape[2] if image.ndim == 3 else 1
        strengths = np.asarray(self.parameters["strength"], np.float32).reshape(-1)
        if (
            len(strengths) not in (1, channels)
            or not np.isfinite(strengths).all()
            or (strengths < 0).any()
        ):
            raise ValueError("Strength must be one nonnegative value or one per channel")
        if self.parameters["mode"] == "channels":
            norm = cv2.NORM_L1 if self.parameters["norm"] == "l1" else cv2.NORM_L2
            if image.dtype == np.uint16 and norm != cv2.NORM_L1:
                raise ValueError("uint16 denoising requires the L1 norm")
            output = cv2.fastNlMeansDenoising(image, strengths.tolist(), None, patch, window, norm)
        else:
            encoding = image_format(source)
            if len(strengths) != 1 or image.dtype != np.uint8 or encoding not in ("bgr", "rgb"):
                raise ValueError("Luminance/chroma mode requires uint8 BGR/RGB and one strength")
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb" else image
            color = finite(self.parameters["color_strength"], "Color strength", 0)
            output = cv2.fastNlMeansDenoisingColored(
                bgr, None, float(strengths[0]), color, patch, window
            )
            if encoding == "rgb":
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return encoded_result(output, source)


def aligned_observations(entries):
    if any(not isinstance(entry.value, np.ndarray) or not entry.value.size for entry in entries):
        raise ValidationError("Every observation must be a nonempty image")
    originals = [entry.value for entry in entries]
    center = originals[0]
    encoding = image_format(center)
    if any(
        frame.shape != center.shape
        or frame.dtype != center.dtype
        or image_format(frame) != encoding
        for frame in originals
    ):
        raise ValidationError("Temporal frames must have matching size, depth and channel order")
    if sum(frame.nbytes for frame in originals) > 128 * 1024**2:
        raise ValidationError("Observations exceed the 128 MiB source limit")
    roots = {}
    for entry in entries:
        context = metadata(entry)
        check_spatial(context, [center.shape[1], center.shape[0]])
        root = context.get("spatial", {}).get("root_id")
        if root is not None:
            require_same_frame(
                (roots.setdefault(root, entry), entry),
                message="Frames sharing a source root must use the same pixel grid",
            )
    return originals


class TemporalNonLocalMeans(ImageTool):
    type_id, title, subtype = "cv.temporal_nlm", "Temporal non-local means", "Filters"
    inputs = (Port("frames", "any"), *SELECTION_INPUTS[1:])
    outputs = (Port("image"), Port("info", "data"))
    defaults = {**NonLocalMeans.defaults, "target_index": -1, "temporal_window": 3}
    parameter_choices = NonLocalMeans.parameter_choices
    parameter_ranges = {
        "color_strength": NumericRange(0, 255),
        "template_window": NumericRange(1, 63, whole=True, odd=True),
        "search_window": NumericRange(1, 127, whole=True, odd=True),
        "target_index": NumericRange(-1, 63, whole=True),
        "temporal_window": NumericRange(1, 31, whole=True, odd=True),
    }
    parameter_help = {
        **NonLocalMeans.parameter_help,
        "target_index": "Zero-based frame to denoise. −1 chooses the earlier middle frame.",
        "temporal_window": "Odd number of frames centered on the target.",
    }
    input_help = {"frames": "Ordered batch of aligned frames of the same scene."}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["search_window"] < p["template_window"]:
            raise ValidationError(
                "Search window must be at least the template size",
                "search_window",
                "template_window",
            )

    def process(self, inputs):
        group = inputs["frames"]
        if not isinstance(group, FlowGroup):
            raise ValidationError("Connect an ordered image batch to Frames")
        p = self.parameters
        size = int(p["temporal_window"])
        half = size // 2
        target = int(p["target_index"])
        if target == -1:
            target = (len(group.items) - 1) // 2
        if target - half < 0 or target + half >= len(group.items):
            raise ValidationError("The temporal window does not fit around the target frame")
        entries = group.items[target - half : target + half + 1]
        originals = aligned_observations(entries)
        center = originals[half]
        encoding = image_format(center)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Temporal denoising requires gray or RGB/BGR channel encoding")
        frames = [image_value(frame, (np.uint8, np.uint16)) for frame in originals]
        payload = metadata(entries[half])
        mask = selection_mask(inputs, image_with_payload(center, payload))
        alpha = encoding in ("rgba", "bgra")
        if alpha:
            frames = [np.ascontiguousarray(frame[..., :3]) for frame in frames]
        channels = 1 if center.ndim == 2 else 3
        try:
            strengths = np.asarray(p["strength"], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Strength must be numeric", "strength") from exc
        if (
            len(strengths) not in (1, channels)
            or not np.isfinite(strengths).all()
            or np.any((strengths < 0) | (strengths > np.iinfo(center.dtype).max))
        ):
            raise ValidationError(
                "Strength needs one value or one per filtered channel within the pixel range",
                "strength",
            )
        patch, search = int(p["template_window"]), int(p["search_window"])
        if p["mode"] == "luminance_chroma":
            if channels != 3 or center.dtype != np.uint8 or len(strengths) != 1:
                raise ValidationError("Luminance/chroma mode requires 8-bit color and one strength")
            rgb = encoding in ("rgb", "rgba")
            if rgb:
                frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
            output = cv2.fastNlMeansDenoisingColoredMulti(
                frames,
                half,
                size,
                None,
                float(strengths[0]),
                float(p["color_strength"]),
                patch,
                search,
            )
            if rgb:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        else:
            norm = cv2.NORM_L1 if p["norm"] == "l1" else cv2.NORM_L2
            if center.dtype == np.uint16 and norm != cv2.NORM_L1:
                raise ValidationError("16-bit temporal denoising requires the L1 norm")
            output = cv2.fastNlMeansDenoisingMulti(
                frames, half, size, strengths.tolist(), None, patch, search, norm
            )
        if alpha:
            output = np.concatenate((output, np.asarray(center)[..., 3:]), axis=2)
        if mask is not None:
            output[mask == 0] = np.asarray(center)[mask == 0]
        info = {"target_index": target, "first_index": target - half, "window_size": size}
        return Result(
            {"image": ImageArray(output, encoding), "info": info},
            payloads={"image": payload, "info": payload},
            message=f"Denoised frame {target} using {size} frames",
        )


class DenoiseTVL1(ImageTool):
    type_id, title, subtype = "cv.denoise_tvl1", "TV-L1 denoise", "Filters"
    inputs = (
        Port("image", required=False, required_when_connected=True),
        Port("observations", "any", required=False, required_when_connected=True),
        *SELECTION_INPUTS[1:],
    )
    outputs = (Port("image"), Port("observation_count", "number"))
    defaults = {"lambda_weight": 1.0, "iterations": 30}
    parameter_ranges = {
        "lambda_weight": NumericRange(1e-6, 1e6),
        "iterations": NumericRange(1, 200, whole=True),
    }
    parameter_help = {
        "lambda_weight": "Weight of the L1 observation error relative to total variation.",
        "iterations": "Number of primal-dual solver iterations.",
    }
    input_help = {
        "image": "One 8-bit monochrome observation. Connect Image or Observations.",
        "observations": "Ordered batch of aligned 8-bit monochrome views of the same scene.",
    }

    def process(self, inputs):
        if ("image" in inputs) == ("observations" in inputs):
            raise ValidationError("Connect exactly one of Image or Observations")
        if "observations" in inputs:
            group = inputs["observations"]
            if not isinstance(group, FlowGroup):
                raise ValidationError("Connect an ordered image batch to Observations")
            entries = group.items
        else:
            entries = (ValuePacket(inputs["image"], self.input_payloads["image"]),)

        originals = aligned_observations(entries)
        reference = (len(entries) - 1) // 2
        source = originals[reference]
        if source.ndim != 2 or source.dtype != np.uint8:
            raise ValidationError("TV-L1 requires 8-bit monochrome images")
        # OpenCV allocates an eight-byte residual for each input and three work planes.
        if (len(entries) + 3) * source.size * 8 > 256 * 1024**2:
            raise ValidationError(
                "TV-L1 workspace exceeds 256 MiB. Reduce the batch or image size."
            )
        p = self.parameters
        iterations = int(p["iterations"])
        if (len(entries) + 1) * source.size * iterations > 512_000_000:
            raise ValidationError(
                "TV-L1 work exceeds the limit. Reduce batch, image or iterations."
            )
        payload = metadata(entries[reference])
        mask = selection_mask(inputs, image_with_payload(source, payload))
        output = np.empty(source.shape, dtype=np.uint8)
        cv2.denoise_TVL1(
            [np.asarray(image) for image in originals],
            output,
            lambda_=float(p["lambda_weight"]),
            niters=iterations,
        )
        if mask is not None:
            output[mask == 0] = np.asarray(source)[mask == 0]
        outputs = {"image": ImageArray(output, "gray"), "observation_count": len(entries)}
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            message=f"Denoised {len(entries)} observations",
        )


class Inpaint(ImageTool):
    type_id, title, subtype = "cv.inpaint", "Inpaint selected pixels", "Filters"
    inputs = SELECTION_INPUTS
    defaults = {"method": "telea", "radius": 3.0}
    parameter_choices = {"method": ("telea", "navier_stokes")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.float32))
        if image_format(source) not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValueError("Inpainting accepts monochrome, BGR, RGB, BGRA, or RGBA")
        mask = selection_mask(inputs, image)
        if mask is None:
            raise ValueError("Connect an inpainting mask or region")
        if np.all(mask):
            raise ValueError("Inpainting needs unselected neighboring pixels")
        radius = finite(self.parameters["radius"], "Radius", 1)
        if radius > 100:
            raise ValueError("Inpainting radius must be at most 100 pixels")
        method = cv2.INPAINT_TELEA if self.parameters["method"] == "telea" else cv2.INPAINT_NS
        if image.ndim == 2:
            return encoded_result(cv2.inpaint(image, mask, radius, method), source)
        colors = image[..., :3]
        if image.dtype == np.uint8:
            output = cv2.inpaint(colors, mask, radius, method)
        else:
            output = cv2.merge(
                [cv2.inpaint(channel, mask, radius, method) for channel in cv2.split(colors)]
            )
        if image.shape[2] == 4:
            output = np.concatenate((output, image[..., 3:]), axis=2)
        return encoded_result(output, source)


class AlignedGuideFilter(ImageTool, ABC):
    subtype = "Filters"
    inputs = (*SELECTION_INPUTS, Port("guide", required=False))
    parameter_choices = {"output_depth": ("unchanged", "uint8", "uint16", "float32")}
    input_help = {"guide": "Optional aligned guide. Otherwise the block guides with the source."}

    def process(self, inputs):
        source = inputs["image"]
        guide = inputs.get("guide", source)
        image_value(source, (np.uint8, np.uint16, np.float32))
        image_value(guide, (np.uint8, np.uint16, np.float32))
        if guide.shape[:2] != source.shape[:2]:
            raise ValidationError("Guide must have the same dimensions as the source")
        try:
            require_same_frame((source, guide))
        except ValueError as exc:
            raise ValidationError("Guide must share the source pixel frame") from exc
        if image_format(source) in ("hsv", "lab") or image_format(guide) in ("hsv", "lab"):
            raise ValidationError("Use grayscale or RGB/BGR intensity data for filtering")

        work = convert_depth(source, "float32")
        reference = work if guide is source else convert_depth(guide, "float32")
        has_alpha = work.ndim == 3 and work.shape[2] == 4
        signal = work[..., :3] if has_alpha else work
        if reference.ndim == 3:
            reference = reference[..., :3]
        result = self.filter_image(reference, signal)
        if not np.isfinite(result).all():
            raise ValidationError(f"{self.title} produced nonfinite samples. Check the input range")
        if has_alpha:
            result = np.concatenate((result, work[..., 3:]), axis=2)
        mask = selection_mask(inputs, source)
        if mask is not None:
            result[mask == 0] = work[mask == 0]
        depth = self.parameters["output_depth"]
        if depth == "unchanged":
            depth = source.dtype.name
        return encoded_result(convert_depth(result, depth), source)

    @abstractmethod
    def filter_image(self, reference, signal):
        """Filter the float32 signal using the aligned reference."""


class GuidedFilter(AlignedGuideFilter):
    type_id, title = "cv.guided_filter", "Guided filter"
    defaults = {"radius": 8, "epsilon": 0.001, "scale": 1.0, "output_depth": "unchanged"}
    parameter_ranges = {
        "radius": NumericRange(1, 4095, whole=True),
        "epsilon": NumericRange(1e-8, 1e8),
        "scale": NumericRange(0.01, 1),
    }
    parameter_help = {
        "radius": "Neighborhood radius in pixels.",
        "epsilon": "Regularization in squared guide units. Integer guide samples are normalized.",
        "scale": "Internal subsampling fraction. One filters at full resolution.",
    }

    def filter_image(self, reference, signal):
        p = self.parameters
        scale = float(p["scale"])
        if scale * min(signal.shape[:2]) < 1:
            raise ValidationError("Scale is too small for the image dimensions")
        return cv2.ximgproc.guidedFilter(
            reference, signal, int(p["radius"]), float(p["epsilon"]), scale=scale
        )


class DomainTransformFilter(AlignedGuideFilter):
    type_id, title = "cv.domain_transform_filter", "Domain transform filter"
    defaults = {
        "sigma_spatial": 8.0,
        "sigma_color": 0.08,
        "mode": "normalized_convolution",
        "iterations": 3,
        "output_depth": "unchanged",
    }
    modes = {
        "normalized_convolution": cv2.ximgproc.DTF_NC,
        "interpolated_convolution": cv2.ximgproc.DTF_IC,
        "recursive_filtering": cv2.ximgproc.DTF_RF,
    }
    parameter_choices = {**AlignedGuideFilter.parameter_choices, "mode": tuple(modes)}
    parameter_ranges = {
        "sigma_spatial": NumericRange(0.01, 4096),
        "sigma_color": NumericRange(1e-6, 1e4),
        "iterations": NumericRange(1, 16, whole=True),
    }
    parameter_help = {
        "sigma_spatial": "Spatial smoothing scale in pixels.",
        "sigma_color": "Edge sensitivity in guide units. Integer guides are normalized to [0, 1].",
        "iterations": "Number of horizontal and vertical filtering passes.",
    }

    def filter_image(self, reference, signal):
        p = self.parameters
        return cv2.ximgproc.dtFilter(
            reference,
            signal,
            float(p["sigma_spatial"]),
            float(p["sigma_color"]),
            mode=self.modes[p["mode"]],
            numIters=int(p["iterations"]),
        )


class SeamlessClone(ImageTool):
    type_id, title, subtype = "cv.seamless_clone", "Seamless clone", "Compositing"
    inputs = (Port("source"), Port("destination"), *SELECTION_INPUTS[1:])
    payload_reference = "destination"
    modes = {
        "normal": cv2.NORMAL_CLONE,
        "mixed": cv2.MIXED_CLONE,
        "monochrome_transfer": cv2.MONOCHROME_TRANSFER,
        "normal_wide": cv2.NORMAL_CLONE_WIDE,
        "mixed_wide": cv2.MIXED_CLONE_WIDE,
        "monochrome_transfer_wide": cv2.MONOCHROME_TRANSFER_WIDE,
    }
    defaults = {"mode": "normal", "center_x": -1, "center_y": -1}
    parameter_choices = {"mode": tuple(modes)}
    parameter_ranges = {
        "center_x": NumericRange(-1, whole=True),
        "center_y": NumericRange(-1, whole=True),
    }
    parameter_help = {
        "mode": "Wide modes place the full source center. Other modes center the selected bounds.",
        "center_x": "Destination column in pixels. Use -1 for the destination center.",
        "center_y": "Destination row in pixels. Use -1 for the destination center.",
    }
    input_help = {
        "source": "8-bit color patch to blend.",
        "destination": "8-bit color image that supplies the output size and coordinate frame.",
        "mask": "Optional 8-bit binary mask in source coordinates. Intersects the region if given.",
        "region": "Optional rectangle, circle, or polygon in source coordinates.",
    }

    def process(self, inputs):
        source, destination = inputs["source"], inputs["destination"]
        patch = image_value(source, (np.uint8,), canonical=True)
        target = image_value(destination, (np.uint8,), canonical=True)
        if patch.ndim != 3 or target.ndim != 3 or not patch.size or not target.size:
            raise ValidationError("Seamless clone requires two nonempty 8-bit color images")
        mask = selection_mask({**inputs, "image": source}, patch)
        if mask is None:
            mask = np.full(patch.shape[:2], 255, np.uint8)
        x, y, width, height = cv2.boundingRect(mask)
        if not width:
            return encoded_result(
                np.asarray(destination).copy(), destination, message="Empty source selection"
            )
        if min(width, height) < 3:
            raise ValidationError("The selected source bounds must be at least 3 × 3 pixels")

        p = self.parameters
        target_height, target_width = target.shape[:2]
        cx = target_width // 2 if p["center_x"] == -1 else p["center_x"]
        cy = target_height // 2 if p["center_y"] == -1 else p["center_y"]
        offset_x, offset_y = width // 2, height // 2
        if p["mode"].endswith("_wide"):
            offset_x, offset_y = patch.shape[1] // 2 - x, patch.shape[0] // 2 - y
        left, top = cx - offset_x, cy - offset_y
        if (
            not 0 <= cx < target_width or not 0 <= cy < target_height
            or not 0 <= left <= target_width - width or not 0 <= top <= target_height - height
        ):
            raise ValidationError(
                "The selected source bounds must fit at the destination center",
                "center_x", "center_y",
            )

        # The native binding clears the outer mask pixels. Padding preserves a
        # selection that reaches the edge without shifting either center convention.
        padded_patch = cv2.copyMakeBorder(patch[..., :3], 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        target = target[..., :3]
        output = cv2.seamlessClone(
            padded_patch, target, padded_mask, (cx, cy), self.modes[p["mode"]]
        )
        encoded = format_image(output, image_format(destination), "unchanged")
        result = np.asarray(destination).copy()
        changed = np.any(output != target, axis=2)
        result[..., :3][changed] = encoded[..., :3][changed]
        return encoded_result(result, destination)


RESTORATION_BLOCKS = (
    NonLocalMeans,
    TemporalNonLocalMeans,
    DenoiseTVL1,
    WienerDeblur,
    Inpaint,
    GuidedFilter,
    DomainTransformFilter,
    SeamlessClone,
)
