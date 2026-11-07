"""Native exposure-stack registration and fusion for static inspection scenes."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, frame, image_with_payload, metadata, moved_payload
from .processing import ImageTool, gray_image, image_value
from .restoration import aligned_observations
from .selection import SELECTION_INPUTS, selection_mask


class _ExposureStack(ImageTool):
    subtype = "Exposure stacks"
    inputs = (Port("exposures", "any"),)
    input_help = {"exposures": "Ordered batch of 2–16 uint8 gray, BGR or RGB exposures."}
    parameter_ranges = {"reference_index": NumericRange(0, 15, whole=True)}
    payload_reference = "exposures"
    process_inputs_readonly = True

    def stack(self, group):
        if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 16:
            raise ValidationError("Connect an ordered batch of 2–16 exposure images")
        reference = int(self.parameters["reference_index"])
        if reference >= len(group.items):
            raise ValidationError(
                "Reference index is outside the exposure batch", "reference_index"
            )
        if any(
            not isinstance(entry.value, np.ndarray) or entry.value.ndim not in (2, 3)
            for entry in group.items
        ):
            raise ValidationError("Each exposure must be a two-dimensional image")
        originals = aligned_observations(group.items)
        if min(originals[0].shape[:2]) < 2:
            raise ValidationError("Each exposure must be at least 2 × 2 pixels")
        if image_format(originals[0]) not in ("gray", "bgr", "rgb"):
            raise ValidationError("Use matching grayscale, BGR or RGB channel layouts")
        frames = [image_value(image, (np.uint8,)) for image in originals]
        if sum(image.size for image in frames) > 24_000_000:
            raise ValidationError("Exposure batch exceeds 24 million channel samples")
        return originals, frames, reference

    def image_context(self, entry, image):
        source = metadata(entry)
        size = [image.shape[1], image.shape[0]]
        batch = self.input_payloads["exposures"]
        return {
            **{k: value for k, value in batch.items() if k not in ("spatial", "pyramid")},
            **{k: source[k] for k in ("spatial", "pyramid") if k in source},
            "spatial": source.get("spatial", frame(size)),
        }


class AlignExposures(_ExposureStack):
    type_id, title = "cv.align_exposures", "Align exposures (MTB)"
    outputs = (Port("aligned", "any"), Port("image"), Port("info", "data"))
    defaults = {
        "reference_index": 0, "max_bits": 4, "exclude_range": 4, "border_mode": "crop",
    }
    parameter_choices = {"border_mode": ("crop", "zero_pad")}
    parameter_ranges = {
        **_ExposureStack.parameter_ranges,
        "max_bits": NumericRange(1, 10, whole=True),
        "exclude_range": NumericRange(0, 127, whole=True),
    }
    parameter_help = {
        "reference_index": "Zero-based exposure that defines the output grid.",
        "max_bits": "Shift search levels. Up to 2^levels − 1 pixels per axis, limited by size.",
        "exclude_range": "Ignore brightness values this close to the median when matching.",
        "border_mode": "Crop to the area shared by all exposures, or zero-pad the original canvas.",
    }

    def process(self, inputs):
        group = inputs["exposures"]
        originals, frames, reference = self.stack(group)
        height, width = frames[0].shape[:2]
        bits = min(int(self.parameters["max_bits"]), min(width, height).bit_length() - 1)
        aligner = cv2.createAlignMTB(bits, int(self.parameters["exclude_range"]), False)
        grays = [gray_image(image, (np.uint8,)) for image in originals]
        for index, gray in enumerate(grays):
            _, valid = aligner.computeBitmaps(gray)
            if not np.any(valid):
                raise ValidationError(f"Exposure {index} has no usable luminance variation")

        shifts = [
            (0, 0) if i == reference else aligner.calculateShift(grays[reference], gray)
            for i, gray in enumerate(grays)
        ]
        dx, dy = zip(*shifts, strict=True)
        left, top = max(dx), max(dy)
        right, bottom = width + min(dx), height + min(dy)
        if min(right - left, bottom - top) < 2:
            raise ValidationError("The estimated shifts leave too little common image area")
        overlap = [left, top, right - left, bottom - top]
        if self.parameters["border_mode"] == "zero_pad":
            left, top, right, bottom = 0, 0, width, height
        size = [right - left, bottom - top]
        crop_to_reference = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]])
        reference_context = self.image_context(group.items[reference], originals[reference])
        context = moved_payload(reference_context, crop_to_reference, size)
        encoding = image_format(originals[reference])
        packets = []
        for entry, image, shift in zip(group.items, frames, shifts, strict=True):
            shifted = aligner.shiftMat(image, shift) if shift != (0, 0) else image
            value = ImageArray(shifted[top:bottom, left:right].copy(), encoding)
            payload = metadata(entry)
            for key in ("spatial", "pyramid"):
                payload.pop(key, None)
                if key in context:
                    payload[key] = context[key]
            packets.append(ValuePacket(value, payload))
        info = {
            "reference_index": reference,
            "effective_max_bits": bits,
            "shifts": [
                {"label": label, "dx": x, "dy": y}
                for label, (x, y) in zip(group.labels, shifts, strict=True)
            ],
            "overlap_xywh": overlap,
            "output_xywh": [left, top, *size],
        }
        return Result(
            {
                "aligned": FlowGroup(tuple(packets), group.labels),
                "image": packets[reference].value,
                "info": info,
            },
            payloads={"image": context, "info": context},
            message=f"Aligned {len(packets)} exposures · {size[0]} × {size[1]}",
        )


class FuseExposures(_ExposureStack):
    type_id, title = "cv.fuse_exposures", "Fuse exposures (Mertens)"
    inputs = (*_ExposureStack.inputs, *SELECTION_INPUTS[1:])
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "reference_index": 0, "contrast_weight": 1.0, "saturation_weight": 1.0,
        "exposure_weight": 0.0, "output_depth": "uint8",
    }
    parameter_choices = {"output_depth": ("uint8", "float32")}
    parameter_ranges = {
        **_ExposureStack.parameter_ranges,
        "contrast_weight": NumericRange(0, 10),
        "saturation_weight": NumericRange(0, 10),
        "exposure_weight": NumericRange(0, 10),
    }
    parameter_help = {
        "reference_index": "Zero-based source for the output grid and pixels outside a selection.",
        "contrast_weight": "Favor detail where local contrast is higher.",
        "saturation_weight": "Favor more saturated pixels in color images.",
        "exposure_weight": "Favor pixels near midrange intensity.",
        "output_depth": "Clip and scale to uint8, or preserve the native float32 samples.",
    }
    input_help = {
        **_ExposureStack.input_help,
        "exposures": "Ordered batch of 2–16 aligned uint8 gray, BGR or RGB exposures.",
        "mask": "Optional mask on the reference grid. Outside pixels come from the reference.",
        "region": "Optional output selection on the reference grid.",
    }

    def process(self, inputs):
        group = inputs["exposures"]
        originals, frames, reference = self.stack(group)
        source = originals[reference]
        context = self.image_context(group.items[reference], source)
        mask = selection_mask(inputs, image_with_payload(source, context))
        encoding = image_format(source)
        # MergeMertens computes its contrast term in RGB order.
        native = frames
        if encoding == "bgr":
            native = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in frames]
        p = self.parameters
        merger = cv2.createMergeMertens(
            float(p["contrast_weight"]), float(p["saturation_weight"]), float(p["exposure_weight"])
        )
        output = merger.process(native)
        if output is None or output.shape != source.shape or not np.isfinite(output).all():
            raise ValidationError("OpenCV returned an invalid exposure fusion image")
        if encoding == "bgr":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        depth = p["output_depth"]
        if depth == "uint8":
            output = convert_depth(output, depth)
        if mask is not None:
            outside = mask == 0
            base = frames[reference] if depth == "uint8" else convert_depth(source, depth)
            output[outside] = base[outside]
        info = {"count": len(frames), "reference_index": reference, "output_depth": depth}
        return Result(
            {"image": ImageArray(output, encoding), "info": info},
            payloads={"image": context, "info": context},
            message=f"Fused {len(frames)} exposures",
        )
