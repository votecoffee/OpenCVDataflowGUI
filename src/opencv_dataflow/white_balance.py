"""Native gray-world white balance with explicit estimation support."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import selection_mask

_CHANNELS = ("B", "G", "R")
_CHUNK_PIXELS = 1_048_576


def _sample_statistics(image, selection, code, cutoff, support):
    """Count usable pixels and sum channels without overflowing native accumulators."""
    colors = image.reshape(-1, 3)
    flat_mask = selection.reshape(-1) if selection is not None else None
    sums = np.zeros(3, dtype=np.uint64)
    accepted = used = 0
    for start in range(0, len(colors), _CHUNK_PIXELS):
        stop = start + _CHUNK_PIXELS
        pixels = colors[start:stop]
        high = pixels.max(axis=1).astype(np.uint32)
        spread = high - pixels.min(axis=1)
        admitted = spread * code <= high * cutoff
        if flat_mask is not None:
            admitted &= flat_mask[start:stop] != 0
        accepted += int(np.count_nonzero(admitted))
        informative = admitted & (high != 0)
        used += int(np.count_nonzero(informative))
        sums += np.sum(pixels, axis=0, where=informative[:, None], dtype=np.uint64)
        if support is not None:
            support[start:stop] = informative.astype(np.uint8) * 255
    return sums, accepted, used


class GrayWorldWhiteBalance(Block):
    type_id, title = "cv.gray_world_white_balance", "Gray-world white balance"
    category, block_type, subtype = "Calibration", "Calibration", "Color balance"
    description = "Estimate a gray-world balance while excluding overly saturated pixels."
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("gains", "data", required=False),
        Port("report", "data"),
    )
    defaults = {"saturation_threshold": 0.9}
    parameter_ranges = {"saturation_threshold": NumericRange(0, 1)}
    parameter_help = {
        "saturation_threshold": "Maximum (brightest − darkest channel) / brightest channel.",
    }
    input_help = {
        "image": "Unsigned 8- or 16-bit BGR/RGB or BGRA/RGBA. Alpha is preserved.",
        "mask": "Only selected pixels can contribute to the gain estimate.",
        "region": "Restrict estimation to a shape on the input image.",
    }
    output_help = {
        "image": "Balanced image on the original grid, depth and channel encoding.",
        "gains": "B/G/R gains normalized and quantized as in OpenCV's fixed-point application.",
        "report": "Candidate, saturated, black and informative sample counts and channel means.",
    }

    def process(self, inputs):
        source = inputs["image"]
        if source.ndim != 3 or source.shape[2] not in (3, 4):
            raise ValidationError("Gray-world white balance requires three color channels")
        encoding = image_format(source)
        if encoding not in ("bgr", "rgb", "bgra", "rgba") or source.dtype not in (
            np.uint8,
            np.uint16,
        ):
            raise ValidationError(
                "Gray-world white balance requires unsigned 8- or 16-bit RGB/BGR color"
            )
        image = np.ascontiguousarray(opencv_image(source)[..., :3])
        if not image.size or image.size > np.iinfo(np.int32).max:
            raise ValidationError("Image exceeds the size supported by OpenCV gray-world balance")
        threshold = float(self.parameters["saturation_threshold"])
        selection = selection_mask(inputs, source)
        count = image.size // 3
        support = np.empty(count, np.uint8) if self.generate_graphics else None
        max_code = np.iinfo(image.dtype).max
        # OpenCV rounds the binary32 threshold on the source integer-code scale.
        cutoff = int(np.rint(np.float32(threshold) * np.float32(max_code)))
        sums, accepted, used = _sample_statistics(image, selection, max_code, cutoff, support)
        candidates = count if selection is None else int(np.count_nonzero(selection))
        report = {
            "saturation_threshold": threshold,
            "effective_saturation_threshold": cutoff / max_code,
            "candidate_pixels": candidates,
            "used_pixels": used,
            "rejected_saturated_pixels": candidates - accepted,
            "black_pixels": accepted - used,
            "mean_bgr": dict(zip(_CHANNELS, (sums / used).tolist(), strict=True)) if used else None,
        }
        graphics = {}
        if support is not None:
            graphics["Estimation support"] = support.reshape(image.shape[:2])
        if not np.all(sums):
            message = "White balance needs admitted nonzero samples in all three color channels"
            report["status"] = "insufficient_channel_support"
            return Result(
                {"image": NO_VALUE, "gains": NO_VALUE, "report": report},
                passed=False,
                message=message,
                fail_reasons=[{"code": "gray_world_no_channel_support", "message": message}],
                graphics=graphics,
            )

        xphoto = getattr(cv2, "xphoto", None)
        if not hasattr(xphoto, "createGrayworldWB"):
            raise ValidationError("Gray-world white balance requires OpenCV contrib xphoto")
        inverse = (sums.max() / sums).astype(np.float32)
        normalized = inverse / inverse.max()
        scale = max_code + 1
        gains = (np.rint(normalized * scale) / scale).tolist()

        # The xphoto estimator has no selection input and uses uint32 sums for uint8.
        # Apply equivalent gains explicitly for selections or when those sums can overflow.
        overflow = image.dtype == np.uint8 and sums.max() > np.iinfo(np.uint32).max
        if selection is not None or overflow:
            corrected = cv2.xphoto.applyChannelGains(image, *inverse.tolist())
            report["application"] = (
                "selected_gains" if selection is not None else "wide_accumulator_gains"
            )
        else:
            balance = cv2.xphoto.createGrayworldWB()
            balance.setSaturationThreshold(threshold)
            corrected = balance.balanceWhite(image)
            report["application"] = "native_gray_world"
        if encoding in ("rgb", "rgba"):
            corrected = corrected[..., ::-1]
        if source.shape[2] == 4:
            alpha_preserved = np.asarray(source).copy()
            alpha_preserved[..., :3] = corrected
            corrected = alpha_preserved
        else:
            corrected = np.ascontiguousarray(corrected)
        report["status"] = "ok"
        return Result(
            {
                "image": ImageArray(corrected, encoding),
                "gains": dict(zip(_CHANNELS, gains, strict=True)),
                "report": report,
            },
            message=f"Gray-world balance · {used:,} estimation pixels",
            graphics=graphics,
        )
