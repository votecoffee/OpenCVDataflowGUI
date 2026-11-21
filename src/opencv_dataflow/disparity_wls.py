"""Guide externally supplied stereo disparity with weighted least squares."""

import cv2
import numpy as np

from .core import Port, Result
from .disparity_inputs import disparity_image, valid_samples
from .geometry3d_common import Tool3D
from .images import ImageArray, opencv_image
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .selection import selection_mask

_NATIVE_SCALE = 16
_MIN_CODE, _MAX_CODE = -32768, 32767


def _encode(samples, valid, scale, sign):
    """Convert to native Q4 units, retaining a separate validity mask."""
    factor = sign * _NATIVE_SCALE / (1.0 if samples.dtype.kind == "f" else scale)
    with np.errstate(over="ignore", invalid="ignore"):
        codes = np.rint(samples[valid].astype(np.float64) * factor)
    if not np.isfinite(codes).all() or np.any((codes < _MIN_CODE) | (codes > _MAX_CODE)):
        raise ValidationError("Disparity exceeds the signed 16-bit range at 16 samples/pixel")
    encoded = np.zeros(samples.shape, np.int16)
    encoded[valid] = codes
    return encoded


def _projected_support(encoded, left_valid, right_valid):
    width = encoded.shape[1]
    # Use the same floor-to-pixel rule as native WLS's left-right lookup.
    match_x = np.arange(width, dtype=np.int32) - (encoded >> 4)
    known = left_valid & (match_x >= 0) & (match_x < width)
    np.clip(match_x, 0, width - 1, out=match_x)
    return known & np.take_along_axis(right_valid, match_x, axis=1)


def _weighted_smooth(guide, encoded, weights, strength, sigma):
    # Match the native WLS normalized smoothing and give missing samples zero weight.
    smooth = cv2.ximgproc.createFastGlobalSmootherFilter(guide, strength, sigma)
    normalizer = smooth.filter(weights)
    numerator = smooth.filter(encoded.astype(np.float32) * weights)
    filtered = np.full(weights.shape, np.nan, np.float32)
    np.divide(numerator, normalizer, out=filtered, where=normalizer > 1e-8)
    return filtered


class FilterExternalDisparity(Tool3D):
    type_id, title, subtype = (
        "3d.filter_disparity_wls",
        "Filter external stereo disparity",
        "Stereo",
    )
    inputs = (
        Port("disparity"),
        Port("guide"),
        Port("right_disparity", required=False, required_when_connected=True),
        Port("valid_mask", required=False, required_when_connected=True),
        Port("right_valid_mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
    )
    outputs = (
        Port("disparity"),
        Port("valid_mask"),
        Port("confidence", required=False),
        Port("right_support", required=False),
        Port("report", "data"),
    )
    defaults = {
        "alignment": "pixel_grid",
        "samples_per_pixel": 16.0,
        "invalid_mode": "nonfinite_only",
        "invalid_code": 0.0,
        "right_samples_per_pixel": 16.0,
        "right_invalid_mode": "nonfinite_only",
        "right_invalid_code": 0.0,
        "right_sign": "right_minus_left",
        "wls_lambda": 8000.0,
        "sigma_color": 1.5,
        "left_right_threshold_px": 1.5,
        "depth_discontinuity_radius": 2,
        "minimum_confidence": 0.0,
    }
    parameter_choices = {
        "alignment": ("pixel_grid", "same_frame"),
        "invalid_mode": ("nonfinite_only", "also_use_code"),
        "right_invalid_mode": ("nonfinite_only", "also_use_code"),
        "right_sign": ("right_minus_left", "left_minus_right"),
    }
    parameter_ranges = {
        "samples_per_pixel": NumericRange(1e-6, 65536),
        "right_samples_per_pixel": NumericRange(1e-6, 65536),
        "invalid_code": NumericRange(),
        "right_invalid_code": NumericRange(),
        "wls_lambda": NumericRange(0, 1e6),
        "sigma_color": NumericRange(0.001, 255),
        "left_right_threshold_px": NumericRange(0, 256),
        "depth_discontinuity_radius": NumericRange(0, 255, whole=True),
        "minimum_confidence": NumericRange(0, 1),
    }
    parameter_help = {
        "alignment": "Assert aligned pixel grids, or check known coordinate frames for each view.",
        "samples_per_pixel": "Left integer samples per pixel. Float inputs are in pixels.",
        "invalid_mode": "Exclude nonfinite samples and optionally an exact finite sample code.",
        "invalid_code": "Missing left sample in original units, before conversion.",
        "right_samples_per_pixel": "Right integer samples per disparity pixel.",
        "right_invalid_mode": "Exclude nonfinite right samples and an optional code.",
        "right_invalid_code": "Missing right sample in original units, before conversion.",
        "right_sign": "Coordinate difference stored by the right-view matcher.",
        "wls_lambda": "Larger values increase guided smoothing.",
        "sigma_color": "Larger values allow smoothing across stronger color edges.",
        "left_right_threshold_px": "Largest consistent difference between left and right matches.",
        "depth_discontinuity_radius": "Confidence penalty radius near disparity discontinuities.",
        "minimum_confidence": "Keep supported left samples meeting this normalized confidence.",
    }
    input_help = {
        "disparity": "Rectified left-view map. Each disparity is x_left minus x_right.",
        "guide": "Aligned 8-bit grayscale or color image of the left view.",
        "right_disparity": "Optional rectified right-view map indexed in right coordinates.",
        "valid_mask": "Optional nonzero valid samples on the left grid.",
        "right_valid_mask": "Optional nonzero valid samples on the right grid.",
        "region": "Optional region restricting which left samples contribute and are emitted.",
    }
    payload_reference = "disparity"
    parallel_safe = process_inputs_readonly = preview_safe = True

    def _samples(self, inputs, *, right=False):
        p = self.parameters
        prefix = "right_" if right else ""
        source = inputs[prefix + "disparity"]
        samples = disparity_image(source)
        mode = p[prefix + "invalid_mode"]
        field = prefix + "invalid_code"
        invalid = float(p[field]) if mode == "also_use_code" else None
        valid = valid_samples(samples, invalid, field)
        selection = {"mask": inputs.get(prefix + "valid_mask")}
        if not right:
            selection["region"] = inputs.get("region")
        mask = selection_mask(selection, source, match_mask_frame=p["alignment"] == "same_frame")
        if mask is not None:
            valid &= mask != 0
        scale = 1.0 if samples.dtype.kind == "f" else float(p[prefix + "samples_per_pixel"])
        return samples, valid, scale

    def process(self, inputs):
        p = self.parameters
        source, left_valid, scale = self._samples(inputs)
        guide = opencv_image(inputs["guide"])
        if guide.dtype != np.uint8 or guide.shape[:2] != source.shape:
            raise ValidationError("Use an aligned 8-bit guide with the same disparity dimensions")
        if min(source.shape) < 2 or max(source.shape) > 16384 or source.size > 8_388_608:
            raise ValidationError("WLS needs at least 2×2 pixels and supports up to 8,388,608")
        if guide.ndim == 3 and guide.shape[2] == 4:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGRA2BGR)
        if p["alignment"] == "same_frame":
            require_same_frame(
                [inputs["disparity"], inputs["guide"]],
                message="Left disparity and guide must share a pixel frame",
            )

        paired = "right_disparity" in inputs and inputs["right_disparity"] is not None
        if not paired and (inputs.get("right_valid_mask") is not None or p["minimum_confidence"]):
            raise ValidationError("Connect right disparity to use right masks or confidence gating")
        left = _encode(source, left_valid, scale, 1)
        strength, sigma = float(p["wls_lambda"]), float(p["sigma_color"])
        confidence = right_support = None
        right_scale = None
        right_count = None
        if paired:
            right_source, right_valid, right_scale = self._samples(inputs, right=True)
            if right_source.shape != source.shape:
                raise ValidationError(
                    "Right disparity must match the rectified left grid dimensions"
                )
            sign = -1 if p["right_sign"] == "left_minus_right" else 1
            right = _encode(right_source, right_valid, right_scale, sign)
            right_count = int(np.count_nonzero(right_valid))
            right_support = _projected_support(left, left_valid, right_valid)
            confidence = np.zeros(source.shape, np.float32)

        filtered = np.full(source.shape, np.nan, np.float32)
        threshold = int(np.floor(float(p["left_right_threshold_px"]) * _NATIVE_SCALE))
        if paired and right_support.any():
            wls = cv2.ximgproc.createDisparityWLSFilterGeneric(True)
            wls.setLambda(strength)
            wls.setSigmaColor(sigma)
            wls.setLRCthresh(threshold + 1)  # Native comparison is strict, in Q4 units.
            wls.setDepthDiscontinuityRadius(int(p["depth_discontinuity_radius"]))
            # OpenCV 5 accepts float samples in the same Q4 units to preserve filter precision.
            filtered = wls.filter(left.astype(np.float32), guide, disparity_map_right=right)
            confidence = wls.getConfidenceMap() / np.float32(255)
            confidence[~right_support | ~np.isfinite(confidence)] = 0
            np.clip(confidence, 0, 1, out=confidence)
            if not left_valid.all() or not right_valid.all():
                filtered = _weighted_smooth(guide, left, confidence, strength, sigma)
        elif not paired and left_valid.any():
            if left_valid.all():
                wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
                wls.setLambda(strength)
                wls.setSigmaColor(sigma)
                filtered = wls.filter(left.astype(np.float32), guide)
            else:
                filtered = _weighted_smooth(
                    guide, left, left_valid.astype(np.float32), strength, sigma
                )

        valid = left_valid & np.isfinite(filtered)
        if paired:
            valid &= (confidence > 0) & (confidence >= float(p["minimum_confidence"]))
        filtered[~valid] = np.nan
        disparity = filtered / np.float32(_NATIVE_SCALE)
        count = int(np.count_nonzero(valid))
        left_count = int(np.count_nonzero(left_valid))
        report = {
            "model": "external_disparity_wls_v1",
            "input_samples_per_pixel": scale,
            "right_input_samples_per_pixel": right_scale,
            "native_samples_per_pixel": _NATIVE_SCALE,
            "left_valid_pixels": left_count,
            "right_valid_pixels": right_count,
            "right_supported_pixels": int(np.count_nonzero(right_support)) if paired else None,
            "output_valid_pixels": count,
            "effective_left_right_threshold_px": threshold / _NATIVE_SCALE if paired else None,
            "minimum_confidence": float(p["minimum_confidence"]) if paired else None,
        }
        graphics = {}
        if self.generate_graphics and count:
            samples = disparity[valid]
            minimum, maximum = float(samples.min()), float(samples.max())
            intensity = np.zeros(source.shape, np.uint8)
            if maximum > minimum:
                intensity[valid] = np.uint8(
                    np.clip((samples - minimum) / (maximum - minimum), 0, 1) * 255
                )
            color = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            graphics["Filtered disparity"] = ImageArray(color, "bgr")
        return Result(
            {
                "disparity": ImageArray(disparity, "gray"),
                "valid_mask": ImageArray(np.uint8(valid) * 255, "gray"),
                "confidence": ImageArray(confidence, "gray") if paired else NO_VALUE,
                "right_support": ImageArray(np.uint8(right_support) * 255, "gray")
                if paired
                else NO_VALUE,
                "report": report,
            },
            graphics=graphics,
            passed=bool(count),
            score=count / left_count if left_count else 0.0,
            fail_reasons=[] if count else ["external_disparity_wls_no_valid_pixels"],
            message=f"Filtered {count}/{left_count} valid left disparity samples",
        )
