"""OpenCV F-transform smoothing and bounded inpainting."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, image_value
from .selection import selection_mask

_DEPTHS = (np.uint8, np.uint16, np.float32)
_CHANNELS = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}
_TAP_BUDGET = 140_000_000
_INPUTS = (
    Port("image"),
    Port("mask", required=False, required_when_connected=True),
    Port("region", "data", required=False, required_when_connected=True),
)


def _profile(values, field):
    try:
        weights = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Provide a list of fuzzy membership weights", field) from exc
    if (
        weights.ndim != 1
        or not 3 <= weights.size <= 129
        or weights.size % 2 == 0
        or not np.isfinite(weights).all()
        or weights.min() < 0
        or weights.max() > 1
    ):
        raise ValidationError("Use an odd list of 3 to 129 weights in [0, 1]", field)
    center = weights.size // 2
    if (
        weights[0] != 0
        or weights[-1] != 0
        or weights[center] != 1
        or not np.allclose(weights[1:center] + weights[center + 1 : -1], 1, atol=1e-6, rtol=0)
    ):
        raise ValidationError(
            "Use zero endpoints, a unit center and opposing overlaps that sum to one", field
        )
    return weights


def _native():
    api = getattr(cv2, "ft", None)
    if api is None or not hasattr(api, "FT02D_iteration"):
        raise ValidationError("Fuzzy restoration requires the OpenCV contrib fuzzy module")
    return api


def _signal(source):
    image = image_value(source, _DEPTHS)
    encoding = image_format(source)
    channels = image.shape[2] if image.ndim == 3 else 1
    if not image.size or channels != _CHANNELS.get(encoding):
        raise ValidationError("Use a nonempty uint8, uint16 or float32 gray or RGB/BGR image")
    if image.shape[0] * image.shape[1] > 8_400_000:
        raise ValidationError("Fuzzy restoration is limited to 8.4 million input pixels")
    if channels == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
        raise ValidationError("Unassociate premultiplied alpha before fuzzy restoration")
    colors = image[..., :3] if image.ndim == 3 else image
    scale = max(1.0, float(np.abs(colors).max())) if colors.dtype == np.float32 else 1.0
    return image, encoding, np.ascontiguousarray(colors, dtype=np.float32) / scale, scale


def _padding_and_work(signal, radius_x, radius_y):
    h, w = signal.shape[:2]
    bottom, right = (-h) % radius_y, (-w) % radius_x
    channels = signal.shape[2] if signal.ndim == 3 else 1
    updates = ((h + bottom) // radius_y + 1) * ((w + right) // radius_x + 1)
    work = updates * (2 * radius_x + 1) * (2 * radius_y + 1) * channels
    return bottom, right, work


def _extend(image, bottom, right):
    # Complete the last interval of the fuzzy grid on the right and bottom.
    return cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_REPLICATE)


def _publish(image, encoding, signal, scale, selection):
    if signal.shape[:2] != image.shape[:2] or not np.isfinite(signal).all():
        raise ValidationError("Fuzzy restoration returned invalid samples")
    measured = signal
    if image.dtype.kind == "u":
        measured = np.rint(measured).clip(0, np.iinfo(image.dtype).max)
    elif scale != 1:
        peak = float(np.finfo(np.float32).max)
        measured = (measured.astype(np.float64) * scale).clip(-peak, peak)
    output = image.copy()
    target = output[..., :3] if output.ndim == 3 else output
    if selection is None:
        target[:] = measured
    else:
        target[selection] = measured[selection]
    return ImageArray(output, encoding)


class FuzzyFilter(ImageTool):
    type_id, title, subtype = "cv.fuzzy_filter", "Fuzzy image smoothing", "Filters"
    description = "Smooth intensity using the OpenCV F-transform and a separable membership kernel."
    inputs = _INPUTS
    defaults = {
        "kernel": "linear",
        "radius": 3,
        "horizontal_weights": [0.0, 0.5, 1.0, 0.5, 0.0],
        "vertical_weights": [0.0, 0.5, 1.0, 0.5, 0.0],
    }
    parameter_choices = {"kernel": ("linear", "custom")}
    parameter_ranges = {"radius": NumericRange(1, 64, whole=True)}
    parameter_help = {
        "kernel": "Native triangular kernel or separate horizontal and vertical weight profiles.",
        "radius": "Linear half-width in pixels. One is an identity operation.",
        "horizontal_weights": "Custom horizontal membership weights. Zero ends and a unit center.",
        "vertical_weights": "Custom vertical weights. Overlapping weights must sum to one.",
    }
    input_help = {
        "image": "uint8, uint16 or float32 intensity or color. Straight alpha is preserved.",
        "mask": "Replace nonzero output pixels. Nearby unselected samples still contribute.",
        "region": "Limit output writes to this region and any connected mask.",
    }
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["kernel"] == "custom":
            for field in ("horizontal_weights", "vertical_weights"):
                _profile(p[field], field)

    def process(self, inputs):
        source = inputs["image"]
        image, encoding, signal, scale = _signal(source)
        mask = selection_mask(inputs, source)
        selected = None if mask is None else mask != 0
        if selected is not None and not selected.any():
            return Result({"image": ImageArray(image.copy(), encoding)}, message="Empty selection")
        api = _native()
        p = self.parameters
        channels = signal.shape[2] if signal.ndim == 3 else 1
        if p["kernel"] == "linear":
            rx = ry = int(p["radius"])
            kernel = api.createKernel(api.LINEAR, rx, channels)
        else:
            a = _profile(p["horizontal_weights"], "horizontal_weights")
            b = _profile(p["vertical_weights"], "vertical_weights")
            rx, ry = a.size // 2, b.size // 2
            kernel = api.createKernel1(a[None, :], b[:, None], channels)
        if rx == ry == 1:
            return Result({"image": ImageArray(image.copy(), encoding)})
        bottom, right, cost = _padding_and_work(signal, rx, ry)
        if cost > _TAP_BUDGET:
            raise ValidationError("Crop the image to fit the fuzzy smoothing work budget")
        if rx == 1 or ry == 1:
            # FT02D_process visits one unsupported cell outside any radius-one axis.
            # FT02D_iteration skips that cell and leaves the image contributions intact.
            expanded = _extend(signal, bottom, right)
            known = np.ones(expanded.shape[:2], np.uint8)
            _, filtered, _ = api.FT02D_iteration(expanded, kernel, known, False)
        else:
            filtered = api.filter(_extend(signal, bottom, right), kernel)
        h, w = image.shape[:2]
        return Result({"image": _publish(image, encoding, filtered[:h, :w], scale, selected)})


class FuzzyInpaint(ImageTool):
    type_id, title, subtype = "cv.fuzzy_inpaint", "Fuzzy inpainting", "Filters"
    description = "Repair selected pixels with fuzzy transforms and a bounded radius search."
    inputs = _INPUTS
    outputs = (Port("image"), Port("repair_mask", visual_type="mask"), Port("report", "data"))
    defaults = {"algorithm": "iterative", "radius": 2, "maximum_radius": 16}
    parameter_choices = {"algorithm": ("one_step", "multi_step", "iterative")}
    parameter_ranges = {
        "radius": NumericRange(1, 64, whole=True),
        "maximum_radius": NumericRange(1, 64, whole=True),
    }
    parameter_help = {
        "algorithm": "One step uses one linear radius. Multi-step searches for full support. "
        "Iterative also reuses completed estimates.",
        "radius": "Starting radius of the triangular membership kernel in pixels.",
        "maximum_radius": "Inclusive upper radius for the adaptive methods.",
    }
    input_help = {
        "image": "uint8, uint16 or float32 intensity or color. Straight alpha is preserved.",
        "mask": "Nonzero pixels are missing. The rest are measured context.",
        "region": "Select a repair area. With a mask, their intersection is repaired.",
    }
    output_help = {
        "repair_mask": "Synthesized pixels on the source grid.",
        "report": "Algorithm, successful radius and number of passes.",
    }
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["algorithm"] != "one_step" and int(p["radius"]) > int(p["maximum_radius"]):
            raise ValidationError(
                "Maximum radius must be at least the start radius", "radius", "maximum_radius"
            )

    def _reconstruct(self, signal, known):
        p, api = self.parameters, _native()
        method = p["algorithm"]
        first = int(p["radius"])
        last = first if method == "one_step" else int(p["maximum_radius"])
        if last < 2:
            raise ValidationError("Radius one has no neighboring support for inpainting", "radius")
        channels = signal.shape[2] if signal.ndim == 3 else 1
        h, w = signal.shape[:2]
        cost = 0
        passes = 0
        for radius in range(max(first, 2), last + 1):
            bottom, right, work = _padding_and_work(signal, radius, radius)
            cost += work
            if cost > _TAP_BUDGET:
                raise ValidationError(
                    "Crop the image or reduce the adaptive radius budget", "maximum_radius"
                )
            passes += 1
            extended = _extend(signal, bottom, right)
            mask = _extend(known, bottom, right)
            if method == "one_step":
                output = api.inpaint(extended, mask, radius, api.LINEAR, api.ONE_STEP)[:h, :w]
                if not np.isfinite(output).all():
                    raise ValidationError(
                        "One-step support is incomplete. Increase the radius.", "radius"
                    )
                return output, radius, passes
            kernel = api.createKernel(api.LINEAR, radius, channels)
            missing, result, coverage = api.FT02D_iteration(extended, kernel, mask, False)
            output = result[:h, :w]
            if method == "iterative":
                signal[known == 0] = output[known == 0]
                known = np.ascontiguousarray(coverage[:h, :w])
            if missing == 0:
                return signal if method == "iterative" else output, radius, passes
        raise ValidationError(
            "Fuzzy repair still lacks support at the maximum radius", "maximum_radius"
        )

    def process(self, inputs):
        source = inputs["image"]
        image, encoding, signal, scale = _signal(source)
        mask = selection_mask(inputs, source)
        if mask is None:
            raise ValidationError("Connect a mask or region to repair")
        selected = mask != 0
        count = int(np.count_nonzero(selected))
        report = {
            "algorithm": self.parameters["algorithm"],
            "repaired_pixels": count,
            "used_radius": None,
            "passes": 0,
        }
        if count == mask.size:
            raise ValidationError("Fuzzy repair requires known pixels outside the selection")
        output = ImageArray(image.copy(), encoding)
        if count:
            known = (~selected).astype(np.uint8)
            predicted, report["used_radius"], report["passes"] = self._reconstruct(signal, known)
            output = _publish(image, encoding, predicted, scale, selected)
        return Result(
            {
                "image": output,
                "repair_mask": ImageArray(mask, "gray"),
                "report": report,
            },
            message=f"Repaired {count:,} selected pixels",
        )
