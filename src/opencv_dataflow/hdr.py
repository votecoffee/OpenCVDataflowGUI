"""Timed HDR exposure stacks and OpenCV display tone mapping."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .exposure import _ExposureStack
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value

_DEBEVEC_RESPONSE = "opencv_debevec_response_v1"
_ROBERTSON_RESPONSE = "opencv_robertson_response_v1"
_TIME_LIMIT = NumericRange(1e-9, 86400)


def exposure_times(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 16:
        raise ValidationError("Supply 2–16 exposure times in seconds", "exposure_times")
    if any(type(value) not in (int, float) for value in values):
        raise ValidationError("Exposure times must be numbers", "exposure_times")
    times = np.array(
        [_TIME_LIMIT.check(value, "Exposure time", "exposure_times") for value in values],
        dtype=np.float32,
    )
    if np.unique(times).size < 2:
        raise ValidationError("Supply at least two distinct exposure times", "exposure_times")
    return times


class _TimedExposureStack(_ExposureStack):
    inputs = (
        *_ExposureStack.inputs,
        Port("times", "data", required=False, required_when_connected=True),
    )
    defaults = {"reference_index": 0, "exposure_times": []}
    input_help = {
        "exposures": "Ordered batch of 2–16 aligned uint8 gray, BGR or RGB exposures.",
        "times": "Optional list of exposure durations in seconds, in batch order.",
    }
    parameter_help = {
        "reference_index": "Zero-based exposure that defines the output coordinate frame.",
        "exposure_times": "Exposure durations in seconds, in batch order. Times input overrides.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["exposure_times"] != []:
            exposure_times(p["exposure_times"])

    def timed_stack(self, inputs):
        group = inputs["exposures"]
        originals, frames, reference = self.stack(group)
        times = exposure_times(inputs.get("times", self.parameters["exposure_times"]))
        if len(times) != len(frames):
            raise ValidationError(
                "Exposure time count must match the image count", "exposure_times"
            )
        context = self.image_context(group.items[reference], originals[reference])
        return frames, times, image_format(originals[reference]), context


class CalibrateCameraResponse(_TimedExposureStack):
    type_id, title = "cv.calibrate_debevec", "Calibrate camera response (Debevec)"
    outputs = (Port("response", "data"),)
    defaults = {**_TimedExposureStack.defaults, "samples": 70, "smoothness": 10.0, "random": False}
    parameter_ranges = {
        **_TimedExposureStack.parameter_ranges,
        "samples": NumericRange(4, 256, whole=True),
        "smoothness": NumericRange(0.001, 1000),
    }
    parameter_help = {
        **_TimedExposureStack.parameter_help,
        "samples": "Pixel locations used to fit each channel's 256 response levels.",
        "smoothness": "Penalty on curvature of the fitted log response.",
        "random": "Sample random pixel locations. Off uses a repeatable rectangular grid.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["random"]) is not bool:
            raise ValidationError("Random sampling must be boolean", "random")

    def process(self, inputs):
        frames, times, encoding, _ = self.timed_stack(inputs)
        p = self.parameters
        samples = int(p["samples"])
        height, width = frames[0].shape[:2]
        if not p["random"]:
            columns = int(math.sqrt(samples * width / height))
            if not 1 <= columns <= width or not 1 <= samples // columns <= height:
                raise ValidationError("Sample grid does not fit the image dimensions", "samples")
        solver = cv2.createCalibrateDebevec(samples, float(p["smoothness"]), p["random"])
        response = solver.process([np.ascontiguousarray(image) for image in frames], times)
        channels = 1 if encoding == "gray" else 3
        if response is None or response.size != 256 * channels:
            raise ValidationError("OpenCV returned an incomplete camera response")
        values = response.reshape(256, channels)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValidationError("Camera response must contain finite positive values")
        return Result(
            {"response": {
                "model": _DEBEVEC_RESPONSE, "encoding": encoding, "values": values.tolist(),
            }},
            message=f"Estimated 256 response levels for {channels} channel(s)",
        )


def robertson_frames(frames, encoding):
    # OpenCV's calibration and merge implementations use three-channel LUTs.
    if encoding == "gray":
        if sum(image.size for image in frames) > 8_000_000:
            raise ValidationError("Grayscale Robertson batches exceed 8 million pixels")
        return [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in frames]
    return [np.ascontiguousarray(image) for image in frames]


class CalibrateRobertson(_TimedExposureStack):
    type_id, title = "cv.calibrate_robertson", "Calibrate camera response (Robertson)"
    outputs = (Port("response", "data"), Port("info", "data"))
    defaults = {
        **_TimedExposureStack.defaults,
        "max_iterations": 30,
        "convergence_threshold": 0.01,
    }
    parameter_ranges = {
        **_TimedExposureStack.parameter_ranges,
        "max_iterations": NumericRange(1, 100, whole=True),
        "convergence_threshold": NumericRange(1e-8, 1),
    }
    parameter_help = {
        **_TimedExposureStack.parameter_help,
        "max_iterations": "Maximum number of solver iterations.",
        "convergence_threshold": (
            "Native early-stop threshold. With unsampled levels the solver may run all iterations."
        ),
    }
    input_help = {
        **_TimedExposureStack.input_help,
        "exposures": (
            "Aligned uint8 exposures with midtone coverage in each channel, including level 128. "
            "Grayscale batches are limited to 8 million pixels."
        ),
    }

    def process(self, inputs):
        frames, times, encoding, context = self.timed_stack(inputs)
        channels = 1 if encoding == "gray" else 3
        observed = np.zeros((256, channels), dtype=bool)
        for image in frames:
            for channel in range(channels):
                histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
                observed[:, channel] |= histogram.reshape(-1) > 0
        if not observed[128].all():
            raise ValidationError("Robertson calibration needs intensity 128 in every channel")
        if np.any(observed[1:-1].sum(axis=0) < 2):
            raise ValidationError(
                "Robertson calibration needs multiple midrange levels per channel"
            )

        p = self.parameters
        solver = cv2.createCalibrateRobertson(
            int(p["max_iterations"]), float(p["convergence_threshold"])
        )
        response = solver.process(robertson_frames(frames, encoding), times)
        if response is None or response.shape != (256, 1, 3):
            raise ValidationError("OpenCV returned an incomplete Robertson camera response")
        values = response[:, 0, :channels].copy()
        measured = values[observed]
        if not np.isfinite(measured).all() or np.any(measured < 0):
            raise ValidationError("Robertson returned invalid values at observed intensities")
        # Native fits leave unsampled levels undefined. Preserve the measured levels
        # and use the nearest endpoints or linear interpolation for later captures.
        for channel in range(channels):
            missing = np.flatnonzero(~observed[:, channel])
            known = np.flatnonzero(observed[:, channel])
            values[missing, channel] = np.interp(missing, known, values[known, channel])
        info = {"observed_levels": observed.sum(axis=0).tolist()}
        return Result(
            {
                "response": {"model": _ROBERTSON_RESPONSE, "encoding": encoding,
                             "values": values.tolist()},
                "info": info,
            },
            payloads={"response": context, "info": context},
            message=f"Estimated Robertson response for {channels} channel(s)",
        )


def response_table(model, encoding, expected_model):
    channels = 1 if encoding == "gray" else 3
    if model is None:
        linear = np.arange(256, dtype=np.float32) / 128
        if expected_model == _DEBEVEC_RESPONSE:
            linear[0] = linear[1]
        values = np.repeat(linear[:, None], channels, axis=1)
    else:
        if not isinstance(model, dict) or model.get("model") != expected_model:
            method = "Debevec" if expected_model == _DEBEVEC_RESPONSE else "Robertson"
            raise ValidationError(f"Connect a {method} camera response")
        supplied_encoding = model.get("encoding")
        if supplied_encoding not in (("gray",) if channels == 1 else ("bgr", "rgb")):
            raise ValidationError("Camera response channel encoding differs from the image batch")
        data = model.get("values")
        if not isinstance(data, list) or len(data) != 256 or any(
            not isinstance(row, list) or len(row) != channels for row in data
        ):
            raise ValidationError("Camera response needs 256 rows with one value per channel")
        if any(type(value) not in (int, float) for row in data for value in row):
            raise ValidationError("Camera response entries must be numeric")
        with np.errstate(over="ignore", under="ignore"):
            values = np.asarray(data, dtype=np.float32)
        if expected_model == _DEBEVEC_RESPONSE:
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValidationError("Camera response must contain finite positive values")
        elif not np.isfinite(values).all() or np.any(values < 0):
            raise ValidationError("Camera response must contain finite nonnegative values")
        if supplied_encoding != encoding:
            values = values[:, ::-1].copy()
    return values.reshape(256, 1) if channels == 1 else values.reshape(256, 1, channels)


class _MergeHDR(_TimedExposureStack):
    inputs = (
        *_TimedExposureStack.inputs,
        Port("response", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("info", "data"))
    input_help = {
        **_TimedExposureStack.input_help,
        "response": "Optional camera response. If disconnected, assume linear response.",
    }

    def process(self, inputs):
        frames, times, encoding, context = self.timed_stack(inputs)
        table = response_table(inputs.get("response"), encoding, self.response_model)
        image = self.merge_radiance(frames, times, table, encoding)
        if image is None or image.shape != frames[0].shape or not np.isfinite(image).all():
            raise ValidationError("OpenCV returned invalid HDR radiance")
        info = {
            "count": len(frames), "times_seconds": times.tolist(),
            "response": "calibrated" if "response" in inputs else "assumed_linear",
            "radiance_min": float(image.min()), "radiance_max": float(image.max()),
        }
        return Result(
            {"image": ImageArray(image, encoding), "info": info},
            payloads={"image": context, "info": context},
            message=f"Merged {len(frames)} exposures into float32 radiance",
        )


class MergeHDR(_MergeHDR):
    type_id, title = "cv.merge_debevec", "Merge HDR radiance (Debevec)"
    response_model = _DEBEVEC_RESPONSE

    def merge_radiance(self, frames, times, table, _encoding):
        return cv2.createMergeDebevec().process(frames, times, table)


class MergeRobertson(_MergeHDR):
    type_id, title = "cv.merge_robertson", "Merge HDR radiance (Robertson)"
    response_model = _ROBERTSON_RESPONSE
    input_help = {
        **_MergeHDR.input_help,
        "exposures": (
            "Aligned uint8 exposures. Grayscale batches are limited to 8 million pixels."
        ),
        "response": "Optional Robertson response. If disconnected, assume linear response.",
    }

    def merge_radiance(self, frames, times, table, encoding):
        native = robertson_frames(frames, encoding)
        if encoding == "gray":
            table = cv2.merge((table, table, table))
        image = cv2.createMergeRobertson().process(native, times, table)
        if encoding == "gray" and image is not None:
            image = np.ascontiguousarray(image[..., 0])
        return image


class ToneMapHDR(ImageTool):
    type_id, title, subtype = "cv.tonemap_hdr", "Tone map HDR", "Exposure stacks"
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "method": "linear", "gamma": 1.0, "output_depth": "uint8",
        "saturation": 1.0, "bias": 0.85, "intensity": 0.0, "light_adaptation": 1.0,
        "color_adaptation": 0.0, "contrast_scale": 0.7,
    }
    parameter_choices = {
        "method": ("linear", "drago", "reinhard", "mantiuk"),
        "output_depth": ("uint8", "float32"),
    }
    parameter_ranges = {
        "gamma": NumericRange(0.1, 8), "saturation": NumericRange(0.01, 5),
        "bias": NumericRange(0.01, 1), "intensity": NumericRange(-8, 8),
        "light_adaptation": NumericRange(0, 1), "color_adaptation": NumericRange(0, 1),
        "contrast_scale": NumericRange(0.01, 1),
    }
    parameter_help = {
        "method": "Operator for mapping radiance to a display range.",
        "gamma": "Display gamma. Values above 1 brighten midtones.",
        "output_depth": "8-bit output or float32 display samples in [0, 1].",
        "saturation": "Color saturation for Drago and Mantiuk.",
        "bias": "Drago logarithmic compression bias.",
        "intensity": "Reinhard brightness adjustment.",
        "light_adaptation": "Reinhard global-to-local light adaptation.",
        "color_adaptation": "Reinhard channel adaptation. Retains a tiny shared luminance term.",
        "contrast_scale": "Mantiuk contrast response scale.",
    }
    input_help = {"image": "Finite nonnegative float32 radiance in gray, BGR or RGB."}
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.float32,))
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb") or float(image.min()) < 0:
            raise ValidationError("Use nonnegative float32 gray, BGR or RGB radiance")
        if image.size > 24_000_000:
            raise ValidationError("Tone mapping is limited to 24 million channel samples")
        p = self.parameters
        method, gamma = p["method"], float(p["gamma"])
        if encoding == "gray":
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif encoding == "bgr":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        minimum, maximum = float(image.min()), float(image.max())
        black = None
        if method != "linear":
            if np.all(rgb == rgb[0, 0]):
                method = "linear"
            else:
                if min(image.shape[:2]) < 2:
                    raise ValidationError("Nonlinear tone mapping needs images at least 2 × 2")
                black = np.all(rgb == minimum, axis=2)
                if black.any():
                    # Nonlinear operators divide by luminance after their min-max normalization.
                    rgb = (rgb - minimum) / (maximum - minimum)
                    rgb[black, 1] = 1e-6
        if method == "drago":
            mapper = cv2.createTonemapDrago(1.0, float(p["saturation"]), float(p["bias"]))
        elif method == "reinhard":
            # Preserve a shared luminance term for channels that go to exactly zero.
            adaptation = min(float(p["color_adaptation"]), 1 - float(np.finfo(np.float32).eps))
            mapper = cv2.createTonemapReinhard(
                1.0, float(p["intensity"]),
                float(p["light_adaptation"]), adaptation,
            )
        elif method == "mantiuk":
            mapper = cv2.createTonemapMantiuk(
                1.0, float(p["contrast_scale"]), float(p["saturation"])
            )
        else:
            mapper = cv2.createTonemap(1.0)
        output = mapper.process(rgb)
        if output is None or not np.isfinite(output).all():
            raise ValidationError("Tone mapping returned nonfinite display samples")
        if black is not None:
            output[black] = 0
        # Native normalization can put a nominal zero just below zero in float32.
        # Clamp that display roundoff before raising samples to a fractional power.
        np.clip(output, 0, 1, out=output)
        if gamma != 1:
            np.power(output, 1.0 / gamma, out=output)
        if encoding == "gray":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        elif encoding == "bgr":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        depth = p["output_depth"]
        if depth == "uint8":
            output = convert_depth(output, depth)
        return Result(
            {"image": ImageArray(output, encoding), "info": {
                "method": method, "requested_method": p["method"], "output_depth": depth,
            }},
            message=f"Tone mapped with {method}",
        )
