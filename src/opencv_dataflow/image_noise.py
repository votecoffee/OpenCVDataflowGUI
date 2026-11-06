"""Seeded synthetic images and selected additive noise in stored sample units."""

import math

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import OUTPUT_DEPTHS
from .processing import image_value
from .selection import selection_mask

_SEED = NumericRange(0, 2**32 - 1, whole=True)
_SEED_INPUT = Port("seed", "number", required=False, required_when_connected=True)
_OUTPUTS = (Port("image"), Port("seed", "number"))
_MAX_SAMPLES = 8_388_608


class _SeededDistribution(Block):
    defaults = {
        "distribution": "gaussian",
        "seed": 0,
        "mean": 0.0,
        "standard_deviation": 8.0,
        "minimum": -8.0,
        "maximum": 8.0,
    }
    parameter_choices = {"distribution": ("gaussian", "uniform")}
    parameter_ranges = {
        "seed": _SEED,
        "mean": NumericRange(),
        "standard_deviation": NumericRange(0),
        "minimum": NumericRange(),
        "maximum": NumericRange(),
    }
    parameter_help = {
        "seed": "Repeat the same draw on each activation. A wired Seed takes precedence.",
        "mean": "Gaussian mean in stored sample units.",
        "standard_deviation": "Gaussian spread in stored sample units. Zero gives a constant.",
        "minimum": "Uniform lower bound in stored sample units.",
        "maximum": "Uniform upper bound in stored sample units.",
    }
    input_help = {"seed": "Optional whole-number seed between 0 and 4,294,967,295."}
    outputs = _OUTPUTS

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["distribution"] == "uniform":
            width = float(p["maximum"]) - float(p["minimum"])
            if width < 0 or not math.isfinite(width):
                raise ValidationError(
                    "Uniform bounds must define a finite ordered range", "minimum", "maximum"
                )

    def draw_samples(self, inputs, shape, integer_depth=None):
        p = self.parameters
        seed = _SEED.check(inputs.get("seed", p["seed"]), "Seed", "seed")
        # A private generator keeps the result independent of branch order and other blocks.
        rng = np.random.Generator(np.random.PCG64(seed))
        if p["distribution"] == "gaussian":
            samples = rng.normal(float(p["mean"]), float(p["standard_deviation"]), size=shape)
        elif integer_depth is not None:
            samples = rng.integers(
                math.ceil(float(p["minimum"])),
                math.floor(float(p["maximum"])),
                size=shape,
                dtype=integer_depth,
                endpoint=True,
            )
        else:
            samples = rng.uniform(float(p["minimum"]), float(p["maximum"]), size=shape)
        if not np.isfinite(samples).all():
            raise ValidationError("Random samples exceed the float64 range")
        return samples, seed


class RandomImage(_SeededDistribution):
    type_id, title, category = "input.random_image", "Generate random image", "Input"
    block_type, subtype = "Input", "Synthetic"
    inputs = (_SEED_INPUT,)
    defaults = {
        **_SeededDistribution.defaults,
        "distribution": "uniform",
        "mean": 127.5,
        "standard_deviation": 32.0,
        "minimum": 0.0,
        "maximum": 255.0,
        "width": 320,
        "height": 240,
        "color_format": "gray",
        "output_depth": "uint8",
    }
    parameter_choices = {
        **_SeededDistribution.parameter_choices,
        "color_format": ("gray", "bgr", "rgb"),
        "output_depth": tuple(depth for depth in OUTPUT_DEPTHS if depth != "same"),
    }
    parameter_ranges = {
        **_SeededDistribution.parameter_ranges,
        "width": NumericRange(1, 8192, whole=True),
        "height": NumericRange(1, 8192, whole=True),
    }
    parameter_help = {
        **_SeededDistribution.parameter_help,
        "output_depth": "Integer output rounds and saturates Gaussian samples. "
        "Uniform integers are drawn inclusively between the configured bounds.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        channels = 1 if p["color_format"] == "gray" else 3
        if int(p["width"]) * int(p["height"]) * channels > _MAX_SAMPLES:
            raise ValidationError(
                "Random image is limited to 8,388,608 channel samples",
                "width",
                "height",
                "color_format",
            )
        dtype = np.dtype(p["output_depth"])
        if p["distribution"] == "uniform" and dtype.kind in "iu":
            limits = np.iinfo(dtype)
            low, high = float(p["minimum"]), float(p["maximum"])
            if not limits.min <= low <= high <= limits.max or math.ceil(low) > math.floor(high):
                raise ValidationError(
                    "Uniform bounds must fit the output depth and include an integer",
                    "minimum",
                    "maximum",
                    "output_depth",
                )

    def process(self, inputs):
        p = self.parameters
        encoding = p["color_format"]
        shape = (int(p["height"]), int(p["width"]))
        if encoding != "gray":
            shape = (*shape, 3)
        dtype = np.dtype(p["output_depth"])
        integer_depth = dtype if dtype.kind in "iu" else None
        samples, seed = self.draw_samples(inputs, shape, integer_depth)
        image = ImageArray(
            samples if samples.dtype == dtype else cast_intensity(samples, dtype), encoding
        )
        return Result({"image": image, "seed": seed}, message=f"Random image from seed {seed}")


class AddImageNoise(_SeededDistribution):
    type_id, title = "cv.add_image_noise", "Add image noise"
    category, block_type, subtype = "Image processing", "Image processing", "Image math"
    process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        _SEED_INPUT,
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    defaults = {**_SeededDistribution.defaults, "preserve_alpha": True}
    parameter_help = {
        **_SeededDistribution.parameter_help,
        "preserve_alpha": "Keep the alpha samples of a four-channel image.",
    }
    input_help = {
        **_SeededDistribution.input_help,
        "image": "Grayscale or RGB/BGR image. Output keeps the input depth and encoding.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Image noise requires grayscale or RGB/BGR samples")
        mask = selection_mask(inputs, source)
        selected = None if mask is None else mask != 0
        output = image.copy()
        signal = output
        if output.ndim == 3 and output.shape[2] == 4 and self.parameters["preserve_alpha"]:
            signal = output[..., :3]
        samples = signal if selected is None else signal[selected]
        noise, seed = self.draw_samples(inputs, samples.shape)
        if samples.size:
            perturbed = cv2.add(
                samples.astype(np.float64, copy=False).reshape(-1, 1), noise.reshape(-1, 1)
            ).reshape(samples.shape)
            if not np.isfinite(perturbed).all():
                raise ValidationError("Noisy samples exceed the float64 range")
            stored = cast_intensity(perturbed, image.dtype)
            if selected is None:
                signal[...] = stored
            else:
                signal[selected] = stored
        pixels = image.shape[0] * image.shape[1] if selected is None else int(selected.sum())
        return Result(
            {"image": ImageArray(output, encoding), "seed": seed},
            message=f"Applied noise to {pixels} selected pixels from seed {seed}",
        )
