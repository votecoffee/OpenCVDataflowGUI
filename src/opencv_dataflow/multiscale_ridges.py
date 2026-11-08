"""Multiscale Sato ridge strength and strongest Gaussian scale."""

from numbers import Real

import numpy as np

from .core import Port, Result
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .selection import RegionalOperation
from .skeleton import SKIMAGE_INSTALL


class MultiscaleRidges(RegionalOperation, ImageTool):
    type_id, title, subtype = "inspect.multiscale_ridges", "Multiscale ridge response", "Filtering"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("scale", required=False))
    region_outputs = ("image", "scale")
    region_background = "zero"
    defaults = {"sigmas": [1, 2, 4], "polarity": "dark", "border": "reflect", "border_value": 0.0}
    parameter_choices = {
        "polarity": ("dark", "bright", "both"),
        "border": ("reflect", "constant", "nearest", "mirror", "wrap"),
    }
    parameter_ranges = {"border_value": NumericRange(-1e12, 1e12)}
    parameter_help = {
        "sigmas": "One to eight distinct Gaussian scales from 1 to 12 pixels. "
        "The scale image reports the strongest response's sigma (zero where response is zero). "
        "Sigma is a filter setting, not a direct width measurement.",
        "polarity": "Detect dark or bright ridges, or keep the stronger of both.",
        "border": "How the filter samples beyond the image edge.",
        "border_value": "Original grayscale intensity outside the frame with constant border mode.",
    }
    input_help = {
        "image": "Grayscale or color uint8, uint16 or float32. Response uses grayscale intensity "
        "units. Requires the morphology install extra.",
        "mask": "Zero both outputs outside the mask while retaining full-image neighborhoods.",
        "region": "Zero both outputs outside the region. Intersected with a wired mask.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        sigmas = p["sigmas"]
        if (
            not isinstance(sigmas, (list, tuple))
            or not 1 <= len(sigmas) <= 8
            or any(
                isinstance(sigma, bool) or not isinstance(sigma, Real) or not 1 <= sigma <= 12
                for sigma in sigmas
            )
            or len(set(sigmas)) != len(sigmas)
        ):
            raise ValidationError(
                "Enter one to eight distinct numeric sigmas from 1 to 12 pixels", "sigmas"
            )

    def process(self, inputs):
        try:
            from skimage.filters import sato
        except ImportError as exc:
            raise RuntimeError(SKIMAGE_INSTALL) from exc
        source = inputs["image"]
        p = self.parameters
        sigmas = sorted(p["sigmas"])
        polarity = p["polarity"]
        signs = (True, False) if polarity == "both" else (polarity == "dark",)
        pixels = source.shape[0] * source.shape[1]
        if not 0 < pixels <= 2_100_000 or pixels * len(sigmas) * len(signs) > 16_800_000:
            raise ValidationError(
                "Use up to 2.1 million pixels and 16.8 million pixel-scale-polarity evaluations"
            )
        gray = gray_image(source).astype(np.float32, copy=False)
        if not np.isfinite(gray).all():
            raise ValidationError("Ridge filtering requires finite grayscale intensities")
        strength = np.zeros_like(gray)
        scale = np.zeros_like(gray)
        border_value = float(p["border_value"])
        for sigma in sigmas:
            for dark in signs:
                # Sato negates bright-mode pixels internally. Match the padding to that sign.
                response = sato(
                    gray, sigmas=(sigma,), black_ridges=dark, mode=p["border"],
                    cval=border_value if dark else -border_value,
                )
                if not np.isfinite(response).all():
                    raise ValidationError("Nonfinite ridge response. Reduce the intensity range.")
                stronger = response > strength
                strength[stronger] = response[stronger]
                scale[stronger] = sigma
        return Result(
            {"image": strength, "scale": scale},
            message=f"{polarity.title()} ridges across {len(sigmas)} scales",
        )
