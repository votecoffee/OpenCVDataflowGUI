"""Linear and circular cross-correlation over monochrome images."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

MAX_TRANSFORM_PIXELS = 8_388_608


def correlation_signal(image, mask, label, subtract_mean, normalize):
    signal = image.astype(np.float64)
    if mask is not None:
        selected = mask != 0
        if not selected.any():
            raise ValidationError(f"{label} selection contains no pixels")
        signal[~selected] = 0
    if normalize:
        scale = float(np.max(np.abs(signal)))
        if scale:
            signal /= scale
    if subtract_mean:
        if mask is None:
            signal -= np.mean(signal)
        else:
            signal[selected] -= np.mean(signal[selected])
    if not np.isfinite(signal).all():
        raise ValidationError(f"{label} centering overflowed. Reduce the sample range")
    if not np.any(signal):
        raise ValidationError(f"{label} has zero correlation energy")
    if normalize:
        signal /= cv2.norm(signal, cv2.NORM_L2)
    return signal


def linear_lag_indices(sample_size, reference_size, fft_size):
    return np.arange(1 - reference_size, sample_size) % fft_size


class FourierCorrelation(Block):
    type_id, title, category = "inspect.fourier_correlation", "Fourier correlation", "Inspection"
    block_type, subtype = "Inspection", "Registration"
    inputs = (
        Port("image"),
        Port("reference"),
        *SELECTION_INPUTS[1:],
        Port("reference_mask", required=False, required_when_connected=True),
    )
    outputs = (Port("correlation"), Port("shift", "data"), Port("peak", "number"))
    defaults = {"boundary": "linear", "normalization": "energy", "subtract_mean": False}
    parameter_choices = {
        "boundary": ("linear", "circular"),
        "normalization": ("energy", "none"),
    }
    parameter_help = {
        "boundary": "Linear zero-pads to avoid wrapping. Circular compares equal-size images.",
        "normalization": "Energy scales each whole signal to unit L2 norm before correlation.",
        "subtract_mean": "Remove each input's mean over its selected samples before correlation.",
    }
    input_help = {
        "image": "Monochrome sample image in which to locate the reference displacement.",
        "reference": "Monochrome reference. Linear mode accepts a different image size.",
        "reference_mask": "Optional uint8 selection in the reference image frame.",
    }

    def process(self, inputs):
        source, target = inputs["reference"], inputs["image"]
        reference, sample = image_value(source), image_value(target)
        if reference.ndim != 2 or sample.ndim != 2:
            raise ValidationError("Fourier correlation needs two monochrome images")
        linear = self.parameters["boundary"] == "linear"
        rh, rw = reference.shape
        sh, sw = sample.shape
        if linear:
            height = cv2.getOptimalDFTSize(rh + sh - 1)
            width = cv2.getOptimalDFTSize(rw + sw - 1)
        else:
            if reference.shape != sample.shape:
                raise ValidationError("Circular correlation requires equal image dimensions")
            height, width = sample.shape
        if height < 1 or width < 1 or height * width > MAX_TRANSFORM_PIXELS:
            raise ValidationError("Correlation transform exceeds the 8,388,608-pixel limit")

        sample_mask = selection_mask(inputs, target)
        reference_mask = selection_mask(
            {"image": source, "mask": inputs.get("reference_mask")}, source
        )
        normalized = self.parameters["normalization"] == "energy"
        center = self.parameters["subtract_mean"]
        a = correlation_signal(sample, sample_mask, "Sample", center, normalized)
        b = correlation_signal(reference, reference_mask, "Reference", center, normalized)

        a = np.pad(a, ((0, height - sh), (0, width - sw))) if (height, width) != a.shape else a
        b = np.pad(b, ((0, height - rh), (0, width - rw))) if (height, width) != b.shape else b
        a = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
        b = cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)
        product = cv2.mulSpectrums(a, b, 0, conjB=True)
        if not np.isfinite(product).all():
            raise ValidationError("Fourier correlation overflowed. Try energy normalization")
        scores = cv2.idft(product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        if linear:
            x = linear_lag_indices(sw, rw, width)
            y = linear_lag_indices(sh, rh, height)
            scores = scores[np.ix_(y, x)]
            origin = [rw - 1, rh - 1]
        else:
            scores = np.fft.fftshift(scores)
            origin = [width // 2, height // 2]
        if not np.isfinite(scores).all():
            raise ValidationError("Fourier correlation produced a nonfinite lag map")
        low, peak, _, location = cv2.minMaxLoc(scores)
        magnitude = max(abs(low), abs(peak))
        contrast = peak / magnitude - low / magnitude if magnitude else 0.0
        if scores.size > 1 and contrast <= 64 * np.finfo(np.float64).eps:
            raise ValidationError("Correlation is flat across all displacements")
        dx, dy = location[0] - origin[0], location[1] - origin[1]
        graphics = {}
        if self.generate_graphics:
            view = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(view, location, (70, 220, 140), cv2.MARKER_CROSS, 11, 1)
            graphics["lag map"] = view
        return Result(
            {
                "correlation": scores,
                "shift": {"x": dx, "y": dy, "direction": "reference to image", "units": "pixels"},
                "peak": peak,
            },
            graphics=graphics,
            payloads={
                "correlation": {
                    "lag_origin": origin,
                    "boundary": self.parameters["boundary"],
                    "normalization": self.parameters["normalization"],
                }
            },
            message=f"Peak {peak:.5g} at ({dx}, {dy}) pixels",
        )
