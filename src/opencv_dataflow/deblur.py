"""Wiener deconvolution with a centered point-spread function and explicit boundaries."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, RegionalOperation


def motion_psf(length, angle):
    """Integrate a centered line over the bilinear pixel basis."""
    theta = math.radians(angle % 180)
    extent = np.round((length / 2) * np.array([math.cos(theta), math.sin(theta)]), 12)
    half_x, half_y = np.ceil(np.abs(extent)).astype(int)
    width, height = 2 * half_x + 1, 2 * half_y + 1
    samples = max(1, math.ceil(16 * length))
    t = 2 * (np.arange(samples) + 0.5) / samples - 1
    x, y = (t[:, None] * extent + [half_x, half_y]).T
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - left, y - top
    kernel = np.zeros((height, width), dtype=np.float64)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            np.add.at(
                kernel,
                (np.minimum(top + dy, height - 1), np.minimum(left + dx, width - 1)),
                wx * wy,
            )
    return kernel / kernel.sum()


def defocus_psf(radius):
    """Approximate circular aperture coverage by sampling on a fine grid."""
    side = 2 * math.ceil(radius) + 1
    scale = 9
    fine = np.zeros((side * scale, side * scale), dtype=np.uint8)
    center = fine.shape[0] // 2
    cv2.circle(fine, (center, center), round(radius * scale), 1, thickness=-1)
    coverage = fine.reshape(side, scale, side, scale).mean(axis=(1, 3))
    return coverage / coverage.sum()


class WienerDeblur(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.wiener_deblur", "Wiener deblur", "Filters"
    inputs = (*SELECTION_INPUTS, Port("psf", "data", required=False, required_when_connected=True))
    outputs = (Port("image"), Port("psf", "data"))
    defaults = {
        "model": "gaussian",
        "kernel_width": 9,
        "kernel_height": 9,
        "sigma_x": 2.0,
        "sigma_y": 2.0,
        "motion_length": 8.0,
        "motion_angle": 0.0,
        "defocus_radius": 3.0,
        "custom_psf": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "noise_to_signal": 0.002,
        "boundary": "reflect_101",
        "output_depth": "same",
    }
    parameter_choices = {
        "model": ("gaussian", "motion", "defocus", "custom"),
        "boundary": ("reflect_101", "circular"),
        "output_depth": ("same", "float32", "float64"),
    }
    parameter_ranges = {
        "kernel_width": NumericRange(1, 127, whole=True, odd=True),
        "kernel_height": NumericRange(1, 127, whole=True, odd=True),
        "sigma_x": NumericRange(0.01, 100),
        "sigma_y": NumericRange(0.01, 100),
        "motion_length": NumericRange(0, 126),
        "motion_angle": NumericRange(),
        "defocus_radius": NumericRange(0, 63),
        "noise_to_signal": NumericRange(1e-12, 1e3),
    }
    parameter_help = {
        "model": "Model of the blur. A wired PSF overrides this setting.",
        "custom_psf": "Odd-sized, nonnegative convolution kernel with the origin at its center.",
        "sigma_x": "Gaussian blur standard deviation along X, in pixels.",
        "sigma_y": "Gaussian blur standard deviation along Y, in pixels.",
        "motion_length": "Length of the centered motion path, in pixels.",
        "motion_angle": "Degrees clockwise from horizontal.",
        "defocus_radius": "Radius of the circular blur disk, in pixels.",
        "noise_to_signal": "Noise-to-signal power ratio. Increase to suppress noise amplification.",
        "boundary": "Reflect before deblurring or treat opposite edges as adjacent.",
        "output_depth": "Floating output preserves signed values in the original sample units.",
    }
    input_help = {"psf": "Optional centered point-spread function as a numeric table."}

    def point_spread(self, inputs):
        p = self.parameters
        model = "custom" if "psf" in inputs else p["model"]
        if model == "motion":
            return motion_psf(float(p["motion_length"]), float(p["motion_angle"]))
        if model == "defocus":
            return defocus_psf(float(p["defocus_radius"]))
        if model == "gaussian":
            x = cv2.getGaussianKernel(int(p["kernel_width"]), float(p["sigma_x"]), cv2.CV_64F)
            y = cv2.getGaussianKernel(int(p["kernel_height"]), float(p["sigma_y"]), cv2.CV_64F)
            return y @ x.T
        message = "PSF must be an odd-sized nonnegative matrix with positive weight"
        try:
            kernel = np.asarray(inputs.get("psf", p["custom_psf"]), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(message, "custom_psf") from exc
        if (
            kernel.ndim != 2
            or any(size % 2 == 0 or size > 127 for size in kernel.shape)
            or not np.isfinite(kernel).all()
            or np.min(kernel) < 0
            or np.max(kernel) <= 0
        ):
            raise ValidationError(message, "custom_psf")
        scaled = kernel / kernel.max()
        return scaled / scaled.sum()

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if not image.size or encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
            raise ValidationError("Wiener deblur requires a nonempty grayscale or RGB/BGR image")
        kernel = self.point_spread(inputs)
        height, width = image.shape[:2]
        kh, kw = kernel.shape
        p = self.parameters
        py, px = (kh // 2, kw // 2) if p["boundary"] == "reflect_101" else (0, 0)
        shape = (height + 2 * py, width + 2 * px)
        if shape[0] < kh or shape[1] < kw:
            raise ValidationError("The PSF must fit inside the working image")
        if shape[0] * shape[1] > 4_000_000:
            raise ValidationError("Wiener deblur is limited to 4 million working pixels")

        impulse = np.zeros(shape, dtype=np.float64)
        rows = (np.arange(kh) - kh // 2) % shape[0]
        columns = (np.arange(kw) - kw // 2) % shape[1]
        impulse[np.ix_(rows, columns)] = kernel
        transfer = cv2.dft(impulse, flags=cv2.DFT_COMPLEX_OUTPUT)
        power = np.square(transfer).sum(axis=2)
        transfer /= (power + float(p["noise_to_signal"]))[..., None]

        planes = [image] if image.ndim == 2 else [image[..., i] for i in range(3)]
        restored = []
        for plane in planes:
            padded = cv2.copyMakeBorder(
                plane.astype(np.float64), py, py, px, px, cv2.BORDER_REFLECT_101
            )
            spectrum = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
            estimate = cv2.mulSpectrums(spectrum, transfer, 0, conjB=True)
            plane = cv2.idft(estimate, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
            restored.append(plane[py : py + height, px : px + width])
        if encoding in ("rgba", "bgra"):
            restored.append(image[..., 3])
        output = restored[0] if image.ndim == 2 else np.stack(restored, axis=-1)
        depth = image.dtype if p["output_depth"] == "same" else np.dtype(p["output_depth"])
        if not np.isfinite(output).all():
            raise ValidationError("Wiener deblur produced nonfinite samples")
        if depth.kind in "iu":
            limits = np.iinfo(depth)
            output = np.rint(output).clip(limits.min, limits.max)
        elif np.max(np.abs(output)) > np.finfo(depth).max:
            raise ValidationError("Restored samples exceed the requested depth")
        return Result(
            {"image": ImageArray(output.astype(depth), encoding), "psf": kernel.tolist()},
            payloads={"psf": {}},
        )
