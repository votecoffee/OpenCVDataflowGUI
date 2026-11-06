"""Spectral coefficient packets and reversible image transforms."""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, convert_depth
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value

DCT_FLAGS = {"2d": 0, "rows": cv2.DCT_ROWS}
DFT_FLAGS = {"2d": 0, "rows": cv2.DFT_ROWS}
IMAGE_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)


def magnitude_graphics(magnitude):
    display = cv2.normalize(np.log1p(magnitude), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return {"magnitude": display}


def transform_input(image, operation):
    image = image_value(image, IMAGE_DEPTHS)
    if image.ndim != 2 or image.size == 0:
        raise ValidationError(f"{operation} needs a nonempty monochrome image")
    return convert_depth(image, "float32") if image.dtype.kind == "u" else image


def restore_image(output, depth, operation):
    if not np.isfinite(output).all():
        raise ValidationError(f"{operation} produced nonfinite pixels. Check the coefficient range")
    if depth in ("uint8", "uint16"):
        output = convert_depth(output, depth)
    elif output.dtype.name != depth:
        if np.max(np.abs(output)) > np.finfo(depth).max:
            raise ValidationError(f"{operation} pixels exceed the requested floating-point depth")
        output = output.astype(depth)
    return Result({"image": ImageArray(output, "gray")})


def dct_shape(shape, mode):
    height, width = shape
    return height + (height % 2 if mode == "2d" and height > 1 else 0), (
        width + (width % 2 if width > 1 else 0)
    )


@dataclass(frozen=True)
class DCTCoefficients:
    values: np.ndarray
    mode: str
    image_shape: tuple[int, int]
    image_depth: str

    def describe(self):
        return {
            "model": "image_dct_v1",
            "mode": self.mode,
            "image_shape": list(self.image_shape),
            "image_depth": self.image_depth,
            "values": self.values,
        }


class DiscreteCosineTransform(Block):
    type_id, title = "cv.dct", "Discrete cosine transform (DCT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = (Port("image"),)
    outputs = (Port("coefficients", "dct", visual_type="data"),)
    defaults = {"mode": "2d", "odd_size": "pad_even"}
    parameter_choices = {"mode": tuple(DCT_FLAGS), "odd_size": ("pad_even", "reject")}
    parameter_help = {
        "mode": "Transform the entire image or transform each row independently.",
        "odd_size": "Extend odd transformed axes by repeating the last pixel, or reject them.",
    }

    def process(self, inputs):
        image = inputs["image"]
        working = transform_input(image, "DCT")
        mode = self.parameters["mode"]
        height, width = image.shape
        rows, columns = dct_shape(image.shape, mode)
        if (rows, columns) != image.shape and self.parameters["odd_size"] == "reject":
            raise ValidationError("DCT needs even transformed dimensions or Pad even", "odd_size")
        if (rows, columns) != image.shape:
            working = np.pad(working, ((0, rows - height), (0, columns - width)), mode="edge")
        coefficients = cv2.dct(working, flags=DCT_FLAGS[mode])
        if not np.isfinite(coefficients).all():
            raise ValidationError("DCT produced nonfinite coefficients. Check the image range")
        graphics = magnitude_graphics(np.abs(coefficients)) if self.generate_graphics else {}
        return Result(
            {"coefficients": DCTCoefficients(coefficients, mode, image.shape, image.dtype.name)},
            graphics=graphics,
        )


class InverseDiscreteCosineTransform(Block):
    type_id, title = "cv.idct", "Inverse discrete cosine transform (IDCT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = DiscreteCosineTransform.outputs
    outputs = (Port("image"),)

    def process(self, inputs):
        packet = inputs["coefficients"]
        mode, shape, depth = packet.mode, packet.image_shape, packet.image_depth
        coefficients = packet.values
        if (
            not isinstance(mode, str)
            or mode not in DCT_FLAGS
            or depth not in ("uint8", "uint16", "float32", "float64")
            or not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)
            or not isinstance(coefficients, np.ndarray)
            or coefficients.shape != dct_shape(shape, mode)
            or coefficients.dtype not in (np.float32, np.float64)
            or not np.isfinite(coefficients).all()
        ):
            raise ValidationError("Invalid DCT coefficient values or metadata")
        height, width = shape
        output = cv2.idct(coefficients, flags=DCT_FLAGS[mode])[:height, :width]
        return restore_image(output, depth, "IDCT")


@dataclass(frozen=True)
class DFTSpectrum:
    values: np.ndarray
    mode: str
    image_shape: tuple[int, int]
    image_depth: str

    def describe(self):
        return {
            "model": "image_dft_v1",
            "mode": self.mode,
            "image_shape": list(self.image_shape),
            "image_depth": self.image_depth,
            "values": self.values,
        }

    def validated_values(self):
        values = self.values
        shape = self.image_shape
        if (
            not isinstance(self.mode, str)
            or self.mode not in DFT_FLAGS
            or self.image_depth not in ("uint8", "uint16", "float32", "float64")
            or not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)
            or not isinstance(values, np.ndarray)
            or values.ndim != 3
            or values.shape[2] != 2
            or values.shape[0] < shape[0]
            or values.shape[1] < shape[1]
            or (self.mode == "rows" and values.shape[0] != shape[0])
            or values.dtype not in (np.float32, np.float64)
            or not np.isfinite(values).all()
        ):
            raise ValidationError("Invalid Fourier spectrum values or metadata")
        return values

    def preview(self):
        magnitude = np.hypot(self.values[..., 0], self.values[..., 1], dtype=np.float64)
        axes = 1 if self.mode == "rows" else (0, 1)
        return magnitude_graphics(np.fft.fftshift(magnitude, axes=axes))


class HannWindow(Block):
    type_id, title = "cv.hann_window", "Hann window"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = (Port("image"),)
    outputs = (Port("image"), Port("window"), Port("coherent_gain", "number"))
    defaults = {"axes": "2d", "window": "hann", "precision": "auto"}
    parameter_choices = {
        "axes": ("2d", "rows"),
        "window": ("hann", "sqrt_hann"),
        "precision": ("auto", "float32", "float64"),
    }
    parameter_help = {
        "axes": "Taper both dimensions or taper each row independently.",
        "window": "Square-root Hann matches OpenCV's phase-correlation coefficients.",
        "precision": "Auto uses float64 for float64 input and float32 otherwise.",
    }

    def process(self, inputs):
        image = image_value(inputs["image"], IMAGE_DEPTHS)
        if image.ndim != 2 or image.size == 0:
            raise ValidationError("Hann window requires a nonempty monochrome image")
        p = self.parameters
        height, width = image.shape
        rows = p["axes"] == "rows"
        if width < 3 or (not rows and height < 3):
            raise ValidationError("Every tapered axis must have at least three samples")
        precision = p["precision"]
        if precision == "auto":
            precision = "float64" if image.dtype == np.float64 else "float32"
        native_depth = cv2.CV_64F if precision == "float64" else cv2.CV_32F
        window = cv2.createHanningWindow((width, 3 if rows else height), native_depth)
        if p["window"] == "hann":
            np.square(window, out=window)
        if rows:
            window = np.broadcast_to(window[1], image.shape).copy()
        if image.dtype.kind == "u":
            working = image.astype(np.float64)
            working /= np.iinfo(image.dtype).max
            working = working.astype(precision, copy=False)
        else:
            if precision == "float32" and np.max(np.abs(image)) > np.finfo(np.float32).max:
                raise ValidationError("Image samples exceed float32 range. Use float64 precision.")
            working = image.astype(precision, copy=False)
        return Result(
            {
                "image": ImageArray(working * window, "gray"),
                "window": ImageArray(window, "gray"),
                "coherent_gain": float(window.mean(dtype=np.float64)),
            }
        )


class DiscreteFourierTransform(Block):
    type_id, title = "cv.dft", "Discrete Fourier transform (DFT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = DiscreteCosineTransform.inputs
    outputs = (Port("spectrum", "dft", visual_type="data"),)
    defaults = {"mode": "2d", "size": "exact"}
    parameter_choices = {"mode": tuple(DFT_FLAGS), "size": ("exact", "optimal")}
    parameter_help = {
        "mode": "Transform the entire image or transform each row independently.",
        "size": "Optimal zero-pads transformed axes to OpenCV's next efficient size.",
    }

    def process(self, inputs):
        image = inputs["image"]
        working = transform_input(image, "DFT")
        mode = self.parameters["mode"]
        height, width = image.shape
        if self.parameters["size"] == "optimal":
            rows = cv2.getOptimalDFTSize(height) if mode == "2d" else height
            columns = cv2.getOptimalDFTSize(width)
            if rows < height or columns < width:
                raise ValidationError("OpenCV could not choose a padded Fourier transform size")
            if (rows, columns) != image.shape:
                working = np.pad(working, ((0, rows - height), (0, columns - width)))
        values = cv2.dft(working, flags=DFT_FLAGS[mode] | cv2.DFT_COMPLEX_OUTPUT)
        if not np.isfinite(values).all():
            raise ValidationError("DFT produced nonfinite coefficients. Check the image range")
        spectrum = DFTSpectrum(values, mode, image.shape, image.dtype.name)
        return Result(
            {"spectrum": spectrum}, graphics=spectrum.preview() if self.generate_graphics else {}
        )


class FilterFourierSpectrum(Block):
    type_id, title = "cv.filter_fourier_spectrum", "Filter Fourier spectrum"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = outputs = DiscreteFourierTransform.outputs
    defaults = {"mode": "low_pass", "cutoff": 0.1, "upper_cutoff": 0.3}
    parameter_choices = {"mode": ("low_pass", "high_pass", "band_pass", "band_stop")}
    parameter_ranges = {
        "cutoff": NumericRange(0, 2**-0.5),
        "upper_cutoff": NumericRange(0, 2**-0.5),
    }
    parameter_help = {
        "cutoff": "Cutoff in cycles per pixel. For a band, this is the lower edge.",
        "upper_cutoff": "Upper edge in cycles per pixel. Used by the band modes.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["mode"] in ("band_pass", "band_stop") and p["upper_cutoff"] <= p["cutoff"]:
            raise ValidationError("Upper cutoff must exceed cutoff", "cutoff", "upper_cutoff")

    def process(self, inputs):
        spectrum = inputs["spectrum"]
        values = spectrum.validated_values()
        height, width = values.shape[:2]
        x = np.fft.fftfreq(width)
        frequencies = (
            np.abs(x) if spectrum.mode == "rows" else np.hypot(np.fft.fftfreq(height)[:, None], x)
        )
        cutoff = float(self.parameters["cutoff"])
        mode = self.parameters["mode"]
        if mode == "low_pass":
            keep = frequencies <= cutoff
        elif mode == "high_pass":
            keep = frequencies >= cutoff
        else:
            keep = (frequencies >= cutoff) & (frequencies <= float(self.parameters["upper_cutoff"]))
            if mode == "band_stop":
                keep = ~keep
        filtered = replace(spectrum, values=values * keep[..., None])
        return Result(
            {"spectrum": filtered}, graphics=filtered.preview() if self.generate_graphics else {}
        )


class MaskFourierSpectrum(Block):
    type_id, title = "cv.mask_fourier_spectrum", "Mask Fourier spectrum"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs = (*DiscreteFourierTransform.outputs, Port("gain"))
    outputs = DiscreteFourierTransform.outputs
    payload_reference = "spectrum"
    defaults = {"layout": "centered", "symmetry": "require"}
    parameter_choices = {
        "layout": ("centered", "native"),
        "symmetry": ("require", "average", "minimum", "maximum"),
    }
    parameter_help = {
        "layout": "Centered matches the spectrum preview. Native has zero frequency at index 0.",
        "symmetry": "Require matching paired gains, or combine each pair by average, min, or max.",
    }
    input_help = {
        "gain": "Monochrome gain image at the full spectrum size, including any DFT padding.",
    }

    def process(self, inputs):
        spectrum = inputs["spectrum"]
        values = spectrum.validated_values()
        gain = image_value(inputs["gain"], IMAGE_DEPTHS)
        if gain.ndim != 2 or gain.shape != values.shape[:2]:
            raise ValidationError("Gain must be monochrome and match the full spectrum dimensions")
        if gain.dtype.kind == "u":
            gain = convert_depth(gain, "float32")
        elif np.any(gain < 0):
            raise ValidationError("Fourier gains must be nonnegative")

        rows_only = spectrum.mode == "rows"
        if self.parameters["layout"] == "centered":
            gain = np.fft.ifftshift(gain, axes=1 if rows_only else (0, 1))
        height, width = gain.shape
        paired_x = -np.arange(width) % width
        paired = (
            gain[:, paired_x] if rows_only else gain[np.ix_(-np.arange(height) % height, paired_x)]
        )
        mode = self.parameters["symmetry"]
        if mode == "require":
            if not np.array_equal(gain, paired):
                raise ValidationError("Fourier gains must match at paired frequencies", "symmetry")
        elif mode == "average":
            gain = gain * 0.5 + paired * 0.5
        elif mode == "minimum":
            gain = np.minimum(gain, paired)
        else:
            gain = np.maximum(gain, paired)
        with np.errstate(over="ignore", invalid="ignore"):
            filtered_values = values * gain[..., None]
        if not np.isfinite(filtered_values).all():
            raise ValidationError("Fourier gains overflowed the spectrum")
        filtered = replace(spectrum, values=filtered_values)
        return Result(
            {"spectrum": filtered}, graphics=filtered.preview() if self.generate_graphics else {}
        )


class InverseDiscreteFourierTransform(Block):
    type_id, title = "cv.idft", "Inverse discrete Fourier transform (IDFT)"
    category, block_type, subtype = "Image processing", "Image processing", "Spectral"
    inputs, outputs = DiscreteFourierTransform.outputs, InverseDiscreteCosineTransform.outputs
    defaults = {"output_depth": "unchanged"}
    parameter_choices = {"output_depth": ("unchanged", "float32", "float64")}
    parameter_help = {
        "output_depth": "Use floating point to retain signed or out-of-range filtered samples.",
    }

    def process(self, inputs):
        spectrum = inputs["spectrum"]
        values = spectrum.validated_values()
        flags = DFT_FLAGS[spectrum.mode] | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
        height, width = spectrum.image_shape
        output = cv2.idft(values, flags=flags)[:height, :width]
        depth = self.parameters["output_depth"]
        if depth == "unchanged":
            depth = spectrum.image_depth
        return restore_image(output, depth, "IDFT")


SPECTRAL_BLOCKS = (
    DiscreteCosineTransform,
    InverseDiscreteCosineTransform,
    HannWindow,
    DiscreteFourierTransform,
    FilterFourierSpectrum,
    MaskFourierSpectrum,
    InverseDiscreteFourierTransform,
)
