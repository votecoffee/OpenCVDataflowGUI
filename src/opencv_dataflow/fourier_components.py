"""Fourier component editing and signed image reconstruction with explicit grids."""

import copy
from dataclasses import replace

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import image_value
from .selection import selection_mask
from .spectral import DFT_FLAGS, IMAGE_DEPTHS, DFTSpectrum, magnitude_graphics

MAX_PIXELS = 2_100_000


def paired_values(values, mode):
    x = -np.arange(values.shape[1]) % values.shape[1]
    paired = (
        values[:, x]
        if mode == "rows"
        else values[np.ix_(-np.arange(values.shape[0]) % values.shape[0], x)]
    )
    paired = paired.copy()
    paired[..., 1] *= -1
    return paired


def component_spectrum(source, parameters):
    values = source.validated_values()
    if values.shape[0] * values.shape[1] > MAX_PIXELS or max(values.shape[:2]) > 16384:
        raise ValidationError("Fourier components are limited to 2.1M bins and 16384 per axis")
    p, native_depth = parameters, values.dtype
    working = values.astype(np.float64)
    paired = paired_values(working, source.mode)
    axes = (1, 2) if source.mode == "rows" else (0, 1, 2)
    peak = np.max(np.abs(working), axis=axes, keepdims=True)
    divisor = np.where(peak > 0, peak, 1)
    error = float(np.max(np.abs(working / divisor - paired / divisor)))
    if p["symmetry"] == "require" and error > 64 * np.finfo(native_depth).eps:
        raise ValidationError(
            "The spectrum is not conjugate symmetric; choose Average explicitly", "symmetry"
        )
    # Preserve equal pairs, including subnormals; average only discrepancies.
    working = np.where(working == paired, working, working * 0.5 + paired * 0.5)
    if p["dc"] == "remove":
        if source.mode == "rows":
            working[:, 0] = 0
        else:
            working[0, 0] = 0
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.hypot(working[..., 0], working[..., 1])
    if not np.isfinite(magnitude).all():
        raise ValidationError("Fourier magnitudes exceed float64. Reduce the coefficient range")
    axes = 1 if source.mode == "rows" else (0, 1)
    peak = magnitude.max(axis=axes, keepdims=True)
    threshold = np.maximum(p["absolute_floor"], peak * p["relative_floor"])
    valid = (magnitude > threshold) & (magnitude > 0)
    phase = np.zeros_like(working)
    np.divide(working, magnitude[..., None], out=phase, where=valid[..., None])
    mode = p["components"]
    if mode == "phase_only":
        modified = phase
    elif mode == "magnitude_only":
        modified = np.stack((magnitude, np.zeros_like(magnitude)), axis=2)
    else:
        modified = working
    precision = native_depth.name if p["precision"] == "auto" else p["precision"]
    if np.max(np.abs(modified)) > np.finfo(precision).max:
        raise ValidationError(
            "Components exceed the requested precision; choose float64", "precision"
        )
    modified = modified.astype(precision)
    angles = np.where(valid, np.arctan2(working[..., 1], working[..., 0]), 0.0)
    info = {
        "model": "fourier_components_v1",
        "components": mode,
        "dc": p["dc"],
        "transform_mode": source.mode,
        "image_size": list(source.image_shape[::-1]),
        "bin_size": list(values.shape[1::-1]),
        "source_depth": source.image_depth,
        "precision": precision,
        "phase_valid_bins": int(valid.sum()),
        "phase_undefined_bins": int(valid.size - valid.sum()),
        "relative_floor": p["relative_floor"],
        "absolute_floor": p["absolute_floor"],
        "symmetry": p["symmetry"],
        "relative_symmetry_error": error,
        "phase_units": "radians",
        "phase_range": [-float(np.pi), float(np.pi)],
    }
    return replace(source, values=modified, image_depth=precision), magnitude, angles, valid, info


def inverse_components(spectrum):
    """Scale coefficients before the native inverse to avoid intermediate overflow."""
    values = spectrum.values.astype(np.float64)
    axes = (1, 2) if spectrum.mode == "rows" else (0, 1, 2)
    peak = np.max(np.abs(values), axis=axes, keepdims=True)
    divisor = np.where(peak > 0, peak, 1)
    inverse = cv2.idft(
        values / divisor, flags=DFT_FLAGS[spectrum.mode] | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )
    with np.errstate(over="ignore", invalid="ignore"):
        inverse *= np.squeeze(peak, axis=2)
    h, w = spectrum.image_shape
    inverse = inverse[:h, :w]
    if not np.isfinite(inverse).all():
        raise ValidationError("Reconstructed samples exceed float64")
    return inverse


def normalized_image(image, mode):
    if mode == "none":
        return image
    peak = float(np.max(np.abs(image)))
    if peak == 0:
        return np.zeros_like(image)
    scaled = image / peak
    if mode == "unit_rms":
        return scaled / np.sqrt(np.mean(scaled * scaled))
    low, high = float(scaled.min()), float(scaled.max())
    return (scaled - low) / (high - low) if high > low else np.zeros_like(image)


def reconstruction_payload(payload, mode, shape, owner):
    output = copy.deepcopy(payload)
    output.pop("alpha_representation", None)
    if mode == "magnitude_only":
        source_frame = output.pop("spatial", None)
        source_pyramid = output.pop("pyramid", None)
        output["spatial"] = frame(list(shape[::-1]), f"{owner}:zero_phase")
        output["spectral_source_grid"] = {"spatial": source_frame, "pyramid": source_pyramid}
    return output


class _ComponentTool(Block):
    category = block_type = "Image processing"
    subtype = "Spectral"
    parallel_safe = process_inputs_readonly = True
    defaults = {
        "components": "phase_only",
        "dc": "keep",
        "relative_floor": 1e-6,
        "absolute_floor": 0.0,
        "symmetry": "require",
        "precision": "auto",
    }
    parameter_choices = {
        "components": ("full", "phase_only", "magnitude_only"),
        "dc": ("keep", "remove"),
        "symmetry": ("require", "average"),
        "precision": ("auto", "float32", "float64"),
    }
    parameter_ranges = {
        "relative_floor": NumericRange(0, 1),
        "absolute_floor": NumericRange(0),
    }
    parameter_help = {
        "components": "Full spectrum, phase with unit magnitudes, or magnitudes with zero phase.",
        "dc": "Keep or remove zero frequency before thresholding, per transformed signal.",
        "relative_floor": "Strict phase floor relative to peak magnitude; per row in Rows mode.",
        "absolute_floor": "Additional strict phase floor in native, unscaled DFT magnitude units.",
        "symmetry": "Require conjugate pairs within native precision, or explicitly average them.",
        "precision": "Auto follows the spectrum; image input uses float64 for uint16/float64.",
    }


class FourierComponents(_ComponentTool):
    type_id, title = "cv.fourier_components", "Fourier magnitude and phase components"
    inputs = (Port("spectrum", "dft", visual_type="data"),)
    outputs = (
        Port("spectrum", "dft", visual_type="data"),
        Port("info", "data"),
        Port("magnitude"),
        Port("phase"),
        Port("valid_phase"),
    )
    payload_reference = "spectrum"
    defaults = {**_ComponentTool.defaults, "layout": "centered"}
    parameter_choices = {**_ComponentTool.parameter_choices, "layout": ("centered", "native")}
    parameter_help = {
        **_ComponentTool.parameter_help,
        "layout": "Magnitude, phase and validity image layout. The typed spectrum stays native.",
    }
    input_help = {
        "spectrum": "Monochrome real-image DFT, including its mode, padding and source grid."
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["spectrum", "info"]
        self.documentation = (
            "<p>Phase-only uses F/|F| where phase exceeds both floors, and zero elsewhere. "
            "Magnitude-only uses |F| + 0i. Full retains both components. DC removal precedes "
            "thresholding. Outputs retain conjugate symmetry for a real inverse.</p>"
            "<p>The typed spectrum requests floating inverse output. Magnitude-only receives a "
            "new spatial frame; it has no source-position mapping. Diagnostic images are "
            "frequency-bin grids, with invalid phase values zero and a separate validity mask.</p>"
            '<p><a href="https://docs.opencv.org/5.0/main_modules/core_array.html">'
            "OpenCV DFT reference</a>.</p>"
        )

    def process(self, inputs):
        spectrum, magnitude, phase, valid, info = component_spectrum(
            inputs["spectrum"], self.parameters
        )
        layout = self.parameters["layout"]
        if layout == "centered":
            axes = 1 if spectrum.mode == "rows" else (0, 1)
            magnitude, phase, valid = (
                np.fft.fftshift(value, axes=axes) for value in (magnitude, phase, valid)
            )
        original = self.input_payloads["spectrum"]
        if original.get("spatial", {}).get("image_size", list(spectrum.image_shape[::-1])) != list(
            spectrum.image_shape[::-1]
        ):
            raise ValidationError("Spectrum source frame must match its original image size")
        payload = reconstruction_payload(
            original, self.parameters["components"], spectrum.image_shape, self.id
        )
        payload["spectral_components"] = {
            "components": self.parameters["components"],
            "dc": self.parameters["dc"],
        }
        frequency = copy.deepcopy(original)
        frequency.pop("pyramid", None)
        frequency["spatial"] = frame(list(magnitude.shape[::-1]), f"{self.id}:frequency")
        frequency["frequency_grid"] = {
            "layout": layout,
            "mode": spectrum.mode,
            "x_units": "cycles/current_pixel",
            "y_units": "source_row" if spectrum.mode == "rows" else "cycles/current_pixel",
            "source_spatial": copy.deepcopy(original.get("spatial")),
            "source_pyramid": copy.deepcopy(original.get("pyramid")),
        }
        info["diagnostic_layout"] = layout
        info["diagnostic_stage"] = "DC-adjusted input spectrum"
        graphics = magnitude_graphics(magnitude) if self.generate_graphics else {}
        return Result(
            {
                "spectrum": spectrum,
                "info": info,
                "magnitude": magnitude,
                "phase": phase,
                "valid_phase": valid.astype(np.uint8) * 255,
            },
            payloads={
                "spectrum": payload,
                "info": payload,
                "magnitude": {**frequency, "spectral_value": {"units": "DFT_magnitude"}},
                "phase": {**frequency, "spectral_value": {"units": "radians", "undefined": 0}},
                "valid_phase": {**frequency, "spectral_value": {"units": "binary_validity"}},
            },
            graphics=graphics,
            message=f"{info['components']} · {info['phase_valid_bins']} reliable phase bins",
        )


class ReconstructFourierImage(_ComponentTool):
    type_id, title = "cv.reconstruct_fourier_image", "Reconstruct Fourier image components"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("info", "data"), Port("source"), Port("preview", required=False))
    payload_reference = "image"
    defaults = {
        **_ComponentTool.defaults,
        "mode": "2d",
        "size": "exact",
        "channels": "independent",
        "normalization": "none",
        "normalization_scope": "per_channel",
    }
    parameter_choices = {
        **_ComponentTool.parameter_choices,
        "mode": tuple(DFT_FLAGS),
        "size": ("exact", "optimal"),
        "channels": ("independent", "grayscale"),
        "normalization": ("none", "unit_rms", "minmax"),
        "normalization_scope": ("per_channel", "shared"),
    }
    parameter_help = {
        **_ComponentTool.parameter_help,
        "mode": "Transform the whole plane or each row independently.",
        "size": "Exact size or efficient zero-padding on transformed axes, followed by cropping.",
        "channels": "Independent color channels or grayscale. Alpha remains on Source only.",
        "normalization": "Signed raw inverse, unit RMS, or range [0,1]. Flat min/max becomes zero.",
        "normalization_scope": "Normalize each color channel separately or share one color scale.",
    }
    input_help = {
        "image": "Gray or RGB(A)/BGR(A), uint8/16/float32/64; Source retains original data.",
        "mask": "Optional uint8 selection: zero other source samples before the global transform.",
        "region": "Source region intersected with the mask. Reconstruction can extend outside it.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "info"]
        self.documentation = (
            "<p>Reconstruct full, phase-only or zero-phase magnitude components per color channel "
            "or in grayscale. Unsigned input maps to [0,1]; floating input retains its values. "
            "Output stays floating and signed unless explicit intensity normalization is chosen. "
            "Color output has three channels; original alpha remains on Source.</p>"
            "<p>Selections zero input samples before the global transform. Native inverse scaling "
            "uses the padded transform length; the result is cropped back. Magnitude-only images "
            "use a new origin-based grid, while phase/full images retain source coordinates.</p>"
        )

    def process(self, inputs):
        original, p = inputs["image"], self.parameters
        raw = image_value(original, IMAGE_DEPTHS)
        encoding = image_format(original)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Convert to grayscale or RGB(A)/BGR(A) before reconstruction")
        expected = 1 if encoding == "gray" else 4 if encoding in ("rgba", "bgra") else 3
        if (1 if raw.ndim == 2 else raw.shape[2]) != expected:
            raise ValidationError("Image encoding does not match its channel layout")
        h, w = raw.shape[:2]
        rows = cv2.getOptimalDFTSize(h) if p["size"] == "optimal" and p["mode"] == "2d" else h
        columns = cv2.getOptimalDFTSize(w) if p["size"] == "optimal" else w
        if rows < h or columns < w or rows * columns > MAX_PIXELS or max(rows, columns) > 16384:
            raise ValidationError(
                "Padded reconstruction is limited to 2.1M pixels and 16384 per axis"
            )
        depth = p["precision"]
        if depth == "auto":
            depth = "float64" if raw.dtype in (np.uint16, np.float64) else "float32"
        if np.max(np.abs(raw)) > np.finfo(depth).max:
            raise ValidationError(
                "Image exceeds the requested precision; choose float64", "precision"
            )
        working = raw.astype(np.float64)
        if raw.dtype.kind == "u":
            working /= np.iinfo(raw.dtype).max
        if working.ndim == 3:
            working = working[..., :3]
            encoding = encoding.removesuffix("a")
            if p["channels"] == "grayscale":
                weights = [0.299, 0.587, 0.114] if encoding == "rgb" else [0.114, 0.587, 0.299]
                working = np.sum(working * weights, axis=2)
                encoding = "gray"
        selection = selection_mask(inputs, original)
        if selection is not None:
            working = (
                np.where((selection != 0)[..., None], working, 0)
                if working.ndim == 3
                else np.where(selection != 0, working, 0)
            )
        planes = [working] if working.ndim == 2 else [working[..., i] for i in range(3)]
        reconstructed, reports = [], []
        for plane in planes:
            padded = np.pad(plane.astype(depth), ((0, rows - h), (0, columns - w)))
            values = cv2.dft(padded, flags=DFT_FLAGS[p["mode"]] | cv2.DFT_COMPLEX_OUTPUT)
            if not np.isfinite(values).all():
                raise ValidationError("DFT overflowed; reduce image range or use float64")
            packet = DFTSpectrum(values, p["mode"], (h, w), raw.dtype.name)
            spectrum, _, _, _, report = component_spectrum(packet, p)
            reconstructed.append(inverse_components(spectrum))
            reports.append(report)
        output = reconstructed[0] if len(reconstructed) == 1 else np.stack(reconstructed, axis=2)
        if output.ndim == 3 and p["normalization_scope"] == "per_channel":
            output = np.stack(
                [normalized_image(output[..., i], p["normalization"]) for i in range(3)], axis=2
            )
        else:
            output = normalized_image(output, p["normalization"])
        if not np.isfinite(output).all() or np.max(np.abs(output)) > np.finfo(depth).max:
            raise ValidationError("Reconstructed image exceeds the requested precision")
        source_payload = self.input_payloads["image"]
        payload = reconstruction_payload(source_payload, p["components"], (h, w), self.id)
        payload["spectral_reconstruction"] = {
            "components": p["components"],
            "normalization": p["normalization"],
            "source_depth": raw.dtype.name,
            "input_scaling": "unsigned_full_scale" if raw.dtype.kind == "u" else "floating_samples",
        }
        info = {
            "model": "fourier_reconstruction_v1",
            "components": p["components"],
            "input_format": image_format(original),
            "output_format": encoding,
            "source_depth": raw.dtype.name,
            "output_depth": depth,
            "alpha_omitted": raw.ndim == 3 and raw.shape[2] == 4,
            "normalization": p["normalization"],
            "normalization_scope": p["normalization_scope"],
            "selected_source_pixels": int(np.count_nonzero(selection))
            if selection is not None
            else h * w,
            "planes": reports,
        }
        preview, drawings = NO_VALUE, {}
        if self.generate_graphics:
            preview = ImageArray(
                np.rint(normalized_image(output, "minmax") * 255).astype(np.uint8), encoding
            )
            label = graphic_object(
                "text", [[5, 16]], (60, 220, 140), text=f"{p['components']} · {p['normalization']}"
            )
            drawings = {"Fourier reconstruction": GraphicDrawing((label,), "image", True)}
        return Result(
            {
                "image": ImageArray(output.astype(depth), encoding),
                "info": info,
                "source": original,
                "preview": preview,
            },
            payloads={
                "image": payload,
                "info": payload,
                "source": source_payload,
                "preview": payload,
            },
            graphics={"reconstruction": preview} if preview is not NO_VALUE else {},
            drawings=drawings,
            message=f"{p['components']} · {len(planes)} planes · {p['normalization']}",
        )
