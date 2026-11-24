"""Periodic contour sampling, Fourier descriptors and similarity alignment."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from .collection_tools import _local_collection_payload
from .contour_analysis import read_ordered_points
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .tickets import ticket_key

MODEL = "contour_fourier_descriptor_v1"


def native_function(name):
    function = getattr(getattr(cv2, "ximgproc", None), name, None)
    if function is None:
        raise ValidationError(f"Fourier contours require OpenCV contrib ximgproc.{name}")
    return function


def raster_points(block, inputs, key):
    points, context = read_ordered_points(block, inputs, key, block.parameters[f"{key}_index"])
    value = inputs[key]
    if isinstance(value, dict):
        points = points - np.asarray(value.get("origin", [0, 0]), dtype=float)
        context = _local_collection_payload(value, context)
    return points, context


def periodic_samples(points, count):
    """Clean float32 segments for the pinned native sampler and restore pixel units."""
    origin = points[0]
    extent = float(np.ptp(points, axis=0).max())
    if extent < 1e-12:
        raise ValidationError("A closed contour must have nonzero extent and area")
    local = ((points - origin) / extent).astype(np.float32)
    local = local[np.any(local != np.roll(local, -1, axis=0), axis=1)]
    area = cv2.contourArea(local, oriented=True) if len(local) >= 3 else 0.0
    if not area:
        raise ValidationError("A closed contour must enclose nonzero area")
    if area < 0:
        local = np.concatenate((local[:1], local[:0:-1]), axis=0)
    samples = native_function("contourSampling")(local.reshape(-1, 1, 2), count)
    if samples is None or samples.shape != (count, 1, 2) or not np.isfinite(samples).all():
        raise ValidationError("OpenCV could not sample the closed contour evenly")
    return samples.reshape(count, 2).astype(np.float64) * extent + origin


def ordered_frequencies(count):
    indices = np.arange(count)
    return np.where(indices <= count // 2, indices, indices - count)


def fit_phase(sample, reference, frequencies, harmonics):
    """Find a stable spectral phase even if native root selection misses a minimum."""
    band = (frequencies != 0) & (np.abs(frequencies) <= harmonics)
    modes = frequencies[band]
    cross = np.conj(sample) * reference
    band_cross = cross[band]
    size = 8 * len(sample)
    padded = np.zeros(size, np.complex128)
    padded[modes % size] = band_cross
    peak = float(np.argmax(np.abs(np.fft.fft(padded)))) / size

    def negative_magnitude(phase):
        return -abs(np.sum(band_cross * np.exp(-2j * np.pi * phase * modes)))

    refined = minimize_scalar(
        negative_magnitude, bounds=(peak - 1 / size, peak + 1 / size),
        method="bounded", options={"xatol": 1e-11},
    )
    phase = float(refined.x if refined.success else peak)
    shift = np.exp(-2j * np.pi * phase * frequencies[1:])
    multiplier = np.sum(cross[1:] * shift) / np.sum(np.abs(sample[1:])**2)
    delta = reference[0] - sample[0]
    return np.array([[phase, np.angle(multiplier), abs(multiplier), delta.real, delta.imag]])


def choose_alignment(sample, reference, count, harmonics):
    src = sample.coefficients[:, 0, 0] + 1j * sample.coefficients[:, 0, 1]
    dst = reference.coefficients[:, 0, 0] + 1j * reference.coefficients[:, 0, 1]
    frequencies = ordered_frequencies(count)
    band = (frequencies != 0) & (np.abs(frequencies) <= harmonics)

    def fit_error(parameters):
        phase, angle, scale, _, _ = parameters[0]
        phased = src * (scale * np.exp(1j * (angle + phase * 2 * np.pi * frequencies)))
        return np.abs(phased - dst)

    solver = native_function("createContourFitting")(count, harmonics)
    try:
        native, distance = solver.estimateTransformation(
            sample.coefficients, reference.coefficients, fdContour=True,
        )
    except cv2.error:
        native, distance = None, None
    spectral = fit_phase(src, dst, frequencies, harmonics)
    if (
        native is not None and native.shape == (1, 5) and np.isfinite(native).all()
        and np.isfinite(distance) and 0 <= distance < 10_000 and native[0, 2] > 0
        and np.linalg.norm(fit_error(native)[1:]) <= np.linalg.norm(fit_error(spectral)[1:]) + 1e-7
    ):
        return native, float(distance), "opencv"
    return spectral, float(fit_error(spectral)[band].mean()), "spectral_phase"


@dataclass(frozen=True)
class Spectrum:
    coefficients: np.ndarray
    center: np.ndarray
    radius: float

    @classmethod
    def from_samples(cls, samples, kept=0):
        relative = samples - samples[0]
        local_center = relative.mean(axis=0)
        center = samples[0] + local_center
        centered = relative - local_center
        radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if radius < 1e-12:
            raise ValidationError("The sampled contour has zero RMS radius")
        work = (centered / radius).reshape(-1, 1, 2)
        coefficients = native_function("fourierDescriptor")(work, nbElt=len(samples), nbFD=kept or -1)
        if (
            coefficients is None or coefficients.shape != (kept or len(samples), 1, 2)
            or not np.isfinite(coefficients).all()
        ):
            raise ValidationError("OpenCV returned invalid Fourier coefficients")
        return cls(coefficients, center, radius)

    @classmethod
    def from_record(cls, record, count):
        if record.get("sample_count") != count or record.get("layout") != "full":
            raise ValidationError("Contour alignment requires a full spectrum with matching sample count")
        rows = record.get("coefficients")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValidationError("Invalid Fourier coefficient count")
        try:
            coefficients = np.ascontiguousarray(rows, dtype=np.float64)
            center = np.asarray(record.get("center"), dtype=np.float64)
            radius = float(record.get("rms_radius"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Invalid Fourier descriptor geometry") from exc
        if (
            coefficients.shape != (count, 2) or not np.isfinite(coefficients).all()
            or np.max(np.abs(coefficients)) > 1.00001
            or np.max(np.abs(coefficients[0])) > 1e-8
            or not np.isclose(float(np.sum(coefficients**2)), 1.0, rtol=1e-5, atol=1e-8)
            or center.shape != (2,) or not np.isfinite(center).all()
            or np.abs(center).max() > 2e6 or not 1e-12 <= radius <= 4e6
        ):
            raise ValidationError("Expected a finite, centered unit-RMS Fourier descriptor")
        return cls(coefficients.reshape(count, 1, 2), center, radius)

    def pixel_samples(self):
        values = cv2.dft(self.coefficients, flags=cv2.DFT_INVERSE).reshape(-1, 2)
        return values * self.radius + self.center


class DescribeFourierContour(Block):
    type_id, title = "inspect.fourier_contour", "Describe contour spectrum"
    category = block_type = "Inspection"
    subtype = "Contour matching"
    description = "Sample a closed contour by arc length and describe its normalized Fourier spectrum."
    inputs = (Port("contour", "data"),)
    outputs = (Port("samples", "data"), Port("descriptor", "data"))
    payload_reference = "contour"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {"contour_index": 0, "samples": 128, "coefficients": 0}
    parameter_ranges = {
        "contour_index": NumericRange(0, 99_999, whole=True),
        "samples": NumericRange(16, 1024, whole=True),
        "coefficients": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "contour_index": "Zero-based item in a contour or shape collection.",
        "samples": "Periodic arc-length samples around the full perimeter.",
        "coefficients": "Zero keeps the full spectrum. Otherwise use an even count up to half the samples.",
    }
    output_help = {
        "samples": "Ordered perimeter samples in local image pixels with no repeated closing endpoint.",
        "descriptor": "Centered unit-RMS complex coefficients with their signed frequencies and pixel geometry.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["coefficients"] % 2 or p["coefficients"] > p["samples"] // 2:
            raise ValidationError("Coefficient count must be even and at most half the samples", "coefficients")

    def process(self, inputs):
        points, context = raster_points(self, inputs, "contour")
        count, kept = int(self.parameters["samples"]), int(self.parameters["coefficients"])
        samples = periodic_samples(points, count)
        shape = Spectrum.from_samples(samples, kept)
        indices = (
            [*range(1, kept // 2 + 1), *range(-kept // 2, 0)]
            if kept else ordered_frequencies(count).tolist()
        )
        return Result(
            {
                "samples": samples.tolist(),
                "descriptor": {
                    "model": MODEL, "sample_count": count, "layout": "compact" if kept else "full",
                    "center": shape.center.tolist(), "rms_radius": shape.radius, "frequencies": indices,
                    "coefficients": shape.coefficients.reshape(-1, 2).tolist(),
                },
            },
            payloads={"samples": context, "descriptor": context},
            message=f"{count} samples · {len(indices)} Fourier coefficients",
        )


class FitFourierContours(Block):
    type_id, title = "inspect.fit_fourier_contours", "Align contours (Fourier)"
    category = block_type = "Inspection"
    subtype = "Contour matching"
    description = "Estimate the similarity transform that aligns an ordered sample outline to a reference."
    inputs = (
        Port("reference", "data"), Port("sample", "data"),
        Port("reference_image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("transform", "data", required=False), Port("report", "data"),
        Port("aligned", "data", required=False), Port("image", required=False),
    )
    payload_reference = "sample"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "reference_index": 0, "sample_index": 0, "samples": 128, "harmonics": 16,
        "maximum_normalized_rms": 0.07, "minimum_scale": 0.05, "maximum_scale": 20.0,
    }
    parameter_ranges = {
        "reference_index": NumericRange(0, 99_999, whole=True),
        "sample_index": NumericRange(0, 99_999, whole=True),
        "samples": NumericRange(16, 1024, whole=True),
        "harmonics": NumericRange(1, 128, whole=True),
        "maximum_normalized_rms": NumericRange(0, 10),
        "minimum_scale": NumericRange(1e-4, 1000),
        "maximum_scale": NumericRange(1e-4, 1000),
    }
    parameter_help = {
        "reference_index": "Zero-based item when Reference is a collection.",
        "sample_index": "Zero-based item when Sample is a collection.",
        "samples": "Number of periodic samples or full Fourier coefficients on both contours.",
        "harmonics": "Positive and matching negative harmonics used to find phase. "
        "At most floor(samples/2) minus one.",
        "maximum_normalized_rms": "Maximum pointwise RMS residual divided by the reference RMS radius.",
        "minimum_scale": "Smallest accepted sample-to-reference scale.",
        "maximum_scale": "Largest accepted sample-to-reference scale.",
    }
    input_help = {
        "reference": "Reference outline as ordered points, a collection, or a full contour spectrum.",
        "sample": "Outline to align as ordered points, a collection, or a full contour spectrum.",
        "reference_image": "Optional image in the reference grid for the overlay.",
    }
    output_help = {
        "transform": "Sample-to-reference 3 × 3 similarity matrix, emitted for an accepted fit.",
        "aligned": "Sample contour phase-shifted and transformed into the reference pixel grid.",
        "report": "Native distance, cyclic phase, pose, pointwise error and shape acceptance.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["harmonics"] > p["samples"] // 2 - 1:
            raise ValidationError("Harmonics must be less than half the sample count", "harmonics")
        if float(p["minimum_scale"]) > float(p["maximum_scale"]):
            raise ValidationError("Minimum scale exceeds maximum scale", "minimum_scale", "maximum_scale")

    def _read_spectrum(self, inputs, key, count):
        value = inputs[key]
        if isinstance(value, dict) and value.get("model") == MODEL:
            return Spectrum.from_record(value, count), self.input_payloads[key]
        points, payload = raster_points(self, inputs, key)
        return Spectrum.from_samples(periodic_samples(points, count)), payload

    def process(self, inputs):
        p = self.parameters
        count, harmonics = int(p["samples"]), int(p["harmonics"])
        reference, target_context = self._read_spectrum(inputs, "reference", count)
        sample, source_context = self._read_spectrum(inputs, "sample", count)
        image = inputs.get("reference_image", NO_VALUE)
        image_context = {}
        if image is not NO_VALUE:
            image_context = self.input_payloads["reference_image"]
            require_same_frame([image, ValuePacket(None, target_context)])
            if "ticket" in image_context and "ticket" in target_context:
                if ticket_key(image_context["ticket"]) != ticket_key(target_context["ticket"]):
                    raise ValidationError("Reference contour and preview image belong to different tickets")
            target_context = {**image_context, **target_context}
        output_context = {
            key: target_context[key] for key in ("spatial", "pyramid") if key in target_context
        }
        if "ticket" in source_context:
            output_context["ticket"] = source_context["ticket"]
        image_context = {**image_context, **output_context}
        native, distance, backend = choose_alignment(sample, reference, count, harmonics)
        if native[0, 2] <= 0 or not np.isfinite(native).all():
            return Result(
                {"transform": NO_VALUE, "aligned": NO_VALUE, "image": image,
                 "report": {"status": "no_fit", "samples": count, "harmonics": harmonics}},
                payloads={"image": image_context},
                passed=False, fail_reasons=["fourier_contour_no_fit"],
                message="No usable Fourier contour fit",
            )

        phase, angle, native_scale, _, _ = native[0]
        aligned = native_function("transformFD")(
            sample.coefficients.copy(), native, fdContour=True,
        ).reshape(count, 2) * reference.radius + reference.center
        ref_samples = reference.pixel_samples()
        rms = float(np.sqrt(np.mean(np.sum((aligned - ref_samples)**2, axis=1))))
        if not math.isfinite(rms):
            raise ValidationError("OpenCV returned nonfinite alignment residuals")
        scale = float(native_scale * reference.radius / sample.radius)
        linear = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        translation = reference.center - linear @ sample.center
        matrix = np.eye(3)
        matrix[:2, :2], matrix[:2, 2] = linear, translation
        normalized_rms = rms / reference.radius
        failures = []
        if normalized_rms > float(p["maximum_normalized_rms"]):
            failures.append("fourier_contour_residual_above_limit")
        if not float(p["minimum_scale"]) <= scale <= float(p["maximum_scale"]):
            failures.append("fourier_contour_scale_outside_limits")
        drawings = {}
        if image is not NO_VALUE and self.generate_graphics:
            drawings["contour alignment"] = GraphicDrawing(
                (
                    graphic_object("polyline", ref_samples, (230, 185, 65), closed=True),
                    graphic_object("polyline", aligned, REJECTED if failures else ACCEPTED, closed=True),
                ),
                from_output=True,
            )
        transform = {
            "model": "image_similarity_v1", "source_to_target": matrix.tolist(),
            "source_size": source_context.get("spatial", {}).get("image_size"),
            "target_size": target_context.get("spatial", {}).get("image_size"),
            "source_spatial": source_context.get("spatial"),
            "target_spatial": target_context.get("spatial"),
        }
        return Result(
            {
                "transform": NO_VALUE if failures else transform, "aligned": aligned.tolist(), "image": image,
                "report": {
                    "status": "rejected" if failures else "matched", "samples": count, "harmonics": harmonics,
                    "phase_turns": float(phase % 1), "angle_degrees": math.degrees(angle),
                    "scale": scale, "translation": translation.tolist(), "distance": float(distance),
                    "fit_backend": backend, "rms_pixels": rms, "normalized_rms": normalized_rms,
                },
            },
            payloads={"aligned": output_context, "image": image_context},
            drawings=drawings, passed=not failures, fail_reasons=failures,
            message=f"Contour residual {rms:.3f} px · scale {scale:.4f}",
        )
