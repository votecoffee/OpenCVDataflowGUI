"""Relative image transfer gain/phase with explicit excited-frequency support."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import HISTOGRAM_CHANNELS
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import image_value
from .selection import selection_mask
from .tickets import ticket_key

_MAX_PIXELS = 2_100_000
_EPS = np.finfo(np.float64).eps
_MAX_FREQUENCY = float(np.sqrt(0.5))


def signal_plane(image, channel, unsigned_scale):
    pixels = image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
    encoding = image_format(image)
    if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
        raise ValidationError("Use intensity gray/RGB/BGR(A), rather than HSV/Lab or depth")
    values = pixels.astype(np.float64)
    if unsigned_scale == "unit_range" and pixels.dtype.kind == "u":
        values /= np.iinfo(pixels.dtype).max
    if channel == "gray":
        if values.ndim == 2:
            return values
        indices = HISTOGRAM_CHANNELS[encoding]
        return sum(
            values[..., indices.index(c)] * w for c, w in (("R", 0.299), ("G", 0.587), ("B", 0.114))
        )
    channels = HISTOGRAM_CHANNELS[encoding]
    if channel not in channels:
        raise ValidationError("Selected channel is absent from an input image", "channel")
    return values[..., channels.index(channel)]


def frequency_domain(shape, p):
    fy, fx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    radius = np.hypot(fy[:, None], fx[None, :])
    eligible = (radius >= p["minimum_frequency"]) & (radius <= p["maximum_frequency"])
    if p["direction"] != "all":
        angle = {"horizontal": 0, "vertical": 90}.get(p["direction"], p["angle_degrees"])
        orientation = np.degrees(np.arctan2(fy[:, None], fx[None, :]))
        difference = np.abs((orientation - angle + 90) % 180 - 90)
        eligible &= difference <= p["angular_tolerance_degrees"]
    eligible[0, 0] = bool(p["include_dc"] and p["minimum_frequency"] == 0)
    return fx, fy, radius, eligible


def weighted_spectrum(plane, aperture, scale, center):
    values = np.zeros_like(plane)
    np.divide(plane, scale, out=values, where=aperture > 0)
    weight = float(aperture.sum())
    numerical_floor = 64 * _EPS * float(np.sum(np.abs(values) * aperture)) / weight
    if center:
        values -= float(np.sum(values * aperture)) / weight
    values *= aperture
    coefficients = cv2.dft(values, flags=cv2.DFT_COMPLEX_OUTPUT) / weight
    if not np.isfinite(coefficients).all():
        raise ValidationError("Image transform exceeded the working range")
    return coefficients, numerical_floor


def stable_mean(values):
    if not len(values):
        return None
    peak = float(np.max(np.abs(values)))
    return float(np.mean(values / peak)) * peak if peak else 0.0


def gain_in_range(value, low, high):
    lower = 128 * _EPS * max(abs(value), abs(low), np.finfo(float).tiny)
    upper = 128 * _EPS * max(abs(value), abs(high), np.finfo(float).tiny)
    return value >= low - lower and value <= high + upper


def transfer(sample, reference, aperture, eligible, p):
    selected = aperture > 0
    scale = max(float(np.max(np.abs(sample[selected]))), float(np.max(np.abs(reference[selected]))))
    scale = scale or 1.0
    s, s_floor = weighted_spectrum(sample, aperture, scale, p["subtract_mean"])
    r, r_floor = weighted_spectrum(reference, aperture, scale, p["subtract_mean"])
    r_magnitude = np.hypot(r[..., 0], r[..., 1])
    s_magnitude = np.hypot(s[..., 0], s[..., 1])
    with np.errstate(over="ignore", divide="ignore"):
        absolute = float(np.float64(p["absolute_floor"]) / scale)
    threshold = max(absolute, float(r_magnitude.max()) * p["relative_floor"], r_floor)
    supported = eligible & (r_magnitude > threshold) & (r_magnitude > 0)
    phase_valid = supported & (
        s_magnitude > max(float(s_magnitude.max()) * p["phase_relative_floor"], s_floor)
    )
    # Normalize each accepted denominator to unit magnitude before native
    # division, preventing its epsilon term from biasing weak reference bins.
    numerator, denominator = np.zeros_like(s), np.zeros_like(r)
    denominator[..., 0] = 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(s, r_magnitude[..., None], out=numerator, where=supported[..., None])
        np.divide(r, r_magnitude[..., None], out=denominator, where=supported[..., None])
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValidationError("Transfer ratio exceeds float64; raise the reference energy floor")
    ratio = cv2.divSpectrums(numerator, denominator, 0, conjB=False)
    gain = np.hypot(ratio[..., 0], ratio[..., 1])
    phase = np.arctan2(ratio[..., 1], ratio[..., 0])
    if not np.isfinite(gain).all() or not np.isfinite(phase).all():
        raise ValidationError("Native transfer division produced nonfinite values")
    gain[~supported], phase[~phase_valid] = 0, 0
    return (
        gain,
        phase,
        supported,
        phase_valid,
        {
            "working_scale": scale,
            "reference_peak_scaled": float(r_magnitude.max()),
            "reference_floor_scaled": threshold if np.isfinite(threshold) else None,
            "absolute_floor_exceeds_working_range": not np.isfinite(absolute),
            "reference_numerical_floor_scaled": r_floor,
            "sample_numerical_floor_scaled": s_floor,
        },
    )


def radial_profile(radius, gain, phase, supported, phase_valid, p):
    edges = np.linspace(p["minimum_frequency"], p["maximum_frequency"], p["radial_bins"] + 1)
    indices = np.minimum(np.searchsorted(edges, radius, side="right") - 1, p["radial_bins"] - 1)
    size = p["radial_bins"]
    counts = np.bincount(indices[supported], minlength=size)
    gains = gain[supported]
    peak = float(gains.max()) if len(gains) else 0.0
    sums = np.bincount(indices[supported], weights=gains / peak if peak else gains, minlength=size)
    phase_indices = indices[phase_valid]
    phase_counts = np.bincount(phase_indices, minlength=size)
    sine = np.bincount(phase_indices, weights=np.sin(phase[phase_valid]), minlength=size)
    cosine = np.bincount(phase_indices, weights=np.cos(phase[phase_valid]), minlength=size)
    rows = []
    for i in range(p["radial_bins"]):
        circular = None
        if phase_counts[i] and np.hypot(sine[i], cosine[i]) > 64 * _EPS * phase_counts[i]:
            circular = float(np.arctan2(sine[i], cosine[i]))
        rows.append(
            {
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "supported_bins": int(counts[i]),
                "phase_valid_bins": int(phase_counts[i]),
                "mean_gain": float(sums[i] / counts[i]) * peak if counts[i] else None,
                "mean_phase_radians": circular,
            }
        )
    return rows


def response_plot(rows, p):
    image = np.full((320, 640, 3), 24, np.uint8)
    peaks = [row["mean_gain"] for row in rows if row["mean_gain"] is not None]
    ceiling = max(1.0, max(peaks, default=0))
    cv2.putText(
        image,
        "Relative frequency gain (supported bins)",
        (22, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    cv2.line(image, (40, 275), (615, 275), (150, 150, 150), 1)
    cv2.line(image, (40, 50), (40, 275), (150, 150, 150), 1)
    last = None
    for i, row in enumerate(rows):
        if row["mean_gain"] is None:
            last = None
            continue
        point = (
            round(40 + (i + 0.5) * 575 / len(rows)),
            round(275 - row["mean_gain"] / ceiling * 215),
        )
        cv2.circle(image, point, 2, (80, 220, 130), -1)
        if last is not None:
            cv2.line(image, last, point, (80, 220, 130), 1)
        last = point
    text = (
        f"{p['minimum_frequency']:.3g} to {p['maximum_frequency']:.3g} cycles/pixel; "
        f"y max {ceiling:.4g}"
    )
    cv2.putText(image, text, (40, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return image


class MeasureFrequencyResponse(Block):
    type_id, title = "inspect.frequency_response", "Image frequency response"
    category = block_type = "Inspection"
    subtype = "Spectral measurements"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        Port("image"),
        Port("reference"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("reference_mask", required=False, required_when_connected=True),
        Port("reference_region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("gain"),
        Port("phase"),
        Port("supported", visual_type="region"),
        Port("phase_valid", visual_type="region"),
        Port("aperture"),
        Port("metrics", "data"),
        Port("mean_gain", "number"),
        Port("maximum_phase_degrees", "number"),
    )
    defaults = {
        "channel": "gray",
        "unsigned_scale": "raw",
        "reference_alignment": "pixel_grid",
        "ticket_policy": "independent",
        "window": "none",
        "subtract_mean": True,
        "relative_floor": 1e-6,
        "absolute_floor": 0.0,
        "phase_relative_floor": 1e-6,
        "minimum_frequency": 0.0,
        "maximum_frequency": 0.5,
        "direction": "all",
        "angle_degrees": 0.0,
        "angular_tolerance_degrees": 10.0,
        "radial_bins": 64,
        "include_dc": False,
        "minimum_supported_bins": 1,
        "minimum_supported_fraction": 0.0,
        "minimum_mean_gain": 0.0,
        "maximum_mean_gain": 1e6,
        "check_phase": False,
        "maximum_phase_degrees": 180.0,
        "maximum_workspace_mib": 256,
    }
    parameter_choices = {
        "channel": ("gray", "R", "G", "B", "A"),
        "unsigned_scale": ("raw", "unit_range"),
        "reference_alignment": ("pixel_grid", "same_frame"),
        "ticket_policy": ("independent", "same_ticket"),
        "window": ("none", "hann", "sqrt_hann"),
        "direction": ("all", "horizontal", "vertical", "angle"),
    }
    parameter_ranges = {
        "relative_floor": NumericRange(0, 1),
        "absolute_floor": NumericRange(0, 1e100),
        "phase_relative_floor": NumericRange(0, 1),
        "minimum_frequency": NumericRange(0, _MAX_FREQUENCY),
        "maximum_frequency": NumericRange(0, _MAX_FREQUENCY),
        "angle_degrees": NumericRange(-180, 180),
        "angular_tolerance_degrees": NumericRange(0, 90),
        "radial_bins": NumericRange(1, 256, whole=True),
        "minimum_supported_bins": NumericRange(1, _MAX_PIXELS, whole=True),
        "minimum_supported_fraction": NumericRange(0, 1),
        "minimum_mean_gain": NumericRange(0, 1e100),
        "maximum_mean_gain": NumericRange(0, 1e100),
        "maximum_phase_degrees": NumericRange(0, 180),
        "maximum_workspace_mib": NumericRange(1, 256, whole=True),
    }
    parameter_help = {
        "channel": "Gray uses Rec.601 RGB luminance; named channels must exist in both inputs.",
        "unsigned_scale": "Raw uses sample units; Unit range divides unsigned values by depth max.",
        "reference_alignment": "Assert pixel alignment, or require identical known spatial grids.",
        "ticket_policy": "Allow historical reference, or require matching acquisition identities.",
        "window": "Shared none/Hann/square-root Hann aperture; finite windows can bias estimates.",
        "subtract_mean": "Remove each signal's aperture-weighted mean before the transform.",
        "relative_floor": "Reject reference amplitudes at/below this fraction of its full peak.",
        "absolute_floor": "Reference floor in sample units after aperture normalization.",
        "phase_relative_floor": "Phase needs sample amplitude above this fraction of its peak.",
        "minimum_frequency": "Inclusive radial lower frequency in cycles per image pixel.",
        "maximum_frequency": "Inclusive radial upper frequency; corners can extend to sqrt(0.5).",
        "direction": "Antipodal frequency cone; horizontal means varying across image x.",
        "angle_degrees": "Cone orientation with x right/y down, used by Angle.",
        "angular_tolerance_degrees": "Inclusive half-angle of the antipodal frequency cone.",
        "radial_bins": "Equal-width radial summaries; unsupported bins remain undefined.",
        "include_dc": "Include DC at minimum frequency zero; centering can remove it.",
        "minimum_supported_bins": "Minimum excited bins, counting both real conjugate partners.",
        "minimum_supported_fraction": "Excited bins divided by eligible band/direction bins.",
        "minimum_mean_gain": "Inclusive lower bound on unweighted supported mean gain.",
        "maximum_mean_gain": "Inclusive upper bound on supported mean linear gain.",
        "check_phase": "Require phase and limit its maximum absolute wrapped value from zero.",
        "maximum_phase_degrees": "Inclusive phase limit when enabled; shifts affect phase.",
        "maximum_workspace_mib": "Work-buffer estimate; snapshots/previews use additional memory.",
    }
    input_help = {
        "image": "Aligned sample capture. Original type/encoding is preserved on Image.",
        "reference": "Aligned reference with known scene detail; no registration is fitted.",
        "mask": "Optional sample-space uint8 selection; shared support weights both signals.",
        "region": "Optional sample-space region, intersected with the other selections.",
        "reference_mask": "Optional uint8 selection in the reference's own coordinates.",
        "reference_region": "Optional reference-space region; both selections must overlap.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_frequency"] >= p["maximum_frequency"]:
            raise ValidationError(
                "Minimum frequency must be below maximum", "minimum_frequency", "maximum_frequency"
            )
        if p["minimum_mean_gain"] > p["maximum_mean_gain"]:
            raise ValidationError(
                "Minimum mean gain exceeds maximum", "minimum_mean_gain", "maximum_mean_gain"
            )

    def process(self, inputs):
        p = self.parameters
        a, b = inputs["image"], inputs["reference"]
        if a.shape[:2] != b.shape[:2] or min(a.shape[:2]) < 3:
            raise ValidationError("Use equal image grids with at least three pixels per axis")
        h, w = a.shape[:2]
        estimate = h * w * 256 + p["radial_bins"] * 2048
        if (
            h * w > _MAX_PIXELS
            or max(h, w) > 16384
            or estimate > p["maximum_workspace_mib"] * 1024**2
        ):
            raise ValidationError(
                "Frequency response exceeds pixel/workspace limits; crop upstream"
            )
        contexts = [self.input_payloads[name] for name in ("image", "reference")]
        if any(c.get("alpha_representation") == "premultiplied" for c in contexts):
            raise ValidationError("Unassociate premultiplied alpha before response measurement")
        if p["reference_alignment"] == "same_frame":
            grids = [c.get("spatial") for c in contexts]
            if (
                any(g is None or g.get("root_id") is None for g in grids)
                or any(
                    grids[0][key] != grids[1][key] for key in ("root_id", "root_size", "image_size")
                )
                or not np.allclose(grids[0]["to_root"], grids[1]["to_root"], atol=1e-9, rtol=0)
            ):
                raise ValidationError("Same-frame comparison requires matching known pixel grids")
        if p["ticket_policy"] == "same_ticket":
            if any("ticket" not in c for c in contexts) or ticket_key(
                contexts[0]["ticket"]
            ) != ticket_key(contexts[1]["ticket"]):
                raise ValidationError("Frequency response requires the same acquisition ticket")
        sample, reference = (
            signal_plane(value, p["channel"], p["unsigned_scale"]) for value in (a, b)
        )
        if not np.isfinite(sample).all() or not np.isfinite(reference).all():
            raise ValidationError("Selected channel arithmetic exceeded float64")
        selected = np.ones((h, w), bool)
        for mask_inputs, source in (
            (inputs, a),
            (
                {
                    "image": b,
                    "mask": inputs.get("reference_mask"),
                    "region": inputs.get("reference_region"),
                },
                b,
            ),
        ):
            mask = selection_mask(mask_inputs, source)
            if mask is not None:
                selected &= mask != 0
        if not selected.any():
            raise ValidationError("The two selections have no shared pixels")
        aperture = selected.astype(np.float64)
        if p["window"] != "none":
            window = cv2.createHanningWindow((w, h), cv2.CV_64F)
            aperture *= window * window if p["window"] == "hann" else window
        if not aperture.any():
            raise ValidationError("Selection and window have zero effective aperture")
        fx, fy, radius, eligible = frequency_domain((h, w), p)
        gain, phase, supported, phase_valid, numerical = transfer(
            sample, reference, aperture, eligible, p
        )
        count, candidates = int(supported.sum()), int(eligible.sum())
        fraction = count / candidates if candidates else 0.0
        mean = stable_mean(gain[supported])
        maximum_phase = (
            float(np.degrees(np.max(np.abs(phase[phase_valid])))) if phase_valid.any() else None
        )
        reasons = []
        if count < p["minimum_supported_bins"] or fraction < p["minimum_supported_fraction"]:
            reasons.append("frequency_reference_support_insufficient")
        if mean is not None and not gain_in_range(
            mean, p["minimum_mean_gain"], p["maximum_mean_gain"]
        ):
            reasons.append("frequency_gain_out_of_tolerance")
        if p["check_phase"] and (
            maximum_phase is None or maximum_phase > p["maximum_phase_degrees"] + 128 * _EPS * 180
        ):
            reasons.append("frequency_phase_out_of_tolerance")
        metrics = {
            "model": "image_frequency_response_v1",
            "channel": p["channel"],
            "units": "cycles_per_pixel",
            "phase_units": "radians",
            "layout": "fftshift_dc_center",
            "image_size": [w, h],
            "frequency_x": np.fft.fftshift(fx).tolist(),
            "frequency_y": np.fft.fftshift(fy).tolist(),
            "supported_bins": count,
            "eligible_bins": candidates,
            "supported_fraction": fraction,
            "phase_valid_bins": int(phase_valid.sum()),
            "mean_gain": mean,
            "maximum_phase_degrees": maximum_phase,
            "selected_pixels": int(selected.sum()),
            "aperture_weight": float(aperture.sum()),
            "window": p["window"],
            "unsigned_scale": p["unsigned_scale"],
            "reference_alignment": p["reference_alignment"],
            "alignment_evidence": "matching_declared_grid"
            if p["reference_alignment"] == "same_frame"
            else "caller_asserted_pixel_indices",
            "source_contexts": contexts,
            "workspace_estimate_bytes": estimate,
            **numerical,
            "radial": radial_profile(radius, gain, phase, supported, phase_valid, p),
        }
        freq_payload = copy.deepcopy(contexts[0])
        freq_payload["spectral_source_grid"] = {
            key: freq_payload.pop(key, None) for key in ("spatial", "pyramid")
        }
        freq_payload["spatial"] = frame([w, h], self.id + ":frequency")
        freq_payload["frequency_grid"] = {
            "model": "frequency_grid_v1",
            "layout": "fftshift_dc_center",
            "units": "cycles_per_pixel",
            "image_size": [w, h],
        }
        maps = {
            "gain": gain,
            "phase": phase,
            "supported": supported.astype(np.uint8) * 255,
            "phase_valid": phase_valid.astype(np.uint8) * 255,
        }
        outputs = {
            "image": a,
            "aperture": ImageArray(aperture, "gray"),
            "metrics": metrics,
            "mean_gain": mean if mean is not None else NO_VALUE,
            "maximum_phase_degrees": maximum_phase if maximum_phase is not None else NO_VALUE,
        }
        outputs.update(
            {name: ImageArray(np.fft.fftshift(value), "gray") for name, value in maps.items()}
        )
        graphics, drawings = {}, {}
        if self.generate_graphics:
            graphics["radial response"] = response_plot(metrics["radial"], p)
            color = (60, 210, 100) if not reasons else (60, 70, 240)
            label = f"Frequency response: {count}/{candidates} bins; gain " + (
                f"{mean:.4g}" if mean is not None else "undefined"
            )
            drawings["response status"] = GraphicDrawing(
                (graphic_object("text", [(10, 20)], color, text=label),)
            )
        return Result(
            outputs,
            payloads=dict.fromkeys(maps, freq_payload),
            graphics=graphics,
            drawings=drawings,
            passed=not reasons,
            fail_reasons=reasons,
            message=f"Measured {count}/{candidates} excited frequency bins",
        )
