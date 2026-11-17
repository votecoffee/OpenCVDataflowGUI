"""Bounded single-image motion PSF hypotheses with separate spatial-tile validation."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .deblur import motion_psf
from .graphics_layers import GraphicDrawing, graphic_object
from .histogram import HISTOGRAM_CHANNELS
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask


def search_grid(p):
    lengths = np.arange(
        p["minimum_length"], p["maximum_length"] + p["length_step"] * 1e-8, p["length_step"]
    )
    angles = (p["angle_start"] + np.arange(0, p["angle_span"], p["angle_step"])) % 180
    return lengths, angles


def frequency_samples(size, low, high):
    axis = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(fx, fy)
    valid = (radius >= low) & (radius <= high) & ((fy > 0) | ((fy == 0) & (fx > 0)))
    ring = np.floor(radius[valid] * size).astype(int)
    return valid, ring, np.bincount(ring)


def whiten(power, indices, ring, counts):
    floor = max(float(power.max()) * 1e-12, np.finfo(float).tiny)
    values = np.log(np.maximum(power[indices], floor))
    sums = np.bincount(ring, weights=values, minlength=len(counts))
    radial = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return values - radial[ring]


def tile_spectra(plane, tiles, size):
    axis = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    window = axis[:, None] * axis[None, :]
    power = [np.zeros((size, size)), np.zeros((size, size))]
    contrasts = [[], []]
    for i, (x, y) in enumerate(tiles):
        tile = plane[y : y + size, x : x + size]
        values = tile - float(tile.mean())
        contrasts[i % 2].append(float(np.std(values)))
        spectrum = cv2.dft(values * window, flags=cv2.DFT_COMPLEX_OUTPUT)
        power[i % 2] += np.square(spectrum).sum(axis=2) / size**4
    for i in range(2):
        power[i] /= len(contrasts[i])
    return power, [float(np.mean(a)) for a in contrasts]


def model_power(length, angle, size):
    kernel = motion_psf(length, angle)
    h, w = kernel.shape
    impulse = np.zeros((size, size))
    impulse[np.ix_((np.arange(h) - h // 2) % size, (np.arange(w) - w // 2) % size)] = kernel
    spectrum = cv2.dft(impulse, flags=cv2.DFT_COMPLEX_OUTPUT)
    power = np.square(spectrum).sum(axis=2)
    # Expected power for the fixed periodic separable Hann window on stationary data.
    weights = ((-1, 1 / 6), (0, 4 / 6), (1, 1 / 6))
    return sum(a * b * np.roll(power, (y, x), (0, 1)) for y, a in weights for x, b in weights)


def estimate(power, size, p):
    indices, ring, counts = frequency_samples(size, p["minimum_frequency"], p["maximum_frequency"])
    fit, validation = [whiten(a, indices, ring, counts) for a in power]
    variances = [float(np.mean(a * a)) for a in (fit, validation)]
    if min(variances) < 1e-12:
        return {"status": "insufficient_spectral_structure"}, None
    lengths, angles = search_grid(p)
    rows, best_model, best = [], None, None
    for li, length in enumerate(lengths):
        for ai, angle in enumerate(angles):
            predicted = model_power(float(length), float(angle), size)
            for floor in p["model_floors"]:
                model = whiten(predicted + floor, indices, ring, counts)
                error = float(np.mean((fit - model) ** 2))
                row = {
                    "length": float(length),
                    "angle": float(angle),
                    "model_floor": floor,
                    "fit_score": 1 - error / variances[0],
                    "length_index": li,
                    "angle_index": ai,
                }
                rows.append(row)
                if best is None or row["fit_score"] > best["fit_score"]:
                    best, best_model = row, predicted + floor
    # Validation observations never select parameters, the floor or a retry.
    model = whiten(best_model, indices, ring, counts)
    validation_score = 1 - float(np.mean((validation - model) ** 2)) / variances[1]
    ordered = sorted(rows, key=lambda r: -r["fit_score"])
    alternatives = [
        row
        for row in ordered
        if abs(row["length"] - best["length"]) >= p["distinct_length"]
        or abs((row["angle"] - best["angle"] + 90) % 180 - 90) >= p["distinct_angle"]
    ]
    runner = alternatives[0] if alternatives else None
    margin = best["fit_score"] - runner["fit_score"] if runner else None
    boundary = best["length_index"] in (0, len(lengths) - 1) or (
        p["angle_span"] < 180 and best["angle_index"] in (0, len(angles) - 1)
    )
    status = "ok"
    if best["fit_score"] < p["minimum_score"]:
        status = "weak_fit"
    elif validation_score < p["minimum_score"]:
        status = "validation_rejected"
    elif p["minimum_margin"] > 0 and (margin is None or margin < p["minimum_margin"]):
        status = "ambiguous"
    elif boundary and p["boundary_policy"] == "reject":
        status = "search_boundary"

    def public(row):
        return {k: v for k, v in row.items() if not k.endswith("_index")}

    return {
        "status": status,
        "candidate": public(best),
        "validation_score": validation_score,
        "peak_margin": margin,
        "alternative": public(runner) if runner else None,
        "search_boundary": boundary,
        "top_candidates": [public(row) for row in ordered[:8]],
        "frequency_samples": int(indices.sum()),
    }, best_model


def diagnostic(power, model, report):
    size = power[0].shape[0]
    panels = []
    for values in [*power, model if model is not None else np.ones_like(power[0])]:
        values = np.log(np.maximum(values, max(float(values.max()) * 1e-8, np.finfo(float).tiny)))
        low, high = np.percentile(values, [2, 98])
        pixels = np.uint8(
            np.clip((np.fft.fftshift(values) - low) / max(high - low, 1e-12), 0, 1) * 255
        )
        panels.append(cv2.applyColorMap(pixels, cv2.COLORMAP_VIRIDIS))
    image = np.zeros((size + 46, size * 3, 3), np.uint8)
    image[:size] = np.concatenate(panels, axis=1)
    for i, label in enumerate(("Fit spectrum", "Validation", "Candidate model")):
        cv2.putText(
            image,
            label,
            (i * size + 3, size + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (220, 220, 220),
            1,
        )
    cv2.putText(
        image,
        report["status"],
        (4, size + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (70, 220, 140) if report["status"] == "ok" else (60, 90, 240),
        1,
    )
    return image


class EstimateMotionBlur(Block):
    type_id, title = "inspect.estimate_motion_blur", "Estimate linear motion blur"
    category = block_type = "Image processing"
    subtype = "Restoration"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("psf", "data", required=False),
        Port("length", "number", required=False),
        Port("angle", "number", required=False),
        Port("report", "data"),
        Port("diagnostic", required=False),
    )
    defaults = {
        "channel": "gray",
        "tile_size": 128,
        "minimum_tiles": 4,
        "maximum_tiles": 32,
        "minimum_length": 4.0,
        "maximum_length": 24.0,
        "length_step": 1.0,
        "angle_start": 0.0,
        "angle_span": 180.0,
        "angle_step": 5.0,
        "model_floors": [0.0001, 0.001, 0.01, 0.1],
        "minimum_frequency": 0.04,
        "maximum_frequency": 0.45,
        "minimum_contrast": 0.001,
        "minimum_score": 0.6,
        "minimum_margin": 0.03,
        "distinct_length": 2.0,
        "distinct_angle": 10.0,
        "boundary_policy": "reject",
        "maximum_candidates": 4096,
        "maximum_search_work": 64_000_000,
        "maximum_working_bytes": 512 * 1024**2,
    }
    parameter_choices = {
        "channel": ("gray", "R", "G", "B"),
        "tile_size": (64, 128, 256, 512),
        "boundary_policy": ("reject", "allow"),
    }
    parameter_ranges = {
        "tile_size": NumericRange(64, 512, whole=True),
        "minimum_tiles": NumericRange(4, 64, whole=True),
        "maximum_tiles": NumericRange(4, 64, whole=True),
        "minimum_length": NumericRange(2, 63),
        "maximum_length": NumericRange(2, 63),
        "length_step": NumericRange(0.25, 16),
        "angle_start": NumericRange(-360, 360),
        "angle_span": NumericRange(0.25, 180),
        "angle_step": NumericRange(0.25, 180),
        "minimum_frequency": NumericRange(0.01, 0.25),
        "maximum_frequency": NumericRange(0.05, 0.49),
        "minimum_contrast": NumericRange(0, 1),
        "minimum_score": NumericRange(0, 1),
        "minimum_margin": NumericRange(0, 2),
        "distinct_length": NumericRange(0.25, 63),
        "distinct_angle": NumericRange(0.25, 90),
        "maximum_candidates": NumericRange(1, 16384, whole=True),
        "maximum_search_work": NumericRange(1, 256_000_000, whole=True),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
    }
    parameter_help = {
        "channel": "Luminance or a named stored color channel; alpha is excluded.",
        "tile_size": "Disjoint square tiles, fixed periodic Hann window; current-image pixels.",
        "minimum_tiles": "At least this many fully selected, opaque tiles; split fit/validation.",
        "maximum_tiles": "Evenly subsample this many eligible tiles before alternating groups.",
        "minimum_length": "First centered line-path length in current pixels.",
        "maximum_length": "Largest searched length; must be at most one quarter of Tile size.",
        "length_step": "Length grid spacing; last sampled length does not exceed Maximum length.",
        "angle_start": "First direction; output angles wrap modulo 180 degrees.",
        "angle_span": "Search arc in degrees; 180 covers all undirected line orientations.",
        "angle_step": "Grid spacing clockwise from horizontal in image coordinates.",
        "model_floors": "1-8 distinct positive floor ratios <=1; nuisance values, not SNR.",
        "minimum_frequency": "Lower fitted radial frequency, cycles/current pixel.",
        "maximum_frequency": "Upper fitted radial frequency, below axial Nyquist.",
        "minimum_contrast": "Minimum mean tile standard deviation divided by working sample scale.",
        "minimum_score": "Minimum explained whitened log-power variance in fit and validation.",
        "minimum_margin": "Fit score gap to a distinct hypothesis; zero disables ambiguity gating.",
        "distinct_length": "Length difference making an alternative distinct, in current pixels.",
        "distinct_angle": "Modulo-180 angle difference making an alternative distinct.",
        "boundary_policy": "Reject a best length/partial-arc endpoint, or explicitly allow it.",
        "maximum_candidates": "Cap total length/angle/floor trials before spectrum work.",
        "maximum_search_work": "Cap template pixels plus trial frequency samples before DFT.",
        "maximum_working_bytes": "Array estimate, excluding graph copies and native workspaces.",
    }
    input_help = {
        "image": "One spatially uniform linear-motion blur over approximately isotropic texture.",
        "mask": "Only fully selected square tiles contribute to estimation.",
        "region": "Optional region intersected with Mask; no black mask padding in spectra.",
    }
    output_help = {
        "image": "Original pixels, encoding, ticket and coordinates, unchanged even on rejection.",
        "psf": "Accepted centered normalized line PSF for Wiener deblur; withheld on rejection.",
        "length": "Accepted line length in current-image pixels; withheld on rejection.",
        "angle": "Accepted clockwise angle modulo 180 degrees; direction sign is unobservable.",
        "report": "Assumption, tile partition, best/alternative scores and acceptance reason.",
        "diagnostic": "Optional frequency-grid plot, with candidate model even when rejected.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "psf", "length", "angle", "report"]
        self.description = self.description or "Estimate a uniform line PSF on isotropic texture."
        self.documentation = self.documentation or (
            "<p>Compare centered line-PSF spectra on a bounded parameter grid after radial "
            "log-power whitening. Fit and validation use disjoint selected tiles. A constant "
            "spectral floor is a nuisance parameter, not a measured noise-to-signal ratio. "
            "Failed/weak/ambiguous estimates withhold PSF and geometry. The texture and blur "
            "assumptions require qualification; this is not arbitrary blind deconvolution. "
            "See docs/blur-estimation.md.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or (limits.whole and type(p[name]) is not int):
                raise ValidationError("Use typed numeric blur-search settings", name)
        floors = p["model_floors"]
        if (
            not isinstance(floors, list)
            or not 1 <= len(floors) <= 8
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
                for v in floors
            )
            or len(set(floors)) != len(floors)
        ):
            raise ValidationError(
                "Use 1-8 distinct finite spectral floor ratios in (0,1]", "model_floors"
            )
        if not p["minimum_length"] <= p["maximum_length"] <= p["tile_size"] / 4:
            raise ValidationError(
                "Length bounds must increase within Tile size / 4",
                "minimum_length",
                "maximum_length",
                "tile_size",
            )
        if p["minimum_tiles"] > p["maximum_tiles"]:
            raise ValidationError(
                "Minimum tiles exceeds Maximum tiles", "minimum_tiles", "maximum_tiles"
            )
        if p["minimum_frequency"] >= p["maximum_frequency"]:
            raise ValidationError(
                "Frequency limits must increase", "minimum_frequency", "maximum_frequency"
            )
        samples, _, rings = frequency_samples(
            p["tile_size"], p["minimum_frequency"], p["maximum_frequency"]
        )
        if samples.sum() < 64 or np.count_nonzero(rings) < 3:
            raise ValidationError(
                "Frequency band needs at least 64 nonredundant bins across three radial rings",
                "minimum_frequency",
                "maximum_frequency",
                "tile_size",
            )
        lengths, angles = search_grid(p)
        if len(lengths) * len(angles) * len(floors) > p["maximum_candidates"]:
            raise ValidationError("Blur search exceeds candidate budget", "maximum_candidates")

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        if not isinstance(source, np.ndarray) or source.ndim not in (2, 3):
            raise ValidationError("Use one intensity image for blur estimation")
        h, w = source.shape[:2]
        size = p["tile_size"]
        estimate_bytes = source.nbytes + h * w * 32 + size**2 * 192
        if (
            min(h, w) < 1
            or max(h, w) > 16384
            or h * w > 4_194_304
            or estimate_bytes > p["maximum_working_bytes"]
        ):
            raise ValidationError(
                "Blur estimation exceeds pixel/array limits", "maximum_working_bytes"
            )
        pixels = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        encoding, context = image_format(source), copy.deepcopy(self.input_payloads["image"])
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("Use grayscale or RGB/BGR(A) intensities")
        expected = () if encoding == "gray" else (4,) if encoding in ("rgba", "bgra") else (3,)
        if pixels.shape[2:] != expected:
            raise ValidationError("Channel layout must match the stored image encoding")
        selected = selection_mask(inputs, source)
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        if encoding in ("rgba", "bgra"):
            maximum = np.iinfo(pixels.dtype).max if pixels.dtype.kind == "u" else 1
            if context.get("alpha_representation", "straight") != "straight":
                raise ValidationError("Unassociate premultiplied alpha before blur estimation")
            if np.any(pixels[..., 3] < 0) or np.any(pixels[..., 3] > maximum):
                raise ValidationError("Alpha must lie within integer full scale or floating [0,1]")
            selected &= pixels[..., 3] > 0
        channels = HISTOGRAM_CHANNELS[encoding]
        if p["channel"] != "gray" and p["channel"] not in channels:
            raise ValidationError("Selected color channel is absent", "channel")
        values = np.asarray(pixels)[..., :3] if pixels.ndim == 3 else np.asarray(pixels)
        scale = (
            float(np.iinfo(pixels.dtype).max)
            if pixels.dtype.kind == "u"
            else float(np.max(np.abs(values[selected])))
            if selected.any()
            else 1.0
        ) or 1.0
        values = values.astype(np.float64) / scale
        if values.ndim == 2:
            plane = values
        elif p["channel"] != "gray":
            plane = values[..., channels.index(p["channel"])]
        else:
            plane = sum(
                values[..., channels.index(c)] * weight
                for c, weight in (("R", 0.299), ("G", 0.587), ("B", 0.114))
            )
        tiles = [
            (x, y)
            for y in range(0, h - size + 1, size)
            for x in range(0, w - size + 1, size)
            if selected[y : y + size, x : x + size].all()
        ]
        if len(tiles) > p["maximum_tiles"]:
            tiles = [
                tiles[i] for i in np.linspace(0, len(tiles) - 1, p["maximum_tiles"], dtype=int)
            ]
        report = {
            "model": "linear_motion_psf_estimate_v1",
            "status": "insufficient_tiles",
            "assumption": "Uniform centered line motion over approximately isotropic texture",
            "length_units": "current_image_pixels",
            "angle_units": "clockwise_degrees_mod_180",
            "scale": scale,
            "tile_size": size,
            "fit_tiles": tiles[::2],
            "validation_tiles": tiles[1::2],
            "validation_used_for_fit": False,
            "source_spatial": context.get("spatial"),
            "source_pyramid": context.get("pyramid"),
            "estimated_working_bytes": estimate_bytes,
        }
        power, candidate_model = None, None
        if len(tiles) >= p["minimum_tiles"]:
            indices, _, _ = frequency_samples(size, p["minimum_frequency"], p["maximum_frequency"])
            lengths, angles = search_grid(p)
            work = len(tiles) * size**2 + len(lengths) * len(angles) * (
                size**2 + len(p["model_floors"]) * int(indices.sum())
            )
            if work > p["maximum_search_work"]:
                raise ValidationError("Blur search work exceeds its budget", "maximum_search_work")
            report["search_work"] = work
            power, contrasts = tile_spectra(plane, tiles, size)
            report["tile_contrasts"] = contrasts
            report["status"] = "insufficient_contrast"
            if min(contrasts) >= p["minimum_contrast"] and min(contrasts) > 0:
                measured, candidate_model = estimate(power, size, p)
                report.update(measured)
        accepted = report["status"] == "ok"
        candidate = report.get("candidate")
        outputs = {
            "image": source,
            "psf": NO_VALUE,
            "length": NO_VALUE,
            "angle": NO_VALUE,
            "report": report,
            "diagnostic": NO_VALUE,
        }
        kernel_context = copy.deepcopy(context)
        kernel_context.pop("spatial", None)
        kernel_context.pop("pyramid", None)
        kernel_context["blur_estimate"] = {
            "length_units": "current_image_pixels",
            "source_spatial": context.get("spatial"),
            "source_pyramid": context.get("pyramid"),
        }
        payloads = {name: copy.deepcopy(context) for name in outputs}
        payloads["psf"] = kernel_context
        if accepted:
            outputs.update(
                psf=motion_psf(candidate["length"], candidate["angle"]).tolist(),
                length=candidate["length"],
                angle=candidate["angle"],
            )
        objects = []
        if self.generate_graphics:
            if power is not None:
                chart = diagnostic(power, candidate_model, report)
                outputs["diagnostic"] = ImageArray(chart, "bgr")
                payloads["diagnostic"] = copy.deepcopy(context)
                payloads["diagnostic"].pop("pyramid", None)
                payloads["diagnostic"]["spatial"] = frame(
                    list(chart.shape[1::-1]), f"{self.id}:blur_spectrum"
                )
            if accepted:
                x, y = np.array(tiles[0], float) + size / 2
                theta = math.radians(candidate["angle"])
                delta = np.array([math.cos(theta), math.sin(theta)]) * candidate["length"] / 2
                objects.append(
                    graphic_object("polyline", [[x, y] - delta, [x, y] + delta], (70, 220, 140))
                )
            objects.append(
                graphic_object(
                    "text",
                    [[8, 18]],
                    (70, 220, 140) if accepted else (60, 90, 240),
                    text="Motion PSF: " + report["status"],
                )
            )
        return Result(
            outputs,
            payloads=payloads,
            passed=accepted,
            score=float(np.clip(report.get("validation_score", 0), 0, 1)),
            fail_reasons=[] if accepted else ["blur_estimate." + report["status"]],
            drawings={"Motion blur estimate": GraphicDrawing(tuple(objects), "image")}
            if objects
            else {},
            message=report["status"],
        )
