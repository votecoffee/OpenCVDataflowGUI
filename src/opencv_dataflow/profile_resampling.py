"""Coordinate-preserving resampling of measured image profiles onto a uniform grid."""

import copy
import math

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .projection import profile_plot
from .workplane import LENGTH_UNITS, require_workplane_scale


def read_profile(value):
    if not isinstance(value, dict) or value.get("model") not in (None, "image_profile_v1"):
        raise ValidationError("Connect a path intensity profile or image_profile_v1")
    if type(value.get("closed")) is not bool:
        raise ValidationError("Profile must declare open or closed sampling")
    try:
        distances = np.asarray(value["distances"], np.float64)
        rows = value["channels"]
        points = value["points"]
    except KeyError, TypeError, ValueError, OverflowError:
        raise ValidationError("Profile needs distances, source points and channels") from None
    if not isinstance(value["distances"], list) or any(
        type(v) not in (int, float) for v in value["distances"]
    ):
        raise ValidationError("Distances must be numeric image-profile coordinates")
    if distances.ndim != 1 or not 2 <= len(distances) <= 8192 or not np.isfinite(distances).all():
        raise ValidationError("Profile requires 2-8192 finite distances")
    delta = np.diff(distances)
    step = float(delta[0])
    if (
        step <= 0
        or np.max(np.abs(distances)) > 1e9
        or not np.allclose(delta, step, rtol=1e-6, atol=1e-10)
        or np.spacing(max(1, float(np.max(np.abs(distances))))) > step * 0.001
    ):
        raise ValidationError(
            "Profile distances need a uniform increasing grid and a resolved local origin"
        )
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValidationError("Profile needs 1-4 named image channels")
    labels, columns = [], []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("channel"), str)
            or not 1 <= len(row["channel"]) <= 32
        ):
            raise ValidationError("Every profile channel needs a short name")
        values = row.get("values")
        if (
            not isinstance(values, list)
            or len(values) != len(distances)
            or any(
                v is not None
                and (type(v) not in (int, float) or abs(v) > 1e100 or not math.isfinite(v))
                for v in values
            )
        ):
            raise ValidationError(
                "Channel values need finite numbers or null, matching the distance count"
            )
        labels.append(row["channel"])
        columns.append([np.nan if v is None else v for v in values])
    if len(set(labels)) != len(labels):
        raise ValidationError("Profile channel names must be unique")
    coordinates = np.full((len(distances), 2), np.nan)
    if not isinstance(points, list) or len(points) != len(distances):
        raise ValidationError("Source points must match the profile sample count")
    for index, point in enumerate(value["points"]):
        if point is None:
            continue
        try:
            pair = np.asarray(point, np.float64)
        except TypeError, ValueError, OverflowError:
            raise ValidationError("Source points need finite [x,y] coordinates or null") from None
        if pair.shape != (2,) or not np.isfinite(pair).all() or np.max(np.abs(pair)) > 1e7:
            raise ValidationError("Source points exceed finite image coordinate limits")
        coordinates[index] = pair
    samples = np.array(columns, np.float64).T
    if np.any(np.isfinite(samples).any(axis=1) & ~np.isfinite(coordinates).all(axis=1)):
        raise ValidationError("Measured values require source image coordinates")
    units = value.get("units", "px")
    if units not in ("px", *LENGTH_UNITS):
        raise ValidationError("Profile distance units must be px, mm, cm, m or in")
    length = value.get("path_length")
    expected = distances[-1] - distances[0] + (step if value["closed"] else 0)
    if (
        type(length) not in (int, float)
        or abs(length) > 1e100
        or not math.isfinite(length)
        or not np.isclose(length, expected, rtol=1e-6, atol=1e-9)
    ):
        raise ValidationError("Profile path length differs from its declared sampling extent")
    return distances, samples, coordinates, labels, step, units


def mapped_indices(indices, size, boundary, closed):
    outside = (indices < 0) | (indices >= size)
    if closed:
        return indices % size, False
    if boundary == "reflect":
        folded = indices % (2 * (size - 1))
        return np.where(folded < size, folded, 2 * (size - 1) - folded), False
    return np.clip(indices, 0, size - 1), bool(outside.any() and boundary == "invalid")


def sample_weights(position, ratio, p):
    down = p["antialias"] and ratio > 1 + 1e-9
    cutoff = p["rolloff"] / ratio if down else 1.0
    half = math.ceil(p["kernel_lobes"] * (ratio if down else 1))
    anchor = math.floor(position)
    if p["method"] == "linear" and not down:
        return np.array([anchor, anchor + 1]), np.array(
            [1 - (position - anchor), position - anchor]
        )
    if p["method"] == "linear":
        offsets = np.arange(-half, half + 1)
        kernel = cutoff * np.sinc(cutoff * offsets) * np.kaiser(len(offsets), p["window_beta"])
        kernel /= kernel.sum()
        fraction = position - anchor
        weights = np.r_[kernel * (1 - fraction), 0] + np.r_[0, kernel * fraction]
        return anchor + np.arange(-half, half + 2), weights
    indices = anchor + np.arange(-half, half + 2)
    offsets = position - indices
    inside = np.abs(offsets) <= half
    window = np.zeros(len(offsets))
    window[inside] = np.i0(
        p["window_beta"] * np.sqrt(np.maximum(0, 1 - (offsets[inside] / half) ** 2))
    ) / np.i0(p["window_beta"])
    weights = cutoff * np.sinc(cutoff * offsets) * window
    weights /= weights.sum()
    return indices, weights


def resample_values(samples, positions, ratio, closed, p):
    count, channels = samples.shape
    output = np.full((len(positions), channels), np.nan)
    extended = np.zeros(len(positions), bool)
    taps = 2 * math.ceil(p["kernel_lobes"] * (ratio if p["antialias"] and ratio > 1 else 1)) + 2
    if len(positions) * taps * channels > p["maximum_work"]:
        raise ValidationError("Profile/kernel/channel work exceeds limit", "maximum_work")
    for row, position in enumerate(positions):
        if not closed and not -1e-8 <= position <= count - 1 + 1e-8:
            continue
        position = position % count if closed else float(np.clip(position, 0, count - 1))
        indices, weights = sample_weights(position, ratio, p)
        active = np.abs(weights) > 1e-12
        indices, weights = indices[active], weights[active]
        uses_extension = not closed and bool(((indices < 0) | (indices >= count)).any())
        indices, unavailable = mapped_indices(indices, count, p["boundary"], closed)
        if unavailable:
            continue
        values = samples[indices]
        valid = np.isfinite(values).all(axis=0)
        if valid.any():
            output[row, valid] = weights @ values[:, valid]
            extended[row] = uses_extension
    if not np.isfinite(output[~np.isnan(output)]).all():
        raise ValidationError("Resampling overflowed; reduce the input sample range")
    return output, extended


class ResampleImageProfile(Block):
    type_id, title = "cv.resample_image_profile", "Resample image profile"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (
        Port("profile", "data"),
        Port("reference", "data", required=False, required_when_connected=True),
        Port("scale", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("profile", "data"),
        Port("points", "data"),
        Port("valid", "data"),
        Port("selected_count", "number"),
        Port("report", "data"),
    )
    defaults = {
        "grid": "own_extent",
        "pitch": 1.0,
        "start": 0.0,
        "end": 100.0,
        "distance_origin": 0.0,
        "method": "kaiser_sinc",
        "antialias": True,
        "kernel_lobes": 12,
        "window_beta": 8.6,
        "rolloff": 0.9,
        "boundary": "reflect",
        "maximum_work": 8_000_000,
    }
    parameter_choices = {
        "grid": ("own_extent", "explicit_extent", "reference"),
        "method": ("kaiser_sinc", "linear"),
        "boundary": ("reflect", "replicate", "invalid"),
    }
    parameter_ranges = {
        "pitch": NumericRange(1e-6, 1e6),
        "start": NumericRange(-1e9, 1e9),
        "end": NumericRange(-1e9, 1e9),
        "distance_origin": NumericRange(-1e9, 1e9),
        "kernel_lobes": NumericRange(2, 32, whole=True),
        "window_beta": NumericRange(0, 20),
        "rolloff": NumericRange(0.1, 1),
        "maximum_work": NumericRange(1, 32_000_000, whole=True),
    }
    parameter_help = {
        "grid": "Use this profile's extent, explicit start/end, or the Reference distance grid.",
        "pitch": (
            "Exact target spacing in profile units; a partial final interval is "
            "omitted, never stretched."
        ),
        "start": (
            "First target coordinate for explicit extent; outside-source samples stay invalid."
        ),
        "end": (
            "Maximum target coordinate for explicit extent; included only when pitch lands on it."
        ),
        "distance_origin": (
            "Add to source arc distances after physical-scale conversion; aligns "
            "taught path origins."
        ),
        "method": (
            "Windowed sinc, or linear interpolation (with a prefilter for "
            "antialiased downsampling)."
        ),
        "antialias": (
            "Low-pass before decimation; disabling it permits aliases. Gaps never "
            "become invented data."
        ),
        "kernel_lobes": (
            "Kernel half-support in source samples, expanded by the decimation ratio."
        ),
        "window_beta": "Kaiser window beta; controls stopband attenuation and transition width.",
        "rolloff": "Cutoff as a fraction of target Nyquist when reducing sample density.",
        "boundary": (
            "Extend outer ends by reflection or replication, or mark unsupported "
            "kernels invalid. Closed paths wrap."
        ),
        "maximum_work": (
            "Output samples times kernel taps times channels ceiling; not an "
            "elapsed-time guarantee."
        ),
    }
    input_help = {
        "profile": (
            "Measured Path intensity profile, or image_profile_v1, with source "
            "points and image payload."
        ),
        "reference": (
            "Reference target distances/units only; its values and ticket are "
            "never treated as current measurements."
        ),
        "scale": (
            "Optional current rectified-workplane scale converts pixel arc "
            "distance to physical units."
        ),
    }
    preview_safe = parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = (
            self.description or "Align measured image profiles to an exact pixel or physical pitch."
        )
        self.documentation = self.documentation or (
            "<p>Resamples uniform path measurements while retaining source image coordinates, "
            "tickets and explicit invalid gaps. A reference profile supplies "
            "a shared distance grid. "
            "Rate-aware Kaiser filtering suppresses aliases during decimation. "
            "See docs/profile-resampling.md for boundary, calibration and validity contracts.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["grid"] == "explicit_extent" and float(p["end"]) <= float(p["start"]):
            raise ValidationError("End must exceed Start", "start", "end")
        if type(p["antialias"]) is not bool:
            raise ValidationError("Antialias must be boolean", "antialias")

    def process(self, inputs):
        p = dict(self.parameters)
        for name, bounds in self.parameter_ranges.items():
            p[name] = bounds.check(p[name], name, name)
        profile = inputs["profile"]
        distances, samples, points, labels, source_step, units = read_profile(profile)
        payload = self.input_payloads["profile"]
        factor = 1.0
        if "scale" in inputs:
            if units != "px" or "spatial" not in payload:
                raise ValidationError(
                    "Physical conversion requires a pixel profile with its image frame"
                )
            try:
                units, factor = require_workplane_scale(
                    inputs["scale"], payload, self.input_payloads["scale"]
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        domain = distances * factor + p["distance_origin"]
        source_step *= factor
        closed = profile["closed"]
        period = profile["path_length"] * factor
        reference = None
        if p["grid"] == "reference":
            if "reference" not in inputs:
                raise ValidationError("Reference grid requires a current Reference input", "grid")
            target, _, _, _, pitch, ref_units = read_profile(inputs["reference"])
            if ref_units != units or inputs["reference"]["closed"] != closed:
                raise ValidationError(
                    "Reference must use the same distance units and open/closed convention"
                )
            if closed and not np.isclose(inputs["reference"]["path_length"], period, rtol=1e-6):
                raise ValidationError("Closed reference must have the same path period")
            reference = copy.deepcopy(self.input_payloads["reference"])
        else:
            if "reference" in inputs:
                raise ValidationError(
                    "Choose Reference grid to consume the connected reference", "grid"
                )
            pitch = p["pitch"]
            start = domain[0] if p["grid"] == "own_extent" else p["start"]
            end = (
                (domain[0] + period if closed else domain[-1])
                if p["grid"] == "own_extent"
                else p["end"]
            )
            if closed:
                quotient = (end - start) / pitch
                if not np.isclose(end - start, period, rtol=1e-6) or not np.isclose(
                    quotient, round(quotient), rtol=1e-8
                ):
                    raise ValidationError(
                        "Closed target pitch must divide one whole path period", "pitch"
                    )
                count = round(quotient)
            else:
                count = math.floor((end - start) / pitch + 1e-9) + 1
            if not 2 <= count <= 8192:
                raise ValidationError(
                    "Target grid requires 2-8192 samples; adjust pitch/extent", "pitch"
                )
            target = start + np.arange(count) * pitch
        ratio = pitch / source_step
        if not 1 / 256 <= ratio <= 256 or np.max(np.abs(target)) > 1e9:
            raise ValidationError(
                "Pitch ratio exceeds 256 or target coordinates exceed supported bounds"
            )
        if np.spacing(max(1, float(np.max(np.abs(target))))) > pitch * 0.001:
            raise ValidationError("Target pitch needs a resolved local origin", "pitch")
        positions = (target - domain[0]) / source_step
        values, extended = resample_values(samples, positions, ratio, closed, p)
        valid = np.isfinite(values)
        geometry = []
        fractional = []
        source_indices = np.arange(len(points))
        for position in positions:
            if closed:
                position = position % len(points)
                index = math.floor(position)
                pair = points[[index, (index + 1) % len(points)]]
                point = pair[0] + (position - index) * (pair[1] - pair[0])
            elif -1e-8 <= position <= len(points) - 1 + 1e-8:
                position = float(np.clip(position, 0, len(points) - 1))
                point = np.array(
                    [np.interp(position, source_indices, points[:, axis]) for axis in (0, 1)]
                )
            else:
                position, point = None, [np.nan, np.nan]
            geometry.append(np.asarray(point).tolist() if np.isfinite(point).all() else None)
            fractional.append(None if position is None else float(position))
        output = {
            "model": "image_profile_v1",
            "closed": closed,
            "units": units,
            "path_length": float(pitch * len(target) if closed else target[-1] - target[0]),
            "distances": target.tolist(),
            "points": geometry,
            "channels": [
                {
                    "channel": name,
                    "values": [
                        float(v) if keep else None
                        for v, keep in zip(values[:, i], valid[:, i], strict=True)
                    ],
                }
                for i, name in enumerate(labels)
            ],
        }
        report = {
            "model": "profile_resampling_v1",
            "units": units,
            "source_pitch": source_step,
            "target_pitch": pitch,
            "source_count": len(samples),
            "target_count": len(target),
            "source_extent": [float(domain[0]), float(domain[-1])],
            "source_period": period if closed else None,
            "method": p["method"],
            "antialias_applied": bool(p["antialias"] and ratio > 1 + 1e-9),
            "cutoff_input_cycles_per_sample": 0.5
            * (p["rolloff"] / ratio if p["antialias"] and ratio > 1 else 1),
            "boundary": "wrap" if closed else p["boundary"],
            "invalid_policy": "strict_kernel_support",
            "boundary_extended": extended.tolist(),
            "edge_extension_count": int(extended.sum()),
            "channel_valid_counts": dict(
                zip(labels, valid.sum(axis=0).astype(int).tolist(), strict=True)
            ),
            "source_fractional_indices": fractional,
            "source_context": copy.deepcopy(payload),
            "reference_context": reference,
            "geometry": "linear interpolation of input sample polyline",
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            for i, name in enumerate(labels):
                graphics["Profile " + name] = profile_plot(
                    values[:, i : i + 1],
                    valid[:, i],
                    [name],
                    "distance",
                    "intensity",
                    distances=target,
                    distance_units=units,
                )
            runs = []
            for point in geometry:
                if point is None:
                    runs.append([])
                elif not runs:
                    runs.append([point])
                else:
                    runs[-1].append(point)
            objects = tuple(
                graphic_object("polyline", run, (0, 210, 255)) for run in runs if len(run) > 1
            )
            if objects:
                drawings["Resampled sampling path"] = GraphicDrawing(objects, source="profile")
        selected = int(valid.any(axis=1).sum())
        outputs = {
            "profile": output,
            "points": geometry,
            "valid": valid.tolist(),
            "selected_count": selected,
            "report": report,
        }
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            graphics=graphics,
            drawings=drawings,
            passed=selected > 0,
            fail_reasons=[] if selected else ["profile_empty_support"],
            message=f"{selected}/{len(target)} samples supported at {pitch:g} {units}",
        )
