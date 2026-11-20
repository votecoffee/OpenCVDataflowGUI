"""Structured-light fringe generation and absolute per-pixel temporal phase decoding."""

import hashlib
import json

import cv2
import numpy as np

from .calibration_revisions import is_full_grid
from .core import Port, Result
from .geometry3d_common import Tool3D
from .parameter_validation import NumericRange, ValidationError
from .payloads import (
    FlowGroup,
    ValuePacket,
    frame,
    image_with_payload,
    metadata,
    require_same_frame,
)
from .selection import selection_mask

MODEL = "phase_shift_sequence_v1"
AXES = ("x", "y")
PATTERN_ORDER = ("coarse_0", "coarse_120", "coarse_240", "fine_0", "fine_120", "fine_240")
MAX_PIXELS = 4_194_304
TAU = 2 * np.pi


def phase_specification(width, height, axis, periods):
    if axis not in AXES:
        raise ValidationError("Fringe axis must be x or y", "axis")
    for key, value in (("width", width), ("height", height), ("fine_periods", periods)):
        if type(value) is not int:
            raise ValidationError(f"{key} must be a whole number", key)
    if not (8 <= width <= 4096 and 8 <= height <= 4096 and width * height <= MAX_PIXELS):
        raise ValidationError("Projector dimensions must be 8..4096 with at most 4,194,304 pixels")
    length = width if axis == "x" else height
    if not 2 <= periods <= 64 or length // periods < 4:
        raise ValidationError(
            "Choose 2..64 fine periods with at least four pixels per period", "fine_periods"
        )
    spec = {
        "model": MODEL,
        "projector_size": [width, height],
        "axis": axis,
        "fine_periods": periods,
        "fine_period_px": length // periods,
        "order": list(PATTERN_ORDER),
    }
    encoded = json.dumps(spec, separators=(",", ":"), sort_keys=True).encode("ascii")
    return {**spec, "sha256": hashlib.sha256(encoded).hexdigest()}


def read_specification(spec):
    if not isinstance(spec, dict) or spec.get("model") != MODEL:
        raise ValidationError("Connect a phase-shift pattern specification")
    size = spec.get("projector_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValidationError("Phase-shift specification needs the projector dimensions")
    canonical = phase_specification(*size, spec.get("axis"), spec.get("fine_periods"))
    if spec != canonical:
        raise ValidationError(
            "Phase-shift specification does not match its order, period or digest"
        )
    return canonical


class PhaseShiftPatterns(Tool3D):
    type_id, title, subtype = (
        "3d.phase_shift_patterns",
        "Generate phase-shift patterns",
        "Structured light",
    )
    process_inputs_readonly = True
    parallel_safe = True
    outputs = (Port("patterns", "any"), Port("specification", "data"), Port("preview"))
    defaults = {"width": 640, "height": 480, "axis": "x", "fine_periods": 8}
    parameter_choices = {"axis": AXES}
    parameter_ranges = {
        "width": NumericRange(8, 4096, whole=True),
        "height": NumericRange(8, 4096, whole=True),
        "fine_periods": NumericRange(2, 64, whole=True),
    }
    parameter_help = {
        "axis": "Projector coordinate the fringes encode. X uses vertical stripes.",
        "fine_periods": "Requested fine cycles. OpenCV uses a whole-pixel fringe period.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        phase_specification(int(p["width"]), int(p["height"]), p["axis"], int(p["fine_periods"]))

    def process(self, inputs):
        p = self.parameters
        spec = phase_specification(
            int(p["width"]), int(p["height"]), p["axis"], int(p["fine_periods"])
        )
        width, height = spec["projector_size"]
        horizontal = spec["axis"] == "y"
        length = height if horizontal else width

        def analytic_triplet(period):
            angle = np.arange(length, dtype=np.float64) * (TAU / period)
            axis_shape = (-1, 1) if horizontal else (1, -1)
            for i in range(3):
                line = np.rint(127.5 + 127.5 * np.sin(angle + i * TAU / 3)).astype(np.uint8)
                yield np.broadcast_to(line.reshape(axis_shape), (height, width)).copy()

        # OpenCV divides by floor(period_count / 3) before checking markers (5.0.0).
        images = list(analytic_triplet(length))
        if spec["fine_periods"] < 3:
            images.extend(analytic_triplet(spec["fine_period_px"]))
        else:
            if not hasattr(getattr(cv2, "structured_light", None), "SinusoidalPattern"):
                raise ValidationError("OpenCV contrib SinusoidalPattern generation is unavailable")
            params = cv2.structured_light.SinusoidalPattern.Params()
            params.width, params.height, params.horizontal = width, height, horizontal
            params.shiftValue = TAU / 3
            params.methodId = cv2.structured_light.PSP
            params.setMarkers = False
            params.nbrOfPeriods = spec["fine_periods"]
            native = cv2.structured_light.SinusoidalPattern.create(params)
            ok, triplet = native.generate()
            if not ok or len(triplet) != 3:
                raise ValidationError("OpenCV could not generate three sinusoidal patterns")
            images.extend(triplet)
        spatial = frame(spec["projector_size"], f"{self.id}:projector")
        group = FlowGroup(
            tuple(
                ValuePacket(
                    image,
                    {"pattern_index": i, "phase_spec_sha256": spec["sha256"], "spatial": spatial},
                )
                for i, image in enumerate(images)
            ),
            PATTERN_ORDER,
        )
        return Result(
            {"patterns": group, "specification": spec, "preview": images[3]},
            payloads={"preview": {"spatial": spatial}},
            message=f"Six projected frames · {spec['axis'].upper()} coordinate",
        )


def capture_images(group, spec, source_id):
    if not isinstance(group, FlowGroup) or len(group.items) != len(PATTERN_ORDER):
        raise ValidationError("Connect six captures in coarse/fine phase order")
    images, payloads = [], []
    for i, packet in enumerate(group.items):
        payload = metadata(packet)
        if (
            payload.get("pattern_index", i) != i
            or payload.get("phase_spec_sha256", spec["sha256"]) != spec["sha256"]
        ):
            raise ValidationError(
                "Pattern index or phase specification differs from the capture sequence"
            )
        value = packet.value
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 2
            or not value.size
            or value.dtype not in (np.uint8, np.uint16, np.float32, np.float64)
        ):
            raise ValidationError("Captures must be monochrome uint8, uint16 or normalized float")
        images.append(np.asarray(value))
        payloads.append(payload)
    shape = images[0].shape
    if np.prod(shape) > MAX_PIXELS or any(v.shape != shape for v in images):
        raise ValidationError(
            "All six capture grids must match and contain at most 4,194,304 pixels"
        )
    size = list(shape[::-1])
    # A TIFF sequence has a distinct root per captured page. A transformed series
    # needs a shared explicit grid, while independent raw pages use caller-aligned pixels.
    if any("spatial" in p and not is_full_grid(p, size) for p in payloads):
        roots = [p.get("spatial", {}).get("root_id") for p in payloads]
        if None in roots or len(set(roots)) != 1:
            raise ValidationError("Transformed phase captures require a shared known camera grid")
        require_same_frame(group.items, message="Phase captures have different camera grids")
    pyramid = payloads[0].get("pyramid")
    if any(p.get("pyramid") != pyramid for p in payloads):
        raise ValidationError("Phase captures have different pyramid levels or transforms")
    context = {key: payloads[0][key] for key in ("spatial", "pyramid") if key in payloads[0]}
    context.setdefault("spatial", frame(size, f"{source_id}:camera"))
    return images, context


def demodulate(images, p):
    samples = []
    for image in images:
        if image.dtype.kind == "u":
            samples.append(image.astype(np.float32) / np.iinfo(image.dtype).max)
        else:
            finite = np.isfinite(image)
            if np.any(finite & ((image < 0) | (image > 1))):
                raise ValidationError(
                    "Floating phase captures must use normalized intensities in [0, 1]"
                )
            samples.append(np.where(finite, image, np.nan).astype(np.float32))
    a, b, c = samples
    sine = (2 * a - b - c) / 3
    cosine = (b - c) / np.sqrt(np.float32(3))
    amplitude = np.hypot(sine, cosine)
    mean = (a + b + c) / 3
    good = (
        (amplitude > 0)
        & (amplitude >= float(p["minimum_modulation"]))
        & (mean >= float(p["minimum_mean"]))
    )
    saturated = np.logical_or.reduce(tuple((v <= 0) | (v >= 1) for v in samples))
    return np.arctan2(sine, cosine), amplitude, good, saturated


class DecodePhaseShift(Tool3D):
    type_id, title, subtype = (
        "3d.phase_shift_decode",
        "Decode absolute phase-shift coordinates",
        "Structured light",
    )
    process_inputs_readonly = True
    parallel_safe = True
    payload_reference = "patterns"
    inputs = (
        Port("patterns", "any"),
        Port("specification", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("projector_coordinate"),
        Port("wrapped_phase"),
        Port("absolute_phase"),
        Port("modulation"),
        Port("cycle_error"),
        Port("valid_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_modulation": 0.03,
        "minimum_mean": 0.04,
        "maximum_cycle_error_fraction": 0.25,
        "reject_saturation": False,
    }
    parameter_ranges = {
        "minimum_modulation": NumericRange(0, 1),
        "minimum_mean": NumericRange(0, 1),
        "maximum_cycle_error_fraction": NumericRange(0.001, 0.49),
    }
    parameter_help = {
        "minimum_modulation": "Minimum normalized sine amplitude in each triplet.",
        "minimum_mean": "Minimum normalized mean exposure in each triplet.",
        "maximum_cycle_error_fraction": "Maximum coarse-to-fine error, in fine periods.",
        "reject_saturation": "Reject a pixel if any capture reaches either intensity rail.",
    }
    input_help = {
        "patterns": "Aligned monochrome captures in coarse then fine 0/120/240 degree order.",
        "specification": "Projector axis and temporal sequence from Generate phase-shift patterns.",
        "mask": "Limit phase decoding to the selected camera pixels.",
        "region": "Select camera pixels, intersecting the mask when supplied.",
    }

    def process(self, inputs):
        spec = read_specification(inputs["specification"])
        images, context = capture_images(inputs["patterns"], spec, self.id)
        p = self.parameters
        width, height = spec["projector_size"]
        length = width if spec["axis"] == "x" else height
        period = spec["fine_period_px"]
        chosen = selection_mask(inputs, image_with_payload(images[0], context))
        context["phase_shift"] = {
            "axis": spec["axis"],
            "projector_size": [width, height],
            "fine_period_px": period,
            "spec_sha256": spec["sha256"],
        }
        selected = np.ones(images[0].shape, bool) if chosen is None else chosen != 0

        coarse, coarse_amp, coarse_good, coarse_sat = demodulate(images[:3], p)
        fine, fine_amp, fine_good, fine_sat = demodulate(images[3:], p)
        signal = coarse_good & fine_good
        saturation = (coarse_sat | fine_sat) if p["reject_saturation"] else np.zeros_like(signal)
        eligible = selected & signal & ~saturation

        # Anchor the one-cycle phase to the legal continuous pixel interval.
        coarse_px = np.mod(coarse * (length / TAU) + 0.5, length) - 0.5
        fine_px = fine * (period / TAU)
        coordinate = fine_px + period * np.rint((coarse_px - fine_px) / period)
        error = np.abs(coordinate - coarse_px)
        consistent = error <= float(p["maximum_cycle_error_fraction"]) * period
        inside = (coordinate >= -0.5) & (coordinate < length - 0.5)
        valid = eligible & consistent & inside
        count, selected_count = int(valid.sum()), int(selected.sum())

        outputs = {
            "projector_coordinate": np.where(valid, coordinate, np.nan),
            "wrapped_phase": np.where(valid, fine, np.nan),
            "absolute_phase": np.where(valid, coordinate * (TAU / period), np.nan),
            "modulation": np.where(selected, np.minimum(coarse_amp, fine_amp), np.nan),
            "cycle_error": np.where(eligible, error, np.nan),
            "valid_mask": valid.astype(np.uint8) * 255,
            "report": {
                "axis": spec["axis"],
                "projector_size": [width, height],
                "fine_period_px": period,
                "selected_pixels": selected_count,
                "valid_pixels": count,
                "low_signal_pixels": int(np.count_nonzero(selected & ~signal)),
                "saturated_pixels": int(np.count_nonzero(selected & signal & saturation)),
                "unwrapping_rejected_pixels": int(
                    np.count_nonzero(eligible & ~(consistent & inside))
                ),
                "coordinate_units": "projector_pixels",
                "phase_units": "radians",
            },
        }
        graphics = {}
        if self.capture_graphics:
            display = np.where(valid, np.clip((coordinate + 0.5) / length, 0, 1), 0)
            colored = cv2.applyColorMap(np.rint(display * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            colored[~valid] = 0
            graphics[f"Projector {spec['axis'].upper()}"] = colored
        return Result(
            outputs,
            graphics=graphics,
            payloads={key: context for key in outputs},
            passed=bool(count),
            score=count / selected_count if selected_count else 0.0,
            fail_reasons=[] if count else ["no_valid_phase_correspondences"],
            message=f"{count}/{selected_count} valid projector {spec['axis'].upper()} coordinates",
        )


PHASE_SHIFT_BLOCKS = (PhaseShiftPatterns, DecodePhaseShift)
