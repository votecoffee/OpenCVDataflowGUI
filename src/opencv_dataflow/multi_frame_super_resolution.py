"""Classical super-resolution for a bounded window of translated source frames."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, NO_VALUE, frame, image_with_payload, metadata, moved_payload
from .processing import gray_image, image_value
from .restoration import aligned_observations
from .selection import SELECTION_INPUTS, selection_mask
from .translation_reconstruction import BoxObservation, footprint_support, reconstruct

MAX_OUTPUT_PIXELS = 2_097_152
MAX_ITERATED_SAMPLES = 480_000_000
DEPTHS = (np.uint8, np.uint16, np.float32)
BASELINES = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4}
FAILURES = {
    "super_resolution_too_few_aligned_frames": "Too few frames aligned to the reference",
    "super_resolution_insufficient_subpixel_phases": "Too few distinct subpixel sampling phases",
    "super_resolution_no_supported_pixels": "No selected pixels have enough observation support",
}


def phase_alignment(anchor, candidate, window, settings):
    shift, response = cv2.phaseCorrelate(anchor.copy(), candidate.copy(), window)
    metrics = {"phase_response": float(response) if np.isfinite(response) else None, "ecc": None}
    if (
        not np.isfinite(shift).all() or not np.isfinite(response)
        or response < float(settings["minimum_phase_response"])
    ):
        return None, metrics, "weak_phase_correlation"
    if max(map(abs, shift)) > float(settings["maximum_shift"]) + 2:
        return None, metrics, "outside_search_range"
    seed = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    termination = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-5)
    try:
        coefficient, matrix = cv2.findTransformECC(
            anchor, candidate, seed, cv2.MOTION_TRANSLATION, termination, None, 5,
        )
    except cv2.error:
        return None, metrics, "ecc_did_not_converge"
    metrics["ecc"] = float(coefficient) if np.isfinite(coefficient) else None
    if metrics["ecc"] is None or coefficient < float(settings["minimum_ecc"]):
        return None, metrics, "weak_ecc"
    return matrix[:, 2].astype(float), metrics, None


def explicit_offsets(value, count, anchor):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValidationError("Supply one ordered [dx, dy] shift per frame in the batch", "shifts")
    try:
        offsets = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Shifts must be finite [dx, dy] pairs", "shifts") from exc
    if offsets.shape != (count, 2) or not np.isfinite(offsets).all():
        raise ValidationError("Shifts must be finite [dx, dy] pairs", "shifts")
    if np.max(np.abs(offsets[anchor])) > 1e-6:
        raise ValidationError("The selected anchor shift must be [0, 0]", "shifts")
    return offsets


class MultiFrameSuperResolution(Block):
    type_id, title = "sequence.multi_frame_super_resolution", "Multi-frame super-resolution"
    category = block_type = "Sequence"
    subtype = "Temporal processing"
    description = "Reconstruct a higher-resolution reference from subpixel-shifted observations."
    inputs = (Port("frames", "any"), *SELECTION_INPUTS[1:], Port("shifts", "data", required=False))
    outputs = (
        Port("image", required=False), Port("baseline"), Port("coverage"),
        Port("processed_mask", visual_type="mask"), Port("geometry", "data"), Port("report", "data"),
    )
    payload_reference = "frames"
    process_inputs_readonly = preview_safe = True
    defaults = {
        "scale": 2,
        "target_index": -1,
        "temporal_window": 5,
        "alignment": "estimate",
        "shifts": [],
        "maximum_shift": 4.0,
        "minimum_phase_response": 0.2,
        "minimum_ecc": 0.8,
        "minimum_frames": 3,
        "minimum_phase_bins": 2,
        "iterations": 90,
        "residual_limit": 0.12,
        "regularization": 0.003,
        "edge_threshold": 0.06,
        "baseline_interpolation": "cubic",
    }
    parameter_choices = {"alignment": ("estimate", "provided"), "baseline_interpolation": tuple(BASELINES)}
    parameter_ranges = {
        "scale": NumericRange(2, 4, whole=True),
        "target_index": NumericRange(-1, 63, whole=True),
        "temporal_window": NumericRange(3, 15, whole=True, odd=True),
        "maximum_shift": NumericRange(0.01, 64),
        "minimum_phase_response": NumericRange(0.0, 1.0),
        "minimum_ecc": NumericRange(0.0, 1.0),
        "minimum_frames": NumericRange(2, 15, whole=True),
        "minimum_phase_bins": NumericRange(2, 16, whole=True),
        "iterations": NumericRange(1, 240, whole=True),
        "residual_limit": NumericRange(0.001, 1),
        "regularization": NumericRange(0, 0.1),
        "edge_threshold": NumericRange(0.001, 1),
    }
    parameter_help = {
        "scale": "Integer high-resolution samples per source pixel.",
        "target_index": "Zero-based reference frame. −1 selects the earlier middle.",
        "temporal_window": "Odd number of observations centered on the target.",
        "alignment": "Estimate translations with phase correlation and ECC, or use known offsets.",
        "shifts": "For provided alignment, ordered [dx, dy] offsets from anchor to each raw frame, in source pixels.",
        "maximum_shift": "Reject observations with a translation component above this limit in source pixels.",
        "minimum_phase_response": "Reject weak phase correlation before refinement.",
        "minimum_ecc": "Reject low correlation after translation refinement.",
        "minimum_frames": "Required accepted frames overall and required per reconstructed pixel.",
        "minimum_phase_bins": "Require distinct rounded subpixel phases on the output sampling grid.",
        "iterations": "Number of projected backprojection steps.",
        "residual_limit": "Huber transition for observation residuals, in normalized intensity units.",
        "regularization": "Strength of the neighboring-pixel Huber penalty. Zero disables it.",
        "edge_threshold": "Huber transition for neighboring-pixel differences.",
        "baseline_interpolation": "Resize the anchor for initialization and pixels outside reconstruction support.",
    }
    input_help = {
        "frames": "Ordered batch of raw same-grid observations. Frame motion is modeled as translation.",
        "mask": "Limit reconstructed writes in the anchor raster. Full observations supply the estimate.",
        "region": "Intersect the write mask with a region on the anchor raster.",
        "shifts": "Override configured offsets in Provided mode. One pair per batch item.",
    }
    output_help = {
        "image": "Reconstruction with the anchor's encoding, depth, ticket and output pixel grid.",
        "baseline": "Single-frame interpolation from the anchor.",
        "coverage": "Count of sensor observations with a valid footprint at each output pixel.",
        "processed_mask": "Pixels where the reconstruction replaces interpolation.",
        "report": "Input window, alignment decisions, phase diversity and output support.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_frames"] > p["temporal_window"]:
            raise ValidationError("Minimum frames exceeds the temporal window", "minimum_frames")
        if p["minimum_phase_bins"] > min(p["temporal_window"], p["scale"] ** 2):
            raise ValidationError("Minimum phase bins exceeds the window or scale", "minimum_phase_bins")

    def _observations(self, images, entries, first, half, raw_offsets):
        p = self.parameters
        auto = raw_offsets is None
        anchor_gray = window = None
        if auto:
            gray = [gray_image(im, (np.float32,), auto_depth=True) for im in images]
            anchor_gray = gray[half]
            window = cv2.createHanningWindow(anchor_gray.shape[::-1], cv2.CV_32F)
        scale = int(p["scale"])
        rows, phases, observations = [], set(), []
        for i, image in enumerate(images):
            if self._cancelled():
                raise InterruptedError("Multi-frame alignment cancelled")
            shift, details, reason = np.zeros(2), {}, None
            if i != half:
                if auto:
                    shift, details, reason = phase_alignment(anchor_gray, gray[i], window, p)
                else:
                    shift = raw_offsets[first + i]
            if shift is not None and (
                not np.isfinite(shift).all() or np.max(np.abs(shift)) > float(p["maximum_shift"])
            ):
                reason = "outside_search_range"
            if reason is None:
                observation = BoxObservation.create(image, scale, shift)
                if observation.valid.any():
                    phases.add(tuple((np.rint(shift * scale).astype(int) % scale).tolist()))
                    observations.append(observation)
                else:
                    reason = "no_shared_aperture"
            clean_shift = shift is not None and bool(np.isfinite(shift).all())
            status = {
                "index": first + i, "ticket": metadata(entries[i]).get("ticket"),
                "shift_pixels": shift.tolist() if clean_shift else None, "accepted": reason is None,
                "reason": reason, **details,
            }
            rows.append(status)
        return observations, phases, rows

    def process(self, inputs):
        group = inputs["frames"]
        if not isinstance(group, FlowGroup):
            raise ValidationError("Connect an ordered image batch to Frames")
        p = self.parameters
        half = int(p["temporal_window"]) // 2
        target = int(p["target_index"])
        if target < 0:
            target = (len(group.items) - 1) // 2
        first, stop = target - half, target + half + 1
        if first < 0 or stop > len(group.items):
            raise ValidationError("Temporal window does not fit around the selected anchor")
        entries = group.items[first:stop]
        frames = aligned_observations(entries)
        anchor = frames[half]
        encoding, dtype = image_format(anchor), anchor.dtype
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or dtype not in DEPTHS:
            raise ValidationError("Use gray or RGB/BGR(A) with uint8, uint16 or normalized float32")
        context = metadata(entries[half])
        alpha = encoding in ("rgba", "bgra")
        if alpha and any(
            metadata(entry).get("alpha_representation", "straight") != "straight" for entry in entries
        ):
            raise ValidationError("Unassociate premultiplied alpha before reconstruction")
        height, width = anchor.shape[:2]
        scale = int(p["scale"])
        size = [width * scale, height * scale]
        if min(height, width) < 8 or max(size) > 8192 or size[0] * size[1] > MAX_OUTPUT_PIXELS:
            raise ValidationError("Use source axes of at least 8 px and at most 2,097,152 output pixels")
        channels = 1 if anchor.ndim == 2 else 3
        if size[0] * size[1] * channels * len(entries) * p["iterations"] > MAX_ITERATED_SAMPLES:
            raise ValidationError("Reconstruction work exceeds the limit. Reduce scale, window or iterations.")
        if p["alignment"] == "provided":
            offsets = explicit_offsets(inputs.get("shifts", p["shifts"]), len(group.items), target)
        else:
            if "shifts" in inputs:
                raise ValidationError("Choose Provided alignment when connecting Shifts", "alignment")
            offsets = None
        raw = [image_value(image, DEPTHS) for image in frames]
        if dtype == np.float32 and any(image.min() < 0 or image.max() > 1 for image in raw):
            raise ValidationError("Floating-point camera samples must be normalized to [0, 1]")
        samples = [convert_depth(image[..., :3] if alpha else image, "float32") for image in raw]
        full_base = cv2.resize(
            convert_depth(raw[half], "float32"), size, interpolation=BASELINES[p["baseline_interpolation"]],
        )
        np.clip(full_base, 0, 1, out=full_base)
        signal_base = full_base[..., :3] if alpha else full_base
        selection = selection_mask(inputs, image_with_payload(anchor, context))
        observations, phases, alignments = self._observations(
            # Preserve the color order so automatic grayscale conversion uses the correct weights.
            [ImageArray(image, encoding[:3] if alpha else encoding) for image in samples],
            entries, first, half, offsets,
        )
        reason = None
        if len(observations) < p["minimum_frames"]:
            reason = "super_resolution_too_few_aligned_frames"
        elif len(phases) < p["minimum_phase_bins"]:
            reason = "super_resolution_insufficient_subpixel_phases"
        support, coverage = footprint_support(observations, signal_base.shape[:2])
        processed = coverage >= p["minimum_frames"]
        if selection is not None:
            processed &= cv2.resize(selection, size, interpolation=cv2.INTER_NEAREST) != 0
        if reason is None and not processed.any():
            reason = "super_resolution_no_supported_pixels"
        baseline = ImageArray(convert_depth(full_base, dtype.name), encoding)
        output = NO_VALUE
        if not reason:
            recon = reconstruct(
                signal_base, observations, support, int(p["iterations"]), float(p["residual_limit"]),
                float(p["regularization"]), float(p["edge_threshold"]), self._cancelled,
            )
            merged = full_base.copy()
            signal = merged[..., :3] if alpha else merged
            signal[processed] = recon[processed]
            output = ImageArray(convert_depth(merged, dtype.name), encoding)
        else:
            processed.fill(False)
        inverse = [[1 / scale, 0, (1 / scale - 1) / 2], [0, 1 / scale, (1 / scale - 1) / 2], [0, 0, 1]]
        low_size = [width, height]
        context.setdefault("spatial", frame(low_size))
        context.pop("raw_sensor", None)
        high = moved_payload(context, inverse, size)
        diagnostics = {
            "model": "translated_burst_super_resolution_v1", "scale": scale,
            "anchor_index": target, "window_indices": list(range(first, stop)),
            "alignment": p["alignment"], "frames": alignments, "accepted_frames": len(observations),
            "phase_bins": [list(phase) for phase in sorted(phases)],
            "processed_pixels": int(processed.sum()), "input_size": low_size, "output_size": size,
            "failure": reason,
        }
        high_mono = {key: value for key, value in high.items() if key != "alpha_representation"}
        return Result(
            {
                "image": output, "baseline": baseline,
                "coverage": ImageArray(coverage, "gray"),
                "processed_mask": ImageArray(processed.astype(np.uint8) * 255, "gray"),
                "geometry": {"input_size": low_size, "output_size": size, "output_to_input": inverse,
                             "coordinate_convention": "nominal half-pixel reference grid"},
                "report": diagnostics,
            },
            payloads={
                "image": high, "baseline": high, "coverage": high_mono, "processed_mask": high_mono,
                "geometry": context, "report": context,
            },
            passed=reason is None,
            fail_reasons=[{"code": reason, "message": FAILURES[reason]}] if reason else [],
            message=FAILURES[reason] if reason else (
                f"{len(observations)} aligned frames · {len(phases)} phases · "
                f"{int(processed.sum())} pixels reconstructed"
            ),
        )
