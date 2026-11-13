"""Reference-guided, region-constant attenuation correction with explicit validity."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_object
from .illumination import INTENSITY_FORMATS
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata, require_same_frame
from .processing import image_value
from .selection import SELECTION_INPUTS, selection_mask

MAX_PIXELS = 4_194_304
MAX_ROI_WORK = 33_554_432


def _brightness_chroma(samples):
    """Nonnegative mean and channel fractions without summation overflow."""
    if samples.ndim == 2:
        return samples, None
    peak = samples.max(axis=2)
    scaled = samples / np.where(peak > 0, peak, 1)[..., None]
    mean = scaled.mean(axis=2) * peak
    chroma = scaled / np.where(scaled.sum(axis=2) > 0, scaled.sum(axis=2), 1)[..., None]
    return mean, chroma


def _fill_small_holes(candidate, eligible, limit, connectivity):
    if not limit or not candidate.any():
        return candidate
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~candidate).astype(np.uint8),
        connectivity=12 - connectivity,
    )
    allowed = stats[:, cv2.CC_STAT_AREA] <= limit
    allowed[0] = False
    allowed[np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))] = False
    # An enclosed hole may contain a real defect, but never unselected or invalid reference pixels.
    allowed[np.unique(labels[~eligible])] = False
    assert len(allowed) == count
    return candidate | allowed[labels]


class CompensateReferenceShadows(Block):
    type_id, title = "cv.reference_shadows", "Detect and compensate reference shadows"
    category, block_type, subtype = "Image processing", "Image processing", "Illumination"
    inputs = (
        *SELECTION_INPUTS,
        Port("reference"),
        Port("fit_mask", required=False, required_when_connected=True, visual_type="region"),
    )
    outputs = (
        Port("image"),
        Port("shadow_mask", visual_type="region"),
        Port("invalid_mask", visual_type="region"),
        Port("clipped_mask", visual_type="region"),
        Port("gain_map"),
        Port("valid", "data"),
        Port("measurements", "data"),
    )
    payload_reference = "image"
    parallel_safe = process_inputs_readonly = True
    defaults = {
        "reference_alignment": "pixel_grid",
        "min_reference": 0.01,
        "min_ratio": 0.2,
        "max_ratio": 0.9,
        "max_chroma_error": 0.05,
        "minimum_area": 128,
        "connectivity": 8,
        "fill_holes_area": 0,
        "fit_erosion": 1,
        "minimum_fit_pixels": 32,
        "max_ratio_spread": 0.15,
        "max_gain": 8.0,
        "clip_policy": "reject",
        "maximum_components": 256,
    }
    parameter_choices = {
        "reference_alignment": ("pixel_grid", "same_frame"),
        "connectivity": (4, 8),
        "clip_policy": ("reject", "clip"),
    }
    parameter_ranges = {
        "min_reference": NumericRange(0),
        "min_ratio": NumericRange(0.000001, 0.999999),
        "max_ratio": NumericRange(0.000001, 0.999999),
        "max_chroma_error": NumericRange(0, 1),
        "minimum_area": NumericRange(1, MAX_PIXELS, whole=True),
        "fill_holes_area": NumericRange(0, MAX_PIXELS, whole=True),
        "fit_erosion": NumericRange(0, 32, whole=True),
        "minimum_fit_pixels": NumericRange(1, MAX_PIXELS, whole=True),
        "max_ratio_spread": NumericRange(0, 10),
        "max_gain": NumericRange(1, 64),
        "maximum_components": NumericRange(1, 4096, whole=True),
    }
    parameter_help = {
        "reference_alignment": "Pixel-aligned captures may have different roots/tickets. "
        "Same frame additionally checks known coordinate grids; no registration is performed.",
        "min_reference": "Minimum reference mean: integer full-scale fraction, or float raw units.",
        "min_ratio": "Minimum current/reference mean ratio for candidate shadow pixels.",
        "max_ratio": "Maximum attenuation ratio; must exceed minimum and be below one.",
        "max_chroma_error": "Maximum absolute channel-fraction difference; ignored for gray.",
        "minimum_area": "Discard smaller candidate components as possible local defects.",
        "connectivity": "4 or 8 connected shadow neighbors; hole topology uses the complement.",
        "fill_holes_area": "Fill enclosed holes up to this area, including possible defects. "
        "Zero disables it. Cannot cross selection or invalid reference pixels.",
        "fit_erosion": "Erode component fit support by this radius, not the output mask.",
        "minimum_fit_pixels": "Minimum original candidate pixels in eroded support and Fit mask.",
        "max_ratio_spread": "Reject if (90th-10th percentile)/median ratio exceeds this limit.",
        "max_gain": "Reject a component needing a larger reciprocal-median gain.",
        "clip_policy": "Reject withholds Image/Gain if correction exceeds output depth. "
        "Clip explicitly permits saturation and reports the affected pixels.",
        "maximum_components": "Bound retained shadow components and per-component reports.",
    }
    input_help = {
        "reference": "Aligned unshadowed reference, matching intensity encoding/depth; alpha "
        "may be omitted. Changes in material or geometry can be confused with shadows.",
        "fit_mask": "Optional uint8 same-grid mask of trusted unchanged material for gain fitting. "
        "It limits estimation only; correction still covers the complete detected region.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["min_ratio"] >= p["max_ratio"]:
            raise ValidationError("Minimum ratio must be below maximum ratio", "min_ratio")

    def process(self, inputs):
        p = self.parameters
        source, reference = inputs["image"], inputs["reference"]
        encoding = image_format(source)
        image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64))
        ref = image_value(reference, (image.dtype,))
        height, width = image.shape[:2]
        if not height or not width or height * width > MAX_PIXELS:
            raise ValidationError(f"Shadow correction requires 1..{MAX_PIXELS} pixels")
        if encoding not in INTENSITY_FORMATS:
            raise ValidationError("Use grayscale or RGB/BGR intensity images")
        alpha = encoding in ("rgba", "bgra")
        if alpha and any(
            metadata(value).get("alpha_representation", "straight") != "straight"
            for value in (source, reference)
        ):
            raise ValidationError("Unassociate premultiplied images before shadow correction")
        if not (
            (ref.shape == image.shape and image_format(reference) == encoding)
            or (
                alpha
                and ref.shape == image.shape[:2] + (3,)
                and image_format(reference) == encoding.removesuffix("a")
            )
        ):
            raise ValidationError("Reference must match image dimensions, encoding and depth")
        if p["reference_alignment"] == "same_frame":
            require_same_frame((source, reference), message="Reference coordinate frame differs")
        values = (image[..., :3] if alpha else image).astype(np.float64)
        ref = (ref[..., :3] if alpha else ref).astype(np.float64)
        if np.any(values < 0) or np.any(ref < 0):
            raise ValidationError("Shadow modeling requires nonnegative intensity samples")
        mask = selection_mask(inputs, source)
        selected = mask != 0 if mask is not None else np.ones((height, width), bool)
        fit_mask = None
        if "fit_mask" in inputs:
            fit_mask = selection_mask({"image": source, "mask": inputs["fit_mask"]}, source) != 0
        current_mean, current_chroma = _brightness_chroma(values)
        ref_mean, ref_chroma = _brightness_chroma(ref)
        floor = float(p["min_reference"])
        if image.dtype.kind == "u":
            if floor > 1:
                raise ValidationError(
                    "Integer minimum reference must be between zero and one", "min_reference"
                )
            floor *= np.iinfo(image.dtype).max
        eligible = (ref_mean > 0) & (ref_mean >= floor)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            ratios = np.divide(current_mean, ref_mean, out=np.zeros_like(ref_mean), where=eligible)
        eligible &= np.isfinite(ratios)
        # A numerically underflowed nonzero ratio is not valid zero evidence.
        eligible &= (ratios != 0) | (current_mean == 0)
        if values.ndim == 3:
            eligible &= (current_mean != 0) | ~np.any(values != 0, axis=2)
        invalid = selected & ~eligible
        candidate = selected & eligible & (ratios >= p["min_ratio"]) & (ratios <= p["max_ratio"])
        if current_chroma is not None:
            candidate &= (
                np.max(np.abs(current_chroma - ref_chroma), axis=2) <= p["max_chroma_error"]
            )
        original_candidate = candidate.copy()
        candidate = _fill_small_holes(
            candidate, selected & eligible, int(p["fill_holes_area"]), int(p["connectivity"])
        )
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=int(p["connectivity"]),
        )
        retained = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= p["minimum_area"])
        retained = retained[retained != 0]
        if len(retained) > p["maximum_components"]:
            raise ValidationError("Too many shadow components; tighten selection or minimum area")
        lookup = np.zeros(len(stats), bool)
        lookup[retained] = True
        shadow = lookup[labels]
        output = image.copy()
        destination = output[..., :3] if alpha else output
        gain_map = np.ones((height, width), np.float64)
        clipped = np.zeros((height, width), bool)
        rows, drawings, work = [], [], 0
        radius = int(p["fit_erosion"])
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        maximum = (
            np.iinfo(image.dtype).max if image.dtype.kind == "u" else np.finfo(image.dtype).max
        )
        for label in retained:
            x, y, w, h, area = map(int, stats[label])
            work += w * h
            if work > MAX_ROI_WORK:
                raise ValidationError("Shadow bounding boxes exceed the bounded work budget")
            roi = np.s_[y : y + h, x : x + w]
            component = labels[roi] == label
            support = (
                cv2.erode(
                    component.astype(np.uint8),
                    kernel,
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                != 0
                if radius
                else component.copy()
            )
            support &= original_candidate[roi]
            if fit_mask is not None:
                support &= fit_mask[roi]
            fit = ratios[roi][support]
            row = {
                "id": int(label),
                "bbox": [x, y, w, h],
                "area": area,
                "fit_pixels": int(fit.size),
                "ratio": None,
                "gain": None,
                "relative_spread": None,
                "status": "insufficient_fit",
            }
            if fit.size >= p["minimum_fit_pixels"]:
                low, median, high = np.quantile(fit, [0.1, 0.5, 0.9])
                gain, spread = float(1 / median), float((high - low) / median)
                row.update(ratio=float(median), gain=gain, relative_spread=spread)
                row["status"] = (
                    "inconsistent_attenuation"
                    if spread > p["max_ratio_spread"]
                    else "gain_limit"
                    if gain > p["max_gain"]
                    else "accepted"
                )
                if row["status"] == "accepted":
                    with np.errstate(over="ignore", invalid="ignore"):
                        corrected = values[roi][component] * gain
                    saturated = (~np.isfinite(corrected)) | (corrected > maximum)
                    saturated = saturated.any(axis=1) if corrected.ndim == 2 else saturated
                    clipped[roi][component] = saturated
                    if saturated.any() and p["clip_policy"] == "reject":
                        row["status"] = "output_range"
                    else:
                        corrected = np.clip(corrected, 0, maximum)
                        if image.dtype.kind == "u":
                            corrected = np.rint(corrected)
                        destination[roi][component] = corrected
                        gain_map[roi][component] = gain
            accepted = row["status"] == "accepted"
            if not accepted:
                invalid[roi] |= component
            rows.append(row)
            if self.generate_graphics:
                color = (100, 220, 90) if accepted else (70, 80, 240)
                drawings.extend(
                    (
                        graphic_box(x, y, w, h, color),
                        graphic_object(
                            "label",
                            [(x, y)],
                            color,
                            text=f"Shadow gain {row['gain']:.3g}" if accepted else row["status"],
                        ),
                    )
                )
        valid = bool(selected.any() and not invalid.any())
        failures = []
        if not selected.any():
            failures.append({"code": "shadow_empty_selection", "message": "No selected pixels"})
        if np.any(selected & ~eligible):
            failures.append(
                {
                    "code": "shadow_invalid_reference",
                    "message": "Selected reference/ratio pixels are unusable",
                }
            )
        for status in sorted({row["status"] for row in rows} - {"accepted"}):
            count = sum(row["status"] == status for row in rows)
            failures.append(
                {
                    "code": "shadow_" + status,
                    "message": f"{count} shadow component(s): {status}; see Measurements",
                }
            )
        measurements = {
            "model": "reference_shadow_compensation_v1",
            "valid": valid,
            "status": "corrected" if valid and rows else "no_candidates" if valid else "invalid",
            "selected_pixels": int(selected.sum()),
            "candidate_pixels": int(original_candidate.sum()),
            "shadow_pixels": int(shadow.sum()),
            "corrected_pixels": int(shadow.sum()) if valid else 0,
            "filled_pixels": int((shadow & ~original_candidate).sum()),
            "invalid_pixels": int(invalid.sum()),
            "clipped_pixels": int(clipped.sum()),
            "components": rows,
            "reference_alignment": p["reference_alignment"],
            "clip_policy": p["clip_policy"],
            "encoding": encoding,
        }
        payload = copy.deepcopy(self.input_payloads["image"])
        payload.pop("alpha_representation", None)
        return Result(
            {
                "image": ImageArray(output, encoding) if valid else NO_VALUE,
                "shadow_mask": shadow.astype(np.uint8) * 255,
                "invalid_mask": invalid.astype(np.uint8) * 255,
                "clipped_mask": clipped.astype(np.uint8) * 255,
                "gain_map": gain_map if valid else NO_VALUE,
                "valid": valid,
                "measurements": measurements,
            },
            payloads={
                name: copy.deepcopy(payload)
                for name in ("shadow_mask", "invalid_mask", "clipped_mask", "gain_map")
            },
            drawings={"shadow regions": GraphicDrawing(tuple(drawings))} if drawings else {},
            passed=valid,
            fail_reasons=failures,
            message=f"{measurements['status']} · {len(rows)} shadows · "
            f"{measurements['corrected_pixels']} corrected pixels",
        )
