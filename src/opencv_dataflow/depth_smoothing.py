"""Confidence-weighted bilateral filtering of physical depth on its original grid."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D
from .parameter_validation import NumericRange, ValidationError
from .payloads import require_same_frame
from .selection import selection_mask
from .tickets import ticket_key


class SmoothDepth(Tool3D):
    type_id, title, subtype = "3d.smooth_depth", "Smooth depth while preserving edges", "Depth"
    payload_reference = "depth"
    parallel_safe = True
    inputs = (
        Port("depth", "depth"),
        Port("confidence", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("depth", "depth"),
        Port("valid_mask", visual_type="region"),
        Port("rejected_mask", visual_type="region"),
        Port("report", "data"),
    )
    defaults = {
        "radius": 4,
        "sigma_space_px": 2.5,
        "sigma_depth_m": 0.08,
        "minimum_confidence": 0.25,
        "border": "reflect_101",
        "input_units": "m",
        "units_per_sample": 0.001,
    }
    parameter_choices = {
        "border": ("reflect_101", "reflect", "replicate"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        "radius": NumericRange(1, 32, whole=True),
        "sigma_space_px": NumericRange(0.01, 100),
        "sigma_depth_m": NumericRange(0.000001, 1000),
        "minimum_confidence": NumericRange(0, 1),
        "units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "radius": "Neighborhood radius in pixels, giving a diameter of 2 × radius + 1.",
        "sigma_space_px": "Spatial Gaussian width in pixels.",
        "sigma_depth_m": "Depth-difference Gaussian width in meters regardless of input units.",
        "minimum_confidence": "Reject lower confidence. Zero confidence is always invalid.",
        "border": "How to extend neighborhoods at the image edges.",
        "input_units": "Physical units for an input without declared depth units.",
        "units_per_sample": "Physical units per uint16 count.",
    }
    input_help = {
        "confidence": (
            "Aligned monochrome float in [0, 1] or uint8/uint16 full scale. "
            "Positive confidence weights neighbors."
        ),
        "mask": (
            "Limit writes to selected pixels. Trusted depth outside the selection can contribute."
        ),
        "region": "Limit writes to a region, intersecting it with the mask if supplied.",
    }

    def confidence_weights(self, source, confidence):
        if confidence.shape != source.shape:
            raise ValidationError("Confidence must be monochrome with the same dimensions as depth")
        require_same_frame(
            (source, confidence), message="Confidence and depth use different coordinate frames"
        )
        depth_ticket = self.input_payloads["depth"].get("ticket")
        confidence_ticket = self.input_payloads["confidence"].get("ticket")
        if (
            depth_ticket
            and confidence_ticket
            and ticket_key(depth_ticket) != ticket_key(confidence_ticket)
        ):
            raise ValidationError("Confidence and depth must refer to the same ticket")
        if confidence.dtype in (np.uint8, np.uint16):
            return np.asarray(confidence, np.float32) / np.iinfo(confidence.dtype).max
        values = np.asarray(confidence)
        if (
            values.dtype not in (np.float32, np.float64)
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValidationError(
                "Confidence must be finite float in [0, 1] or uint8/uint16 full scale"
            )
        return np.asarray(values, np.float32)

    def process(self, inputs):
        source = inputs["depth"]
        values = np.asarray(source)
        p = self.parameters
        units = getattr(source, "units", p["input_units"])
        if units not in ("m", "mm"):
            raise ValidationError("Depth units must be m or mm", "input_units")
        integer_depth = values.dtype == np.uint16
        meters_per_sample = (0.001 if units == "mm" else 1.0) * (
            float(p["units_per_sample"]) if integer_depth else 1.0
        )
        valid = np.isfinite(values) & (values > 0)
        selection = selection_mask(inputs, source)
        selected = np.ones(values.shape, bool) if selection is None else selection != 0
        trusted = valid.copy()
        confidence = inputs.get("confidence")
        if confidence is None:
            weights = trusted.astype(np.float32)
        else:
            weights = self.confidence_weights(source, confidence)
            trusted &= (weights > 0) & (weights >= float(p["minimum_confidence"]))
            # Keep a positive center weight representable through native normalization.
            weights = np.where(trusted, np.maximum(weights, np.finfo(np.float32).tiny), 0)

        output = source.copy()
        rejected = selected & valid & ~trusted
        output[rejected] = 0 if integer_depth else np.nan
        targets = selected & trusted
        count = int(targets.sum())
        if count:
            samples = values[trusted].astype(np.float64)
            minimum, maximum = samples.min(), samples.max()
            # Center the guide and express it in sigma units. Native float filtering uses
            # 4096 lookup intervals across the guide range. Keep at least one per sigma.
            sigma_depth = float(p["sigma_depth_m"]) / meters_per_sample
            if maximum - minimum > 4096 * sigma_depth:
                raise ValidationError(
                    "Trusted depth spans more than 4096 depth sigmas; discard outliers "
                    "using confidence or increase depth sigma",
                    "sigma_depth_m",
                )
            guide = np.zeros(values.shape, np.float32)
            guide[trusted] = (samples - minimum) / sigma_depth
            border = {
                "reflect_101": cv2.BORDER_REFLECT_101,
                "reflect": cv2.BORDER_REFLECT,
                "replicate": cv2.BORDER_REPLICATE,
            }[p["border"]]
            if not hasattr(getattr(cv2, "ximgproc", None), "jointBilateralFilter"):
                raise ValidationError(
                    "Depth smoothing requires OpenCV contrib jointBilateralFilter"
                )
            diameter = 2 * int(p["radius"]) + 1
            sigma_space = float(p["sigma_space_px"])
            weighted = cv2.ximgproc.jointBilateralFilter(
                guide, guide * weights, diameter, 1.0, sigma_space, borderType=border
            )
            support = cv2.ximgproc.jointBilateralFilter(
                guide, weights, diameter, 1.0, sigma_space, borderType=border
            )
            # The shared kernel's normalization cancels. Excluded samples have zero weight.
            offsets = weighted[targets].astype(np.float64) / support[targets]
            filtered = minimum + offsets * sigma_depth
            np.clip(filtered, minimum, maximum, out=filtered)
            output[targets] = np.rint(filtered) if integer_depth else filtered

        selected_count = int(selected.sum())
        rejected_count = int(rejected.sum())
        return Result(
            {
                "depth": output,
                "valid_mask": (valid & ~rejected).astype(np.uint8) * 255,
                "rejected_mask": rejected.astype(np.uint8) * 255,
                "report": {
                    "selected_samples": selected_count,
                    "smoothed_samples": count,
                    "invalid_input_samples": selected_count - count - rejected_count,
                    "confidence_rejected_samples": rejected_count,
                    "units": units,
                    "meters_per_sample": meters_per_sample,
                },
            },
            passed=bool(count),
            fail_reasons=[] if count else ["no_trusted_depth_in_selection"],
            score=count / selected_count if selected_count else 0.0,
            message=f"Smoothed {count} depth pixels · {rejected_count} rejected by confidence",
        )
