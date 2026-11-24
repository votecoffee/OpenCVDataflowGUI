"""Pei–Lin affine normalization with source-grid geometry and moment checks."""

import cv2
import numpy as np

from .core import Port
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image, image_value
from .registration import BORDERS, INTERPOLATION, output_size, warped
from .selection import selection_mask

_GRAY_DEPTHS = (np.uint8, np.uint16, np.float32)


def _fit_moments(weights):
    m = cv2.moments(weights)
    mass = m["m00"]
    if not mass > 0:
        raise ValidationError("Normalization needs nonzero foreground support")
    center = np.array([m["m10"], m["m01"]]) / mass
    covariance = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / mass
    eigenvalues, axes = np.linalg.eigh(covariance)
    if eigenvalues[0] <= max(1e-10, eigenvalues[1] * 1e-9):
        raise ValidationError("Normalization needs two-dimensional foreground support")
    whitener = (axes / np.sqrt(eigenvalues)) @ axes.T
    inverse = whitener.T @ whitener
    terms = [inverse[0, 0], 2 * inverse[0, 1], inverse[1, 1]]
    # Contract the third central moments with inverse covariance before whitening.
    skew = (
        np.array([[m["mu30"], m["mu21"], m["mu12"]], [m["mu21"], m["mu12"], m["mu03"]]])
        @ terms
        / mass
    )
    direction = whitener @ skew
    return center, covariance, whitener, direction


class PeiLinNormalization(ImageTool):
    type_id, title, subtype = "cv.peilin_normalization", "Normalize shape (Pei–Lin)", "Geometry"
    description = "Normalize a shape's position, spread and orientation from its spatial moments."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("transform", "data"),
        Port("geometry", "data"),
        Port("report", "data"),
    )
    defaults = {
        "weight_source": "intensity",
        "threshold": 0.5,
        "polarity": "bright",
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
    }
    parameter_choices = {
        "weight_source": ("intensity", "threshold", "selection"),
        "polarity": ("bright", "dark"),
        "interpolation": tuple(INTERPOLATION),
        "border": tuple(BORDERS),
    }
    parameter_ranges = {
        "threshold": NumericRange(0, 1),
        "width": NumericRange(8, 16384, whole=True, zero_allowed=True),
        "height": NumericRange(8, 16384, whole=True, zero_allowed=True),
    }
    parameter_help = {
        "weight_source": "Fit from nonnegative grayscale intensity, a binary threshold, or "
        "the wired mask and region treated as a binary silhouette.",
        "threshold": "Threshold relative to full uint8/uint16 scale or in float [0, 1] units.",
        "polarity": "Threshold mode selects samples at or above the cutoff for bright foreground "
        "or below it for dark foreground.",
        "width": "Output canvas width. Zero uses the source width.",
        "height": "Output canvas height. Zero uses the source height.",
        "interpolation": "Sampling method when warping the image.",
        "border": "Sampling rule outside the source grid.",
        "border_value": "Exterior sample value for constant borders, in source units.",
    }
    input_help = {
        "image": "Warp the original encoding and depth. Intensity fitting uses grayscale.",
        "mask": "Restrict the moment estimate. Selection mode fits the mask as the silhouette.",
        "region": "Restrict the moment estimate and intersect with any wired mask.",
    }
    output_help = {
        "transform": "Source-to-normalized affine transform and inverse for points or image warps.",
        "geometry": "Output-to-input coordinate mapping on the source pixel grid.",
        "report": "Estimator, foreground sample count, and whether orientation is resolved.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def _weights(self, source, selection):
        p = self.parameters
        if p["weight_source"] == "selection":
            if selection is None:
                raise ValidationError(
                    "Connect a mask or region to fit a silhouette", "weight_source"
                )
            return selection
        gray = gray_image(source, _GRAY_DEPTHS)
        if p["weight_source"] == "threshold":
            peak = np.iinfo(gray.dtype).max if gray.dtype.kind == "u" else 1
            cutoff = float(p["threshold"]) * peak
            kept = gray >= cutoff if p["polarity"] == "bright" else gray < cutoff
            weights = kept.astype(np.uint8)
        else:
            weights = gray if selection is None else np.where(selection != 0, gray, 0)
            if weights.min() < 0:
                raise ValidationError(
                    "Intensity fitting requires nonnegative samples", "weight_source"
                )
            if weights.dtype.kind == "f" and weights.max() > 0:
                # A common rescale keeps floating-point moments away from numeric extremes.
                weights = weights / weights.max()
            return np.ascontiguousarray(weights)
        if selection is not None:
            weights[selection == 0] = 0
        return weights

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        height, width = image.shape[:2]
        if min(height, width) < 8 or height * width > 16_000_000:
            raise ValidationError(
                "Pei–Lin normalization requires at least 8×8 and up to 16M pixels"
            )
        native = getattr(getattr(cv2, "ximgproc", None), "PeiLinNormalization", None)
        if native is None:
            raise ValidationError("Pei–Lin normalization requires OpenCV contrib ximgproc")
        p = self.parameters
        target = output_size(p, image)
        weights = self._weights(source, selection_mask(inputs, source))
        center, covariance, whitener, direction = _fit_moments(weights)
        strength = float(np.linalg.norm(direction))
        oriented = strength > 1e-6
        spread = min(height, width) // 8
        target_spread = min(target) // 8
        target_center = np.array(target) // 2
        candidate = native(weights)
        estimator = "opencv"
        if (
            oriented
            and candidate is not None
            and candidate.shape == (2, 3)
            and np.isfinite(candidate).all()
            and np.allclose(
                candidate[:, :2] @ covariance @ candidate[:, :2].T / spread**2,
                np.eye(2),
                atol=1e-6,
            )
        ):
            if target == (width, height):
                matrix = candidate
            else:
                linear = candidate[:, :2] * (target_spread / spread)
                matrix = np.column_stack((linear, target_center - linear @ center))
        else:
            # OpenCV's eigenvector formula can divide by zero on axis-aligned shapes.
            # Symmetric whitening also gives a stable axis choice when skew has no direction.
            angle = np.arctan2(-direction[0], direction[1]) if oriented else 0
            cos, sin = np.cos(angle), np.sin(angle)
            linear = target_spread * np.array([[cos, sin], [-sin, cos]]) @ whitener
            matrix = np.column_stack((linear, target_center - linear @ center))
            estimator = "stable_moments"
        result = warped(source, matrix, target, p["interpolation"], p["border"], p["border_value"])
        result.outputs["transform"] = {
            "model": "image_affine_v1",
            "source_to_target": matrix.tolist(),
            "target_to_source": result.outputs["geometry"]["output_to_input"],
            "source_size": [width, height],
            "target_size": list(target),
        }
        result.outputs["report"] = {
            "estimator": estimator,
            "weight_source": p["weight_source"],
            "support_pixels": int(np.count_nonzero(weights)),
            "source_center": center.tolist(),
            "orientation_resolved": bool(oriented),
            "orientation_strength": strength,
            "target_spread": target_spread,
        }
        result.message = (
            "Shape normalized" if oriented else "Shape normalized with ambiguous orientation"
        )
        return result
