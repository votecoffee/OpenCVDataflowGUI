"""Intensity-based image alignment with explicit forward and inverse transforms."""

import cv2
import numpy as np

from .core import Block, Port
from .processing import finite, gray_image, integer
from .registration import BORDERS, INTERPOLATION, matrix_value, warped
from .selection import SELECTION_INPUTS, selection_mask

MOTIONS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


class AlignECC(Block):
    type_id, title = "inspect.align_ecc", "Align images (ECC)"
    category, block_type, subtype = "Inspection", "Inspection", "Registration"
    inputs = (
        *SELECTION_INPUTS,
        Port("reference"),
        Port("reference_mask", required=False),
        Port("initial_transform", "data", required=False),
    )
    outputs = (
        Port("image"),
        Port("geometry", "data"),
        Port("transform", "data"),
        Port("coefficient", "number"),
    )
    defaults = {
        "motion": "affine",
        "pyramid_levels": 1,
        "max_iterations": 50,
        "iterations_per_level": [],
        "epsilon": 1e-6,
        "gaussian_size": 5,
        "min_coefficient": 0.0,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
    }
    parameter_choices = {
        "motion": tuple(MOTIONS),
        "interpolation": tuple(INTERPOLATION),
        "border": tuple(BORDERS),
    }
    parameter_help = {
        "pyramid_levels": "One for direct ECC, more for OpenCV's multiscale ECC.",
        "max_iterations": "Maximum iterations per pyramid level.",
        "iterations_per_level": "Optional list with one iteration limit per multiscale level.",
        "epsilon": "Stop on coefficient convergence. Zero uses the iteration limit only.",
    }

    def process(self, inputs):
        source, reference = inputs["image"], inputs["reference"]
        gray, target = (
            gray_image(image, (np.uint8, np.uint16, np.float32)) for image in (source, reference)
        )
        if gray.shape != target.shape or min(gray.shape) < 4:
            raise ValueError("ECC requires matching image dimensions of at least 4 by 4")
        moving_mask = selection_mask(inputs, gray)
        reference_mask = selection_mask(
            {"image": reference, "mask": inputs.get("reference_mask")}, target
        )
        if any(mask is not None and not mask.any() for mask in (moving_mask, reference_mask)):
            raise ValueError("ECC masks must each include at least one pixel")
        p = self.parameters
        motion = MOTIONS[p["motion"]]
        levels = integer(p["pyramid_levels"], "Pyramid levels", 1, 8)
        if min(gray.shape) < 4 * 2 ** (levels - 1):
            raise ValueError("ECC pyramid has too many levels for this image size")
        iterations = integer(p["max_iterations"], "Maximum iterations", 1, 10000)
        epsilon = finite(p["epsilon"], "Epsilon", 0)
        criteria = (
            cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if epsilon else 0),
            iterations,
            epsilon,
        )
        gaussian = integer(p["gaussian_size"], "Gaussian size", 1, 99, odd=True)
        minimum = finite(p["min_coefficient"], "Minimum coefficient")
        if not -1 <= minimum <= 1:
            raise ValueError("Minimum coefficient must be between -1 and 1")
        shape = (3, 3) if motion == cv2.MOTION_HOMOGRAPHY else (2, 3)
        initial = inputs.get("initial_transform")
        if initial is None:
            inverse = np.eye(3)
        else:
            inverse = np.linalg.inv(matrix_value(initial, (3, 3)))
            inverse = matrix_value(inverse, shape)
        guess = inverse[: shape[0]].astype(np.float32)
        target, gray = target.astype(np.float32), gray.astype(np.float32)
        if levels > 1:
            settings = cv2.ECCParameters()
            settings.motionType = motion
            settings.criteria = criteria
            settings.gaussFiltSize = gaussian
            settings.nlevels = levels
            limits = p["iterations_per_level"]
            if limits:
                if len(limits) != levels:
                    raise ValueError("Provide one iteration limit per ECC pyramid level")
                settings.itersPerLevel = tuple(
                    integer(value, "Level iteration limit", 1, 10000) for value in limits
                )
            settings.interpolation = INTERPOLATION[p["interpolation"]]
            coefficient, matrix = cv2.findTransformECCMultiScale(
                target, gray, guess, settings, reference_mask, moving_mask
            )
        else:
            coefficient, matrix = cv2.findTransformECCWithMask(
                target, gray, reference_mask, moving_mask, guess, motion, criteria, gaussian
            )
        matrix = matrix_value(matrix, shape)
        if not np.isfinite(coefficient):
            raise ValueError("ECC produced a nonfinite coefficient")
        result = warped(
            source,
            matrix,
            tuple(target.shape[::-1]),
            p["interpolation"],
            p["border"],
            finite(p["border_value"], "Border value"),
            inverse_map=True,
        )
        geometry = result.outputs["geometry"]
        result.outputs.update(
            transform={
                "model": "image_ecc_v1",
                "source_to_target": np.linalg.inv(geometry["output_to_input"]).tolist(),
                "source_size": geometry["input_size"],
                "target_size": geometry["output_size"],
            },
            coefficient=coefficient,
        )
        result.passed = coefficient >= minimum
        result.score = float(np.clip(coefficient, 0, 1))
        result.message = f"ECC coefficient {coefficient:.6g}"
        if not result.passed:
            result.fail_reasons = ["ecc_below_minimum"]
        return result
