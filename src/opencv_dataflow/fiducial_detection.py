"""Shared ArUco dictionary choices and marker detector settings."""

import cv2
import numpy as np

from .aruco_dictionary import DICTIONARIES as DICTIONARIES
from .aruco_dictionary import DICTIONARY_CHOICES, MAX_MARKERS
from .parameter_validation import NumericRange, ValidationError
from .processing import finite, integer

_CORNER_REFINEMENT = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpixel": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}
MARKER_OPTIONS = {
    "corner_refinement": "subpixel",
    "adaptive_min": 3,
    "adaptive_max": 23,
    "adaptive_step": 10,
    "adaptive_constant": 7.0,
    "min_perimeter_rate": 0.03,
    "max_perimeter_rate": 4.0,
    "min_border_distance": 3,
    "border_bits": 1,
    "error_correction_rate": 0.6,
}
MARKER_RANGES = {
    "adaptive_min": NumericRange(3, 255, whole=True),
    "adaptive_max": NumericRange(3, 1000, whole=True),
    "adaptive_step": NumericRange(1, 100, whole=True),
    "adaptive_constant": NumericRange(),
    "min_perimeter_rate": NumericRange(0.0001),
    "max_perimeter_rate": NumericRange(0.0001),
    "min_border_distance": NumericRange(0, 1000, whole=True),
    "border_bits": NumericRange(1, 10, whole=True),
    "error_correction_rate": NumericRange(0, 1),
}
MARKER_CHOICES = {
    "dictionary": DICTIONARY_CHOICES,
    "corner_refinement": tuple(_CORNER_REFINEMENT),
}


def board_marker_ids(ids, count, dictionary):
    capacity = MAX_MARKERS if dictionary is None else len(dictionary.bytesList)
    if count > capacity:
        raise ValidationError("The dictionary has fewer IDs than board markers", "dictionary")
    if not isinstance(ids, list) or (
        ids
        and (
            len(ids) != count
            or any(type(i) is not int or not 0 <= i < capacity for i in ids)
            or len(set(ids)) != count
        )
    ):
        raise ValidationError(
            f"Provide {count} unique marker IDs in 0..{capacity - 1}, or an empty list",
            "marker_ids",
        )
    return np.asarray(ids or range(count), dtype=np.int32)


def detector_parameters(p):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = _CORNER_REFINEMENT[p["corner_refinement"]]
    parameters.adaptiveThreshWinSizeMin = integer(p["adaptive_min"], "Adaptive min", 3, 255)
    parameters.adaptiveThreshWinSizeMax = integer(
        p["adaptive_max"], "Adaptive max", parameters.adaptiveThreshWinSizeMin, 1000
    )
    parameters.adaptiveThreshWinSizeStep = integer(p["adaptive_step"], "Adaptive step", 1, 100)
    parameters.adaptiveThreshConstant = finite(p["adaptive_constant"], "Adaptive constant")
    parameters.minMarkerPerimeterRate = finite(
        p["min_perimeter_rate"], "Min perimeter rate", 0.0001
    )
    parameters.maxMarkerPerimeterRate = finite(
        p["max_perimeter_rate"], "Max perimeter rate", parameters.minMarkerPerimeterRate
    )
    parameters.minDistanceToBorder = integer(p["min_border_distance"], "Border distance", 0, 1000)
    parameters.markerBorderBits = integer(p["border_bits"], "Border bits", 1, 10)
    parameters.errorCorrectionRate = finite(p["error_correction_rate"], "Error correction rate", 0)
    if parameters.errorCorrectionRate > 1:
        raise ValueError("Error correction rate must be in [0,1]")
    return parameters
