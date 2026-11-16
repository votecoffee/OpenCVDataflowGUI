"""Explicit QR/ArUco detector controls, independent of the reader's result contract."""

import math

import cv2

from .parameter_validation import NumericRange, ValidationError

# Each row is (native property, default, range, help). Explicit defaults keep saved
# programs stable across library updates; native objects are created per invocation.
QR_OPTIONS = {
    "qr_min_module_size": (
        "minModuleSizeInPyramid",
        4.0,
        NumericRange(1, 128),
        "ArUco QR: smallest pyramid module size in pixels.",
    ),
    "qr_max_rotation_degrees": (
        "maxRotation",
        15.0,
        NumericRange(0.01, 90),
        "ArUco QR: relative finder-pattern rotation tolerance in degrees; not label angle.",
    ),
    "qr_max_module_size_mismatch": (
        "maxModuleSizeMismatch",
        1.75,
        NumericRange(1, 100),
        "ArUco QR: allowed ratio between finder-pattern module sizes.",
    ),
    "qr_max_timing_pattern_mismatch": (
        "maxTimingPatternMismatch",
        2.0,
        NumericRange(0.0001, 100),
        "ArUco QR: timing-module size mismatch tolerance.",
    ),
    "qr_max_penalties": (
        "maxPenalties",
        0.4,
        NumericRange(0, 1),
        "ArUco QR: permitted fraction of timing-pattern penalty points.",
    ),
    "qr_max_colors_mismatch": (
        "maxColorsMismatch",
        0.2,
        NumericRange(0, 1),
        "ArUco QR: relative timing-pattern intensity mismatch tolerance.",
    ),
    "qr_scale_timing_pattern_score": (
        "scaleTimingPatternScore",
        0.9,
        NumericRange(0.0001, 1),
        "ArUco QR: timing-score scale used to select the finder-pattern hypothesis.",
    ),
}
ARUCO_OPTIONS = {
    "aruco_adaptive_min": (
        "adaptiveThreshWinSizeMin",
        3,
        NumericRange(3, 1001, whole=True, odd=True),
        "Smallest adaptive threshold window, pixels.",
    ),
    "aruco_adaptive_max": (
        "adaptiveThreshWinSizeMax",
        23,
        NumericRange(3, 1001, whole=True, odd=True),
        "Largest adaptive threshold window, pixels; at least the minimum.",
    ),
    "aruco_adaptive_step": (
        "adaptiveThreshWinSizeStep",
        10,
        NumericRange(1, 1000, whole=True),
        "Threshold window step; at most 128 window scales per invocation.",
    ),
    "aruco_adaptive_constant": (
        "adaptiveThreshConstant",
        7.0,
        NumericRange(-255, 255),
        "Adaptive threshold subtraction constant in uint8 intensity units.",
    ),
    "aruco_min_perimeter_rate": (
        "minMarkerPerimeterRate",
        0.02,
        NumericRange(0.000001, 16),
        "Finder perimeter divided by the image's largest dimension.",
    ),
    "aruco_max_perimeter_rate": (
        "maxMarkerPerimeterRate",
        4.0,
        NumericRange(0.000001, 16),
        "Largest permitted finder perimeter ratio; at least the minimum.",
    ),
    "aruco_polygon_accuracy": (
        "polygonalApproxAccuracyRate",
        0.03,
        NumericRange(0.000001, 1),
        "Polygon approximation error as a fraction of contour perimeter.",
    ),
    "aruco_min_corner_distance": (
        "minCornerDistanceRate",
        0.05,
        NumericRange(0, 1),
        "Smallest corner separation relative to finder perimeter.",
    ),
    "aruco_min_border_distance": (
        "minDistanceToBorder",
        3,
        NumericRange(0, 16384, whole=True),
        "Finder-corner exclusion distance from image border, pixels.",
    ),
    "aruco_min_marker_distance": (
        "minMarkerDistanceRate",
        0.125,
        NumericRange(0, 1),
        "Close-candidate grouping distance relative to finder perimeter.",
    ),
    "aruco_min_group_distance": (
        "minGroupDistance",
        0.21,
        NumericRange(0, 10),
        "Candidate group separation measured in module-size units.",
    ),
    "aruco_refinement_window": (
        "cornerRefinementWinSize",
        5,
        NumericRange(1, 255, whole=True),
        "Corner refinement half-window, pixels.",
    ),
    "aruco_relative_refinement_window": (
        "relativeCornerRefinmentWinSize",
        0.3,
        NumericRange(0.0001, 10),
        "Refinement window relative to estimated module size.",
    ),
    "aruco_refinement_iterations": (
        "cornerRefinementMaxIterations",
        30,
        NumericRange(1, 1000, whole=True),
        "Maximum corner-refinement iterations.",
    ),
    "aruco_refinement_accuracy": (
        "cornerRefinementMinAccuracy",
        0.1,
        NumericRange(0.000001, 100),
        "Corner-refinement stopping displacement, pixels.",
    ),
    "aruco_border_bits": (
        "markerBorderBits",
        1,
        NumericRange(1, 16, whole=True),
        "Finder marker border width in modules; QR finders normally require one.",
    ),
    "aruco_pixels_per_cell": (
        "perspectiveRemovePixelPerCell",
        4,
        NumericRange(1, 64, whole=True),
        "Canonical pixels per finder module for identification.",
    ),
    "aruco_cell_margin": (
        "perspectiveRemoveIgnoredMarginPerCell",
        0.13,
        NumericRange(0, 0.49),
        "Fraction ignored on each side of a canonical module; below one half.",
    ),
    "aruco_max_border_error": (
        "maxErroneousBitsInBorderRate",
        0.35,
        NumericRange(0, 1),
        "Allowed erroneous border-bit fraction.",
    ),
    "aruco_min_otsu_stddev": (
        "minOtsuStdDev",
        5.0,
        NumericRange(0, 255),
        "Minimum canonical intensity deviation for Otsu thresholding.",
    ),
    "aruco_error_correction": (
        "errorCorrectionRate",
        0.6,
        NumericRange(0, 1),
        "Fraction of the internal finder dictionary's correction capacity.",
    ),
    "aruco_valid_bit_threshold": (
        "validBitIdThreshold",
        0.49,
        NumericRange(0, 1),
        "Native valid-bit identification threshold (OpenCV 5).",
    ),
    "aruco_apriltag_decimate": (
        "aprilTagQuadDecimate",
        0.0,
        NumericRange(0, 16),
        "AprilTag candidate extraction only: quad decimation, zero disables.",
    ),
    "aruco_apriltag_sigma": (
        "aprilTagQuadSigma",
        0.0,
        NumericRange(0, 16),
        "AprilTag candidate extraction only: blur sigma.",
    ),
    "aruco_apriltag_min_cluster": (
        "aprilTagMinClusterPixels",
        5,
        NumericRange(1, 100000, whole=True),
        "AprilTag candidate extraction only: smallest connected cluster.",
    ),
    "aruco_apriltag_max_maxima": (
        "aprilTagMaxNmaxima",
        10,
        NumericRange(4, 128, whole=True),
        "AprilTag candidate extraction only: maximum corner maxima.",
    ),
    "aruco_apriltag_critical_degrees": (
        "aprilTagCriticalRad",
        10.0,
        NumericRange(0, 89.9),
        "AprilTag candidate extraction only: critical angle, degrees.",
    ),
    "aruco_apriltag_max_line_error": (
        "aprilTagMaxLineFitMse",
        10.0,
        NumericRange(0.000001, 1000000),
        "AprilTag candidate extraction only: largest line-fit mean square error.",
    ),
    "aruco_apriltag_min_contrast": (
        "aprilTagMinWhiteBlackDiff",
        5,
        NumericRange(0, 255, whole=True),
        "AprilTag candidate extraction only: minimum white/black contrast.",
    ),
    "aruco_min_canonical_side": (
        "minSideLengthCanonicalImg",
        32,
        NumericRange(1, 4096, whole=True),
        "ArUco3 only: smallest canonical finder side, pixels.",
    ),
    "aruco_min_length_ratio": (
        "minMarkerLengthRatioOriginalImg",
        0.0,
        NumericRange(0, 1),
        "ArUco3 only: smallest finder length relative to original image size.",
    ),
}
REFINEMENT = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpixel": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}
ARUCO_FLAGS = {
    "aruco_detect_inverted": ("detectInvertedMarker", False),
    "aruco_use_aruco3": ("useAruco3Detection", False),
    "aruco_apriltag_deglitch": ("aprilTagDeglitch", False),
}
OPTIONS = {**QR_OPTIONS, **ARUCO_OPTIONS}
DEFAULTS = {
    "backend": "standard",
    **{key: value[1] for key, value in OPTIONS.items()},
    "aruco_corner_refinement": "none",
    **{key: value[1] for key, value in ARUCO_FLAGS.items()},
    "maximum_pixels": 16777216,
}
RANGES = {key: value[2] for key, value in OPTIONS.items()}
RANGES["maximum_pixels"] = NumericRange(441, 67108864, whole=True)
HELP = {
    key: ("ArUco backend only. " if key in ARUCO_OPTIONS else "") + value[3]
    for key, value in OPTIONS.items()
}
HELP.update(
    {
        "backend": (
            "Standard preserves existing projects; ArUco uses finder-marker candidates. "
            "No automatic fallback."
        ),
        "aruco_corner_refinement": (
            "ArUco finder refinement; AprilTag selects its candidate extractor. "
            "ArUco3 requires subpixel."
        ),
        "aruco_use_aruco3": (
            "ArUco backend only: native multi-scale candidate extraction; use subpixel refinement."
        ),
        "aruco_detect_inverted": (
            "ArUco backend only: inverted finder markers; no guarantee of inverted QR decoding."
        ),
        "aruco_apriltag_deglitch": (
            "AprilTag candidate extraction only: remove isolated threshold artifacts."
        ),
        "maximum_pixels": (
            "Reject larger images before grayscale/selection/native decoding; "
            "not a native workspace or time bound."
        ),
    }
)


def validate_qr(parameters):
    p = parameters
    if p["backend"] == "aruco" and p["mode"] == "curved":
        raise ValidationError(
            "ArUco QR supports single or multi mode, not curved", "backend", "mode"
        )
    if p["aruco_adaptive_min"] > p["aruco_adaptive_max"]:
        raise ValidationError(
            "Threshold minimum exceeds maximum", "aruco_adaptive_min", "aruco_adaptive_max"
        )
    if (p["aruco_adaptive_max"] - p["aruco_adaptive_min"]) // p["aruco_adaptive_step"] + 1 > 128:
        raise ValidationError("Use at most 128 threshold windows", "aruco_adaptive_step")
    if p["aruco_min_perimeter_rate"] > p["aruco_max_perimeter_rate"]:
        raise ValidationError(
            "Finder minimum perimeter exceeds maximum",
            "aruco_min_perimeter_rate",
            "aruco_max_perimeter_rate",
        )
    if p["aruco_use_aruco3"] and p["aruco_corner_refinement"] != "subpixel":
        raise ValidationError(
            "ArUco3 requires subpixel refinement", "aruco_use_aruco3", "aruco_corner_refinement"
        )
    for name in ARUCO_FLAGS:
        if not isinstance(p[name], bool):
            raise ValidationError("Use a checkbox boolean", name)


def make_detector(parameters):
    p = parameters
    if p["backend"] == "standard":
        detector = cv2.QRCodeDetector()
        detector.setEpsX(float(p["eps_x"]))
        detector.setEpsY(float(p["eps_y"]))
        detector.setUseAlignmentMarkers(p["use_alignment_markers"])
        return detector
    if not hasattr(cv2, "QRCodeDetectorAruco"):
        raise ValidationError("This OpenCV build has no ArUco QR detector", "backend")
    detector = cv2.QRCodeDetectorAruco()
    qr, aruco = detector.getDetectorParameters(), detector.getArucoParameters()
    for options, native in ((QR_OPTIONS, qr), (ARUCO_OPTIONS, aruco)):
        for name, (attribute, default, _, _) in options.items():
            if not hasattr(native, attribute):
                raise ValidationError(f"This OpenCV build lacks {attribute}", name)
            value = int(p[name]) if type(default) is int else float(p[name])
            if name in ("qr_max_rotation_degrees", "aruco_apriltag_critical_degrees"):
                value = math.radians(value)
            setattr(native, attribute, value)
    aruco.cornerRefinementMethod = REFINEMENT[p["aruco_corner_refinement"]]
    for name, (attribute, _) in ARUCO_FLAGS.items():
        setattr(aruco, attribute, int(p[name]) if name == "aruco_apriltag_deglitch" else p[name])
    detector.setDetectorParameters(qr)
    detector.setArucoParameters(aruco)
    return detector
