"""OpenCV dot detection and board coordinates for circular calibration targets."""

import cv2
import numpy as np

from .parameter_validation import NumericRange, ValidationError

CIRCLE_FLAGS = {
    "circles_symmetric": cv2.CALIB_CB_SYMMETRIC_GRID,
    "circles_asymmetric": cv2.CALIB_CB_ASYMMETRIC_GRID,
}
CIRCLE_OPTIONS = {
    "circle_polarity": "dark",
    "circle_clustering": False,
    "circle_max_rectified_distance": 0.5,
    "circle_min_area": 25.0,
    "circle_max_area": 5000.0,
    "circle_min_distance": 10.0,
    "circle_min_threshold": 50.0,
    "circle_max_threshold": 220.0,
    "circle_threshold_step": 10.0,
    "circle_min_repeatability": 2,
    "circle_min_circularity": 0.0,
    "circle_min_convexity": 0.95,
    "circle_min_inertia_ratio": 0.1,
}
CIRCLE_RANGES = {
    "circle_max_rectified_distance": NumericRange(1e-6, 10),
    "circle_min_area": NumericRange(1e-6, 64_000_000),
    "circle_max_area": NumericRange(1e-6, 64_000_000),
    "circle_min_distance": NumericRange(1e-6, 16_384),
    "circle_min_threshold": NumericRange(0, 255),
    "circle_max_threshold": NumericRange(0, 255),
    "circle_threshold_step": NumericRange(1, 255),
    "circle_min_repeatability": NumericRange(1, 255, whole=True),
    "circle_min_circularity": NumericRange(0, 1),
    "circle_min_convexity": NumericRange(0, 1),
    "circle_min_inertia_ratio": NumericRange(0, 1),
}
CIRCLE_HELP = {
    "circle_polarity": "Dark dots on a light field or light dots on a dark field.",
    "circle_clustering": "Use clustering for perspective distortion on an uncluttered target.",
    "circle_max_rectified_distance": "Clustering tolerance in normalized board spacing units.",
    "circle_min_area": "Minimum candidate area in image pixels squared.",
    "circle_max_area": "Exclusive upper candidate area in image pixels squared.",
    "circle_min_distance": "Minimum separation of distinct blobs in image pixels.",
    "circle_min_threshold": "Start of the 8-bit candidate threshold sweep.",
    "circle_max_threshold": "Exclusive end of the 8-bit candidate threshold sweep.",
    "circle_threshold_step": "Increment for the candidate threshold sweep.",
    "circle_min_repeatability": "Minimum number of thresholds that must detect a candidate.",
    "circle_min_circularity": "Minimum contour circularity. Zero disables the filter.",
    "circle_min_convexity": "Minimum contour convexity. Zero disables the filter.",
    "circle_min_inertia_ratio": "Minimum minor-to-major inertia ratio. Zero disables the filter.",
}


def board_xy(columns, rows, asymmetric=False):
    xy = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    if asymmetric:
        xy[:, 0] = 2 * xy[:, 0] + xy[:, 1] % 2
    return xy


def validate_circle_options(parameters):
    for low, high, label in (
        ("circle_min_area", "circle_max_area", "Circle area"),
        ("circle_min_threshold", "circle_max_threshold", "Circle threshold"),
    ):
        if float(parameters[low]) >= float(parameters[high]):
            raise ValidationError(f"{label} minimum must be below its maximum", low, high)


def _blob_detector(p):
    options = cv2.SimpleBlobDetector_Params()
    options.blobColor = 0 if p["circle_polarity"] == "dark" else 255
    options.minArea = float(p["circle_min_area"])
    options.maxArea = float(p["circle_max_area"])
    options.minDistBetweenBlobs = float(p["circle_min_distance"])
    options.minThreshold = float(p["circle_min_threshold"])
    options.maxThreshold = float(p["circle_max_threshold"])
    options.thresholdStep = float(p["circle_threshold_step"])
    options.minRepeatability = int(p["circle_min_repeatability"])
    options.filterByCircularity = float(p["circle_min_circularity"]) > 0
    options.filterByConvexity = float(p["circle_min_convexity"]) > 0
    options.filterByInertia = float(p["circle_min_inertia_ratio"]) > 0
    if options.filterByCircularity:
        options.minCircularity = float(p["circle_min_circularity"])
    if options.filterByConvexity:
        options.minConvexity = float(p["circle_min_convexity"])
    if options.filterByInertia:
        options.minInertiaRatio = float(p["circle_min_inertia_ratio"])
    return cv2.SimpleBlobDetector_create(options)


def _order_error(board, centers):
    matrix, _ = cv2.findHomography(board, centers, 0)
    if matrix is None or not np.isfinite(matrix).all():
        return np.inf
    predicted = cv2.perspectiveTransform(board[:, None, :], matrix).reshape(-1, 2)
    return float(np.mean(np.sum((predicted - centers) ** 2, axis=1)))


def find_circle_grid(gray, size, p):
    flags = CIRCLE_FLAGS[p["detector"]]
    options = cv2.CirclesGridFinderParameters()
    if p["circle_clustering"]:
        flags |= cv2.CALIB_CB_CLUSTERING
        options.maxRectifiedDistance = float(p["circle_max_rectified_distance"])
    found, centers = cv2.findCirclesGrid(
        gray, size, flags=flags, blobDetector=_blob_detector(p), parameters=options
    )
    if not found:
        return False, None
    columns, rows = size
    centers = centers.reshape(-1, 2).astype(np.float32)
    if len(centers) != columns * rows or not np.isfinite(centers).all():
        raise ValueError("Circle grid returned invalid center coordinates")
    if p["detector"] == "circles_asymmetric" and rows % 2 == 0:
        # Some OpenCV detections swap adjacent rows on even-row staggered targets.
        board = board_xy(columns, rows, asymmetric=True).astype(np.float32)
        swapped = centers.reshape(-1, 2, columns, 2)[:, ::-1].reshape(-1, 2).copy()
        if _order_error(board, swapped) < _order_error(board, centers):
            centers = swapped
    return True, centers
