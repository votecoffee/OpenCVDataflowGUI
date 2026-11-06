"""Identify board-space calibration points in partial ChArUco observations."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .fiducial_detection import DICTIONARIES, MARKER_CHOICES, MARKER_OPTIONS, detector_parameters
from .geometry3d_common import require_pinhole
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .result_graphics import ACCEPTED, REJECTED


def _make_board(p):
    size = (int(p["columns"]), int(p["rows"]))
    square = float(p["square_length"])
    marker = float(p["marker_length"])
    if marker >= square:
        raise ValidationError("Marker length must be less than square length", "marker_length")
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]])
    capacity = len(dictionary.bytesList)
    marker_count = size[0] * size[1] // 2
    if marker_count > capacity:
        raise ValidationError("The dictionary has fewer IDs than board markers", "dictionary")
    ids = p["marker_ids"]
    if not isinstance(ids, list) or (
        ids
        and (
            len(ids) != marker_count
            or any(type(i) is not int or not 0 <= i < capacity for i in ids)
            or len(set(ids)) != marker_count
        )
    ):
        raise ValidationError(
            f"Provide {marker_count} unique marker IDs in 0..{capacity - 1}, or an empty list",
            "marker_ids",
        )
    board = cv2.aruco.CharucoBoard(
        size, square, marker, dictionary, np.asarray(ids, dtype=np.int32) if ids else None
    )
    board.setLegacyPattern(bool(p["legacy_pattern"]))
    return board


def _points(board, corners, ids, total):
    if ids is None or len(ids) == 0:
        return [], [], [], False
    ids = ids.reshape(-1)
    xy = corners.reshape(-1, 2)
    if (
        len(ids) != len(xy)
        or np.any(ids < 0)
        or np.any(ids >= total)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(xy).all()
    ):
        raise ValidationError("OpenCV returned invalid ChArUco correspondences")
    order = np.argsort(ids)
    ids = ids[order]
    xyz, xy = board.matchImagePoints(xy[order], ids)
    collinear = bool(board.checkCharucoCornersCollinear(ids))
    return ids.tolist(), xy.reshape(-1, 2).tolist(), xyz.reshape(-1, 3).tolist(), collinear


def _drawings(board_markers, corners, ids, passed):
    layers = {}
    if board_markers:
        objects = []
        for marker, number in board_markers:
            points = np.asarray(marker).reshape(4, 2)
            objects.extend(
                (
                    graphic_object("polyline", points, (255, 180, 0), closed=True),
                    graphic_object("text", [points[0]], (255, 180, 0), text=f"M{number}"),
                )
            )
        layers["Board markers"] = GraphicDrawing(tuple(objects))
    if ids:
        color = ACCEPTED if passed else REJECTED
        objects = []
        for point, number in zip(corners, ids):
            objects.extend(
                (
                    graphic_object("marker", [point], color),
                    graphic_object(
                        "text", [[point[0] + 5, point[1] - 5]], color, text=f"C{number}"
                    ),
                )
            )
        layers["ChArUco corners"] = GraphicDrawing(tuple(objects))
    return layers


class DetectCharucoBoard(Block):
    type_id, title, category = "calibration.charuco", "Detect ChArUco board", "Calibration"
    block_type, subtype = "Calibration", "Grid and coordinates"
    inputs = (
        Port("image"),
        Port("camera", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("observation", "data"),
        Port("corner_count", "number"),
        Port("corner_ids", "data"),
        Port("image_points", "data"),
        Port("object_points", "data"),
    )
    defaults = {
        "columns": 7,
        "rows": 5,
        "square_length": 0.03,
        "marker_length": 0.022,
        "units": "m",
        "dictionary": "4x4_50",
        "marker_ids": [],
        "legacy_pattern": False,
        "minimum_corners": 6,
        "min_adjacent_markers": 2,
        "check_markers": True,
        "try_refine_markers": False,
        "refine_max_distance": 10.0,
        "refine_error_correction": 3.0,
        "refine_check_all_orders": True,
        **MARKER_OPTIONS,
        "corner_refinement": "none",
    }
    parameter_choices = {**MARKER_CHOICES, "units": ("m", "mm", "unitless")}
    parameter_ranges = {
        "columns": NumericRange(3, 64, whole=True),
        "rows": NumericRange(3, 64, whole=True),
        "square_length": NumericRange(1e-6, 1e6),
        "marker_length": NumericRange(1e-6, 1e6),
        "minimum_corners": NumericRange(4, 3969, whole=True),
        "min_adjacent_markers": NumericRange(1, 2, whole=True),
        "refine_max_distance": NumericRange(1e-6, 1000),
        "refine_error_correction": NumericRange(0, 10),
    }
    parameter_help = {
        "columns": "Number of squares across the board.",
        "rows": "Number of squares down the board.",
        "square_length": "Physical side length of a square in the chosen units.",
        "marker_length": "Physical side length of an ArUco marker in the same units.",
        "marker_ids": "IDs in board marker order. An empty list uses the dictionary's first IDs.",
        "legacy_pattern": "Use the OpenCV pre-4.6 chessboard pattern for older printed boards.",
        "minimum_corners": "Minimum identified corners required to emit point correspondences.",
        "min_adjacent_markers": "Minimum detected neighboring markers for an interpolated corner.",
        "check_markers": "Check whether detected markers belong to the ChArUco pattern.",
        "try_refine_markers": "Look for missed board markers near their predicted positions.",
        "refine_max_distance": "Maximum marker recovery reprojection distance in pixels.",
        "refine_error_correction": "Marker error correction multiplier during recovery.",
        "refine_check_all_orders": "Check all four possible corner orderings during recovery.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        p = self.parameters
        gray = gray_image(inputs["image"], (np.uint8,))
        size = list(gray.shape[::-1])
        board = _make_board(p)
        total = (int(p["columns"]) - 1) * (int(p["rows"]) - 1)
        minimum = int(p["minimum_corners"])
        if minimum > total:
            raise ValidationError(
                f"Minimum corners cannot exceed the {total} board corners", "minimum_corners"
            )
        parameters = cv2.aruco.CharucoParameters()
        parameters.minMarkers = int(p["min_adjacent_markers"])
        parameters.checkMarkers = p["check_markers"]
        parameters.tryRefineMarkers = p["try_refine_markers"]
        if "camera" in inputs:
            camera, distortion, calibrated_size = require_pinhole(inputs["camera"])
            if list(calibrated_size) != size:
                raise ValidationError("Camera model dimensions must match the observation image")
            parameters.cameraMatrix = camera
            parameters.distCoeffs = distortion
        refine = cv2.aruco.RefineParameters(
            float(p["refine_max_distance"]),
            float(p["refine_error_correction"]),
            p["refine_check_all_orders"],
        )
        detector = cv2.aruco.CharucoDetector(board, parameters, detector_parameters(p), refine)
        corners, ids, markers, marker_ids = detector.detectBoard(gray)
        ids, xy, xyz, collinear = _points(board, corners, ids, total)
        board_ids = board.getIds().reshape(-1).tolist()
        known = set(board_ids)
        detected = [] if marker_ids is None else marker_ids.reshape(-1).tolist()
        board_markers = [(marker, i) for marker, i in zip(markers, detected) if i in known]
        count = len(ids)
        passed = count >= minimum and not collinear
        if count < minimum:
            reason = {
                "code": "charuco_corner_count",
                "message": f"Found {count} of {minimum} required corners",
            }
        elif collinear:
            reason = {
                "code": "charuco_collinear",
                "message": "Observed board corners are collinear",
            }
        else:
            reason = None
        observation = {
            "model": "charuco_observation_v1",
            "board": {
                "columns": int(p["columns"]),
                "rows": int(p["rows"]),
                "square_length": float(p["square_length"]),
                "marker_length": float(p["marker_length"]),
                "units": p["units"],
                "dictionary": p["dictionary"],
                "marker_ids": board_ids,
                "legacy_pattern": bool(p["legacy_pattern"]),
            },
            "image_size": size,
            "corner_ids": ids,
            "image_points": xy,
            "object_points": xyz,
            "visible_marker_ids": sorted({i for _, i in board_markers}),
            "collinear": collinear,
        }
        drawings = {}
        if self.generate_graphics:
            drawings = _drawings(board_markers, xy, ids, passed)
        return Result(
            {
                "observation": observation,
                "corner_count": count,
                "corner_ids": ids,
                "image_points": xy if passed else NO_VALUE,
                "object_points": xyz if passed else NO_VALUE,
            },
            payloads={"object_points": {"object_point_units": p["units"]}} if passed else {},
            passed=passed,
            fail_reasons=[reason] if reason else [],
            score=count / total,
            message=f"{count} of {total} board corners",
            drawings=drawings,
        )
