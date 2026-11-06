"""Offline camera calibration and grid-derived planar measurement coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np

from .circle_grid import (
    CIRCLE_FLAGS,
    CIRCLE_HELP,
    CIRCLE_OPTIONS,
    CIRCLE_RANGES,
    board_xy,
    find_circle_grid,
    validate_circle_options,
)
from .core import Block, Graph, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import opencv_image, preview_rgb
from .processing import finite, integer
from .toolblocks import Binding, ToolBlock


def grid_settings(parameters):
    columns, rows = int(parameters["columns"]), int(parameters["rows"])
    spacing = float(parameters["spacing"])
    if columns < 3 or rows < 3 or columns * rows > 10000:
        raise ValueError("Grid requires at least 3 by 3 features and at most 10000 features")
    if not np.isfinite(spacing) or spacing <= 0 or not parameters["units"].strip():
        raise ValueError("Grid spacing must be positive and units must be specified")
    world = np.zeros((columns * rows, 3), np.float32)
    asymmetric = parameters.get("detector") == "circles_asymmetric"
    world[:, :2] = board_xy(columns, rows, asymmetric) * spacing
    return (columns, rows), world


GRID_OPTIONS = {
    "detector": "sb",
    "normalize": True,
    "exhaustive": True,
    "accuracy": False,
    "adaptive_threshold": True,
    "fast_check": False,
    "subpixel_window": 11,
    "subpixel_iterations": 30,
    "subpixel_epsilon": 0.001,
    **CIRCLE_OPTIONS,
}
CALIB_FLAGS = {
    "use_intrinsic_guess": cv2.CALIB_USE_INTRINSIC_GUESS,
    "fix_principal_point": cv2.CALIB_FIX_PRINCIPAL_POINT,
    "fix_aspect_ratio": cv2.CALIB_FIX_ASPECT_RATIO,
    "zero_tangent": cv2.CALIB_ZERO_TANGENT_DIST,
    "rational_model": cv2.CALIB_RATIONAL_MODEL,
    "thin_prism_model": cv2.CALIB_THIN_PRISM_MODEL,
    "tilted_model": cv2.CALIB_TILTED_MODEL,
    **{f"fix_k{i}": getattr(cv2, f"CALIB_FIX_K{i}") for i in range(1, 7)},
}


def find_grid(image, size, parameters=None):
    # Detection copy only: numeric source pixels remain unchanged.
    display = cv2.cvtColor(preview_rgb(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    p = GRID_OPTIONS if parameters is None else parameters
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE if p["normalize"] else 0
    is_circle_grid = p["detector"] in CIRCLE_FLAGS
    if is_circle_grid:
        found, corners = find_circle_grid(gray, size, p)
    elif p["detector"] == "sb":
        flags |= cv2.CALIB_CB_EXHAUSTIVE if p["exhaustive"] else 0
        flags |= cv2.CALIB_CB_ACCURACY if p["accuracy"] else 0
        found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
    else:
        flags |= cv2.CALIB_CB_ADAPTIVE_THRESH if p["adaptive_threshold"] else 0
        flags |= cv2.CALIB_CB_FAST_CHECK if p["fast_check"] else 0
        found, corners = cv2.findChessboardCorners(gray, size, flags=flags)
        if found:
            window = integer(p["subpixel_window"], "Subpixel window", 1, 255)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                integer(p["subpixel_iterations"], "Subpixel iterations", 1, 10000),
                finite(p["subpixel_epsilon"], "Subpixel epsilon", 0),
            )
            corners = cv2.cornerSubPix(gray, corners, (window, window), (-1, -1), criteria)
    if not found:
        feature = "circle centers" if is_circle_grid else "inner corners"
        raise ValueError(f"Calibration grid not found: expected {size[0]} by {size[1]} {feature}")
    return corners.reshape(-1, 2).astype(np.float32), display


class _GridCalibration(Block):
    parameter_choices = {
        "detector": ("sb", "classic", *CIRCLE_FLAGS),
        "circle_polarity": ("dark", "light"),
    }
    parameter_ranges = CIRCLE_RANGES
    parameter_help = {
        **CIRCLE_HELP,
        "columns": "Number of inner chessboard corners or circle centers in each row.",
        "rows": "Number of rows of inner chessboard corners or circle centers.",
        "spacing": "Feature pitch. For asymmetric circles, same-row pitch is twice this value.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["detector"] in CIRCLE_FLAGS:
            validate_circle_options(p)


class CalibrationGrid(_GridCalibration):
    type_id, title, category = "calibration.grid", "Calibration grid", "Calibration"
    block_type, subtype = "Calibration", "Grid and coordinates"
    inputs = (Port("image"),)
    outputs = (Port("mapping", "data"), Port("corners", "data"), Port("overlay"))
    defaults = {
        **GRID_OPTIONS,
        "columns": 9,
        "rows": 6,
        "spacing": 10.0,
        "units": "mm",
        "origin_column": 0,
        "origin_row": 0,
        "reverse_order": False,
        "max_error": 1.0,
    }

    def process(self, inputs):
        size, world = grid_settings(self.parameters)
        column, row = int(self.parameters["origin_column"]), int(self.parameters["origin_row"])
        maximum = float(self.parameters["max_error"])
        if (
            not (0 <= column < size[0] and 0 <= row < size[1])
            or not np.isfinite(maximum)
            or maximum <= 0
        ):
            raise ValueError("Origin must be a grid corner and max error must be positive")
        corners, overlay = find_grid(inputs["image"], size, self.parameters)
        if self.parameters["reverse_order"]:
            corners = corners[::-1].copy()
            if self.parameters["detector"] == "circles_asymmetric":
                world[:, :2] = -world[::-1, :2]
        origin_index = row * size[0] + column
        world[:, :2] -= world[origin_index, :2].copy()
        matrix, _ = cv2.findHomography(corners, world[:, :2], 0)
        if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError("Degenerate calibration grid geometry")
        # Anchor the selected observed corner exactly, rather than leaving it at
        # the small residual offset of the unconstrained least-squares fit.
        origin_h = matrix @ np.r_[corners[origin_index], 1.0]
        if abs(origin_h[2]) < 1e-12:
            raise ValueError("Calibration origin maps to infinity")
        anchor = origin_h[:2] / origin_h[2]
        matrix = np.array([[1, 0, -anchor[0]], [0, 1, -anchor[1]], [0, 0, 1]]) @ matrix
        projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - world[:, :2]) ** 2, axis=1))))
        mapping = {
            "model": "planar_homography_v1",
            "pixel_to_physical": matrix.tolist(),
            "units": self.parameters["units"],
            "rms_error": error,
            "image_size": [int(overlay.shape[1]), int(overlay.shape[0])],
            "origin_pixel": corners[origin_index].tolist(),
            "origin_grid": [column, row],
            "axes": "+X increasing detected column, +Y increasing detected row",
            "grid_orientation": "reversed" if self.parameters["reverse_order"] else "detector",
            "plane_only": True,
        }
        cv2.drawChessboardCorners(overlay, size, corners.reshape(-1, 1, 2), True)
        origin = tuple(np.rint(corners[origin_index]).astype(int))
        cv2.drawMarker(overlay, origin, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(overlay, "Origin", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        axes = cv2.perspectiveTransform(
            np.float64([[[self.parameters["spacing"], 0], [0, self.parameters["spacing"]]]]),
            np.linalg.inv(matrix),
        ).reshape(-1, 2)
        labels = ("+X", "+Y")
        colors = ((0, 0, 255), (0, 180, 0))
        for endpoint, label, color in zip(axes, labels, colors):
            end = tuple(np.rint(endpoint).astype(int))
            cv2.arrowedLine(overlay, origin, end, color, 2, tipLength=0.2)
            cv2.putText(overlay, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        drawings = {}
        if self.generate_graphics:
            origin_pixel = corners[origin_index]
            axis_objects = [graphic_object("marker", [origin_pixel], colors[0])]
            for endpoint, label, color in zip(axes, labels, colors):
                axis_objects.append(graphic_object("polyline", [origin_pixel, endpoint], color))
                axis_objects.append(graphic_object("text", [endpoint], color, text=label))
            drawings = {
                "Grid features": GraphicDrawing(
                    tuple(graphic_object("marker", [point], (255, 180, 0)) for point in corners)
                ),
                "Grid origin and axes": GraphicDrawing(tuple(axis_objects)),
            }
        return Result(
            {"mapping": mapping, "corners": corners.tolist(), "overlay": overlay},
            graphics={"grid origin": overlay},
            drawings=drawings,
            passed=error <= maximum,
            score=max(0.0, 1 - error / maximum),
            message=f"Planar fit RMS {error:.4g} {self.parameters['units']}",
        )


def map_points(mapping, points):
    if mapping.get("model") != "planar_homography_v1":
        raise ValueError("Expected a calibration-grid planar mapping")
    matrix = np.asarray(mapping["pixel_to_physical"], np.float64)
    pixels = np.asarray(points, np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
    ):
        raise ValueError("Invalid planar mapping matrix")
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels) or not np.isfinite(pixels).all():
        raise ValueError("Points must be a nonempty list of finite [x, y] pixel coordinates")
    homogeneous = np.column_stack((pixels, np.ones(len(pixels)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("Point maps to infinity")
    transformed = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(transformed).all():
        raise ValueError("Mapping produced nonfinite coordinates")
    return transformed


class PixelToPhysical(Block):
    type_id, title, category = "calibration.pixel_to_physical", "Pixel to physical", "Calibration"
    block_type, subtype = "Calibration", "Grid and coordinates"
    inputs = (Port("mapping", "data"), Port("points", "data", required=False))
    outputs = (Port("points", "data"),)
    defaults = {"pixel_points": [[0.0, 0.0]], "max_mapping_error": 1.0}

    def process(self, inputs):
        mapping = inputs["mapping"]
        maximum = float(self.parameters["max_mapping_error"])
        error = float(mapping["rms_error"])
        if not np.isfinite([maximum, error]).all() or maximum <= 0 or error < 0 or error > maximum:
            raise ValueError("Mapping exceeds the allowed physical-space calibration error")
        pixels = inputs.get("points", self.parameters["pixel_points"])
        output = map_points(mapping, pixels)
        return Result(
            {
                "points": {
                    "pixels": pixels,
                    "physical": output.tolist(),
                    "units": mapping["units"],
                    "plane_only": True,
                }
            },
            message=f"Mapped {len(output)} points on calibration plane",
        )


def camera_parameters(calibration):
    if calibration.get("model") != "opencv_pinhole_v1":
        raise ValueError("Expected OpenCV pinhole calibration data")
    matrix = np.asarray(calibration["camera_matrix"], np.float64)
    distortion = np.asarray(calibration["distortion"], np.float64).reshape(-1)
    if (
        matrix.shape != (3, 3)
        or distortion.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(matrix).all()
        or not np.isfinite(distortion).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
    ):
        raise ValueError("Invalid camera matrix or distortion coefficients")
    return matrix, distortion


class LensCalibration(_GridCalibration):
    type_id, title, category = "calibration.lens", "Calibrate lens", "Calibration"
    block_type, subtype = "Calibration", "Lens distortion"
    outputs = (Port("calibration", "data"), Port("rms", "number"))
    defaults = {
        **GRID_OPTIONS,
        **dict.fromkeys(CALIB_FLAGS, False),
        "solver": "svd",
        "initial_camera_matrix": [],
        "initial_distortion": [],
        "max_iterations": 30,
        "epsilon": float(np.finfo(float).eps),
        "image_paths": [],
        "columns": 9,
        "rows": 6,
        "spacing": 10.0,
        "units": "mm",
        "max_rms_pixels": 1.0,
    }
    parameter_choices = {**_GridCalibration.parameter_choices, "solver": ("svd", "lu", "qr")}

    def process(self, inputs):
        size, world = grid_settings(self.parameters)
        paths = self.parameters["image_paths"]
        maximum = float(self.parameters["max_rms_pixels"])
        if not isinstance(paths, list) or len(paths) < 3 or len(set(paths)) != len(paths):
            raise ValueError("Provide at least three distinct grid image paths at varied poses")
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Maximum RMS must be positive")
        points, image_size = [], None
        for path in paths:
            image = cv2.imdecode(
                np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            if image is None:
                raise ValueError(f"Cannot decode calibration image: {path}")
            current_size = (image.shape[1], image.shape[0])
            if image_size is not None and current_size != image_size:
                raise ValueError("All lens calibration images must have identical dimensions")
            image_size = current_size
            corners, _ = find_grid(image, size, self.parameters)
            if any(np.sqrt(np.mean((corners - previous) ** 2)) < 0.5 for previous in points):
                raise ValueError("Calibration views are duplicates; use varied board poses")
            points.append(corners)
        p = self.parameters
        matrix = (
            np.asarray(p["initial_camera_matrix"], np.float64)
            if p["initial_camera_matrix"]
            else None
        )
        distortion = (
            np.asarray(p["initial_distortion"], np.float64) if p["initial_distortion"] else None
        )
        if matrix is not None and (matrix.shape != (3, 3) or not np.isfinite(matrix).all()):
            raise ValueError("Initial camera matrix must be finite 3x3")
        if distortion is not None and (
            distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()
        ):
            raise ValueError("Initial distortion must contain 4,5,8,12 or 14 finite values")
        if (
            p["use_intrinsic_guess"] or p["fix_aspect_ratio"] or p["fix_principal_point"]
        ) and matrix is None:
            raise ValueError("Selected intrinsic constraints require an initial camera matrix")
        flags = sum(value for key, value in CALIB_FLAGS.items() if p[key])
        flags |= {"svd": 0, "lu": cv2.CALIB_USE_LU, "qr": cv2.CALIB_USE_QR}[p["solver"]]
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            integer(p["max_iterations"], "Maximum iterations", 1, 10000),
            finite(p["epsilon"], "Epsilon", 0),
        )
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            [world.copy() for _ in points],
            points,
            image_size,
            matrix,
            distortion,
            flags=flags,
            criteria=criteria,
        )
        calibration = {
            "model": "opencv_pinhole_v1",
            "camera_matrix": matrix.tolist(),
            "distortion": distortion.reshape(-1).tolist(),
            "image_size": list(image_size),
            "rms_pixels": float(rms),
            "views": len(points),
            "units": self.parameters["units"],
        }
        camera_parameters(calibration)
        if not np.isfinite(rms):
            raise ValueError("Calibration failed to produce a finite reprojection error")
        return Result(
            {"calibration": calibration, "rms": float(rms)},
            passed=rms <= maximum,
            score=max(0.0, 1 - rms / maximum),
            message=f"{len(points)} views · reprojection RMS {rms:.4g} px",
        )


class Undistort(Block):
    type_id, title, category = "calibration.undistort", "Undistort lens", "Calibration"
    block_type, subtype = "Calibration", "Lens distortion"
    inputs = (Port("image"), Port("calibration", "data"))
    outputs = (Port("image"),)
    defaults = {"max_rms_pixels": 1.0, "new_camera_matrix": []}

    def process(self, inputs):
        calibration = inputs["calibration"]
        matrix, distortion = camera_parameters(calibration)
        image = opencv_image(inputs["image"])
        maximum, rms = float(self.parameters["max_rms_pixels"]), float(calibration["rms_pixels"])
        if not np.isfinite([maximum, rms]).all() or maximum <= 0 or rms < 0 or rms > maximum:
            raise ValueError("Lens calibration exceeds the allowed reprojection error")
        if calibration["image_size"] != [image.shape[1], image.shape[0]]:
            raise ValueError("Image dimensions do not match the lens calibration")
        replacement = self.parameters["new_camera_matrix"]
        replacement = np.asarray(replacement, np.float64) if replacement else None
        if replacement is not None and (
            replacement.shape != (3, 3) or not np.isfinite(replacement).all()
        ):
            raise ValueError("New camera matrix must be finite 3x3")
        output = cv2.undistort(image, matrix, distortion, newCameraMatrix=replacement)
        return Result({"image": output}, graphics={"undistorted": output})


class ColorBalance(Block):
    type_id, title, category = "calibration.color_balance", "White / color balance", "Calibration"
    block_type, subtype = "Calibration", "Color balance"
    inputs = (Port("image"),)
    outputs = (Port("image"), Port("gains", "data"))
    defaults = {
        "mode": "gray_world",
        "roi": [],
        "red_gain": 1.0,
        "green_gain": 1.0,
        "blue_gain": 1.0,
    }
    parameter_choices = {"mode": ("gray_world", "manual")}

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        if (
            image.ndim != 3
            or image.shape[2] not in (3, 4)
            or image.dtype not in (np.uint8, np.uint16, np.float32, np.float64)
        ):
            raise ValueError("Color balance requires BGR/RGB color pixels")
        samples = image[..., :3].astype(np.float64)
        if not np.isfinite(samples).all() or np.any(samples < 0):
            raise ValueError("Color samples must be finite and nonnegative")
        region = samples
        roi = self.parameters["roi"]
        if roi:
            if len(roi) != 4 or any(int(v) != v for v in roi):
                raise ValueError("ROI must be [x, y, width, height] in whole pixels")
            x, y, width, height = map(int, roi)
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > image.shape[1]
                or y + height > image.shape[0]
            ):
                raise ValueError("ROI must be inside the image")
            region = samples[y : y + height, x : x + width]
        mode = self.parameters["mode"]
        if mode == "gray_world":
            means = region.mean(axis=(0, 1))
            if np.any(means <= 1e-12):
                raise ValueError("Reference region must contain nonzero samples in every channel")
            gains = means.mean() / means
        elif mode == "manual":
            gains = np.array(
                [self.parameters[k] for k in ("blue_gain", "green_gain", "red_gain")], float
            )
        else:
            raise ValueError("Unknown color balance mode")
        if not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Color gains must be finite and positive")
        corrected = samples * gains
        if np.issubdtype(image.dtype, np.integer):
            corrected = np.rint(np.clip(corrected, 0, np.iinfo(image.dtype).max))
        output = image.copy()
        output[..., :3] = corrected.astype(image.dtype)
        return Result(
            {"image": output, "gains": dict(zip(("B", "G", "R"), gains.tolist()))},
            graphics={"balanced": output},
            message=f"{mode} color correction",
        )


def validate_profile(profile):
    if profile.get("model") == "planar_homography_v1":
        map_points(profile, [profile["origin_pixel"]])
        error = float(profile["rms_error"])
    elif profile.get("model") == "opencv_pinhole_v1":
        camera_parameters(profile)
        error = float(profile["rms_pixels"])
    else:
        raise ValueError("Unknown calibration profile model")
    size = profile["image_size"]
    if (
        len(size) != 2
        or any(not isinstance(v, int) or v <= 0 for v in size)
        or not np.isfinite(error)
        or error < 0
        or not profile["units"].strip()
    ):
        raise ValueError("Invalid calibration image size, error or units")


class LoadCalibration(Block):
    type_id, title, category = "input.calibration", "Load calibration profile", "Input"
    block_type, subtype = "Input", "Calibration profiles"
    outputs = (Port("profile", "data"),)
    defaults = {"path": "", "path_base": "working_directory"}
    parameter_choices = {"path_base": ("working_directory", "project")}

    def process(self, inputs):
        path = Path(self.parameters["path"])
        if self.parameters["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValueError("Save or open the project before loading a relative profile")
            path = self.project_dir / path
        profile = json.loads(path.read_text(encoding="utf-8"))
        validate_profile(profile)
        return Result({"profile": profile}, message=profile["model"])


class SaveCalibration(Block):
    type_id, title, category = "output.calibration", "Save calibration profile", "Output"
    block_type, subtype = "Output", "Calibration profiles"
    inputs = (Port("profile", "data"),)
    defaults = {"path": "calibration.json"}

    def process(self, inputs):
        profile = inputs["profile"]
        validate_profile(profile)
        content = json.dumps(profile, indent=2, allow_nan=False) + "\n"
        with Path(self.parameters["path"]).open("x", encoding="utf-8") as stream:
            stream.write(content)
        return Result(message="Calibration profile saved")


def calibration_templates(registry):
    templates = []
    for type_id in (
        "calibration.grid",
        "calibration.lens",
        "calibration.undistort",
        "calibration.color_balance",
        "calibration.pixel_to_physical",
    ):
        graph = Graph()
        child = graph.add(registry.create(type_id))
        block = ToolBlock(
            graph,
            {p.name: Binding(child.id, p.name) for p in child.inputs},
            {p.name: Binding(child.id, p.name) for p in child.outputs},
        )
        block.name = child.title + " ToolBlock"
        block.block_type, block.subtype = child.block_type, child.subtype
        templates.append(block.definition())
    return templates
