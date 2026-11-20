"""Rasterize planar fiducials with explicit print size and physical correspondences."""

from dataclasses import dataclass

import cv2
import numpy as np

from .core import Block, Port, Result
from .fiducial_detection import DICTIONARIES, board_marker_ids
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError

_LAYOUTS = ("aruco_marker", "aruco_grid", "charuco")
_MAX_PIXELS = 16_777_216
_MAX_AXIS = 16_384


@dataclass(frozen=True)
class _Raster:
    target_size: tuple[int, int]
    margin: int
    marker: int
    square: int | None
    gap: int | None
    pixels_per_unit: float

    @property
    def image_size(self):
        return tuple(side + 2 * self.margin for side in self.target_size)


def _layout(p, dictionary):
    mode = p["layout"]
    columns, rows = int(p["columns"]), int(p["rows"])
    count = {"aruco_marker": 1, "aruco_grid": columns * rows, "charuco": columns * rows // 2}
    ids = board_marker_ids(p["marker_ids"], count[mode], dictionary)
    mm_per_unit = 1000 if p["units"] == "m" else 1
    requested_scale = float(p["dpi"]) * mm_per_unit / 25.4
    marker_length = float(p["marker_length"])
    square_px, gap_px = None, None
    if mode == "charuco":
        if min(columns, rows) < 3:
            raise ValidationError(
                "ChArUco needs at least three squares per axis", "columns", "rows"
            )
        square_length = float(p["square_length"])
        if square_length <= marker_length:
            raise ValidationError(
                "ChArUco marker length must be less than square length", "marker_length"
            )
        square_px = round(square_length * requested_scale)
        scale = square_px / square_length
        # Match parity so the marker has the same whole-pixel inset on both sides.
        inset = round((square_length - marker_length) * scale / 2)
        if inset < 1:
            raise ValidationError(
                "ChArUco needs a white inset of at least one pixel around every marker",
                "dpi",
                "marker_length",
                "square_length",
            )
        marker_px = square_px - 2 * inset
        target_size = (columns * square_px, rows * square_px)
    else:
        marker_px = round(marker_length * requested_scale)
        scale = marker_px / marker_length
        if mode == "aruco_grid":
            gap_px = round(float(p["marker_separation"]) * scale)
            if gap_px < 1:
                raise ValidationError(
                    "The grid needs at least one pixel between markers",
                    "marker_separation",
                    "dpi",
                )
            target_size = tuple(n * marker_px + (n - 1) * gap_px for n in (columns, rows))
        else:
            target_size = (marker_px, marker_px)
    modules = dictionary.markerSize + 2 * int(p["border_bits"])
    if marker_px < modules:
        raise ValidationError(
            f"Each marker needs at least {modules} pixels per side for its code and border",
            "dpi",
            "marker_length",
            "border_bits",
        )
    margin_px = round(float(p["margin"]) * scale)
    raster = _Raster(target_size, margin_px, marker_px, square_px, gap_px, scale)
    width, height = raster.image_size
    if max(width, height) > _MAX_AXIS or width * height > _MAX_PIXELS:
        raise ValidationError(
            "Target exceeds 16,384 pixels per axis or 16,777,216 image pixels",
            "dpi",
            "margin",
        )
    return raster, ids


class GenerateFiducialTarget(Block):
    type_id, title = "calibration.generate_fiducial_target", "Generate planar fiducial target"
    category = block_type = "Calibration"
    subtype = "Grid and coordinates"
    outputs = (Port("image"), Port("target", "data"))
    defaults = {
        "layout": "charuco",
        "dictionary": "4x4_50",
        "marker_ids": [],
        "columns": 7,
        "rows": 5,
        "marker_length": 22.0,
        "marker_separation": 8.0,
        "square_length": 30.0,
        "legacy_pattern": False,
        "border_bits": 1,
        "margin": 8.0,
        "units": "mm",
        "dpi": 300,
    }
    parameter_choices = {
        "layout": _LAYOUTS,
        "dictionary": tuple(DICTIONARIES),
        "units": ("mm", "m"),
    }
    parameter_ranges = {
        "columns": NumericRange(1, 64, whole=True),
        "rows": NumericRange(1, 64, whole=True),
        "marker_length": NumericRange(0.000001, 10_000),
        "marker_separation": NumericRange(0.000001, 10_000),
        "square_length": NumericRange(0.000001, 10_000),
        "border_bits": NumericRange(1, 10, whole=True),
        "margin": NumericRange(0, 10_000),
        "dpi": NumericRange(36, 2400, whole=True),
    }
    parameter_help = {
        "layout": "Single encoded marker, regular marker grid or a ChArUco chessboard.",
        "dictionary": "Predefined ArUco or AprilTag family. Match it in the detector.",
        "marker_ids": "IDs in board order. Leave empty to start at zero.",
        "columns": "Markers across an ArUco grid or squares across a ChArUco board.",
        "rows": "Markers down an ArUco grid or squares down a ChArUco board.",
        "marker_length": "Outer black edge length in the selected units.",
        "marker_separation": "White edge-to-edge spacing in an ArUco grid.",
        "square_length": "ChArUco square pitch. Preserved at the reported print size.",
        "legacy_pattern": "Use the OpenCV pre-4.6 ChArUco parity for even row counts.",
        "border_bits": "Width of the black frame in code modules. Match it in the detector.",
        "margin": "White page margin on every side, in the selected length units.",
        "units": "Length units for the target configuration, geometry and print size.",
        "dpi": "Desired render density. The reported effective density accounts for rounding.",
    }
    parallel_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["legacy_pattern"]) is not bool:
            raise ValidationError("Legacy pattern must be boolean", "legacy_pattern")
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]])
        _layout(p, dictionary)

    def process(self, inputs):
        p = self.parameters
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[p["dictionary"]])
        raster, ids = _layout(p, dictionary)
        mode = p["layout"]
        border = int(p["border_bits"])
        size = (int(p["columns"]), int(p["rows"]))
        charuco_points = np.empty((0, 3))
        if mode == "aruco_marker":
            marker = cv2.aruco.generateImageMarker(
                dictionary, int(ids[0]), raster.marker, borderBits=border
            )
            gap = raster.margin
            pixels = cv2.copyMakeBorder(marker, gap, gap, gap, gap, cv2.BORDER_CONSTANT, value=255)
            s = raster.marker
            corners = np.asarray([[[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]]])
        else:
            if mode == "charuco":
                board = cv2.aruco.CharucoBoard(size, raster.square, raster.marker, dictionary, ids)
                board.setLegacyPattern(p["legacy_pattern"])
                charuco_points = board.getChessboardCorners()
            else:
                board = cv2.aruco.GridBoard(size, raster.marker, raster.gap, dictionary, ids)
            pixels = board.generateImage(
                raster.image_size, marginSize=raster.margin, borderBits=border
            )
            corners = np.asarray(board.getObjPoints())
        scale = raster.pixels_per_unit
        # The integer geometry is the actual raster layout, anchored to one exact physical length.
        report = {
            "model": "fiducial_target_v1",
            "layout": mode,
            "dictionary": p["dictionary"],
            "units": p["units"],
            "border_bits": border,
            "marker_ids": ids.tolist(),
            "marker_length": raster.marker / scale,
            "marker_corners_xyz": (corners.astype(float) / scale).tolist(),
            "target_size": [value / scale for value in raster.target_size],
            "image_size": list(raster.image_size),
            "target_rect_px": [raster.margin, raster.margin, *raster.target_size],
            "pixels_per_unit": scale,
            "effective_dpi": scale * 25.4 / (1000 if p["units"] == "m" else 1),
            "print_size": [value / scale for value in raster.image_size],
        }
        if mode == "aruco_grid":
            report.update(columns=size[0], rows=size[1], marker_separation=raster.gap / scale)
        if mode == "charuco":
            report.update(
                columns=size[0],
                rows=size[1],
                square_length=raster.square / scale,
                legacy_pattern=p["legacy_pattern"],
                charuco_corners_xyz=(charuco_points.astype(float) / scale).tolist(),
            )
        label = "marker" if len(ids) == 1 else "markers"
        return Result(
            {"image": ImageArray(pixels, "gray"), "target": report},
            message=f"{len(ids)} {label} · {pixels.shape[1]} × {pixels.shape[0]} pixels",
        )
