"""Detect an oversized sector-based checkerboard in a stable marked coordinate frame."""

import cv2
import numpy as np

from .calibration import chessboard_sb_flags
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image

_MAX_CORNERS = 4096
_MAX_IMAGE_PIXELS = 16_777_216


def _native_grid(corners, metadata, image_shape, requested, allow_larger):
    meta = np.asarray(metadata)
    pixels = np.asarray(corners, np.float64)
    if (
        meta.ndim != 2
        or meta.dtype != np.uint8
        or min(meta.shape) < 3
        or meta.size > _MAX_CORNERS
        or pixels.shape not in ((meta.size, 2), (meta.size, 1, 2))
    ):
        raise ValueError("OpenCV returned invalid or oversized checkerboard metadata")
    actual_size = sorted(meta.shape)
    requested_size = sorted(requested)
    if any(a < b for a, b in zip(actual_size, requested_size, strict=True)) or (
        not allow_larger and actual_size != requested_size
    ):
        raise ValueError("OpenCV returned unexpected checkerboard dimensions")
    xy = pixels.reshape(-1, 2)
    if (
        not np.isfinite(xy).all()
        or np.any(xy < -0.5)
        or np.any(xy > np.asarray(image_shape[::-1]) - 0.5)
        or len(np.unique(xy, axis=0)) != meta.size
        or np.any(meta[-1, :])
        or np.any(meta[:, -1])
        or np.any((meta[:-1, :-1] < 1) | (meta[:-1, :-1] > 4))
    ):
        raise ValueError("OpenCV returned invalid checkerboard corners or cell codes")
    return xy.reshape(*meta.shape, 2), meta


def _marker_frame(grid, metadata):
    centers = np.argwhere(metadata == 4)
    white_dot_cells = np.argwhere(metadata == 3)
    if len(centers) != 1 or len(white_dot_cells) != 2:
        return None
    center = centers[0]
    neighbors = white_dot_cells - center
    if not np.all(np.abs(neighbors).sum(axis=1) == 1) or neighbors[0] @ neighbors[1] != 0:
        return None

    # Metadata refers to cells. Grid indices refer to intersections. Reversing a
    # cell axis changes which of its four corners is the canonical origin.
    row, col = center
    steps = np.array([grid[row + 1, col] - grid[row, col], grid[row, col + 1] - grid[row, col]])
    first, second = neighbors @ steps
    handedness = first[0] * second[1] - first[1] * second[0]
    if abs(handedness) <= 1e-6 * np.linalg.norm(first) * np.linalg.norm(second):
        return None
    # On the front of the target the two dots lie east and north of the
    # black-dot cell. Camera image axes have X right and Y down.
    if handedness > 0:
        neighbors = neighbors[::-1]
    x_step, y_step = neighbors[0], -neighbors[1]
    origin = center + (1 - x_step - y_step) // 2
    indices = np.indices(metadata.shape).reshape(2, -1).T
    lattice = (indices - origin) @ np.stack((x_step, y_step)).T
    order = np.lexsort((lattice[:, 0], lattice[:, 1]))
    return center, origin, x_step, y_step, lattice[order], grid.reshape(-1, 2)[order]


def _board_drawings(grid, origin, x_end, y_end):
    grid_color = (80, 210, 230)
    x_color, y_color = (65, 85, 250), (105, 225, 115)
    return {
        "Detected checkerboard": GraphicDrawing(
            tuple(
                graphic_object("polyline", line, grid_color, thickness=1)
                for view in (grid, np.swapaxes(grid, 0, 1))
                for line in view
            )
        ),
        "Marked origin and axes": GraphicDrawing(
            (
                graphic_object("marker", [origin], (80, 240, 240)),
                graphic_object("arrow", [origin, x_end], x_color),
                graphic_object("arrow", [origin, y_end], y_color),
                graphic_object("text", [x_end], x_color, text="+X"),
                graphic_object("text", [y_end], y_color, text="+Y"),
            )
        ),
    }


class DetectMarkedCheckerboard(Block):
    type_id = "calibration.marked_checkerboard"
    title, category = "Detect marked checkerboard", "Calibration"
    block_type, subtype = "Calibration", "Grid and coordinates"
    description = "Identify visible corners of an oversized checkerboard by its three-dot origin."
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("image"),)
    outputs = (
        Port("observation", "data", required=False),
        Port("image_points", "data", required=False),
        Port("object_points", "data", required=False),
        Port("corner_ids", "data", required=False),
        Port("native_metadata", "data", required=False),
        Port("corner_count", "number"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_columns": 7,
        "minimum_rows": 5,
        "allow_larger": True,
        "normalize": True,
        "exhaustive": False,
        "accuracy": False,
        "target_id": "checkerboard",
        "spacing": 10.0,
        "units": "mm",
    }
    parameter_choices = {"units": ("mm", "m", "unitless")}
    parameter_ranges = {
        "minimum_columns": NumericRange(3, 64, whole=True),
        "minimum_rows": NumericRange(3, 64, whole=True),
        "spacing": NumericRange(1e-8, 1e6),
    }
    parameter_help = {
        "minimum_columns": "Requested inner corner columns. Larger mode treats it as a minimum.",
        "minimum_rows": "Requested inner corner rows. Larger mode treats it as a minimum.",
        "allow_larger": "Allow OpenCV to extend beyond the requested grid when visible.",
        "normalize": "Equalize the detector's grayscale input.",
        "exhaustive": "Enable OpenCV's expanded search for the grid.",
        "accuracy": "Upsample internally for subpixel localization.",
        "target_id": "Name this physical target consistently across captures.",
        "spacing": "Physical distance between adjacent corner intersections.",
    }
    output_help = {
        "observation": "Identified image and board points, ordered by board row and column.",
        "image_points": "Subpixel XY in the input image coordinates.",
        "object_points": "Metric XYZ relative to the marked origin with Z=0.",
        "corner_ids": "Signed [column, row] IDs relative to the marked origin.",
        "native_metadata": "Unmodified OpenCV cell-code matrix in detector order.",
        "corner_count": "Number of detected grid intersections.",
        "report": "Detection status, native grid shape and the visible marked-frame bounds.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        name = p["target_id"]
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValidationError("Name the physical target with up to 128 characters", "target_id")
        for key in ("allow_larger", "normalize", "exhaustive", "accuracy"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox value", key)

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        if image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS:
            raise ValidationError("Checkerboard search is limited to 16,777,216 image pixels")
        gray = gray_image(image, (np.uint8,), auto_depth=True)
        requested = int(p["minimum_columns"]), int(p["minimum_rows"])
        flags = cv2.CALIB_CB_MARKER | chessboard_sb_flags(p)
        if p["allow_larger"]:
            flags |= cv2.CALIB_CB_LARGER
        report = {
            "model": "marked_checkerboard_report_v1",
            "target_id": p["target_id"],
            "requested_inner_size": list(requested),
            "allow_larger": p["allow_larger"],
            "image_size": list(gray.shape[::-1]),
        }
        found, points, metadata = cv2.findChessboardCornersSBWithMeta(gray, requested, flags)
        if not found:
            return self._unanchored(
                report, "not_found", "No marked checkerboard found at the requested size"
            )

        grid, meta = _native_grid(points, metadata, gray.shape, requested, p["allow_larger"])
        native = meta.tolist()
        report["native_inner_size"] = list(meta.shape[::-1])
        marked = _marker_frame(grid, meta)
        if marked is None:
            return self._unanchored(
                report,
                "unresolved_marker",
                "The native metadata did not identify a three-dot origin",
                count=meta.size,
                metadata=native,
            )

        center, origin, x_step, y_step, ids, xy = marked
        origin_pixel = grid[tuple(origin)]
        xyz = np.column_stack((ids * float(p["spacing"]), np.zeros(len(ids))))
        xy, ids, xyz = xy.tolist(), ids.tolist(), xyz.tolist()
        low, high = ids[0], ids[-1]
        report.update(
            status="ok",
            corner_count=len(ids),
            board_index_bounds=[low, high],
            marker_cell_native=center[::-1].tolist(),
            origin_corner_native=origin[::-1].tolist(),
            origin_pixel=origin_pixel.tolist(),
        )
        observation = {
            "model": "marked_checkerboard_observation_v1",
            "target_id": p["target_id"],
            "image_size": report["image_size"],
            "spacing": float(p["spacing"]),
            "units": p["units"],
            "image_points": xy,
            "object_points": xyz,
            "corner_ids": ids,
            "origin_pixel": report["origin_pixel"],
        }
        drawings = {}
        if self.generate_graphics:
            drawings = _board_drawings(
                grid, origin_pixel, grid[tuple(origin + x_step)], grid[tuple(origin + y_step)]
            )
        return Result(
            {
                "observation": observation,
                "image_points": xy,
                "object_points": xyz,
                "corner_ids": ids,
                "native_metadata": native,
                "corner_count": len(ids),
                "report": report,
            },
            payloads={"object_points": {"object_point_units": p["units"]}},
            drawings=drawings,
            message=f"{len(ids)} identified corners, {meta.shape[1]} x {meta.shape[0]} native grid",
        )

    @staticmethod
    def _unanchored(report, status, message, *, count=0, metadata=NO_VALUE):
        report.update(status=status, corner_count=count)
        return Result(
            {
                "observation": NO_VALUE,
                "image_points": NO_VALUE,
                "object_points": NO_VALUE,
                "corner_ids": NO_VALUE,
                "native_metadata": metadata,
                "corner_count": count,
                "report": report,
            },
            passed=False,
            message=message,
            fail_reasons=[f"marked_checkerboard_{status}"],
        )
