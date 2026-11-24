"""Ordered Edge Drawing paths, linked lines and round primitive collections."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle, graphic_object
from .managed_process import ManagedProcess
from .objects import circle_item, collection, ellipse_item, line_item
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, gray_image
from .segmentation import bgr8
from .selection import SELECTION_INPUTS, contains, selection_mask

WORKER_SCRIPT = Path(__file__).with_name("edge_drawing_worker.py").resolve()
MAX_PIXELS = 1_048_576
MAX_RESULT_BYTES = 8 * 1024**2


def native_geometry(image, settings, cancelled):
    if cancelled():
        raise InterruptedError("Edge Drawing cancelled")
    if not hasattr(getattr(cv2, "ximgproc", None), "createEdgeDrawing"):
        raise ValidationError("Edge Drawing requires OpenCV contrib ximgproc")
    with TemporaryDirectory(prefix="dataflow-edge-drawing-") as temporary:
        work = Path(temporary)
        np.save(work / "image.npy", image, allow_pickle=False)
        worker = ManagedProcess(
            [sys.executable, "-I", str(WORKER_SCRIPT), json.dumps(settings, allow_nan=False)],
            work, os.environ.copy(), memory_limit=1024**3,
        )
        try:
            exit_code = worker.wait(float(settings["timeout_seconds"]), cancelled)
            if exit_code:
                diagnostic = [
                    line for line in worker.tail().strip().splitlines()
                    if not line.startswith("DATAFLOW_CHILD_PID=")
                ]
                detail = f" {diagnostic[-1]}" if diagnostic else ""
                raise ValidationError(f"Edge Drawing worker failed (exit {exit_code}).{detail}")
        except TimeoutError as exc:
            raise TimeoutError("Edge Drawing timed out") from exc
        finally:
            worker.close()
        output = work / "result.npz"
        if output.stat().st_size > MAX_RESULT_BYTES:
            raise ValidationError("Edge Drawing result exceeds 8 MiB. Reduce edge complexity or crop.")
        with np.load(output, allow_pickle=False) as archive:
            return tuple(
                archive[key] for key in ("edge", "gradient", "lengths", "points", "lines", "line_paths", "shapes")
            )


def ordered_paths(lengths, points, selected):
    paths = {}
    offset = 0
    for index, length in enumerate(lengths.tolist()):
        chain = points[offset : offset + length]
        offset += length
        if len(chain) and (selected is None or selected[chain[:, 1], chain[:, 0]].any()):
            paths[index] = chain
    return paths


def round_primitives(rows, selection, width, height):
    circles, ellipses = [], []
    for index, row in enumerate(rows):
        x, y, radius, a, b, angle = map(float, row)
        if (
            not np.isfinite(row).all()
            or not (0 <= x < width and 0 <= y < height)
            or not contains(selection, (x, y))
        ):
            continue
        if radius > 0:
            circles.append(circle_item((x, y, radius), None, index))
        elif radius == 0 and min(a, b) > 0:
            if a < b:
                a, b, angle = b, a, angle + 90
            ellipses.append(
                ellipse_item(
                    {"shape": "ellipse", "center": [x, y], "axis_lengths": [2 * a, 2 * b],
                     "angle_degrees": angle % 180},
                    index,
                )
            )
    return circles, ellipses


def edge_drawings(paths, lines, circles, ellipses):
    # Bound preview complexity while returning complete geometry on data ports.
    path_objects = []
    for chain in list(paths.values())[:256]:
        step = (len(chain) + 511) // 512
        sampled = chain[::step]
        if step > 1:
            sampled = np.vstack((sampled, chain[-1]))
        path_objects.append(graphic_object("polyline", sampled, (125, 210, 255), thickness=1))
    layers = {
        "edge paths": tuple(path_objects),
        "fitted lines": tuple(
            graphic_object("polyline", item["geometry"]["points"], (80, 245, 150))
            for item in lines[:512]
        ),
        "circles": tuple(
            graphic_circle(*item["geometry"]["center"], item["geometry"]["radius"], (255, 200, 85))
            for item in circles[:256]
        ),
        "ellipses": tuple(
            graphic_object("polyline", item["geometry"]["points"], (255, 160, 185), closed=True)
            for item in ellipses[:256]
        ),
    }
    return {name: GraphicDrawing(objects) for name, objects in layers.items() if objects}


class EdgeDrawingPrimitives(ImageTool):
    type_id, title, subtype = "inspect.edge_drawing", "Edge Drawing", "Features"
    category = block_type = "Inspection"
    description = "Trace ordered edge paths, then fit straight lines, circles and rotated ellipses."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("edges", visual_type="mask"),
        Port("gradient"),
        Port("segments", "data"),
        Port("lines", "data"),
        Port("circles", "data"),
        Port("ellipses", "data"),
        Port("report", "data"),
    )
    defaults = {
        "input_mode": "grayscale",
        "gradient": "prewitt",
        "gradient_threshold": 20,
        "anchor_threshold": 0,
        "scan_interval": 1,
        "minimum_path_length": 10,
        "sigma": 1.0,
        "sum_gradient": True,
        "parameter_free": False,
        "validate_nfa": True,
        "minimum_line_length": -1,
        "line_join_distance": 6.0,
        "line_fit_error": 1.0,
        "maximum_line_error": 1.3,
        "fit_lines": True,
        "fit_round": True,
        "timeout_seconds": 30.0,
    }
    parameter_choices = {
        "input_mode": ("grayscale", "color"),
        "gradient": ("prewitt", "sobel", "scharr", "lsd"),
    }
    parameter_ranges = {
        "gradient_threshold": NumericRange(1, 4095, whole=True),
        "anchor_threshold": NumericRange(0, 4095, whole=True),
        "scan_interval": NumericRange(1, 128, whole=True),
        "minimum_path_length": NumericRange(3, 8192, whole=True),
        "sigma": NumericRange(0, 12),
        "line_join_distance": NumericRange(0, 128),
        "line_fit_error": NumericRange(1e-4, 32),
        "maximum_line_error": NumericRange(1e-4, 32),
        "timeout_seconds": NumericRange(0.1, 120),
    }
    parameter_help = {
        "input_mode": "Grayscale ED or native color ED. Color uses a Di Zenzo gradient on color channels.",
        "gradient": "Native derivative operator for grayscale ED.",
        "gradient_threshold": "Minimum gradient to follow when computing edge paths.",
        "anchor_threshold": "Required gradient difference between an anchor and nearby pixels.",
        "scan_interval": "Spacing of candidate anchor scans in pixels.",
        "minimum_path_length": "Shortest candidate path admitted before native validation.",
        "sigma": "Native smoothing sigma. In grayscale mode, values below 1 skip initial smoothing.",
        "sum_gradient": "Grayscale gradient sums absolute derivatives when enabled.",
        "parameter_free": "Run the grayscale native parameter-free segment validation stage.",
        "validate_nfa": "Use native false-alarm validation for grayscale lines and fitted round shapes.",
        "minimum_line_length": "-1 uses native automatic length. Otherwise enter 9..8192 pixels.",
        "line_join_distance": "Maximum endpoint gap when joining candidate straight lines.",
        "line_fit_error": "Maximum native error in an initial line fit in pixels.",
        "maximum_line_error": "Maximum error when merging collinear candidates in pixels.",
        "fit_lines": "Fit lines and link them to native edge segment IDs.",
        "fit_round": "Fit circles and ellipses using the native edge paths.",
        "timeout_seconds": "Maximum wait for the isolated native worker.",
    }
    input_help = {
        "image": "Grayscale mode converts to 8-bit. Color mode accepts an 8-bit color image.",
        "mask": "Restrict output maps by pixel, paths by overlap, lines by both endpoints and round fits by center.",
        "region": "Intersect the output selection with the optional mask. Native fitting uses the full source.",
    }
    output_help = {
        "image": "Original input with optional drawable result layers.",
        "edges": "Native uint8 edge map, zero outside the output selection.",
        "gradient": "Native uint16 gradient map, zero outside the output selection.",
        "segments": "Original ordered pixel paths with stable native IDs.",
        "lines": "Line collection. Each line refers to its source path by edge_segment_id.",
        "circles": "Circle collection with fitted center and radius in source pixels.",
        "ellipses": "Ellipse collection with full major/minor axes and angle from the x axis in degrees.",
    }
    preview_safe = process_inputs_readonly = True
    payload_reference = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "edges", "segments", "lines", "circles", "ellipses", "report"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("sum_gradient", "parameter_free", "validate_nfa", "fit_lines", "fit_round"):
            if type(p[name]) is not bool:
                raise ValidationError(f"{name.replace('_', ' ').capitalize()} must be boolean", name)
        minimum = p["minimum_line_length"]
        if type(minimum) is not int or (minimum != -1 and not 9 <= minimum <= 8192):
            raise ValidationError("Minimum line length must be -1 or 9..8192", "minimum_line_length")
        if p["parameter_free"] and p["input_mode"] != "grayscale":
            raise ValidationError("Parameter-free Edge Drawing requires grayscale", "parameter_free")

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 8 or max(height, width) > 4096 or height * width > MAX_PIXELS:
            raise ValidationError("Edge Drawing requires 8..4096 pixels per axis and at most 1,048,576 pixels")
        p = self.parameters
        selected = selection_mask(inputs, source)
        if p["input_mode"] == "color":
            if source.ndim != 3 or source.dtype != np.uint8:
                raise ValidationError("Color Edge Drawing requires an 8-bit color image", "input_mode")
            work = bgr8(source)
        else:
            work = gray_image(source, (np.uint8,), auto_depth=True)
        edge, gradient, lengths, points, rows, line_paths, shapes = native_geometry(
            np.ascontiguousarray(work), p, self._cancelled,
        )
        if (
            edge.shape != (height, width) or edge.dtype != np.uint8
            or gradient.shape != edge.shape or gradient.dtype != np.uint16
            or int(lengths.sum()) != len(points) or len(rows) != len(line_paths)
            or np.any(points < 0) or np.any(points >= (width, height))
        ):
            raise ValidationError("OpenCV returned inconsistent Edge Drawing geometry")
        paths = ordered_paths(lengths, points, selected)
        lines = []
        for index, (row, segment) in enumerate(zip(rows, line_paths.tolist(), strict=True)):
            if (
                segment in paths and np.isfinite(row).all()
                and contains(selected, row[:2]) and contains(selected, row[2:])
            ):
                item = line_item(row, index)
                item["geometry"]["edge_segment_id"] = segment
                lines.append(item)
        circles, ellipses = round_primitives(shapes, selected, width, height)
        if selected is not None:
            edge[selected == 0] = 0
            gradient[selected == 0] = 0
        incoming = self.input_payloads["image"]
        counts = {
            "segments": len(paths), "lines": len(lines), "circles": len(circles), "ellipses": len(ellipses),
            "edge_pixels": int(np.count_nonzero(edge)),
        }
        simple_context = {k: v for k, v in incoming.items() if k not in ("alpha_representation", "raw_sensor")}
        return Result(
            {
                "image": source, "edges": edge, "gradient": gradient,
                "segments": [{"id": index, "points": chain.tolist()} for index, chain in paths.items()],
                "lines": collection("lines", lines, source, incoming),
                "circles": collection("circles", circles, source, incoming),
                "ellipses": collection("ellipses", ellipses, source, incoming),
                "report": {**counts, "fit_lines": p["fit_lines"], "fit_round": p["fit_round"]},
            },
            payloads={name: simple_context for name in ("edges", "gradient")},
            drawings=edge_drawings(paths, lines, circles, ellipses) if self.generate_graphics else {},
            message=f"{len(paths)} paths · {len(lines)} lines · {len(circles)} circles · {len(ellipses)} ellipses",
        )
