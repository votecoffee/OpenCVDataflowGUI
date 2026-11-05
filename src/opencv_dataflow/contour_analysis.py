"""Compare and measure ordered contours and detected object collections."""

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import finite, integer

MATCH_METHODS = {
    "i1": cv2.CONTOURS_MATCH_I1,
    "i2": cv2.CONTOURS_MATCH_I2,
    "i3": cv2.CONTOURS_MATCH_I3,
}


def read_ordered_points(block, inputs, name, item_index, minimum=3):
    source = inputs[name]
    payload = block.input_payloads.get(name, {})
    if isinstance(source, dict):
        collection, payload = read_collection(block, inputs, name)
        items = collection["items"]
        index = integer(item_index, "Contour item index", 0, len(items) - 1)
        geometry = items[index]["geometry"]
        source = geometry.get("points")
        if source is None:
            source = max(
                geometry["contours"],
                key=lambda contour: cv2.contourArea(np.asarray(contour, np.float32)),
            )
    contour = np.asarray(source, np.float64)
    if (
        contour.ndim != 2
        or contour.shape[1] != 2
        or not minimum <= len(contour) <= 100_000
        or not np.isfinite(contour).all()
        or np.max(np.abs(contour)) > 1_000_000
    ):
        raise ValueError(f"Contour requires {minimum}–100,000 ordered finite [x,y] points")
    return contour, payload


def read_contour(block, inputs, name, item_index):
    contour, payload = read_ordered_points(block, inputs, name, item_index)
    if np.array_equal(contour[0], contour[-1]):
        contour = contour[:-1]
    native = contour.astype(np.float32)
    area = cv2.contourArea(native) if len(native) >= 3 else 0.0
    if area <= 0:
        raise ValueError("Contour must enclose a nonzero area")
    return contour, native, area, payload


class CompareContours(Block):
    type_id, title = "inspect.compare_contours", "Compare contours (Hu moments)"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("reference", "data"), Port("sample", "data"))
    outputs = (Port("distance", "number"),)
    defaults = {
        "method": "i2",
        "reference_index": 0,
        "sample_index": 0,
        "check_distance": False,
        "maximum_distance": 0.1,
    }
    parameter_choices = {"method": tuple(MATCH_METHODS)}
    parameter_help = {
        "reference_index": "Zero-based reference collection item. Ignored for point-list inputs.",
        "sample_index": "Zero-based sample collection item. Ignored for point-list inputs.",
        "maximum_distance": "Maximum Hu-moment distance. Smaller values indicate closer shapes.",
    }

    def process(self, inputs):
        p = self.parameters
        _, reference, _, _ = read_contour(self, inputs, "reference", p["reference_index"])
        _, sample, _, _ = read_contour(self, inputs, "sample", p["sample_index"])
        distance = cv2.matchShapes(reference, sample, MATCH_METHODS[p["method"]], 0.0)
        if not np.isfinite(distance):
            raise ValueError("OpenCV returned a nonfinite contour distance")
        passed = not p["check_distance"] or distance <= finite(
            p["maximum_distance"], "Maximum distance", 0
        )
        return Result(
            {"distance": distance},
            passed=passed,
            message=f"Hu contour distance {distance:.6g}",
            fail_reasons=[] if passed else ["contour_distance_out_of_tolerance"],
        )


class ContourHull(Block):
    type_id, title = "inspect.contour_hull", "Contour hull and convexity defects"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("contour", "data"),)
    outputs = (
        Port("hull", "data"),
        Port("defects", "data"),
        Port("count", "number"),
        Port("solidity", "number"),
    )
    defaults = {
        "item_index": 0,
        "clockwise": False,
        "measure_defects": True,
        "minimum_depth": 0.0,
    }
    parameter_help = {
        "item_index": "Zero-based collection item. A blob uses its largest boundary.",
        "clockwise": "Hull winding in OpenCV Cartesian coordinates (positive Y up).",
        "measure_defects": "Return concavity measurements for a simple whole-pixel contour.",
        "minimum_depth": "Ignore convexity defects shallower than this many pixels.",
    }

    def process(self, inputs):
        p = self.parameters
        points, native, area, payload = read_contour(self, inputs, "contour", p["item_index"])
        hull = cv2.convexHull(native, clockwise=p["clockwise"], returnPoints=False).ravel()
        hull_area = cv2.contourArea(native[hull])
        defects = []
        if p["measure_defects"]:
            if not np.array_equal(points, np.rint(points)):
                raise ValueError("Convexity defects require whole-pixel contour coordinates")
            minimum = finite(p["minimum_depth"], "Minimum depth", 0)
            raw = cv2.convexityDefects(points.astype(np.int32), hull)
            if raw is not None:
                for start, end, deepest, depth in raw.reshape(-1, 4):
                    depth_pixels = float(depth) / 256
                    if depth_pixels >= minimum:
                        defects.append(
                            {
                                "start": points[start].tolist(),
                                "end": points[end].tolist(),
                                "deepest": points[deepest].tolist(),
                                "depth": depth_pixels,
                            }
                        )
        outputs = {
            "hull": points[hull].tolist(),
            "defects": defects,
            "count": len(defects),
            "solidity": area / hull_area,
        }
        return Result(outputs, payloads={name: payload for name in outputs})


class TestContourPoints(Block):
    type_id, title = "inspect.contour_points", "Test points against contour"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("contour", "data"), Port("points", "data"))
    outputs = (Port("measurements", "data"), Port("points", "data"), Port("count", "number"))
    defaults = {"item_index": 0, "mode": "signed_distance", "minimum_distance": 0.0}
    parameter_choices = {"mode": ("signed_distance", "location")}
    parameter_help = {
        "minimum_distance": "Minimum signed distance in pixels. Outside distances are negative.",
        "mode": "Return signed pixel distances or only classify points as inside, edge or outside.",
    }

    def process(self, inputs):
        p = self.parameters
        _, contour, _, contour_payload = read_contour(self, inputs, "contour", p["item_index"])
        payload = self.input_payloads.get("points", {})
        require_same_frame((ValuePacket(None, contour_payload), ValuePacket(None, payload)))
        points = np.asarray(inputs["points"], np.float64)
        if points.shape == (0,):
            points = points.reshape(0, 2)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) > 100_000
            or not np.isfinite(points).all()
            or (np.abs(points) > 1_000_000).any()
        ):
            raise ValueError("Contour testing requires finite [x,y] points")
        distances = p["mode"] == "signed_distance"
        minimum = finite(p["minimum_distance"], "Minimum signed distance") if distances else 0
        records, accepted = [], []
        for point in points.tolist():
            value = cv2.pointPolygonTest(contour, point, distances)
            location = "inside" if value > 0 else "outside" if value < 0 else "edge"
            keep = value >= minimum
            records.append(
                {
                    "point": point,
                    "location": location,
                    "signed_distance": value if distances else None,
                    "accepted": keep,
                }
            )
            if keep:
                accepted.append(point)
        outputs = {"measurements": records, "points": accepted, "count": len(accepted)}
        context = {**contour_payload, **payload}
        return Result(outputs, payloads={name: context for name in outputs})


class ApproximateContour(Block):
    type_id, title = "inspect.approximate_contour", "Approximate contour"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("contour", "data"), Port("image", required=False))
    outputs = (
        Port("points", "data"),
        Port("vertex_count", "number"),
        Port("measurements", "data"),
        Port("overlay", required=False),
    )
    defaults = {
        "item_index": 0,
        "closed": True,
        "epsilon_mode": "pixels",
        "epsilon_pixels": 2.0,
        "length_fraction": 0.01,
    }
    parameter_choices = {"epsilon_mode": ("pixels", "length_fraction")}
    parameter_ranges = {
        "item_index": NumericRange(0, whole=True),
        "epsilon_pixels": NumericRange(0, 1_000_000),
        "length_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "item_index": "Zero-based contour collection item. A blob uses its largest boundary.",
        "closed": "Connect the last vertex to the first. Disable for an open path.",
        "epsilon_pixels": "Maximum approximation tolerance when Epsilon Mode is pixels.",
        "length_fraction": "Tolerance as a fraction of the original perimeter or path length.",
    }
    input_help = {"image": "Optional aligned background for viewing the simplified contour."}

    def process(self, inputs):
        p = self.parameters
        closed = p["closed"]
        try:
            if closed:
                _, native, _, payload = read_contour(self, inputs, "contour", p["item_index"])
            else:
                points, payload = read_ordered_points(self, inputs, "contour", p["item_index"], 2)
                native = points.astype(np.float32)
                if np.array_equal(native[0], native[-1]):
                    raise ValueError("An open contour requires distinct first and last points")
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        original_length = cv2.arcLength(native, closed)
        epsilon = (
            float(p["epsilon_pixels"])
            if p["epsilon_mode"] == "pixels"
            else original_length * float(p["length_fraction"])
        )
        simplified = cv2.approxPolyDP(native, epsilon, closed).reshape(-1, 2)
        count = len(simplified)
        area = float(cv2.contourArea(simplified)) if closed and count >= 3 else 0.0
        passed = area > 0 if closed else count >= 2
        overlay = NO_VALUE
        drawings = {}
        image = inputs.get("image")
        drawing_source = "contour"
        if image is not None:
            size = [image.shape[1], image.shape[0]]
            if payload.get("spatial", {}).get("image_size", size) != size:
                raise ValidationError("Overlay image dimensions must match the contour frame")
            try:
                require_same_frame((image, ValuePacket(None, payload)))
            except ValueError as exc:
                raise ValidationError("Overlay image must share the contour frame") from exc
            overlay = overlay_image(image)
            drawing_source = "image"
        if passed and (image is not None or self.generate_graphics):
            graphic = graphic_object("polyline", simplified, (70, 220, 140), closed=closed)
            if image is not None:
                draw_object(overlay, graphic, np.eye(3))
            if self.generate_graphics:
                drawings["simplified contour"] = GraphicDrawing((graphic,), source=drawing_source)
        outputs = {
            "points": simplified.tolist(),
            "vertex_count": count,
            "measurements": {
                "closed": closed,
                "source_vertices": len(native),
                "source_length": original_length,
                "length": cv2.arcLength(simplified, closed),
                "area": area if closed else None,
                "epsilon_pixels": epsilon,
            },
            "overlay": overlay,
        }
        return Result(
            outputs,
            payloads={name: payload for name in outputs if name != "overlay" and payload},
            drawings=drawings,
            passed=passed,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "contour_approximation_degenerate",
                    "message": "The simplified contour is degenerate at this tolerance",
                }
            ],
            message=f"Simplified {len(native)} vertices to {count} at {epsilon:.3g} px tolerance",
        )


CONTOUR_ANALYSIS_BLOCKS = (CompareContours, ContourHull, TestContourPoints, ApproximateContour)
