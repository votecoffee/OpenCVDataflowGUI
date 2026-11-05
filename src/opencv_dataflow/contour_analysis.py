"""Compare and measure ordered contours and detected object collections."""

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .payloads import ValuePacket, require_same_frame
from .processing import finite, integer

MATCH_METHODS = {
    "i1": cv2.CONTOURS_MATCH_I1,
    "i2": cv2.CONTOURS_MATCH_I2,
    "i3": cv2.CONTOURS_MATCH_I3,
}


def read_contour(block, inputs, name, item_index):
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
        or not 3 <= len(contour) <= 100_000
        or not np.isfinite(contour).all()
        or np.max(np.abs(contour)) > 1_000_000
    ):
        raise ValueError("Contour requires 3–100,000 ordered finite [x,y] points")
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


CONTOUR_ANALYSIS_BLOCKS = (CompareContours, ContourHull, TestContourPoints)
