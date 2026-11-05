"""Compare and measure ordered contours and detected object collections."""

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .objects import collection, contour_item
from .parameter_validation import NumericRange, ValidationError
from .parameters import CONTOUR_APPROXIMATIONS
from .payloads import NO_VALUE, ValuePacket, require_same_frame
from .processing import finite, integer
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, selection_mask

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


def profile_samples(points, step):
    """Include contour vertices and subdivide each edge to the requested spacing."""
    delta = np.roll(points, -1, axis=0) - points
    steps = np.ceil(np.linalg.norm(delta, axis=1) / step).astype(np.int64)
    count = int(steps.sum())
    if count > 8192:
        raise ValidationError("Contour profile exceeds 8192 samples per contour")
    indices = np.repeat(np.arange(len(points)), steps)
    offsets = steps.cumsum() - steps
    fraction = (np.arange(count) - offsets[indices]) / steps[indices]
    return points[indices] + delta[indices] * fraction[:, None]


class ContourProfileTolerance(Block):
    type_id, title = "inspect.contour_profile", "Contour profile tolerance"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = CompareContours.inputs
    outputs = (
        Port("maximum_deviation", "number"),
        Port("rms_deviation", "number"),
        Port("violation_count", "number"),
        Port("profiles", "data"),
    )
    defaults = {
        "reference_index": 0,
        "sample_index": 0,
        "alignment": "same_frame",
        "sample_spacing": 2.0,
        "tolerance": 2.0,
    }
    parameter_choices = {"alignment": ("same_frame", "pixel_grid")}
    parameter_ranges = {
        "reference_index": NumericRange(0, whole=True),
        "sample_index": NumericRange(0, whole=True),
        "sample_spacing": NumericRange(0.25),
        "tolerance": NumericRange(0),
    }
    parameter_help = {
        "reference_index": CompareContours.parameter_help["reference_index"],
        "sample_index": CompareContours.parameter_help["sample_index"],
        "alignment": "Pixel grid compares coordinates directly for already aligned images.",
        "sample_spacing": "Maximum distance between boundary samples, in pixels.",
        "tolerance": "Maximum absolute deviation in pixels for a passing profile.",
    }

    def process(self, inputs):
        p = self.parameters
        try:
            reference, _, _, ref_payload = read_contour(
                self, inputs, "reference", p["reference_index"]
            )
            sample, _, _, sample_payload = read_contour(self, inputs, "sample", p["sample_index"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        ref_frame = ref_payload.get("spatial")
        sample_frame = sample_payload.get("spatial")
        if ref_frame and sample_frame and ref_frame["image_size"] != sample_frame["image_size"]:
            raise ValidationError("Contour profile requires matching pixel dimensions")
        if p["alignment"] == "same_frame":
            require_same_frame(
                (ValuePacket(None, ref_payload), ValuePacket(None, sample_payload)),
                message="Use the same coordinate frame or an aligned pixel grid",
            )

        spacing = float(p["sample_spacing"])
        ref_points = profile_samples(reference, spacing)
        sample_points = profile_samples(sample, spacing)
        if len(ref_points) * len(sample) + len(sample_points) * len(reference) > 16_000_000:
            raise ValidationError("Contour profile is too dense. Simplify the contours first.")
        tolerance = float(p["tolerance"])
        origin = (reference[0] + sample[0]) / 2
        profiles = {}
        distances = []
        violations = []
        for name, points, target in (
            ("reference_to_sample", ref_points, sample),
            ("sample_to_reference", sample_points, reference),
        ):
            native = (target - origin).astype(np.float32)
            signed = np.array(
                [cv2.pointPolygonTest(native, point, True) for point in (points - origin).tolist()]
            )
            distances.append(signed)
            violations.append(points[np.abs(signed) > tolerance])
            profiles[name] = np.column_stack((points, signed)).tolist()
        combined = np.concatenate(distances)
        outliers = np.concatenate(violations)
        maximum = float(np.max(np.abs(combined)))
        passed = len(outliers) == 0
        outputs = {
            "maximum_deviation": maximum,
            "rms_deviation": float(np.sqrt(np.mean(np.square(combined)))),
            "violation_count": len(outliers),
            "profiles": profiles,
        }
        drawing_source = "sample" if sample_frame or not ref_frame else "reference"
        payload = dict(sample_payload)
        if drawing_source == "reference":
            for key in ("spatial", "pyramid"):
                if key in ref_payload:
                    payload[key] = ref_payload[key]
        drawings = {}
        if self.generate_graphics:
            for name, contour, color in (
                ("reference outline", reference, (0, 190, 240)),
                ("sample outline", sample, ACCEPTED),
            ):
                outline = graphic_object("polyline", contour, color, closed=True)
                drawings[name] = GraphicDrawing((outline,), drawing_source)
            if len(outliers):
                marks = tuple(graphic_object("marker", (point,), REJECTED) for point in outliers)
                drawings["out of tolerance"] = GraphicDrawing(marks, drawing_source)
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            passed=passed,
            message=f"Maximum deviation {maximum:.4g} px, {len(outliers)} violations",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "contour_profile_out_of_tolerance",
                    "message": f"Maximum deviation {maximum:.4g} px exceeds {tolerance:.4g} px",
                }
            ],
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
        Port("maximum_depth", "number"),
    )
    defaults = {
        "item_index": 0,
        "clockwise": False,
        "measure_defects": True,
        "minimum_depth": 0.0,
        "check_defect_count": False,
        "maximum_defects": 0,
    }
    parameter_ranges = {
        "item_index": NumericRange(0, whole=True),
        "minimum_depth": NumericRange(0),
        "maximum_defects": NumericRange(0, whole=True),
    }
    parameter_help = {
        "item_index": "Zero-based collection item. A blob uses its largest boundary.",
        "clockwise": "Hull winding in OpenCV Cartesian coordinates (positive Y up).",
        "measure_defects": "Return concavity measurements for a simple whole-pixel contour.",
        "minimum_depth": "Count concavities at or above this perpendicular depth in pixels.",
        "check_defect_count": "Judge the count after applying the minimum depth.",
        "maximum_defects": "Largest qualifying count that passes. Zero allows no defects.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["check_defect_count"] and not p["measure_defects"]:
            raise ValidationError(
                "Enable Measure defects to check the defect count",
                "measure_defects",
                "check_defect_count",
            )

    def process(self, inputs):
        p = self.parameters
        points, native, area, payload = read_contour(self, inputs, "contour", p["item_index"])
        hull = cv2.convexHull(native, clockwise=p["clockwise"], returnPoints=False).ravel()
        hull_points = points[hull]
        hull_area = cv2.contourArea(native[hull])
        draw = self.generate_graphics
        objects = {name: [] for name in ("qualifying defects", "shallow defects")}
        defects = []
        if p["measure_defects"]:
            if not np.array_equal(points, np.rint(points)):
                raise ValidationError("Convexity defects require whole-pixel contour coordinates")
            minimum = float(p["minimum_depth"])
            raw = cv2.convexityDefects(points.astype(np.int32), hull)
            if raw is not None:
                for start, end, deepest, depth in raw.reshape(-1, 4):
                    depth_pixels = float(depth) / 256
                    qualifies = depth_pixels >= minimum
                    if qualifies:
                        defects.append(
                            {
                                "start": points[start].tolist(),
                                "end": points[end].tolist(),
                                "deepest": points[deepest].tolist(),
                                "depth": depth_pixels,
                            }
                        )
                    if draw:
                        a, b, tip = points[[start, end, deepest]]
                        edge = b - a
                        foot = a + edge * ((tip - a) @ edge / (edge @ edge))
                        layer = "qualifying defects" if qualifies else "shallow defects"
                        color = REJECTED if qualifies else (0, 190, 240)
                        objects[layer].append(graphic_object("polyline", (foot, tip), color))
        count = len(defects)
        maximum = int(p["maximum_defects"])
        passed = not p["check_defect_count"] or count <= maximum
        outputs = {
            "hull": hull_points.tolist(),
            "defects": defects,
            "count": count,
            "solidity": area / hull_area,
            "maximum_depth": max((row["depth"] for row in defects), default=0.0),
        }
        drawings = {}
        if draw:
            outline = graphic_object("polyline", hull_points, ACCEPTED, closed=True)
            drawings["convex hull"] = GraphicDrawing((outline,), "contour")
            drawings.update(
                {
                    name: GraphicDrawing(tuple(graphics), "contour")
                    for name, graphics in objects.items()
                    if graphics
                }
            )
        return Result(
            outputs,
            payloads=dict.fromkeys(outputs, payload),
            drawings=drawings,
            passed=passed,
            message=f"{count} qualifying convexity defects"
            if p["measure_defects"]
            else "Convex hull",
            fail_reasons=[]
            if passed
            else [
                {
                    "code": "too_many_convexity_defects",
                    "message": f"Found {count} qualifying defects, maximum {maximum}",
                }
            ],
        )


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


class ContourHierarchy(Block):
    type_id, title = "inspect.contour_hierarchy", "Contour hierarchy"
    category, block_type, subtype = "Inspection", "Inspection", "Shapes"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("hierarchy", "data"),
        Port("collection", "data"),
        Port("count", "number"),
        Port("hole_count", "number"),
        Port("overlay", required=False),
    )
    defaults = {"approximation": "simple", "draw_overlay": True}
    parameter_choices = {"approximation": tuple(CONTOUR_APPROXIMATIONS)}
    input_help = {"image": "Monochrome binary image. Every nonzero pixel is foreground."}

    def process(self, inputs):
        source = inputs["image"]
        if source.ndim != 2 or source.dtype not in (np.uint8, np.bool_):
            raise ValidationError("Contour hierarchy requires a monochrome uint8 or boolean mask")
        if max(source.shape) > 16384 or source.size > 64_000_000:
            raise ValidationError("Contour hierarchy is limited to 64M pixels and 16384 per axis")
        binary = source != 0
        mask = selection_mask(inputs, source)
        if mask is not None:
            binary &= mask != 0
        binary = binary.astype(np.uint8) * 255
        contours, links = cv2.findContours(
            binary, cv2.RETR_TREE, CONTOUR_APPROXIMATIONS[self.parameters["approximation"]]
        )
        if len(contours) > 100_000:
            raise ValidationError("Contour hierarchy supports up to 100,000 contours")

        rows, roots = [], []
        for index, link in enumerate(links[0] if links is not None else ()):
            next_id, previous, first_child, parent = (int(i) if i >= 0 else None for i in link)
            rows.append(
                {
                    "id": index,
                    "next_id": next_id,
                    "previous_id": previous,
                    "first_child_id": first_child,
                    "parent_id": parent,
                    "child_ids": [],
                }
            )
            if parent is None:
                roots.append(index)
        for row in rows:
            if row["parent_id"] is not None:
                rows[row["parent_id"]]["child_ids"].append(row["id"])

        holes = 0
        pending = [(index, 0) for index in roots]
        while pending:
            index, depth = pending.pop()
            row = rows[index]
            row["depth"] = depth
            row["kind"] = "hole" if depth % 2 else "island" if depth else "outer"
            holes += int(row["kind"] == "hole")
            pending.extend((child, depth + 1) for child in row["child_ids"])

        overlay = overlay_image(binary) if self.parameters["draw_overlay"] else NO_VALUE
        draw = self.generate_graphics
        objects = []
        if overlay is not NO_VALUE or draw:
            colors = {"outer": (80, 230, 140), "hole": (80, 145, 255), "island": (240, 225, 60)}
            for contour, row in zip(contours, rows, strict=True):
                color = colors[row["kind"]]
                if overlay is not NO_VALUE:
                    cv2.drawContours(overlay, [contour], -1, color, 2)
                if draw:
                    objects.append(
                        graphic_object("polyline", contour.reshape(-1, 2), color, closed=True)
                    )

        payload = self.input_payloads["image"]
        items = [contour_item(contour, i, links[0][i]) for i, contour in enumerate(contours)]
        return Result(
            {
                "hierarchy": {"model": "contour_hierarchy_v1", "root_ids": roots, "nodes": rows},
                "collection": collection("contours", items, source, payload),
                "count": len(rows),
                "hole_count": holes,
                "overlay": overlay,
            },
            drawings={"contour hierarchy": GraphicDrawing(tuple(objects))} if objects else {},
            message=f"{len(rows)} contours across {len(roots)} outer boundaries · {holes} holes",
        )


CONTOUR_ANALYSIS_BLOCKS = (
    CompareContours,
    ContourProfileTolerance,
    ContourHull,
    TestContourPoints,
    ApproximateContour,
    ContourHierarchy,
)
