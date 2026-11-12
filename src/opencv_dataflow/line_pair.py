"""Frame-aware corner and transverse gap measurements from image line pairs."""

import math

import cv2
import numpy as np

from .collection_tools import read_collection
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .result_graphics import ACCEPTED, REJECTED
from .selection import selection_mask
from .tickets import ticket_key
from .workplane import LENGTH_UNITS, require_workplane_scale


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


class _MissingLine(ValueError):
    pass


def _line(block, inputs, port, target):
    value = inputs[port]
    payload = block.input_payloads[port]
    identifier = None
    origin = [0, 0]
    if isinstance(value, dict) and value.get("model") == "vision_collection_v1":
        data, payload = read_collection(block, inputs, port)
        if data["kind"] != "lines":
            raise ValidationError(f"{port} requires a lines collection")
        index = int(block.parameters[f"index_{port[-1]}"])
        if index >= len(data["items"]):
            raise _MissingLine(f"{port} has no line at the selected index")
        item = data["items"][index]
        value, identifier, origin = item["geometry"].get("points"), item["id"], data["origin"]
        if "spatial" not in payload and data["image_size"] != target["image_size"]:
            raise ValidationError("Unframed lines must use the image's pixel grid")
    elif isinstance(value, dict) and value.get("shape") == "line":
        value = value.get("endpoints")
    else:
        raise ValidationError(f"{port} requires a fitted line or a lines collection")
    try:
        points = np.asarray(value, np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Line endpoints must be numeric [x,y] pairs") from exc
    if points.shape != (2, 2) or not np.isfinite(points).all() or np.max(np.abs(points)) > 1e7:
        raise ValidationError("Each line needs two finite endpoints within coordinate limits")
    source = payload.get("spatial")
    if source is None:
        points = points - origin
    elif source != target:
        if (
            source.get("root_id") is None
            or source["root_id"] != target.get("root_id")
            or source["root_size"] != target["root_size"]
        ):
            raise ValidationError("Lines and image require compatible coordinate frames")
        transform = np.linalg.solve(np.asarray(target["to_root"]), source["to_root"])
        mapped = np.column_stack((points, np.ones(2))) @ transform.T
        if not (np.all(mapped[:, 2] > 1e-12) or np.all(mapped[:, 2] < -1e-12)):
            raise ValidationError("Line support crosses the coordinate-frame horizon")
        points = mapped[:, :2] / mapped[:, 2, None]
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e7:
        raise ValidationError("Mapped line endpoints exceed coordinate limits")
    delta = points[1] - points[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValidationError("Line endpoints must be distinct")
    direction = delta / length
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return points, direction, payload, identifier


def _extension(point, endpoints, direction):
    station = float((point - endpoints[0]) @ direction)
    end = float((endpoints[1] - endpoints[0]) @ direction)
    lo, hi = sorted((0.0, end))
    return max(lo - station, station - hi, 0.0)


def _covered(paths, mask):
    """Require complete path coverage plus a pixel of surrounding support."""
    points = np.vstack(paths)
    if np.any(points.min(axis=0) < 1) or np.any(
        points.max(axis=0) > np.array(mask.shape[::-1]) - 2
    ):
        return False
    footprint = np.zeros(mask.shape, np.uint8)
    for path in paths:
        cv2.polylines(
            footprint,
            [np.rint(path * 256).astype(np.int32)],
            False,
            1,
            1,
            lineType=cv2.LINE_8,
            shift=8,
        )
    footprint = cv2.dilate(footprint, np.ones((3, 3), np.uint8))
    return bool(np.all(mask[footprint != 0] != 0))


class MeasureLinePair(Block):
    type_id, title = "inspect.line_pair", "Measure line pair"
    category = block_type = "Inspection"
    subtype = "Measurements"
    payload_reference = "image"
    process_inputs_readonly = True
    parallel_safe = True
    inputs = (
        Port("image"),
        Port("line_a", "data"),
        Port("line_b", "data"),
        Port("scale", "data", required=False, required_when_connected=True),
        Port("valid_mask", visual_type="region", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("measurements", "data"),
        Port("vertex", "data"),
        Port("vertex_pixels", "data"),
        Port("angle", "number"),
        Port("gap", "number"),
    )
    defaults = {
        "mode": "corner",
        "angle_mode": "acute",
        "units": "px",
        "index_a": 0,
        "index_b": 0,
        "minimum_intersection_angle": 1.0,
        "maximum_extension": 0.0,
        "require_vertex_in_image": True,
        "maximum_parallel_angle": 1.0,
        "minimum_overlap": 0.0,
        "check_angle": False,
        "target_angle": 90.0,
        "angle_tolerance": 2.0,
        "check_gap": False,
        "minimum_gap": 0.0,
        "maximum_gap": 0.0,
    }
    parameter_choices = {
        "mode": ("corner", "angle", "parallel_gap"),
        "angle_mode": ("acute", "obtuse"),
        "units": ("px", *LENGTH_UNITS),
    }
    parameter_ranges = {
        "index_a": NumericRange(0, 99999, whole=True),
        "index_b": NumericRange(0, 99999, whole=True),
        "minimum_intersection_angle": NumericRange(0.000001, 90),
        "maximum_extension": NumericRange(0, 1e9),
        "maximum_parallel_angle": NumericRange(0, 45),
        "minimum_overlap": NumericRange(0, 1e9),
        "target_angle": NumericRange(0, 180),
        "angle_tolerance": NumericRange(0, 180),
        "minimum_gap": NumericRange(0, 1e9),
        "maximum_gap": NumericRange(0, 1e9),
    }
    parameter_help = {
        "mode": "Measure a corner, angle alone, or transverse gap over common support.",
        "angle_mode": "Acute reports 0..90 degrees; Obtuse reports the 90..180 supplement.",
        "units": "Physical units require a matching Rectify workplane scale and validity mask.",
        "index_a": "Zero-based entry when Line A receives a lines collection.",
        "index_b": "Zero-based entry when Line B receives a lines collection.",
        "minimum_intersection_angle": "Reject nearly parallel, poorly conditioned corner fits.",
        "maximum_extension": "Maximum segment extension in selected units; zero disables.",
        "require_vertex_in_image": "Reject a corner outside the current image.",
        "maximum_parallel_angle": "Maximum acute angle allowed for gap measurement, in degrees.",
        "minimum_overlap": "Minimum shared longitudinal support, in the selected length units.",
        "check_angle": "Apply target-angle tolerance after geometric qualification.",
        "target_angle": "Target for the selected acute or obtuse angle, in degrees.",
        "angle_tolerance": "Inclusive absolute deviation from the target angle.",
        "check_gap": "Require the entire measured overlap to meet the gap limits.",
        "minimum_gap": "Inclusive minimum separation in the selected length units.",
        "maximum_gap": "Inclusive maximum separation. Zero disables the upper limit.",
    }
    input_help = {
        "image": "Grid for geometry, coverage and graphics; pixels are passed through.",
        "line_a": "Line fit from Fit shape, or a lines collection with a selected index.",
        "line_b": "Second fit or collection. Compatible frame transforms are applied.",
        "scale": "Uniform physical scale and origin from Rectify workplane.",
        "valid_mask": "Optional coverage mask; required for physical measurements.",
        "region": "Optional inspection region containing all measured paths.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("require_vertex_in_image", "check_angle", "check_gap"):
            if type(p[key]) is not bool:
                raise ValidationError(f"{key} must be a checkbox value", key)
        if p["maximum_gap"] and p["maximum_gap"] < p["minimum_gap"]:
            raise ValidationError("Maximum gap must be at least the minimum or zero", "maximum_gap")

    def process(self, inputs):
        image = image_value(inputs["image"])
        height, width = image.shape[:2]
        if image.shape[0] * image.shape[1] > 8_388_608 or max(height, width) >= 32767:
            raise ValidationError("Line measurement image exceeds the supported grid limits")
        payload = self.input_payloads["image"]
        space = payload["spatial"]
        try:
            a, da, pa, ia = _line(self, inputs, "line_a", space)
            b, db, pb, ib = _line(self, inputs, "line_b", space)
        except _MissingLine as exc:
            return Result(
                {
                    "image": inputs["image"],
                    "measurements": {
                        "model": "line_pair_v1",
                        "status": "unmeasurable",
                        "mode": self.parameters["mode"],
                        "units": self.parameters["units"],
                        "within_limits": False,
                    },
                    "vertex": NO_VALUE,
                    "vertex_pixels": NO_VALUE,
                    "angle": NO_VALUE,
                    "gap": NO_VALUE,
                },
                passed=False,
                fail_reasons=[{"code": "line_pair_missing_line", "message": str(exc)}],
                message=str(exc),
                drawings={
                    "Measurement": GraphicDrawing(
                        (
                            graphic_object(
                                "label", [[width / 2, height / 2]], REJECTED, text=str(exc)
                            ),
                        )
                    )
                }
                if self.generate_graphics
                else {},
            )
        contexts = [
            payload,
            pa,
            pb,
            *(
                self.input_payloads[key]
                for key in ("scale", "valid_mask", "region")
                if key in inputs
            ),
        ]
        tickets = [p["ticket"] for p in contexts if "ticket" in p]
        if tickets and any(ticket_key(t) != ticket_key(tickets[0]) for t in tickets[1:]):
            raise ValidationError("Line measurements and image must share one ticket")
        context = {**(pa if "ticket" in pa else pb), **payload}
        context.pop("pyramid", None)
        if "pyramid" in payload:
            context["pyramid"] = payload["pyramid"]
        p = self.parameters
        units, step, origin = p["units"], 1.0, np.zeros(2)
        if units != "px":
            if "scale" not in inputs or "valid_mask" not in inputs:
                raise ValidationError("Physical measurements require Scale and Valid mask")
            actual, step = require_workplane_scale(
                inputs["scale"], payload, self.input_payloads["scale"]
            )
            if actual != units:
                raise ValidationError(
                    "Workplane scale units must match the selected units", "units"
                )
            origin = np.asarray(inputs["scale"].get("origin"), np.float64)
            if origin.shape != (2,) or not np.isfinite(origin).all():
                raise ValidationError("Workplane origin must be a finite [x,y] point")
        selection = selection_mask(
            {
                "image": inputs["image"],
                "mask": inputs.get("valid_mask"),
                "region": inputs.get("region"),
            },
            inputs["image"],
        )
        sine, cosine = abs(_cross(da, db)), abs(float(da @ db))
        acute = math.degrees(math.atan2(sine, cosine))
        angle = acute if p["angle_mode"] == "acute" else 180.0 - acute
        report = {
            "model": "line_pair_v1",
            "mode": p["mode"],
            "units": units,
            "status": "unmeasurable",
            "acute_angle_degrees": acute,
            "condition_number": (1 + min(cosine, 1.0)) / sine if sine else None,
            "source_ids": [ia, ib],
            "line_a_pixels": a.tolist(),
            "line_b_pixels": b.tolist(),
        }
        reasons, paths, vertex, gap = [], [a, b], None, None
        gap_points = None

        def reject(code, message):
            reasons.append({"code": f"line_pair_{code}", "message": message})

        if p["mode"] == "corner":
            if acute < p["minimum_intersection_angle"]:
                reject("parallel", "The line pair is too nearly parallel for a reliable corner")
            else:
                vertex = a[0] + _cross(b[0] - a[0], db) / _cross(da, db) * da
                extensions = [_extension(vertex, a, da) * step, _extension(vertex, b, db) * step]
                report.update(candidate_vertex_pixels=vertex.tolist(), extension=extensions)
                if not np.isfinite(vertex).all() or np.max(np.abs(vertex)) > 1e7:
                    reject("unstable_intersection", "The intersection exceeds coordinate limits")
                elif p["require_vertex_in_image"] and (
                    np.any(vertex < 0) or np.any(vertex > [width - 1, height - 1])
                ):
                    reject("outside_image", "The intersection lies outside the current image")
                elif p["maximum_extension"] and max(extensions) > p["maximum_extension"]:
                    reject(
                        "extrapolation", "The intersection exceeds the allowed segment extension"
                    )
                else:
                    paths += [
                        np.array([edge[np.argmin(np.linalg.norm(edge - vertex, axis=1))], vertex])
                        for edge in (a, b)
                    ]
        elif p["mode"] == "parallel_gap":
            if acute > p["maximum_parallel_angle"]:
                reject("not_parallel", "The line pair exceeds the allowed parallel angle")
            else:
                aa, bb = (edge @ da for edge in (a, b))
                lo, hi = max(min(aa), min(bb)), min(max(aa), max(bb))
                overlap = (hi - lo) * step
                report["overlap"] = max(0.0, overlap)
                if hi - lo <= 1e-9 or overlap < p["minimum_overlap"]:
                    reject("no_overlap", "The lines have insufficient shared longitudinal support")
                else:
                    normal = np.array([-da[1], da[0]])
                    samples = a[0] + (np.array([lo, (lo + hi) / 2, hi]) - a[0] @ da)[:, None] * da
                    offsets = np.array(
                        [_cross(b[0] - point, db) / _cross(normal, db) for point in samples]
                    )
                    other = samples + offsets[:, None] * normal
                    paths = [samples[[0, 2]], other[[0, 2]], np.array([samples[1], other[1]])]
                    gap_points = paths[-1]
                    gap = abs(float(offsets[1])) * step
                    minimum = (
                        0.0 if offsets[0] * offsets[-1] <= 0 else float(min(abs(offsets[[0, 2]])))
                    )
                    report.update(
                        gap=gap,
                        gap_min=minimum * step,
                        gap_max=float(max(abs(offsets[[0, 2]]))) * step,
                        signed_gap=float(offsets[1]) * step,
                    )
        if not reasons and selection is not None and not _covered(paths, selection):
            reject("invalid_coverage", "The measured paths touch invalid coverage or region bounds")
        measurable = not reasons
        if measurable:
            report.update(status="measured", angle_degrees=angle)
            if vertex is not None:
                report["vertex"] = (origin + vertex * step).tolist()
            if p["check_angle"] and abs(angle - p["target_angle"]) > p["angle_tolerance"] + 1e-10:
                reject("angle_out_of_range", f"Angle {angle:.4g} degrees is outside tolerance")
            if (
                p["mode"] == "parallel_gap"
                and p["check_gap"]
                and (
                    report["gap_min"] < p["minimum_gap"] - 1e-10
                    or (p["maximum_gap"] and report["gap_max"] > p["maximum_gap"] + 1e-10)
                )
            ):
                reject("gap_out_of_range", "Separation across the overlap is outside gap limits")
        passed = not reasons
        report["within_limits"] = passed
        message = (
            f"{angle:.4g} degrees"
            if measurable and gap is None
            else f"{gap:.4g} {units} gap, {angle:.4g} degrees"
            if measurable
            else reasons[0]["message"]
        )
        drawings = {}
        if self.generate_graphics:
            color = ACCEPTED if passed else REJECTED
            drawings["Line support"] = GraphicDrawing(
                tuple(graphic_object("polyline", edge, color) for edge in (a, b))
            )
            marks = []
            if measurable and vertex is not None:
                marks.append(graphic_circle(*vertex, 5, color))
                ray = db if da @ db >= 0 else -db
                if p["angle_mode"] == "obtuse":
                    ray = -ray
                start = math.atan2(da[1], da[0])
                sweep = math.atan2(_cross(da, ray), float(da @ ray))
                angles = start + np.linspace(0, sweep, 25)
                radius = min(24.0, np.linalg.norm(a[1] - a[0]) / 4, np.linalg.norm(b[1] - b[0]) / 4)
                marks.append(
                    graphic_object(
                        "polyline",
                        vertex + radius * np.column_stack((np.cos(angles), np.sin(angles))),
                        color,
                    )
                )
            elif measurable and gap_points is not None:
                marks.append(graphic_object("polyline", gap_points, color))
            anchor = (
                vertex
                if measurable and vertex is not None
                else (
                    gap_points.mean(axis=0)
                    if measurable and gap_points is not None
                    else a.mean(axis=0)
                )
            )
            marks.append(graphic_object("label", [anchor], color, text=message))
            drawings["Measurement"] = GraphicDrawing(tuple(marks))
        outputs = {
            "image": inputs["image"],
            "measurements": report,
            "vertex": report["vertex"] if measurable and vertex is not None else NO_VALUE,
            "vertex_pixels": vertex.tolist() if measurable and vertex is not None else NO_VALUE,
            "angle": angle if measurable else NO_VALUE,
            "gap": gap if measurable and gap is not None else NO_VALUE,
        }
        payloads = dict.fromkeys(outputs, context)
        payloads["vertex"] = {**context, "coordinates": {"units": units, "space": "image"}}
        if units != "px":
            payloads["vertex"] = {
                key: value for key, value in context.items() if key not in ("spatial", "pyramid")
            }
            payloads["vertex"]["coordinates"] = {
                "space": "rectified_workplane",
                "units": units,
                "origin": origin.tolist(),
                "units_per_pixel": step,
                "pixel_spatial": space,
            }
        return Result(
            outputs,
            payloads=payloads,
            passed=passed,
            fail_reasons=reasons,
            message=message,
            drawings=drawings,
        )
