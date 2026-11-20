"""Inspect a metric point scan against the exact surface of a reference mesh."""

import math

import numpy as np

from .cloud_comparison import deviation_statistics
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D
from .mesh_distance import TriangleSurface
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud

_UNSUPPORTED, _BOUNDARY, _IN_TOLERANCE, _DEFECT, _NO_SIGN = range(5)
_MAX_POINTS = 1_000_000
_MAX_TRIANGLES = 250_000


def _values(array):
    return [value if math.isfinite(value) else None for value in array.tolist()]


def _failure(code, message):
    return {"code": code, "message": message}


def _classify(matches, p):
    supported = (matches.face >= 0) & ~matches.boundary
    signed = np.isfinite(matches.signed)
    if p["comparison"] == "normal":
        tested = signed
        good = (matches.signed >= -float(p["negative_tolerance_m"])) & (
            matches.signed <= float(p["positive_tolerance_m"])
        )
    else:
        tested = supported
        good = matches.distance <= float(p["distance_tolerance_m"])
    states = np.full(len(supported), _UNSUPPORTED, np.uint8)
    states[matches.boundary] = _BOUNDARY
    states[supported & ~tested] = _NO_SIGN
    states[tested] = np.where(good[tested], _IN_TOLERANCE, _DEFECT)
    return states, supported, tested


def _review(scan, states, signed):
    palette = np.array(
        [
            [170, 170, 170],
            [241, 180, 55],
            [85, 211, 137],
            [240, 77, 67],
            [160, 105, 220],
        ],
        np.uint8,
    )
    colors = palette[states]
    if signed is not None:
        colors[(states == _DEFECT) & (signed < 0)] = [78, 147, 242]
    display = PointCloud(scan.points, scan.units, scan.frame, colors=colors)
    return render_cloud(display)


class CompareScanToMesh(Tool3D):
    type_id = "3d.scan_mesh_compare"
    title, subtype = "Compare scan to reference mesh", "Mesh geometry"
    description = "Measure scan deviations from triangles and isolate open boundaries."
    payload_reference = "scan"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (Port("scan", "cloud"), Port("reference", "mesh"))
    outputs = (
        Port("defects", "cloud"),
        Port("boundary", "cloud"),
        Port("unsupported", "cloud"),
        Port("correspondences", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "comparison": "distance",
        "maximum_distance_m": 0.01,
        "distance_tolerance_m": 0.002,
        "negative_tolerance_m": 0.002,
        "positive_tolerance_m": 0.002,
        "minimum_support_fraction": 0.8,
        "minimum_in_tolerance_fraction": 1.0,
        "minimum_signed_fraction": 0.95,
        "maximum_shared_normal_angle_deg": 30.0,
        "max_points": 250_000,
        "max_triangles": 100_000,
        "max_work": 10_000_000,
    }
    parameter_choices = {"comparison": ("distance", "normal")}
    parameter_ranges = {
        "maximum_distance_m": NumericRange(1e-9, 1e12),
        "distance_tolerance_m": NumericRange(0, 1e12),
        "negative_tolerance_m": NumericRange(0, 1e12),
        "positive_tolerance_m": NumericRange(0, 1e12),
        "minimum_support_fraction": NumericRange(0, 1),
        "minimum_in_tolerance_fraction": NumericRange(0, 1),
        "minimum_signed_fraction": NumericRange(0, 1),
        "maximum_shared_normal_angle_deg": NumericRange(0, 180),
        "max_points": NumericRange(1, _MAX_POINTS, whole=True),
        "max_triangles": NumericRange(1, _MAX_TRIANGLES, whole=True),
        "max_work": NumericRange(1, 200_000_000, whole=True),
    }
    parameter_help = {
        "comparison": "Judge Euclidean distance or signed distance along reference winding.",
        "maximum_distance_m": "Inclusive search distance for surface support.",
        "distance_tolerance_m": "Maximum Euclidean deviation in distance mode.",
        "negative_tolerance_m": "Allowed magnitude opposite the local normal.",
        "positive_tolerance_m": "Allowed magnitude along the local normal.",
        "minimum_support_fraction": "Minimum scan fraction matched away from open mesh boundaries.",
        "minimum_in_tolerance_fraction": "Minimum in-tolerance fraction of evaluated scan rows.",
        "minimum_signed_fraction": "Minimum signed fraction of supported scan rows in normal mode.",
        "maximum_shared_normal_angle_deg": "Largest compatible normal angle at shared features.",
        "max_points": "Maximum number of measured scan points.",
        "max_triangles": "Maximum number of input reference triangles.",
        "max_work": "Maximum point-to-box and point-to-triangle tests combined.",
    }
    input_help = {
        "scan": "Measured XYZ in m or mm in the reference frame.",
        "reference": "Triangle surface in the same frame. Winding defines outward direction.",
    }
    output_help = {
        "defects": "Supported scan rows beyond the selected tolerance.",
        "boundary": "Scan rows closest to an open reference edge or boundary vertex.",
        "unsupported": "Scan rows with no usable triangle within the search distance.",
        "correspondences": "Per scan row triangle index, closest XYZ, distance, sign, and state.",
        "report": "Triangle work, support counts, tolerance results and deviation statistics.",
    }

    def process(self, inputs):
        scan, mesh, p = inputs["scan"], inputs["reference"], self.parameters
        if scan.frame != mesh.frame:
            raise ValidationError("Align the scan and mesh to the same frame", "scan", "reference")
        if scan.units not in METERS or mesh.units not in METERS:
            raise ValidationError("Scan and reference mesh must both use m or mm")
        if len(scan.points) > p["max_points"] or len(mesh.faces) > p["max_triangles"]:
            raise ValidationError("Scan or mesh exceeds the configured point or triangle budget")

        radius = float(p["maximum_distance_m"])
        xyz = scan.points.astype(np.float64) * METERS[scan.units]
        vertices = mesh.vertices.astype(np.float64) * METERS[mesh.units]
        report = {
            "frame": scan.frame,
            "comparison": p["comparison"],
            "maximum_distance_m": radius,
            "scan_points": len(xyz),
            "reference_triangles": len(mesh.faces),
        }
        outputs = {port.name: NO_VALUE for port in self.outputs}
        outputs["report"] = report

        def failed(code, message):
            return Result(
                outputs, passed=False, message=message, fail_reasons=[_failure(code, message)]
            )

        if self._cancelled():
            return failed("scan_mesh_cancelled", "Scan-to-mesh comparison cancelled")
        surface = TriangleSurface(vertices, mesh.faces, float(p["maximum_shared_normal_angle_deg"]))
        report["usable_triangles"] = len(surface.face_ids)
        report["degenerate_triangles"] = len(mesh.faces) - len(surface.face_ids)
        matches, work = surface.query(xyz, radius, int(p["max_work"]), self._cancelled)
        report.update(work)
        if work["budget_exceeded"]:
            return failed("scan_mesh_work_limit", "Scan-to-mesh comparison exceeds the work limit")
        if matches is None or self._cancelled():
            return failed("scan_mesh_cancelled", "Scan-to-mesh comparison cancelled")

        states, supported, tested = _classify(matches, p)
        support_count = int(np.count_nonzero(supported))
        tested_count = int(np.count_nonzero(tested))
        signed_count = int(np.count_nonzero(np.isfinite(matches.signed)))
        good_count = int(np.count_nonzero(states == _IN_TOLERANCE))
        coverage = support_count / len(xyz) if len(xyz) else 0.0
        signed_fraction = signed_count / support_count if support_count else 0.0
        tolerance = good_count / tested_count if tested_count else None
        report.update(
            supported_points=support_count,
            support_fraction=coverage,
            boundary_points=int(np.count_nonzero(states == _BOUNDARY)),
            unsupported_points=int(np.count_nonzero(states == _UNSUPPORTED)),
            ambiguous_points=int(np.count_nonzero(matches.ambiguous)),
            signed_points=signed_count,
            signed_fraction=signed_fraction,
            evaluated_points=tested_count,
            in_tolerance_points=good_count,
            in_tolerance_fraction=tolerance,
            euclidean_m=deviation_statistics(matches.distance[supported]),
            signed_m=deviation_statistics(matches.signed[np.isfinite(matches.signed)]),
        )
        reasons = []
        if not len(xyz):
            reasons.append(_failure("scan_mesh_empty_scan", "The measured scan is empty"))
        if not len(surface.face_ids):
            reasons.append(
                _failure("scan_mesh_degenerate_reference", "Reference has no usable triangles")
            )
        if not support_count or coverage < float(p["minimum_support_fraction"]):
            reasons.append(
                _failure("scan_mesh_low_support", f"Usable scan support is {coverage:.2%}")
            )
        if (
            support_count
            and p["comparison"] == "normal"
            and (not signed_count or signed_fraction < float(p["minimum_signed_fraction"]))
        ):
            reasons.append(
                _failure(
                    "scan_mesh_low_signed_coverage", f"Signed coverage is {signed_fraction:.2%}"
                )
            )
        if tolerance is not None and tolerance < float(p["minimum_in_tolerance_fraction"]):
            reasons.append(
                _failure("scan_mesh_tolerance", f"In-tolerance fraction is {tolerance:.2%}")
            )

        for port, state in (
            ("defects", _DEFECT),
            ("boundary", _BOUNDARY),
            ("unsupported", _UNSUPPORTED),
        ):
            outputs[port] = scan.subset(states == state)
        if "correspondences" in self.exposed_outputs:
            outputs["correspondences"] = {
                "triangle_index": matches.face.tolist(),
                "closest_xyz_m": [
                    point if index >= 0 else None
                    for index, point in zip(
                        matches.face.tolist(), matches.closest.tolist(), strict=True
                    )
                ],
                "distance_m": _values(matches.distance),
                "signed_distance_m": _values(matches.signed),
                "state": states.tolist(),
            }

        graphics = {}
        if self.capture_graphics:
            signed = matches.signed if p["comparison"] == "normal" else None
            graphics["Scan-to-mesh deviations"] = _review(scan, states, signed)
        scores = [coverage, tolerance or 0.0]
        if p["comparison"] == "normal":
            scores.append(signed_fraction)
        return Result(
            outputs,
            graphics=graphics,
            passed=not reasons,
            score=min(scores),
            fail_reasons=reasons,
            message=f"Evaluated {tested_count} of {len(xyz)} scan points",
        )
