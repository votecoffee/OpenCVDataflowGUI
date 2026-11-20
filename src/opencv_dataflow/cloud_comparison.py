"""Bounded, bidirectional dimensional inspection of aligned point clouds."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .pointclouds import PointCloud, render_cloud

_UNMATCHED, _IN_TOLERANCE, _OUT_OF_TOLERANCE, _SIGN_UNAVAILABLE = range(4)
_QUERY_BATCH = 8192


@dataclass
class NearestRows:
    index: np.ndarray
    distance: np.ndarray
    ambiguous: np.ndarray


def _nearest_rows(source, target, radius, tie_tolerance, cancelled):
    """Return exact nearest distances and unique target row IDs within radius."""
    index = np.full(len(source), -1, np.int32)
    distance = np.full(len(source), np.nan)
    ambiguous = np.zeros(len(source), bool)
    result = NearestRows(index, distance, ambiguous)
    if not len(source) or not len(target):
        return result
    # Coincident rows remain distinct observations. Indexing each unique XYZ
    # avoids building a large KD-tree leaf made entirely of duplicate points.
    xyz, original, counts = np.unique(target, axis=0, return_index=True, return_counts=True)
    if cancelled():
        return None
    tree = cKDTree(xyz)
    upper = np.nextafter(radius + tie_tolerance, np.inf)
    for start in range(0, len(source), _QUERY_BATCH):
        if cancelled():
            return None
        band = slice(start, start + _QUERY_BATCH)
        distances, indices = tree.query(source[band], k=[1, 2], distance_upper_bound=upper)
        rows = np.flatnonzero(distances[:, 0] <= radius)
        peer = indices[rows, 0]
        tied = (counts[peer] > 1) | (distances[rows, 1] <= distances[rows, 0] + tie_tolerance)
        distance[start + rows] = distances[rows, 0]
        ambiguous[start + rows] = tied
        index[start + rows[~tied]] = original[peer[~tied]]
    return result


def deviation_statistics(values):
    if not len(values):
        return None
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p95_absolute": float(np.percentile(np.abs(values), 95)),
    }


@dataclass
class CloudDirection:
    nearest: NearestRows
    signed: np.ndarray
    state: np.ndarray
    report: dict

    def records(self):
        def nullable(values):
            return [v if math.isfinite(v) else None for v in values.tolist()]

        return {
            "nearest_index": self.nearest.index.tolist(),
            "distance_m": nullable(self.nearest.distance),
            "signed_distance_m": nullable(self.signed),
            "state": self.state.tolist(),
        }


def _compare_direction(
    xyz, reference_xyz, measured_normals, reference_normals, nearest, reverse, p
):
    length = len(nearest.index)
    matched = np.isfinite(nearest.distance)
    signed = np.full(length, np.nan)
    normal_rejected = 0
    if reference_normals is not None:
        rows = np.flatnonzero(nearest.index >= 0)
        peers = nearest.index[rows]
        measured_rows, reference_rows = (peers, rows) if reverse else (rows, peers)
        normals = reference_normals[reference_rows].astype(np.float64)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        if measured_normals is not None:
            observed = measured_normals[measured_rows].astype(np.float64)
            observed /= np.linalg.norm(observed, axis=1)[:, None]
            cosine = np.cos(np.deg2rad(float(p["maximum_normal_angle_deg"])))
            agrees = np.einsum("ij,ij->i", normals, observed) >= cosine - 1e-12
            normal_rejected = len(rows) - int(np.count_nonzero(agrees))
            rows, measured_rows, reference_rows, normals = (
                rows[agrees],
                measured_rows[agrees],
                reference_rows[agrees],
                normals[agrees],
            )
        delta = xyz[measured_rows] - reference_xyz[reference_rows]
        signed[rows] = np.einsum("ij,ij->i", delta, normals)
    usable_sign = np.isfinite(signed)
    if p["comparison"] == "normal":
        tested = usable_sign
        good = (signed >= -float(p["negative_tolerance_m"])) & (
            signed <= float(p["positive_tolerance_m"])
        )
    else:
        tested = matched
        good = nearest.distance <= float(p["distance_tolerance_m"])
    state = np.full(length, _UNMATCHED, np.uint8)
    state[matched & ~tested] = _SIGN_UNAVAILABLE
    state[tested] = np.where(good[tested], _IN_TOLERANCE, _OUT_OF_TOLERANCE)
    matched_count = int(np.count_nonzero(matched))
    signed_count = int(np.count_nonzero(usable_sign))
    tested_count = int(np.count_nonzero(tested))
    good_count = int(np.count_nonzero(state == _IN_TOLERANCE))
    return CloudDirection(
        nearest,
        signed,
        state,
        {
            "total_points": length,
            "matched_points": matched_count,
            "overlap_fraction": matched_count / length if length else 0.0,
            "ambiguous_matches": int(np.count_nonzero(nearest.ambiguous)),
            "normal_agreement_failures": normal_rejected,
            "signed_points": signed_count,
            "signed_fraction_of_matches": signed_count / matched_count if matched_count else 0.0,
            "evaluated_points": tested_count,
            "in_tolerance_points": good_count,
            "in_tolerance_fraction": good_count / tested_count if tested_count else None,
            "euclidean_m": deviation_statistics(nearest.distance[matched]),
            "signed_m": deviation_statistics(signed[usable_sign]),
        },
    )


def _failure(code, message):
    return {"code": code, "message": message}


def _judgment(stats, label, p):
    reasons = []
    name = label.capitalize()
    overlap, minimum = stats["overlap_fraction"], float(p[f"minimum_{label}_overlap"])
    if overlap < minimum:
        reasons.append(
            _failure(
                f"{label}_overlap_below_minimum",
                f"{name} overlap {overlap:.2%} is below {minimum:.2%}",
            )
        )
    signed, minimum_sign = stats["signed_fraction_of_matches"], float(p["minimum_signed_fraction"])
    if stats["matched_points"] and p["comparison"] == "normal":
        if not stats["signed_points"]:
            reasons.append(
                _failure(
                    f"{label}_signed_coverage_below_minimum", f"{name} matches lack reliable signs"
                )
            )
        elif signed < minimum_sign:
            reasons.append(
                _failure(
                    f"{label}_signed_coverage_below_minimum",
                    f"{name} signed match coverage {signed:.2%} is below {minimum_sign:.2%}",
                )
            )
    tolerance = stats["in_tolerance_fraction"]
    required = float(p["minimum_in_tolerance_fraction"])
    if tolerance is not None and tolerance < required:
        reasons.append(
            _failure(
                f"{label}_tolerance_below_minimum",
                f"{name} in-tolerance fraction {tolerance:.2%} is below {required:.2%}",
            )
        )
    return reasons


def _color_review(cloud, direction, signed_mode):
    # RGB review colors leave the original cloud's scientific attributes intact.
    palette = np.array([[238, 173, 61], [86, 220, 145], [238, 76, 73], [155, 155, 155]], np.uint8)
    rgb = palette[direction.state]
    if signed_mode:
        inward = (direction.state == _OUT_OF_TOLERANCE) & (direction.signed < 0)
        rgb[inward] = [81, 143, 247]
    view = PointCloud(cloud.points, cloud.units, cloud.frame, colors=rgb)
    return render_cloud(view)


class CompareCloudDimensions(Tool3D):
    type_id = "3d.cloud_compare"
    title, subtype = "Compare point-cloud dimensions", "Point clouds"
    description = "Measure bidirectional overlap and deviations between aligned scans."
    payload_reference = "measured"
    parallel_safe = preview_safe = process_inputs_readonly = True
    inputs = (Port("measured", "cloud"), Port("reference", "cloud"))
    outputs = (
        Port("measured_defects", "cloud"),
        Port("unmatched_measured", "cloud"),
        Port("missing_reference", "cloud"),
        Port("measured_pairs", "data", required=False),
        Port("reference_pairs", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "comparison": "distance",
        "correspondence_radius_m": 0.01,
        "distance_tolerance_m": 0.002,
        "negative_tolerance_m": 0.002,
        "positive_tolerance_m": 0.002,
        "minimum_measured_overlap": 0.9,
        "minimum_reference_overlap": 0.9,
        "minimum_in_tolerance_fraction": 1.0,
        "minimum_signed_fraction": 0.95,
        "maximum_normal_angle_deg": 60.0,
        "tie_tolerance_m": 1e-9,
        "max_points_per_cloud": 250_000,
    }
    parameter_choices = {"comparison": ("distance", "normal")}
    parameter_ranges = {
        "correspondence_radius_m": NumericRange(1e-9, 1e12),
        "distance_tolerance_m": NumericRange(0, 1e12),
        "negative_tolerance_m": NumericRange(0, 1e12),
        "positive_tolerance_m": NumericRange(0, 1e12),
        "minimum_measured_overlap": NumericRange(0, 1),
        "minimum_reference_overlap": NumericRange(0, 1),
        "minimum_in_tolerance_fraction": NumericRange(0, 1),
        "minimum_signed_fraction": NumericRange(0, 1),
        "maximum_normal_angle_deg": NumericRange(0, 180),
        "tie_tolerance_m": NumericRange(0, 1e12),
        "max_points_per_cloud": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "comparison": "Judge Euclidean distance or signed projection onto reference normals.",
        "correspondence_radius_m": "Inclusive maximum Euclidean distance for overlap.",
        "distance_tolerance_m": "Maximum Euclidean deviation in distance mode.",
        "negative_tolerance_m": "Allowed magnitude opposite the reference normal in normal mode.",
        "positive_tolerance_m": "Allowed magnitude along the reference normal in normal mode.",
        "minimum_measured_overlap": "Minimum measured rows with a nearby reference, as a fraction.",
        "minimum_reference_overlap": "Minimum fraction of reference rows with a measured match.",
        "minimum_in_tolerance_fraction": "Minimum in-tolerance fraction in each direction.",
        "minimum_signed_fraction": "Minimum signed fraction of matches in normal mode.",
        "maximum_normal_angle_deg": "Maximum normal angle if both clouds have normals.",
        "tie_tolerance_m": "Withhold row identity and sign for near-tied distances.",
        "max_points_per_cloud": "Maximum number of points admitted per cloud.",
    }
    input_help = {
        "measured": "Measured XYZ in the same named frame as the reference, in m or mm.",
        "reference": "Nominal XYZ with optional oriented normals, in the measured frame.",
    }
    output_help = {
        "measured_defects": "Measured points whose evaluated deviation exceeded tolerance.",
        "unmatched_measured": "Measured points with no reference neighbor within the radius.",
        "missing_reference": "Reference points with no measured neighbor within the radius.",
        "measured_pairs": "One row per measured point with reference index and deviation.",
        "reference_pairs": "One row per reference point with measured index and deviation.",
        "report": "Bidirectional overlap, signed coverage and deviation statistics.",
    }

    def process(self, inputs):
        measured, reference, p = inputs["measured"], inputs["reference"], self.parameters
        if measured.frame != reference.frame:
            raise ValidationError(
                "Align both clouds to the same coordinate frame", "measured", "reference"
            )
        if measured.units not in METERS or reference.units not in METERS:
            raise ValidationError("Dimensional comparison requires both clouds in m or mm")
        if max(len(measured.points), len(reference.points)) > p["max_points_per_cloud"]:
            raise ValidationError(
                "Cloud exceeds the configured point budget", "max_points_per_cloud"
            )

        radius, tie = float(p["correspondence_radius_m"]), float(p["tie_tolerance_m"])
        report = {
            "frame": measured.frame,
            "comparison": p["comparison"],
            "correspondence_radius_m": radius,
            "reference_has_normals": reference.normals is not None,
            "measured_has_normals": measured.normals is not None,
        }
        outputs = {port.name: NO_VALUE for port in self.outputs}
        outputs["report"] = report

        def cancelled_result():
            return Result(
                outputs,
                passed=False,
                fail_reasons=[
                    _failure("cloud_comparison_cancelled", "Point-cloud comparison cancelled")
                ],
                message="Point-cloud comparison cancelled",
            )

        if self._cancelled():
            return cancelled_result()
        xyz = measured.points.astype(np.float64) * METERS[measured.units]
        nominal = reference.points.astype(np.float64) * METERS[reference.units]
        forward = _nearest_rows(xyz, nominal, radius, tie, self._cancelled)
        if forward is None or self._cancelled():
            return cancelled_result()
        reverse = _nearest_rows(nominal, xyz, radius, tie, self._cancelled)
        if reverse is None or self._cancelled():
            return cancelled_result()
        measured_result = _compare_direction(
            xyz, nominal, measured.normals, reference.normals, forward, False, p
        )
        reference_result = _compare_direction(
            xyz, nominal, measured.normals, reference.normals, reverse, True, p
        )
        if self._cancelled():
            return cancelled_result()
        a, b = measured_result.report, reference_result.report
        report.update(measured_to_reference=a, reference_to_measured=b)

        reasons = []
        if not len(xyz):
            reasons.append(_failure("empty_measured_cloud", "The measured cloud is empty"))
        if not len(nominal):
            reasons.append(_failure("empty_reference_cloud", "The reference cloud is empty"))
        if len(xyz) and len(nominal) and not (a["matched_points"] or b["matched_points"]):
            reasons.append(_failure("no_overlapping_pairs", "No points overlap within the radius"))
        reasons.extend(_judgment(a, "measured", p))
        reasons.extend(_judgment(b, "reference", p))

        outputs.update(
            measured_defects=measured.subset(measured_result.state == _OUT_OF_TOLERANCE),
            unmatched_measured=measured.subset(measured_result.state == _UNMATCHED),
            missing_reference=reference.subset(reference_result.state == _UNMATCHED),
        )
        for port, direction in (
            ("measured_pairs", measured_result),
            ("reference_pairs", reference_result),
        ):
            if port in self.exposed_outputs:
                outputs[port] = direction.records()
        graphics = {}
        if self.capture_graphics:
            graphics["Measured deviations"] = _color_review(
                measured, measured_result, p["comparison"] == "normal"
            )
        payloads = {
            "measured_defects": self.input_payloads["measured"],
            "unmatched_measured": self.input_payloads["measured"],
            "missing_reference": self.input_payloads["reference"],
        }
        scores = [
            a["overlap_fraction"],
            b["overlap_fraction"],
            a["in_tolerance_fraction"] or 0,
            b["in_tolerance_fraction"] or 0,
        ]
        if p["comparison"] == "normal":
            scores.extend((a["signed_fraction_of_matches"], b["signed_fraction_of_matches"]))
        return Result(
            outputs,
            graphics=graphics,
            payloads=payloads,
            passed=not reasons,
            score=min(scores),
            fail_reasons=reasons,
            message=f"Matched {a['matched_points']}/{len(xyz)} measured and "
            f"{b['matched_points']}/{len(nominal)} reference points",
        )
