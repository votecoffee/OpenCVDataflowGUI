"""Measure observed point-cloud profiles in a metric virtual slice."""

import numpy as np

from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, rigid
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .projection import profile_plot

_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
_MAX_POINTS = 2_000_000
_COORDINATE = NumericRange(-1e12, 1e12)


def _basis(pose, cloud):
    if pose is None:
        return cloud.frame, np.eye(3), np.zeros(3)
    if (
        not isinstance(pose, dict)
        or pose.get("target_frame") != cloud.frame
        or not isinstance(pose.get("source_frame"), str)
        or not pose["source_frame"].strip()
        or pose.get("units") not in ("m", "mm")
    ):
        raise ValidationError("Pose must map a named metric frame into the cloud frame", "pose")
    try:
        transform = rigid(pose)
    except ValueError as exc:
        raise ValidationError(str(exc), "pose") from exc
    # The pose contract admits small rotational drift. Use unit measurement axes.
    left, _, right = np.linalg.svd(transform[:3, :3])
    return pose["source_frame"], left @ right, transform[:3, 3] * METERS[pose["units"]]


def _edges(p):
    edges = np.linspace(float(p["u_min_m"]), float(p["u_max_m"]), int(p["bins"]) + 1)
    if np.any(np.diff(edges) <= 0):
        raise ValidationError(
            "U limits must define distinct profile bin edges", "u_min_m", "u_max_m", "bins"
        )
    return edges


def _extreme_rows(bins, v, rows, extrema):
    # Keep the earliest input row when several points have the same extreme value.
    chosen = np.full(len(extrema), np.iinfo(np.intp).max, dtype=np.intp)
    matches = v == extrema[bins]
    np.minimum.at(chosen, bins[matches], rows[matches])
    return chosen


def _row_reference(cloud, row):
    return {
        "input_row": int(row),
        **{
            name: int(cloud.attributes[name][row])
            for name in ("source_index", "source_scan")
            if name in cloud.attributes
        },
    }


def _profile(cloud, rows, uv, edges, min_count):
    total = len(edges) - 1
    indices = np.minimum(np.searchsorted(edges, uv[:, 0], side="right") - 1, total - 1)
    v = uv[:, 1]
    counts = np.bincount(indices, minlength=total)
    supported = counts >= min_count
    lows = np.full(total, np.inf)
    highs = np.full(total, -np.inf)
    np.minimum.at(lows, indices, v)
    np.maximum.at(highs, indices, v)
    means = np.divide(
        np.bincount(indices, weights=v, minlength=total),
        counts,
        out=np.zeros(total),
        where=counts > 0,
    )
    min_rows = _extreme_rows(indices, v, rows, lows)
    max_rows = _extreme_rows(indices, v, rows, highs)
    centers = edges[:-1] + np.diff(edges) / 2

    bins = []
    for i, count in enumerate(counts):
        record = {
            "u_start_m": float(edges[i]),
            "u_end_m": float(edges[i + 1]),
            "u_center_m": float(centers[i]),
            "point_count": int(count),
            "supported": bool(supported[i]),
            "v_min_m": None,
            "v_mean_m": None,
            "v_max_m": None,
            "width_m": None,
            "minimum_point": None,
            "maximum_point": None,
        }
        if supported[i]:
            record.update(
                v_min_m=float(lows[i]),
                v_mean_m=float(means[i]),
                v_max_m=float(highs[i]),
                width_m=float(highs[i] - lows[i]),
                minimum_point=_row_reference(cloud, min_rows[i]),
                maximum_point=_row_reference(cloud, max_rows[i]),
            )
        bins.append(record)
    return bins, supported, np.column_stack((lows, means, highs)), centers


class MeasureCloudSection(Tool3D):
    type_id = "3d.cloud_section"
    title, subtype = "Measure point-cloud section", "Point clouds"
    description = "Select a metric slab and measure observed widths on a fixed section grid."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("cloud", "cloud"),
        Port("pose", "data", visual_type="pose", required=False, required_when_connected=True),
    )
    outputs = (Port("section", "cloud"), Port("profile", "data"), Port("report", "data"))
    defaults = {
        "plane": "xy",
        "offset_m": 0.0,
        "thickness_m": 0.004,
        "u_min_m": -0.05,
        "u_max_m": 0.05,
        "restrict_v": False,
        "v_min_m": -0.05,
        "v_max_m": 0.05,
        "bins": 50,
        "minimum_points_per_bin": 2,
        "minimum_coverage": 0.0,
        "max_input_points": 250_000,
    }
    parameter_choices = {"plane": tuple(_AXES)}
    parameter_ranges = {
        "offset_m": _COORDINATE,
        "thickness_m": NumericRange(1e-9, 1e12),
        "u_min_m": _COORDINATE,
        "u_max_m": _COORDINATE,
        "v_min_m": _COORDINATE,
        "v_max_m": _COORDINATE,
        "bins": NumericRange(1, 2048, whole=True),
        "minimum_points_per_bin": NumericRange(1, _MAX_POINTS, whole=True),
        "minimum_coverage": NumericRange(0, 1),
        "max_input_points": NumericRange(1, _MAX_POINTS, whole=True),
    }
    parameter_help = {
        "plane": "First axis is U, second is V, and the unused positive axis is the normal.",
        "offset_m": "Move the section origin along the positive normal, in meters.",
        "thickness_m": "Full symmetric thickness about the section plane, in meters.",
        "u_min_m": "First limit of the sampled U interval, in meters.",
        "u_max_m": "Last limit of the sampled U interval, in meters.",
        "restrict_v": "Also limit the selection across the section width.",
        "v_min_m": "Inclusive lower V limit in meters when restriction is enabled.",
        "v_max_m": "Inclusive upper V limit in meters when restriction is enabled.",
        "bins": "Number of evenly spaced profile bins across the U interval.",
        "minimum_points_per_bin": "A bin needs this many observed points to report dimensions.",
        "minimum_coverage": "Required fraction of the configured U interval with supported bins.",
        "max_input_points": "Maximum number of cloud rows accepted for measurement.",
    }
    input_help = {
        "cloud": "Metric point cloud in meters or millimeters.",
        "pose": "Optional rigid transform from the section reference frame into the cloud.",
    }
    output_help = {
        "section": "Original selected points in their input order, frame and units.",
        "profile": "Metric U bins with observed V envelope, counts and extrema source rows.",
        "report": "Plane provenance, signed distance statistics, extents and support coverage.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _edges(p)
        if type(p["restrict_v"]) is not bool:
            raise ValidationError("Restrict V must be a checkbox value", "restrict_v")
        if p["restrict_v"] and p["v_min_m"] >= p["v_max_m"]:
            raise ValidationError("Maximum V must exceed minimum V", "v_min_m", "v_max_m")

    def process(self, inputs):
        cloud, p = inputs["cloud"], self.parameters
        if cloud.units not in METERS:
            raise ValidationError("Section measurement requires a cloud in m or mm", "cloud")
        if len(cloud.points) > p["max_input_points"]:
            raise ValidationError("Cloud exceeds the section point budget", "max_input_points")
        reference, basis, anchor = _basis(inputs.get("pose"), cloud)
        u, v, n = _AXES[p["plane"]]
        offset, thickness = float(p["offset_m"]), float(p["thickness_m"])
        edges = _edges(p)
        origin = anchor + offset * basis[:, n]
        measure = {
            "model": "cloud_section_v1",
            "reference_frame": reference,
            "cloud_frame": cloud.frame,
            "cloud_units": cloud.units,
            "plane": p["plane"],
            "origin_cloud_m": origin.tolist(),
            "u_axis_cloud": basis[:, u].tolist(),
            "v_axis_cloud": basis[:, v].tolist(),
            "normal_cloud": basis[:, n].tolist(),
            "signed_offset_m": offset,
            "full_thickness_m": thickness,
            "u_interval_m": [float(edges[0]), float(edges[-1])],
            "v_interval_m": [float(p["v_min_m"]), float(p["v_max_m"])] if p["restrict_v"] else None,
        }
        report = {**measure, "input_points": len(cloud.points)}

        def cancelled():
            report["status"] = "cancelled"
            return Result(
                {"section": NO_VALUE, "profile": NO_VALUE, "report": report},
                passed=False,
                fail_reasons=["cloud_section_cancelled"],
                message="Point-cloud section cancelled",
            )

        if self._cancelled():
            return cancelled()
        with np.errstate(over="ignore", invalid="ignore"):
            local = (cloud.points.astype(np.float64) * METERS[cloud.units] - anchor) @ basis
        if not np.isfinite(local).all():
            raise ValidationError("Section pose produces unrepresentable local coordinates", "pose")
        d = local[:, n] - offset
        keep = (np.abs(d) <= thickness / 2) & (local[:, u] >= edges[0]) & (local[:, u] <= edges[-1])
        if p["restrict_v"]:
            keep &= (p["v_min_m"] <= local[:, v]) & (local[:, v] <= p["v_max_m"])
        rows = np.flatnonzero(keep)
        uv = local[np.ix_(rows, (u, v))]
        bins, supported, values, centers = _profile(
            cloud,
            rows,
            uv,
            edges,
            int(p["minimum_points_per_bin"]),
        )
        if self._cancelled():
            return cancelled()

        extent = float(edges[-1] - edges[0])
        valid_bins = int(supported.sum())
        fraction = valid_bins / len(supported)
        section = cloud.subset(rows)
        if "source_index" not in section.attributes:
            section.attributes["source_index"] = rows.astype(np.int32)
        section.metadata["cloud_section"] = measure
        report.update(
            section_points=len(rows),
            supported_bins=valid_bins,
            total_bins=len(supported),
            coverage=fraction,
            covered_length_m=fraction * extent,
            sample_pitch_m=extent / len(supported),
            minimum_coverage=float(p["minimum_coverage"]),
            observed_u_extent_m=float(np.ptp(uv[:, 0])) if len(rows) else None,
            observed_v_width_m=float(np.ptp(uv[:, 1])) if len(rows) else None,
            signed_distance_range_m=[float(d[rows].min()), float(d[rows].max())]
            if len(rows)
            else None,
            rms_distance_m=float(np.sqrt(np.mean(d[rows] ** 2))) if len(rows) else None,
        )
        if not len(rows):
            status = "no_section_points"
        elif not valid_bins:
            status = "no_supported_bins"
        elif fraction < p["minimum_coverage"]:
            status = "insufficient_coverage"
        else:
            status = "ok"
        report["status"] = status
        profile = {
            "model": "cloud_section_profile_v1",
            "section": measure,
            "bins": bins,
            "coverage": fraction,
            "sample_pitch_m": report["sample_pitch_m"],
        }
        outputs = {"section": section, "profile": profile, "report": report}
        payload = {**self.input_payloads["cloud"], "cloud_section": measure}
        graphics = {}
        if self.capture_graphics and valid_bins:
            graphics["Section profile"] = profile_plot(
                values,
                supported,
                ["V min", "V mean", "V max"],
                "U",
                "section envelope (m)",
                distances=centers,
                distance_units="m",
            )
        return Result(
            outputs,
            payloads={name: payload for name in outputs},
            passed=status == "ok",
            score=fraction,
            fail_reasons=[] if status == "ok" else [f"cloud_section_{status}"],
            graphics=graphics,
            message=f"{len(rows):,} points · {valid_bins}/{len(supported)} bins · {status}",
        )
