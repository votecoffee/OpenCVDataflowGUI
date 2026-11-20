"""Merge registered point clouds with metric overlap suppression and row provenance."""

import copy
from dataclasses import dataclass

import numpy as np

from .cloud_comparison import nearest_cloud_rows
from .cloud_projection import METERS
from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, rigid
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup
from .point_attributes import merge_point_attributes
from .pointclouds import PointCloud
from .tickets import ticket_key

_SLOTS = tuple((f"cloud_{i}", f"pose_{i}") for i in range(1, 5))
_ATTRIBUTES = ("colors", "normals")


@dataclass
class Scan:
    name: str
    cloud: PointCloud
    context: dict
    pose: dict | None = None

    def in_frame(self, target):
        cloud = self.cloud
        if cloud.units not in METERS:
            raise ValidationError(f"{self.name} requires metric cloud units")
        if self.pose is None:
            if cloud.frame != target:
                raise ValidationError(f"{self.name} needs a pose from {cloud.frame} to {target}")
            return np.eye(4)
        pose = self.pose
        if (
            not isinstance(pose, dict)
            or pose.get("source_frame") != cloud.frame
            or pose.get("target_frame") != target
            or pose.get("units") not in METERS
        ):
            raise ValidationError(f"{self.name} pose must map {cloud.frame} to {target} in m or mm")
        try:
            matrix = rigid(pose).copy()
        except ValueError as exc:
            raise ValidationError(f"{self.name} pose must be rigid") from exc
        matrix[:3, 3] *= METERS[pose["units"]]
        return matrix


def _scans(inputs, payloads):
    if "scans" in inputs:
        group = inputs["scans"]
        if len(inputs) != 1 or not isinstance(group, FlowGroup):
            raise ValidationError("Connect a flow group of registered clouds or individual scans")
        scans = []
        for label, packet in zip(group.labels, group.items, strict=True):
            if not isinstance(label, str) or not label:
                raise ValidationError("Grouped scans require nonempty text labels", "scans")
            if not isinstance(packet.value, PointCloud):
                raise ValidationError("Every grouped scan must be a point cloud", "scans")
            scans.append(Scan(label, packet.value, {**packet.value.metadata, **packet.payload}))
        return scans, payloads["scans"].get("ticket")
    scans = []
    for cloud_port, pose_port in _SLOTS:
        if cloud_port in inputs:
            scans.append(
                Scan(cloud_port, inputs[cloud_port], payloads[cloud_port], inputs.get(pose_port))
            )
        elif pose_port in inputs:
            raise ValidationError(f"{pose_port} needs {cloud_port}", pose_port)
    tickets = [scan.context["ticket"] for scan in scans if "ticket" in scan.context]
    if len({ticket_key(ticket) for ticket in tickets}) > 1:
        raise ValidationError(
            "Individual scans need a shared ticket. Batch distinct tickets first."
        )
    return scans, tickets[0] if tickets else None


class MergeCloudScans(Tool3D):
    type_id, title, subtype = "3d.cloud_merge", "Merge registered point-cloud scans", "Point clouds"
    description = "Combine registered scans in one metric frame with optional overlap suppression."
    payload_reference = ""
    process_inputs_readonly = parallel_safe = preview_safe = True
    inputs = (
        *(
            port
            for cloud, pose in _SLOTS
            for port in (
                Port(cloud, "cloud", required=False, required_when_connected=True),
                Port(pose, "data", required=False, required_when_connected=True),
            )
        ),
        Port("scans", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("cloud", "cloud"),
        Port("report", "data"),
        Port("provenance", "data", required=False),
    )
    defaults = {
        "target_frame": "",
        "output_units": "m",
        "merge_mode": "append",
        "overlap_radius_m": 0.001,
        "attribute_policy": "common",
        "max_input_points": 250_000,
    }
    parameter_choices = {
        "output_units": ("m", "mm"),
        "merge_mode": ("append", "prefer_earlier"),
        "attribute_policy": ("common", "require_matching"),
    }
    parameter_ranges = {
        "overlap_radius_m": NumericRange(1e-12, 1e20),
        "max_input_points": NumericRange(1, 1_000_000, whole=True),
    }
    parameter_help = {
        "target_frame": "Named output frame. Blank uses the first scan or its pose target.",
        "output_units": "Metric units used in the emitted point cloud.",
        "merge_mode": "Append all points or drop later points near earlier retained scans.",
        "overlap_radius_m": "Inclusive Euclidean distance in meters for overlap suppression.",
        "attribute_policy": (
            "Keep common RGB and normals plus available scalar columns, or require matching inputs."
        ),
        "max_input_points": "Combined point budget before allocation or overlap suppression.",
    }
    input_help = {
        **{
            cloud: "Metric cloud to merge. Earlier ports have priority in overlap."
            for cloud, _ in _SLOTS
        },
        **{
            pose: "Rigid source-to-target pose for the associated cloud, in m or mm."
            for _, pose in _SLOTS
        },
        "scans": "Alternative input with 2–64 grouped metric clouds already in one named frame.",
    }
    output_help = {
        "cloud": "Merged metric cloud with shared RGB/normals and available inspection scalars.",
        "report": "Per-scan input, retained and suppressed counts and the applied metric poses.",
        "provenance": "Output row spans, original rows and original pixels for each input scan.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["target_frame"], str) or p["target_frame"] != p["target_frame"].strip():
            raise ValidationError("Target frame must be a name or empty", "target_frame")

    def process(self, inputs):
        p = self.parameters
        scans, ticket = _scans(inputs, self.input_payloads)
        if len(scans) < 2:
            raise ValidationError("Connect at least two scans or a group of at least two scans")
        count = sum(len(scan.cloud.points) for scan in scans)
        if count > p["max_input_points"]:
            raise ValidationError("Scans exceed the combined point budget", "max_input_points")
        first = scans[0]
        target = p["target_frame"] or (
            first.pose.get("target_frame") if isinstance(first.pose, dict) else first.cloud.frame
        )
        if not isinstance(target, str) or not target.strip():
            raise ValidationError("A named target frame is required", "target_frame")
        units = p["output_units"]
        coordinate_limit = float(np.finfo(np.float32).max) * METERS[units]
        report = {"frame": target, "units": units, "mode": p["merge_mode"], "input_points": count}
        outputs = dict(cloud=NO_VALUE, report=report, provenance=NO_VALUE)
        context = {"ticket": ticket} if ticket is not None else {}
        payloads = dict.fromkeys(outputs, context)
        transforms = [scan.in_frame(target) for scan in scans]
        active = [scan.cloud for scan in scans if len(scan.cloud.points)]
        different_availability = (
            any(
                len({getattr(cloud, name) is not None for cloud in active}) > 1
                for name in _ATTRIBUTES
            )
            or len({frozenset(cloud.attributes) for cloud in active}) > 1
        )
        if p["attribute_policy"] == "require_matching" and different_availability:
            raise ValidationError(
                "Nonempty scans must agree on RGB and normal availability and inspection fields",
                "attribute_policy",
            )

        def cancelled():
            return Result(
                outputs, payloads=payloads, passed=False, fail_reasons=["cloud_merge_cancelled"]
            )

        if self._cancelled():
            return cancelled()
        points = np.empty((count, 3), dtype=np.float64)
        stop = 0
        provenance, retained, reports = [], [], []
        for scan, transform in zip(scans, transforms, strict=True):
            cloud = scan.cloud
            rotation = transform[:3, :3]
            xyz = (cloud.points.astype(np.float64) * METERS[cloud.units]) @ rotation.T
            xyz += transform[:3, 3]
            if np.max(np.abs(xyz), initial=0) > coordinate_limit:
                raise ValidationError(
                    "Merged coordinates exceed the float32 point-cloud range", "output_units"
                )
            selected = np.arange(len(xyz))
            if p["merge_mode"] == "prefer_earlier" and stop and len(xyz):
                nearest = nearest_cloud_rows(
                    xyz,
                    points[:stop],
                    float(p["overlap_radius_m"]),
                    0,
                    self._cancelled,
                )
                if nearest is None:
                    return cancelled()
                selected = np.flatnonzero(~np.isfinite(nearest.distance))
            if self._cancelled():
                return cancelled()
            end = stop + len(selected)
            points[stop:end] = xyz[selected]
            details = {
                "scan": scan.name,
                "frame": cloud.frame,
                "units": cloud.units,
                "input_points": len(xyz),
                "retained_points": len(selected),
                "suppressed_points": len(xyz) - len(selected),
                "source_to_target_m": transform.tolist(),
            }
            if "ticket" in scan.context:
                details["ticket"] = scan.context["ticket"]
            reports.append(details)
            if len(selected):
                retained.append((cloud, rotation, selected))
            if "provenance" in self.exposed_outputs:
                origin = {
                    "scan": scan.name,
                    "source_frame": cloud.frame,
                    "output_span": [stop, end],
                    "source_rows": selected.tolist(),
                    "source_size": list(cloud.source_size) if cloud.pixels is not None else None,
                    "source_pixels": cloud.pixels[selected].tolist()
                    if cloud.pixels is not None
                    else None,
                }
                for key in ("ticket", "spatial", "pyramid"):
                    if key in scan.context:
                        origin[key] = copy.deepcopy(scan.context[key])
                provenance.append(origin)
            stop = end

        colors = normals = None
        if retained and all(cloud.colors is not None for cloud, _, _ in retained):
            colors = np.concatenate([cloud.colors[ix] for cloud, _, ix in retained])
        if retained and all(cloud.normals is not None for cloud, _, _ in retained):
            normals = np.concatenate(
                [cloud.normals[ix] @ rotation.T for cloud, rotation, ix in retained]
            )
        xyz = points[:stop] / METERS[units]
        attributes, namespaces = merge_point_attributes(
            [(cloud.attributes, ix) for cloud, _, ix in retained]
        )
        if namespaces:
            for detail, span in zip(
                (d for d in reports if d["retained_points"]), namespaces, strict=True
            ):
                detail["source_scan_range"] = span
        result = PointCloud(
            xyz, units, target, colors=colors, normals=normals, attributes=attributes
        )
        report.update(
            retained_points=stop,
            suppressed_points=count - stop,
            colors=colors is not None,
            normals=normals is not None,
            scans=reports,
        )
        if attributes:
            report["inspection_attributes"] = list(attributes)
        if self._cancelled():
            return cancelled()
        outputs["cloud"] = result
        if "provenance" in self.exposed_outputs:
            outputs["provenance"] = {"model": "cloud_merge_provenance_v1", "scans": provenance}
        return Result(
            outputs,
            payloads=payloads,
            graphics=cloud_graphics(self, result),
            passed=bool(stop),
            fail_reasons=[] if stop else ["cloud_merge_empty"],
            message=f"Kept {stop} of {count} points from {len(scans)} scans",
        )
