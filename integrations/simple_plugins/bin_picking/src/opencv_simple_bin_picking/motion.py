"""Numeric result graphics for planned and measured-joint TCP paths."""

import numpy as np

STAGES = {
    "observation": [210, 210, 210],
    "pregrasp": [70, 150, 255],
    "approach": [255, 170, 30],
    "lift": [30, 230, 240],
    "transfer": [185, 100, 255],
    "place": [255, 230, 50],
    "retreat": [40, 220, 110],
    "park": [180, 180, 180],
}
FEEDBACK = [255, 70, 170]


def _points(value):
    points = np.asarray(value, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) > 20000
        or not np.isfinite(points).all()
    ):
        raise ValueError("Path review needs bounded finite XYZ points")
    return points


def _line(points, color, output, colors):
    if not len(points):
        return
    segments = [points[:1]]
    for a, b in zip(points, points[1:]):
        count = min(200, max(2, int(np.ceil(np.linalg.norm(b - a) / 0.003)) + 1))
        segments.append(np.linspace(a, b, count)[1:])
    sampled = np.vstack(segments)
    output.append(sampled)
    colors.append(np.tile(color, (len(sampled), 1)))


def path_graphics(report, *, planned=True, feedback=True):
    points, colors, rows = [], [], []
    model_hash = (
        report.get("kinematic_model", {}).get("sha256") if report.get("kinematic_model") else None
    )
    for receipt in report.get("receipts", []):
        result = receipt.get("result", {})
        stage = receipt.get("motion_stage", "transfer")
        if "tcp_path" in result and planned:
            path = result["tcp_path"]
            if (
                path.get("frame") != "world"
                or path.get("units") != "m"
                or (model_hash and path.get("kinematic_sha256") != model_hash)
            ):
                raise ValueError("Planned path frame/units/robot model differ")
            xyz = _points(path["positions_m"])
            _line(xyz, STAGES.get(stage, STAGES["transfer"]), points, colors)
            rows.append(
                dict(
                    stage=stage,
                    kind="planned",
                    samples=len(xyz),
                    duration_s=result.get("duration_s"),
                    fk_validation=path.get("native_validation"),
                )
            )
        if "observed_tcp_path" in result and feedback:
            path = result["observed_tcp_path"]
            if (
                path.get("frame") != "world"
                or path.get("units") != "m"
                or (model_hash and path.get("kinematic_sha256") != model_hash)
            ):
                raise ValueError("Feedback path frame/units/robot model differ")
            samples = path.get("samples", [])
            if samples:
                xyz = _points([s["position_m"] for s in samples])
                # Discrete samples preserve what was observed; no filled feedback gaps.
                points.append(xyz)
                colors.append(np.tile(FEEDBACK, (len(xyz), 1)))
            rows.append(
                dict(
                    stage=stage,
                    kind="measured_joint_feedback_fk",
                    samples=len(samples),
                    status=path.get("status"),
                    discarded=path.get("discarded_samples", 0),
                )
            )
    if sum(len(p) for p in points) > 300000:
        raise ValueError("Motion graphic exceeds 300000 vertices")
    return (
        (np.vstack(points) if points else np.empty((0, 3))),
        (np.vstack(colors).astype(np.uint8) if colors else np.empty((0, 3), np.uint8)),
        dict(
            source="planned joint-waypoint FK lines; measured joint-feedback FK dots",
            independent_pose_measurement=False,
            collision_certificate=False,
            stage_colors_rgb=STAGES,
            feedback_color_rgb=FEEDBACK,
            paths=rows,
        ),
    )
