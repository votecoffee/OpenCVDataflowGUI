"""Cached poses to native RGB evidence: cup, tool sweep, TCP/approach and witnesses."""

from dataclasses import dataclass

import numpy as np

COLORS = {"eligible": [50, 230, 115], "rejected": [245, 75, 75],
          "unknown": [255, 175, 45], "selected": [255, 230, 50], "scene": [115, 125, 140],
          "bin": [140, 125, 230], "sweep": [70, 180, 245], "collision": [255, 45, 210]}
LAYERS = ("scene", "bin", "cups", "sweep", "poses", "witnesses")


@dataclass(frozen=True)
class CloudGraphics:
    points: np.ndarray
    colors: np.ndarray
    units: str
    frame: str
    metadata: dict


def result_cloud(result, *, selected=None, layers=LAYERS):
    if any(name not in LAYERS for name in layers):
        raise ValueError("Choose known bin-picking graphic layers")
    if selected is not None and selected not in {c["id"] for c in result.candidates}:
        raise ValueError("Choose an existing candidate")
    recipe, scene = result.scene.recipe, result.scene
    xyz, rgb = [], []

    def add(data, color):
        data = np.asarray(data).reshape(-1, 3)
        xyz.append(data)
        rgb.append(np.tile(color, (len(data), 1)))

    def line(a, b, color, count=24):
        add(np.linspace(a, b, count), color)

    if "scene" in layers and len(scene.points):
        rows = np.linspace(0, len(scene.points) - 1, min(20000, len(scene.points)), dtype=int)
        add(scene.points[rows], COLORS["scene"])
    if "bin" in layers:
        corners = np.array([[x, y, z] for x in (recipe.minimum[0], recipe.maximum[0])
                            for y in (recipe.minimum[1], recipe.maximum[1])
                            for z in (recipe.minimum[2], recipe.maximum[2])])
        for i in range(8):
            for bit in (1, 2, 4):
                if i < i ^ bit:
                    line(corners[i], corners[i ^ bit], COLORS["bin"])
    if selected is None:
        eligible = [c for c in result.candidates if c["status"] == "eligible"][:20]
        other = [c for c in result.candidates if c["status"] != "eligible" and
                 c["reasons"] not in (["overlap_with_higher_ranked"], ["pick_count_limit"]) and
                 c["contact_pose"] is not None]
        if len(other) > 12:
            other = [other[i] for i in np.linspace(0, len(other) - 1, 12, dtype=int)]
        candidates = eligible + other
    else:
        candidates = [c for c in result.candidates if c["id"] == selected]
    primary = candidates[0]["id"] if candidates else None
    # All alternatives remain in the report/table. Draw one swept envelope in overview.
    for item in candidates:
        if item["contact_pose"] is None:
            continue
        transform = np.asarray(item["contact_pose"])
        center, rotation = transform[:3, 3], transform[:3, :3]
        color = COLORS["selected"] if item["id"] == selected else COLORS[item["status"]]
        theta = np.linspace(0, 2 * np.pi, 80)
        circle = np.c_[np.cos(theta), np.sin(theta), np.zeros(len(theta))]
        s = recipe.settings
        if "cups" in layers:
            add(circle * s.cup_radius @ rotation.T + center, color)
        if "sweep" in layers and (selected is not None or item["id"] == primary):
            start, end = s.surface_tolerance, s.body_length + s.pregrasp_distance
            radius = s.body_radius + s.clearance
            for t in (start, end):
                add(circle * radius @ rotation.T + center + t * rotation[:, 2], COLORS["sweep"])
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                offset = radius * (rotation[:, 0] * np.cos(angle) + rotation[:, 1] * np.sin(angle))
                line(center + start * rotation[:, 2] + offset,
                     center + end * rotation[:, 2] + offset, COLORS["sweep"])
        if "poses" in layers and (selected is not None or item["id"] == primary):
            tcp = np.asarray(item["tcp_pose"])[:3, 3]
            pregrasp = np.asarray(item["pregrasp_pose"])[:3, 3]
            line(tcp, pregrasp, color, 50)
            for axis in range(3):
                axis_color = np.zeros(3)
                axis_color[axis] = 255
                line(tcp, tcp + s.cup_radius * 1.5 * rotation[:, axis], axis_color)
        if "witnesses" in layers:
            add(scene.points[item["collision_rows"]], COLORS["collision"])
            add(item["unknown_samples"], COLORS["unknown"])
    points = np.ascontiguousarray(np.concatenate(xyz) if xyz else np.empty((0, 3)), np.float32)
    colors = np.ascontiguousarray(np.concatenate(rgb) if rgb else np.empty((0, 3)), np.uint8)
    points.flags.writeable = colors.flags.writeable = False
    return CloudGraphics(points, colors, recipe.units, recipe.frame,
        {"tool": "SimpleBinPicking", "legend": COLORS, "layers": list(layers),
         "selected_candidate": selected, "eligible_picks": len(result.picks),
         "displayed_candidate_ids": [c["id"] for c in candidates],
         "overview_sweep_candidate": primary,
         "pose_convention": "+Z retracts from surface; approach motion is -Z",
         "note": "Cups, cylinders, axes and arrows are derived configured geometry; "
                 "unknown samples are not free-space observations."})
