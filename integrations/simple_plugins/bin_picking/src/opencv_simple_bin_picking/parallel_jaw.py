"""Observed box-top proposals for a top-down parallel-jaw demonstration.

This is geometric candidate generation, not force-closure or robot feasibility.
Original first-return depth remains the evidence; connectivity dilation is used
only to propose components and never creates support samples.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import ConvexHull, QhullError

from .specs import Camera, number, rigid


@dataclass(frozen=True)
class BoxTopSettings:
    floor_z: float = 0.2
    minimum: tuple = (0.28, 0.05)
    maximum: tuple = (0.62, 0.35)
    minimum_height: float = 0.015
    part_height: float = 0.04
    minimum_side: float = 0.025
    maximum_side: float = 0.06
    grid_step: float = 0.0015
    top_tolerance: float = 0.002
    minimum_support: float = 0.6
    maximum_flatness: float = 0.002
    jaw_margin: float = 0.014
    maximum_opening: float = 0.08
    tcp_below_top: float = 0.005
    approach_height: float = 0.14
    minimum_points: int = 80
    maximum_candidates: int = 32

    def __post_init__(self):
        number(self.floor_z, "Floor", -100, 100)
        if (
            len(self.minimum) != 2
            or len(self.maximum) != 2
            or not np.isfinite([*self.minimum, *self.maximum]).all()
            or any(a >= b for a, b in zip(self.minimum, self.maximum))
        ):
            raise ValueError("Use ordered finite XY bin-interior bounds in meters")
        for name in ("minimum", "maximum"):
            object.__setattr__(
                self, name, tuple(number(v, name, -1000, 1000) for v in getattr(self, name))
            )
        for name in (
            "minimum_height",
            "part_height",
            "minimum_side",
            "maximum_side",
            "grid_step",
            "top_tolerance",
            "maximum_flatness",
            "jaw_margin",
            "maximum_opening",
            "tcp_below_top",
            "approach_height",
        ):
            number(getattr(self, name), name, 1e-6, 10)
        number(self.minimum_support, "Support", 0.1, 1)
        if (
            self.minimum_side >= self.maximum_side
            or self.tcp_below_top >= self.part_height
            or self.maximum_side + self.jaw_margin > self.maximum_opening
        ):
            raise ValueError("Box dimensions and jaw stroke must be physically compatible")
        if type(self.minimum_points) is not int or not 8 <= self.minimum_points <= 100000:
            raise ValueError("Use 8..100000 minimum observed top points")
        if type(self.maximum_candidates) is not int or not 1 <= self.maximum_candidates <= 256:
            raise ValueError("Use 1..256 candidates")
        shape = np.ceil((np.asarray(self.maximum) - self.minimum) / self.grid_step)
        if np.prod(shape) > 2_000_000:
            raise ValueError("Bin proposal grid exceeds two million cells")


def rectangle(points):
    """Minimum observed XY rectangle; square ties prefer world-axis alignment."""
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 8))
    best = None
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        axes = np.array([[c, s], [-s, c]])
        aligned = hull @ axes.T
        minimum, maximum = aligned.min(0), aligned.max(0)
        dimensions = maximum - minimum
        area = float(np.prod(dimensions))
        alignment = min(angle, math.pi / 2 - angle)
        if (
            best is None
            or area < best[0] - 1e-8
            or (abs(area - best[0]) <= 1e-8 and alignment < best[1])
        ):
            corners = (
                np.array(
                    [
                        [minimum[0], minimum[1]],
                        [maximum[0], minimum[1]],
                        [maximum[0], maximum[1]],
                        [minimum[0], maximum[1]],
                    ]
                )
                @ axes
            )
            best = (
                area,
                alignment,
                (minimum + maximum) / 2 @ axes,
                dimensions,
                float(angle),
                corners,
            )
    return best[2:]


def propose_box_tops(depth, camera, world_from_camera, settings=None, top_template=None):
    settings = settings or BoxTopSettings()
    if not isinstance(camera, Camera):
        camera = Camera.from_record(camera)
    transform = rigid(world_from_camera)
    original = np.asarray(depth)
    if (
        original.dtype.kind != "f"
        or original.shape != tuple(reversed(camera.image_size))
        or original.ndim != 2
    ):
        raise ValueError("Use matching float axial depth in meters")
    valid = np.isfinite(original) & (original > 0)
    y, x = np.nonzero(valid)
    z = original[y, x].astype(float)
    k = np.asarray(camera.matrix)
    optical = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    world = optical @ transform[:3, :3].T + transform[:3, 3]
    selected = (
        np.all(world[:, :2] >= settings.minimum, axis=1)
        & np.all(world[:, :2] < settings.maximum, axis=1)
        & (world[:, 2] > settings.floor_z + settings.minimum_height)
    )
    points, pixels = world[selected], np.c_[x[selected], y[selected]]
    shape = np.ceil((np.asarray(settings.maximum) - settings.minimum) / settings.grid_step).astype(
        int
    )
    cells = np.floor((points[:, :2] - settings.minimum) / settings.grid_step).astype(int)
    occupancy = np.zeros(tuple(shape), bool)
    if len(cells):
        occupancy[cells[:, 0], cells[:, 1]] = True
    labels, count = label(binary_dilation(occupancy), structure=np.ones((3, 3), bool))
    components = labels[cells[:, 0], cells[:, 1]] if len(cells) else np.zeros(0, int)
    candidates, rejected = [], []
    support_image = np.zeros(original.shape, np.int32)
    for component in range(1, count + 1):
        indices = np.flatnonzero(components == component)
        cloud = points[indices]
        if len(cloud) < settings.minimum_points:
            continue
        top_z = float(np.percentile(cloud[:, 2], 98))
        top = np.abs(cloud[:, 2] - top_z) <= settings.top_tolerance
        surface = cloud[top]
        reason = None
        if len(surface) < settings.minimum_points:
            reason = "too_few_observed_top_points"
        else:
            try:
                center, dimensions, angle, corners = rectangle(surface[:, :2])
            except QhullError:
                reason = "degenerate_observed_top"
        if reason:
            rejected.append({"component": component, "reason": reason})
            continue
        width, length = map(float, dimensions)
        support_cells = np.unique(cells[indices[top]], axis=0)
        support = min(1.0, len(support_cells) * settings.grid_step**2 / max(width * length, 1e-12))
        box_support = support
        expected = matched = None
        if top_template is not None:
            support, expected, matched = top_template.coverage(surface[:, :2], center, angle)
        flatness = float(np.std(surface[:, 2]))
        if not all(settings.minimum_side <= v <= settings.maximum_side for v in (width, length)):
            reason = "box_top_dimensions"
        elif support < settings.minimum_support:
            reason = "insufficient_observed_top_support"
        elif flatness > settings.maximum_flatness:
            reason = "nonflat_top"
        if reason:
            rejected.append(
                {
                    "component": component,
                    "reason": reason,
                    "observed_dimensions_m": [width, length],
                    "support": support,
                }
            )
            continue
        # A box can be grasped about either side; choose the orientation closest to world X.
        if angle > math.pi / 4:
            angle -= math.pi / 2
            width, length = length, width
        grasp = np.eye(4)
        c, s = math.cos(angle), math.sin(angle)
        grasp[:3, :3] = [[c, s, 0], [s, -c, 0], [0, 0, -1]]
        grasp[:3, 3] = [*center, float(np.median(surface[:, 2])) - settings.tcp_below_top]
        pregrasp = grasp.copy()
        pregrasp[2, 3] += settings.approach_height
        object_center = [*center, float(np.median(surface[:, 2])) - settings.part_height / 2]
        candidate = {
            "component": component,
            "kind": "observed_box_top",
            "world_from_tcp": grasp.tolist(),
            "world_from_pregrasp": pregrasp.tolist(),
            "estimated_center_m": object_center,
            "top_z_m": float(np.median(surface[:, 2])),
            "dimensions_m": [width, length, settings.part_height],
            "yaw_rad": angle,
            "jaw_opening_m": min(settings.maximum_opening, width + settings.jaw_margin),
            "jaw_grasp_m": width,
            "support": support,
            "flatness_std_m": flatness,
            "observed_points": len(surface),
            "corners_xy_m": corners.tolist(),
            "score": float(support * math.exp(-flatness / settings.maximum_flatness)),
            "score_semantics": "geometric_support_not_grasp_probability",
        }
        candidates.append(candidate)
        if expected is not None:
            candidate.update(
                kind="observed_taught_planar_top",
                rectangular_fill_fraction=box_support,
                expected_surface_xy_m=expected.tolist(),
                expected_surface_matched=matched.tolist(),
                matching_tolerance_m=top_template.tolerance_m,
            )
        retained = pixels[indices[top]]
        support_image[retained[:, 1], retained[:, 0]] = component
    candidates.sort(
        key=lambda row: (
            -row["top_z_m"],
            row["estimated_center_m"][0],
            row["estimated_center_m"][1],
        )
    )
    candidates = candidates[: settings.maximum_candidates]
    return {
        "model": "observed_taught_top_candidates_v1"
        if top_template is not None
        else "observed_box_top_candidates_v1",
        "units": "m",
        "frame": "world",
        "camera_frame": camera.frame,
        "candidates": candidates,
        "rejected": rejected,
        "observed_roi_points": len(points),
        "components": count,
        "privileged_pose_input": False,
        "robot_feasibility": "not_assessed",
    }, support_image
