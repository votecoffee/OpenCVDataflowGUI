"""Bounded suction proposals, independent of object matching or robot motion."""

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .scene import Scene


def basis(normal):
    x = np.array([1., 0., 0.]) - normal * normal[0]
    if np.linalg.norm(x) < .1:
        x = np.cross([0., 1., 0.], normal)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(normal, x), normal])


def pose(rotation, center):
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = rotation, center
    return value.tolist()


def overlap(first, second, radius):
    distance = np.linalg.norm(np.asarray(first) - second)
    if distance >= 2 * radius:
        return 0.
    return float((2 * radius**2 * np.arccos(distance / (2 * radius)) -
                  .5 * distance * np.sqrt(max(0., 4 * radius**2 - distance**2))) /
                 (np.pi * radius**2))


@dataclass(frozen=True)
class PickResult:
    scene: Scene
    candidates: tuple
    searched_cells: int

    @property
    def picks(self):
        return tuple(c for c in self.candidates if c["status"] == "eligible")

    def report(self):
        return {"model": "simple_bin_picking_v1", "units": self.scene.recipe.units,
                "frame": self.scene.recipe.frame, "camera_frame": self.scene.recipe.camera.frame,
                "bin_to_camera": self.scene.pose.tolist(),
                "observed_points": len(self.scene.points),
                "invalid_depth_pixels": self.scene.depth.size - len(self.scene.points),
                "searched_cells": self.searched_cells, "evaluated_candidates": len(self.candidates),
                "candidate_budget": self.scene.recipe.settings.max_candidates,
                "eligible_picks": len(self.picks),
                "candidates": copy.deepcopy(list(self.candidates)),
                "row_index_scope": "positive_finite_depth_pixels_in_y_then_x_order",
                "score_definition": "geometric_contact_planarity_tilt_not_success_probability",
                "clearance_definition": "observed_points_and_sampled_first_return_free_rays",
                "unobserved_space": "unknown_not_free", "motion_execution": False,
                "qualification": "no_robot_ik_full_robot_collision_or_physical_seal_success",
                "depth_sha256": hashlib.sha256(self.scene.depth.tobytes()).hexdigest()}


class Picker:
    def __init__(self, recipe):
        self.recipe = recipe
        # Full disk samples check interiors as well as the rim; holes are not filled.
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        self.disk = np.r_[np.zeros((1, 3)), np.array([
            [radius * np.cos(t), radius * np.sin(t), 0.]
            for radius in (.4, .7, 1.) for t in theta]) * recipe.settings.cup_radius]
        self.disk.flags.writeable = False

    def _candidate(self, scene, seed, candidate_id):
        s = self.recipe.settings
        point = scene.points[seed]
        pixel = scene.pixels[seed]
        instance = (None if scene.instances is None else int(scene.instances[pixel[1], pixel[0]]))
        item = {"id": candidate_id, "seed_row": int(seed), "seed_pixel": pixel.tolist(),
                "instance_id": instance, "status": "rejected", "reasons": [], "score": 0.,
                "contact_status": "unavailable", "clearance_status": "not_evaluated",
                "center": point.tolist(), "contact_pose": None, "tcp_pose": None,
                "pregrasp_pose": None, "collision_rows": [], "unknown_samples": []}
        neighbors = scene.tree.query_ball_point(point, s.cup_radius * 1.25)
        if len(neighbors) < 12:
            item["reasons"] = ["insufficient_patch_points"]
            return item
        neighbors = np.asarray(neighbors)
        if len(neighbors) > 512:
            neighbors = neighbors[np.linspace(0, len(neighbors) - 1, 512, dtype=int)]
        patch = scene.points[neighbors]
        origin = patch.mean(0)
        eigenvalues, axes = np.linalg.eigh((patch - origin).T @ (patch - origin))
        if eigenvalues[1] <= max(eigenvalues[2] * 1e-8, 1e-20):
            item["reasons"] = ["degenerate_patch"]
            return item
        normal = axes[:, 0]
        if normal[2] < 0:
            normal = -normal
        tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
        rotation = basis(normal)
        center = point - normal * ((point - origin) @ normal)
        item.update(center=center.tolist(), normal=normal.tolist(), tilt_degrees=tilt,
            contact_pose=pose(rotation, center),
            tcp_pose=pose(rotation, center + normal * s.tcp_offset),
            pregrasp_pose=pose(rotation, center + normal * (s.tcp_offset + s.pregrasp_distance)),
            approach_direction=(-normal).tolist(), retract_direction=normal.tolist())
        if tilt > s.max_tilt_degrees:
            item["reasons"] = ["surface_tilt"]
            return item
        samples = self.disk @ rotation.T + center
        observed, gaps, same = scene.contact(samples, center, normal, instance)
        supported = observed & same & (gaps <= s.surface_tolerance)
        contact_fraction = float(supported.mean())
        rmse = float(np.sqrt(np.mean(((patch - origin) @ normal)**2)))
        item.update(contact_fraction=contact_fraction, contact_samples=len(samples),
            observed_contact_samples=int(observed.sum()), patch_rmse=rmse,
            contact_status="supported" if observed.all() and supported.mean() >=
                s.min_contact_fraction else "unknown" if not observed.all() else "unsupported")
        if not observed.all():
            item["status"] = "unknown"
            item["reasons"].append("unobserved_contact")
        if contact_fraction < s.min_contact_fraction:
            item["reasons"].append("insufficient_contact_support")
        if scene.instances is not None and (instance == 0 or not same.all()):
            item["reasons"].append("instance_boundary")
        if item["reasons"]:
            return item
        # Conservative cylinder covers the entire configured gripper and its retract sweep.
        start, end = s.surface_tolerance, s.body_length + s.pregrasp_distance
        radius = s.body_radius + s.clearance
        extent = radius * np.sqrt(np.maximum(0., 1 - normal**2))
        low = np.minimum(center + start * normal, center + end * normal) - extent
        high = np.maximum(center + start * normal, center + end * normal) + extent
        if (np.any(low[:2] < np.asarray(self.recipe.minimum)[:2] + s.clearance) or
                np.any(high[:2] > np.asarray(self.recipe.maximum)[:2] - s.clearance) or
                low[2] < self.recipe.minimum[2]):
            item["reasons"] = ["configured_bin_collision"]
            item["clearance_status"] = "bin_collision"
            return item
        middle = center + (start + end) / 2 * normal
        near = np.asarray(scene.tree.query_ball_point(middle,
                          np.hypot((end - start) / 2, radius)), int)
        delta = scene.points[near] - center
        axial = delta @ normal
        radial = np.linalg.norm(delta - axial[:, None] * normal, axis=1)
        collisions = near[(axial > start) & (axial <= end) & (radial <= radius)]
        item["collision_points"] = len(collisions)
        item["collision_rows"] = collisions[:128].tolist()
        if len(collisions):
            item["reasons"] = ["observed_tool_sweep_collision"]
            item["clearance_status"] = "observed_collision"
            return item
        # Sample boundary and interior of the swept cylinder. Query depth on four rays;
        # anything behind/missing/outside a first return remains unknown.
        count = min(64, max(3, int(np.ceil(end / max(s.body_radius, s.surface_tolerance)))))
        t = np.linspace(2 * s.surface_tolerance, end, count)
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        radial_samples = np.r_[np.zeros((1, 3)), np.array([
            [r * np.cos(a), r * np.sin(a), 0.] for r in (.5 * radius, radius) for a in theta])]
        sweep = (radial_samples[None, :, :] @ rotation.T +
                 center + t[:, None, None] * normal).reshape(-1, 3)
        known, nearest, _, _, z = scene.sample(sweep)
        free = known & (z < nearest - s.surface_tolerance)
        unknown = sweep[~free]
        item.update(clearance_samples=len(sweep), free_samples=int(free.sum()),
                    unknown_sample_count=len(unknown), unknown_samples=unknown[:128].tolist())
        if len(unknown):
            item["status"], item["clearance_status"] = "unknown", "unknown"
            item["reasons"] = ["unobserved_or_occluded_tool_sweep"]
            return item
        item["clearance_status"] = "sampled_free"
        item["status"] = "eligible"
        item["score"] = float(.5 * contact_fraction + .3 * max(0., 1 - rmse /
                                   s.surface_tolerance) + .2 * (1 - tilt / s.max_tilt_degrees))
        return item

    def plan(self, depth, *, bin_to_camera=None, instances=None):
        scene = Scene(self.recipe, depth, bin_to_camera=bin_to_camera, instances=instances)
        s = self.recipe.settings
        rows = np.flatnonzero(np.all(scene.points >= self.recipe.minimum, axis=1) &
                             np.all(scene.points <= self.recipe.maximum, axis=1) &
                             (scene.points[:, 2] >= self.recipe.minimum[2] + s.min_pick_height))
        if scene.instances is not None:
            px = scene.pixels[rows]
            rows = rows[scene.instances[px[:, 1], px[:, 0]] > 0]
        # One highest visible proposal per physical cup-radius XY cell, then a fixed budget.
        order = rows[np.lexsort((rows, -scene.points[rows, 2]))]
        cells = np.floor(scene.points[order, :2] / s.cup_radius).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        seeds = order[np.sort(first)][:s.max_candidates]
        jobs = list(enumerate(seeds))
        def evaluate(job):
            i, seed = job
            return self._candidate(scene, seed, i)
        if s.workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=s.workers, thread_name_prefix="bin-pick") as pool:
                candidates = list(pool.map(evaluate, jobs))
        else:
            candidates = list(map(evaluate, jobs))
        candidates.sort(key=lambda c: (c["status"] != "eligible", -c["score"],
                                       -c["center"][2], c["id"]))
        accepted = []
        for candidate in candidates:
            if candidate["status"] != "eligible":
                continue
            other = next((c for c in accepted if overlap(c["center"], candidate["center"],
                                                        s.cup_radius) > s.overlap_threshold), None)
            if other is not None:
                candidate.update(status="rejected", reasons=["overlap_with_higher_ranked"],
                                 overlap_candidate=other["id"])
            elif len(accepted) >= s.max_picks:
                candidate.update(status="rejected", reasons=["pick_count_limit"])
            else:
                candidate["rank"] = len(accepted) + 1
                accepted.append(candidate)
        return PickResult(scene, tuple(candidates), len(first))
