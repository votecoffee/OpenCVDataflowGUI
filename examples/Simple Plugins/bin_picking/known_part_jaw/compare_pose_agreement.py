"""Frozen CAD/reconstruction and bounded-ICP study; never updates runtime recipes or poses."""

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin


@dataclass(frozen=True)
class StudySettings:
    model_samples: int = 40_000
    sample_seed: int = 27
    tile_pixels: int = 16
    boundary_pixels: int = 2
    iterations: int = 30
    correspondence_tolerances: float = 5.0
    retained_fraction: float = 0.75
    translation_tolerances: float = 3.0
    translation_diagonal: float = 0.05
    maximum_rotation_degrees: float = 10.0


SETTINGS = StudySettings()
CASES = ((2, 6, (3, 13, 24, 255, 494)), (3, 8, (5, 10, 16, 261, 493)))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_masks(mask, settings=SETTINGS):
    """Disjoint spatial tiles within a capture, using no depth values."""
    width = settings.boundary_pixels * 2 + 1
    region = cv2.erode(np.asarray(mask, np.uint8), np.ones((width, width), np.uint8)) > 0
    y, x = np.indices(region.shape)
    first = (x // settings.tile_pixels + y // settings.tile_pixels) % 2 == 0
    return region & first, region & ~first


def training_points(depth, selected, camera):
    y, x = np.nonzero(selected & np.isfinite(depth) & (depth > 0))
    z = depth[y, x].astype(float)
    return np.c_[(x - camera[0, 2]) * z / camera[0, 0], (y - camera[1, 2]) * z / camera[1, 1], z]


def rigid_fit(source, target):
    """Proper point-to-point least-squares transform; no scale or calibration fit."""
    a, b = source.mean(0), target.mean(0)
    u, singular, vt = np.linalg.svd((source - a).T @ (target - b))
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    delta = np.eye(4)
    delta[:3, :3], delta[:3, 3] = rotation, b - rotation @ a
    return delta, singular


def refine(model, initial, observed, train_mask, camera, tolerance, settings=SETTINGS):
    """Research proposal using training pixels only; production acceptance is separate."""
    from opencv_simple_bin_picking.specs import rigid

    pose = np.array(initial, copy=True)
    tree = cKDTree(observed) if len(observed) >= 30 else None
    limit = max(
        tolerance * settings.translation_tolerances,
        np.linalg.norm(np.ptp(model.points, axis=0)) * settings.translation_diagonal,
    )
    reason, used, iterations = "insufficient_training_points", 0, 0
    for step in range(settings.iterations if tree is not None else 0):
        iterations = step + 1
        points = model.points @ pose[:3, :3].T + pose[:3, 3]
        normals = model.normals @ pose[:3, :3].T
        ids = np.flatnonzero((np.sum(normals * -points, axis=1) > 0) & (points[:, 2] > 0))
        projected = points[ids] @ camera.T
        uv = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
        inside = np.all(uv >= 0, axis=1) & np.all(uv < train_mask.shape[::-1], axis=1)
        ids, uv = ids[inside], uv[inside]
        ids = ids[train_mask[uv[:, 1], uv[:, 0]]]
        if len(ids) < 30:
            reason = "insufficient_projected_training_points"
            break
        distance, rows = tree.query(points[ids])
        keep = distance < tolerance * settings.correspondence_tolerances
        cutoff = np.quantile(distance[keep], settings.retained_fraction) if keep.any() else 0
        keep &= distance <= cutoff
        used = int(keep.sum())
        if used < 30:
            reason = "insufficient_correspondences"
            break
        delta, singular = rigid_fit(points[ids[keep]], observed[rows[keep]])
        if singular[1] <= max(singular[0], 1e-30) * 1e-8:
            reason = "degenerate_correspondences"
            break
        candidate = delta @ pose
        change = candidate[:3, :3] @ initial[:3, :3].T
        angle = np.rad2deg(np.arccos(np.clip((np.trace(change) - 1) / 2, -1, 1)))
        if angle > settings.maximum_rotation_degrees or (
            np.linalg.norm(candidate[:3, 3] - initial[:3, 3]) > limit
        ):
            reason = "update_bound"
            break
        pose = rigid(candidate)
        reason = "iteration_limit"
        if np.linalg.norm(delta[:3, 3]) < 1e-6 and np.linalg.norm(delta[:3, :3] - np.eye(3)) < 1e-5:
            reason = "converged"
            break
    change = pose[:3, :3] @ initial[:3, :3].T
    return pose, dict(
        termination=reason,
        iterations=iterations,
        correspondences=used,
        translation_change_m=float(np.linalg.norm(pose[:3, 3] - initial[:3, 3])),
        rotation_change_degrees=float(
            np.rad2deg(np.arccos(np.clip((np.trace(change) - 1) / 2, -1, 1)))
        ),
        translation_bound_m=float(limit),
        maximum_rotation_degrees=settings.maximum_rotation_degrees,
        production_pose_updated=False,
    )


def ratio(evidence):
    count = evidence["known_samples"]
    return evidence["supported_samples"] / count if count else None


def raster_evidence(mesh, pose, camera, depth, held, tolerance):
    from opencv_simple_3d_locate.projective import render_depth

    predicted, factors, stride, reason = render_depth(mesh, pose, camera)
    measured = depth[::stride, ::stride]
    expected = np.isfinite(predicted)
    known = expected & np.isfinite(measured) & (measured > 0)
    delta = predicted - measured
    support = known & (abs(delta) <= tolerance)
    contradiction = known & (delta < -tolerance)
    occluded = known & (delta > tolerance)
    withheld = known & held[::stride, ::stride]
    n = int(expected.sum())
    details = dict(
        grid_stride_pixels=stride,
        renderer_reason=reason,
        rendered_rays=n,
        known_rays=int(known.sum()),
        supported_rays=int(support.sum()),
        contradiction_rays=int(contradiction.sum()),
        occluded_rays=int(occluded.sum()),
        unknown_rays=int((expected & ~known).sum()),
        expected_ray_support=float(support.sum() / n) if n else None,
        heldout_known_rays=int(withheld.sum()),
        heldout_known_ray_support=float(support[withheld].mean()) if withheld.any() else None,
        normal_residual_p05_p50_p95_m=(
            np.quantile(delta[known] * factors[known], [0.05, 0.5, 0.95]).tolist()
            if known.any()
            else None
        ),
        scope="Exact triangle rays at reported stride; axial tolerance, no gate replacement",
    )
    state = np.zeros(depth.shape, np.uint8)
    sub = state[::stride, ::stride]
    for selected, code in ((support, 1), (contradiction, 2), (occluded, 3), (expected & ~known, 4)):
        sub[selected] = code
    return details, state


def panel(rgb, state, mask, title, text):
    y, x = np.nonzero(mask)
    low = np.maximum([x.min() - 18, y.min() - 18], 0)
    high = np.minimum([x.max() + 19, y.max() + 19], rgb.shape[1::-1])
    crop = rgb[low[1] : high[1], low[0] : high[0]].copy()
    labels = state[low[1] : high[1], low[0] : high[0]]
    colors = np.array(
        [
            [0, 0, 0],
            [100, 230, 65],
            [210, 55, 245],
            [245, 135, 70],
            [45, 165, 255],
            [65, 230, 245],
            [235, 95, 165],
        ]
    )
    active = labels > 0
    crop[active] = np.rint(crop[active] * 0.35 + colors[labels[active]] * 0.65).astype(np.uint8)
    scale = min(300 / crop.shape[1], 320 / crop.shape[0])
    crop = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    image = np.full((398, 300, 3), (27, 24, 22), np.uint8)
    x, y = (300 - crop.shape[1]) // 2, 50 + (320 - crop.shape[0]) // 2
    image[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    cv2.putText(image, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(image, text, (8, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1)
    cv2.putText(
        image,
        "Annotation/mask assisted study",
        (8, 388),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (210, 210, 210),
        1,
    )
    return image


def replay_project(output, object_id, capture, views):
    """Frozen derived-query review using built-in file inputs, never a pose detector."""
    from opencv_simple_bin_picking.jaw_fit import POSE_FIT_COLORS

    from opencv_dataflow.assistant_api import ProjectSession
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.pointclouds import PointCloud

    folder = output / f"review_object{object_id}"
    (folder / "assets").mkdir(parents=True)
    palette = np.array(list(POSE_FIT_COLORS.values()), np.uint8)
    manifest = []
    with ProjectSession(builtin_registry(), directory=folder) as session:
        for kind, name, fit, evidence in views:
            cloud = PointCloud(fit.points, "m", "study_camera", colors=palette[fit.states])
            relative = f"assets/{kind}_{name}.ply"
            write_cloud(folder / relative, cloud, binary=True)
            details = dict(
                model="frozen_pose_agreement_view_v1",
                capture_id=capture,
                derived_queries=True,
                sensor_observations=False,
                camera_from_study_frame=np.eye(4).tolist(),
                sampled_pose_gate=evidence,
                summary=fit.summary,
                colors_rgb=POSE_FIT_COLORS,
                original_model_rows=fit.model_rows.tolist(),
                qualification="Annotation/mask-assisted diagnostic; not an accepted runtime pose",
            )
            write_json(folder / f"assets/{kind}_{name}.json", details)
            block = session.graph.add(
                session.registry.create(
                    "input.cloud",
                    block_id=f"{kind}_{name}",
                    parameters=dict(path=relative, units="m", frame="study_camera"),
                )
            )
            block.name = f"{kind} {name}: {evidence['status']} / derived"
            block.position = (400 if kind == "reconst" else 0, 300 if name == "after" else 0)
            block.preview_output = "cloud"
            manifest.append(
                dict(
                    block=block.id,
                    path=relative,
                    sha256=hashlib.sha256((folder / relative).read_bytes()).hexdigest(),
                )
            )
        session.save(folder / "review.json")
        assert session.inspect()["validation"]["valid"]
        session.run()
        assert all(block.last_result is not None for block in session.graph.blocks.values())
    write_json(folder / "views.json", manifest)


def summarize(cases):
    changes = [
        row["after"]["heldout_known_support"] - row["before"]["heldout_known_support"]
        for row in cases
    ]
    result = dict(
        model_capture_cases=len(cases),
        captures=len(cases) // 2,
        heldout_improved=sum(value > 1e-12 for value in changes),
        heldout_degraded=sum(value < -1e-12 for value in changes),
        heldout_unchanged=sum(abs(value) <= 1e-12 for value in changes),
        pose_gate_accepted_before=sum(
            row["before"]["sampled_pose_gate"]["status"] == "accepted" for row in cases
        ),
        pose_gate_accepted_after=sum(
            row["after"]["sampled_pose_gate"]["status"] == "accepted" for row in cases
        ),
        model_averages={},
        production_refinement_enabled=False,
    )
    for kind in ("cad", "reconst"):
        selected = [row for row in cases if row["model"] == kind]
        result["model_averages"][kind] = {
            stage: dict(
                sampled_pose_coverage=float(
                    np.mean([row[stage]["sampled_pose_gate"]["coverage"] for row in selected])
                ),
                triangle_expected_ray_support=float(
                    np.mean([row[stage]["triangle"]["expected_ray_support"] for row in selected])
                ),
            )
            for stage in ("before", "after")
        }
    return result


def compare(root, source, output):
    from opencv_simple_3d_locate import Camera as RenderCamera
    from opencv_simple_3d_locate import read_ply_mesh
    from opencv_simple_bin_picking.jaw import JawPlanner
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_specs import PartInstance, PartSurface
    from opencv_simple_bin_picking.scene import Scene
    from opencv_simple_bin_picking.specs import Camera, rigid

    output.mkdir(parents=True, exist_ok=False)
    provenance_file = source / "reference_provenance.json"
    if not provenance_file.exists():
        provenance_file = source / "provenance.json"
    provenance = json.loads(provenance_file.read_bytes())
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["revision"] == "5fd309a04476a842d93abfb584fba9ee7caecdf1"
    copied = {}

    def member(name):
        path = source / name
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert digest == provenance["sha256"][name], name
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        copied[name] = digest
        return path

    recipe_path = root / "tless/frame000013/assets/trained.sbpj"
    base = load_jaw_recipe(recipe_path)
    meshes, models = {}, {}
    for _, obj, _ in CASES:
        for kind in ("cad", "reconst"):
            mesh = read_ply_mesh(member(f"models_{kind}/obj_{obj:06}.ply"), scale=0.001)
            points = mesh.sample(SETTINGS.model_samples, SETTINGS.sample_seed).astype(float)
            points[:, 3:] /= np.linalg.norm(points[:, 3:], axis=1)[:, None]
            meshes[obj, kind] = mesh
            models[obj, kind] = PartSurface(points[:, :3], points[:, 3:], f"tless{obj}")
    rows, images = [], []
    for scene_id, obj, frames in CASES:
        prefix = f"test_primesense/{scene_id:06}"
        cameras = json.loads(member(prefix + "/scene_camera.json").read_bytes())
        poses = json.loads(member(prefix + "/scene_gt.json").read_bytes())
        for frame in frames:
            meta = cameras[str(frame)]
            index, target = next(
                (i, g) for i, g in enumerate(poses[str(frame)]) if g["obj_id"] == obj
            )
            raw = cv2.imread(str(member(prefix + f"/depth/{frame:06}.png")), -1)
            rgb = cv2.imread(str(member(prefix + f"/rgb/{frame:06}.png")))
            mask = cv2.imread(str(member(prefix + f"/mask_visib/{frame:06}_{index:06}.png")), 0) > 0
            depth = (raw.astype(float) * meta["depth_scale"] * 0.001).astype(np.float32)
            k = np.array(meta["cam_K"]).reshape(3, 3)
            train, held = split_masks(mask)
            observed = training_points(depth, train, k)
            pose = np.eye(4)
            pose[:3, :3] = np.array(target["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.array(target["cam_t_m2c"]) * 0.001
            pose = rigid(pose)
            bin_recipe = replace(
                base.bin,
                camera=Camera(k, depth.shape[1::-1]),
                reference_depth=depth,
                bin_to_camera=np.eye(4),
                frame="study_camera",
            )
            full_scene = Scene(bin_recipe, depth)
            held_scene = Scene(bin_recipe, np.where(held, depth, 0))
            render_camera = RenderCamera(k, depth.shape[1::-1], "m")
            pictures, views = [], []
            for kind in ("cad", "reconst"):
                planner = JawPlanner(replace(base, bin=bin_recipe, part=models[obj, kind]))
                refined, diagnostics = refine(
                    models[obj, kind], pose, observed, train, k, base.settings.pose_tolerance
                )
                record = dict(
                    scene=scene_id,
                    frame=frame,
                    object_id=obj,
                    model=kind,
                    annotation_pose=pose.tolist(),
                    research_pose=refined.tolist(),
                    refinement=diagnostics,
                    segmentation_and_pose_assistance=True,
                )
                for name, matrix in (("before", pose), ("after", refined)):
                    instance = PartInstance("study", matrix, True, f"scene{scene_id}_frame{frame}")
                    review = []
                    evidence = planner._pose_evidence(full_scene, instance, review)
                    views.append((kind, name, review[0], evidence))
                    held_evidence = planner._pose_evidence(held_scene, instance)
                    raster, state = raster_evidence(
                        meshes[obj, kind],
                        matrix,
                        render_camera,
                        depth,
                        held,
                        base.settings.pose_tolerance,
                    )
                    record[name] = dict(
                        sampled_pose_gate=evidence,
                        heldout_sampled=held_evidence,
                        heldout_known_support=ratio(held_evidence),
                        triangle=raster,
                    )
                    pictures.append(
                        panel(
                            rgb,
                            state,
                            mask,
                            f"{kind}: {name}",
                            f"Point gate {evidence['coverage']:.1%} / {evidence['status']}",
                        )
                    )
                rows.append(record)
                print(
                    scene_id,
                    frame,
                    kind,
                    record["before"]["sampled_pose_gate"]["coverage"],
                    record["after"]["sampled_pose_gate"]["coverage"],
                    flush=True,
                )
            picture = np.concatenate(pictures, axis=1)
            assert cv2.imwrite(str(output / f"scene{scene_id}_frame{frame:06}.png"), picture)
            if frame in (3, 13, 255, 5, 261, 493):
                images.append(picture)
            if (scene_id, frame) in ((2, 13), (3, 261)):
                replay_project(output, obj, f"scene{scene_id}_frame{frame}", views)
    assert cv2.imwrite(str(output / "overview.png"), np.concatenate(images, axis=0))
    write_json(
        output / "source/provenance.json",
        dict(
            dataset="T-LESS",
            license=provenance["license"],
            revision=provenance["revision"],
            source="https://cmp.felk.cvut.cz/t-less/",
            sha256=copied,
            changes="Unchanged selected members; poses/masks are explicit diagnostic assistance",
        ),
    )
    write_json(
        output / "comparison.json",
        dict(
            model="pose_agreement_study_v1",
            settings=vars(SETTINGS),
            cases=rows,
            recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            source_pose_gate_settings=dict(
                tolerance_m=base.settings.pose_tolerance,
                minimum_support=base.settings.minimum_pose_support,
                maximum_contradiction=base.settings.maximum_pose_contradiction,
            ),
            split_scope="Disjoint capture tiles; not independent sessions or pose truth",
            fitting_scope="Training depth only; no held-out normals, values or calibration fitting",
            runtime_recipes_and_poses_modified=False,
            robot_execution=False,
            publisher_paper="https://cmp.felk.cvut.cz/~hodanto2/data/hodan2017tless.pdf",
        ),
    )
    write_json(output / "summary.json", summarize(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    registry = builtin_registry()
    load_plugin(args.root / "plugin.py", registry)
    compare(args.root.resolve(), args.source.resolve(), args.output.resolve())
