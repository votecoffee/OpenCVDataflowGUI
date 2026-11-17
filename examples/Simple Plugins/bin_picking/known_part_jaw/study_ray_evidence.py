"""Fixed diagnostic ablations; no recipe, pose, threshold or robot updates."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import PointCloud, render_cloud

SAMPLES = (4_000, 40_000, 50_000)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify(depth, x, y, z, tolerance, neighbors):
    """Same integer ray domain; only the depth neighborhood changes."""
    from opencv_simple_bin_picking.jaw_fit import classify_pose_fit

    values = depth[y, x, None]
    if neighbors:
        values = depth[y[:, None] + [0, 0, 1, 1], x[:, None] + [0, 1, 0, 1]]
    known = np.isfinite(values).all(1) & (values > 0).all(1) & (values <= 1e9).all(1)
    nearest, farthest = values.min(1), values.max(1)
    same = np.ones(len(z), bool)
    support = known & (np.maximum(abs(z - nearest), abs(z - farthest)) <= tolerance)
    bad = known & (z < nearest - tolerance)
    states = classify_pose_fit(known, nearest, farthest, z, same, support, bad, tolerance)
    return states, (known, nearest, farthest, z, same, support, bad, tolerance)


def mesh_queries(mesh, pose, camera):
    from opencv_simple_bin_picking.jaw_raster import render_depth

    predicted, incidence, stride, reason = render_depth(mesh, pose, camera)
    iy, ix = np.nonzero(np.isfinite(predicted))
    x, y = ix * stride, iy * stride
    # Hold the domain fixed for one- and four-neighbor policies, including borders.
    keep = (x < camera.image_size[0] - 1) & (y < camera.image_size[1] - 1)
    return (
        x[keep],
        y[keep],
        predicted[iy[keep], ix[keep]],
        incidence[iy[keep], ix[keep]],
        stride,
        reason,
    )


def sample_queries(part, pose, camera, stride):
    """Nearest front-facing sample per rounded full-image pixel, then grid intersection.

    This is an approximate sampled representation, not an exact visibility oracle.
    Depth is evaluated at the integer pixel for the controlled common-ray ablation;
    production evaluates four neighbors at the original fractional sample position.
    """
    points = part.points @ pose[:3, :3].T + pose[:3, 3]
    normals = part.normals @ pose[:3, :3].T
    rows = np.flatnonzero((np.sum(normals * -points, axis=1) > 1e-12) & (points[:, 2] > 1e-8))
    projected = points[rows] @ np.asarray(camera.matrix).T
    uv = projected[:, :2] / projected[:, 2:]
    valid = np.all(uv >= 0, axis=1) & np.all(uv < np.array(camera.image_size) - 1, axis=1)
    rows, uv = rows[valid], np.rint(uv[valid]).astype(int)
    valid = np.all(uv % stride == 0, axis=1) & np.all(uv < np.array(camera.image_size) - 1, axis=1)
    rows, uv = rows[valid], uv[valid]
    keys = uv[:, 1] * camera.image_size[0] + uv[:, 0]
    order = np.lexsort((rows, points[rows, 2], keys))
    _, first = np.unique(keys[order], return_index=True)
    selected = order[first]
    return uv[selected, 0], uv[selected, 1], points[rows[selected], 2]


def evaluate(mesh, parts, pose, camera, depth, tolerance):
    from opencv_simple_bin_picking.jaw_metrics import mesh_statistics, ray_statistics

    x, y, z, incidence, stride, reason = mesh_queries(mesh, pose, camera)
    single, _ = classify(depth, x, y, z, tolerance, False)
    square, _ = classify(depth, x, y, z, tolerance, True)
    result = dict(
        rendered_rays=len(z),
        stride=stride,
        renderer_reason=reason,
        integer_neighborhood_disagreements=int((single != square).sum()),
        same_ray_single=mesh_statistics(single, z, incidence, camera, stride),
        same_ray_four=mesh_statistics(square, z, incidence, camera, stride),
        sampled_representation=[],
    )
    keys = y * camera.image_size[0] + x
    for count, part in parts.items():
        sx, sy, sz = sample_queries(part, pose, camera, stride)
        _, im, ip = np.intersect1d(keys, sy * camera.image_size[0] + sx, return_indices=True)
        px, py, pz = sx[ip], sy[ip], sz[ip]
        common = dict(
            samples=count,
            sampled_grid_rays=len(sz),
            common_mesh_rays=len(im),
            mesh_occupancy_fraction=len(im) / len(z) if len(z) else None,
            sampled_only_rays=len(sz) - len(im),
            predicted_z_difference_p05_p50_p95_m=(
                np.quantile(pz - z[im], [0.05, 0.5, 0.95]).tolist() if len(im) else None
            ),
        )
        for name, neighbors in (("single", False), ("four", True)):
            sampled, _ = classify(depth, px, py, pz, tolerance, neighbors)
            exact, _ = classify(depth, px, py, z[im], tolerance, neighbors)
            common[name] = dict(
                sampled=ray_statistics(sampled),
                triangle=ray_statistics(exact),
                state_disagreements=int((sampled != exact).sum()),
            )
        result["sampled_representation"].append(common)
    return result


def analytic_controls():
    """Independent plane equations supply measured depth; the renderer is under test."""
    from opencv_simple_bin_picking import Camera, PartMesh

    camera = Camera(((80, 0, 32), (0, 80, 32), (0, 0, 1)), (64, 64))
    uv = np.array([[12.5, 12.5], [12.5, 51.5], [51.5, 51.5], [51.5, 12.5]])
    ray = np.c_[(uv - [32, 32]) / 80, np.ones(4)]
    normal = np.array([0.3, 0, 1])
    vertices = ray * (0.5 / (ray @ normal))[:, None]
    triangles = vertices[[[0, 1, 2], [0, 2, 3]]]
    # Both outward-facing layers are visible by normal alone, but only the front
    # layer may contribute model queries at a given ray.
    mesh = PartMesh(np.concatenate([triangles, triangles * 1.2]))
    yy, xx = np.indices((64, 64))
    truth = 0.5 / (1 + 0.3 * (xx - 32) / 80)
    variants = dict(
        complete=truth,
        foreground_occlusion=np.where(xx < 32, truth - 0.05, truth),
        missing_half=np.where(xx < 32, 0, truth),
        mixed_neighbors=np.where((xx + yy) % 2, truth, truth + 0.05),
        sparse_one_ray=np.where((xx == 32) & (yy == 32), truth, 0),
        all_occluded=truth - 0.05,
        all_missing=np.zeros_like(truth),
        wrong_in_front=truth + 0.05,
    )
    result = {}
    for name, depth in variants.items():
        result[name] = evaluate(mesh, {}, np.eye(4), camera, depth, 0.003)
    # Expected first-surface support is analytic, not a render-to-self assertion.
    full = result["complete"]["same_ray_single"]["pixel"]
    assert full["rays"] == 39**2 and full["supported_expected_fraction"] == 1
    assert result["sparse_one_ray"]["same_ray_single"]["pixel"]["testable_mass"] == 1
    assert result["all_occluded"]["same_ray_single"]["pixel"]["support_testable_fraction"] is None
    assert result["wrong_in_front"]["same_ray_single"]["pixel"]["supported_expected_fraction"] == 0
    return result


def write_review(output, obj, camera, views):
    from opencv_simple_bin_picking.jaw_fit import POSE_FIT_COLORS, retain_pose_fit

    folder = output / f"review_object{obj}"
    (folder / "assets").mkdir(parents=True)
    palette = np.array(list(POSE_FIT_COLORS.values()), np.uint8)
    manifest = []
    with ProjectSession(builtin_registry(), directory=folder) as session:
        for index, (name, depth, x, y, z, neighbors, tolerance, capture) in enumerate(views):
            _, args = classify(depth, x, y, z, tolerance, neighbors)
            k = np.asarray(camera.matrix)
            points = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
            fit = retain_pose_fit(name, points, y * camera.image_size[0] + x, *args)
            relative = f"assets/{name}.ply"
            cloud = PointCloud(fit.points, "m", "study_camera", colors=palette[fit.states])
            write_cloud(folder / relative, cloud, binary=True)
            write_json(
                folder / f"assets/{name}.json",
                dict(
                    capture_id=capture,
                    summary=fit.summary,
                    observations=False,
                    integer_ray_queries=True,
                    acceptance_applied=False,
                    classification_neighbors=4 if neighbors else 1,
                    qualification="Frozen diagnostic queries; not accepted poses or sensor data",
                ),
            )
            assert cv2.imwrite(
                str(folder / f"{name}.png"),
                render_cloud(cloud, 900, 640, yaw=155, pitch=-35, stable_scale=True),
            )
            block = session.graph.add(
                session.registry.create(
                    "input.cloud",
                    block_id=name,
                    parameters=dict(path=relative, units="m", frame="study_camera"),
                )
            )
            block.name, block.position = (
                name.replace("_", " "),
                ((index % 3) * 400, (index // 3) * 300),
            )
            block.preview_output = "cloud"
            manifest.append(
                dict(
                    block=name,
                    path=relative,
                    sha256=hashlib.sha256((folder / relative).read_bytes()).hexdigest(),
                )
            )
        session.save(folder / "review.json")
        assert session.inspect()["validation"]["valid"]
        session.run()
        assert all(block.last_result is not None for block in session.graph.blocks.values())
    write_json(folder / "views.json", manifest)


def run(root, output):
    from compare_pose_agreement import panel
    from opencv_simple_3d_locate import read_ply_mesh
    from opencv_simple_bin_picking import Camera, PartInstance, PartMesh
    from opencv_simple_bin_picking.jaw import JawPlanner
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_specs import PartSurface
    from opencv_simple_bin_picking.scene import Scene

    output.mkdir(parents=True, exist_ok=False)
    source = root / "pose_agreement"
    provenance = json.loads((source / "source/provenance.json").read_bytes())
    for name, digest in provenance["sha256"].items():
        assert hashlib.sha256((source / "source" / name).read_bytes()).hexdigest() == digest
    write_json(output / "controls.json", analytic_controls())
    study = json.loads((source / "comparison.json").read_bytes())
    base = load_jaw_recipe(root / "tless/frame000013/assets/trained.sbpj")
    geometry, inputs, records, images = {}, {}, [], []
    baseline_exact = 0
    for record in study["cases"]:
        scene_id, frame, obj, kind = (record[k] for k in ("scene", "frame", "object_id", "model"))
        if (obj, kind) not in geometry:
            original = read_ply_mesh(source / f"source/models_{kind}/obj_{obj:06}.ply", scale=0.001)
            parts = {}
            for count in SAMPLES:
                samples = original.sample(count, 27).astype(float)
                samples[:, 3:] /= np.linalg.norm(samples[:, 3:], axis=1)[:, None]
                parts[count] = PartSurface(samples[:, :3], samples[:, 3:], f"tless{obj}")
            geometry[obj, kind] = PartMesh(original.triangles, f"tless{obj}"), parts
        if (scene_id, frame) not in inputs:
            folder = source / f"source/test_primesense/{scene_id:06}"
            meta = json.loads((folder / "scene_camera.json").read_bytes())[str(frame)]
            raw = cv2.imread(str(folder / f"depth/{frame:06}.png"), -1)
            depth = (raw.astype(float) * meta["depth_scale"] * 0.001).astype(np.float32)
            rgb = cv2.imread(str(folder / f"rgb/{frame:06}.png"))
            camera = Camera(np.array(meta["cam_K"]).reshape(3, 3), depth.shape[1::-1])
            inputs[scene_id, frame] = camera, depth, rgb
        camera, depth, rgb = inputs[scene_id, frame]
        mesh, parts = geometry[obj, kind]
        planner = JawPlanner(
            replace(
                base,
                part=parts[40_000],
                bin=replace(
                    base.bin,
                    camera=camera,
                    reference_depth=depth,
                    bin_to_camera=np.eye(4),
                    frame="study_camera",
                ),
            )
        )
        scene = Scene(planner.recipe.bin, depth)
        initial = np.array(record["annotation_pose"])
        diagonal = float(np.linalg.norm(np.ptp(mesh.triangles.reshape(-1, 3), axis=0)))
        poses = {"before": initial, "after": np.array(record["research_pose"])}
        for name, axis, sign in (
            ("lateral", 0, 1),
            ("toward_camera", 2, -1),
            ("away_camera", 2, 1),
        ):
            poses[name] = initial.copy()
            poses[name][axis, 3] += sign * diagonal * 0.25
        views, pictures = [], []
        for name, pose in poses.items():
            instance = PartInstance("study", pose, True, f"scene{scene_id}_frame{frame}")
            pose = instance.camera_from_part
            actual = planner._pose_evidence(scene, instance)
            if name in ("before", "after"):
                assert actual == record[name]["sampled_pose_gate"]
                baseline_exact += 1
            result = evaluate(mesh, parts, pose, camera, depth, base.settings.pose_tolerance)
            result.update(
                scene=scene_id,
                frame=frame,
                object_id=obj,
                geometry=kind,
                hypothesis=name,
                production_point_gate=actual,
                camera_from_part=pose.tolist(),
            )
            records.append(result)
            if kind == "cad" and (scene_id, frame) in ((2, 13), (3, 261)):
                x, y, z, _, stride, _ = mesh_queries(mesh, pose, camera)
                if name == "before":
                    sx, sy, sz = sample_queries(parts[40_000], pose, camera, stride)
                    views += [
                        ("sample_single", depth, sx, sy, sz, False, 0.003, instance.capture_id),
                        ("sample_four", depth, sx, sy, sz, True, 0.003, instance.capture_id),
                        ("mesh_single", depth, x, y, z, False, 0.003, instance.capture_id),
                        ("mesh_four", depth, x, y, z, True, 0.003, instance.capture_id),
                    ]
                if name in ("toward_camera", "away_camera"):
                    views.append((name, depth, x, y, z, False, 0.003, instance.capture_id))
            if kind == "cad" and name in ("before", "toward_camera"):
                x, y, z, _, _, _ = mesh_queries(mesh, pose, camera)
                for label, neighbors in (("1 ray", False), ("4 neighbors", True)):
                    states, _ = classify(depth, x, y, z, 0.003, neighbors)
                    state = np.zeros(depth.shape, np.uint8)
                    state[y, x] = states + 1
                    region = np.zeros(depth.shape, bool)
                    region[y, x] = True
                    metric = result[f"same_ray_{'four' if neighbors else 'single'}"]["pixel"]
                    conditional = metric["support_testable_fraction"]
                    score = "n/a" if conditional is None else f"{conditional:.1%}"
                    pictures.append(
                        panel(
                            rgb,
                            state,
                            region,
                            f"{name}: {label}; frame {frame}",
                            f"Testable {metric['testable_fraction']:.1%}; S/T {score}",
                        )
                    )
        if views:
            write_review(output, obj, camera, views)
        if pictures and frame in (3, 13, 255, 5, 261, 493):
            images.append(np.concatenate(pictures, axis=1))
        print(f"scene {scene_id} frame {frame} {kind}: five hypotheses", flush=True)
    assert cv2.imwrite(str(output / "overview.png"), np.concatenate(images, axis=0))
    write_json(
        output / "comparison.json",
        dict(
            model="mesh_ray_ablation_v1",
            cases=records,
            baseline_reports_exact=baseline_exact,
            original_member_hashes=provenance["sha256"],
            settings=dict(
                samples=SAMPLES,
                seed=27,
                tolerance_m=base.settings.pose_tolerance,
                negative_translation_model_diagonal=0.25,
            ),
            acceptance_applied=False,
            production_poses_or_recipes_changed=False,
            scope=(
                "Diagnostic perturbations and within-capture comparisons; "
                "no independent grasp labels"
            ),
        ),
    )
    write_json(output / "summary.json", summarize(records, baseline_exact))


def summarize(records, baseline_exact):
    result = dict(cases=len(records), frozen_baseline_reports_exact=baseline_exact, groups={})
    for name in ("before", "after", "lateral", "toward_camera", "away_camera"):
        rows = [r for r in records if r["hypothesis"] == name]
        group = dict(
            cases=len(rows),
            point_gate_accepted=sum(
                r["production_point_gate"]["status"] == "accepted" for r in rows
            ),
        )
        for neighbors in ("single", "four"):
            for weighting in ("pixel", "visible_area"):
                metrics = [r[f"same_ray_{neighbors}"][weighting] for r in rows]
                means = {}
                valid_cases = {}
                for field in (
                    "testable_fraction",
                    "supported_expected_fraction",
                    "support_testable_fraction",
                ):
                    values = [m[field] for m in metrics if m.get(field) is not None]
                    means[field] = float(np.mean(values)) if values else None
                    valid_cases[field] = len(values)
                means["valid_metric_cases"] = valid_cases
                group[f"{neighbors}_{weighting}"] = means
        result["groups"][name] = group
    result["interpretation"] = (
        "Means are descriptive capture/model averages, not probabilities or accuracy; "
        "perturbations are controlled hypotheses, not independently labeled negatives. "
        "Conditional support requires observability and sample counts. No gate is selected."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    registry = builtin_registry()
    load_plugin(args.root.resolve() / "plugin.py", registry)
    run(args.root.resolve(), args.output.resolve())
