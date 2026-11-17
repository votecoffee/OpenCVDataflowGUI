"""Fixed jaw controls and attributed T-LESS pose-assisted replay; no robot calls."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import render_cloud


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def control():
    from opencv_simple_bin_picking import Camera, JawGrasp, JawRecipe, PartSurface, Recipe

    points, normals = [], []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            uv = np.stack(np.meshgrid(*[np.linspace(-0.02, 0.02, 41)] * 2), -1).reshape(-1, 2)
            data, normal = np.zeros((len(uv), 3)), np.zeros((len(uv), 3))
            data[:, axis], data[:, others], normal[:, axis] = sign * 0.02, uv, sign
            points.append(data)
            normals.append(normal)
    depth = np.full((240, 320), 0.5, np.float32)
    y, x = np.indices(depth.shape)
    target = (abs((x - 160) * 0.46 / 1200) <= 0.02) & (abs((y - 120) * 0.46 / 1200) <= 0.02)
    depth[target] = 0.46
    camera = Camera(((1200, 0, 160), (0, 1200, 120), (0, 0, 1)), (320, 240))
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 0.5
    bin_recipe = Recipe(camera, depth, pose, (-0.1, -0.08, 0), (0.1, 0.08, 0.1), units="m")
    surface = PartSurface(np.concatenate(points), np.concatenate(normals), "box")
    grasps = (
        JawGrasp.from_contacts("sides_x", [-0.02, 0, 0], [0.02, 0, 0], [0, 0, 1]),
        JawGrasp.from_contacts("sides_y", [0, -0.02, 0], [0, 0.02, 0], [0, 0, 1]),
    )
    recipe = JawRecipe(bin_recipe, surface, grasps, notes={"source": "authored first-return box"})
    pose = pose.copy()
    pose[2, 3] = 0.48
    return recipe, depth, pose


def record(recipe, pose, identity, *, assisted):
    return dict(
        capture_id=identity,
        pose_assistance=assisted,
        camera=dict(
            model="camera_model_v1",
            matrix=recipe.bin.camera.matrix,
            image_size=recipe.bin.camera.image_size,
            frame=recipe.bin.camera.frame,
            units="m",
            lens="pinhole",
            distortion=[],
        ),
        part_instances=[
            dict(
                model="part_instance_v1",
                id="part-1",
                accepted=True,
                capture_id=identity,
                pose=dict(
                    model="pose_3d_v1",
                    source_frame=recipe.part.frame,
                    target_frame=recipe.bin.camera.frame,
                    units="m",
                    source_to_target=pose.tolist(),
                ),
            )
        ],
    )


def project(output, registry, recipe, depth, pose, *, assisted):
    from opencv_simple_bin_picking.jaw_persistence import save_jaw_recipe

    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    digest = save_jaw_recipe(assets / "trained.sbpj", recipe)
    np.save(assets / "depth.npy", depth, allow_pickle=False)
    write_json(assets / "records.json", record(recipe, pose, output.name, assisted=assisted))
    with ProjectSession(registry, directory=output) as session:
        graph = session.graph
        source = graph.add(
            registry.create(
                "input.simple_jaw_example",
                block_id="capture",
                parameters={
                    "depth_path": "assets/depth.npy",
                    "records_path": "assets/records.json",
                },
            )
        )
        source.position = [0, 0]
        source.name = "Recorded first return + supplied pose (assisted)"
        picker = graph.add(
            registry.create(
                "3d.simple_bin_picking_jaw",
                block_id="jaw",
                parameters={"recipe_path": "assets/trained.sbpj", "sha256": digest},
            )
        )
        picker.position = [480, 0]
        picker.description = (
            "CAD-assisted jaw alternatives. Inspect graphics in native fly-around. "
            "No force-closure, held-part transport or robot execution qualification."
        )
        for key in ("depth", "camera", "part_instances", "capture_id"):
            graph.connect("capture", key, "jaw", key)
        assert session.inspect()["validation"]["valid"]
        session.save(output / "project.json")
        session.run()
        result = picker.last_result
        assert result is not None, picker.error
        report = result.outputs["report"]
        write_json(output / "expected.json", report)
        cloud = result.outputs["graphics"]
        write_cloud(output / "result.ply", cloud, binary=True)
        write_json(output / "graphics.json", cloud.metadata)
        cv2.imwrite(
            str(output / "result.png"),
            render_cloud(
                cloud, width=1000, height=740, yaw=155, pitch=-35, point_size=2, limit=40000
            ),
        )
    with ProjectSession.open(output / "project.json", registry) as reopened:
        reopened.run()
        picker = reopened.graph.blocks["jaw"]
        assert picker.last_result.outputs["report"] == report
        cached = picker._planner
        picker.set_graphics_mode("off")
        reopened.run()
        assert picker.last_result.outputs["report"] == report and picker._planner is cached
    return dict(
        case=output.name,
        eligible=report["eligible_grasps"],
        statuses=[c["status"] for c in report["candidates"]],
        reasons=[c["reasons"] for c in report["candidates"]],
        reopen_exact=True,
        cached_graphics_off_exact=True,
        pose_assistance=assisted,
    )


def tless(source, output, registry):
    from opencv_simple_bin_picking import Camera, JawGrasp, JawRecipe, PartSurface, Recipe
    from opencv_simple_bin_picking.specs import rigid

    provenance = json.loads((source / "provenance.json").read_bytes())
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["revision"] == "5fd309a04476a842d93abfb584fba9ee7caecdf1"
    names = [
        "models_cad/obj_000006.ply",
        "test_primesense/000002/scene_camera.json",
        "test_primesense/000002/scene_gt.json",
    ]
    names += [
        f"test_primesense/000002/{kind}/{frame:06}.png"
        for frame in (3, 13)
        for kind in ("rgb", "depth")
    ]
    for name in names:
        raw = (source / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["sha256"][name], name
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    write_json(
        output / "provenance.json",
        dict(
            dataset="T-LESS",
            license="CC-BY-4.0",
            source="https://cmp.felk.cvut.cz/t-less/",
            revision=provenance["revision"],
            sha256={n: provenance["sha256"][n] for n in names},
            changes="mm to m; deterministic area-weighted CAD face sampling; authored axial grasps",
            assistance="Ground-truth part poses; calibrated tabletop bounds; no grasp labels",
        ),
    )
    shutil.copy2(output / "provenance.json", output / "source/provenance.json")
    shutil.copy2(Path(__file__).with_name("ATTRIBUTION_TLESS.md"), output / "ATTRIBUTION.md")
    # Pinned upstream ASCII triangle PLY only; no general CAD import claim.
    lines = (source / names[0]).read_text(encoding="ascii").splitlines()
    assert lines[1] == "format ascii 1.0"
    nv = int(next(s for s in lines if s.startswith("element vertex ")).split()[-1])
    nf = int(next(s for s in lines if s.startswith("element face ")).split()[-1])
    start = lines.index("end_header") + 1
    vertices = (
        np.array([[float(v) for v in s.split()[:3]] for s in lines[start : start + nv]]) * 0.001
    )
    faces = np.array([[int(v) for v in s.split()[1:]] for s in lines[start + nv : start + nv + nf]])
    assert faces.shape == (nf, 3)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(normals, axis=1)
    valid = area > 1e-15
    triangles, normals, area = triangles[valid], normals[valid], area[valid]
    normals /= area[:, None]
    rng = np.random.default_rng(27)
    rows = rng.choice(len(area), 40_000, p=area / area.sum())
    uv = rng.random((len(rows), 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    points = (
        triangles[rows, 0]
        + uv[:, :1] * (triangles[rows, 1] - triangles[rows, 0])
        + uv[:, 1:] * (triangles[rows, 2] - triangles[rows, 0])
    )
    surface = PartSurface(points, normals[rows], "tless6")
    lo, hi = vertices.min(0), vertices.max(0)
    center = (lo + hi) / 2
    grasps = []
    for axis in (0, 1):
        a, b = center.copy(), center.copy()
        a[axis], b[axis] = lo[axis], hi[axis]
        grasps.append(JawGrasp.from_contacts(f"authored_axis_{axis}", a, b, [0, 0, 1]))
    cameras = json.loads((source / names[1]).read_bytes())
    gt = json.loads((source / names[2]).read_bytes())
    reports = []
    for number in (3, 13):
        key = str(number)
        raw_depth = cv2.imread(str(source / f"test_primesense/000002/depth/{number:06}.png"), -1)
        depth = (raw_depth.astype(np.float64) * cameras[key]["depth_scale"] * 0.001).astype(
            np.float32
        )
        camera = Camera(np.array(cameras[key]["cam_K"]).reshape(3, 3), depth.shape[1::-1])
        pose = np.eye(4)
        pose[:3, :3] = np.array(cameras[key]["cam_R_w2c"]).reshape(3, 3)
        pose[:3, 3] = np.array(cameras[key]["cam_t_w2c"]) * 0.001
        base = Recipe(
            camera,
            depth,
            rigid(pose),
            (-0.3, -0.3, -0.05),
            (0.3, 0.3, 0.3),
            units="m",
            frame="world",
        )
        recipe = JawRecipe(
            base,
            surface,
            tuple(grasps),
            notes=dict(
                dataset="T-LESS",
                license="CC-BY-4.0",
                attribution="Hodan et al., WACV 2017",
                source="https://cmp.felk.cvut.cz/t-less/",
                upstream_sha256=provenance["sha256"][names[0]],
                changes=("CAD mm to m; sampled face normals; authored grasps, not grasp truth"),
            ),
        )
        target = next(g for g in gt[key] if g["obj_id"] == 6)
        object_pose = np.eye(4)
        object_pose[:3, :3] = np.array(target["cam_R_m2c"]).reshape(3, 3)
        object_pose[:3, 3] = np.array(target["cam_t_m2c"]) * 0.001
        reports.append(
            project(
                output / f"frame{number:06}",
                registry,
                recipe,
                depth,
                object_pose,
                assisted="T-LESS annotated object pose; not independent localization",
            )
        )
    return reports


def build(output, wheel, source=None, locator_wheel=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / "releases").mkdir()
    shutil.copy2(wheel, output / "releases" / wheel.name)
    if locator_wheel is None:
        locator_wheel = Path(__file__).with_name("releases") / (
            "opencv_simple_3d_locate-0.1.0.dev20-py3-none-any.whl"
        )
    shutil.copy2(locator_wheel, output / "releases" / locator_wheel.name)
    shutil.copy2(Path(__file__).with_name("plugin.py"), output / "plugin.py")
    registry = builtin_registry()
    load_plugin(output / "plugin.py", registry)
    recipe, depth, pose = control()
    cases = {"clear": depth.copy(), "obstacle": depth.copy(), "unknown": depth.copy()}
    cases["obstacle"][105:135, 242:254] = 0.43
    cases["unknown"][113:127, 246:250] = 0
    reports = [
        project(output / name, registry, recipe, data, pose, assisted="Authored analytic box pose")
        for name, data in cases.items()
    ]
    if source is not None:
        reports += tless(source, output / "tless", registry)
    assert reports[0]["eligible"] >= 1
    # The Y alternative can remain usable when an X-side obstacle is observed.
    assert reports[1]["eligible"] < reports[0]["eligible"] or any(
        "observed_gripper_sweep_collision" in reasons for reasons in reports[1]["reasons"]
    )
    write_json(output / "verification.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--locator-wheel", type=Path)
    args = parser.parse_args()
    build(args.output, args.wheel, args.source, args.locator_wheel)
