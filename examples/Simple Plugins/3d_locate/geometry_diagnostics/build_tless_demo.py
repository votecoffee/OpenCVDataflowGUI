"""Build an attributed real RGB-D example and retain hard clutter/occlusion results.

The supplied object mask is an upstream segmentation input for the isolated
case, NOT a prediction by Simple3DLocate. The full-scene benchmark uses no mask.
No test pose is supplied to the locator. Ground truth is evaluation-only.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from opencv_simple_3d_locate import (
    Camera,
    DepthScene,
    Locator,
    Settings,
    TriangleMesh,
    save_recipe,
    train_cad,
)
from opencv_simple_3d_locate.optimization import pose_agreement

FRAMES = (220, 230, 347, 488)
SYMMETRY = np.diag([-1.0, -1.0, 1.0, 1.0])


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def tless_mesh(path):
    """Adapter for the upstream ASCII triangle PLY; not a general PLY importer."""
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Selected T-LESS mesh exceeds 8 MiB.")
    lines = path.read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    header = lines[:end]
    if header[:2] != ["ply", "format ascii 1.0"]:
        raise ValueError("This adapter requires the original ASCII T-LESS CAD mesh.")
    vertices = int(next(x.split()[-1] for x in header if x.startswith("element vertex ")))
    faces = int(next(x.split()[-1] for x in header if x.startswith("element face ")))
    if (
        not 3 <= vertices <= 100000
        or not 1 <= faces <= 200000
        or len(lines) != end + 1 + vertices + faces
    ):
        raise ValueError("Invalid or oversized selected mesh.")
    xyz = np.asarray(
        [[float(v) for v in row.split()[:3]] for row in lines[end + 1 : end + 1 + vertices]]
    )
    indices = []
    for row in lines[end + 1 + vertices :]:
        values = [int(v) for v in row.split()]
        if len(values) != 4 or values[0] != 3 or min(values[1:]) < 0 or max(values[1:]) >= vertices:
            raise ValueError("Expected valid triangle vertex indices.")
        indices.append(values[1:])
    return TriangleMesh(xyz[np.asarray(indices)])


def source_frame(source, frame, *, isolated):
    folder = Path(source) / "test_primesense/000001"
    metadata = json.loads((folder / "scene_camera.json").read_bytes())[str(frame)]
    image = cv2.imread(str(folder / f"rgb/{frame:06}.png"))
    depth = cv2.imread(str(folder / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise ValueError("Missing real RGB/depth input.")
    camera = Camera(np.asarray(metadata["cam_K"]).reshape(3, 3), (image.shape[1], image.shape[0]))
    if isolated:
        mask = cv2.imread(str(folder / f"mask_visib/{frame:06}_000002.png"), 0)
        if mask is None:
            raise ValueError("Missing supplied object segmentation.")
        depth = np.where(mask, depth, 0)
    else:
        # One fixed camera workspace for every view; no test boxes or poses.
        z = depth * metadata["depth_scale"]
        depth = np.where((z >= 300) & (z <= 1100), depth, 0)
    points = camera.backproject(depth, scale=metadata["depth_scale"])
    record = json.loads((folder / "scene_gt.json").read_bytes())[str(frame)][2]
    assert record["obj_id"] == 29
    truth = np.eye(4)
    truth[:3, :3] = np.asarray(record["cam_R_m2c"]).reshape(3, 3)
    truth[:3, 3] = record["cam_t_m2c"]
    return image, points, camera, truth


def full_depth_scene(source, frame, camera):
    """All measured returns from this capture, without masks or workspace clipping."""
    folder = Path(source) / "test_primesense/000001"
    metadata = json.loads((folder / "scene_camera.json").read_bytes())[str(frame)]
    depth = cv2.imread(str(folder / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Missing full-frame depth input.")
    return DepthScene(
        camera.backproject(depth, scale=metadata["depth_scale"]), camera.units, camera.frame
    )


def image_overlay(image, recipe, result, camera, *, alternatives=False):
    from opencv_simple_3d_locate.model import transformed

    image = image.copy()
    if alternatives:
        for match in result.matches:
            for other in (match.pose_alternatives or {}).get("alternatives", [])[:1]:
                xyz = transformed(recipe.samples, np.asarray(other["source_to_target"]))[:, :3]
                xyz = xyz[xyz[:, 2] > 0]
                pixels = xyz @ camera.matrix.T
                pixels = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
                for u, v in pixels:
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        cv2.circle(image, (int(u), int(v)), 2, (35, 170, 255), -1)
    for match in reversed(result.matches):
        aligned = transformed(recipe.samples, match.pose)
        xyz = aligned[:, :3]
        visible = (
            match.visible_model if match.visible_model is not None else np.ones(len(xyz), bool)
        )
        for supported, color in ((False, (30, 30, 245)), (True, (50, 230, 60))):
            selected = visible & (xyz[:, 2] > 0) & (match.inlier_model == supported)
            pixels = xyz[selected] @ camera.matrix.T
            pixels = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
            for u, v in pixels:
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv2.circle(image, (int(u), int(v)), 1, color, -1)
        if match.free_space_model is not None:
            selected = match.free_space_model & (xyz[:, 2] > 0)
            pixels = xyz[selected] @ camera.matrix.T
            pixels = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
            for u, v in pixels:
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv2.circle(image, (int(u), int(v)), 2, (220, 50, 235), -1)
        origin = np.zeros((4, 3))
        origin[1:] = np.eye(3) * recipe.diameter * 0.2
        origin = origin @ match.pose[:3, :3].T + match.pose[:3, 3]
        pixels = origin @ camera.matrix.T
        if np.all(pixels[:, 2] > 0):
            pixels = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
            for axis, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
                cv2.line(image, tuple(pixels[0]), tuple(pixels[axis]), color, 2)
    text = (
        f"{len(result.matches)} poses | green fit / red missing / magenta free | T-LESS CC BY 4.0"
    )
    if alternatives:
        text = "Orange: competing pose (not a detection) | T-LESS CC BY 4.0"
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
    )
    return image


def build(
    source,
    output,
    *,
    host_project=False,
    backend="ppf",
    project_scene="isolated",
    depth_verification="auto",
    separate_depth=False,
    project_frame=230,
    show_alternatives=False,
    workers=1,
    pose_scoring="geometric",
):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = train_cad(
        tless_mesh(source / "models_cad/obj_000029.ply"),
        units="mm",
        frame="tless_object29",
        settings=Settings(
            candidate_backend=backend,
            depth_verification=depth_verification,
            execution_workers=workers,
            pose_scoring=pose_scoring,
        ),
    )
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    pin = save_recipe(assets / "object29.s3l", recipe)
    locator = Locator(recipe)
    records = []
    for isolated in (True, False):
        for frame in FRAMES:
            image, data, camera, truth = source_frame(source, frame, isolated=isolated)
            depth = full_depth_scene(source, frame, camera) if separate_depth else None
            result = locator.locate(
                data, units="mm", frame="camera", camera=camera, depth_scene=depth
            )
            agreement = pose_agreement(
                [m.pose for m in result.matches],
                [truth],
                position_tolerance=5,
                angle_degrees=5,
                symmetries=(SYMMETRY,),
            )
            label = f"{'isolated' if isolated else 'full'}_{frame:06}"
            record = {
                "case": label,
                "segmentation_supplied": isolated,
                "validation": agreement,
                "result": result.report(),
            }
            records.append(record)
            write_json(output / f"{label}.json", record)
            cv2.imwrite(
                str(output / f"{label}.png"),
                image_overlay(image, recipe, result, camera, alternatives=show_alternatives),
            )
            print(label, agreement, flush=True)
            if label == f"{project_scene}_{project_frame:06}":
                np.savez_compressed(
                    assets / "scene.npz", points_normals=data, units="mm", frame="camera"
                )
                write_json(assets / "camera.json", camera.record())
                write_json(assets / "truth.json", truth.tolist())
                if depth is not None:
                    np.savez_compressed(
                        assets / "depth.npz",
                        points_normals=depth.points,
                        units=depth.units,
                        frame=depth.frame,
                    )
                if host_project:
                    _host(output, data, camera, pin, depth=depth)
                    if show_alternatives:
                        from opencv_simple_3d_locate.graphics import overlay

                        cv2.imwrite(
                            str(output / "competing_result.png"),
                            overlay(recipe, result, alternatives=True),
                        )
    write_json(
        output / "benchmark.json",
        {
            "candidate_backend": backend,
            "depth_verification": depth_verification,
            "separate_depth": separate_depth,
            "scope": "Four selected real T-LESS captures, not full-dataset qualification. "
            "Isolated cases use supplied segmentation; full cases do not. "
            "Fit thresholds unchanged. Position 5 mm / angle 5 degrees, discrete symmetry allowed.",
            "cases": records,
        },
    )
    return records


def _host(output, data, camera, pin, *, depth=None):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph
    from opencv_dataflow.pointclouds import PointCloud

    assets = output / "assets"
    with TemporaryDirectory(dir=assets) as scratch:
        path = Path(scratch) / "scene.ply"
        write_cloud(path, PointCloud(data, "mm", "camera"), binary=True)
        path.replace(assets / "scene.ply")
        if depth is not None:
            path = Path(scratch) / "depth.ply"
            write_cloud(path, PointCloud(depth.points, depth.units, depth.frame), binary=True)
            path.replace(assets / "depth.ply")
    graph, registry = Graph(), builtin_registry()
    try:
        scene = graph.add(
            registry.create(
                "input.cloud",
                block_id="scene",
                parameters={"path": "assets/scene.ply", "units": "mm", "frame": "camera"},
            )
        )
        k = camera.matrix
        calibration = graph.add(
            registry.create(
                "3d.camera",
                block_id="camera",
                parameters={
                    "fx": float(k[0, 0]),
                    "fy": float(k[1, 1]),
                    "cx": float(k[0, 2]),
                    "cy": float(k[1, 2]),
                    "width": camera.size[0],
                    "height": camera.size[1],
                    "units": "mm",
                    "frame": "camera",
                },
            )
        )
        locate = graph.add(
            registry.create(
                "3d.simple_locate",
                block_id="locate",
                parameters={"recipe_path": "assets/object29.s3l", "sha256": pin},
            )
        )
        scene.position, calibration.position, locate.position = (0, 0), (0, 200), (380, 0)
        graph.connect(scene.id, "cloud", locate.id, "scene")
        graph.connect(calibration.id, "camera", locate.id, "camera")
        if depth is not None:
            evidence = graph.add(
                registry.create(
                    "input.cloud",
                    block_id="depth_scene",
                    parameters={
                        "path": "assets/depth.ply",
                        "units": depth.units,
                        "frame": depth.frame,
                    },
                )
            )
            evidence.position = (0, 400)
            graph.connect(evidence.id, "cloud", locate.id, "depth_scene")
        locate.preview_output = "fit"
        locate.parameters["render_overlay"] = True  # Secondary documentation image.
        graph.save(output / "project.json")
        graph.run()
        if locate.error:
            raise RuntimeError(locate.error)
        write_json(output / "expected.json", locate.last_result.outputs["report"])
        cv2.imwrite(str(output / "result.png"), locate.last_result.outputs["overlay"])
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Selected upstream T-LESS files.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--host-project", action="store_true")
    parser.add_argument(
        "--backend", choices=("ppf", "ppf_planes", "ppf_features", "ppf_regions"), default="ppf"
    )
    parser.add_argument("--project-scene", choices=("isolated", "full"), default="isolated")
    parser.add_argument("--depth-verification", choices=("auto", "report", "off"), default="auto")
    parser.add_argument("--separate-depth", action="store_true")
    parser.add_argument("--project-frame", choices=FRAMES, type=int, default=230)
    parser.add_argument("--show-alternatives", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pose-scoring", choices=("geometric", "projective"), default="geometric")
    args = parser.parse_args()
    build(
        args.source,
        args.output,
        host_project=args.host_project,
        backend=args.backend,
        project_scene=args.project_scene,
        depth_verification=args.depth_verification,
        separate_depth=args.separate_depth,
        project_frame=args.project_frame,
        show_alternatives=args.show_alternatives,
        workers=args.workers,
        pose_scoring=args.pose_scoring,
    )
