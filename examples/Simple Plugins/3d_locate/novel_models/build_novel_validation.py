"""Train frozen new CAD models and retain positive/negative qualification failures."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .build_tless_demo import image_overlay, tless_mesh, write_json
    from .tless_validation import validation_captures
else:
    from build_tless_demo import image_overlay, tless_mesh, write_json
    from tless_validation import validation_captures

from opencv_simple_3d_locate import (
    Locator,
    Settings,
    compare_profiles,
    profile_recipe,
    save_recipe,
    train_cad,
)


def build(source, output, *, repeats=2, host_project=False):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / "selection.json").read_bytes())
    provenance = json.loads((source / "provenance.json").read_bytes())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
    summaries = []
    validation = output / "validation"
    validation.mkdir(exist_ok=True)
    for item in selection["models"]:
        object_id = item["object_id"]
        folder = output / f"object_{object_id:06}"
        folder.mkdir(exist_ok=True)
        captures = validation_captures(source, item)
        recipe = train_cad(
            tless_mesh(source / f"models_cad/obj_{object_id:06}.ply"),
            units="mm",
            frame=f"tless_object{object_id}",
            settings=Settings(**selection["recipe"]),
        )
        write_json(folder / "training_health.json", recipe.health)
        report = compare_profiles(
            recipe,
            [capture.validation() for capture in captures],
            repeats=repeats,
            profiles=tuple(selection["profiles"]),
            position_tolerance=selection["position_tolerance_mm"],
            angle_tolerance=selection["angle_tolerance_degrees"],
            progress=print,
        )
        report["dataset"] = provenance
        report["selection"] = item
        report["scope"] += (
            " Frozen novel models/views with all measured depth, supplied masks for isolated "
            "cases and target-absent full-scene negatives. No pose seeds or per-model settings. "
            "This development validation exposed a training-budget failure before matching; "
            "it is not an independent production benchmark."
        )
        write_json(folder / "comparison.json", report)
        manifest = {"version": 1, "scenes": []}
        for capture in captures:
            full_name = f"scene{capture.scene_id:06}_frame{capture.frame_id:06}_full.npz"
            full_path = validation / full_name
            if not full_path.exists():
                np.savez_compressed(
                    full_path, points_normals=capture.depth.points, units="mm", frame="camera"
                )
            cloud_name = full_name
            if capture.isolated:
                cloud_name = f"object{object_id:06}_{capture.name}.npz"
                np.savez_compressed(
                    validation / cloud_name,
                    points_normals=capture.points,
                    units="mm",
                    frame="camera",
                )
            manifest["scenes"].append(
                dict(
                    name=capture.name,
                    cloud=f"validation/{cloud_name}",
                    depth_cloud=f"validation/{full_name}",
                    camera=capture.camera.record(),
                    truth=[p.tolist() for p in capture.truth],
                    symmetries=[s.tolist() for s in capture.symmetries],
                )
            )
        write_json(output / f"validation_object{object_id:06}.json", manifest)
        # One fixed saved route for both models, chosen before matching outcomes.
        selected = profile_recipe(recipe, "regions")
        locator = Locator(selected)
        for capture in captures:
            result = locator.locate(
                capture.points,
                units="mm",
                frame="camera",
                camera=capture.camera,
                depth_scene=capture.depth,
            )
            write_json(folder / f"{capture.name}.json", result.report())
            cv2.imwrite(
                str(folder / f"{capture.name}.png"),
                image_overlay(capture.image, selected, result, capture.camera),
            )
        if host_project:
            make_project(folder, selected, captures[0])
        summaries.append(
            dict(
                object_id=object_id,
                health=recipe.health,
                recommended=report["recommended"],
                profiles=[
                    dict(
                        profile=p["profile"],
                        correct=sum(s["runs"][0]["matched"] for s in p["scenes"]),
                        missed=sum(s["runs"][0]["missed"] for s in p["scenes"]),
                        extra=sum(s["runs"][0]["extra"] for s in p["scenes"]),
                    )
                    for p in report["profiles"]
                ],
            )
        )
    write_json(
        output / "summary.json",
        dict(
            models=summaries,
            repeats=repeats,
            positive_scenes=12,
            target_absent_scenes=6,
            expected_instances_per_repeat=12,
            scope="Novel-model development validation; default fit settings and fixed selection. "
            "Not production qualification.",
        ),
    )
    return summaries


def make_project(folder, recipe, capture):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph
    from opencv_dataflow.pointclouds import PointCloud

    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    digest = save_recipe(assets / "model.s3l", recipe)
    write_cloud(assets / "scene.ply", PointCloud(capture.points, "mm", "camera"), binary=True)
    write_cloud(assets / "depth.ply", PointCloud(capture.depth.points, "mm", "camera"), binary=True)
    write_json(assets / "camera.json", capture.camera.record())
    write_json(assets / "truth.json", [p.tolist() for p in capture.truth])
    registry = builtin_registry()
    graph = Graph()
    try:
        scene = graph.add(
            registry.create(
                "input.cloud",
                block_id="scene",
                parameters={"path": "assets/scene.ply", "units": "mm", "frame": "camera"},
            )
        )
        depth = graph.add(
            registry.create(
                "input.cloud",
                block_id="depth",
                parameters={"path": "assets/depth.ply", "units": "mm", "frame": "camera"},
            )
        )
        k = capture.camera.matrix
        camera = graph.add(
            registry.create(
                "3d.camera",
                block_id="camera",
                parameters={
                    "fx": float(k[0, 0]),
                    "fy": float(k[1, 1]),
                    "cx": float(k[0, 2]),
                    "cy": float(k[1, 2]),
                    "width": capture.camera.size[0],
                    "height": capture.camera.size[1],
                    "units": "mm",
                    "frame": "camera",
                },
            )
        )
        locate = graph.add(
            registry.create(
                "3d.simple_locate",
                block_id="locate",
                parameters={"recipe_path": "assets/model.s3l", "sha256": digest},
            )
        )
        for i, block in enumerate((scene, depth, camera)):
            block.position = (0, i * 200)
        locate.position = (380, 0)
        graph.connect("scene", "cloud", "locate", "scene")
        graph.connect("depth", "cloud", "locate", "depth_scene")
        graph.connect("camera", "camera", "locate", "camera")
        locate.preview_output = "fit"
        locate.parameters["render_overlay"] = True  # Secondary documentation image.
        graph.save(folder / "project.json")
        graph.run()
        if locate.error:
            raise RuntimeError(locate.error)
        write_json(folder / "expected.json", locate.last_result.outputs["report"])
        cv2.imwrite(str(folder / "result.png"), locate.last_result.outputs["overlay"])
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--host-project", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, repeats=args.repeats, host_project=args.host_project)
