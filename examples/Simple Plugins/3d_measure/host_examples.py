"""Portable native-cloud project/export helper; requires OpenCVDataflowGUI."""

import hashlib
import json
from pathlib import Path

import cv2
from opencv_simple_3d_measure import save_recipe
from opencv_simple_3d_measure.dataflow import register

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def project(folder, recipe, scene, *, scene_frame=None, fixture=None, description=""):
    folder = Path(folder)
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    digest = save_recipe(assets / "measurement.s3m", recipe)
    write_cloud(assets / "scene.ply", PointCloud(scene, recipe.units, scene_frame or recipe.frame),
                binary=True)
    registry = builtin_registry()
    register(registry)
    graph = Graph()
    graph.bind_project_dir(folder)
    try:
        cloud = graph.add(registry.create("input.cloud", block_id="scene", parameters={
            "path": "assets/scene.ply", "units": recipe.units,
            "frame": scene_frame or recipe.frame, "max_points": 2000000}))
        cloud.position = [0, 0]
        measure = graph.add(registry.create("3d.simple_measure", block_id="measure", parameters={
            "recipe_path": "assets/measurement.s3m", "sha256": digest}))
        measure.position = [400, 0]
        measure.description = description
        measure.preview_output = "fit"
        graph.connect(cloud.id, "cloud", measure.id, "scene")
        if fixture is not None:
            pose = graph.add(registry.create("3d.pose", block_id="fixture", parameters={
                "rvec": cv2.Rodrigues(fixture[:3, :3])[0].ravel().tolist(),
                "translation": fixture[:3, 3].tolist(), "source_frame": recipe.frame,
                "target_frame": scene_frame, "units": recipe.units}))
            pose.name = "Supplied annotation fixture (not a detected pose)"
            pose.position = [0, 250]
            graph.connect(pose.id, "transform", measure.id, "fixture")
        graph.save(folder / "project.json")
        graph.run()
        if any(block.state == "error" for block in graph.blocks.values()):
            raise RuntimeError({b.id: b.error for b in graph.blocks.values()})
        report = measure.last_result.outputs["report"]
        write_json(folder / "expected.json", report)
        write_cloud(folder / "result.ply", measure.last_result.outputs["fit"], binary=True)
        return report
    finally:
        graph.close()


def manifest(folder):
    root = Path(folder)
    write_json(root / "sha256.json", {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256.json"
        and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")})
