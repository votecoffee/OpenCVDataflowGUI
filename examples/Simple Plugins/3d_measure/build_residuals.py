"""Residual views of unchanged generated controls and licensed real T-LESS projects."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
from host_examples import manifest, write_json
from opencv_simple_3d_measure import Measurer, ResidualReview, load_recipe
from opencv_simple_3d_measure.dataflow import fixture_matrix, register
from opencv_simple_3d_measure.persistence import strict_json
from opencv_simple_3d_measure.qt_residuals import ResidualHistogram
from PySide6.QtWidgets import QApplication

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import render_cloud

CASES = {"good": "generated/good", "raised": "generated/raised_feature",
         "missing": "generated/missing_surface", "line_good": "lines/generated/good",
         "line_nonparallel": "lines/generated/nonparallel",
         "tless_frame13": "real_tless/frame000013", "tless_frame24": "real_tless/frame000024"}


def build(source, output):
    source, output = Path(source), Path(output)
    hashes = strict_json((source / "sha256.json").read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    application = QApplication.instance() or QApplication([])
    registry = builtin_registry()
    register(registry)
    for name, relative in CASES.items():
        base, target = source / relative, output / name
        for path in [base / "project.json", base / "expected.json", *(base / "assets").iterdir()]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != hashes[path.relative_to(source).as_posix()]:
                raise ValueError(f"Pinned example mismatch: {path.name}")
        shutil.copytree(base / "assets", target / "assets")
        definition = strict_json((base / "project.json").read_bytes())
        block = next(b for b in definition["blocks"] if b["id"] == "measure")
        block["parameters"]["residual_graphics"] = True
        block["exposed_outputs"] += ["residuals", "residual_report"]
        write_json(target / "project.json", definition)
        graph = Graph.load(target / "project.json", registry)
        try:
            block = graph.blocks["measure"]
            block.preview_output = "residuals"
            graph.save(target / "project.json")
            graph.run()
            if block.state == "error":
                raise RuntimeError(block.error)
            report = block.last_result.outputs["report"]
            expected = strict_json((base / "expected.json").read_bytes())
            if strict_json(json.dumps(report, allow_nan=False)) != expected:
                raise ValueError("Residual review changed the numerical report")
            write_json(target / "expected.json", report)
            write_json(target / "residual_report.json",
                       block.last_result.outputs["residual_report"])
            cloud = block.last_result.outputs["residuals"]
            write_cloud(target / "residuals.ply", cloud, binary=True)
            cv2.imwrite(str(target / "residuals.png"), render_cloud(
                cloud, width=900, height=650, yaw=25, pitch=-55, point_size=2))
            recipe = load_recipe(target / block.parameters["recipe_path"])
            scene = graph.blocks["scene"].last_result.outputs["cloud"]
            fixture = fixture_matrix(block.last_inputs.get("fixture"), recipe, scene)
            result = Measurer(recipe).measure(scene.points, units=scene.units,
                                               frame=scene.frame, fixture=fixture)
            review = ResidualReview(result)
            review.export(target / "residuals.npz")
            chart = ResidualHistogram()
            for i, (region, row) in enumerate(review.report()["regions"].items()):
                chart.set_region(row, region, recipe.units)
                chart.save_image(target / f"histogram_{i}.png")
            chart.deleteLater()
            write_json(target / "source_identity.json", {"base_project": relative,
                "base_project_sha256": hashes[relative + "/project.json"],
                "recipe_sha256": block.parameters["sha256"], "numerical_report": "unchanged"})
            print(name, block.state, list(review.report()["regions"]))
        finally:
            graph.close()
            application.processEvents()
    manifest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
