"""Rebuild result views from saved public projects into a fresh output folder."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from opencv_simple_3d_locate import Locator, load_recipe
from opencv_simple_3d_locate.dataflow import depth_input, oriented
from opencv_simple_3d_locate.graphics import fit_cloud, overlay

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph


def build(output=None):
    source = Path(__file__).resolve().parent
    root = source.parent
    target = source if output is None else Path(output)
    if any(target.glob("*_accepted_fit.ply")):
        raise ValueError("Result clouds already exist; use --output with a fresh folder.")
    target.mkdir(parents=True, exist_ok=True)
    cases = [("generated", ""), ("real_positive", "real_tless/depth_verified"),
             ("real_no_match", "novel_models/object_000006")]
    receipt = {}
    for name, relative in cases:
        folder = root / relative
        graph = Graph.load(folder / "project.json", builtin_registry())
        try:
            graph.run()
            block = graph.blocks["locate"]
            if block.state not in ("pass", "fail"):
                raise ValueError(block.error)
            recipe = load_recipe(folder / block.parameters["recipe_path"],
                                 sha256=block.parameters["sha256"])
            inputs = block.last_inputs
            source = inputs["scene"]
            result = Locator(recipe).locate(
                oriented(source), units=source.units, frame=source.frame,
                camera=inputs.get("camera"), depth_scene=depth_input(inputs.get("depth_scene")),
            )
            assert len(result.matches) == block.last_result.outputs["count"]
            cv2.imwrite(str(target / f"{name}.png"), overlay(recipe, result))
            write_cloud(target / f"{name}_accepted_fit.ply", fit_cloud(recipe, result), binary=True)
            report = result.report()
            (target / f"{name}.json").write_bytes(
                (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
            )
            if report["execution"].get("candidate_review"):
                cv2.imwrite(str(target / f"{name}_rejected.png"),
                            overlay(recipe, result, rejected_index=0))
                write_cloud(target / f"{name}_REJECTED_candidate.ply",
                            fit_cloud(recipe, result, rejected_index=0), binary=True)
            paths = [folder / "project.json", folder / block.parameters["recipe_path"]]
            paths.extend(folder / b.parameters["path"] for b in graph.blocks.values()
                         if b.type_id == "input.cloud")
            receipt[name] = {
                "project": (folder / "project.json").relative_to(root).as_posix(),
                "input_sha256": {
                    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths
                },
                "accepted_count": len(result.matches), "assessment": report["assessment"],
                "note": "Rejected candidate display is not an accepted detection.",
            }
            print(name, report["assessment"]["message"])
        finally:
            graph.close()
    (target / "receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Use a fresh folder to preserve existing views")
    build(parser.parse_args().output)
