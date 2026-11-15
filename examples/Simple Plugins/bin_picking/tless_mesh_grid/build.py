"""Build a disabled mesh-picking example from the existing serial grid workflow."""

import json
from pathlib import Path

from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry


def build(root):
    root = Path(root)
    registry = builtin_registry()
    register(registry)
    with ProjectSession.open(root.parent / "bin_to_grid/project.json", registry) as session:
        session.apply(
            [
                dict(op="update", id="grid", fields={"parameters": {"part_center_z_m": 0.223}}),
                dict(
                    op="update",
                    id="pick",
                    fields={
                        "parameters": {
                            "enabled": False,
                            "part_height_m": 0.046,
                            "top_template_path": "assets/top-template.json",
                            "top_template_sha256": json.loads(
                                (root / "provenance.json").read_bytes()
                            )["top_template_sha256"],
                        }
                    },
                ),
            ]
        )
        if not session.inspect()["validation"]["valid"]:
            raise RuntimeError("Dataset-backed example is invalid")
        session.save(root / "project.json")


def review(root):
    root = Path(root)
    registry = builtin_registry()
    register(registry)
    with ProjectSession(registry, directory=root) as session:
        operations = [
            dict(
                op="add",
                type="input.sim_pick_review",
                id="review",
                parameters={"archive_directory": "assets/run", "cell": 6},
            )
        ]
        for identifier, name in (
            ("final_camera", "final-camera.png"),
            ("summary", "summary.png"),
            ("original_ground_truth", "source-review.png"),
        ):
            operations.append(
                dict(
                    op="add",
                    type="input.file",
                    id=identifier,
                    parameters={"path": name, "path_base": "project", "filename_ticket": False},
                )
            )
        operations.append(
            dict(
                op="add",
                type="input.cloud",
                id="source_ground_truth_cloud",
                parameters={"path": "source-cloud.ply", "units": "m", "frame": "tless_camera"},
            )
        )
        operations.append(dict(op="arrange"))
        session.apply(operations)
        session.graph.blocks["review"].preview_output = "motion"
        session.graph.blocks["source_ground_truth_cloud"].preview_output = "cloud"
        assert session.inspect()["validation"]["valid"]
        session.save(root / "review.json")


if __name__ == "__main__":
    root = Path(__file__).parent
    if not (root / "project.json").exists():
        build(root)
    if not (root / "review.json").exists():
        review(root)
