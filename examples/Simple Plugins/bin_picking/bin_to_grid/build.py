"""Build a disabled live graph and an optional recorded review through the app API."""

import argparse
from pathlib import Path

from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry


def build(root, *, enabled=False, review=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    registry = builtin_registry()
    register(registry)
    session = ProjectSession(registry, directory=root)
    try:
        if review:
            session.apply(
                [
                    dict(
                        op="add",
                        type="input.sim_pick_review",
                        id="review",
                        parameters={"archive_directory": "assets/run", "cell": 6},
                    ),
                    dict(
                        op="add",
                        type="input.file",
                        id="final_image",
                        parameters={
                            "path": "result.png",
                            "path_base": "project",
                            "filename_ticket": False,
                        },
                    ),
                    dict(
                        op="add",
                        type="input.cloud",
                        id="final_cloud",
                        parameters={"path": "result.ply", "units": "m", "frame": "world"},
                    ),
                    dict(op="arrange"),
                ]
            )
            session.graph.blocks["review"].preview_output = "graphics"
            session.graph.blocks["final_cloud"].preview_output = "cloud"
            name = "review.json"
        else:
            operations = [
                dict(op="add", type=t, id=i, parameters=p)
                for t, i, p in (
                    ("input.sim_bin_grid", "grid", {}),
                    ("flow.unbatch", "cells", {}),
                    ("3d.sim_bin_pick_one", "pick", {"enabled": enabled}),
                    ("flow.batch", "batch", {"size": 6}),
                    ("3d.sim_bin_grid_summary", "summary", {}),
                )
            ]
            for a, b, c, d in (
                ("grid", "group", "cells", "group"),
                ("cells", "value", "pick", "task"),
                ("pick", "outcome", "batch", "item_1"),
                ("batch", "group", "summary", "group"),
            ):
                operations.append(dict(op="connect", source=a, output=b, target=c, input=d))
            operations.append(dict(op="arrange"))
            session.apply(operations)
            session.graph.blocks[
                "grid"
            ].description = "Six destination cells in world metres; app-owned pick order."
            session.graph.blocks["pick"].description = (
                "Fresh first-return depth per pick, MoveIt trajectories, observed assisted "
                "grasp and grid placement. Disabled by default."
            )
            session.graph.blocks["pick"].preview_output = "motion"
            session.graph.blocks["summary"].preview_output = "image"
            name = "live-enabled.local.json" if enabled else "project.json"
        if not session.inspect()["validation"]["valid"]:
            raise RuntimeError("Example failed static validation")
        session.save(root / name)
        return root / name
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    print(build(args.directory, review=args.review))
