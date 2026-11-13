"""Compare global and regional PPF without changing training or fit gates."""

import argparse
import json
from pathlib import Path

from build_tless_demo import (
    FRAMES,
    SYMMETRY,
    full_depth_scene,
    source_frame,
    tless_mesh,
    write_json,
)
from opencv_simple_3d_locate import Settings, ValidationScene, compare_profiles, train_cad


def benchmark(source, output, *, repeats=3, workers=1, pose_scoring="projective"):
    source = Path(source)
    recipe = train_cad(
        tless_mesh(source / "models_cad/obj_000029.ply"),
        units="mm",
        frame="tless_object29",
        settings=Settings(execution_workers=workers, pose_scoring=pose_scoring),
    )
    scenes = []
    for isolated in (True, False):
        for frame in FRAMES:
            _, data, camera, truth = source_frame(source, frame, isolated=isolated)
            scenes.append(
                ValidationScene(
                    f"{'isolated' if isolated else 'full'}_{frame:06}",
                    data,
                    "mm",
                    "camera",
                    (truth,),
                    camera,
                    (SYMMETRY,),
                    full_depth_scene(source, frame, camera),
                )
            )
    report = compare_profiles(
        recipe,
        scenes,
        profiles=("current", "regions"),
        repeats=repeats,
        position_tolerance=5,
        angle_tolerance=5,
        progress=print,
    )
    report["dataset"] = json.loads((source / "provenance.json").read_bytes())
    report["scope"] += (
        " Real T-LESS development subset, four supplied-mask and four fixed-workspace inputs. "
        "Both routes use the same scoring, geometric/depth gates, total verification budget "
        "and full-frame measured depth. Regions add bounded proposal-only native searches. "
        "No truth, test boxes or input pose seeds enter search."
    )
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pose-scoring", choices=("geometric", "projective"), default="projective")
    args = parser.parse_args()
    benchmark(
        args.source,
        args.output,
        repeats=args.repeats,
        workers=args.workers,
        pose_scoring=args.pose_scoring,
    )
