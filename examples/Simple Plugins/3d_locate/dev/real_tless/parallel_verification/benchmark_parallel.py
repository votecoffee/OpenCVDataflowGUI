"""Compare serial and bounded parallel verification on retained real T-LESS inputs."""

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


def benchmark(source, output, *, repeats=3):
    source = Path(source)
    recipe = train_cad(
        tless_mesh(source / "models_cad/obj_000029.ply"),
        units="mm",
        frame="tless_object29",
        settings=Settings(execution_workers=1),
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
        profiles=("current", "parallel"),
        repeats=repeats,
        position_tolerance=5,
        angle_tolerance=5,
        progress=lambda message: print(message, flush=True),
    )
    report["dataset"] = json.loads((source / "provenance.json").read_bytes())
    report["scope"] += (
        " Serial and parallel use identical trained samples, candidate backend and fit gates. "
        "Four cases use supplied masks; four use a fixed depth workspace. "
        "All use full-frame measured depth. No input pose seeds. "
        "These retained cases do not qualify production accuracy or cycle time."
    )
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.source, args.output, repeats=args.repeats)
