"""Compare PPF and the optional plane/feature hybrids on all retained real T-LESS cases."""

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


def benchmark(source, output, *, repeats=1, depth_verification="auto", separate_depth=False):
    source, output = Path(source), Path(output)
    recipe = train_cad(
        tless_mesh(source / "models_cad/obj_000029.ply"),
        units="mm",
        frame="tless_object29",
        settings=Settings(depth_verification=depth_verification),
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
                    full_depth_scene(source, frame, camera) if separate_depth else None,
                )
            )
    report = compare_profiles(
        recipe,
        scenes,
        profiles=("current", "planes", "features"),
        repeats=repeats,
        position_tolerance=5,
        angle_tolerance=5,
        progress=print,
    )
    report["dataset"] = json.loads((source / "provenance.json").read_bytes())
    report["depth_verification"] = depth_verification
    report["separate_depth"] = separate_depth
    report["scope"] += (
        " Real T-LESS development subset. Isolated cases use supplied masks; "
        "full cases use a fixed depth workspace only. No input pose seeds."
    )
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--depth-verification", choices=("auto", "report", "off"), default="auto")
    parser.add_argument("--separate-depth", action="store_true")
    args = parser.parse_args()
    benchmark(
        args.source,
        args.output,
        repeats=args.repeats,
        depth_verification=args.depth_verification,
        separate_depth=args.separate_depth,
    )
