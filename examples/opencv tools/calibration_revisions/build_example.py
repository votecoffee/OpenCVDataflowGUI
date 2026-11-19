"""Create deterministic, hardware-free calibration revision example projects."""

import argparse
import json
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import packet_for

HERE = Path(__file__).resolve().parent
STORE = "assets/revisions"
MAP = "assets/left-r2.npz"


def assets(registry, output):
    for version, baseline in (("r1", 0.10), ("r2", 0.12)):
        source = registry.create("input.stereo_demo", parameters={"baseline_m": baseline})
        result = source.execute({})
        assert result and result.passed, source.error
        for profile_id, profile in (
            ("stereo", result.outputs["rig"]),
            ("left-camera", {**result.outputs["camera"], "rms_pixels": 0.0}),
        ):
            writer = registry.create(
                "output.calibration",
                parameters={
                    "path": str(output / STORE),
                    "profile_id": profile_id,
                    "revision": version,
                },
            )
            saved = writer.execute({"profile": profile})
            assert saved and saved.passed, writer.error
    loader = registry.create(
        "input.calibration",
        parameters={"path": str(output / STORE), "profile_id": "left-camera", "revision": "r2"},
    )
    result = loader.execute({})
    assert result and result.passed, loader.error
    builder = registry.create("calibration.lens_maps", parameters={"width": 240, "height": 135})
    maps = builder.execute({"calibration": result.outputs["profile"]})
    assert maps and maps.passed, builder.error
    writer = registry.create("output.lens_maps", parameters={"path": str(output / MAP)})
    saved = writer.execute({key: packet_for(maps, key) for key in ("map_x", "map_y")})
    assert saved and saved.passed, writer.error


def build_project(registry, path, mono=False, revision="r2"):
    graph = Graph()

    def add(type_id, key, xy, **settings):
        block = graph.add(registry.create(type_id, block_id=key, parameters=settings))
        block.position = xy
        return block

    def stamp(side, name, xy):
        ref = json.loads((path.parent / STORE / name / "r2.json").read_bytes())
        add(
            "calibration.declare_capture",
            side,
            xy,
            camera_frame=f"{side}_camera",
            profile_id=name,
            revision="r2",
            sha256=ref["sha256"],
        )
        graph.connect("scene", side, side, "image")

    add("input.stereo_demo", "scene", [0, 0])
    if mono:
        stamp("left", "left-camera", [330, 0])
        add("input.lens_maps", "maps", [330, 300], path=MAP, path_base="project")
        add("cv.remap", "corrected", [680, 0])
        graph.connect("left", "image", "corrected", "image")
        for name in ("map_x", "map_y"):
            graph.connect("maps", name, "corrected", name)
    else:
        stamp("left", "stereo", [330, 0])
        stamp("right", "stereo", [330, 300])
        add(
            "input.calibration",
            "profile",
            [330, 635],
            path=STORE,
            path_base="project",
            profile_id="stereo",
            revision=revision,
        )
        add("3d.stereo_rectify", "rectify", [680, 0])
        add("3d.disparity", "disparity", [1030, 0], num_disparities=32)
        add("3d.reproject", "cloud", [1380, 0])
        for side in ("left", "right"):
            graph.connect(side, "image", "rectify", side)
            graph.connect("rectify", side, "disparity", side)
        graph.connect("profile", "profile", "rectify", "rig")
        graph.connect("rectify", "rectification", "cloud", "rectification")
        graph.connect("rectify", "left", "cloud", "image")
        graph.connect("disparity", "disparity", "cloud", "disparity")
    note = add("layout.comment", "credits", [0, -300])
    note.name, note.node_size = "Calibration revision example", "large"
    note.description = (
        "Original synthetic stereo texture and known calibration by OpenCVDataflowGUI contributors"
        " (MIT). The scene was tagged independently as revision r2. The r2 stereo rig has a"
        " baseline of 0.12 m, and the older r1 rig has a baseline of 0.10 m."
    )
    try:
        verdict = validate_graph(graph)
        assert verdict["valid"], verdict["issues"]
        graph.save(path)
    finally:
        graph.close()


def build(output):
    if (output / "assets").exists() or any(output.glob("*.json")):
        raise FileExistsError("Choose an empty destination. Existing example files are preserved.")
    registry = builtin_registry()
    assets(registry, output)
    build_project(registry, output / "stereo_r2.json")
    build_project(registry, output / "stereo_mismatch.json", revision="r1")
    build_project(registry, output / "camera_maps.json", mono=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=HERE)
    build(cli.parse_args().output.resolve())
