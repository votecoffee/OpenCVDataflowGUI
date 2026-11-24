"""Build a lossless moving-part comparison from the empty-conveyor reference fixture."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ASSETS = Path(__file__).resolve().parents[1] / "background_reference/assets"
FPS = 20
FRAME_NAMES = ("clean",) * 22 + ("part_a0", "part_a0", "part_b0", "part_b0", "clean")


def write_video(path):
    images = {
        name: cv2.imread(str(ASSETS / f"{name}.png"), cv2.IMREAD_COLOR) for name in set(FRAME_NAMES)
    }
    if any(image is None for image in images.values()):
        raise RuntimeError("The background reference fixture images are required")
    height, width = images["clean"].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (width, height)
    )
    try:
        if not writer.isOpened():
            raise RuntimeError("FFmpeg could not encode the lossless conveyor clip")
        for name in FRAME_NAMES:
            writer.write(images[name])
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        for index, name in enumerate(FRAME_NAMES):
            ok, image = capture.read()
            assert ok and np.array_equal(image, images[name]), index
            assert abs(capture.get(cv2.CAP_PROP_POS_MSEC) - index * 1000 / FPS) < 1e-6, index
        assert not capture.read()[0]
    finally:
        capture.release()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    write_video(assets / "conveyor.avi")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare three adaptive models", 0, -260)
        note.node_size = "large"
        note.description = (
            "Run All. The original MIT fixture starts with an empty conveyor, shows a part "
            "in two positions, then returns to an empty scene. Compare readiness and masks. "
            "The filmstrip shows GMG's foreground mask."
        )
        add(
            "input.video_file",
            "video",
            "Conveyor replay",
            0,
            100,
            path="assets/conveyor.avi",
            path_base="project",
            backend="ffmpeg",
            color_format="gray",
        )
        for key, name, y, settings in (
            ("mog", "MOG mixtures", 5, {}),
            ("gmg", "GMG color histograms", 243, {"initialization_frames": 12}),
            ("cnt", "CNT stability counts", 481, {"min_stability": 10}),
        ):
            block = add(f"sequence.{key}_background", key, name, 300, y, **settings)
            block.exposed_inputs = ["image"]
            graph.connect("video", "image", key, "image")
        add("output.filmstrip_image", "filmstrip", "Review GMG masks", 615, 243)
        graph.connect("gmg", "mask", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "adaptive_foreground_models.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
