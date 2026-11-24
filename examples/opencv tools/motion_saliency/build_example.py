"""Build a timestamped replay with a known moving part and stationary intervals."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (128, 96)
FPS = 20
START = 72
STOP = 87
FRAME_COUNT = 100


def frames():
    width, height = SIZE
    yy, xx = np.indices((height, width))
    background = (60 + 6 * np.sin(xx / 11) + 5 * np.cos(yy / 14)).astype(np.uint8)
    for index in range(FRAME_COUNT):
        image = background.copy()
        if START <= index < STOP:
            left = 20 + (index - START) * 4
            image[35:57, left : left + 22] = 220
        yield cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / "moving_part.avi"
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), FPS, SIZE)
    try:
        if not writer.isOpened():
            raise RuntimeError("The FFmpeg backend could not encode the lossless fixture")
        for image in frames():
            writer.write(image)
    finally:
        writer.release()

    reader = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        for index, expected in enumerate(frames()):
            ok, image = reader.read()
            assert ok and np.array_equal(image, expected), index
            assert abs(reader.get(cv2.CAP_PROP_POS_MSEC) - index * 1000 / FPS) < 1e-6, index
        assert not reader.read()[0]
    finally:
        reader.release()

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Compare two motion detectors on a timed clip", 0, -225
        )
        note.node_size = "large"
        note.description = (
            "Run All. The original MIT clip starts with a stationary scene, then the bright "
            "part moves across the frame. Inspect the masks and readiness from both branches. "
            "The filmstrip shows the Bin Wang mask."
        )
        add(
            "input.video_file",
            "video",
            "Stationary scene and moving part",
            0,
            65,
            path="assets/moving_part.avi",
            path_base="project",
            backend="ffmpeg",
        )
        for key, name, method, y in (
            ("wang", "Bin Wang motion mask", "bin wang", 30),
            ("retina", "Retina transient areas", "retina transient", 260),
        ):
            tool = add("sequence.motion_saliency", key, name, 295, y, method=method)
            tool.exposed_inputs = ["image"]
            graph.connect("video", "image", key, "image")
        add("output.filmstrip_image", "filmstrip", "Review Bin Wang motion", 605, 45)
        graph.connect("wang", "mask", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "motion_saliency.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
