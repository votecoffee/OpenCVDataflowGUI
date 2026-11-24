"""Build a timestamped silhouette replay with known rightward and downward motion."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (128, 96)
FPS = 20
FRAMES = 44


def frames():
    width, height = SIZE
    yy, xx = np.indices((height, width))
    background = (52 + 7 * np.sin(xx / 14) + 6 * np.cos(yy / 18)).astype(np.uint8)
    for index in range(FRAMES):
        image = background.copy()
        if 2 <= index <= 12:
            left = 18 + 3 * (index - 2)
            image[18:34, left : left + 15] = 220
        if 22 <= index <= 32:
            top = 18 + 3 * (index - 22)
            image[top : top + 16, 74:91] = 220
        yield cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    media = assets / "two_directions.avi"
    writer = cv2.VideoWriter(str(media), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), FPS, SIZE)
    try:
        if not writer.isOpened():
            raise RuntimeError("The FFmpeg backend could not encode the lossless fixture")
        for image in frames():
            writer.write(image)
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(media), cv2.CAP_FFMPEG)
    try:
        for index, expected in enumerate(frames()):
            ok, image = reader.read()
            assert ok and np.array_equal(image, expected), index
            assert abs(reader.get(cv2.CAP_PROP_POS_MSEC) - index * 1000 / FPS) < 1e-6, index
        assert not reader.read()[0]
    finally:
        reader.release()

    graph, registry = Graph(), builtin_registry()

    def add(kind, id, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=id, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Read motion direction from a fading trail", 0, -195)
        note.node_size = "large"
        note.description = (
            "Run All. The original MIT clip shows a bright part moving right, then down. "
            "The filmstrip displays 250 ms of motion history. Bright pixels moved most "
            "recently. Review the history block for direction, age and region labels."
        )
        add(
            "input.video_file",
            "video",
            "Two-direction test clip",
            0,
            45,
            path="assets/two_directions.avi",
            path_base="project",
            backend="ffmpeg",
            color_format="gray",
        )
        add("cv.threshold", "threshold", "Make the moving part silhouette", 260, 45, value=150)
        history = add("sequence.motion_history", "history", "Measure motion history", 550, 45)
        history.exposed_inputs = ["image", "timestamp_ms"]
        add("output.filmstrip_image", "filmstrip", "Review the fading trail", 850, 45)
        graph.connect("video", "image", "threshold", "image")
        graph.connect("threshold", "image", "history", "image")
        graph.connect("history", "recency", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "motion_history.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
