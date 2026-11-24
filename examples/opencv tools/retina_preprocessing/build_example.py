"""Build a lossless replay with known illumination and motion intervals."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (112, 80)
FPS = 20
FRAME_COUNT = 45


def frames():
    width, height = SIZE
    y, x = np.indices((height, width))
    base = (52 + 8 * np.sin(x / 6) + 6 * np.cos(y / 7)).astype(np.uint8)
    base[25:55, 24:45] += 18
    for index in range(FRAME_COUNT):
        image = base.copy()
        if index >= 12:
            image += 45
        if 30 <= index < 38:
            left = 60 + (index - 30) * 3
            image[30:49, left : left + 15] += 63
        yield cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def write_video(path):
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), FPS, SIZE)
    try:
        if not writer.isOpened():
            raise RuntimeError("FFmpeg could not encode the Retina replay")
        for frame in frames():
            writer.write(frame)
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        for index, expected in enumerate(frames()):
            ok, frame = reader.read()
            assert ok and np.array_equal(frame, expected), index
            assert abs(reader.get(cv2.CAP_PROP_POS_MSEC) - index * 1000 / FPS) < 1e-6, index
        assert not reader.read()[0]
    finally:
        reader.release()


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    write_video(assets / "exposure_and_motion.avi")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Exposure step and moving part", 0, -245)
        note.node_size = "large"
        note.description = (
            "Run All. Frame 12 brightens the entire scene. A part then crosses the right "
            "side in frames 30 through 37. The filmstrip shows the Magno motion response. "
            "The source is an original MIT-licensed synthetic replay."
        )
        add(
            "input.video_file",
            "video",
            "Exposure and part replay",
            0,
            90,
            path="assets/exposure_and_motion.avi",
            path_base="project",
            backend="ffmpeg",
            color_format="gray",
        )
        model = add("sequence.retina", "retina", "Adapt contrast and detect motion", 320, 90)
        model.exposed_inputs = ["image"]
        model.exposed_outputs.append("motion_raw")
        graph.connect("video", "image", "retina", "image")
        add("output.filmstrip_image", "filmstrip", "Review motion response", 645, 90)
        graph.connect("retina", "motion", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "retina_preprocessing.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
