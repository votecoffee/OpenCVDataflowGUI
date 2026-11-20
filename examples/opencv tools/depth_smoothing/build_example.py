"""Build the original synthetic depth-smoothing project and its packed input."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 120, 80


def packed_fixture():
    rng = np.random.default_rng(2701)
    truth = np.full((HEIGHT, WIDTH), 1000, np.uint16)
    truth[:, WIDTH // 2 :] = 2000
    depth = np.rint(truth + rng.normal(0, 21, truth.shape)).astype(np.uint16)
    confidence = np.full_like(depth, 65535)
    depth[14:23, 16:24] = 0
    depth[46:52, 58:65] = 0
    depth[30:35, 74:84] = 1000  # False foreground on the distant plane.
    confidence[30:35, 74:84] = 6554
    confidence[54:62, 29:42] = 32768
    return np.dstack((depth, confidence, truth))


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "depth_smoothing.json").exists():
        raise FileExistsError("Choose an empty destination. Existing fixtures are preserved.")
    assets.mkdir(parents=True)
    ok, image = cv2.imencode(".png", packed_fixture())
    if not ok:
        raise ValueError("Could not encode the synthetic fixture")
    (assets / "depth_planes.png").write_bytes(image.tobytes())
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        add(
            "input.file",
            "source",
            [0, 0],
            path="assets/depth_planes.png",
            path_base="project",
            color_format="bgr",
            pixel_depth="uint16",
        ).name = "Packed depth, confidence and truth"
        add("cv.split_channels", "split", [330, 0]).name = "Depth / confidence / truth planes"
        add("3d.smooth_depth", "smooth", [660, 0])
        add(
            "3d.camera",
            "camera",
            [660, 310],
            fx=150.0,
            fy=150.0,
            cx=59.5,
            cy=39.5,
            width=WIDTH,
            height=HEIGHT,
        )
        add("3d.depth_cloud", "cloud", [995, 0], sample_step=1)
        comment = add("layout.comment", "credit", [0, -310])
        comment.name, comment.node_size = "Synthetic noisy depth step", "large"
        comment.description = (
            "Original fixture by OpenCVDataflowGUI contributors (MIT). Near Z = 1 m, "
            "far Z = 2 m. Channel 0 is noisy depth in mm, channel 1 is scaled confidence, "
            "channel 2 is the unchanged ground truth. Zero depth marks holes."
        )
        graph.connect("source", "image", "split", "image")
        graph.connect("split", "channel_0", "smooth", "depth")
        graph.connect("split", "channel_1", "smooth", "confidence")
        graph.connect("smooth", "depth", "cloud", "depth")
        graph.connect("camera", "camera", "cloud", "camera")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / "depth_smoothing.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
