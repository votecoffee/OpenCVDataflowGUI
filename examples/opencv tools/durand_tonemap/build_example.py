"""Build a reproducible comparison of local and global HDR tone mapping."""

import argparse
from pathlib import Path

import cv2
import numpy as np

def fixture():
    y, x = np.indices((84, 144))
    band = x // 48
    illumination = np.array([1.0, 12.0, 144.0], np.float32)[band]
    texture = 1.0 + np.where(y % 4 < 2, -0.10, 0.10)
    bgr = np.array([[1.10, 1.0, 0.90], [0.97, 1.0, 1.03], [0.93, 1.0, 1.07]])
    return (illumination[..., None] * texture[..., None] * bgr[band]).astype(np.float32)


def build(output):
    from opencv_dataflow.assistant_api import validate_graph
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.core import Graph

    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".tiff", fixture(), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    if not ok:
        raise OSError("Could not encode the HDR fixture")
    (assets / "textured_bands.tiff").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare shadow detail", 0, -240)
        note.node_size = "large"
        note.description = (
            "Run Once and compare both tone-mapped images. Inspect the fine horizontal "
            "texture in the dark band on the left. The filmstrip shows the Durand result. "
            "Original MIT-licensed synthetic float32 radiance."
        )
        add(
            "input.file", "source", "Three HDR bands", 0, 130, path="assets/textured_bands.tiff",
            path_base="project", color_format="unchanged", pixel_depth="unchanged",
        )
        add("cv.tonemap_hdr", "linear", "Global linear mapping", 310, 20, output_depth="float32")
        add("cv.tonemap_durand", "local", "Durand local mapping", 310, 265, output_depth="float32")
        add("output.filmstrip_image", "filmstrip", "Review the local result", 620, 265)
        graph.connect("source", "image", "linear", "image")
        graph.connect("source", "image", "local", "image")
        graph.connect("local", "image", "filmstrip", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "durand_tonemap.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
