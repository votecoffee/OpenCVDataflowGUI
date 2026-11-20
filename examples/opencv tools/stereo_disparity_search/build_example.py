"""Save a hardware-free comparison of manual and depth-bounded disparity search."""

import argparse
from pathlib import Path

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def build(output):
    if output.exists():
        raise FileExistsError("Choose a new output path to preserve the example")
    graph, registry = Graph(), builtin_registry()

    def add(type_id, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=name, parameters=parameters))
        block.position = [x, y]
        return block

    try:
        comment = add("layout.comment", "notes", 0, -215)
        comment.name = "Stereo disparity search and support"
        comment.node_size = "large"
        comment.description = (
            "Original synthetic stereo pair with 8 px disparity and 3 m depth. "
            "Both branches have the same image support. A mask marks an unavailable "
            "area on the right. Compare the support masks and search reports."
        )
        pair = add("input.stereo_demo", "pair", 0, 0, disparity=8)
        rectify = add("3d.stereo_rectify", "rectify", 285, 0)
        left = add("region.shape", "left_selection", 585, 0, rectangle=[12, 8, 290, 168])
        left.name = "Left inspection support"
        right = add(
            "region.shape",
            "right_selection",
            585,
            215,
            rectangle=[135, 64, 8, 45],
            invert=True,
        )
        right.name = "Right occlusion exclusion"
        manual = add("3d.disparity", "manual", 925, 25, num_disparities=64)
        manual.name = "Manual search over 64 disparities"
        depth = add(
            "3d.disparity",
            "depth",
            925,
            405,
            search_mode="depth",
            minimum_depth=2.4,
            maximum_depth=4.4,
        )
        depth.name = "Depth-bounded search"
        cloud = add("3d.reproject", "cloud", 1340, 405, sample_step=4, max_depth=5.0)
        for name in ("left", "right", "rig"):
            graph.connect(pair.id, name, rectify.id, name)
        graph.connect(rectify.id, "left", left.id, "image")
        graph.connect(rectify.id, "right", right.id, "image")
        for block in (manual, depth):
            for name in ("left", "right", "rectification"):
                graph.connect(rectify.id, name, block.id, name)
            graph.connect(left.id, "mask", block.id, "left_mask")
            graph.connect(right.id, "mask", block.id, "right_mask")
        graph.connect(depth.id, "disparity", cloud.id, "disparity")
        graph.connect(rectify.id, "rectification", cloud.id, "rectification")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("stereo_disparity_search.json")
    )
    build(parser.parse_args().output)
