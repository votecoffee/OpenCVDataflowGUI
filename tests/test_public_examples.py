"""Verify the public gallery runs with its bundled images from any working directory."""

from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

GALLERY = Path(__file__).resolve().parents[1] / "examples" / "public"


def test_brick_edges_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(GALLERY / "brick_edges" / "brick_edges.json", builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        edges = graph.blocks["edges"].last_output_images["image"]
        assert edges.shape == (512, 512)
        np.testing.assert_array_equal(np.unique(edges), [0, 255])
    finally:
        graph.close()


def test_coffee_histograms_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(
        GALLERY / "coffee_histograms" / "coffee_histograms.json", builtin_registry()
    )
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        for block_id, count in [("color_histogram", 3), ("gray_histogram", 1)]:
            outputs = graph.blocks[block_id].last_result.outputs
            channels = outputs["histogram"]["channels"]
            assert len(channels) == count
            assert all(sum(channel["counts"]) == 400 * 600 for channel in channels)
    finally:
        graph.close()
