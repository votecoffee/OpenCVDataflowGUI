"""Verify the photograph examples run with bundled images from any working directory."""

from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_brick_edges_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(
        EXAMPLES / "opencv tools/brick_edges/brick_edges.json", builtin_registry()
    )
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
        EXAMPLES / "custom tools/coffee_histograms/coffee_histograms.json", builtin_registry()
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


def test_pcb_outline_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(EXAMPLES / "applications/pcb_outline/pcb_outline.json", builtin_registry())
    try:
        for sample in ("0000", "0621"):
            graph.run()
            assert all(block.state == "pass" for block in graph.blocks.values())
            source = graph.blocks["source"].last_result.outputs
            assert source["info"]["filename"] == f"visa-pcb2-normal-{sample}.jpg"
            outline = graph.blocks["outline"].last_result.outputs
            assert outline["count"] == 1
            assert 330_000 < outline["data"]["areas"][0] < 400_000
            overlay = graph.blocks["overlay"].last_result
            assert "Overlaid 1 layers" in overlay.message
            assert np.any(overlay.outputs["image"] != source["image"])
            np.testing.assert_array_equal(overlay.outputs["image"][0, 0], source["image"][0, 0])
        assert graph.blocks["source"].exhausted
    finally:
        graph.close()


def test_pcb_motion_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(EXAMPLES / "flow tools/pcb_motion/pcb_motion.json", builtin_registry())
    try:
        dis = graph.blocks["dis"]
        graph.run()
        assert dis.state == "waiting"
        assert not dis.last_result.outputs["ready"]
        assert not dis.current_emissions["flow"]
        graph.run()
        assert dis.state == "pass", dis.error
        result = dis.last_result.outputs
        assert result["ready"]
        assert result["flow"].vectors.shape == (1070, 1404, 2)
        assert np.isfinite(result["flow"].vectors).all()
        assert result["magnitude"].max() > 1
        assert result["status"]["previous_ticket"]["value"] == "visa-pcb2-normal-0000.jpg"
        assert result["status"]["current_ticket"]["value"] == "visa-pcb2-normal-0621.jpg"
    finally:
        graph.close()
