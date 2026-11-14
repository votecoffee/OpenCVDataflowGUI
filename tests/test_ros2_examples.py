"""Public graphs remain offline by default and usable through editor/API/CLI."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import ros2_blocks as ros
from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/industrial integration/ros2_fastdds"


@pytest.mark.parametrize(
    "name",
    [
        "offline_image_roundtrip",
        "image_subscriber",
        "triggered_image_publisher",
        "joint_state_relay",
        "offline_cloud_roundtrip",
        "cloud_subscriber",
        "triggered_cloud_publisher",
    ],
)
def test_saved_graphs_validate_and_run_without_ros(monkeypatch, name):
    monkeypatch.setattr(
        ros, "ROSClient", lambda *args: pytest.fail("Disabled example contacted ROS")
    )
    graph = Graph.load(BASE / f"{name}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.run()
        assert all(b.state in ("pass", "waiting", "inactive") for b in graph.processing_blocks())
        if name == "offline_image_roundtrip":
            source = graph.blocks["pcb_photo"].last_result.outputs["image"]
            result = graph.blocks["decode_rgb"].last_result.outputs["image"]
            np.testing.assert_array_equal(result[..., ::-1], source)
            assert result.color_format == "rgb"
            assert graph.blocks["edges"].last_result.outputs["image"].any()
            assert "CC BY 4.0" in graph.blocks["source_and_scope"].description
        if name == "offline_cloud_roundtrip":
            original = graph.blocks["project"].last_result.outputs["cloud"]
            restored = graph.blocks["decode_cloud"].last_result.outputs["cloud"]
            np.testing.assert_array_equal(restored.points, original.points)
            np.testing.assert_array_equal(restored.colors, original.colors)
            np.testing.assert_array_equal(restored.pixels, original.pixels)
            assert restored.units == "m" and restored.source_size == original.source_size
            assert 10000 < len(restored.points) < 30000
            assert graph.blocks["decode_cloud"].last_result.outputs["details"]["dropped"] > 0
            assert "CC BY 4.0" in graph.blocks["showcase_source"].description
        for b in graph.processing_blocks():
            if isinstance(b, ros.ReadROS):
                assert b.parameters["enabled"] is False
            if isinstance(b, ros.PublishROS):
                assert b.parameters["write_enabled"] is False
    finally:
        graph.close()


def test_examples_rebuild_with_identical_bytes():
    import runpy

    paths = sorted(BASE.glob("*.json"))
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(ROOT / "examples/assets/build_ros2_examples.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before


def test_ros_blocks_have_editor_dropdowns_and_offline_image_result(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph.load(BASE / "offline_image_roundtrip.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["encode_rgb"].setSelected(True)
        assert isinstance(window.inspector.fields["encoding"], QComboBox)
        assert window.inspector.can_preview()
        assert graph.blocks["decode_rgb"].last_output_images["image"].ndim == 3
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_real_cloud_example_uses_existing_3d_editor_and_optional_controls(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph.load(BASE / "offline_cloud_roundtrip.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["decode_cloud"].setSelected(True)
        assert isinstance(window.inspector.fields["units"], QComboBox)
        assert isinstance(window.inspector.fields["invalid_points"], QComboBox)
        assert window.inspector.can_preview()
        from opencv_dataflow.pointclouds import cloud_outputs

        assert len(cloud_outputs(graph.blocks["decode_cloud"])["cloud"].points) > 10000
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
