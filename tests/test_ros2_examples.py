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
        "offline_calibrated_rgbd",
        "registered_rgbd_subscriber",
        "offline_pose_roundtrip",
        "offline_run_identity",
        "exact_time_tf_subscriber",
        "offline_static_camera",
        "static_camera_subscriber",
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
        if name == "offline_static_camera":
            result = graph.blocks["checked"].last_result.outputs["info"]
            assert result["calibration_timing"] == "bound_static"
            assert not result["timestamps_nonzero"]
            assert result["static_calibration"]["calibration_header"]["stamp"] == {
                "sec": 0,
                "nanosec": 0,
            }
            cloud = graph.blocks["project"].last_result.outputs["cloud"]
            assert cloud.units == "m" and 10000 < len(cloud.points) < 30000
            assert "CC BY 4.0" in graph.blocks["showcase_source"].description
            first_ticket = result["ticket"]
            graph.run()
            assert graph.blocks["project"].current_emissions.get("cloud")
            assert graph.blocks["checked"].last_result.outputs["info"]["ticket"] != first_ticket
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
        if name == "offline_calibrated_rgbd":
            validated = graph.blocks["checked"].last_result
            assert validated.outputs["info"]["ticket"]["value"] == "TLESS:scene1:220"
            assert not validated.outputs["info"]["timestamps_nonzero"]
            assert not validated.outputs["info"]["clock_domain_verified"]
            depth = validated.outputs["depth"]
            source = graph.blocks["depth"].last_result.outputs["depth"]
            valid = np.isfinite(source) & (source > 0)
            np.testing.assert_array_equal(depth[valid], source[valid])
            assert np.isnan(depth[~valid]).all()
            cloud = graph.blocks["project"].last_result.outputs["cloud"]
            assert cloud.units == "m" and 10000 < len(cloud.points) < 30000
            assert "CC BY 4.0" in graph.blocks["showcase_source"].description
        if name == "offline_pose_roundtrip":
            original = graph.blocks["pose"].last_result.outputs["transform"]
            restored = graph.blocks["decode"].last_result.outputs["pose"]
            np.testing.assert_allclose(
                restored["source_to_target"], original["source_to_target"], atol=1e-10
            )
            inverse = graph.blocks["inverse"].last_result.outputs["pose"]
            assert inverse["source_frame"] == "base_link" and inverse["target_frame"] == "tool0"
            np.testing.assert_allclose(inverse["translation"], [0.2, 0.1, -0.3], atol=1e-10)
        if name == "offline_run_identity":
            context = graph.blocks["run"].last_result.outputs["context"]
            attached = graph.blocks["attach"].last_result.outputs["acquisition"]
            assert attached["context"] == context
            assert graph.blocks["guard"].last_result.outputs["info"]["current"]
            assert (
                graph.blocks["gray"].last_result.payloads["image"]["acquisition"]["context"]
                == context
            )
            assert "CC BY 4.0" in graph.blocks["source_and_scope"].description
        for b in graph.processing_blocks():
            if isinstance(b, ros.ReadROS):
                assert b.parameters["enabled"] is False
            if isinstance(b, ros.PublishROS):
                assert b.parameters["write_enabled"] is False
            if b.type_id == "industrial.ros2_tf":
                assert b.parameters["enabled"] is False
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


def test_rgbd_block_editor_shows_units_and_timing_controls(app):
    from PySide6.QtWidgets import QCheckBox, QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph.load(BASE / "offline_calibrated_rgbd.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["checked"].setSelected(True)
        assert isinstance(window.inspector.fields["float_depth_units"], QComboBox)
        assert isinstance(window.inspector.fields["allow_zero_stamps"], QCheckBox)
        assert window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_pose_and_run_guard_inspector_enum_choices(app):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph.load(BASE / "offline_pose_roundtrip.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["decode"].setSelected(True)
        assert isinstance(window.inspector.fields["quaternion_policy"], QComboBox)
        assert window.inspector.can_preview()
        graph = Graph.load(BASE / "offline_run_identity.json", window.registry)
        graph.run()
        window.replace_graph(graph)
        window.scene.nodes["guard"].setSelected(True)
        assert isinstance(window.inspector.fields["identity_source"], QComboBox)
        assert window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
