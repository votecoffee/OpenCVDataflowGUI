"""Execution-coherent plugin input captures without a Qt dependency."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, metadata
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock


class Acquisition(Block):
    outputs = (
        Port("scene", "cloud"),
        Port("depth_scene", "cloud"),
        Port("camera", "data"),
        Port("image"),
        Port("depth", "depth"),
    )
    defaults = {"generation": 1, "passed": True}

    def process(self, inputs):
        generation = self.parameters["generation"]
        cloud = PointCloud(
            [[generation, 2, 3], [4, 5, 6]],
            units="mm",
            frame="camera",
            colors=[[1, 2, 3], [4, 5, 6]],
            normals=[[0, 0, 1], [0, 1, 0]],
            pixels=[[0, 0], [1, 1]],
            source_size=(3, 2),
            metadata={"acquisition": {"generation": generation}},
        )
        values = {
            "scene": cloud,
            "depth_scene": cloud,
            "camera": {"matrix": [[100, 0, 1], [0, 100, 1], [0, 0, 1]]},
            "image": ImageArray(np.full((2, 3, 3), generation, np.uint16), "rgb"),
            "depth": DepthArray(np.full((2, 3), generation), units="mm", display_max=100),
        }
        ticket = TicketClock("plugin-snapshot-test").issue("camera", "external", generation)
        payload = {"ticket": ticket, "capture": {"generation": generation}}
        return Result(
            values,
            passed=self.parameters["passed"],
            payloads={name: payload for name in values},
        )


class Inspection(Block):
    inputs = (*Acquisition.outputs, Port("optional", "data", required=False))
    defaults = {"passed": True}

    def process(self, inputs):
        return Result(passed=self.parameters["passed"])


@pytest.fixture
def capture_graph():
    graph = Graph()
    source = graph.add(Acquisition())
    target = graph.add(Inspection())
    for port in source.outputs:
        graph.connect(source.id, port.name, target.id, port.name)
    context = PluginDialogContext(graph, target)
    graph.run()
    assert source.state == target.state == "pass"
    try:
        yield graph, source, target, context
    finally:
        context._close()
        graph.close()


def test_all_processed_ports_are_copied_together_with_typed_metadata(capture_graph):
    _, _, target, context = capture_graph
    captured = context.snapshot_processed_inputs()
    assert set(captured.inputs) == {port.name for port in Acquisition.outputs}
    assert captured.block_id == target.id and captured.execution_count == 1
    assert captured.output_revision == 0 and captured.passed
    assert len(captured.snapshot_id) == 32
    for packet in captured.inputs.values():
        assert isinstance(packet, ValuePacket)
        assert packet.payload["ticket"]["value"] == 1
        assert packet.payload["capture"] == {"generation": 1}
    scene = captured.inputs["scene"].value
    assert scene.units == "mm" and scene.frame == "camera" and scene.source_size == (3, 2)
    scene.points[:] = 90
    scene.colors[:] = 90
    scene.normals[:] = 0
    scene.pixels[:] = 0
    scene.metadata["acquisition"]["generation"] = 90
    captured.inputs["camera"].value["matrix"][0][0] = 90
    captured.inputs["image"].value[:] = 90
    captured.inputs["depth"].value[:] = 90
    captured.inputs["depth"].value.units = "m"
    captured.inputs["image"].value.metadata["capture"]["generation"] = 90
    captured.inputs["image"].payload["capture"]["generation"] = 90
    fresh = context.snapshot_processed_inputs()
    assert fresh.snapshot_id != captured.snapshot_id
    np.testing.assert_array_equal(fresh.inputs["scene"].value.points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(fresh.inputs["scene"].value.colors, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(fresh.inputs["scene"].value.normals, [[0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(fresh.inputs["scene"].value.pixels, [[0, 0], [1, 1]])
    assert fresh.inputs["scene"].value.metadata["acquisition"]["generation"] == 1
    assert fresh.inputs["camera"].value["matrix"][0][0] == 100
    image = fresh.inputs["image"].value
    assert image.color_format == "rgb" and image.dtype == np.uint16 and np.all(image == 1)
    assert metadata(image) == fresh.inputs["image"].payload
    assert fresh.inputs["depth"].value.units == "mm"
    assert fresh.inputs["depth"].value.display_max == 100
    assert np.all(fresh.inputs["depth"].value == 1)
    assert target.input_payloads["image"]["capture"]["generation"] == 1


def test_newer_upstream_outputs_do_not_change_the_processed_bundle(capture_graph):
    graph, source, target, context = capture_graph
    first = context.snapshot_processed_inputs(("scene", "depth_scene", "camera"))
    source.parameters["generation"] = 2
    source.execute({})
    context._outputs_published()
    context.refresh_images()
    assert context.snapshot_input_cloud("scene").value.points[0, 0] == 2
    retained = context.snapshot_processed_inputs(("scene", "depth_scene", "camera"))
    assert retained.output_revision == 1 and retained.execution_count == first.execution_count
    assert all(packet.payload["capture"]["generation"] == 1 for packet in retained.inputs.values())
    graph.run()
    newer = context.snapshot_processed_inputs()
    assert newer.execution_count == 2 and target.timing.count == 2
    assert all(packet.payload["capture"]["generation"] == 2 for packet in newer.inputs.values())
    assert first.inputs["scene"].value.points[0, 0] == 1


def test_rejected_inputs_keep_the_last_completed_bundle(capture_graph):
    graph, source, target, context = capture_graph
    target.input_failure_policies = {"scene": "reject_failed"}
    source.parameters.update(generation=2, passed=False)
    graph.run()
    assert target.input_rejected and target.state == "inactive"
    captured = context.snapshot_processed_inputs()
    assert captured.execution_count == 1 and captured.passed
    assert captured.inputs["scene"].payload["ticket"]["value"] == 1


def test_failed_inspection_is_still_a_completed_capture(capture_graph):
    graph, _, target, context = capture_graph
    target.parameters["passed"] = False
    graph.run()
    captured = context.snapshot_processed_inputs()
    assert target.state == "fail" and not captured.passed and captured.execution_count == 2


@pytest.mark.parametrize(
    ("ports", "error", "message"),
    [
        ("scene", TypeError, "collection"),
        (("scene", "scene"), ValueError, "unique"),
        (("scene", "missing"), ValueError, "declared"),
        (("scene", 1), ValueError, "declared"),
        (("scene", "optional"), ValueError, "retained for: optional"),
    ],
)
def test_invalid_port_sets_fail_without_partial_capture_or_held_lock(
    capture_graph, ports, error, message
):
    _, _, _, context = capture_graph
    with pytest.raises(error, match=message):
        context.snapshot_processed_inputs(ports)
    assert "scene" in context.snapshot_processed_inputs(["scene"]).inputs


@pytest.mark.parametrize("run_profile", ("balanced", "lean"))
def test_profiles_without_input_history_require_an_explicit_new_debug_run(
    capture_graph, run_profile
):
    graph, _, _, context = capture_graph
    graph.set_run_profile(run_profile)
    graph.run()
    with pytest.raises(ValueError, match="not retained.*Debug"):
        context.snapshot_processed_inputs()
    graph.set_run_profile("debug")
    with pytest.raises(ValueError, match="unavailable.*Debug"):
        context.snapshot_processed_inputs()
    graph.run()
    assert len(context.snapshot_processed_inputs().inputs) == 5


def test_no_completed_result_and_exceptions_do_not_capture_unfinished_inputs():
    class Broken(Block):
        inputs = (Port("value", "data"),)

        def process(self, inputs):
            raise ValueError("Synthetic processing failure")

    graph = Graph()
    target = graph.add(Broken())
    context = PluginDialogContext(graph, target)
    try:
        with pytest.raises(ValueError, match="No completed"):
            context.snapshot_processed_inputs()
        target.execute({"value": 5})
        assert target.state == "error"
        with pytest.raises(ValueError, match="No completed"):
            context.snapshot_processed_inputs()
    finally:
        context._close()
        graph.close()


def test_active_execution_is_rejected_without_blocking_and_capture_can_retry():
    entered, finish = Event(), Event()

    class Slow(Block):
        inputs = (Port("value", "data"),)

        def process(self, inputs):
            entered.set()
            if not finish.wait(timeout=5):
                raise RuntimeError("Test worker did not finish")
            return Result()

    graph = Graph()
    target = graph.add(Slow())
    context = PluginDialogContext(graph, target)
    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(target.execute, {"value": 12})
            try:
                assert entered.wait(timeout=5)
                with pytest.raises(RuntimeError, match="block is active"):
                    context.snapshot_processed_inputs()
            finally:
                finish.set()
            assert future.result(timeout=5) is not None
        assert context.snapshot_processed_inputs().inputs["value"].value == 12
    finally:
        context._close()
        graph.close()


def test_capture_runs_on_owner_thread_and_private_copies_outlive_the_dialog(capture_graph):
    _, _, _, context = capture_graph
    with ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(RuntimeError, match="owning thread"):
            workers.submit(context.snapshot_processed_inputs).result(timeout=5)
    captured = context.snapshot_processed_inputs()
    context._close()
    assert captured.inputs["scene"].value.points[0, 0] == 1
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_processed_inputs()


def test_bundle_preserves_consumed_pyramid_level_even_when_draft_selection_differs():
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    pyramid = graph.add(registry.create("cv.image_pyramid", parameters={"levels": 2}))
    target = graph.add(registry.create("cv.gray"))
    target.input_levels = {"image": "1"}
    graph.connect(source.id, "image", pyramid.id, "image")
    graph.connect(pyramid.id, "images", target.id, "image")
    graph.run()
    context = PluginDialogContext(graph, target, input_levels={"image": "0"})
    try:
        captured = context.snapshot_processed_inputs().inputs["image"]
        assert captured.payload["pyramid"]["level"] == 1
        assert captured.value.shape == target.last_inputs["image"].shape
        assert metadata(captured.value) == captured.payload
        assert captured.payload["ticket"] == target.input_payloads["image"]["ticket"]
    finally:
        context._close()
        graph.close()


def test_bundle_does_not_claim_that_consumed_inputs_share_an_acquisition():
    class Target(Block):
        inputs = (Port("a", "data"), Port("b", "data"), Port("absent", "any"))

        def process(self, inputs):
            return Result()

    graph = Graph()
    target = graph.add(Target())
    tickets = TicketClock("different-acquisitions")
    a, b = tickets.issue("a"), tickets.issue("b")
    target.execute(
        {
            "a": ValuePacket({"value": 1}, {"ticket": a}),
            "b": ValuePacket({"value": 2}, {"ticket": b}),
            "absent": NO_VALUE,
        }
    )
    context = PluginDialogContext(graph, target)
    try:
        captured = context.snapshot_processed_inputs()
        assert captured.inputs["a"].payload["ticket"] == a
        assert captured.inputs["b"].payload["ticket"] == b
        assert a != b and captured.inputs["absent"].value is NO_VALUE
    finally:
        context._close()
        graph.close()
