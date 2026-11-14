"""New tools capture one private cached publication without executing it."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugin_dialog import PluginDialogContext


class Source(Block):
    outputs = (Port("depth", "depth"), Port("camera", "data"),
               Port("absent", "data", required=False))

    def process(self, inputs):
        return Result({"depth": np.full((3, 4), 500, np.uint16),
                       "camera": {"nested": {"value": 5}}, "absent": NO_VALUE},
                      payloads={"depth": {"acquisition": "original"}})


class Unconfigured(Block):
    inputs = (Port("depth", "depth"), Port("camera", "data", required=False),
              Port("optional", "data", required=False))

    def process(self, inputs):
        raise AssertionError("Capture must not execute the target")


@pytest.fixture
def connected():
    graph = Graph()
    source, target = graph.add(Source()), graph.add(Unconfigured())
    graph.connect(source.id, "depth", target.id, "depth")
    graph.connect(source.id, "camera", target.id, "camera")
    source.execute({})
    context = PluginDialogContext(graph, target)
    try:
        yield graph, source, target, context
    finally:
        context._close()
        graph.close()


def test_capture_precedes_target_execution_and_copies_all_packets(connected):
    graph, source, target, context = connected
    result = context.snapshot_connected_inputs()
    assert set(result.inputs) == {"depth", "camera"}
    assert target.timing.count == 0 and result.output_revision == 0
    assert result.sources["depth"]["execution_count"] == source.timing.count
    assert result.inputs["depth"].payload == {"acquisition": "original"}
    result.inputs["depth"].value[:] = 0
    result.inputs["camera"].value["nested"]["value"] = -1
    assert source.last_result.outputs["depth"][0, 0] == 500
    assert source.last_result.outputs["camera"]["nested"]["value"] == 5
    assert context.snapshot_connected_inputs().snapshot_id != result.snapshot_id


def test_thread_busy_and_closed_guards_release_locks(connected):
    graph, source, target, context = connected
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owning thread"):
            pool.submit(context.snapshot_connected_inputs).result()
        ready, release = Event(), Event()
        def hold():
            with source._execution_lock:
                ready.set()
                release.wait(5)
        future = pool.submit(hold)
        assert ready.wait(3)
        try:
            with pytest.raises(RuntimeError, match="active"):
                context.snapshot_connected_inputs()
        finally:
            release.set()
            future.result()
    assert context.snapshot_connected_inputs().inputs
    context._close()
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_connected_inputs()


@pytest.mark.parametrize("ports", ["depth", ["depth", "depth"], ["unknown"]])
def test_port_selection(connected, ports):
    with pytest.raises((ValueError, TypeError)):
        connected[-1].snapshot_connected_inputs(ports)


def test_required_value_and_typed_input_validation(connected):
    graph, source, target, context = connected
    source.last_result.outputs["depth"] = NO_VALUE
    with pytest.raises(ValueError, match="run its source"):
        context.snapshot_connected_inputs()
    source.last_result.outputs["depth"] = np.zeros((1,), np.uint16)
    with pytest.raises(TypeError, match="depth image"):
        context.snapshot_connected_inputs()
    source.last_result.outputs["depth"] = np.zeros((3, 4), np.uint16)
    assert context.snapshot_connected_inputs().inputs
