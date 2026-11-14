"""Train unconfigured tools from one cached editor publication without execution."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, metadata
from opencv_dataflow.plugin_dialog import PluginDialogContext


class Source(Block):
    outputs = (Port("depth", "depth"), Port("camera", "data"), Port("absent", "data"))

    def process(self, inputs):
        return Result(
            {
                "depth": DepthArray(np.full((3, 4), 500, np.uint16), units="mm", display_max=1000),
                "camera": {"fx": 100, "frame": "camera"},
                "absent": NO_VALUE,
            },
            payloads={"depth": {"capture": {"frame": 2}}, "camera": {"frame": "camera"}},
        )


class Untrained(Block):
    inputs = (
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("bin_pose", "data", required=False),
        Port("absent", "data", required=False),
    )

    def process(self, inputs):
        raise ValueError("Train this tool first")


@pytest.fixture
def untrained():
    graph = Graph()
    source, target = graph.add(Source()), graph.add(Untrained())
    for name in ("depth", "camera", "absent"):
        graph.connect(source.id, name, target.id, name)
    graph.run()
    assert target.state == "error"
    context = PluginDialogContext(graph, target)
    try:
        yield graph, source, target, context
    finally:
        context._close()
        graph.close()


def test_untrained_target_captures_private_typed_bundle_with_missing_optional(untrained):
    _, source, target, context = untrained
    before = source.timing.count, target.timing.count
    first = context.snapshot_connected_inputs()
    assert set(first.inputs) == {"depth", "camera"}
    assert first.missing_inputs == ("bin_pose", "absent")
    assert first.sources["depth"]["block_id"] == source.id
    assert first.inputs["depth"].value.units == "mm"
    assert first.inputs["depth"].value.dtype == source.last_result.outputs["depth"].dtype
    assert metadata(first.inputs["depth"].value) == first.inputs["depth"].payload
    first.inputs["depth"].value[:] = 0
    first.inputs["depth"].payload["capture"]["frame"] = 99
    first.inputs["camera"].value["fx"] = 99
    second = context.snapshot_connected_inputs()
    assert first.snapshot_id != second.snapshot_id
    assert first.publication_id == second.publication_id
    assert np.all(second.inputs["depth"].value == 500)
    assert second.inputs["camera"].value["fx"] == 100
    assert second.inputs["depth"].payload["capture"]["frame"] == 2
    assert (source.timing.count, target.timing.count) == before
    context._outputs_published()
    assert context.snapshot_connected_inputs().publication_id != second.publication_id


@pytest.mark.parametrize(
    "ports,error",
    [
        ("depth", TypeError),
        (["depth", "depth"], ValueError),
        (["typo"], ValueError),
        (["depth", 1], ValueError),
    ],
)
def test_invalid_port_sets_release_no_partial_bundle(untrained, ports, error):
    *_, context = untrained
    with pytest.raises(error):
        context.snapshot_connected_inputs(ports)
    assert context.snapshot_connected_inputs(["camera"]).inputs["camera"].value["fx"] == 100


def test_required_missing_active_and_wrong_thread_reject_without_waiting(untrained):
    graph, source, target, context = untrained
    source._execution_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            context.snapshot_connected_inputs()
    finally:
        source._execution_lock.release()
    assert target._execution_lock.acquire(blocking=False)
    target._execution_lock.release()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owning thread"):
            pool.submit(context.snapshot_connected_inputs).result(timeout=5)
    source.clear_result()
    with pytest.raises(ValueError, match="required input depth"):
        context.snapshot_connected_inputs()
    assert source._execution_lock.acquire(blocking=False)
    source._execution_lock.release()
    context._close()
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_connected_inputs()


def test_connected_capture_uses_draft_pyramid_level():
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    pyramid = graph.add(registry.create("cv.image_pyramid", parameters={"levels": 2}))
    target = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", pyramid.id, "image")
    graph.connect(pyramid.id, "images", target.id, "image")
    graph.run()
    context = PluginDialogContext(graph, target, input_levels={"image": "1"})
    try:
        bundle = context.snapshot_connected_inputs(["image"])
        assert bundle.inputs["image"].payload["pyramid"]["level"] == 1
        assert bundle.inputs["image"].value.shape == (180, 240, 3)
    finally:
        context._close()
        graph.close()


def test_publication_change_rejects_capture_and_releases_locks(untrained, monkeypatch):
    import opencv_dataflow.plugin_dialog as module

    _, source, target, context = untrained
    original = module.prepare_inputs

    def changed(*args, **kwargs):
        context._outputs_published()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_inputs", changed)
    with pytest.raises(RuntimeError, match="publication changed"):
        context.snapshot_connected_inputs()
    for block in (source, target):
        assert block._execution_lock.acquire(blocking=False)
        block._execution_lock.release()
