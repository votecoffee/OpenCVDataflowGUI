"""Independent ordered stream references, integer timing and bounded lifecycle."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import ImageArray
from opencv_dataflow.outcomes import failed_input
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, metadata
from opencv_dataflow.ros2_context import AttachAcquisition, GuardRunIdentity, RunIdentity
from opencv_dataflow.ros2_pairing import PairROSImages
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def image(ns, source="left", *, frame="camera", failed=False):
    header = {"frame_id": frame, "stamp": {"sec": ns // 10**9, "nanosec": ns % 10**9}}
    value = ImageArray(np.full((3, 4), 7, np.uint16), "gray", {"ros2": {"header": header}})
    payload = {"ticket": TicketClock("pair-test").issue(source, "external", str(ns))}
    if failed:
        payload["outcome"] = {
            "model": "tool_outcome_v1",
            "passed": False,
            "chain_passed": False,
            "source": {"id": source},
            "score": None,
            "message": "bad",
            "data": {},
        }
    return ValuePacket(value, payload)


def run(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def pair(**parameters):
    block = PairROSImages(parameters=parameters)
    now = [100.0]
    block.now = lambda: now[0]
    return block, now


def test_near_stamps_get_one_shared_ticket_and_preserve_original_headers_and_bytes():
    block, _ = pair(tolerance_ms=2)
    left, right = image(2_000_000_001), image(2_002_000_001, "right")
    assert run(block, input_1=left).outputs["group"] is NO_VALUE
    result = run(block, input_2=right)
    assert result.outputs["info"]["spread_ns"] == 2_000_000
    assert result.outputs["info"]["policy"] == "ordered_fifo"
    assert not result.outputs["info"]["physical_sync_verified"]
    group = result.outputs["group"]
    assert group.labels == ("input_1", "input_2")
    assert ticket_key(metadata(group.items[0])["ticket"]) == ticket_key(
        metadata(group.items[1])["ticket"]
    )
    assert result.outputs["info"]["members"]["input_1"]["ticket"] == left.payload["ticket"]
    for port, original in (("input_1", left), ("input_2", right)):
        np.testing.assert_array_equal(result.outputs[port], original.value)
        assert (
            metadata(result.outputs[port])["ros2"]["header"]
            == original.value.metadata["ros2"]["header"]
        )
    assert not block.runtime["bytes"]


def test_fifo_uses_first_feasible_not_a_future_nearest_neighbor():
    block, _ = pair(tolerance_ms=5)
    run(block, input_1=image(10_000_000))
    run(block, input_1=image(14_000_000))
    result = run(block, input_2=image(15_000_000, "right"))
    assert result.outputs["info"]["members"]["input_1"]["ticket"]["value"] == "10000000"
    assert len(block.runtime["queues"]["input_1"]) == 1


def test_drop_provably_too_old_heads_then_pair_earliest_feasible():
    block, _ = pair(tolerance_ms=1)
    run(block, input_1=image(1_000_000))
    run(block, input_1=image(5_000_000))
    result = run(block, input_2=image(5_100_000, "right"))
    assert result.outputs["info"]["spread_ns"] == 100_000
    assert block.runtime["dropped"] == 1
    assert not block.runtime["bytes"]


def test_nanosecond_comparison_never_rounds_epoch_times_to_float():
    block, _ = pair(tolerance_ms=0)
    epoch = 2_000_000_000 * 10**9
    run(block, input_1=image(epoch + 1))
    assert run(block, input_2=image(epoch + 2, "right")).outputs["group"] is NO_VALUE
    assert block.runtime["dropped"] == 1
    assert run(block, input_1=image(epoch + 2)).outputs["info"]["spread_ns"] == 0


def test_duplicates_do_not_reuse_images_and_backward_stamps_hold_until_reset():
    block, _ = pair()
    run(block, input_1=image(100))
    run(block, input_2=image(100, "right"))
    assert run(block, input_1=image(100)).outputs["group"] is NO_VALUE
    assert block.runtime["duplicates"] == 1
    run(block, input_2=image(200, "right"))
    result = run(block, input_1=image(90))
    assert not result.passed and block.runtime["held"] and not block.runtime["bytes"]
    assert run(block, input_1=image(300), input_2=image(300, "right")).outputs["group"] is NO_VALUE
    run(block, reset=True)
    run(block, input_1=image(1))
    run(block, reset=True)  # Held reset level does not clear the new image.
    assert run(block, input_2=image(1, "right")).outputs["group"] is not NO_VALUE


def test_monotonic_deadline_includes_stopped_time_and_late_arrival_cannot_revive_image():
    block, clock = pair(max_wait_ms=10)
    run(block, input_1=image(10))
    clock[0] += 0.011
    assert run(block).outputs["group"] is NO_VALUE
    assert block.runtime["dropped"] == 1 and not block.runtime["bytes"]
    assert run(block, input_2=image(10, "right")).outputs["group"] is NO_VALUE


@pytest.mark.parametrize("parameters", [{"max_pending": 1}, {"max_bytes_mb": 0.001}])
def test_overflow_is_contained_and_clears_queues(parameters):
    block, _ = pair(**parameters)
    value = image(10)
    if "max_bytes_mb" in parameters:
        value.value = ImageArray(np.ones((100, 100), np.uint16), "gray", value.value.metadata)
    first = block.execute({"input_1": value})
    result = block.execute({"input_1": image(11)}) if first is not None else first
    assert result is None and block.runtime["bytes"] == 0
    assert not any(block.runtime["queues"].values())


@pytest.mark.parametrize("change", ["zero", "header", "ticket", "binding"])
def test_bad_image_data_is_contained(change):
    block, _ = pair()
    value = image(10)
    if change == "zero":
        value = image(0)
    elif change == "header":
        value.value.metadata["ros2"].clear()
    elif change == "ticket":
        value.payload.clear()
    else:
        value.payload["ros2_static_binding"] = {"old": True}
    assert block.execute({"input_1": value}) is None
    assert not block.current_emissions and not block.runtime["bytes"]


def test_frame_mismatch_is_explicit_and_failed_member_status_is_preserved():
    block, _ = pair()
    run(block, input_1=image(10))
    assert block.execute({"input_2": image(10, "right", frame="other")}) is None
    block.parameters["require_same_frame"] = False
    run(block, input_1=image(20, failed=True))
    result = run(block, input_2=image(20, "right", frame="other"))
    assert failed_input(result.outputs["group"])
    assert failed_input(result.emissions["input_1"][0])
    assert not failed_input(result.emissions["input_2"][0])


def test_context_change_flushes_old_queue_and_paired_capture_passes_current_guard():
    source, attach = RunIdentity(), AttachAcquisition()
    current = run(source).outputs["context"]
    block, _ = pair(require_context=True)
    run(block, context=current)

    def tagged(context, ns, source="left"):
        result = run(attach, value=image(ns, source), context=context)
        return ValuePacket(result.outputs["value"], result.payloads["value"])

    run(block, input_1=tagged(current, 10))
    new = run(source, reset=True).outputs["context"]
    run(block, context=new)
    assert not block.runtime["bytes"]
    assert block.execute({"input_2": tagged(current, 10, "right")}) is None
    run(block, context=new)
    run(block, input_1=tagged(new, 20))
    result = run(block, input_2=tagged(new, 21, "right"))
    group = result.emissions["group"][0]
    assert run(GuardRunIdentity(), value=group, context=new).outputs["info"]["checked_members"] == 2
    assert result.outputs["info"]["members"]["input_1"]["acquisition"]["capture"]["value"] == "20"


@pytest.mark.parametrize(
    "parameters",
    [{"tolerance_ms": -1}, {"max_wait_ms": True}, {"clock_domain": ""}, {"max_pending": 1.5}],
)
def test_parameter_limits_are_visible_validation_errors(parameters):
    block = PairROSImages(parameters=parameters)
    assert block.execute({}) is None
    assert block.error_kind == "invalid"


def test_changed_clock_or_matching_configuration_never_reuses_pending_images():
    block, _ = pair()
    run(block, input_1=image(10))
    block.parameters["clock_domain"] = "new_epoch"
    assert run(block, input_2=image(10, "right")).outputs["group"] is NO_VALUE
    assert not block.runtime["queues"]["input_1"]


def test_context_can_arrive_after_images_without_permanent_startup_failure():
    source, attach = RunIdentity(), AttachAcquisition()
    context = run(source).outputs["context"]
    block, _ = pair(require_context=True)
    tagged = run(attach, value=image(10), context=context)
    value = ValuePacket(tagged.outputs["value"], tagged.payloads["value"])
    assert run(block, input_1=value).outputs["group"] is NO_VALUE
    assert block.runtime["needs_context"] and not block.runtime["bytes"]
    run(block, context=context)
    assert not block.runtime["needs_context"]
    assert run(block, input_2=value).outputs["group"] is NO_VALUE  # Dropped image is not replayed.
    assert run(block, input_1=value).outputs["group"] is not NO_VALUE


def test_recipe_conflicts_do_not_get_relabelled_as_one_pair():
    block, _ = pair()
    left, right = image(10), image(10, "right")
    ref = {"id": "recipe", "revision": 1, "digest": "a" * 64}
    left.payload["recipe_ref"] = ref
    right.payload["recipe_ref"] = {**ref, "revision": 2}
    run(block, input_1=left)
    assert block.execute({"input_2": right}) is None
    assert not block.runtime["bytes"]


def test_queue_copy_is_private_and_project_xml_never_contains_buffered_images(tmp_path):
    block, _ = pair()
    run(block, input_1=image(10))
    other = PairROSImages()
    other.copy_runtime_from(block)
    other.runtime["queues"]["input_1"][0]["packet"].value[:] = 99
    assert block.runtime["queues"]["input_1"][0]["packet"].value[0, 0] == 7
    graph = Graph()
    graph.add(block)
    path = tmp_path / "pair.xml"
    graph.save(path)
    assert "ros_time_pair_v1" not in path.read_text()
    restored = Graph.load(path, builtin_registry())
    loaded = next(iter(restored.blocks.values()))
    assert loaded.runtime["bytes"] == 0
    assert block_entry(loaded).path == ("Industrial integration", "ROS 2")
    restored.close()
    graph.close()


def test_graph_multi_emission_and_idle_deadline_polling():
    class Source(Block):
        outputs = (Port("left"), Port("right"))

        def process(self, inputs):
            left, right = image(10), image(11, "right")
            return Result(
                {"left": left.value, "right": right.value},
                payloads={"left": metadata(left), "right": metadata(right)},
            )

    graph = Graph()
    source = graph.add(Source())
    block = graph.add(PairROSImages())
    graph.connect(source.id, "left", block.id, "input_1")
    graph.connect(source.id, "right", block.id, "input_2")
    graph.run()
    assert len(block.current_emissions["group"]) == 1, block.error
    assert (
        block.last_result.payloads["input_1"]["ticket"]
        == block.last_result.payloads["input_2"]["ticket"]
    )
    graph.close()


def test_many_ordered_streams_match_an_independent_fifo_reference():
    rng = np.random.default_rng(451)
    for _ in range(20):
        left = np.cumsum(rng.integers(1, 20, 25)).tolist()
        right = np.cumsum(rng.integers(1, 20, 25)).tolist()
        i, j, expected = 0, 0, []
        while i < len(left) and j < len(right):
            if abs(left[i] - right[j]) <= 3:
                expected.append((left[i], right[j]))
                i, j = i + 1, j + 1
            elif left[i] < right[j]:
                i += 1
            else:
                j += 1
        block, _ = pair(tolerance_ms=0.000003, max_wait_ms=0)
        # Different interleavings preserve each stream's own acquisition order.
        pending = {"input_1": list(left), "input_2": list(right)}
        actual = []
        while any(pending.values()):
            names = [name for name in pending if pending[name]]
            name = names[int(rng.integers(len(names)))]
            result = run(block, **{name: image(pending[name].pop(0), name)})
            for event in result.emissions["info"]:
                members = event.value["members"]
                actual.append(
                    tuple(int(members[n]["ticket"]["value"]) for n in ("input_1", "input_2"))
                )
        assert actual == expected


def test_graph_polls_timeout_after_upstream_becomes_unavailable():
    class Once(Block):
        outputs = (Port("image"),)
        emitted = False

        def process(self, inputs):
            value = image(10)
            output = NO_VALUE if self.emitted else value.value
            self.emitted = True
            return Result({"image": output}, payloads={"image": metadata(value)})

    graph = Graph()
    source = graph.add(Once())
    block, clock = pair(max_wait_ms=1)
    graph.add(block)
    graph.connect(source.id, "image", block.id, "input_1")
    graph.run()
    assert block.runtime["bytes"] > 0
    clock[0] += 0.01
    graph.run()
    assert block.runtime["bytes"] == 0 and block.runtime["dropped"] == 1
    graph.close()


def test_pair_controls_are_available_and_live_preview_is_disabled(app):
    from PySide6.QtWidgets import QCheckBox

    from opencv_dataflow.app import MainWindow

    window = MainWindow()
    try:
        graph = Graph()
        block = graph.add(PairROSImages())
        window.replace_graph(graph)
        window.scene.nodes[block.id].setSelected(True)
        assert isinstance(window.inspector.fields["require_same_frame"], QCheckBox)
        assert "tolerance_ms" in window.inspector.fields
        assert not window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
