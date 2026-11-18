"""Coherent private completed input/output captures for reusable plugin review."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.plugin_dialog import PluginDialogContext, PluginProcessedExecution
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock


class Inspection(Block):
    inputs = (Port("image"), Port("region", "data", required=False))
    outputs = (Port("image"), Port("review", "data"), Port("absent", "data", required=False))
    defaults = {"passed": True, "broken": False}

    def process(self, inputs):
        if self.parameters["broken"]:
            raise ValueError("Synthetic review failure")
        number = int(inputs["image"][0, 0])
        return Result(
            {"image": inputs["image"], "review": {"matches": [number]}, "absent": NO_VALUE},
            passed=self.parameters["passed"],
            score=0.7,
            message=f"review {number}",
            payloads={"review": {"plugin_completed_review": {"token": f"completion-{number}"}}},
        )


@pytest.fixture
def setup():
    graph = Graph()
    block = graph.add(Inspection())
    context = PluginDialogContext(graph, block)
    clock = TicketClock("paired-review")

    def execute(number=1):
        return block.execute(
            {
                "image": ValuePacket(
                    np.full((2, 3), number, np.uint16),
                    {
                        "ticket": clock.issue("camera", "external", "same-ticket"),
                        "calibration": {"id": "camera"},
                    },
                )
            }
        )

    execute()
    try:
        yield graph, block, context, execute
    finally:
        context._close()
        graph.close()


def test_selected_packets_keep_completion_payloads_and_own_their_arrays_and_metadata(setup):
    _, block, context, execute = setup
    result = context.snapshot_processed_execution(["image"], ["review"])
    assert isinstance(result, PluginProcessedExecution)
    assert (
        result.block_id == block.id and result.execution_count == 1 and result.output_revision == 0
    )
    assert result.passed and result.score == 0.7 and result.message == "review 1"
    assert set(result.inputs) == {"image"} and set(result.outputs) == {"review"}
    assert result.outputs["review"].value == {"matches": [1]}
    assert result.outputs["review"].payload["plugin_completed_review"]["token"] == "completion-1"
    result.outputs["review"].value["matches"][0] = 99
    result.outputs["review"].payload["plugin_completed_review"]["token"] = "changed"
    result.inputs["image"].value[:] = 99
    result.inputs["image"].payload["calibration"]["id"] = "changed"
    fresh = context.snapshot_processed_execution(["image"], ["review"])
    assert fresh.snapshot_id != result.snapshot_id
    assert np.all(fresh.inputs["image"].value == 1)
    assert fresh.inputs["image"].payload["calibration"]["id"] == "camera"
    assert fresh.outputs["review"].value["matches"] == [1]
    assert fresh.outputs["review"].payload["plugin_completed_review"]["token"] == "completion-1"
    execute(2)
    assert np.all(fresh.inputs["image"].value == 1)
    assert fresh.outputs["review"].value["matches"] == [1]
    newer = context.snapshot_processed_execution()
    assert newer.execution_count == 2 and newer.outputs["review"].value["matches"] == [2]


def test_default_ports_omit_absent_values_and_explicit_empty_subsets_avoid_image_copies(setup):
    _, block, context, _ = setup
    result = context.snapshot_processed_execution()
    assert set(result.inputs) == {"image"}
    assert set(result.outputs) == {"image", "review"}
    assert result.inputs["image"].payload["ticket"] == result.outputs["image"].payload["ticket"]
    assert not np.shares_memory(result.outputs["image"].value, block.last_result.outputs["image"])
    result = context.snapshot_processed_execution([], ["review"])
    assert result.inputs == {} and set(result.outputs) == {"review"}
    result = context.snapshot_processed_execution([], [])
    assert not result.inputs and not result.outputs and result.passed
    assert block.timing.count == 1  # Capturing never runs the tool.


@pytest.mark.parametrize(
    "side,ports,kind",
    [
        ("input_ports", "image", TypeError),
        ("input_ports", ["region"], ValueError),
        ("input_ports", ["image", "image"], ValueError),
        ("output_ports", "review", TypeError),
        ("output_ports", ["missing"], ValueError),
        ("output_ports", ["review", "review"], ValueError),
        ("output_ports", ["absent"], ValueError),
        ("output_ports", [1], ValueError),
    ],
)
def test_invalid_or_missing_subsets_fail_without_partial_capture_or_lock_leaks(
    setup, side, ports, kind
):
    _, _, context, _ = setup
    with pytest.raises(kind):
        context.snapshot_processed_execution(**{side: ports})
    assert context.snapshot_processed_execution(["image"], ["review"]).outputs["review"].value[
        "matches"
    ] == [1]


def test_previous_completion_remains_coherent_after_rejected_input_but_not_an_exception(setup):
    _, block, context, _ = setup
    block.input_failure_policies["image"] = "reject_failed"
    # Use the same real rejection payload emitted by a failed block.
    source = Inspection(parameters={"passed": False})
    failed = source.execute(
        {
            "image": ValuePacket(
                np.full((2, 3), 9, np.uint16), {"ticket": TicketClock().issue("camera")}
            )
        }
    )
    rejected = ValuePacket(failed.outputs["image"], failed.payloads["image"])
    block.execute({"image": rejected})
    assert block.input_rejected
    result = context.snapshot_processed_execution(["image"], ["review"])
    assert result.execution_count == 1 and result.outputs["review"].value["matches"] == [1]
    block.parameters["broken"] = True
    block.execute({"image": np.ones((2, 3), np.uint16)})
    with pytest.raises(ValueError, match="No completed"):
        context.snapshot_processed_execution()


def test_failed_inspection_is_available_and_capture_outlives_context(setup):
    _, block, context, execute = setup
    block.parameters["passed"] = False
    execute()
    result = context.snapshot_processed_execution()
    assert not result.passed
    context._close()
    assert result.outputs["review"].value["matches"] == [1]
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_processed_execution()


def test_capture_is_owner_thread_only(setup):
    _, _, context, _ = setup
    with ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(RuntimeError, match="owning thread"):
            workers.submit(context.snapshot_processed_execution).result(timeout=5)


def test_active_execution_is_rejected_without_waiting_and_later_capture_is_coherent():
    entered, finish = Event(), Event()

    class Slow(Inspection):
        def process(self, inputs):
            entered.set()
            assert finish.wait(timeout=5)
            return super().process(inputs)

    graph = Graph()
    block = graph.add(Slow())
    context = PluginDialogContext(graph, block)
    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(block.execute, {"image": np.full((2, 3), 4, np.uint16)})
            try:
                assert entered.wait(timeout=5)
                with pytest.raises(RuntimeError, match="block is active"):
                    context.snapshot_processed_execution()
            finally:
                finish.set()
            assert future.result(timeout=5) is not None
        result = context.snapshot_processed_execution()
        assert (
            result.inputs["image"].value[0, 0] == result.outputs["review"].value["matches"][0] == 4
        )
    finally:
        context._close()
        graph.close()


@pytest.mark.parametrize("profile", ["balanced", "lean"])
def test_missing_input_history_requires_a_new_debug_execution(setup, profile):
    graph, _, context, execute = setup
    graph.set_run_profile(profile)
    execute()
    with pytest.raises(ValueError, match="not retained.*Debug"):
        context.snapshot_processed_execution()
    graph.set_run_profile("debug")
    with pytest.raises(ValueError, match="unavailable|not retained"):
        context.snapshot_processed_execution()
    execute()
    assert context.snapshot_processed_execution().inputs


def test_missing_output_retention_refuses_even_with_retained_inputs_and_after_profile_switch(setup):
    graph, block, context, execute = setup
    block.run_profile = replace(PROFILES["debug"], output_previews=False)
    execute()
    with pytest.raises(ValueError, match="outputs were not retained"):
        context.snapshot_processed_execution(["image"], ["review"])
    graph.set_run_profile("debug")
    with pytest.raises(ValueError, match="outputs were not retained"):
        context.snapshot_processed_execution(["image"], ["review"])
    execute()
    assert context.snapshot_processed_execution().outputs


def test_no_input_block_cannot_expose_described_lean_outputs_after_debug_switch():
    class Source(Block):
        outputs = (Port("cloud", "cloud"),)

        def process(self, inputs):
            return Result({"cloud": PointCloud([[1, 2, 3]], units="mm", frame="camera")})

    graph = Graph()
    block = graph.add(Source())
    worker = Source()
    context = PluginDialogContext(graph, block)
    try:
        worker.run_profile = PROFILES["lean"]
        worker.execute({})
        block.copy_results_from(worker)
        assert isinstance(block.last_result.outputs["cloud"], dict)
        graph.set_run_profile("debug")
        with pytest.raises(ValueError, match="outputs were not retained"):
            context.snapshot_processed_execution([], ["cloud"])
        worker.run_profile = PROFILES["debug"]
        worker.execute({})
        block.copy_results_from(worker)
        np.testing.assert_array_equal(
            context.snapshot_processed_execution([], ["cloud"]).outputs["cloud"].value.points,
            [[1, 2, 3]],
        )
    finally:
        context._close()
        graph.close()
        worker.close()


class ManifestInspection(Block):
    inputs = (Port("image", required=False), Port("manifest", "data", required=False))
    outputs = (Port("review", "data"),)
    defaults = {"broken": False, "passed": True}

    def process(self, inputs):
        if self.parameters["broken"]:
            raise ValueError("Manifest processing failed")
        return Result(
            {"review": {"captures": ["manifest-image"], "run": self.timing.count + 1}},
            passed=self.parameters["passed"],
            payloads={"review": {"completed_review": {"model": "executed-model"}}},
        )


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("published", [False, True])
def test_explicit_empty_input_capture_uses_optional_input_completion_without_execution(
    passed, published
):
    graph = Graph()
    block = graph.add(ManifestInspection())
    worker = ManifestInspection(parameters={"passed": passed}) if published else block
    context = PluginDialogContext(graph, block)
    try:
        worker.parameters["passed"] = passed
        assert worker.execute({}) is not None
        if published:
            block.copy_results_from(worker)
        assert block.last_inputs == {}
        with pytest.raises(ValueError, match="inputs are unavailable"):
            context.snapshot_processed_execution()
        with pytest.raises(ValueError, match="inputs are unavailable"):
            context.snapshot_processed_inputs(())
        pair = context.snapshot_processed_execution((), ["review"])
        assert pair.inputs == {} and pair.passed is passed
        assert pair.outputs["review"].value["captures"] == ["manifest-image"]
        assert pair.outputs["review"].payload["completed_review"]["model"] == "executed-model"
        assert pair.execution_count == worker.timing.count == 1
        pair.outputs["review"].value["captures"].clear()
        fresh = context.snapshot_processed_execution(iter(()), ["review"])
        assert fresh.outputs["review"].value["captures"] == ["manifest-image"]
        assert fresh.snapshot_id != pair.snapshot_id and worker.timing.count == 1
        worker.parameters["broken"] = True
        worker.execute({})
        if published:
            block.copy_results_from(worker)
        with pytest.raises(ValueError, match="No completed"):
            context.snapshot_processed_execution((), ["review"])
    finally:
        context._close()
        graph.close()
        if worker is not block:
            worker.close()


@pytest.mark.parametrize("profile", ["balanced", "lean"])
@pytest.mark.parametrize("switch_before_execution", [False, True])
def test_optional_empty_capture_never_recreates_evicted_input_history(
    profile, switch_before_execution
):
    graph = Graph()
    block = graph.add(ManifestInspection())
    context = PluginDialogContext(graph, block)
    try:
        if switch_before_execution:
            graph.set_run_profile(profile)
        block.execute({})
        graph.set_run_profile(profile)
        with pytest.raises(ValueError, match="not retained.*Debug"):
            context.snapshot_processed_execution((), ["review"])
        graph.set_run_profile("debug")
        with pytest.raises(ValueError, match="unavailable|not retained"):
            context.snapshot_processed_execution((), ["review"])
        block.execute({})
        assert context.snapshot_processed_execution((), ["review"]).outputs
        # Restored snapshots must honor the receiving graph's retention budget too.
        captured = {block.id: {"last_result": block.last_result}}
        graph.set_run_profile(profile)
        graph.restore_captured(captured)
        graph.set_run_profile("debug")
        with pytest.raises(ValueError, match="unavailable|not retained"):
            context.snapshot_processed_execution((), ["review"])
    finally:
        context._close()
        graph.close()


def test_explicit_empty_selection_does_not_hide_missing_required_input_history(setup):
    _, block, context, _ = setup
    block.last_inputs = {}
    with pytest.raises(ValueError, match="inputs are unavailable"):
        context.snapshot_processed_execution((), ["review"])
