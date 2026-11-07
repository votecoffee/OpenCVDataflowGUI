"""Observation budgets must preserve program semantics, state and mutable-input isolation."""

import copy
import json

import numpy as np
import pytest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.basic_processing import GaussianBlur
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.examples import parallel_parts_graph
from opencv_dataflow.queued_execution import QueuedExecution
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.runner import main
from opencv_dataflow.validation import run_set


@pytest.mark.parametrize("name", PROFILES)
def test_default_processing_outputs_and_ticket_counts_match_across_profiles(name):
    registry = builtin_registry()
    graph = demo_graph(registry)
    reference = demo_graph(registry)
    graph.set_run_profile(name)
    for _ in range(2):
        graph.run()
        reference.run()
        for block, expected in zip(graph.blocks.values(), reference.blocks.values()):
            assert block.state == expected.state
            for port in block.outputs:
                a, b = block.last_result.outputs[port.name], expected.last_result.outputs[port.name]
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
            assert block.last_result.passed == expected.last_result.passed
        assert (
            graph.ticket_results.statistics()["total"]
            == reference.ticket_results.statistics()["total"]
        )
    if name != "debug":
        assert all(not b.last_inputs and not b.last_graphics for b in graph.blocks.values())
    if name == "lean":
        assert all(not b.last_output_images for b in graph.blocks.values())


@pytest.mark.parametrize("name", PROFILES)
def test_custom_mutating_inputs_still_have_private_process_and_finalize_snapshots(name):
    class Mutate(Block):
        inputs, outputs = (Port("image"),), (Port("image"),)

        def process(self, inputs):
            inputs["image"][:] = 99
            return Result({"image": inputs["image"]})

        def finalize_result(self, result, inputs):
            assert not inputs["image"].any()
            return result

    graph = Graph()
    block = graph.add(Mutate())
    graph.set_run_profile(name)
    original = np.zeros((12, 16), np.uint8)
    block.execute({"image": original})
    assert not block.error, block.error
    assert not original.any()
    assert block.last_result.outputs["image"].min() == 99


def test_readonly_optimization_is_not_accidentally_inherited_by_custom_override():
    class MutatingSubclass(GaussianBlur):
        def process(self, inputs):
            inputs["image"][:] = 33
            return Result({"image": inputs["image"]})

        def finalize_result(self, result, inputs):
            assert not inputs["image"].any()
            return result

    graph = Graph()
    block = graph.add(MutatingSubclass())
    graph.set_run_profile("lean")
    block.execute({"image": np.zeros((12, 16), np.uint8)})
    assert not block.error, block.error


def test_capture_filter_runs_before_formatting_or_parameter_snapshot():
    class BadString:
        def __str__(self):
            raise AssertionError("Filtered message was formatted")

    graph = demo_graph(builtin_registry())
    graph.set_run_profile("lean")
    block = next(iter(graph.blocks.values()))
    block.logger.debug("Ignored %s", BadString())
    assert not block.logger.isEnabledFor("INFO")
    assert block.logger.isEnabledFor("WARNING")
    before = graph.activity.revision()
    graph.run()
    assert graph.activity.revision() == before
    block.logger.warning("Visible", parameters={"score": 0.5})
    assert graph.activity.snapshot()["events"][-1]["parameters"] == {"score": 0.5}


def test_lean_keeps_fail_reasons_errors_and_final_decisions():
    graph = Graph.load(
        "examples/flow tools/validation_images/validation_images.json", builtin_registry()
    )
    graph.set_run_profile("lean")
    graph.blocks["validation_source"].parameters["path"] = "../../assets/validation/fail.png"
    graph.run()
    final = graph.blocks["validation_final"]
    assert final.last_result.final_judgment is False
    assert final.last_result.fail_reasons[0]["code"] == "part_count"
    assert graph.ticket_results.statistics()["fail"] == 1
    assert any(
        e["kind"] == "finish" and e["judgment"] == "fail"
        for e in graph.activity.snapshot()["events"]
    )
    graph.blocks["validation_source"].parameters["path"] = "missing.png"
    graph.run()
    errors = [e for e in graph.activity.snapshot()["events"] if e["kind"] == "exception"]
    assert errors and errors[-1]["exception"]["traceback"]
    assert errors[-1]["parameters"]["path"] == "missing.png"


def test_balanced_keeps_activity_spans_but_skips_routine_parameter_dumps():
    graph = demo_graph(builtin_registry())
    graph.set_run_profile("balanced")
    before = graph.activity.revision()
    graph.run()
    events = [e for e in graph.activity.snapshot()["events"] if e["sequence"] > before]
    assert sum(e["kind"] == "start" for e in events) == len(graph.blocks)
    assert sum(e["kind"] == "finish" for e in events) == len(graph.blocks)
    assert all(e["parameters"] == {} for e in events)


@pytest.mark.parametrize("name", PROFILES)
def test_profile_roundtrip_and_validation_scoring(name, tmp_path):
    registry = builtin_registry()
    graph = Graph.load("examples/flow tools/validation_images/validation_images.json", registry)
    graph.set_run_profile(name)
    definition = copy.deepcopy(graph.to_dict())
    loaded = Graph.from_dict(definition, registry, project_dir=graph.project_dir)
    assert loaded.run_profile.name == name
    assert all(b.run_profile.name == name for b in loaded.blocks.values())
    report = run_set(definition, graph.validation_sets[0], registry, graph.project_dir)
    assert report["summary"]["score_percent"] == 100 and report["summary"]["passed"]


def test_invalid_saved_profile_is_rejected():
    definition = demo_graph(builtin_registry()).to_dict()
    definition["run_profile"] = "unsafe"
    with pytest.raises(ValueError, match="Run profile"):
        Graph.from_dict(definition, builtin_registry())


def test_parallel_inspection_profiles_reach_each_job_and_preserve_results():
    graph = parallel_parts_graph(builtin_registry())
    graph.set_run_profile("lean")
    graph.run()
    inspector = next(b for b in graph.blocks.values() if b.type_id == "flow.inspect_each")
    assert inspector.last_result.outputs["pass_count"] == 25
    assert all(b.run_profile.name == "lean" for b in inspector.graph.blocks.values())
    assert not inspector.last_graphics and not inspector.last_output_images


def test_lean_continuous_worker_sampling_final_state_and_stop_resume(app):
    from test_editor import pump_until

    window = MainWindow()
    window.profile_choice.setCurrentText("lean")
    window.start_run(True)
    assert not window.profile_choice.isEnabled()
    worker = window.runner
    pump_until(app, lambda: worker.graph.ticket_results.statistics()["total"] >= 4)
    window.stop_run()
    pump_until(app, lambda: window.runner is None)
    count = window.root_graph.ticket_results.statistics()["total"]
    assert count >= 4 and window.profile_choice.isEnabled()
    assert all(not b.last_inputs and not b.last_output_images for b in window.graph.blocks.values())
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert window.root_graph.ticket_results.statistics()["total"] == count + 1
    window.dirty = False
    window.close()


def test_lean_display_copy_contains_no_image_arrays():
    registry = builtin_registry()
    graph = demo_graph(registry)
    graph.set_run_profile("lean")
    graph.run()
    display = Graph.from_dict(graph.to_dict(), registry)
    for key, block in graph.blocks.items():
        display.blocks[key].copy_results_from(block)
        shown = display.blocks[key]
        assert not shown.current_emissions and not shown.last_output_images
        assert all(not isinstance(v, np.ndarray) for v in shown.last_result.outputs.values())
    json.dumps([b.last_result.outputs for b in display.blocks.values()], default=str)


def test_custom_graphic_factory_can_avoid_work_without_changing_declared_outputs():
    class Graphic(Block):
        outputs = (Port("image"),)

        def process(self, inputs):
            self.generated = self.generate_graphics
            return Result(
                {"image": np.zeros((12, 16), np.uint8)},
                graphics={"detail": np.ones((12, 16), np.uint8)} if self.generate_graphics else {},
            )

    graph = Graph()
    block = graph.add(Graphic())
    graph.set_run_profile("lean")
    block.execute({})
    assert not block.generated and block.last_result.outputs["image"].shape == (12, 16)


def test_graphics_defaults_and_block_overrides_round_trip_in_editor(app, tmp_path):
    window = MainWindow()
    registry = window.registry
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    region = graph.add(registry.create("region.shape"))
    graph.connect(source.id, "image", region.id, "image")
    window.replace_graph(graph)
    window.inspector.select(region)
    graph.run()
    assert region.last_result.drawings
    overlay = region.last_result.outputs["overlay"].copy()
    timing = region.timing.snapshot()

    window.project_graphics_choice.setCurrentText("Off")
    assert not region.last_graphics and not region.last_result.drawings
    assert region.timing.snapshot() == timing
    window.inspector.graphics_choice.setCurrentText("On")
    graph.run()
    assert region.last_result.drawings
    assert not source.last_graphics
    np.testing.assert_array_equal(region.last_result.outputs["overlay"], overlay)
    assert window.dirty
    path = tmp_path / "job.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    window.replace_graph(restored, path)
    window.inspector.select(restored.blocks[region.id])
    assert window.project_graphics_choice.currentText() == "Off"
    assert window.inspector.graphics_choice.currentText() == "On"
    assert not window.dirty
    window.inspector.graphics_choice.setCurrentText("Off")
    window.project_graphics_choice.setCurrentText("On")
    restored.run()
    assert not restored.blocks[region.id].last_result.drawings
    np.testing.assert_array_equal(
        restored.blocks[region.id].last_result.outputs["overlay"], overlay
    )
    window.dirty = False
    window.close()


def test_lean_background_acquisition_keeps_fifo_data_without_restoring_previews():
    registry, graph = builtin_registry(), Graph()
    source = graph.add(registry.create("input.demo"))
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": 2}
        )
    )
    gray = graph.add(registry.create("cv.gray"))
    graph.connect(source.id, "image", queue.id, "image")
    graph.connect(queue.id, "image", gray.id, "image")
    graph.set_run_profile("lean")
    queued = QueuedExecution(graph, registry)
    try:
        queued.start()
        for index in range(1, 4):
            assert queued.run_one()
            assert graph.ticket_results.statistics()["total"] == index
            assert gray.last_result.outputs["image"].shape == (360, 480)
            assert not source.last_output_images and not source.last_inputs
        assert all(b.run_profile.name == "lean" for b in queued.producer.blocks.values())
    finally:
        queued.close()
        graph.close()


def test_cli_profile_override_does_not_modify_saved_program(tmp_path):
    path = tmp_path / "program.json"
    graph = demo_graph(builtin_registry())
    graph.save(path)
    before = path.read_bytes()
    receipt = tmp_path / "receipt.json"
    assert main([str(path), "--profile", "lean", "--repeat", "2", "--report", str(receipt)]) == 0
    report = json.loads(receipt.read_text("utf-8"))
    assert report["run_profile"] == "lean" and report["passed"]
    assert report["ticket_results"]["production_statistics"]["total"] == 2
    assert path.read_bytes() == before


def test_capture_profile_notice_changes_without_new_events(app):
    from opencv_dataflow.diagnostic_viewer import DiagnosticViewer

    graph = demo_graph(builtin_registry())
    view = DiagnosticViewer(lambda: graph.activity)
    graph.set_run_profile("lean")
    view.refresh()
    assert "Capture profile: lean" in view.summary.text()
    view.close()
