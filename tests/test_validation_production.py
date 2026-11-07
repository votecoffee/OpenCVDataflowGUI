"""Production counts, failure explanations and executable validation image sets."""

import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import parallel_parts_graph, validation_graph
from opencv_dataflow.fail_reasons import normalize_reasons
from opencv_dataflow.outcomes import TicketResults
from opencv_dataflow.production import ProductionStatistics
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolblocks import Binding, ToolBlock
from opencv_dataflow.validation import main, run_set, validate_sets
from opencv_dataflow.validation_editor import ValidationManager


@pytest.fixture
def program():
    registry = builtin_registry()
    graph = Graph.load("examples/flow tools/validation_images/validation_images.json", registry)
    return registry, graph


@pytest.mark.local_examples
def test_saved_example_and_all_expected_classes(program):
    registry, graph = program
    assert graph.to_dict() == validation_graph(registry).to_dict()
    report = run_set(graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir)
    assert report["summary"] == {
        "completed": 5,
        "scored": 4,
        "unknown": 1,
        "matched": 4,
        "score_percent": 100,
        "must_mismatches": 0,
        "advisory_mismatches": 0,
        "execution_errors": 0,
        "cancelled": False,
        "passed": True,
    }
    assert [r["actual"] for r in report["cases"]] == [True, False, True, False, False]
    assert report["cases"][-1]["matched"] is None
    assert report["cases"][1]["reason_matched"]
    assert all(len(r["image_sha256"]) == 64 for r in report["cases"])
    assert graph.ticket_results.statistics()["total"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.local_examples
def test_production_counts_are_per_ticket_not_block(program):
    _, graph = program
    graph.run()
    graph.blocks["validation_source"].parameters["path"] = "../../assets/validation/fail.png"
    graph.run()
    stats = graph.ticket_results.statistics()
    assert (stats["total"], stats["pass"], stats["fail"], stats["pass_rate"]) == (2, 1, 1, 50)
    assert stats["average_ms"] > 0 and stats["timed"] == 2


def test_ordinary_demo_counts_terminal_results_without_assigning_final_judgments():
    graph = demo_graph(builtin_registry())
    for _ in range(3):
        graph.run()
    assert graph.ticket_results.statistics()["total"] == 3
    assert graph.ticket_results.statistics()["pass"] == 3
    assert all(not row["judgments"] for row in graph.ticket_results.snapshot()["tickets"])


def test_parallel_child_tickets_do_not_inflate_image_counts():
    graph = parallel_parts_graph(builtin_registry())
    graph.run()
    assert graph.ticket_results.statistics()["total"] == 1
    assert graph.ticket_results.statistics()["pass"] == 1


@pytest.mark.local_examples
def test_duplicate_and_conflicting_final_judgments_count_once(program):
    _, graph = program
    source, final = graph.blocks["validation_source"], graph.blocks["validation_final"]
    source.ticket_settings = {"mode": "fixed", "value": "same part"}
    graph.run()
    graph.run()
    assert graph.ticket_results.statistics()["total"] == 1
    final.parameters["mode"] = "fail"
    graph.run()
    stats = graph.ticket_results.statistics()
    assert (stats["total"], stats["pass"], stats["fail"]) == (1, 0, 1)


def test_numeric_ticket_identity_and_snapshot_deduplication():
    clock, stats = TicketClock(), ProductionStatistics()
    for value in (1, 1.0):
        stats.record(clock.issue("camera", "fixed", value), True, 2)
    assert stats.summary()["total"] == 1
    restored = ProductionStatistics()
    restored.restore(copy.deepcopy(stats.snapshot()))
    restored.record(clock.issue("camera", "fixed", 1), True, 4)
    assert restored.summary()["total"] == 1 and restored.summary()["average_ms"] == 4


def test_totals_survive_ticket_history_eviction():
    consumer, clock = TicketResults(max_tickets=1), TicketClock()
    tickets = [clock.issue("camera") for _ in range(1500)]
    for index, ticket in enumerate(tickets):
        consumer.consume(
            {
                "ticket": ticket,
                "source": {"id": "final"},
                "kind": "final_judgment",
                "passed": index % 2 == 0,
                "chain_passed": index % 2 == 0,
                "score": None,
                "message": "",
                "data": {},
                "final_judgment": index % 2 == 0,
            }
        )
    assert len(consumer.snapshot()["tickets"]) == 1
    assert consumer.statistics()["total"] == 1500
    assert consumer.statistics()["pass"] == 750
    consumer.complete(tickets[0], True)
    assert consumer.statistics()["total"] == 1500
    restored = TicketResults()
    restored.restore(consumer.snapshot())
    assert restored.statistics() == consumer.statistics()
    restored.reset_statistics()
    assert restored.statistics()["total"] == 0
    assert len(restored.snapshot()["tickets"]) == 1


@pytest.mark.local_examples
def test_fail_reasons_propagate_and_are_output_with_final_judgment(program):
    _, graph = program
    graph.blocks["validation_source"].parameters["path"] = "../../assets/validation/fail.png"
    graph.run()
    final = graph.blocks["validation_final"]
    reasons = final.last_result.outputs["judgment"]["fail_reasons"]
    assert reasons[0] == {"code": "part_count", "message": "Expected exactly one part"}
    assert any(r["code"] == "inspect.contours.failed" for r in reasons)
    assert final.last_result.payloads["value"]["outcome"]["fail_reasons"] == reasons
    row = graph.ticket_results.find(final.last_result.payloads["value"]["ticket"])
    assert row["final_fail_reasons"] == reasons


@pytest.mark.parametrize("bad", [[""], [{"code": "x", "message": 12}], ["x"] * 33, "x" * 129])
def test_invalid_fail_reason_contract_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        normalize_reasons(bad)


@pytest.mark.local_examples
def test_must_mismatch_and_preferred_reason_are_required(program):
    registry, graph = program
    manifest = copy.deepcopy(graph.validation_sets[0])
    manifest["cases"] = [manifest["cases"][1]]
    manifest["cases"][0]["preferred_fail_reason"] = "wrong_reason"
    report = run_set(graph.to_dict(), manifest, registry, graph.project_dir)
    assert report["cases"][0]["actual"] is False
    assert report["summary"]["score_percent"] == 0 and not report["summary"]["passed"]
    assert report["summary"]["must_mismatches"] == 1


@pytest.mark.local_examples
def test_should_mismatch_is_advisory_and_unknown_is_unscored(program):
    registry, graph = program
    manifest = copy.deepcopy(graph.validation_sets[0])
    manifest["cases"] = [manifest["cases"][0], manifest["cases"][-1]]
    manifest["cases"][0]["expectation"] = "should_fail"
    report = run_set(graph.to_dict(), manifest, registry, graph.project_dir)
    assert report["summary"]["passed"] and report["summary"]["score_percent"] == 0
    assert report["summary"]["advisory_mismatches"] == 1
    assert report["summary"]["unknown"] == 1


@pytest.mark.local_examples
def test_missing_image_is_execution_error_not_expected_inspection_failure(program):
    registry, graph = program
    manifest = copy.deepcopy(graph.validation_sets[0])
    manifest["cases"] = [manifest["cases"][1]]
    manifest["cases"][0]["path"] = "does-not-exist.png"
    report = run_set(graph.to_dict(), manifest, registry, graph.project_dir)
    assert report["summary"]["execution_errors"] == 1
    assert not report["summary"]["passed"] and report["cases"][0]["matched"] is False


@pytest.mark.local_examples
def test_validation_replaces_camera_without_opening_hardware(program, monkeypatch):
    import cv2

    registry, graph = program
    definition = graph.to_dict()
    definition["blocks"][0].update(type="input.camera", parameters={})
    definition["blocks"][0]["exposed_outputs"] = ["image", "ticket"]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: pytest.fail("Validation opened a camera"))
    report = run_set(definition, graph.validation_sets[0], registry, graph.project_dir)
    assert report["summary"]["passed"]


@pytest.mark.local_examples
def test_validation_disables_output_writes_and_keeps_live_graph_unchanged(program, tmp_path):
    registry, graph = program
    output = graph.add(
        registry.create("output.image", parameters={"path": str(tmp_path / "must-not-write.png")})
    )
    graph.connect("validation_inspect", "overlay", output.id, "image")
    before = copy.deepcopy(graph.to_dict())
    report = run_set(before, graph.validation_sets[0], registry, graph.project_dir)
    assert report["summary"]["passed"]
    assert not (tmp_path / "must-not-write.png").exists()
    assert graph.to_dict() == before and graph.ticket_results.statistics()["total"] == 0


@pytest.mark.local_examples
def test_manifest_roundtrip_and_cli_report_preserves_existing_file(program, tmp_path):
    registry, graph = program
    saved = tmp_path / "program.json"
    graph.save(saved)
    assert Graph.load(saved, registry).validation_sets == graph.validation_sets
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "examples/flow tools/validation_images/validation_images.json",
                "--set",
                "validation_smoke",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    original = report.read_bytes()
    assert (
        main(
            [
                "examples/flow tools/validation_images/validation_images.json",
                "--set",
                "validation_smoke",
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert report.read_bytes() == original
    content = json.loads(original)
    assert (
        content["cases"][0]["image_sha256"]
        == hashlib.sha256(Path("examples/assets/validation/pass.png").read_bytes()).hexdigest()
    )


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "change", ["bad_expectation", "duplicate_case", "bad_source", "bad_preference"]
)
def test_manifest_validation(program, change):
    _, graph = program
    catalog = copy.deepcopy(graph.validation_sets)
    if change == "bad_expectation":
        catalog[0]["cases"][0]["expectation"] = "sometimes"
    elif change == "duplicate_case":
        catalog[0]["cases"].append(catalog[0]["cases"][0])
    elif change == "bad_source":
        catalog[0]["source"] = ""
    else:
        catalog[0]["cases"][0]["preferred_fail_reason"] = "wrong_for_pass"
    with pytest.raises(ValueError):
        validate_sets(catalog)


def test_statusbar_statistics_survive_worker_runs_and_reset(app):
    from test_editor import pump_until

    window = MainWindow()
    for count in (1, 2):
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        window.refresh_production()
        assert f"Total {count}" in window.production_label.text()
        assert f"Pass {count}" in window.production_label.text()
        assert "100.0%" in window.production_label.text()
    window.reset_production()
    assert "Total 0" in window.production_label.text()
    window.close()


@pytest.mark.local_examples
def test_validation_manager_background_run_and_saved_criteria(app, program):
    from test_editor import pump_until

    _, graph = program
    window = MainWindow()
    window.replace_graph(graph)
    dialog = ValidationManager(window)
    assert dialog.images.rowCount() == 5
    assert dialog.images.cellWidget(1, 1).currentText() == "must_fail"
    assert dialog.images.item(1, 2).text() == "part_count"
    dialog.run_images()
    assert not dialog.editor.isEnabled()
    pump_until(app, lambda: dialog.worker is None)
    assert dialog.report["summary"]["score_percent"] == 100
    assert "Score 100.0%" in dialog.status.text()
    assert graph.ticket_results.statistics()["total"] == 0
    dialog.images.setCurrentCell(0, 0)
    dialog.move(1)
    assert dialog.images.item(1, 0).data(256) == "known_pass"
    assert dialog.save_set()
    assert graph.validation_sets[0]["cases"][1]["id"] == "known_pass"
    dialog.close()
    window.dirty = False
    window.close()


@pytest.mark.local_examples
def test_sequence_validation_can_warm_up_and_isolated_validation_starts_fresh(program):
    registry, original = program
    graph = Graph()
    source = graph.add(registry.create("input.file", block_id="source"))
    memory = graph.add(registry.create("sequence.image_memory", parameters={"warmup": 2}))
    final = graph.add(registry.create("inspect.final_judgment"))
    graph.connect(source.id, "image", memory.id, "image")
    graph.connect(memory.id, "image", final.id, "value")
    cases = [
        {"id": "warmup", "path": "../../assets/validation/pass.png", "expectation": "unknown"},
        {"id": "inspect", "path": "../../assets/validation/pass.png", "expectation": "must_pass"},
    ]
    manifest = {
        "id": "memory",
        "name": "Memory",
        "source": source.id,
        "judgment": final.id,
        "state_mode": "sequence",
        "cases": cases,
    }
    report = run_set(graph.to_dict(), manifest, registry, original.project_dir)
    assert report["summary"]["passed"] and report["cases"][0]["actual"] is None
    manifest["state_mode"] = "isolated"
    report = run_set(graph.to_dict(), manifest, registry, original.project_dir)
    assert not report["summary"]["passed"] and report["cases"][1]["actual"] is None


@pytest.mark.local_examples
def test_validation_cancellation_keeps_partial_report(program):
    registry, graph = program
    stop = threading.Event()
    report = run_set(
        graph.to_dict(),
        graph.validation_sets[0],
        registry,
        graph.project_dir,
        cancelled=stop.is_set,
        progress=lambda row: stop.set(),
    )
    assert report["summary"]["completed"] == 1
    assert report["summary"]["cancelled"] and not report["summary"]["passed"]


@pytest.mark.local_examples
def test_wired_reason_codes_are_consumable_and_forced_pass_does_not_publish_failure(program):
    registry, graph = program
    final = graph.blocks["validation_final"]
    graph.blocks["validation_source"].parameters["path"] = "../../assets/validation/fail.png"
    graph.run()
    packet = graph.blocks["validation_inspect"].current_emissions["overlay"][0]
    tool = registry.create("inspect.final_judgment", parameters={"mode": "boolean_input"})
    tool.execute(
        {
            "value": packet,
            "passed": False,
            "reasons": [{"code": "surface", "message": "Scratch found"}],
        }
    )
    assert tool.last_result.outputs["judgment"]["fail_reasons"][0]["code"] == "surface"
    final.parameters["mode"] = "pass"
    graph.run()
    ticket = final.last_result.payloads["value"]["ticket"]
    row = graph.ticket_results.find(ticket)
    assert row["status"] == "final_pass"
    assert row["events"][-1]["fail_reasons"] == []
    assert final.last_result.outputs["judgment"]["fail_reasons"] == []


@pytest.mark.local_examples
def test_validation_can_use_automatic_terminal_results_with_no_final_tool(program):
    registry, original = program
    graph = demo_graph(registry)
    inspector = list(graph.blocks.values())[-1]
    inspector.parameters["expected_count"] = 1
    inspector.parameters["min_area"] = 20.0
    source = next(iter(graph.blocks))
    manifest = {
        "id": "automatic",
        "name": "Automatic",
        "source": source,
        "cases": [
            {"id": "one", "path": "../../assets/validation/pass.png", "expectation": "must_pass"}
        ],
    }
    report = run_set(graph.to_dict(), manifest, registry, original.project_dir)
    assert report["summary"]["passed"]


@pytest.mark.local_examples
def test_automatic_counts_include_private_toolblock_inspection_failures(program):
    registry, original = program
    inner, outer = Graph(), Graph()
    gray = inner.add(registry.create("cv.gray"))
    inspection = inner.add(registry.create("inspect.contours", parameters={"expected_count": 2}))
    inner.connect(gray.id, "image", inspection.id, "image")
    tool = outer.add(
        ToolBlock(inner, {"image": Binding(gray.id, "image")}, {"image": Binding(gray.id, "image")})
    )
    source = outer.add(
        registry.create(
            "input.file",
            parameters={
                "path": str((original.project_dir / "../../assets/validation/pass.png").resolve())
            },
        )
    )
    outer.connect(source.id, "image", tool.id, "image")
    outer.run()
    assert tool.state == "fail"
    assert outer.ticket_results.statistics()["fail"] == 1
    assert outer.ticket_results.statistics()["total"] == 1
