"""Independent geometric/count/AP references and real ONNX validation controls."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.assistant_cli import main as api_main
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.detection_evaluation import (
    box_iou,
    capture_detections,
    detection_config,
    evaluate_detections,
    truth,
)
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.validation import run_set, validate_sets

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/detector_evaluation"


def obj(identity, label="A", box=None, score=None):
    result = {"id": str(identity), "label": label, "bbox": box or [0, 0, 10, 10]}
    if score is not None:
        result["score"] = score
    return result


def row(identity="image", expected=None, predictions=None, status="predicted"):
    return {
        "id": identity,
        "size": [100, 100],
        "truth": {"size": [100, 100], "objects": expected or []},
        "predictions": predictions or [],
        "status": status,
    }


def test_continuous_edge_iou_exact_and_independent_native_polygon_area():
    a = [obj("a", box=[0, 0, 10, 10]), obj("b", box=[10, 0, 10, 10])]
    b = [obj("a"), obj("b", box=[5, 0, 10, 10]), obj("c", box=[1, 1, 2, 2])]
    np.testing.assert_allclose(box_iou(a, b), [[1, 1 / 3, 0.04], [0, 1 / 3, 0]])
    rng = np.random.default_rng(731)
    for _ in range(20):
        boxes = [[*rng.uniform(0, 20, 2), *rng.uniform(1, 20, 2)] for _ in range(2)]
        polygons = [
            np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
            for x, y, w, h in boxes
        ]
        area, _ = cv2.intersectConvexConvex(*polygons)
        reference = area / (sum(b[2] * b[3] for b in boxes) - area)
        assert box_iou([obj(0, box=boxes[0])], [obj(1, box=boxes[1])])[0, 0] == pytest.approx(
            reference, abs=1e-6
        )


def test_duplicates_wrong_classes_and_missed_images_keep_all_object_denominators():
    report = evaluate_detections(
        [
            row(
                "one",
                [obj("a"), obj("b", "B", [50, 0, 10, 10])],
                [
                    obj("a", score=0.9),
                    obj("duplicate", score=0.8),
                    obj("wrong", "C", [50, 0, 10, 10], 0.7),
                ],
            ),
            row("missing", [obj("m")], status="missing"),
        ],
        classes=["A", "B"],
    )
    summary = report["summary"]
    assert (summary["tp"], summary["fp"], summary["fn"], summary["support"]) == (1, 2, 2, 3)
    assert summary["precision"] == pytest.approx(1 / 3)
    assert summary["recall"] == pytest.approx(1 / 3)
    assert report["classes"] == ["A", "B", "C"]
    a, b, c = report["per_class"]
    assert a["ap"] == 0.5 and b["ap"] == 0 and c["ap"] is None
    assert summary["map"] == 0.25 and summary["ap_classes"] == 2
    assert not summary["passed"]


def test_score_greedy_matching_is_one_to_one_and_not_global_cardinality():
    report = evaluate_detections(
        [
            row(
                expected=[obj("a"), obj("b", box=[8, 0, 10, 10])],
                predictions=[obj("first", box=[0, 0, 18, 10], score=0.9), obj("second", score=0.8)],
            )
        ]
    )
    assert (report["summary"]["tp"], report["summary"]["fp"], report["summary"]["fn"]) == (1, 1, 1)
    assert report["images"][0]["matches"][0]["truth"] == "a"
    assert report["images"][0]["missed_truth"] == ["b"]


def test_tiny_translated_boxes_have_stable_iou_and_underflow_is_rejected():
    boxes = [obj("tiny", box=[1000, 1000, 1e-10, 1e-10])]
    assert box_iou(boxes, boxes)[0, 0] == 1
    with pytest.raises(ValidationError, match="representable area"):
        evaluate_detections(
            [row(predictions=[obj("underflow", box=[0, 0, 1e-300, 1e-300], score=0.9)])]
        )


def test_equal_scores_grouped_for_ap_and_id_order_is_stable():
    example = row(
        expected=[obj("a")],
        predictions=[obj("found", score=0.9), obj("false", box=[50, 0, 10, 10], score=0.9)],
    )
    report = evaluate_detections([example])
    assert report["summary"]["map"] == 0.5
    assert len(report["per_class"][0]["curve"]) == 1
    example["predictions"].reverse()
    assert evaluate_detections([example]) == report


def test_ap_envelope_and_101_samples_have_known_independent_values():
    example = row(
        expected=[obj("a"), obj("b", box=[30, 0, 10, 10])],
        predictions=[
            obj("first", score=0.9),
            obj("duplicate", score=0.8),
            obj("last", box=[30, 0, 10, 10], score=0.7),
        ],
    )
    all_points = evaluate_detections([example])
    assert all_points["summary"]["map"] == pytest.approx(5 / 6)
    sampled = evaluate_detections([example], ap_interpolation="101_points")
    assert sampled["summary"]["map"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert sampled["per_class"][0]["curve"][-1]["recall"] == 1


def test_ap_ranks_across_images_and_keeps_negative_image_false_positives():
    report = evaluate_detections(
        [
            row("first", [obj("a")], [obj("good", score=0.9)]),
            row("negative", predictions=[obj("false", score=0.8)]),
            row("last", [obj("b")], [obj("good", score=0.7)]),
        ]
    )
    assert report["summary"]["tp"] == 2 and report["summary"]["fp"] == 1
    assert report["summary"]["map"] == pytest.approx(5 / 6)
    assert report["per_class"][0]["curve"][-1]["recall"] == 1


def test_score_sweep_can_include_rows_below_operating_cutoff_but_does_not_change_ap():
    example = row(
        expected=[obj("a"), obj("b", box=[30, 0, 10, 10])],
        predictions=[
            obj("first", score=0.9),
            obj("duplicate", score=0.8),
            obj("last", box=[30, 0, 10, 10], score=0.7),
        ],
    )
    report = evaluate_detections(
        [example], score_threshold=0.85, score_thresholds=[0.9, 0.8, 0.7, 1]
    )
    assert (report["summary"]["tp"], report["summary"]["fp"], report["summary"]["fn"]) == (1, 0, 1)
    assert [(s["tp"], s["fp"], s["fn"]) for s in report["score_sweeps"]] == [
        (1, 0, 1),
        (1, 1, 1),
        (2, 1, 0),
        (0, 0, 2),
    ]
    assert report["summary"]["map"] == pytest.approx(5 / 6)


def test_localization_cutoffs_and_mean_ap_are_explicit():
    example = row(expected=[obj("a")], predictions=[obj("shifted", box=[2, 0, 10, 10], score=0.9)])
    report = evaluate_detections([example], iou_threshold=0.5, iou_thresholds=[0.5, 0.75])
    assert [r["summary"]["tp"] for r in report["iou_reports"]] == [1, 0]
    assert report["summary"]["map"] == 1 and report["summary"]["mean_ap_over_iou"] == 0.5
    boundary = evaluate_detections([example], iou_threshold=2 / 3)
    assert boundary["summary"]["tp"] == 1


def test_negative_and_unlabeled_images_are_distinct_and_errors_never_pass():
    negative = row()
    clean = evaluate_detections([negative])
    assert clean["summary"]["passed"] and clean["summary"]["recall"] is None
    false_positive = evaluate_detections([row(predictions=[obj("fp", score=0.9)])])
    assert false_positive["summary"]["fp"] == 1 and not false_positive["summary"]["passed"]
    unlabeled = {**row(predictions=[obj("unscored", score=0.9)]), "truth": None}
    report = evaluate_detections([negative, {**unlabeled, "id": "unlabeled"}])
    assert report["summary"]["passed"] and report["summary"]["unlabeled_images"] == 1
    assert report["summary"]["fp"] == 0
    assert not evaluate_detections([unlabeled])["summary"]["passed"]
    for state in ("error", "missing"):
        assert not evaluate_detections([row(status=state)], min_precision=0, min_recall=0)[
            "summary"
        ]["passed"]


def test_expected_ids_truth_and_group_checks_and_missing_rows():
    expected = [
        {"id": "image", "detection_truth": row()["truth"]},
        {"id": "missing", "detection_truth": {"size": [100, 100], "objects": [obj("part")]}},
    ]
    report = evaluate_detections([row()], expected_cases=expected)
    assert report["expected_ids_verified"] and report["summary"]["fn"] == 1
    assert report["summary"]["status_counts"]["missing"] == 1
    assert next(r for r in report["records"] if r["id"] == "missing")["size"] is None
    for value in ([row("unexpected")], [row(expected=[obj(1)])], [{**row(), "group": "different"}]):
        with pytest.raises(ValidationError):
            evaluate_detections(value, expected_cases=expected)
    with pytest.raises(ValidationError, match="Duplicate"):
        evaluate_detections([row(), row()])


@pytest.mark.parametrize(
    "change",
    [
        {"bbox": [0, 0, 0, 2]},
        {"bbox": [0, 0, float("nan"), 2]},
        {"bbox": [0, 0, True, 2]},
        {"label": ""},
        {"score": float("inf")},
        {"id": 1},
        {"extra": "unsupported"},
    ],
)
def test_malformed_objects_are_rejected(change):
    with pytest.raises(ValidationError):
        evaluate_detections([row(predictions=[{**obj("a", score=0.9), **change}])])


@pytest.mark.parametrize(
    "settings",
    [
        {"iou_threshold": 0},
        {"min_precision": 1.1},
        {"score_thresholds": [True]},
        {"iou_thresholds": [0.5, 0.5]},
        {"classes": ["A", "A"]},
        {"ap_interpolation": "voc11"},
        {"score_threshold": float("nan")},
        {"typo": 1},
    ],
)
def test_invalid_parameter_contracts(settings):
    with pytest.raises(ValidationError):
        evaluate_detections([], **settings)


def test_bounds_and_training_leakage_reject_before_overlap_work(monkeypatch):
    import opencv_dataflow.detection_evaluation as module

    with pytest.raises(ValidationError):
        truth({"size": [100, 100], "objects": [obj(1, box=[99, 0, 10, 10])]})
    with pytest.raises(ValidationError):
        evaluate_detections([row(status="missing", predictions=[obj(1, score=0.9)])])
    with pytest.raises(ValidationError, match="overlap"):
        detection_config(
            {"block": "det", "training_groups": ["train"]},
            [{"id": "case", "capture_group": "train"}],
        )
    monkeypatch.setattr(module, "MAX_PAIRS", 1)
    monkeypatch.setattr(module, "box_iou", lambda *args: pytest.fail("Allocation before budget"))
    with pytest.raises(ValidationError, match="IoU pairs"):
        evaluate_detections([row(expected=[obj(1), obj(2)], predictions=[obj(1, score=0.9)])])


def capture_fixture():
    ticket = TicketClock().issue("source")
    context = {"ticket": ticket, "spatial": frame([100, 100], "source-grid")}
    value = {
        "model": "vision_collection_v1",
        "kind": "boxes",
        "count": 1,
        "origin": [0, 0],
        "image_size": [100, 100],
        "context": {"spatial": context["spatial"]},
        "items": [
            {
                "id": 0,
                "label": "A",
                "score": 0.9,
                "metrics": {},
                "geometry": {"bbox": [0, 0, 10, 10]},
            }
        ],
    }
    source = SimpleNamespace(
        current_emissions={"image": [ValuePacket(np.zeros((100, 100), np.uint8), context)]}
    )
    detector = SimpleNamespace(
        state="pass", current_emissions={"collection": [ValuePacket(value, copy.deepcopy(context))]}
    )
    graph = SimpleNamespace(blocks={"det": detector, "source": source})
    case = {"id": "case", "detection_truth": {"size": [100, 100], "objects": [obj("a")]}}
    return graph, detection_config({"block": "det"}), case, ticket


@pytest.mark.parametrize(
    "failure",
    ["ticket", "grid", "embedded_grid", "size", "origin", "rotated", "count", "missing", "error"],
)
def test_current_capture_identity_geometry_and_absent_event_guards(failure):
    graph, config, case, ticket = capture_fixture()
    det = graph.blocks["det"]
    packet = det.current_emissions["collection"][0]
    if failure == "ticket":
        packet.payload["ticket"] = TicketClock().issue("another")
    elif failure == "grid":
        packet.payload["spatial"]["to_root"][0][2] = 1
    elif failure == "embedded_grid":
        packet.value["context"]["spatial"] = frame([100, 100], "other")
    elif failure == "size":
        case["detection_truth"]["size"] = [101, 100]
    elif failure == "origin":
        packet.value["origin"] = [1, 0]
    elif failure == "rotated":
        packet.value["items"][0]["orientation_measured"] = True
    elif failure == "count":
        det.current_emissions["collection"].append(packet)
    elif failure == "missing":
        det.current_emissions = {}
    else:
        det.state, det.error = "error", "\U0001f6ab" * 10000
    result = capture_detections(graph, config, case, ticket, [], "source")
    assert result["status"] == ("missing" if failure == "missing" else "error")
    assert not result["predictions"]
    assert not evaluate_detections([result])["summary"]["passed"]


def test_valid_current_capture_and_lean_block_numeric_parity():
    graph, config, case, ticket = capture_fixture()
    result = capture_detections(graph, config, case, ticket, [], "source")
    assert result["status"] == "predicted"
    group = FlowGroup((ValuePacket(result, {}),), ("case",))
    block = builtin_registry().create("ml.evaluate_detections")
    try:
        baseline = block.execute({"records": group})
        assert baseline is not None, block.error
        assert baseline.passed and baseline.outputs["report"]["summary"]["tp"] == 1
        assert baseline.payloads["plot"]["spatial"]["root_id"].endswith(":pr")
        block.set_graphics_mode("off")
        lean = block.execute({"records": group})
        assert lean.outputs["report"] == baseline.outputs["report"]
        assert lean.outputs["plot"] is NO_VALUE and not lean.graphics
    finally:
        block.close()


def test_missing_and_error_images_do_not_invent_observed_dimensions():
    for state in ("missing", "error"):
        report = evaluate_detections([{**row(status=state), "size": None}])
        assert report["records"][0]["size"] is None and not report["summary"]["passed"]
    with pytest.raises(ValidationError, match="source image size"):
        evaluate_detections([{**row(), "size": None}])


@pytest.mark.local_examples
@pytest.mark.parametrize("backend", ["opencv", "default"])
def test_cpu_validation_does_not_probe_accelerators(monkeypatch, backend):
    def unexpected(backend):
        pytest.fail("CPU inference enumerated accelerator devices")

    monkeypatch.setattr(cv2.dnn, "getAvailableTargets", unexpected)
    with ProjectSession.open(HERE / "correct.json") as session:
        session.graph.blocks["infer"].parameters["backend"] = backend
        assert session.run_validation("objects")["summary"]["passed"]
        session.graph.run()
        calls = session.graph.blocks["infer"].last_result.outputs["receipt"]["calls"]
        assert calls[0]["requested_available_target_ids"] == [cv2.dnn.DNN_TARGET_CPU]
        assert calls[0]["requested_availability_scope"] == "requested_cpu_only"
        assert not calls[0]["native_placement_verified"]


@pytest.mark.local_examples
def test_cold_native_validation_worker_process_exits_cleanly():
    script = r"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.validation_editor import ValidationWorker
app = QApplication([])
registry = builtin_registry()
graph = Graph.load("examples/opencv tools/detector_evaluation/correct.json", registry)
worker = ValidationWorker(graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir)
worker.start()
assert worker.wait(20000)
assert not worker.error, worker.error
assert worker.report["summary"]["passed"]
worker.deleteLater()
graph.close()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del worker, graph, app
print("cold validation worker closed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=HERE.parents[2],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cold validation worker closed" in result.stdout


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "case,counts,map_value",
    [
        ("correct", (3, 0, 0), 1),
        ("duplicates", (3, 1, 0), 11 / 12),
        ("low_recall", (1, 0, 2), 0.25),
        ("localization", (2, 1, 1), 0.625),
        ("wrong_class", (2, 1, 1), 0.5),
        ("negative_false_positives", (0, 3, 0), None),
        ("decoder_error", (0, 0, 3), 0),
    ],
)
def test_native_detector_examples_match_independent_counts(case, counts, map_value):
    with ProjectSession.open(HERE / (case + ".json")) as session:
        report = session.run_validation("objects")
        metrics = report["detection"]
        summary = metrics["summary"]
        assert tuple(summary[k] for k in ("tp", "fp", "fn")) == counts
        assert summary["map"] == (pytest.approx(map_value) if map_value is not None else None)
        assert report["summary"]["passed"] == (case == "correct")
        assert metrics["complete"]
        assert session.graph.blocks["decode"].last_result is None
        assert session.graph.ticket_results.statistics()["total"] == 0
        json.dumps(report, allow_nan=False)


@pytest.mark.local_examples
def test_cancelled_set_missing_images_hash_errors_never_reuse_previous_detection():
    registry = builtin_registry()
    graph = Graph.load(HERE / "correct.json", registry)
    try:
        manifest = copy.deepcopy(graph.validation_sets[0])
        manifest["cases"].reverse()  # Process clean negative, then cancel before 3 labeled objects.
        completed = []
        report = run_set(
            graph.to_dict(),
            manifest,
            registry,
            HERE,
            cancelled=lambda: bool(completed),
            progress=completed.append,
        )
        assert report["summary"]["cancelled"] and not report["summary"]["passed"]
        assert report["detection"]["summary"]["fn"] == 3
        assert not report["detection"]["complete"]
        for key, value in (("image_sha256", "0" * 64), ("path", "missing.png")):
            changed = copy.deepcopy(manifest)
            changed["cases"][1][key] = value
            report = run_set(graph.to_dict(), changed, registry, HERE)
            assert report["cases"][1]["detection"]["status"] == "error"
            assert report["detection"]["summary"]["fn"] == 3
            assert not report["summary"]["passed"]
    finally:
        graph.close()


@pytest.mark.local_examples
def test_json_xml_portable_save_as_and_cli_reports(tmp_path, capsys):
    with ProjectSession.open(HERE / "correct.json") as session:
        original = session.graph.to_dict()
        xml = tmp_path / "program.xml"
        export_project_xml(original, xml)
        assert read_project_xml(xml) == original
        session.save(tmp_path / "program.json")
    with ProjectSession.open(tmp_path / "program.json") as session:
        assert (
            session.graph.validation_sets[0]["cases"][0]["detection_truth"]
            == original["validation_sets"][0]["cases"][0]["detection_truth"]
        )
        assert session.run_validation("objects")["summary"]["passed"]
    output = tmp_path / "report.json"
    args = [
        "run-validation",
        str(HERE / "correct.json"),
        "--set",
        "objects",
        "--report",
        str(output),
        "--summary",
    ]
    assert api_main(args) == 0
    assert json.loads(capsys.readouterr().out)["data"]["detection"]["tp"] == 3
    before = output.read_bytes()
    assert api_main(args) == 2
    capsys.readouterr()
    assert output.read_bytes() == before


def test_object_label_editor_validates_and_preserves_drafts(app):
    from opencv_dataflow.detection_validation_editor import DetectionTruthEditor

    original = {"size": [100, 100], "objects": [obj("a")]}
    dialog = DetectionTruthEditor(None, original, [100, 100])
    dialog.table.item(0, 4).setText("0")
    dialog.accept()
    assert "positive" in dialog.error.text()
    assert original["objects"][0]["bbox"][2] == 10
    dialog.table.item(0, 4).setText("9.5")
    dialog.accept()
    assert dialog.result_truth["objects"][0]["bbox"][2] == 9.5
    dialog.close()
    empty = DetectionTruthEditor(None, None, [100, 100])
    empty.unlabeled.setChecked(False)
    empty.accept()
    assert empty.result_truth == {"size": [100, 100], "objects": []}
    empty.close()


@pytest.mark.local_examples
def test_validation_window_detector_roundtrip_and_background_report(app):
    from test_editor import pump_until

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.validation_editor import ValidationManager

    window = MainWindow()
    window.replace_graph(Graph.load(HERE / "correct.json", window.registry))
    app.processEvents()
    dialog = ValidationManager(window)
    try:
        assert dialog.detector.enabled.isChecked() and not dialog.images.isColumnHidden(7)
        assert dialog.images.isColumnHidden(4)
        manifest = dialog.read_set()
        validate_sets([manifest])
        assert len(manifest["cases"][0]["detection_truth"]["objects"]) == 3
        dialog.run_images()
        pump_until(app, lambda: dialog.worker is None)
        assert dialog.report["summary"]["passed"]
        assert dialog.detection_report.classes.item(0, 2).text() == "2"
        assert dialog.result_tabs.isTabVisible(2)
        dialog.detector.enabled.setChecked(False)
        assert "detection" not in dialog.read_set() and dialog.images.isColumnHidden(7)
        assert (
            dialog.read_set()["cases"][0]["detection_truth"]
            == manifest["cases"][0]["detection_truth"]
        )
    finally:
        dialog.close()
        window.dirty = False
        window.close()


@pytest.mark.local_examples
def test_box_editor_accept_stages_changes_cancel_preserves_them(app, monkeypatch):
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.detection_validation_editor import DetectionTruthEditor
    from opencv_dataflow.validation_editor import ValidationManager

    window = MainWindow()
    window.replace_graph(Graph.load(HERE / "correct.json", window.registry))
    app.processEvents()
    original = copy.deepcopy(window.graph.validation_sets)
    dialog = ValidationManager(window)
    try:

        def accept_edit(editor):
            editor.table.item(0, 1).setText("edited")
            editor.accept()
            return editor.result()

        monkeypatch.setattr(DetectionTruthEditor, "exec", accept_edit)
        dialog.edit_object_labels(0, 7)
        accepted = dialog.read_set()["cases"][0]["detection_truth"]
        assert any(o["label"] == "edited" for o in accepted["objects"])
        assert window.graph.validation_sets == original

        def cancel_edit(editor):
            editor.table.item(0, 1).setText("cancelled")
            editor.reject()
            return editor.result()

        monkeypatch.setattr(DetectionTruthEditor, "exec", cancel_edit)
        dialog.edit_object_labels(0, 7)
        assert dialog.read_set()["cases"][0]["detection_truth"] == accepted
        assert window.graph.validation_sets == original
        assert dialog.save_set()
        assert window.graph.validation_sets[0]["cases"][0]["detection_truth"] == accepted
    finally:
        dialog.close()
        window.dirty = False
        window.close()


@pytest.mark.local_examples
def test_public_assets_all_projects_and_exact_regeneration(tmp_path):
    expected = json.loads((HERE / "truth.json").read_bytes())
    for name, digest in expected["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in HERE.glob("*.json"):
        if path.name != "truth.json":
            with ProjectSession.open(path) as session:
                assert validate_graph(session.graph)["valid"], path.name
    spec = importlib.util.spec_from_file_location("detector_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    generated = [p for p in tmp_path.rglob("*") if p.is_file()]
    for path in generated:
        assert path.read_bytes() == (HERE / path.relative_to(tmp_path)).read_bytes(), path
    before = {p: p.read_bytes() for p in generated}
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()
    assert all(path.read_bytes() == raw for path, raw in before.items())
