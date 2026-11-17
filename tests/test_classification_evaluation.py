"""Independent count references and real image-model validation, without training libraries."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.assistant_cli import main as api_main
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classification_evaluation import (
    capture_classification,
    classification_config,
    evaluate_classification,
    split_image_groups,
)
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.validation import run_set, validate_sets

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/classifier_evaluation"


def row(identity, truth="A", prediction="A", status="predicted", score=0.9):
    return {
        "id": str(identity),
        "expected": truth,
        "prediction": prediction,
        "status": status,
        "score": score,
        "score_kind": "agreement",
    }


def samples():
    return [
        {
            "id": f"{g}-{label}",
            "path": f"{g}-{label}.png",
            "class_label": label,
            "capture_group": str(g),
        }
        for g in range(8)
        for label in ("A", "B")
    ]


def test_independent_counts_include_rejections_missing_errors_and_unlabelled():
    rows = [
        row(0),
        row(1),
        row(2, prediction="B"),
        row(3, prediction=None, status="rejected"),
        row(4, prediction=None, status="missing"),
        row(5, prediction=None, status="error"),
        row(6, "B", "A"),
        row(7, "B", "B"),
        row(8, "B", "B"),
        row(9, None),
    ]
    report = evaluate_classification(rows, classes=["A", "B"])
    assert report["confusion"] == [[2, 1, 1, 1, 1], [1, 2, 0, 0, 0]]
    summary = report["summary"]
    assert (summary["total"], summary["labeled"], summary["unlabeled"]) == (10, 9, 1)
    assert summary["accuracy"] == pytest.approx(4 / 9)
    assert summary["coverage"] == pytest.approx(6 / 9)
    assert summary["selective_accuracy"] == pytest.approx(4 / 6)
    a, b = report["per_class"]
    assert (a["tp"], a["fp"], a["fn"], a["support"]) == (2, 1, 4, 6)
    assert (b["tp"], b["fp"], b["fn"], b["support"]) == (2, 1, 1, 3)
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(1 / 3)
    assert a["f1"] == pytest.approx(4 / 9)
    assert summary["balanced_accuracy"] == pytest.approx(0.5)
    assert summary["weighted_f1"] == pytest.approx((6 * 4 / 9 + 3 * 2 / 3) / 9)
    assert not summary["passed"]
    json.dumps(report, allow_nan=False)


def test_absent_expected_rows_are_missing_and_unexpected_ids_rejected():
    expected = [{"id": "0", "class_label": "A"}, {"id": "1", "class_label": "B"}]
    report = evaluate_classification([row(0)], expected_cases=expected)
    assert report["expected_ids_verified"]
    assert report["summary"]["accuracy"] == 0.5
    assert report["summary"]["status_counts"] == {"predicted": 1, "missing": 1}
    with pytest.raises(ValidationError, match="unexpected"):
        evaluate_classification([row(2)], expected_cases=expected)
    with pytest.raises(ValidationError, match="truth differs"):
        evaluate_classification([row(1)], expected_cases=expected)


def test_undefined_metrics_and_novel_predicted_classes_are_explicit():
    report = evaluate_classification([row(0, "A", "novel")], classes=["A", "unused"])
    assert report["classes"] == ["A", "unused", "novel"]
    assert report["per_class"][0]["precision"] is None
    assert report["per_class"][1]["recall"] is None
    assert report["macro"]["recall_classes"] == 1
    empty = evaluate_classification([])
    assert empty["summary"]["accuracy"] is None and not empty["summary"]["passed"]
    rejected = evaluate_classification([row(0, prediction=None, status="rejected")])
    assert rejected["summary"]["selective_accuracy"] is None


@pytest.mark.parametrize("direction,coverages", [("higher", [1, 0.5, 0]), ("lower", [0.5, 1, 1])])
def test_threshold_boundary_and_direction(direction, coverages):
    report = evaluate_classification(
        [row(0, score=0.2), row(1, score=0.8)],
        thresholds=[0.2, 0.8, 0.9],
        score_direction=direction,
    )
    assert [r["summary"]["coverage"] for r in report["thresholds"]] == coverages
    assert report["summary"]["accuracy"] == 1  # Base predictions are not changed.


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"status": "error"}, "disagree"),
        ({"score": float("nan")}, "finite"),
        ({"prediction": 1}, "text"),
        ({"group": []}, "text"),
    ],
)
def test_bad_records_are_rejected(change, pattern):
    with pytest.raises(ValidationError, match=pattern):
        evaluate_classification([{**row(0), **change}])


def test_duplicate_ids_mixed_score_meanings_and_invalid_thresholds_fail():
    with pytest.raises(ValidationError, match="Duplicate"):
        evaluate_classification([row(0), row(0)])
    with pytest.raises(ValidationError, match="mix score"):
        evaluate_classification([row(0), {**row(1), "score_kind": "distance"}], thresholds=[0.5])
    with pytest.raises(ValidationError, match="needs a score"):
        evaluate_classification([row(0, score=None)], thresholds=[0.5])
    for settings in (
        {"thresholds": [True]},
        {"thresholds": [0.5, 0.5]},
        {"min_accuracy": True},
        {"min_coverage": -0.1},
        {"typo": 2},
    ):
        with pytest.raises(ValidationError):
            evaluate_classification([row(0)], **settings)
    for state in ("missing", "error"):
        assert not evaluate_classification(
            [row(0, prediction=None, status=state)], min_accuracy=0, min_coverage=0
        )["summary"]["passed"]


def test_split_is_order_independent_keeps_whole_groups_and_all_classes():
    original = samples()
    first, report = split_image_groups(
        original, test_fraction=0.25, validation_fraction=0.25, seed=42
    )
    second, repeated = split_image_groups(
        original[::-1], test_fraction=0.25, validation_fraction=0.25, seed=42
    )
    assert first == second and report == repeated
    assert report["group_counts"] == {"train": 4, "validation": 2, "test": 2}
    assert not report["stratified"] and not report["image_hashes_verified"]
    ownership = {}
    for name, records in first.items():
        for record in records:
            assert ownership.setdefault(record["capture_group"], name) == name
    assert len(ownership) == 8 and sum(map(len, first.values())) == len(original)
    assert len({s["id"] for rows in first.values() for s in rows}) == len(original)


@pytest.mark.parametrize("duplicate", ["path", "image_sha256"])
def test_duplicate_image_identity_cannot_cross_groups(duplicate):
    records = samples()
    if duplicate == "path":
        records[0]["image_sha256"], records[2]["image_sha256"] = "a" * 64, "b" * 64
    records[0][duplicate] = records[2][duplicate] = "c" * 64
    with pytest.raises(ValidationError, match="crosses capture"):
        split_image_groups(records)


def test_split_class_loss_fraction_errors_and_training_leakage():
    records = [{**s, "class_label": s["capture_group"]} for s in samples()]
    with pytest.raises(ValidationError, match="lacks classes"):
        split_image_groups(records)
    for changes in (
        {"test_fraction": 1},
        {"test_fraction": True},
        {"test_fraction": -1},
        {"seed": 0.5},
        {"require_training_classes": "true"},
    ):
        with pytest.raises(ValidationError):
            split_image_groups(samples(), **changes)
    config = {"block": "predict", "training_groups": ["0"]}
    with pytest.raises(ValidationError, match="overlap"):
        classification_config(config, samples())


def test_metrics_block_group_input_lean_plot_and_numeric_parity():
    registry = builtin_registry()
    block = registry.create("ml.evaluate_classification")
    try:
        group = FlowGroup((ValuePacket(row(0), {}), ValuePacket(row(1), {})), ("0", "1"))
        result = block.execute({"records": group})
        assert result is not None, block.error
        assert result.outputs["report"]["summary"]["accuracy"] == 1
        assert result.outputs["plot"].dtype == np.uint8
        assert result.payloads["plot"]["spatial"]["root_id"].endswith(":confusion")
        baseline = result.outputs["report"]
        block.set_graphics_mode("off")
        result = block.execute({"records": group})
        assert result is not None, block.error
        assert result.outputs["report"] == baseline
        assert result.outputs["plot"] is NO_VALUE and not result.graphics
    finally:
        block.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("case", ["correct", "rejected", "mislabelled_training", "wrong_contract"])
def test_native_image_pipeline_and_independent_expected_confusions(case):
    expected = {
        "correct": ([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], 1, 1, True),
        "rejected": ([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], 0, 0, False),
        "mislabelled_training": ([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], 0, 1, False),
        "wrong_contract": ([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]], 0, 0, False),
    }[case]
    with ProjectSession.open(HERE / (case + ".json")) as session:
        report = session.run_validation("heldout")
        metrics = report["classification"]
        assert metrics["confusion"] == expected[0]
        assert metrics["summary"]["accuracy"] == expected[1]
        assert metrics["summary"]["coverage"] == expected[2]
        assert report["summary"]["passed"] == expected[3]
        assert metrics["complete"]
        assert session.graph.blocks["predict"].last_result is None
        assert session.graph.ticket_results.statistics()["total"] == 0
        assert all(r["image_sha256"] for r in report["cases"])
        json.dumps(report, allow_nan=False)


@pytest.mark.local_examples
def test_hash_mismatch_missing_file_and_cancel_never_reuse_retained_prediction():
    registry = builtin_registry()
    graph = Graph.load(HERE / "correct.json", registry)
    try:
        manifest = copy.deepcopy(graph.validation_sets[0])
        manifest["cases"][1]["image_sha256"] = "0" * 64
        report = run_set(graph.to_dict(), manifest, registry, HERE)
        assert [r["classification"]["status"] for r in report["cases"]] == ["predicted", "error"]
        assert report["classification"]["summary"]["accuracy"] == 0.5
        assert "pinned SHA" in report["cases"][1]["errors"][0]
        manifest["cases"][1]["path"] = "missing.png"
        report = run_set(graph.to_dict(), manifest, registry, HERE)
        assert report["cases"][1]["classification"]["prediction"] is None
        completed = []
        report = run_set(
            graph.to_dict(),
            graph.validation_sets[0],
            registry,
            HERE,
            cancelled=lambda: bool(completed),
            progress=completed.append,
        )
        assert len(report["cases"]) == 1
        assert report["summary"]["cancelled"] and not report["summary"]["passed"]
        assert not report["classification"]["complete"]
        assert report["classification"]["summary"]["status_counts"]["missing"] == 1
    finally:
        graph.close()


def test_capture_uses_current_events_not_last_results_and_checks_ticket():
    from types import SimpleNamespace

    from opencv_dataflow.tickets import TicketClock

    clock = TicketClock()
    ticket, wrong = clock.issue("source"), clock.issue("source")
    config = classification_config({"block": "predict"})
    block = SimpleNamespace(state="pass", current_emissions={}, last_result=object())
    graph = SimpleNamespace(blocks={"predict": block})
    case = {"id": "a", "class_label": "A"}
    assert capture_classification(graph, config, case, ticket, [])["status"] == "missing"
    block.current_emissions["labels"] = [ValuePacket(["A"], {"ticket": wrong})]
    result = capture_classification(graph, config, case, ticket, [])
    assert result["status"] == "error" and "ticket" in result["error"]
    block.current_emissions["labels"] = [ValuePacket(["A", "B"], {"ticket": ticket})]
    assert "one recognized" in capture_classification(graph, config, case, ticket, [])["error"]


@pytest.mark.local_examples
def test_json_xml_save_as_preserve_class_truth_groups_and_hashes(tmp_path):
    with ProjectSession.open(HERE / "correct.json") as session:
        definition = session.graph.to_dict()
        xml_path = tmp_path / "program.xml"
        export_project_xml(definition, xml_path)
        assert read_project_xml(xml_path) == definition
        session.save(tmp_path / "program.json")
    graph = Graph.load(tmp_path / "program.json", builtin_registry())
    try:
        sets = graph.validation_sets
        assert sets[0]["cases"][0]["class_label"] == "vertical"
        assert sets[0]["classification"] == definition["validation_sets"][0]["classification"]
        assert run_set(graph.to_dict(), sets[0], builtin_registry(), tmp_path)["summary"]["passed"]
    finally:
        graph.close()


@pytest.mark.local_examples
def test_cli_reports_exit_code_and_preserves_existing_output(tmp_path, capsys):
    report = tmp_path / "report.json"
    args = [
        "run-validation",
        str(HERE / "correct.json"),
        "--set",
        "heldout",
        "--report",
        str(report),
        "--summary",
    ]
    assert api_main(args) == 0
    assert json.loads(capsys.readouterr().out)["data"]["classification"]["accuracy"] == 1
    raw = report.read_bytes()
    assert api_main(args) != 0
    capsys.readouterr()
    assert report.read_bytes() == raw
    args[1], args[5] = str(HERE / "mislabelled_training.json"), str(tmp_path / "wrong.json")
    assert api_main(args) == 1
    capsys.readouterr()


@pytest.mark.local_examples
def test_validation_dialog_roundtrip_hidden_defaults_and_confusion(app):
    from test_editor import pump_until

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.validation_editor import ValidationManager

    window = MainWindow()
    window.replace_graph(Graph.load(HERE / "correct.json", window.registry))
    app.processEvents()
    dialog = ValidationManager(window)
    try:
        assert dialog.class_enabled.isChecked() and not dialog.images.isColumnHidden(4)
        assert dialog.images.isColumnHidden(6)
        dialog.class_advanced.setChecked(True)
        assert not dialog.images.isColumnHidden(6)
        manifest = dialog.read_set()
        validate_sets([manifest])
        assert manifest["cases"][0]["capture_group"] == "heldout"
        assert manifest["classification"]["score_path"] == "vote_fractions.0"
        dialog.run_images()
        pump_until(app, lambda: dialog.worker is None)
        assert dialog.report["summary"]["passed"]
        assert dialog.confusion.item(0, 0).text() == "1"
        assert dialog.class_metrics.item(0, 2).text() == "100.0%"
        dialog.class_enabled.setChecked(False)
        assert dialog.images.isColumnHidden(4)
        assert "classification" not in dialog.read_set()
    finally:
        dialog.close()
        window.dirty = False
        window.close()


@pytest.mark.local_examples
def test_public_assets_are_pinned_and_all_six_native_projects_validate():
    truth = json.loads((HERE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in HERE.glob("*.json"):
        if path.name == "truth.json":
            continue
        with ProjectSession.open(path) as session:
            assert validate_graph(session.graph)["valid"], path.name
            if path.stem in ("metrics", "split_groups"):
                session.graph.run()
                assert all(b.state != "error" for b in session.graph.blocks.values())


@pytest.mark.local_examples
def test_example_regeneration_is_exact_and_preserves_existing_files(tmp_path):
    spec = importlib.util.spec_from_file_location("evaluation_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    paths = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len([p for p in paths if p.suffix == ".json" and p.parent == tmp_path]) == 7
    before = {p: p.read_bytes() for p in paths}
    for path, raw in before.items():
        assert raw == (HERE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.local_examples
def test_training_feature_rows_match_declared_disjoint_capture_images():
    import cv2

    truth = json.loads((HERE / "truth.json").read_bytes())
    model = json.loads((HERE / "assets/classifier-none.json").read_bytes())
    swapped = json.loads((HERE / "assets/mislabelled.json").read_bytes())
    rows = []
    hog = builtin_registry().create("inspect.hog")
    try:
        for name in truth["training"]:
            image = cv2.imread(str(HERE / "assets" / name), cv2.IMREAD_GRAYSCALE)
            result = hog.execute({"image": image})
            assert result is not None, hog.error
            rows.append(np.asarray(result.outputs["descriptor"]).reshape(-1))
    finally:
        hog.close()
    np.testing.assert_array_equal(
        np.asarray(rows, np.float32), np.asarray(model["samples"], np.float32)
    )
    np.testing.assert_array_equal(model["samples"], swapped["samples"])
    assert [swapped["class_labels"][i] for i in swapped["responses"]] == (
        ["horizontal"] * 3 + ["vertical"] * 3
    )
    train_pixels = [cv2.imread(str(HERE / "assets" / n), 0).tobytes() for n in truth["training"]]
    heldout_pixels = [
        cv2.imread(str(p), 0).tobytes() for p in (HERE / "assets").glob("heldout*.png")
    ]
    assert len(set(train_pixels + heldout_pixels)) == 8


def test_score_capture_rejects_missing_stale_and_malformed_scores():
    from types import SimpleNamespace

    from opencv_dataflow.tickets import TicketClock

    clock = TicketClock()
    ticket, other = clock.issue("source"), clock.issue("source")
    config = classification_config(
        {"block": "predict", "score_output": "scores", "score_path": "winner"}
    )
    block = SimpleNamespace(
        state="pass",
        current_emissions={"labels": [ValuePacket(["A"], {"ticket": ticket})]},
        last_result=None,
    )
    graph = SimpleNamespace(blocks={"predict": block})
    case = {"id": "one", "class_label": "A"}
    for packets in (
        [],
        [ValuePacket({"winner": 0.9}, {"ticket": other})],
        [ValuePacket({"winner": None}, {"ticket": ticket})],
        [ValuePacket({"wrong": 0.9}, {"ticket": ticket})],
    ):
        block.current_emissions["scores"] = packets
        assert capture_classification(graph, config, case, ticket, [])["status"] == "error"
    block.current_emissions["scores"] = [ValuePacket({"winner": 0.9}, {"ticket": ticket})]
    result = capture_classification(graph, config, case, ticket, [])
    assert result["prediction"] == "A" and result["score"] == 0.9
    with pytest.raises(ValidationError):
        classification_config({"block": "predict", "score_kind": ""})


def test_long_unicode_errors_stay_bounded_and_keep_the_failed_case():
    from types import SimpleNamespace

    config = classification_config({"block": "predict"})
    graph = SimpleNamespace(
        blocks={"predict": SimpleNamespace(state="error", error="\U0001f6ab" * 10000)}
    )
    captured = capture_classification(graph, config, {"id": "a", "class_label": "A"}, None, [])
    report = evaluate_classification([captured])
    assert report["summary"]["status_counts"] == {"error": 1}
    assert len(json.dumps(captured["error"]).encode()) <= 4096
