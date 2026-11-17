"""Pinned learned YOLO heads, independent grid/letterbox/NMS references and saved workflows."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/opencv tools/yolo_detection"
RAW = BASE / ".local/yolox_raw/model.onnx"
DERIVED = BASE / ".local/yolox_decoded/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
CASES = TRUTH["cases"]
native = pytest.mark.skipif(
    not RAW.is_file() or not DERIVED.is_file(),
    reason="Prepare the pinned optional raw/derived YOLOX models first",
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def references():
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == (
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
    )
    # Independent fixed-image resize/padding declarations; do not call app preprocessing.
    cases = {
        "coffee": (ROOT / "examples/assets/coffee.png", (640, 427), (0, 106)),
        "portrait": (BASE / "assets/coffee_portrait.png", (496, 640), (72, 0)),
        "coffee_new": (ROOT / "examples/assets/coffee.png", (640, 427), (0, 106)),
    }
    net = cv2.dnn.readNetFromONNX(str(RAW), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    result = {}
    for name, (path, resized_size, offset) in cases.items():
        if name == "coffee_new":
            # Decode equivalence uses the same engine. Cross-engine scored boxes
            # have a separate numerical tolerance check below.
            net = cv2.dnn.readNetFromONNX(str(RAW), cv2.dnn.ENGINE_NEW)
        image = cv2.imread(str(path))
        rgb = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, resized_size)
        canvas = np.full((640, 640, 3), 114, np.float32)
        left, top = offset
        canvas[top : top + resized_size[1], left : left + resized_size[0]] = resized
        net.setInput(np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), "images")
        raw = net.forward("output")[0].copy()
        boxes = raw[:, :4].astype(float)
        grid, strides = [], []
        for stride in (8, 16, 32):
            for y in range(640 // stride):
                for x in range(640 // stride):
                    grid.append((x, y))
                    strides.append(stride)
        strides = np.asarray(strides)[:, None]
        boxes[:, :2] = (boxes[:, :2] + grid) * strides
        boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * strides
        combined = raw[:, 4:5] * raw[:, 5:]
        decoded = np.column_stack((boxes, combined))
        boxes[:, :2] -= boxes[:, 2:4] / 2
        height, width = image.shape[:2]
        boxes[:, :2] -= offset
        boxes *= [width / resized_size[0], height / resized_size[1]] * 2
        result[name] = {
            "raw": raw,
            "decoded": decoded,
            "boxes": boxes,
            "scores": combined.max(axis=1),
            "classes": combined.argmax(axis=1),
            "size": [width, height],
        }
    return result


def native_selected(reference, options):
    boxes, scores, classes = (reference[name] for name in ("boxes", "scores", "classes"))
    indices = np.flatnonzero(scores > options.get("confidence_threshold", 0.35))
    if options.get("classes", [41]):
        indices = indices[np.isin(classes[indices], options.get("classes", [41]))]
    if options.get("nms_enabled", True):
        if options.get("class_agnostic_nms", False):
            keep = cv2.dnn.NMSBoxes(
                boxes[indices].tolist(),
                scores[indices].tolist(),
                0.35,
                options.get("nms_iou", 0.45),
            )
        else:
            keep = cv2.dnn.NMSBoxesBatched(
                boxes[indices].tolist(),
                scores[indices].tolist(),
                classes[indices].tolist(),
                0.35,
                options.get("nms_iou", 0.45),
            )
        indices = indices[np.asarray(keep, dtype=int).reshape(-1)]
    return sorted(indices.tolist(), key=lambda i: (-float(scores[i]), i))


@native
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_current_native_events_judgments_context_primary_pixels_and_graphics(case, profile):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        tool = session.graph.blocks["decode"]
        if case == "wrong_layout":
            assert tool.state == "error" and not tool.current_emissions
            assert "row width" in tool.error
            return
        result = tool.last_result
        assert result is not None, tool.error
        options = CASES[case]
        assert (result.outputs["count"], result.passed) == (options["count"], options["passed"])
        assert result.outputs["report"]["context_checks"] == {
            "ticket_checked": True,
            "frame_checked": True,
        }
        source = session.graph.blocks["source"].last_result
        context = result.payloads["collection"]
        assert ticket_key(context["ticket"]) == ticket_key(source.outputs["ticket"])
        assert context["spatial"]["image_size"] == list(source.outputs["image"].shape[1::-1])
        np.testing.assert_array_equal(source.outputs["image"], result.outputs["image"])
        overlay = session.graph.blocks["overlay"].last_result.outputs["image"]
        if profile == "debug" and options["count"]:
            assert result.drawings and np.any(overlay != result.outputs["image"])
        if profile == "lean":
            assert not result.drawings
        receipt = session.graph.blocks["infer"].last_result.outputs["receipt"]["calls"][0]
        assert receipt["configuration_verified"] and receipt["external_preprocessing_verified"]
        assert receipt["requested_execution"]["engine"] == (
            "new" if options.get("kind") == "decoded" else "classic"
        )
        json.dumps(result.outputs["report"], allow_nan=False)


@native
@pytest.mark.parametrize(
    "case",
    [
        "raw_coffee",
        "decoded_rows",
        "decoded_channels",
        "portrait",
        "duplicates",
        "class_aware",
        "class_agnostic",
    ],
)
def test_source_boxes_scores_classes_and_original_rows_match_independent_native_reference(
    case, references
):
    options = CASES[case]
    reference = references[
        "portrait"
        if case == "portrait"
        else "coffee_new"
        if options.get("kind") == "decoded"
        else "coffee"
    ]
    with ProjectSession.open(BASE / (case + ".json")) as session:
        session.graph.run()
        inference = session.graph.blocks["infer"].last_result
        if options.get("kind") == "decoded":
            rows = selected_tensor(inference.outputs["tensors"], "decoded_rows")[0]
            channels = selected_tensor(inference.outputs["tensors"], "decoded_channels")[0]
            np.testing.assert_array_equal(rows, channels.T)
            np.testing.assert_allclose(rows, reference["decoded"], atol=0.002, rtol=2e-5)
        else:
            np.testing.assert_allclose(
                selected_tensor(inference.outputs["tensors"], "output")[0],
                reference["raw"],
                atol=1e-6,
            )
        (width, height) = reference["size"]
        expected = native_selected(reference, options)
        result = session.graph.blocks["decode"].last_result
        items = result.outputs["collection"]["items"]
        assert [item["id"] for item in items] == expected
        for item, index in zip(items, expected, strict=True):
            x, y, w, h = reference["boxes"][index]
            left, top = np.clip([x, y], 0, [width, height])
            right, bottom = np.clip([x + w, y + h], 0, [width, height])
            np.testing.assert_allclose(
                item["geometry"]["bbox"], [left, top, right - left, bottom - top], atol=0.01
            )
            assert item["score"] == pytest.approx(reference["scores"][index], abs=2e-6)
            assert item["class_id"] == reference["classes"][index]
            assert not item["orientation_measured"]


@native
@pytest.mark.parametrize("case", CASES)
def test_validation_uses_hand_marked_truth_and_retains_failures_duplicates_and_errors(case):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        report = session.run_validation("cup")
        passed = case in (
            "raw_coffee",
            "decoded_rows",
            "decoded_channels",
            "portrait",
            "class_aware",
            "empty",
        )
        assert report["summary"]["passed"] is passed, report["summary"]
        summary = report["detection"]["summary"]
        if case == "duplicates":
            assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 7, 0)
            assert summary["precision"] == 0.125
        elif case == "class_aware":
            assert (summary["tp"], summary["fp"], summary["fn"]) == (2, 0, 0)
        elif case == "class_agnostic":
            assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 0, 1)
        elif case == "wrong_layout":
            assert summary["fn"] == 1 and report["summary"]["execution_errors"]
        elif not passed:
            assert (summary["tp"], summary["fp"], summary["fn"]) == (0, 0, 1)
        elif case == "empty":
            assert summary["support"] == 0 and summary["fp"] == 0
        else:
            assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 0, 0)


@native
def test_confidence_boundary_is_strict_and_lower_neighbor_recovers_one_cup(references):
    highest = max(references["coffee"]["scores"][references["coffee"]["classes"] == 41])
    with ProjectSession.open(BASE / "raw_coffee.json") as session:
        tool = session.graph.blocks["decode"]
        tool.parameters["confidence_threshold"] = float(highest)
        session.graph.run()
        assert tool.last_result.outputs["count"] == 0
        tool.parameters["confidence_threshold"] = float(np.nextafter(highest, np.float32(-np.inf)))
        session.graph.run()
        assert tool.last_result.outputs["count"] == 1


@native
def test_classic_and_new_cpu_paths_retain_the_same_scored_boxes_and_classes(references):
    a, b = references["coffee"], references["coffee_new"]
    first, second = native_selected(a, {"classes": []}), native_selected(b, {"classes": []})
    assert first == second
    np.testing.assert_array_equal(a["classes"][first], b["classes"][second])
    np.testing.assert_allclose(a["scores"][first], b["scores"][second], atol=2e-5)
    np.testing.assert_allclose(a["boxes"][first], b["boxes"][second], atol=0.01)


@native
def test_layout_error_after_valid_activation_has_no_retained_output_event_and_recovers():
    with ProjectSession.open(BASE / "decoded_channels.json") as session:
        tool = session.graph.blocks["decode"]
        session.graph.run()
        assert tool.last_result.outputs["count"] == 1
        tool.parameters["layout"] = "rows"
        session.graph.run()
        assert tool.state == "error" and not tool.current_emissions
        tool.parameters["layout"] = "channels_first"
        session.graph.run()
        assert tool.last_result.outputs["count"] == 1


@native
@pytest.mark.parametrize("case", ["raw_coffee", "decoded_channels"])
def test_portable_save_as_preserves_pinned_model_credits_xml_and_validation(case, tmp_path):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        definition = session.graph.to_dict()
        xml = tmp_path / "project.xml"
        export_project_xml(definition, xml)
        assert read_project_xml(xml) == definition
        session.save(tmp_path / "project.json")
    kind = "raw" if case == "raw_coffee" else "decoded"
    source = RAW if kind == "raw" else DERIVED
    assert (tmp_path / f".local/yolox_{kind}/model.onnx").read_bytes() == source.read_bytes()
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    with ProjectSession.open(tmp_path / "project.json") as session:
        assert session.run_validation("cup")["summary"]["passed"]


@native
def test_export_is_exact_and_all_original_learned_initializers_and_nodes_are_unchanged():
    onnx = pytest.importorskip("onnx")
    prepare = module("yolo_prepare", BASE / "prepare_models.py")
    derived = prepare.decoded_export(RAW.read_bytes())
    assert derived == DERIVED.read_bytes()
    assert len(derived) == prepare.DERIVED_SIZE
    assert hashlib.sha256(derived).hexdigest() == prepare.DERIVED_SHA256
    original = onnx.load(str(RAW))
    converted = onnx.load_model_from_string(derived)
    count = len(original.graph.node)
    assert list(original.graph.node) == list(converted.graph.node[:count])
    original_weights = {item.name: item.SerializeToString() for item in original.graph.initializer}
    kept = {
        item.name: item.SerializeToString()
        for item in converted.graph.initializer
        if item.name in original_weights
    }
    assert original_weights == kept
    assert "Modified by OpenCVDataflowGUI" in converted.doc_string


def test_wrong_weights_reject_before_export_download_or_publication(tmp_path, monkeypatch):
    prepare = module("yolo_prepare", BASE / "prepare_models.py")
    source = tmp_path / "other.onnx"
    source.write_bytes(b"preserve me")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="pinned YOLOX"):
        prepare.prepare(source, root=tmp_path / "models")
    assert not (tmp_path / "models").exists() and source.read_bytes() == b"preserve me"


@native
def test_existing_preparation_is_idempotent_and_offline(monkeypatch):
    pytest.importorskip("onnx")
    prepare = module("yolo_prepare", BASE / "prepare_models.py")
    paths = [RAW, DERIVED, RAW.with_name("contract.json"), DERIVED.with_name("contract.json")]
    before = [(path, path.stat().st_mtime_ns) for path in paths]
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert prepare.prepare() == BASE / ".local"
    assert before == [(path, path.stat().st_mtime_ns) for path in paths]


def test_exact_project_and_crop_regeneration_preserves_existing_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("yolo_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in BASE.glob("*.json"):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    assert (tmp_path / "assets/coffee_portrait.png").read_bytes() == (
        BASE / "assets/coffee_portrait.png"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


@native
def test_cold_decoded_validation_worker_exits_cleanly_without_accelerator_discovery():
    script = r"""
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
import cv2
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.validation_editor import ValidationWorker
def unexpected(*args):
    raise AssertionError("CPU inference probed accelerators")
cv2.dnn.getAvailableTargets = unexpected
app = QApplication([])
registry = builtin_registry()
graph = Graph.load("examples/opencv tools/yolo_detection/decoded_channels.json", registry)
worker = ValidationWorker(graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir)
worker.start()
assert worker.wait(25000)
assert not worker.error, worker.error
assert worker.report["summary"]["passed"]
worker.deleteLater()
graph.close()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del worker, graph, app
print("decoded YOLO CPU worker closed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "decoded YOLO CPU worker closed" in result.stdout
