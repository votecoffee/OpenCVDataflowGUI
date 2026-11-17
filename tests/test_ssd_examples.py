"""Pinned native SSD inference, independent preprocessing/geometry and portable workflows."""

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
BASE = ROOT / "examples/opencv tools/ssd_detection"
MODEL = BASE / ".local/ssd_resnet34/model.onnx"
SHA = "5ec1e058e666ef980a6213ac40088877865aafb2b6676b9af204b3828e4bb9d5"
CASES = {
    "coffee": (1, True),
    "selected_out": (0, False),
    "confidence_fail": (0, False),
    "count_fail": (1, False),
    "empty": (0, False),
}
native = pytest.mark.skipif(
    not MODEL.is_file(), reason="Prepare the pinned optional SSD model first"
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def reference():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == SHA
    image = cv2.imread(str(ROOT / "examples/assets/coffee.png"))
    # Independent native preprocessing: float bilinear resize, RGB, then 0..1 normalization.
    resized = cv2.resize(image.astype(np.float32), (1200, 1200))[:, :, ::-1]
    normalized = resized / np.float32(255) - np.array([0.485, 0.456, 0.406], np.float32)
    normalized /= np.array([0.229, 0.224, 0.225], np.float32)
    blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    net = cv2.dnn.readNetFromONNX(str(MODEL), cv2.dnn.ENGINE_NEW)
    net.setInput(blob, "image")
    names = ["bboxes", "labels", "scores"]
    return dict(zip(names, net.forward(names), strict=True))


@native
@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", CASES)
def test_native_projects_current_boxes_tickets_graphics_and_expected_failures(case, profile):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        decoder = session.graph.blocks["decode"]
        result = decoder.last_result
        assert result is not None, decoder.error
        assert (result.outputs["count"], result.passed) == CASES[case]
        assert result.outputs["report"]["context_checks"] == {
            "ticket_checked": True,
            "frame_checked": True,
        }
        source = session.graph.blocks["source"].last_result
        assert ticket_key(result.payloads["collection"]["ticket"]) == ticket_key(
            source.outputs["ticket"]
        )
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        overlay = session.graph.blocks["overlay"].last_result.outputs["image"]
        if profile == "debug" and CASES[case][0]:
            assert result.drawings and np.any(overlay != result.outputs["image"])
        if profile == "lean":
            assert not result.drawings
        receipt = session.graph.blocks["infer"].last_result.outputs["receipt"]["calls"][0]
        assert receipt["artifact"]["sha256"] == SHA
        assert receipt["requested_execution"]["engine"] == "new"
        assert (
            next(item for item in receipt["outputs"] if item["name"] == "labels")["dtype"]
            == "int64"
        )
        json.dumps(result.outputs["report"], allow_nan=False)


@native
def test_ssd_matches_independent_native_preprocessing_classes_and_continuous_edges(reference):
    with ProjectSession.open(BASE / "coffee.json") as session:
        session.graph.run()
        actual = session.graph.blocks["infer"].last_result.outputs["tensors"]
        # Small class IDs survive the public float32 tensor_v1 transport exactly.
        np.testing.assert_array_equal(selected_tensor(actual, "labels"), reference["labels"])
        np.testing.assert_allclose(
            selected_tensor(actual, "bboxes"), reference["bboxes"], atol=2e-5
        )
        np.testing.assert_allclose(
            selected_tensor(actual, "scores"), reference["scores"], atol=2e-5
        )
        (item,) = session.graph.blocks["decode"].last_result.outputs["collection"]["items"]
        ids = np.flatnonzero((reference["labels"][0] == 42) & (reference["scores"][0] > 0.4))
        assert len(ids) == 1
        left, top, right, bottom = reference["bboxes"][0, ids[0]].astype(float) * [
            600,
            400,
            600,
            400,
        ]
        np.testing.assert_allclose(
            item["geometry"]["bbox"], [left, top, right - left, bottom - top], atol=0.01
        )
        assert item["class_id"] == 41 and item["label"] == "cup" and item["score"] > 0.45
        assert not item["orientation_measured"]
        # Independent manual truth, not a box generated from this model's output.
        intersection = max(0, min(right, 415) - max(left, 170)) * max(
            0, min(bottom, 310) - max(top, 18)
        )
        assert intersection / ((right - left) * (bottom - top) + 245 * 292 - intersection) > 0.6


@native
@pytest.mark.parametrize(
    "case,passed,tp,fn",
    [
        ("coffee", True, 1, 0),
        ("selected_out", False, 0, 1),
        ("confidence_fail", False, 0, 1),
        ("count_fail", True, 1, 0),
        ("empty", True, 0, 0),
    ],
)
def test_saved_validation_distinguishes_inspection_gate_from_detector_localization(
    case, passed, tp, fn
):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        report = session.run_validation("cup")
        assert report["summary"]["passed"] is passed
        summary = report["detection"]["summary"]
        assert (summary["tp"], summary["fp"], summary["fn"]) == (tp, 0, fn)


@native
@pytest.mark.parametrize("field,value", [("width", 300), ("engine", "classic")])
def test_bound_settings_reject_changes_before_native_inference_and_recover(field, value):
    with ProjectSession.open(BASE / "coffee.json") as session:
        tool = session.graph.blocks["infer"]
        session.graph.run()
        original = tool.parameters[field]
        tool.parameters[field] = value
        session.graph.run()
        assert tool.state == "error" and not tool.current_emissions
        assert "pinned ONNX contract" in tool.error
        assert not session.graph.blocks["decode"].current_emissions
        tool.parameters[field] = original
        session.graph.run()
        assert session.graph.blocks["decode"].last_result.outputs["count"] == 1


@native
def test_this_pinned_ssd_export_is_rejected_by_the_explicit_native_classic_importer():
    # This unsupported importer can produce a native Windows fault diagnostic.
    # Probe it in a disposable process, separate from all qualified New-engine runs.
    script = f"""
import cv2
try:
    cv2.dnn.readNetFromONNX({str(MODEL)!r}, cv2.dnn.ENGINE_CLASSIC)
except cv2.error:
    print("SSD Classic rejected", flush=True)
else:
    raise AssertionError("Requalify this model: native Classic behavior has changed")
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSD Classic rejected" in result.stdout


@native
def test_project_xml_and_portable_save_as_preserve_pinned_model_and_validation(tmp_path):
    with ProjectSession.open(BASE / "coffee.json") as session:
        definition = session.graph.to_dict()
        xml = tmp_path / "program.xml"
        export_project_xml(definition, xml)
        assert read_project_xml(xml) == definition
        session.save(tmp_path / "program.json")
    copied = tmp_path / ".local/ssd_resnet34/model.onnx"
    assert copied.stat().st_size == 80_366_315
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == SHA
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    with ProjectSession.open(tmp_path / "program.json") as session:
        assert session.run_validation("cup")["summary"]["passed"]


def test_preparation_rejects_different_weights_without_creating_bundle(tmp_path, monkeypatch):
    prepare = module("ssd_prepare", BASE / "prepare_model.py")
    source = tmp_path / "wrong.onnx"
    source.write_bytes(b"unchanged wrong model")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="pinned"):
        prepare.prepare(source, directory=tmp_path / "bundle")
    assert source.read_bytes() == b"unchanged wrong model" and not (tmp_path / "bundle").exists()


def test_project_regeneration_is_exact_and_preserves_existing_projects(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("ssd_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in BASE.glob("*.json"):
        assert json.loads(path.read_bytes()) == json.loads((tmp_path / path.name).read_bytes())
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()
    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}


@native
def test_existing_bundle_verifies_without_network_or_rewrite(monkeypatch):
    prepare = module("ssd_prepare", BASE / "prepare_model.py")
    before = {
        name: (MODEL.parent / name).stat().st_mtime_ns for name in ("model.onnx", "contract.json")
    }
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert prepare.prepare() == MODEL.parent
    assert before == {name: (MODEL.parent / name).stat().st_mtime_ns for name in before}


@native
def test_cold_native_ssd_validation_worker_shuts_down_cleanly():
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
graph = Graph.load("examples/opencv tools/ssd_detection/coffee.json", registry)
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
print("SSD CPU worker closed", flush=True)
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
    assert "SSD CPU worker closed" in result.stdout
