"""Learned tile detections, independent native coordinates, saved graphs and failure controls."""

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
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.toolbox import block_entry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/opencv tools/detection_tile_fusion"
MODEL = BASE / ".local/yolox_raw/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
CASES = TRUTH["cases"]
native = pytest.mark.skipif(not MODEL.is_file(), reason="Prepare the optional pinned YOLOX model")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def references():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == (
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
    )
    image = cv2.imread(str(ROOT / "examples/assets/coffee.png"))
    net = cv2.dnn.readNetFromONNX(str(MODEL), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    result = {}
    # These dimensions/offsets are fixed independently of application preprocessing.
    for name, tile, resized, offset, scale, origin in (
        ("left", image[:380, 80:520], (640, 553), (0, 43), 1, (80, 0)),
        ("right", image[:, 140:560], (640, 610), (0, 15), 1, (140, 0)),
        ("half", cv2.resize(image[:, 140:560], (210, 200)), (640, 610), (0, 15), 2, (140, 0)),
    ):
        rgb = cv2.cvtColor(tile.astype(np.float32), cv2.COLOR_BGR2RGB)
        canvas = np.full((640, 640, 3), 114, np.float32)
        left, top = offset
        canvas[top : top + resized[1], left : left + resized[0]] = cv2.resize(rgb, resized)
        net.setInput(np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), "images")
        raw = net.forward("output")[0].copy()
        grid, strides = [], []
        for stride in (8, 16, 32):
            for y in range(640 // stride):
                for x in range(640 // stride):
                    grid.append((x, y))
                    strides.append(stride)
        stride = np.asarray(strides)[:, None]
        centers = (raw[:, :2].astype(float) + grid) * stride
        sizes = np.exp(raw[:, 2:4].astype(float)) * stride
        combined = raw[:, 4:5] * raw[:, 5:]
        scores, classes = combined.max(axis=1), combined.argmax(axis=1)
        boxes = np.column_stack((centers - sizes / 2, sizes))
        boxes[:, :2] -= offset
        boxes *= [tile.shape[1] / resized[0], tile.shape[0] / resized[1]] * 2
        indices = np.flatnonzero((scores > 0.35) & (classes == 41))
        keep = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(), 0.35, 0.45)
        selected = indices[np.asarray(keep, dtype=int).reshape(-1)]
        assert len(selected) == 1
        i = selected[0]
        # Decoder clips local rectangles; edge mapping then uses scale*edges+origin.
        x, y, w, h = boxes[i]
        xyxy = np.clip([x, y, x + w, y + h], 0, [tile.shape[1], tile.shape[0]] * 2)
        xyxy *= scale
        xyxy += [*origin, *origin]
        result[name] = {"box": np.r_[xyxy[:2], xyxy[2:] - xyxy[:2]], "score": float(scores[i])}
    return result


@native
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_native_outcomes_graphics_context_and_readonly_source(case, profile):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        tool = session.graph.blocks["fuse"]
        if case == "missing_tile":
            assert tool.state == "error" and not tool.current_emissions
            assert "tile count" in tool.error
            return
        result = tool.last_result
        assert result is not None, tool.error
        assert (result.outputs["count"], result.passed) == (
            CASES[case]["count"],
            CASES[case]["passed"],
        )
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(source.outputs["image"], result.outputs["image"])
        context = result.payloads["collection"]
        assert context["spatial"] == source.payloads["image"]["spatial"]
        assert ticket_key(context["ticket"]) == ticket_key(source.outputs["ticket"])
        report = result.outputs["report"]
        assert len(report["tiles"]) == 2 and report["expected_tile_count_checked"]
        assert report["raw_detections"] == (0 if case == "empty" else 2)
        if case == "failed_tile":
            assert [r["code"] for r in result.fail_reasons] == ["tile_detections.failed_tile"]
        if case == "count_fail":
            assert [r["code"] for r in result.fail_reasons] == ["tile_detections.count"]
        if profile == "debug" and result.outputs["count"]:
            assert result.drawings
            overlay = session.graph.blocks["overlay"].last_result.outputs["image"]
            assert np.any(overlay != result.outputs["image"])
        if profile == "lean":
            assert not result.drawings
        json.dumps(report, allow_nan=False)


@native
@pytest.mark.parametrize("case", ["coffee", "resized", "duplicates"])
def test_cross_tile_winner_matches_independently_prepared_native_rows_and_edge_mapping(
    case, references
):
    observations = [references["left"], references["half" if case == "resized" else "right"]]
    boxes = [row["box"].tolist() for row in observations]
    scores = [row["score"] for row in observations]
    expected = (
        sorted(range(2), key=lambda i: -scores[i])
        if case == "duplicates"
        else np.asarray(cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5)).reshape(-1).tolist()
    )
    with ProjectSession.open(BASE / (case + ".json")) as session:
        session.graph.run()
        items = session.graph.blocks["fuse"].last_result.outputs["collection"]["items"]
        assert len(items) == len(expected)
        for item, i in zip(items, expected, strict=True):
            np.testing.assert_allclose(item["geometry"]["bbox"], boxes[i], atol=0.01)
            assert item["score"] == pytest.approx(scores[i], abs=2e-6)
            assert item["provenance"]["tile_index"] == i
            assert item["metrics"]["angle"] is None and not item["orientation_measured"]


@native
@pytest.mark.parametrize("case", CASES)
def test_independent_ground_truth_validation_preserves_duplicate_false_positive_and_missing_event(
    case,
):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        report = session.run_validation("cup")
        assert report["summary"]["passed"] is (
            case in ("coffee", "resized", "count_fail", "failed_tile", "empty")
        )
        detection = report["detection"]["summary"]
        if case == "duplicates":
            assert (detection["tp"], detection["fp"], detection["fn"]) == (1, 1, 0)
        elif case in ("missing_tile", "selected_out"):
            assert (detection["tp"], detection["fp"], detection["fn"]) == (0, 0, 1)
        elif case == "empty":
            assert detection["support"] == 0 and detection["fp"] == 0
        else:
            assert (detection["tp"], detection["fp"], detection["fn"]) == (1, 0, 0)
        if case == "missing_tile":
            assert report["summary"]["execution_errors"]


@native
def test_serial_parallel_portable_save_as_xml_and_error_recovery(tmp_path):
    with ProjectSession.open(BASE / "coffee.json") as session:
        session.graph.set_branch_workers(1)
        session.graph.run()
        expected = session.graph.blocks["fuse"].last_result.outputs["collection"]["items"]
        session.graph.set_branch_workers(2)
        session.graph.run()
        tool = session.graph.blocks["fuse"]
        assert expected == tool.last_result.outputs["collection"]["items"]
        tool.parameters["expected_tiles"] = 3
        session.graph.run()
        assert tool.state == "error" and not tool.current_emissions
        tool.parameters["expected_tiles"] = 2
        session.graph.run()
        assert tool.last_result.passed and tool.current_emissions
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
        assert read_project_xml(tmp_path / "program.xml") == definition
        session.save(tmp_path / "program.json")
    assert (tmp_path / ".local/yolox_raw/model.onnx").read_bytes() == MODEL.read_bytes()
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    with ProjectSession.open(tmp_path / "program.json") as session:
        assert session.run_validation("cup")["summary"]["passed"]


def test_generated_projects_and_image_match_and_existing_projects_are_preserved(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("tile_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in BASE.glob("*.json"):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    assert (tmp_path / "assets/black.png").read_bytes() == (BASE / "assets/black.png").read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_pinned_preparation_rejects_wrong_weights_without_download(tmp_path, monkeypatch):
    prepare = module("tile_prepare", BASE / "prepare_tile_model.py")
    source = tmp_path / "wrong.onnx"
    source.write_bytes(b"preserve")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="pinned"):
        prepare.prepare(source, directory=tmp_path / "bundle")
    assert source.read_bytes() == b"preserve" and not (tmp_path / "bundle").exists()


@native
def test_prepared_bundle_is_idempotent_offline_and_rejects_changed_settings(tmp_path, monkeypatch):
    prepare = module("tile_prepare", BASE / "prepare_tile_model.py")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    paths = [MODEL, MODEL.with_name("contract.json")]
    before = [path.stat().st_mtime_ns for path in paths]
    assert prepare.prepare() == MODEL.parent
    assert before == [path.stat().st_mtime_ns for path in paths]
    destination = tmp_path / "model"
    prepare.prepare(MODEL, directory=destination)
    contract_path = destination / "contract.json"
    contents = json.loads(contract_path.read_bytes())
    contents["preprocessing"]["images"]["color_format"] = "bgr"
    contract_path.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        prepare.prepare(directory=destination)


def test_toolbox_matches_opencv_dnn_group():
    from opencv_dataflow.detection_fusion import FuseTileDetections

    assert block_entry(FuseTileDetections).path[:2] == ("OpenCV", "Deep Neural Networks (dnn)")


@native
def test_cold_parallel_cpu_worker_closes_without_accelerator_discovery():
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
graph = Graph.load("examples/opencv tools/detection_tile_fusion/coffee.json", registry)
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
print("parallel tile fusion worker closed", flush=True)
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
    assert "parallel tile fusion worker closed" in result.stdout
