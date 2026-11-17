"""Saved native detector sequences with independent source truth and explicit identity metrics."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/opencv tools/detection_tracking"
MODEL = BASE / ".local/yolox_raw/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
EXPECTED = {
    "crossing": (18, 0, 0, 0, 0),
    "occluded_swap": (14, 0, 0, 0, 0),
    "motion_only": (14, 0, 0, 2, 0),
    "detection_gap": (6, 0, 3, 0, 1),
    "expire_reentry": (6, 0, 3, 1, 1),
    "yolox_coffee": (5, 0, 0, 0, 0),
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", EXPECTED)
def test_independent_sequence_metrics_count_identity_switches_fragments_and_detector_misses(
    case, profile
):
    if case == "yolox_coffee" and not MODEL.is_file():
        pytest.skip("Prepare the optional pinned YOLOX model")
    evaluator = module("tracking_examples", BASE / "evaluate_examples.py")
    report = evaluator.evaluate(case, profile)
    summary = report["evaluation"]["summary"]
    assert (
        tuple(summary[k] for k in ("tp", "fp", "fn", "id_switches", "fragmentations"))
        == EXPECTED[case]
    )
    assert summary["mean_matched_iou"] > 0.9
    if case == "occluded_swap":
        assert report["predicted_counts"][3:5] == [2, 2]
    if case == "detection_gap":
        assert report["predicted_counts"][3:6] == [1, 1, 1]
    if case == "expire_reentry":
        assert report["predicted_counts"][3:6] == [1, 0, 0]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_current_ticket_frames_graphics_and_original_pixels_are_preserved(profile):
    with ProjectSession.open(BASE / "crossing.json") as session:
        session.graph.set_run_profile(profile)
        for _ in range(3):
            session.graph.run()
            source = session.graph.blocks["source"].last_result
            described = session.graph.blocks["appearance"].last_result
            track = session.graph.blocks["track"].last_result
            assert ticket_key(track.payloads["tracks"]["ticket"]) == ticket_key(
                source.outputs["ticket"]
            )
            assert track.payloads["tracks"]["spatial"] == source.payloads["image"]["spatial"]
            np.testing.assert_array_equal(described.outputs["image"], source.outputs["image"])
            overlay = session.graph.blocks["filmstrip"].last_result.outputs["image"]
            if profile == "debug":
                assert (
                    track.drawings
                    and described.drawings
                    and np.any(overlay != source.outputs["image"])
                )
            else:
                assert not track.drawings and not described.drawings
            assert all(item["appearance_available"] for item in track.outputs["tracks"]["items"])


def test_every_project_is_static_valid_and_has_portable_attribution():
    for case in EXPECTED:
        with ProjectSession.open(BASE / (case + ".json")) as session:
            assert validate_graph(session.graph)["valid"]
            assert "scikit-image CC0" in session.graph.blocks["attribution"].description
            assert session.graph.program["assets"] == ["SOURCES.md", "MODEL-LICENSE.txt"]


def test_generator_preserves_existing_and_reproduces_all_projects_truth_and_assets(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("tracking_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for original in [*BASE.glob("*.json"), *(BASE / "assets").glob("*.png")]:
        assert (tmp_path / original.relative_to(BASE)).read_bytes() == original.read_bytes(), (
            original
        )
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()
    for frames in TRUTH["sequences"].values():
        for row in frames:
            assert hashlib.sha256((BASE / row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.skipif(not MODEL.is_file(), reason="Prepare optional pinned model")
def test_pinned_dnn_project_save_as_xml_and_reopen_carry_model_pixels_and_credits(tmp_path):
    with ProjectSession.open(BASE / "yolox_coffee.json") as session:
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
        assert read_project_xml(tmp_path / "program.xml") == definition
        session.save(tmp_path / "program.json")
    assert (tmp_path / ".local/yolox_raw/model.onnx").read_bytes() == MODEL.read_bytes()
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    with ProjectSession.open(tmp_path / "program.json") as session:
        session.graph.run()
        tool = session.graph.blocks["track"]
        assert tool.last_result is not None, tool.error
        assert tool.last_result.outputs["count"] == 1
        item = tool.last_result.outputs["tracks"]["items"][0]
        assert item["class_id"] == 41 and not item["angle_available"]
        assert item["source_detection_id"] is not None and item["appearance_available"]


@pytest.mark.skipif(not MODEL.is_file(), reason="Prepare optional pinned model")
def test_cold_run_all_worker_retains_track_history_and_shuts_down_without_accelerator_probe():
    script = r"""
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication
import cv2
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
def unexpected(*args):
    raise AssertionError("CPU inference probed accelerators")
cv2.dnn.getAvailableTargets = unexpected
app = QApplication([])
registry = builtin_registry()
graph = Graph.load("examples/opencv tools/detection_tracking/yolox_coffee.json", registry)
worker = Runner(graph, "all", registry)
worker.frame.connect(lambda: worker.ack.set(), Qt.ConnectionType.DirectConnection)
worker.start()
assert worker.wait(25000)
assert not worker.ended_with_error
assert worker.iterations == 6 and graph.sources_exhausted()
tool = graph.blocks["track"]
assert tool.last_result.outputs["count"] == 1
item = tool.last_result.outputs["tracks"]["items"][0]
assert item["id"] == 1 and item["hits"] == 5 and item["class_id"] == 41
worker.deleteLater()
graph.close()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del worker, graph, app
print("tracking Run All worker closed", flush=True)
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
    assert "tracking Run All worker closed" in result.stdout
