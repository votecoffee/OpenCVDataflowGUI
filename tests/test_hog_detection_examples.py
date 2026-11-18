"""Saved native HOG training/search projects, known boxes and portable model checks."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/hog_detection"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_training_detection_and_failure_projects(case, profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        if case == "train":
            graph.blocks["save"].parameters["path"] = str(tmp_path / "detector.json")
        for index, state in enumerate(expected["states"]):
            graph.run()
            block = graph.blocks[expected["target"]]
            assert block.state == state, (case, [(k, b.error) for k, b in graph.blocks.items()])
            if state in ("inactive", "error"):
                assert not block.current_emissions
                continue
            if case == "train":
                assert (
                    json.loads((tmp_path / "detector.json").read_bytes())
                    == graph.blocks["model"].last_result.outputs["model"]
                )
                continue
            result = block.last_result
            assert result.passed is (state == "pass") and result.score is None
            source = graph.blocks["source"].last_result.outputs["image"]
            np.testing.assert_array_equal(result.outputs["image"], source)
            assert result.payloads["collection"]["ticket"]["value"] == expected["images"][index]
            boxes = sorted(
                item["geometry"]["bbox"] for item in result.outputs["collection"]["items"]
            )
            if case in ("single_scale", "multiscale", "high_depth", "count_fail"):
                assert boxes == TRUTH["boxes"]
                assert all(
                    item["native_weight"] > 0.99 for item in result.outputs["collection"]["items"]
                )
            if case == "raw_windows":
                assert len(boxes) == 6
            if case == "low_threshold":
                assert len(boxes) == 6 and all(box in boxes for box in TRUTH["boxes"])
            if case == "selection":
                assert boxes == TRUTH["boxes"][:1]
            if case in ("blank", "default_people", "daimler_people"):
                assert not boxes
            if case == "high_depth":
                assert result.outputs["image"].dtype == np.uint16
            if profile == "debug":
                assert result.drawings
            else:
                assert not result.drawings
    finally:
        graph.close()


def test_independent_exhaustive_window_margin_and_iou_references():
    observed = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    assert observed["windows"] == 501
    for threshold, case in observed["cases"].items():
        assert case["differing_boxes"] == 0 and case["maximum_margin_error"] < 2e-6
        assert case["native_candidates"] == case["independent_candidates"]
        if threshold in ("0.5", "0.8"):
            assert sorted(row["box"] for row in case["retained"]) == TRUTH["boxes"]
        else:
            assert len(case["retained"]) == 6


def test_regeneration_is_exact_with_disjoint_training_query_hashes_and_preserved_destinations(
    tmp_path,
):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }
    hashes = TRUTH["asset_sha256"]
    for name, digest in hashes.items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    assert hashes["scene.png"] not in {hashes[n] for n in TRUTH["training"]}
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build.__globals__["HERE"] = tmp_path
    build()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build()
    assert before == {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }


def test_json_xml_save_as_keep_detector_bytes_and_source_boxes(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "multiscale.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        path = tmp_path / graph.blocks["model"].parameters["path"]
        assert path.read_bytes() == (BASE / "assets/detector.json").read_bytes()
        export_project_xml(graph.to_dict(), tmp_path / "program.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["detect"].last_result.outputs["count"] == 2
    finally:
        graph.close()


def test_failed_inputs_gate_detection_before_last_input_history_changes(monkeypatch):
    graph = Graph.load(BASE / "single_scale.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["detect"]
        previous = block.last_inputs["image"].copy()
        count = block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed)
        graph.blocks["source"].reset_runtime()
        graph.run()
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()


def test_cold_run_all_detects_current_boxes_and_closes():
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'multiscale.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph))
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and len(frames)==1
 assert graph.blocks['detect'].last_result.outputs['count']==2
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
