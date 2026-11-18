"""Portable public cascade projects and cold native worker qualification."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/cascade_detection"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
pytestmark = pytest.mark.local_examples


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return overlap / (aw * ah + bw * bh - overlap)


@pytest.mark.parametrize("case", sorted(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_projects_match_native_and_coarse_face_roi(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["detect"]
        assert block.state == TRUTH["cases"][case]["state"], block.error
        if case == "bad_pin":
            assert not block.current_emissions
            return
        result = block.last_result
        assert result is not None, block.error
        source = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], source)
        assert result.payloads["collection"]["ticket"]["value"] == TRUTH["cases"][case]["image"]
        items = result.outputs["collection"]["items"]
        rows = BENCH["cases"][case]["retained"]
        rows = sorted(rows, key=lambda row: (*row["box"], row["score"] or 0))
        items = sorted(items, key=lambda item: (*item["geometry"]["bbox"], item["score"] or 0))
        assert len(items) == len(rows)
        for item, row in zip(items, rows, strict=True):
            assert item["geometry"]["bbox"] == row["box"]
            if row["score"] is None:
                assert item["score"] is None
            else:
                assert item["score"] == pytest.approx(row["score"], abs=1e-5)
        assert (
            result.outputs["report"]["native_candidates"]
            == BENCH["cases"][case]["native_candidates"]
        )
        if case not in ("raw_windows", "blank", "weight_fail", "strict_size"):
            assert (
                max(iou(item["geometry"]["bbox"], TRUTH["reference_face"]) for item in items) > 0.7
            )
        if case == "two_faces":
            assert (
                len(items) == 2
                and max(
                    iou(item["geometry"]["bbox"], TRUTH["second_reference_face"]) for item in items
                )
                > 0.7
            )
        if case == "raw_windows":
            assert any(item["native_weight"] < 0 for item in items) and len(items) == 40
        if case == "neighbors":
            assert items[0]["neighbor_count"] == 37
        if case == "high_depth":
            assert result.outputs["image"].dtype == np.uint16
        assert bool(result.drawings) is (profile == "debug")
        assert result.score is None
        film = graph.blocks["filmstrip"].last_result
        assert film is not None and film.outputs["image"].shape == source.shape
        if profile == "debug" and items:
            assert np.any(film.outputs["image"] != source)
    finally:
        graph.close()


def test_independent_direct_native_reference_matches_pinned_values():
    observed = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    for case, expected in BENCH["cases"].items():
        assert observed["cases"][case]["native_candidates"] == expected["native_candidates"]
        got = observed["cases"][case]["retained"]
        assert len(got) == len(expected["retained"])
        for row, reference in zip(got, expected["retained"], strict=True):
            assert (
                row["box"] == reference["box"] and row["reject_level"] == reference["reject_level"]
            )
            if row["score"] is not None:
                assert row["score"] == pytest.approx(reference["score"], abs=1e-5)


def test_exact_asset_pins_regeneration_and_original_preservation(tmp_path):
    hashes = TRUTH["asset_sha256"]
    for name, digest in hashes.items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path, BASE / "assets")
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build(tmp_path, BASE / "assets")


def test_save_as_and_xml_carry_exact_model_and_current_images(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "portrait.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        path = tmp_path / graph.blocks["detect"].parameters["path"]
        assert path.read_bytes() == (BASE / "assets/haar.xml").read_bytes()
        export_project_xml(graph.to_dict(), tmp_path / "program.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["detect"].last_result.outputs["count"] == 1
    finally:
        graph.close()


def test_failed_input_gating_retains_previous_history_without_new_output(monkeypatch):
    graph = Graph.load(BASE / "portrait.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["detect"]
        previous = block.last_inputs["image"].copy()
        count = block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        source = graph.blocks["source"]
        original = source.process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(source, "process", failed)
        source.reset_runtime()
        graph.run()
        assert block.input_rejected and not block.current_emissions and block.timing.count == count
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()


@pytest.mark.parametrize("mode", ["run_all", "parallel", "preview"])
def test_cold_native_workers_exit_cleanly(mode):
    script = r"""
import json,sys
from pathlib import Path
import cv2
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner,LivePreview
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);base=Path(sys.argv[1]);mode=sys.argv[2];registry=builtin_registry()
graph=Graph.load(base/'portrait.json',registry)
try:
 if mode=='parallel':
  other=graph.add(registry.create('ml.cascade_detect',block_id='other',parameters=graph.blocks['detect'].parameters))
  graph.connect('source','image','other','image');graph.set_branch_workers(2);graph.run()
  assert other.last_result.outputs['count']==1,other.error
 elif mode=='preview':
  worker=LivePreview(graph.blocks['detect'],{'image':cv2.imread(str(base/'assets/portrait.png'))},1,None)
  worker.start();assert worker.wait(25000);assert not worker.error,worker.error
 else:
  worker=Runner(graph,MainWindow.run_all_mode(SimpleNamespace(root_graph=graph)),registry)
  errors=[];frames=[]
  worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
  worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
  if not worker.wait(25000):
   worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
  assert not errors,errors
  assert worker.iterations==len(frames)==1
 assert graph.blocks['detect'].last_result.outputs['count']==1,graph.blocks['detect'].error
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE), mode],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
