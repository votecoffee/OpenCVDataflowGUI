"""Pinned real model, missing landmarks, portable examples and independent native parity."""

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
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/object_landmarks"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
REFERENCE = json.loads((BASE / "benchmark.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_pretrained_landmark_projects(case, profile):
    expected = REFERENCE["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        graph.run()
        block = graph.blocks["decode"]
        result = block.last_result
        assert result is not None and not block.error, [
            (k, b.error) for k, b in graph.blocks.items()
        ]
        assert result.passed is expected["passed"]
        assert result.outputs["indices"] == expected["accepted_indices"]
        assert result.outputs["object_confidence"] == pytest.approx(
            expected["object_confidence"], abs=1e-6
        )
        items = result.outputs["landmarks"]["items"]
        np.testing.assert_allclose(
            [item["position"] for item in items], expected["positions"], atol=2e-4, rtol=0
        )
        np.testing.assert_allclose(
            [item["confidence"] for item in items], expected["confidence"], atol=1e-6, rtol=0
        )
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        assert (
            graph.blocks["infer"].last_result.outputs["receipt"]["calls"][0]["artifact"]["sha256"]
            == TRUTH["model_sha256"]
        )
        assert graph.blocks["infer"].last_result.outputs["receipt"]["calls"][0][
            "external_preprocessing_verified"
        ]
        assert result.outputs["points"] is NO_VALUE or len(result.outputs["indices"]) == len(items)
        if case == "missing_pose":
            pose = graph.blocks["pose"].last_result
            assert pose is not None and not pose.passed and pose.outputs["pose"] is NO_VALUE
        if profile == "debug":
            assert result.outputs["report"]["drawn_links"] == expected["links"]
            if case != "show_rejected":
                assert result.outputs["report"]["drawn_points"] == expected["accepted_indices"]
            else:
                assert len(result.outputs["report"]["drawn_points"]) >= len(
                    expected["accepted_indices"]
                )
            assert result.drawings
        else:
            assert not result.drawings and result.outputs["report"]["drawn_points"] == []
    finally:
        graph.close()


def test_native_reference_is_repeatable_and_model_predictions_are_not_visibility_truth():
    observed = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    assert observed["model_sha256"] == TRUTH["model_sha256"]
    for case, expected in REFERENCE["cases"].items():
        np.testing.assert_allclose(
            observed["cases"][case]["positions"], expected["positions"], atol=2e-4, rtol=0
        )
        assert observed["cases"][case]["accepted_indices"] == expected["accepted_indices"]
    assert len(observed["cases"]["portrait"]["accepted_indices"]) == 28
    assert len(observed["cases"]["presence_only"]["accepted_indices"]) == 33
    assert not observed["cases"]["blank"]["accepted_indices"]
    assert observed["cases"]["blank"]["object_confidence"] < 1e-6
    # The portrait's high global score is retained alongside its configured point failures.
    assert observed["cases"]["portrait"]["object_confidence"] > 0.99
    assert not observed["cases"]["require_all"]["passed"]


def test_regenerate_into_empty_destination_preserves_license_model_assets_and_projects(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(destination=tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build(destination=tmp_path)
    assert before == {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }


def test_sealed_model_save_as_xml_keep_source_bytes_and_landmark_identities(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "reordered.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    restored = Graph.load(tmp_path / "program.json", registry)
    try:
        restored.run()
        assert restored.blocks["decode"].last_result.outputs["indices"] == [1, 2, 3, 4, 5]
        assert (
            hashlib.sha256((tmp_path / "bundle/model.onnx").read_bytes()).hexdigest()
            == TRUTH["model_sha256"]
        )
        export_project_xml(restored.to_dict(), tmp_path / "program.xml")
    finally:
        restored.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["decode"].last_result.outputs["landmarks"]["items"][0]["channel"] == 16
        assert not graph.blocks["decode"].last_result.outputs["landmarks"]["items"][0]["accepted"]
    finally:
        graph.close()


def test_pinned_model_changes_and_rejected_inputs_have_no_new_decoder_emissions(
    tmp_path, monkeypatch
):
    registry = builtin_registry()
    graph = Graph.load(BASE / "portrait.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        graph.run()
        block = graph.blocks["decode"]
        image, count = block.last_inputs["image"].copy(), block.timing.count
        block.input_failure_policies = {"tensors": "reject_failed"}
        original = graph.blocks["infer"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["infer"], "process", failed)
        graph.run()
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], image)
    finally:
        graph.close()
    model = tmp_path / "bundle/model.onnx"
    model.write_bytes(model.read_bytes() + b"changed")
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        graph.run()
        assert graph.blocks["infer"].state == "error"
        assert not graph.blocks["decode"].current_emissions
    finally:
        graph.close()


def test_cold_background_run_all_produces_current_pretrained_results_and_closes():
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'portrait.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph))
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and len(frames)==1
 assert graph.blocks['decode'].last_result.outputs['count']==28
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
