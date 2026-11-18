"""Measured-pose classification, ticket/geometry retention and native saved projects."""

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
from opencv_dataflow.payloads import FlowGroup
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/rotated_crop_classification"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("case", sorted(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_measured_pose_projects_and_inherited_child_tickets(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks[TRUTH["cases"][case]["target"]]
        if case == "coverage_strict":
            assert (
                block.state == "error"
                and not block.current_emissions
                and "unsupported" in block.error
            )
            return
        if case == "coverage_pad":
            assert block.state == "fail" and block.last_result.outputs["coverage"] == pytest.approx(
                0.40087890625
            )
            return
        result = block.last_result
        assert result is not None, block.error
        expected = BENCH["cases"][case]
        assert (
            result.outputs["pass_count"] == expected["pass_count"]
            and result.outputs["fail_count"] == expected["fail_count"]
        )
        assert result.outputs["error_count"] == 0 and result.outputs["complete"]
        assert block.state == (
            "pass" if expected["fail_count"] == 0 and expected["rows"] else "fail"
        )
        parent = graph.blocks["source"].last_result.payloads["image"]["ticket"]
        rows = result.outputs["results"]["items"]
        assert len(rows) == len(expected["rows"])
        if not rows:
            return
        pictures = result.outputs["images"]
        packets = result.outputs["group"]
        assert isinstance(pictures, FlowGroup) and pictures.labels == packets.labels
        for record, packet, picture, reference in zip(
            rows, packets.items, pictures.items, expected["rows"], strict=True
        ):
            assert record["feature_id"] == reference["feature_id"]
            np.testing.assert_allclose(record["pose"]["origin"], reference["center"], atol=1e-9)
            assert record["outputs"]["label"] == reference["label"]
            assert record["outputs"]["confidence"] == pytest.approx(
                reference["confidence"], abs=1e-6
            )
            assert record["passed"] is reference["passed"]
            geom = record["outputs"]["geometry"]
            np.testing.assert_allclose(
                geom["output_to_input"], reference["source_from_crop"], atol=1e-9
            )
            assert packet.payload["inspection"]["parent_ticket"] == parent
            assert packet.payload["inspection"]["feature_id"] == record["feature_id"]
            assert picture.payload["ticket"] == packet.payload["ticket"]
            assert picture.payload["inspection"]["parent_ticket"] == parent
            np.testing.assert_allclose(
                picture.payload["spatial"]["to_root"], reference["source_from_crop"], atol=1e-9
            )
            assert picture.value.shape == (32, 64)
            assert picture.value.dtype == (np.uint16 if case == "high_depth" else np.uint8)
            if case not in ("no_alignment", "accepted_horizontal"):
                known = min(
                    TRUTH["parts"],
                    key=lambda part: np.linalg.norm(
                        np.asarray(part["center"]) - record["pose"]["origin"]
                    ),
                )
                label = (
                    "vertical"
                    if case == "all_known" and known["label"] == "unknown"
                    else known["label"]
                )
                if label != "unknown":
                    assert record["outputs"]["label"] == label and record["passed"]
                else:
                    assert not record["passed"] and record["outputs"]["confidence"] < 0.51
    finally:
        graph.close()


def test_independent_moments_bilinear_pixels_and_numpy_inference():
    got = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    for case, expected in BENCH["cases"].items():
        assert got["cases"][case]["pass_count"] == expected["pass_count"]
        for row, ref in zip(got["cases"][case]["rows"], expected["rows"], strict=True):
            assert row["label"] == ref["label"] and row["passed"] == ref["passed"]
            assert row["maximum_bilinear_error_in_uint8_units"] < 0.501
            assert row["maximum_logit_error"] < 1e-6
            assert row["confidence"] == pytest.approx(ref["confidence"], abs=1e-6)
    assert got["cases"]["parallel_inspection"]["pass_count"] == 5
    assert got["cases"]["no_alignment"]["pass_count"] == 0


def test_exact_assets_disjoint_training_query_and_preserved_regeneration(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    training = {TRUTH["asset_sha256"]["assets/" + item["file"]] for item in TRUTH["training"]}
    assert TRUTH["asset_sha256"]["assets/scene.png"] not in training and len(training) == 24
    runpy.run_path(str(BASE / "build_example.py"))["build"](tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        runpy.run_path(str(BASE / "build_example.py"))["build"](tmp_path)


def test_nested_model_contract_and_toolblock_survive_save_as_and_xml(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "parallel_inspection.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        contract = graph.blocks["inspect"].graph.blocks["net"].parameters["contract_path"]
        document = json.loads((tmp_path / contract).read_bytes())
        model = (tmp_path / contract).parent / document["artifact"]["file"]
        assert model.read_bytes() == (BASE / "bundle/model.onnx").read_bytes()
        export_project_xml(graph.to_dict(), tmp_path / "program.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["inspect"].last_result.outputs["pass_count"] == 5
    finally:
        graph.close()


def test_failed_source_is_gated_before_new_pose_jobs_and_history_updates(monkeypatch):
    graph = Graph.load(BASE / "parallel_inspection.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["inspect"]
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
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()


def test_cold_worker_runs_one_image_and_closes_nested_native_jobs():
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1])/'parallel_inspection.json',registry)
try:
 worker=Runner(graph,MainWindow.run_all_mode(SimpleNamespace(root_graph=graph)),registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 assert graph.blocks['inspect'].last_result.outputs['pass_count']==5
 assert graph.blocks['inspect'].last_result.outputs['fail_count']==1
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
