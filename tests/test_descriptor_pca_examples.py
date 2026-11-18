"""Saved image PCA training/inference graphs, portability and independent references."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/descriptor_pca"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_image_training_and_heldout_projects(case, profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        training = case.startswith("train")
        if training:
            graph.blocks["save_pca"].parameters["path"] = str(tmp_path / "pca.json")
            graph.blocks["save"].parameters["path"] = str(tmp_path / "classifier.json")
        for i, state in enumerate(expected["states"]):
            graph.run()
            block = graph.blocks[expected["target"]]
            assert block.state == state, block.error
            if state == "inactive":
                assert not block.current_emissions
                continue
            if state == "error":
                assert not block.current_emissions and "preprocessing" in block.error
                assert graph.blocks["overlay"].state == "inactive"
                continue
            if training:
                pca = json.loads((tmp_path / "pca.json").read_bytes())
                classifier = json.loads((tmp_path / "classifier.json").read_bytes())
                assert pca == graph.blocks["fit"].last_result.outputs["model"]
                assert classifier == graph.blocks["train"].last_result.outputs["model"]
                assert len(pca["mean"]) == 324 and len(pca["basis"]) == 2
                assert classifier["feature_contract"]["pca_revision"] == pca["revision"]
                assert (
                    len(graph.blocks["project"].last_result.payloads["samples"]["feature_rows"])
                    == 6
                )
            else:
                result = block.last_result
                source = graph.blocks["source"].last_result.outputs["image"]
                np.testing.assert_array_equal(result.outputs["image"], source)
                filename = (
                    "heldout-diagonal.png"
                    if case == "reject_diagonal"
                    else ("heldout-vertical.png" if i == 0 else "heldout-horizontal.png")
                )
                assert result.payloads["image"]["ticket"]["value"] == filename
                project = graph.blocks["project"].last_result
                if expected["labels"]:
                    assert result.outputs["labels"] == [expected["labels"][i]]
                if case == "reject_diagonal":
                    assert result.outputs["maximum_error"] > 0.14
                    assert result.fail_reasons[0]["code"] == "pca_reconstruction_error"
                elif case in ("classify", "normalized", "variance", "reconstruct"):
                    assert project.outputs["maximum_error"] < 0.017
                if case == "dimensions_only":
                    assert not project.outputs["report"]["contract_verified"]
                    contract = project.payloads["samples"]["feature_contract"]
                    assert contract["verification"] == "dimensions_only"
                if case == "reconstruct":
                    assert result.outputs["feature_count"] == 324
                assert graph.blocks["filmstrip"].state == "pass"
                if profile == "debug":
                    assert result.drawings
                    rendered = graph.blocks["overlay"].last_result.outputs["image"]
                    assert np.any(rendered != source[..., None])
                else:
                    assert not result.drawings
    finally:
        graph.close()


def test_exact_assets_disjoint_pixels_and_isolated_regeneration(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    pixels = [
        cv2.imdecode(np.frombuffer((BASE / "assets" / name).read_bytes(), np.uint8), 0).tobytes()
        for name in TRUTH["asset_sha256"]
        if name.endswith(".png")
    ]
    assert len(set(pixels)) == len(pixels) == 9
    builder = runpy.run_path(str(BASE / "build_example.py"))
    build = builder["build"]
    destination = tmp_path / "generated"
    destination.mkdir()
    build.__globals__["HERE"] = destination
    build()
    for path in destination.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(destination)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build()
    for name, digest in before.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest


def test_independent_native_hog_svd_and_nearest_neighbor_references():
    benchmark = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    for name, metrics in benchmark["modes"].items():
        assert metrics["input_dimensions"] == 324 and metrics["training_rows"] == 6
        assert metrics["components"] == (1 if name == "variance" else 2)
        assert metrics["saved_projector_max_error"] < 1e-10
        assert metrics["retained_training_variance"] > 0.998
        assert [row["reduced_label"] for row in metrics["heldout"][:2]] == [
            "vertical",
            "horizontal",
        ]
        assert all(row["reduced_label"] == row["raw_label"] for row in metrics["heldout"])
        assert max(row["reconstruction_rmse"] for row in metrics["heldout"][:2]) < 0.017
        assert metrics["heldout"][2]["reconstruction_rmse"] > (0.048 if name == "l2" else 0.14)


def test_portable_models_survive_save_as_and_xml(tmp_path):
    graph = Graph.load(BASE / "classify.json", builtin_registry())
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    restored = Graph.load(tmp_path / "program.json", builtin_registry())
    try:
        for block_id, name in (("pca", "pca-none.json"), ("classifier", "classifier-none.json")):
            assert (tmp_path / restored.blocks[block_id].parameters["path"]).read_bytes() == (
                BASE / "assets" / name
            ).read_bytes()
        restored.run()
        assert restored.blocks["predict"].last_result.outputs["label"] == "vertical"
        export_project_xml(restored.to_dict(), tmp_path / "program.xml")
        from_xml = Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            from_xml.run()
            assert from_xml.blocks["predict"].last_result.outputs["label"] == "vertical"
        finally:
            from_xml.close()
    finally:
        restored.close()


def test_failed_inputs_and_absent_wired_descriptors_do_not_update_state(monkeypatch):
    graph = Graph.load(BASE / "classify.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["project"]
        before = block.last_inputs["image"].copy()
        count = block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def fail(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", fail)
        graph.run()
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], before)
        monkeypatch.setattr(graph.blocks["source"], "process", original)
        from opencv_dataflow.payloads import NO_VALUE

        monkeypatch.setattr(
            graph.blocks["hog"],
            "process",
            lambda inputs: Result({"descriptor": NO_VALUE, "info": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count and not block.current_emissions
    finally:
        graph.close()


def test_existing_numeric_pca_project_remains_runnable():
    graph = Graph.load(
        BASE.parent / "principal_components/principal_components.json", builtin_registry()
    )
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
    finally:
        graph.close()


def test_cold_installed_style_worker_run_all_and_clean_shutdown():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'classify.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='all'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(15000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
 assert not errors,errors
 assert worker.iterations==2 and len(frames)==2
 assert graph.blocks['predict'].last_result.outputs['label']=='horizontal'
 print(json.dumps({'iterations':worker.iterations,'frames':len(frames)}))
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"iterations": 2, "frames": 2}
