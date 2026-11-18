"""Known-good image-model training, held-out inspection and portable saved graphs."""

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
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/feature_outliers"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_training_and_outlier_projects(case, profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        if case.startswith("train"):
            graph.blocks["save"].parameters["path"] = str(tmp_path / "model.json")
        for index, state in enumerate(expected["states"]):
            graph.run()
            block = graph.blocks[expected["target"]]
            assert block.state == state, block.error
            if state == "inactive":
                assert not block.current_emissions
                continue
            if state == "error":
                assert not block.current_emissions
                assert graph.blocks["overlay"].state == "inactive"
                if case == "singular_no_ridge":
                    assert "singular" in block.error
                else:
                    assert "preprocessing differs" in block.error
                continue
            if case.startswith("train"):
                model = graph.blocks["fit"].last_result.outputs["model"]
                assert json.loads((tmp_path / "model.json").read_bytes()) == model
                assert len(model["mean"]) == 144 and model["sample_count"] == 9
                assert model["feature_contract"]["settings"]["bins"] == 4
                continue
            result = block.last_result
            source = graph.blocks["source"].last_result.outputs["image"]
            np.testing.assert_array_equal(result.outputs["image"], source)
            assert result.payloads["distances"]["ticket"]["value"] == expected["images"][index]
            assert result.outputs["report"]["contract_verified"]
            assert result.score is None and result.final_judgment is None
            distance = result.outputs["maximum_distance"]
            if case in ("crack", "missing_arc", "empty_view"):
                assert distance > 15 and result.outputs["outlier_count"] == 1
                assert result.fail_reasons[0]["code"] == "mahalanobis_distance_out_of_tolerance"
            if case == "low_contrast_miss":
                assert 2 < distance < 4 and result.passed
            if case == "known_set":
                assert 2 < distance < 3
            if case == "diagnostic_only":
                assert distance > 15 and result.passed and result.outputs["outlier_count"] == 1
            assert graph.blocks["filmstrip"].state == "pass"
            if profile == "debug":
                assert result.drawings and result.graphics
                displayed = graph.blocks["overlay"].last_result.outputs["image"]
                assert np.any(displayed != source[..., None])
            else:
                assert not result.drawings and not result.graphics
    finally:
        graph.close()


def test_exact_assets_pixel_separation_and_isolated_regeneration(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    training = [cv2.imread(str(BASE / "assets" / n), 0).tobytes() for n in TRUTH["training"]]
    queries = [cv2.imread(str(BASE / "assets" / n), 0).tobytes() for n in TRUTH["known"]]
    assert len(set(training + queries)) == 12
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
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


def test_independent_hog_covariance_and_native_quadratic_references():
    result = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    for mode, reference in result["modes"].items():
        assert reference["training_images"] == 9 and reference["dimensions"] == 144
        assert reference["covariance_max_error"] < 1e-14
        assert reference["maximum_native_numpy_error"] < 1e-9
        cases = {r["image"]: r for r in reference["heldout"]}
        assert all(cases[n]["passed"] for n in TRUTH["known"])
        assert not cases["crack.png"]["passed"] and not cases["missing-arc.png"]["passed"]
        assert cases["low-contrast.png"]["passed"]
        assert cases["empty.png"]["supported"] is (mode == "none")


def test_model_save_as_xml_and_pca_revision_keep_exact_assets(tmp_path):
    graph = Graph.load(BASE / "pca_known.json", builtin_registry())
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    restored = Graph.load(tmp_path / "program.json", builtin_registry())
    try:
        for block_id, name in (("model", "covariance-pca.json"), ("pca", "pca-none.json")):
            assert (tmp_path / restored.blocks[block_id].parameters["path"]).read_bytes() == (
                BASE / "assets" / name
            ).read_bytes()
        restored.run()
        assert restored.blocks["inspect"].state == "pass"
        export_project_xml(restored.to_dict(), tmp_path / "program.xml")
        xml_graph = Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            xml_graph.run()
            assert xml_graph.blocks["inspect"].state == "pass"
        finally:
            xml_graph.close()
    finally:
        restored.close()


def test_rejected_and_absent_descriptors_do_not_replace_last_image(monkeypatch):
    graph = Graph.load(BASE / "known_set.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["inspect"]
        previous = block.last_inputs["image"].copy()
        count = block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed)
        graph.run()
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
        monkeypatch.setattr(graph.blocks["source"], "process", original)
        monkeypatch.setattr(
            graph.blocks["hog"],
            "process",
            lambda inputs: Result({"descriptor": NO_VALUE, "info": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count and not block.current_emissions
    finally:
        graph.close()


def test_existing_numeric_covariance_project_still_runs():
    graph = Graph.load(
        BASE.parent / "feature_covariance/feature_covariance.json", builtin_registry()
    )
    try:
        graph.run()
        assert not any(block.error for block in graph.blocks.values())
    finally:
        graph.close()


def test_cold_background_run_all_emits_each_heldout_frame_and_closes():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'known_set.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='all'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==3 and len(frames)==3
 assert graph.blocks['inspect'].last_result.passed
 print(json.dumps({'iterations':worker.iterations,'frames':len(frames)}))
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"iterations": 3, "frames": 3}
