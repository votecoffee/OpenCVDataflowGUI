"""Saved native image-mixture projects, exact assets and current result graphics."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/image_mixtures"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_color_and_texture_mixture_projects(case, profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        if case.startswith("train"):
            graph.blocks["save"].parameters["path"] = str(tmp_path / "model.json")
        graph.run()
        block = graph.blocks[expected["target"]]
        assert block.state == expected["state"], block.error
        if case.startswith("train"):
            assert (
                json.loads((tmp_path / "model.json").read_bytes())
                == graph.blocks["train"].last_result.outputs["model"]
            )
            return
        result = block.last_result
        assert result is not None and not block.error
        source = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], source)
        assert result.payloads["report"]["ticket"]["value"] == expected["image"]
        report = result.outputs["report"]
        if case in (
            "color_known",
            "color_rgb16",
            "region_color",
            "posterior_maps",
            "spherical_color",
            "texture_known",
        ):
            assert report["unknown_count"] == 0
        if case == "color_defect":
            mask = cv2.imread(str(BASE / "assets/color-truth.png"), 0)
            np.testing.assert_array_equal(result.outputs["unknown_mask"], mask)
            assert report["unknown_count"] == 120
            assert result.fail_reasons[0]["code"] == "em_unknown_pixels"
            assert np.min(result.outputs["winning_posterior"][mask != 0]) > 0.99
        if case == "density_disabled":
            assert report["unknown_count"] == 0 and result.passed
        if case == "component_filter":
            assert report["rejected_component"] == 3072
        if case.startswith("texture"):
            assert report["supported_pixels"] == 3400 and report["selected_pixels"] == 4480
            if case == "texture_support":
                assert result.fail_reasons[0]["code"] == "em_feature_support"
            if case == "texture_defect":
                assert report["unknown_count"] == 192
        if case == "posterior_maps":
            group = result.outputs["posteriors"]
            assert len(group.items) == 2
            np.testing.assert_allclose(sum(p.value for p in group.items), 1, atol=1e-7)
            for p in group.items:
                assert p.payload["ticket"] == result.payloads["image"]["ticket"]
        else:
            assert result.outputs["posteriors"] is NO_VALUE
        assert graph.blocks["filmstrip"].state == "pass"
        if profile == "debug":
            assert result.drawings and result.graphics
            displayed = graph.blocks["overlay"].last_result.outputs["image"]
            assert np.any(displayed != (source[..., None] if source.ndim == 2 else source))
        else:
            assert not result.drawings and not result.graphics
    finally:
        graph.close()


def test_isolated_regeneration_exact_hashes_and_disjoint_training_images(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    for family in ("color", "texture"):
        train = cv2.imread(str(BASE / f"assets/train-{family}.png"), -1)
        query = cv2.imread(str(BASE / f"assets/query-{family}.png"), -1)
        assert not np.array_equal(train, query)
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    generated = tmp_path / "generated"
    generated.mkdir()
    build.__globals__["HERE"] = generated
    build()
    for path in generated.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(generated)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build()
    for name, digest in before.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest


def test_independent_native_and_gaussian_reference_and_edit_masks():
    benchmark = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    cases = benchmark["cases"]
    for result in cases.values():
        assert result["native_gaussian_max_error"] < 1e-8
    for name in ("color_known", "spherical_color", "texture_known"):
        assert cases[name]["unknown_pixels"] == 0
    color = cases["color_defect"]
    assert color["unknown_pixels"] == color["true_positive"] == 120
    assert color["false_positive"] == color["false_negative"] == 0
    assert color["minimum_winning_posterior"] > 0.99
    texture = cases["texture_defect"]
    assert texture["supported_pixels"] == 3400 and texture["true_positive"] == 192
    assert texture["false_negative"] == 64 and texture["false_positive"] == 0
    assert cases["spherical_color"]["fitted_covariance"] == "spherical"
    assert cases["spherical_color"]["prediction_covariance"] == "diagonal"


def test_json_xml_save_as_keeps_native_model_and_primary_context(tmp_path):
    graph = Graph.load(BASE / "color_defect.json", builtin_registry())
    try:
        graph.save(tmp_path / "copy.json")
    finally:
        graph.close()
    restored = Graph.load(tmp_path / "copy.json", builtin_registry())
    try:
        assert (tmp_path / restored.blocks["model"].parameters["path"]).read_bytes() == (
            BASE / "assets/model-color.json"
        ).read_bytes()
        restored.run()
        assert restored.blocks["inspect"].last_result.outputs["report"]["unknown_count"] == 120
        export_project_xml(restored.to_dict(), tmp_path / "copy.xml")
        xml_graph = Graph.from_dict(
            read_project_xml(tmp_path / "copy.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            xml_graph.run()
            assert xml_graph.blocks["inspect"].state == "fail"
            assert xml_graph.blocks["inspect"].last_result.outputs["report"]["unknown_count"] == 120
        finally:
            xml_graph.close()
    finally:
        restored.close()


def test_rejected_image_and_absent_wired_region_do_not_replace_last_input(monkeypatch):
    graph = Graph.load(BASE / "region_color.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["inspect"]
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
        monkeypatch.setattr(
            graph.blocks["selection"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count and not block.current_emissions
    finally:
        graph.close()


def test_cold_background_worker_one_image_run_all_is_once_and_shuts_down():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'color_known.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='once'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(15000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and len(frames)==1
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
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"iterations": 1, "frames": 1}
