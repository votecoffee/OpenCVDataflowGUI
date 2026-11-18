"""Saved image-based anomaly workflows and independent localization/error controls."""

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

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/local_anomaly"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_projects_against_independent_distances_masks_and_outcomes(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        target = graph.blocks[TRUTH["cases"][case]["target"]]
        if case in ("wrong_features", "corrupt_reference"):
            assert target.state == "error" and not target.current_emissions
            return
        expected = BENCH["cases"][case]
        result = target.last_result
        assert result is not None, [(k, b.error) for k, b in graph.blocks.items()]
        assert result.passed == expected["passed"] and result.score is None
        assert result.final_judgment is None
        assert result.outputs["anomaly_pixels"] == expected["anomaly_pixels"]
        assert result.outputs["maximum_distance"] == pytest.approx(
            expected["maximum_distance"], abs=2e-7
        )
        report = result.outputs["report"]
        assert report["scored_pixels"] == expected["selected_pixels"]
        assert report["training_image_reused"] is False
        assert report["capture_independence_verified"] is False
        assert report["score_is_probability"] is False
        source = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], source)
        if case in (
            "defect",
            "high_depth",
            "missing_texture",
            "train_then_inspect",
            "sampled_reference",
            "allowed_area",
        ):
            truth = cv2.imread(str(BASE / "assets/truth-defect.png"), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(result.outputs["mask"], truth)
        elif case == "bright_shift":
            assert np.all(result.outputs["mask"] == 255)
        else:
            assert not np.any(result.outputs["mask"])
        assert bool(result.graphics) is (profile == "debug")
        assert bool(result.drawings) is (profile == "debug")
        ticket = graph.blocks["source"].last_result.payloads["image"]["ticket"]
        assert result.payloads["mask"]["ticket"] == ticket
        assert graph.blocks["features"].last_result.outputs["report"]["identity"][
            "tensor_hash_checked"
        ]
        if case == "high_depth":
            assert result.outputs["image"].dtype == np.uint16
        film = graph.blocks["filmstrip"].last_result
        assert film is not None and film.outputs["image"].shape[:2] == source.shape[:2]
        if profile == "debug" and case == "defect":
            # The defect boundary is a drawable polyline, actually rendered in the filmstrip.
            drawings = result.drawings["local anomalies"].objects
            assert any(obj.kind == "polyline" and obj.closed for obj in drawings)
            assert not np.array_equal(
                film.outputs["image"], np.repeat(source[..., None], 3, axis=2)
            )
    finally:
        graph.close()


def test_independent_native_patch_reference_and_deliberate_accuracy_limits():
    measured = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    assert measured["maximum_native_patch_difference"] == 0
    for case, expected in BENCH["cases"].items():
        assert measured["cases"][case] == expected
    assert measured["cases"]["defect"]["mask_iou"] == 1
    assert measured["cases"]["weak_defect"]["ground_truth_defect_pixels"] == 1
    assert measured["cases"]["weak_defect"]["passed"]
    assert measured["cases"]["weak_defect"]["mask_iou"] == 0
    assert not measured["cases"]["bright_shift"]["passed"]
    assert measured["cases"]["bright_shift"]["ground_truth_defect_pixels"] == 0


def test_exact_regeneration_asset_pins_and_source_separation(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    train = {TRUTH["asset_sha256"][f"assets/train-{i}.png"] for i in range(3)}
    assert all(
        TRUTH["asset_sha256"][f"assets/{name}.png"] not in train
        for name in ("normal", "defect", "missing", "weak", "bright")
    )
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build(tmp_path)


def test_reference_and_onnx_assets_survive_save_as_xml_and_reload(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "defect.json", registry)
    try:
        graph.save(tmp_path / "saved.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "saved.json", registry)
    try:
        reference = tmp_path / graph.blocks["reference"].parameters["path"]
        assert reference.read_bytes() == (BASE / "assets/reference.json").read_bytes()
        graph.run()
        assert graph.blocks["inspect"].last_result.outputs["anomaly_pixels"] == 64
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["inspect"].last_result.outputs["anomaly_pixels"] == 64
    finally:
        graph.close()


def test_failed_input_gate_keeps_previous_input_and_emits_nothing(monkeypatch):
    graph = Graph.load(BASE / "normal.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["inspect"]
        previous, count = block.last_inputs["image"].copy(), block.timing.count
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


@pytest.mark.parametrize("case", ["normal", "train_then_inspect"])
def test_cold_run_all_and_model_teardown(case):
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);r=builtin_registry();g=Graph.load(Path(sys.argv[1])/(sys.argv[2]+'.json'),r)
try:
 worker=Runner(g,MainWindow.run_all_mode(SimpleNamespace(root_graph=g)),r);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 pixels=g.blocks['inspect'].last_result.outputs['anomaly_pixels']
 assert pixels==(0 if sys.argv[2]=='normal' else 64)
finally:g.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
