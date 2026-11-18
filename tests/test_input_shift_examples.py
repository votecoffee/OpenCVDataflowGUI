"""Saved model-independent shift evaluation around native ONNX anomaly inspection."""

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
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/input_shift_evaluation"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_shift_reports_against_independent_scores_labels_and_geometry(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        target = graph.blocks[TRUTH["cases"][case]["target"]]
        if case in ("cropped_truth", "missing_variant", "wrong_identity"):
            assert target.state == "error" and not target.current_emissions
            return
        result = target.last_result
        assert result is not None, [(key, b.error) for key, b in graph.blocks.items()]
        expected = BENCH["cases"][case]
        assert result.passed is (
            case in ("clean_stable", "defect_stable", "high_depth", "mixed_parts")
        )
        report = result.outputs["report"]
        summary = report["summary"]
        assert summary["observation_count"] == TRUTH["cases"][case]["observation_count"]
        for field in ("false_accept_rate", "false_reject_rate"):
            assert summary[field] == expected[field]
            assert (result.outputs[field] is NO_VALUE) is (expected[field] is None)
        assert report["complete"] and not report["capture_independence_verified"]
        assert report["score_kind"] == "raw" and not report["generated_shifts_are_new_captures"]
        for key, reference in expected["records"].items():
            index, variant = key.split(":")
            prefix = f"s{index}_{variant}_"
            shift = graph.blocks[prefix + "shift"].last_result
            origin = shift.outputs["shift"]["origin_id"]
            row = next(
                r for r in report["records"] if r["origin_id"] == origin and r["variant"] == variant
            )
            for field in (
                "score",
                "score_delta",
                "truth_iou",
                "baseline_agreement_iou",
                "shared_support_fraction",
            ):
                if reference[field] is None:
                    assert row[field] is None
                else:
                    assert row[field] == pytest.approx(reference[field], abs=3e-7)
            assert row["passed"] == reference["passed"]
            inspect = graph.blocks[prefix + "inspect"].last_result
            assert inspect.outputs["anomaly_pixels"] == reference["anomaly_pixels"]
            np.testing.assert_array_equal(inspect.outputs["image"], shift.outputs["image"])
            if case == "high_depth":
                assert inspect.outputs["image"].dtype == np.uint16
            capture = graph.blocks[prefix + "capture"].last_result.outputs["observation"]
            assert capture["source_ticket"] == shift.payloads["image"]["ticket"]
        assert (result.outputs["plot"] is NO_VALUE) is (profile == "lean")
        if case in ("exposure_sensitive", "blur_sensitive", "pose_sensitive"):
            variant = next(v for v in report["by_variant"] if v != "baseline")
            assert report["by_variant"][variant]["false_reject_rate"] == 1
    finally:
        graph.close()


def test_independent_numeric_reference_exposes_passing_and_failing_pairs():
    observed = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    assert observed == BENCH
    assert observed["maximum_native_patch_difference"] == 0
    assert observed["cases"]["defect_stable"]["records"]["0:pose"]["baseline_agreement_iou"] == 1
    assert observed["cases"]["loose_threshold"]["false_accept_rate"] == 1


def test_exact_assets_and_project_regeneration(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build(tmp_path)


def test_paired_reference_assets_survive_save_as_and_xml(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "defect_stable.json", registry)
    try:
        graph.save(tmp_path / "saved.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "saved.json", registry)
    try:
        graph.run()
        assert graph.blocks["metrics"].last_result.passed
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["metrics"].last_result.outputs["report"]["summary"]["truth_iou"] == 1
    finally:
        graph.close()


def test_failed_input_gate_preserves_last_observed_image_and_blocks_new_record(monkeypatch):
    graph = Graph.load(BASE / "clean_stable.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["s0_pose_capture"]
        previous, count = block.last_inputs["image"].copy(), block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        source = graph.blocks["s0_source"]
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


@pytest.mark.parametrize("case", ["clean_stable", "mixed_parts"])
def test_cold_run_all_completes_one_paired_set_and_closes_native_models(case):
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
 result=g.blocks['metrics'].last_result
 assert result is not None and result.passed
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
