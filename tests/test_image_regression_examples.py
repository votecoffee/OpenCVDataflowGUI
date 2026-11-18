"""Native image measurements, held-out residuals, portable assets and batch rows."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/image_regression"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("case", sorted(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_regression_programs_and_source_receipts(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        for index in range(3 if case == "batch_rows" else 1):
            graph.run()
            if case == "batch_rows" and index < 2:
                assert not graph.blocks["metrics"].current_emissions
        block = graph.blocks[TRUTH["cases"][case]["target"]]
        state = (
            "error"
            if case in ("bad_model", "wrong_layout", "evaluation_mixed")
            else "fail"
            if case in ("low_contrast", "range_fail", "evaluation_lighting")
            else "pass"
        )
        assert block.state == state, [(k, b.error) for k, b in graph.blocks.items()]
        if state == "error":
            assert not block.current_emissions
            return
        result = block.last_result
        assert result is not None and result.score is None
        if case.startswith("evaluation") or case == "batch_rows":
            expected = BENCH["groups"][case]
            for name in ("mae", "rmse", "bias", "maximum_absolute_error"):
                assert result.outputs[name] == pytest.approx(expected[name], abs=2e-5)
            report = result.outputs["report"]
            assert report["sample_count"] == 3 and report["identity_count"] == 1
            assert report["capture_independence_verified"] is False
            if case == "batch_rows":
                assert report["r2"] is None
                assert [r["source_ticket"]["value"] for r in report["records"]] == [
                    "batch-a.png",
                    "batch-b.png",
                    "batch-c.png",
                ]
                assert (
                    graph.blocks["decode"].last_result.outputs["report"]["identity"][
                        "tensor_hash_checked"
                    ]
                    is False
                )
            assert (result.outputs["plot"] is NO_VALUE) is (profile == "lean")
        else:
            image = graph.blocks["source"].last_result.outputs["image"]
            np.testing.assert_array_equal(result.outputs["image"], image)
            measurement = result.outputs["measurement"]
            assert (
                measurement["model_revision"]
                == json.loads(
                    (
                        BASE
                        / "bundles"
                        / ("nchw" if case == "nchw_head" else "vector")
                        / "contract.json"
                    ).read_bytes()
                )["revision"]
            )
            assert result.outputs["report"]["identity"]["tensor_hash_checked"]
            ticket = graph.blocks["source"].last_result.payloads["image"]["ticket"]
            assert (
                measurement["context"]["ticket"] == ticket
                and result.payloads["value"]["ticket"] == ticket
            )
            reference = BENCH["images"][
                "dim.png"
                if case == "low_contrast"
                else "test-0.png"
                if case == "width_small"
                else "test-2.png"
                if case == "width_large"
                else "test-1.png"
            ]
            if case == "brightness":
                assert result.outputs["value"] == pytest.approx(
                    reference["vector_brightness"], abs=0.001
                )
            else:
                assert result.outputs["value"] == pytest.approx(reference["vector_value"], abs=2e-5)
            if case in ("no_reference", "brightness"):
                assert (
                    result.outputs["residual"] is NO_VALUE
                    and not block.current_emissions["residual"]
                )
            else:
                assert result.outputs["residual"] == pytest.approx(
                    result.outputs["value"] - measurement["reference"]
                )
            assert bool(result.drawings) is (profile == "debug")
            if case == "high_depth":
                assert result.outputs["image"].dtype == np.uint16
            film = graph.blocks["filmstrip"].last_result
            assert film is not None and film.outputs["image"].shape[:2] == image.shape[:2]
            if profile == "debug":
                assert np.any(
                    film.outputs["image"]
                    != (
                        np.repeat(image[..., None], 3, axis=2)
                        if film.outputs["image"].ndim == 3 and image.ndim == 2
                        else image
                    )
                )
    finally:
        graph.close()


def test_independent_native_svr_onnx_and_numpy_error_references():
    observed = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    assert observed["maximum_onnx_numpy_difference"] < 2e-5
    assert observed["maximum_svm_numpy_difference"] < 2e-7
    for name, expected in BENCH["images"].items():
        assert observed["images"][name]["vector_value"] == pytest.approx(
            expected["vector_value"], abs=2e-5
        )
    assert observed["groups"]["evaluation"]["mae"] < 0.002
    assert observed["groups"]["evaluation_lighting"]["mae"] > 0.15
    assert [call["shape"][0] for call in observed["batch_calls"]] == [2, 1]


def test_all_asset_pins_training_query_separation_and_exact_regeneration(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    training = {TRUTH["asset_sha256"][f"assets/train-{i}.png"] for i in range(7)}
    assert all(TRUTH["asset_sha256"][f"assets/test-{i}.png"] not in training for i in range(3))
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (BASE / p.relative_to(tmp_path)).read_bytes(), p
    with pytest.raises(FileExistsError, match="preserved"):
        build(tmp_path)


def test_sealed_regression_model_survives_save_as_and_xml(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "width_middle.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "program.json", registry)
    try:
        contract = tmp_path / graph.blocks["net"].parameters["contract_path"]
        assert (contract.parent / "model.onnx").read_bytes() == (
            BASE / "bundles/vector/model.onnx"
        ).read_bytes()
        export_project_xml(graph.to_dict(), tmp_path / "program.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["decode"].last_result.outputs["value"] == pytest.approx(
            1.325, abs=0.002
        )
    finally:
        graph.close()


def test_failed_input_is_rejected_before_history_changes_or_measurement_event(monkeypatch):
    graph = Graph.load(BASE / "width_middle.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["decode"]
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


@pytest.mark.parametrize(
    "project,iterations", [("width_middle", 1), ("evaluation", 1), ("batch_rows", 3)]
)
def test_cold_run_all_closes_native_models_and_completes_expected_rows(project, iterations):
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);r=builtin_registry();name=sys.argv[2];g=Graph.load(Path(sys.argv[1])/(name+'.json'),r)
try:
 worker=Runner(g,MainWindow.run_all_mode(SimpleNamespace(root_graph=g)),r);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==int(sys.argv[3])
 target=g.blocks['decode' if name=='width_middle' else 'metrics']
 assert target.state=='pass',target.error
finally:g.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE), project, str(iterations)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
