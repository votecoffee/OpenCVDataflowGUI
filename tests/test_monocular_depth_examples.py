"""Native project contracts, real RGB-D readback and source-aligned quantity references."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/monocular_depth"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
MODEL = BASE / ".local/native_relative/model.onnx"


def module(name):
    spec = importlib.util.spec_from_file_location("monocular_depth_" + name, BASE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_project_outcomes_source_identity_quantities_preview_and_units(case, profile):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit MiDaS preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["estimate"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed == (case not in ("empty", "coverage_alarm"))
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        assert result.payloads["estimate"]["ticket"] == source.payloads["image"]["ticket"]
        assert result.payloads["estimate"]["spatial"] == source.payloads["image"]["spatial"]
        assert result.outputs["estimate"]["metric_validated"] is False
        if case in ("metric", "inverse_metric", "millimeters", "invalid_hole"):
            assert result.outputs["depth"] is not NO_VALUE
        else:
            assert result.outputs["depth"] is NO_VALUE
        if profile == "lean":
            assert result.outputs["preview"] is NO_VALUE and not result.graphics
            assert session.graph.blocks["filmstrip"].last_result is None
        else:
            np.testing.assert_array_equal(
                result.outputs["preview"],
                session.graph.blocks["filmstrip"].last_result.outputs["image"],
            )
        if case == "metric":
            assert session.graph.blocks["cloud"].last_result.outputs["count"] == 713
        if case == "letterbox":
            assert result.outputs["info"]["valid_pixels"] == 31 * 19
        if case == "invalid_hole":
            assert result.outputs["info"]["valid_pixels"] == 713 - 16
        if case == "empty":
            assert result.outputs["info"]["native_calls"] == 0
        if case == "native_relative":
            values = np.array(result.outputs["estimate"]["values"]).reshape(540, 720)
            for sample in BENCH["source_samples"]:
                x, y = sample["pixel"]
                assert values[y, x] == pytest.approx(sample["value"], abs=2e-4)
            assert result.outputs["estimate"]["units"] == "arbitrary"
            assert result.outputs["receipt"]["calls"][0]["configuration_verified"]
            assert (
                result.outputs["receipt"]["calls"][0]["requested_availability_scope"]
                == "requested_cpu_only"
            )


def test_exact_regeneration_preserves_assets_and_ignores_foreign_helpers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(PARAMETERS={"wrong": True}))
    builder = module("build_example")
    builder.HERE = tmp_path
    builder.build()
    paths = [*BASE.glob("*.json"), *(BASE / "assets").rglob("*"), *(BASE / "models").rglob("*")]
    for path in paths:
        if path.is_file() and path.name != "benchmark.json":
            assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    (tmp_path / "assets/ramp.png").write_bytes(b"preserved image")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit MiDaS preparation required")
def test_real_known_depth_benchmark_is_diagnostic_and_not_runtime_calibration():
    result = module("benchmark").benchmark()
    assert result == BENCH
    assert result["known_depth_pixels"] == 387684 and result["ordering_pairs"] > 1000
    assert result["near_far_order_accuracy"] > 0.95
    evaluation = result["evaluation_only_alignment"]
    assert evaluation["runtime_calibration"] is False and result["metric_output"] is False
    assert evaluation["aligned"]["rmse_m"] > evaluation["constant_calibration_median"]["rmse_m"]
    assert 0.05 < evaluation["aligned"]["rmse_m"] < 0.06


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit MiDaS preparation required")
def test_preparation_is_offline_idempotent_and_preserves_different_bundles(tmp_path, monkeypatch):
    helper = module("prepare_model")
    monkeypatch.setattr(helper.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    before = [p.stat().st_mtime_ns for p in (BASE / ".local").rglob("*") if p.is_file()]
    helper.prepare()
    assert before == [p.stat().st_mtime_ns for p in (BASE / ".local").rglob("*") if p.is_file()]
    (tmp_path / "native_relative").mkdir()
    (tmp_path / "native_relative/contract.json").write_bytes(b"{}")
    with pytest.raises((ValueError, FileExistsError)):
        helper.prepare(directory=tmp_path)


@pytest.mark.parametrize("case", ["metric", "native_relative"])
def test_sealed_save_as_and_xml_preserve_models_and_units(tmp_path, case):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit MiDaS preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        relative = session.graph.blocks["estimate"].parameters["contract_path"]
        session.save(tmp_path / "program.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "program.xml")
    for name in (
        relative,
        str(Path(relative).with_name("model.onnx")),
        "MODEL-LICENSE.txt",
        "SOURCES.md",
    ):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            graph.run()
            result = graph.blocks["estimate"].last_result
            assert result is not None and result.passed
            assert (result.outputs["depth"] is NO_VALUE) == (case == "native_relative")
        finally:
            graph.close()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit MiDaS preparation required")
def test_cold_native_background_run_and_driver_teardown_are_clean():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'native_relative.json',registry)
try:
    mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='once'
    worker=Runner(graph,mode,registry);errors=[];frames=[]
    worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
    worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
    if not worker.wait(20000):
        worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
    assert not errors,errors
    assert worker.iterations==1 and len(frames)==1
    result=graph.blocks['estimate'].last_result
    assert result is not None and result.passed and result.outputs['depth'] is NO_VALUE
    print(json.dumps({'iterations':worker.iterations,'valid_pixels':result.outputs['info']['valid_pixels']}))
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "iterations": 1,
        "valid_pixels": 388800,
    }
