"""Native fixtures, independent retained measurements and sealed project portability."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_restoration"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCHMARK = json.loads((BASE / "benchmark.json").read_bytes())
MODEL = BASE / ".local/native_whole/model.onnx"


def module(name):
    spec = importlib.util.spec_from_file_location(
        "learned_restoration_" + name, BASE / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_projects_current_quality_identity_metadata_and_pixel_outcomes(case, profile):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit DnCNN preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["restore"]
        result = block.last_result
        assert result is not None and result.passed, block.error
        source = session.graph.blocks["source"].last_result
        assert result.outputs["image"].shape == source.outputs["image"].shape
        assert result.outputs["image"].dtype == source.outputs["image"].dtype
        assert result.payloads["image"]["ticket"] == source.payloads["image"]["ticket"]
        assert result.payloads["image"]["spatial"] == source.payloads["image"]["spatial"]
        np.testing.assert_array_equal(
            result.outputs["image"], session.graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        if case in ("fidelity_alarm", "native_low_noise"):
            assert not session.graph.blocks["restored_error"].last_result.passed
        elif "restored_error" in session.graph.blocks:
            assert session.graph.blocks["restored_error"].last_result.passed
        if "seam_error" in session.graph.blocks:
            seam = session.graph.blocks["seam_error"].last_result
            assert seam.passed == (case == "native_tiles")
        if case == "empty":
            assert result.outputs["info"]["native_calls"] == 0
            np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        elif case.startswith("native_"):
            expected = BENCHMARK["cases"]["low_noise" if case == "native_low_noise" else "noisy"][
                "metrics"
            ][
                "halo0"
                if case == "native_no_halo"
                else "halo17"
                if case == "native_tiles"
                else "learned"
            ]
            assert session.graph.blocks["restored_error"].last_result.outputs[
                "value"
            ] == pytest.approx(expected["mse"])
            receipt = result.outputs["receipt"]
            assert all(
                call["configuration_verified"] and call["external_preprocessing_verified"]
                for call in receipt["calls"]
            )
            assert {call["artifact"]["sha256"] for call in receipt["calls"]} == {
                TRUTH["native_onnx_sha256"]
            }
            assert all(
                call["requested_availability_scope"] == "requested_cpu_only"
                for call in receipt["calls"]
            )


def test_exact_public_regeneration_is_independent_of_foreign_preparation_helpers(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(PARAMETERS={"foreign": True}))
    builder = module("build_example")
    builder.HERE = tmp_path
    builder.build()
    paths = [*BASE.glob("*.json"), *(BASE / "assets").rglob("*"), *(BASE / "models").rglob("*")]
    for path in paths:
        if path.is_file() and path.name != "benchmark.json":
            assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
    (tmp_path / "assets/noisy.png").write_bytes(b"preserve this image")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit DnCNN preparation required")
def test_direct_native_benchmark_preserves_clean_truth_and_records_losses():
    result = module("benchmark").benchmark()
    assert result == {
        key: value for key, value in BENCHMARK.items() if key != "torch_maximum_unit_error"
    }
    noisy = result["cases"]["noisy"]
    low = result["cases"]["low_noise"]
    assert noisy["metrics"]["learned"]["mse"] < noisy["metrics"]["input"]["mse"] * 0.1
    assert low["metrics"]["learned"]["mse"] > low["metrics"]["input"]["mse"] * 2
    assert noisy["metrics"]["learned"]["faint_crack_erased_pixels"] == 29
    assert noisy["metrics"]["learned"]["strong_crack_erased_pixels"] == 0
    assert noisy["halo17_maximum_unit_error"] < 5e-6
    assert noisy["halo0_maximum_sample_error"] == 30
    image = cv2.imread(str(BASE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    yy, xx = np.indices(image.shape)
    disk = (xx - 30) ** 2 + (yy - 45) ** 2 <= 17**2
    np.testing.assert_array_equal(image[27:64, 12:49] < 90, disk[27:64, 12:49])
    assert BENCHMARK["torch_maximum_unit_error"] < 5e-6


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit DnCNN preparation required")
def test_idempotent_offline_native_preparation_rejects_different_bundles(tmp_path, monkeypatch):
    helper = module("prepare_model")
    monkeypatch.setattr(helper.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    before = [p.stat().st_mtime_ns for p in (BASE / ".local").rglob("*") if p.is_file()]
    identities = helper.prepare()
    assert {item["sha256"] for item in identities} == {TRUTH["native_onnx_sha256"]}
    assert before == [p.stat().st_mtime_ns for p in (BASE / ".local").rglob("*") if p.is_file()]
    import shutil

    shutil.copytree(BASE / ".local", tmp_path / "prepared")
    contract = tmp_path / "prepared/native_whole/contract.json"
    content = json.loads(contract.read_bytes())
    content["configuration"]["halo"] = 0
    contract.write_text(json.dumps(content), encoding="utf-8", newline="\n")
    with pytest.raises((ValueError, FileExistsError)):
        helper.prepare(directory=tmp_path / "prepared")


def test_sealed_models_are_copied_on_save_as_and_json_xml_reopen(tmp_path):
    with ProjectSession.open(BASE / "tiles.json") as session:
        session.save(tmp_path / "program.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "program.xml")
    for name in (
        "models/tiles/model.onnx",
        "models/tiles/contract.json",
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
            result = graph.blocks["restore"].last_result
            assert result is not None and result.passed
            assert result.outputs["info"]["native_calls"] == 16
        finally:
            graph.close()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit DnCNN preparation required")
def test_cold_native_background_worker_finishes_and_exits_cleanly():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'native_tiles.json',registry)
try:
    mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='once'
    worker=Runner(graph,mode,registry);errors=[];frames=[]
    worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
    worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection)
    worker.start()
    if not worker.wait(20000):
        worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
    assert not errors,errors
    assert worker.iterations==1 and len(frames)==1
    result=graph.blocks['restore'].last_result
    assert result is not None and result.passed
    print(json.dumps({'iterations':worker.iterations,'native_calls':result.outputs['info']['native_calls']}))
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
        "native_calls": 9,
    }
