"""Portable flow projects, independent native quantities and clean background teardown."""

import hashlib
import json
import os
import runpy
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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_flow"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
MODEL = BASE / ".local/raft/model.onnx"


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", TRUTH["cases"])
def test_saved_pair_quantities_geometry_coverage_and_observation_budget(case, profile):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit RAFT preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["estimate"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed == (case not in ("empty", "coverage_alarm"))
        first = session.graph.blocks["first"].last_result
        second = session.graph.blocks["second"].last_result
        np.testing.assert_array_equal(result.outputs["image"], first.outputs["image"])
        payload = result.payloads["flow"]
        assert payload["ticket"] == first.payloads["image"]["ticket"]
        assert payload["ticket"]["value"] != second.payloads["image"]["ticket"]["value"]
        assert payload["spatial"] == first.payloads["image"]["spatial"]
        assert (
            payload["flow_pair"]["frames"]["second"]["spatial"]
            == second.payloads["image"]["spatial"]
        )
        valid = result.outputs["valid_mask"] != 0
        vectors = result.outputs["flow"].vectors
        assert np.isnan(vectors[~valid]).all()
        assert payload["flow_association"]["direction"] == "source_to_target"
        if case.startswith("native_"):
            reference = BENCH["cases"][case]
            for sample in reference["source_samples"]:
                x, y = sample["pixel"]
                if valid[y, x]:
                    np.testing.assert_allclose(vectors[y, x], sample["vector"], atol=3e-5)
            x, y, w, h = reference["evaluation_rectangle"]
            retained = valid[y : y + h, x : x + w]
            assert retained.sum() == reference["retained_pixels"]
            error = np.linalg.norm(
                vectors[y : y + h, x : x + w].astype(np.float64) - reference["truth_motion"], axis=2
            )
            assert error[retained].mean() == pytest.approx(
                reference["retained_mean_endpoint_error"], abs=1e-6
            )
            call = result.outputs["receipt"]["calls"][0]
            assert call["configuration_verified"] and call["external_preprocessing_verified"]
            assert call["selected_execution"]["engine"] == "classic"
            assert call["model_import"]["transport"] == "temporary_snapshot_file"
            assert [x["name"] for x in call["outputs"]] == ["12006"]
        elif valid.any():
            expected = [1.5, 1] if case == "subpixel" else [2, -1]
            np.testing.assert_allclose(
                vectors[valid], np.broadcast_to(expected, (int(valid.sum()), 2)), atol=1e-5
            )
        if case == "plane":
            warp = session.graph.blocks["warp"].last_result
            assert warp is not None and warp.passed
            covered = warp.outputs["coverage"] != 0
            np.testing.assert_array_equal(
                warp.outputs["image"][covered], second.outputs["image"][covered]
            )
        if case == "empty":
            assert result.outputs["info"]["native_calls"] == 0
        if profile == "lean":
            assert result.outputs["preview"] is NO_VALUE and result.outputs["magnitude"] is NO_VALUE
            assert not result.drawings and not result.graphics
            assert session.graph.blocks["filmstrip"].last_result is None
        else:
            np.testing.assert_array_equal(
                result.outputs["preview"],
                session.graph.blocks["filmstrip"].last_result.outputs["image"],
            )
            assert result.drawings["learned motion"].source == "first"


def test_exact_regeneration_and_pinned_assets_do_not_use_foreign_helper(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(PARAMETERS={"wrong": True}))
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build.__globals__["HERE"] = tmp_path
    build()
    for path in [
        *BASE.glob("*.json"),
        *(BASE / "assets").rglob("*"),
        *(BASE / "models").rglob("*"),
    ]:
        if path.is_file() and path.name != "benchmark.json":
            assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    (tmp_path / "assets/first.png").write_bytes(b"preserved image")
    with pytest.raises(FileExistsError, match="preserved"):
        build()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit RAFT preparation required")
def test_direct_native_benchmark_and_offline_bundle_reverification(tmp_path, monkeypatch):
    actual = runpy.run_path(str(BASE / "benchmark.py"))["benchmark"]()
    assert actual == BENCH
    assert actual["cases"]["native_positive"]["mean_endpoint_error"] < 0.025
    assert actual["cases"]["native_negative"]["mean_endpoint_error"] < 0.025
    assert actual["cases"]["native_zero"]["mean_endpoint_error"] < 0.005
    assert all(
        c["retained_fraction"] == 1 and c["bad1_fraction"] == 0 for c in actual["cases"].values()
    )
    helper = runpy.run_path(str(BASE / "prepare_model.py"))
    monkeypatch.setattr(helper["urllib"].request, "urlopen", lambda *a, **k: pytest.fail("network"))
    before = [p.stat().st_mtime_ns for p in MODEL.parent.iterdir()]
    helper["prepare"]()
    assert before == [p.stat().st_mtime_ns for p in MODEL.parent.iterdir()]
    (tmp_path / "contract.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        helper["prepare"](directory=tmp_path)


@pytest.mark.parametrize("case", ["plane", "native_positive"])
def test_save_as_and_xml_keep_exact_models_notices_and_frame_order(tmp_path, case):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit RAFT preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        contract = session.graph.blocks["estimate"].parameters["contract_path"]
        session.save(tmp_path / "project.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "project.xml")
    for name in (
        contract,
        str(Path(contract).with_name("model.onnx")),
        "SOURCES.md",
        "BSD-3-LICENSE.txt",
        "MITLICENSE.txt",
    ):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "project.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "project.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            graph.run()
            result = graph.blocks["estimate"].last_result
            assert result is not None and result.passed, graph.blocks["estimate"].error
            assert result.outputs["info"]["direction"] == "first_to_second"
            assert (
                result.outputs["flow"].metadata["flow_association"]["direction"]
                == "source_to_target"
            )
        finally:
            graph.close()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit RAFT preparation required")
def test_cold_native_background_worker_and_cache_teardown_are_clean():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'native_positive.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='once'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(30000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and len(frames)==1
 result=graph.blocks['estimate'].last_result
 assert result is not None and result.passed and result.outputs['info']['native_calls']==1
 print(json.dumps({'iterations':worker.iterations,'shape':list(result.outputs['flow'].vectors.shape)}))
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "iterations": 1,
        "shape": [180, 240, 2],
    }
