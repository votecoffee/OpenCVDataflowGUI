"""Offline saved stereo projects, native controls, portable models and worker cleanup."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_stereo"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())
MODEL = BASE / ".local/hitnet/model.onnx"


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", TRUTH["cases"])
def test_saved_projects_quantities_current_frames_and_lean_display(case, profile):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit HITNET preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["estimate"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed == (case not in ("empty", "coverage_alarm"))
        source = (
            session.graph.blocks["rectify"].last_result
            if case == "plane"
            else session.graph.blocks["left"].last_result
        )
        port = "left" if case == "plane" else "image"
        np.testing.assert_array_equal(result.outputs["image"], source.outputs[port])
        assert result.payloads["disparity"]["spatial"] == source.payloads[port]["spatial"]
        assert result.outputs["info"]["pair"]["tickets_verified"]
        valid = result.outputs["valid_mask"] != 0
        disparity = result.outputs["disparity"]
        assert np.isnan(disparity[~valid]).all()
        if case.startswith("native_"):
            benchmark = BENCH["cases"][case]
            for sample in benchmark["source_samples"]:
                x, y = sample["pixel"]
                if valid[y, x]:
                    assert disparity[y, x] == pytest.approx(sample["value"], abs=3e-5)
            x, y, w, h = benchmark["evaluation_rectangle"]
            retained = valid[y : y + h, x : x + w]
            assert int(retained.sum()) == benchmark["retained_pixels"]
            error = np.abs(
                disparity[y : y + h, x : x + w].astype(np.float64) - benchmark["truth_disparity"]
            )
            assert error[retained].mean() == pytest.approx(
                benchmark["retained_mean_absolute_error"], abs=1e-6
            )
            call = result.outputs["receipt"]["calls"][0]
            assert call["configuration_verified"] and call["selected_execution"]["engine"] == "new"
            assert call["model_import"]["engine_argument"] == 2
        elif valid.any():
            np.testing.assert_allclose(
                disparity[valid], 1.5 if case == "subpixel" else 3.0, atol=8e-6
            )
        if case == "plane":
            cloud = session.graph.blocks["cloud"].last_result
            assert cloud is not None and cloud.outputs["count"] == int(valid.sum())
            assert result.outputs["rectification"]["units"] == "m"
            np.testing.assert_allclose(cloud.outputs["cloud"].points[:, 2], 8 / 3, atol=8e-6)
        elif case == "right_mask":
            # The right-view strip removes corresponding left positions, not x=8..11.
            assert not valid[:, 12:14].any() and valid[:, 7].all()
        elif case == "empty":
            assert result.outputs["info"]["native_calls"] == 0
        if profile == "lean":
            assert result.outputs["preview"] is NO_VALUE and not result.graphics
            assert session.graph.blocks["filmstrip"].last_result is None
        else:
            np.testing.assert_array_equal(
                result.outputs["preview"],
                session.graph.blocks["filmstrip"].last_result.outputs["image"],
            )


def test_exact_regeneration_isolated_helper_and_preserved_assets(tmp_path, monkeypatch):
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
    (tmp_path / "assets/left.png").write_bytes(b"preserved image")
    with pytest.raises(FileExistsError, match="preserved"):
        build()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit HITNET preparation required")
def test_independent_native_benchmark_and_offline_preparation(tmp_path, monkeypatch):
    actual = runpy.run_path(str(BASE / "benchmark.py"))["benchmark"]()
    assert actual == BENCH
    for name in ("native_8", "native_16"):
        assert actual["cases"][name]["mean_absolute_error"] < 0.04
        assert actual["cases"][name]["bad1_fraction"] == 0
    assert 0.1 < actual["cases"]["native_zero"]["mean_absolute_error"] < 0.3
    helper = runpy.run_path(str(BASE / "prepare_model.py"))
    monkeypatch.setattr(helper["urllib"].request, "urlopen", lambda *a, **k: pytest.fail("network"))
    before = [p.stat().st_mtime_ns for p in MODEL.parent.iterdir()]
    helper["prepare"]()
    assert before == [p.stat().st_mtime_ns for p in MODEL.parent.iterdir()]
    (tmp_path / "contract.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        helper["prepare"](directory=tmp_path)


@pytest.mark.parametrize("case", ["plane", "native_8"])
def test_sealed_save_as_and_xml_preserve_pinned_files_and_reprojection(tmp_path, case):
    if case.startswith("native_") and not MODEL.exists():
        pytest.skip("Explicit HITNET preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        relative = session.graph.blocks["estimate"].parameters["contract_path"]
        session.save(tmp_path / "project.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "project.xml")
    for name in (
        relative,
        str(Path(relative).with_name("model.onnx")),
        "MODEL-LICENSE.txt",
        "SOURCES.md",
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
            assert result.outputs["info"]["units"] == "source_pixels"
        finally:
            graph.close()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit HITNET preparation required")
def test_cold_native_background_worker_stops_cleanly():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'native_8.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='once'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(25000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and len(frames)==1
 result=graph.blocks['estimate'].last_result
 assert result is not None and result.passed and result.outputs['info']['native_calls']==1
 print(json.dumps({'iterations':worker.iterations,'shape':list(result.outputs['disparity'].shape)}))
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "terminate handler" not in result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "iterations": 1,
        "shape": [240, 320],
    }
