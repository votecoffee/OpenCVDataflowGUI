"""Native saved-project evidence, exact source geometry and portable frozen models."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/super_resolution"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCHMARK = json.loads((BASE / "benchmark.json").read_bytes())


def module(name):
    spec = importlib.util.spec_from_file_location("super_resolution_" + name, BASE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_projects_pixels_quality_current_outcomes_and_root_geometry(case, profile):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        learned = session.graph.blocks["learned"]
        result = learned.last_result
        assert result is not None and result.passed, learned.error
        original = session.graph.blocks["source"].last_result
        source = original.outputs["image"]
        assert result.outputs["image"].shape == source.shape
        assert result.outputs["image"].dtype == source.dtype
        payload = result.payloads["image"]
        assert payload["ticket"] == original.payloads["image"]["ticket"]
        np.testing.assert_allclose(payload["spatial"]["to_root"], np.eye(3))
        assert payload["spatial"]["root_size"] == [256, 192]
        current = result.outputs["image"].astype(np.float64)
        mae = np.abs(current - source).mean()
        quality = session.graph.blocks["learned_error"].last_result
        assert quality is not None and quality.passed == (case != "fidelity_alarm")
        assert quality.outputs["value"] == pytest.approx(mae, abs=1e-8)
        baseline = result.outputs["baseline"].astype(np.float64)
        expected = np.abs(baseline - source).mean()
        assert session.graph.blocks["baseline_error"].last_result.outputs["value"] == pytest.approx(
            expected
        )
        np.testing.assert_array_equal(
            result.outputs["image"], session.graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        assert result.outputs["info"]["native_calls"] == (0 if case == "empty_region" else 1)
        if case == "empty_region":
            np.testing.assert_array_equal(current, baseline)
        if case in ("geometry", "noise"):
            clean = cv2.imread(str(BASE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
            measured = module("benchmark").measurements(result.outputs["image"], clean, TRUTH)
            reference = BENCHMARK["cases"]["reference" if case == "geometry" else "noise"][
                "learned"
            ]
            assert measured.keys() == reference.keys()
            for key, value in measured.items():
                assert value == pytest.approx(reference[key], abs=1e-7), key


def test_retained_native_benchmark_matches_independent_geometry_and_exposes_lost_detail():
    benchmark = module("benchmark").benchmark()
    assert benchmark == BENCHMARK
    original = cv2.imread(str(BASE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    y, x = np.indices(original.shape)
    circle = (x - 70) ** 2 + (y - 88) ** 2 <= 36**2
    x0, y0, width, height = TRUTH["circle_roi"]
    np.testing.assert_array_equal(
        original[y0 : y0 + height, x0 : x0 + width] < 110, circle[y0 : y0 + height, x0 : x0 + width]
    )
    assert benchmark["reference"]["circle_width_height"] == [73, 73]
    assert benchmark["reference"]["crack_widths_at_y157"] == [1, 2, 3, 4]
    assert benchmark["reference"]["bead_gap_columns"] == 2
    learned = benchmark["cases"]["reference"]["learned"]
    assert learned["crack_widths_at_y157"][0] == 0
    assert learned["circle_width_height"] == [71, 71]
    assert learned["erased_dark_detail_pixels"] > 0
    assert learned["flat_changed_pixels"] > 0
    assert learned["mae"] < benchmark["cases"]["reference"]["cubic"]["mae"]
    assert (
        benchmark["cases"]["noise"]["learned"]["flat_std"]
        > benchmark["cases"]["noise"]["linear"]["flat_std"]
    )


def test_exact_example_regeneration_and_differing_asset_preservation(tmp_path):
    builder = module("build_example")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/FSRCNN_x2.pb").write_bytes((BASE / "assets/FSRCNN_x2.pb").read_bytes())
    builder.HERE = tmp_path
    builder.build()
    for path in [*BASE.glob("*.json"), *(BASE / "assets").iterdir()]:
        if path.name == "benchmark.json":
            continue
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path.name
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    (tmp_path / "assets/reference.png").write_bytes(b"preserve existing image")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_save_as_copies_exact_model_license_and_json_xml_replay(tmp_path):
    with ProjectSession.open(BASE / "geometry.json") as session:
        session.save(tmp_path / "program.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "program.xml")
    for name in ("assets/FSRCNN_x2.pb", "MODEL-LICENSE.txt", "SOURCES.md"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            graph.run()
            result = graph.blocks["learned"].last_result
            assert result is not None and result.passed
            assert result.outputs["info"]["artifact"]["sha256"] == TRUTH["model_sha256"]
        finally:
            graph.close()


def test_absolute_external_model_stays_linked_and_offline_reference_is_available(tmp_path):
    from opencv_dataflow.opencv_reference import opencv_reference

    with ProjectSession.open(BASE / "geometry.json") as session:
        block = session.graph.blocks["learned"]
        block.parameters["model_path"] = str((BASE / "assets/FSRCNN_x2.pb").resolve())
        reference = opencv_reference(block)
        assert "upsample(img" in reference and "readModel(path)" in reference
        session.save(tmp_path / "program.json")
    assert not (tmp_path / "assets/FSRCNN_x2.pb").exists()
    with ProjectSession.open(tmp_path / "program.json") as session:
        session.graph.run()
        assert session.graph.blocks["learned"].last_result.passed


def test_cold_background_worker_and_native_teardown_exit_cleanly():
    # A fresh interpreter matters: main-thread OpenCL initialization in earlier
    # tests masks a driver teardown crash from an unnecessary worker-side query.
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([])
registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'geometry.json',registry)
try:
    mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph))
    assert mode=='once'
    worker=Runner(graph,mode,registry);errors=[];frames=[]
    worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
    worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection)
    worker.start()
    if not worker.wait(20000):
        worker.stop();worker.ack.set();worker.wait(5000)
        raise AssertionError('Worker did not finish')
    assert not errors,errors
    assert worker.iterations==1 and len(frames)==1
    result=graph.blocks['learned'].last_result
    assert result is not None and result.passed and result.outputs['image'].shape==(192,256)
    print(json.dumps({'iterations':worker.iterations,'native_calls':result.outputs['info']['native_calls']}))
finally:
    graph.close()
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
        "native_calls": 1,
    }
