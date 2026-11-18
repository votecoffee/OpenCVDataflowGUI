"""Runnable native tracking clips, exact portability and background lifecycle proof."""

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

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_tracking"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", TRUTH["cases"])
def test_saved_clip_current_outputs_score_loss_reseed_and_observation_policy(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    records = []
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        while not graph.sources_exhausted():
            assert len(records) < 15
            graph.run()
            block = graph.blocks["track"]
            result = block.last_result
            if case == "resource_limit":
                assert (
                    result is None and block.error_kind == "invalid" and not block.current_emissions
                )
                assert not block.memory["streams"]
                records.append(None)
                continue
            assert result is not None, block.error
            status = result.outputs["status"]
            records.append((status, result.passed, result.outputs["ready"]))
            np.testing.assert_array_equal(
                result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
            )
            assert result.payloads["image"]["ticket"] == status["current_ticket"]
            assert (
                result.outputs["receipt"]["configuration_verified"]
                and result.outputs["receipt"]["preprocessing_verified"]
            )
            assert result.outputs["receipt"]["selected_execution"]["engine"] == (
                "new" if case == "new_engine" else "classic"
            )
            assert result.outputs["overlay"] is NO_VALUE and bool(result.drawings) == (
                profile == "debug"
            )
            if status["phase"] in ("initialized", "seed_required"):
                assert result.outputs["tracking_score"] is NO_VALUE and result.score is None
            else:
                assert result.score == result.outputs["tracking_score"] == status["tracking_score"]
            if not result.outputs["ready"]:
                assert result.outputs["bounds"] is NO_VALUE and result.outputs["pose"] is NO_VALUE
                assert result.outputs["tracked_region"] is NO_VALUE
            elif case in ("translation", "new_engine", "region_seed", "grayscale"):
                box = result.outputs["bounds"]
                expected = TRUTH["annotations"][len(records) - 1]["bounds"]
                iou = runpy.run_path(str(BASE / "benchmark.py"))["iou"](box, expected)
                assert iou > 0.8
                assert (
                    result.outputs["tracked_region"]["coordinate_context"]
                    == result.payloads["bounds"]["spatial"]
                )
        if case in ("translation", "new_engine", "region_seed", "grayscale"):
            assert [r[2] for r in records] == [False, False] + [True] * 5
            assert all(r[1] for r in records)
        elif case == "loss_expiry":
            assert [r[0]["phase"] for r in records[7:]] == [
                "lost",
                "lost",
                "seed_required",
                "seed_required",
                "seed_required",
            ]
            assert all(not r[1] and not r[2] for r in records[7:])
        elif case == "detector_reseed":
            assert [r[0]["phase"] for r in records[4:]] == [
                "lost",
                "seed_required",
                "initialized",
                "tracking",
                "tracking",
            ]
            assert not records[4][1] and not records[5][1]
            assert records[0][0]["track_id"] != records[6][0]["track_id"]
            assert records[-1][2]
        elif case in ("score_alarm", "jump_alarm"):
            assert all(not r[1] and not r[2] for r in records[1:])
            assert records[1][0]["reason"] == (
                "tracking_score_below_threshold"
                if case == "score_alarm"
                else "center_jump_above_maximum"
            )
    finally:
        graph.close()


def test_exact_regeneration_and_offline_model_verification_preserve_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(SHA256="wrong"))
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
    helper = runpy.run_path(str(BASE / "prepare_model.py"))
    monkeypatch.setattr(helper["urllib"].request, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert helper["prepare"]() == BASE / "assets/vit.onnx"
    (tmp_path / "assets/frame_00.png").write_bytes(b"preserved image")
    with pytest.raises(FileExistsError, match="preserved"):
        build()
    (tmp_path / "different.onnx").write_bytes(b"preserved weights")
    with pytest.raises(FileExistsError, match="preserved"):
        helper["prepare"](destination=tmp_path / "different.onnx")


def test_independent_native_annotated_benchmark():
    actual = runpy.run_path(str(BASE / "benchmark.py"))["benchmark"]()
    assert actual == BENCH
    assert actual["minimum_iou"] > 0.93 and actual["maximum_center_error_pixels"] < 1.2
    assert not actual["blank_located"] and actual["blank_score"] < actual["threshold"]
    assert not actual["score_is_probability"]


def test_sealed_save_as_xml_and_opening_do_not_persist_or_create_live_model(tmp_path, monkeypatch):
    with ProjectSession.open(BASE / "translation.json") as session:
        block = session.graph.blocks["track"]
        assert not block.memory["streams"]
        session.graph.run()
        assert block.memory["streams"]
        session.save(tmp_path / "program.json")
        export_project_xml(session.graph.to_dict(), tmp_path / "program.xml")
        contract = block.parameters["contract_path"]
    for name in (
        contract,
        str(Path(contract).with_name("model.onnx")),
        "SOURCES.md",
        "MODEL-LICENSE.txt",
    ):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            block = graph.blocks["track"]
            assert not block.memory["streams"]
            graph.run()
            result = block.last_result
            assert result is not None, block.error
            assert result.outputs["status"]["phase"] == "initialized"
            assert result.outputs["receipt"]["artifact"]["sha256"] == TRUTH["model_sha256"]
        finally:
            graph.close()


def test_model_and_configuration_tampering_are_not_silent_retraining(tmp_path):
    with ProjectSession.open(BASE / "translation.json") as session:
        session.save(tmp_path / "program.json")
    graph = Graph.load(tmp_path / "program.json", builtin_registry())
    try:
        block = graph.blocks["track"]
        block.parameters["mean"] = [0, 0, 0]
        graph.run()
        assert block.last_result is None and "settings differ" in block.error
        block.parameters["mean"] = [0.485, 0.456, 0.406]
        path = tmp_path / Path(block.parameters["contract_path"]).with_name("model.onnx")
        path.write_bytes(b"changed weights")
        graph.run()
        assert block.last_result is None and "checksum" in block.error
        assert not block.memory["streams"]
    finally:
        graph.close()


def test_cold_native_run_all_and_teardown_have_one_frame_event_per_input():
    code = """
import json,sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'translation.json',registry)
try:
 mode=MainWindow.run_all_mode(SimpleNamespace(root_graph=graph));assert mode=='all'
 worker=Runner(graph,mode,registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('Worker timeout')
 assert not errors,errors
 assert worker.iterations==7 and len(frames)==7
 result=graph.blocks['track'].last_result
 assert result is not None and result.passed and result.outputs['ready']
 print(json.dumps({'iterations':worker.iterations,'phase':result.outputs['status']['phase']}))
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
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "iterations": 7,
        "phase": "tracking",
    }
