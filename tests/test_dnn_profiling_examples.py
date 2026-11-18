"""Pinned CNN arithmetic, observation controls, portable graphs and cold worker lifecycle."""

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
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/dnn_profiling"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def expected(image, multiplier=1):
    pixels = cv2.resize(image.astype(np.float32), (16, 16), interpolation=cv2.INTER_LINEAR)[
        ..., ::-1
    ]
    midpoint = 32767.5 if image.dtype == np.uint16 else 127.5
    prepared = (pixels.astype(np.float32) - midpoint) / midpoint * multiplier
    return np.maximum(prepared + np.float32([0.15, -0.1, 0.05]), 0).mean(axis=(0, 1))


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_profiles_keep_image_arithmetic_tickets_and_observation_semantics(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        b = graph.blocks["profile"]
        if case == "required_budget":
            assert b.last_result is None and not b.current_emissions
            assert "profiling unavailable" in b.error and b._net is None
            return
        result = b.last_result
        assert result is not None, b.error
        data = result.outputs["profile"]
        assert data["status"] == (
            "disabled"
            if case == "disabled"
            else "unavailable"
            if case == "optional_budget"
            else "available"
        )
        if case == "batch_tail":
            assert [c["batch"]["real_rows"] for c in data["calls"]] == [2, 1]
            rows = result.outputs["results"].items
            sources = [graph.blocks[k].last_result for k in ("source", "source_1", "source_2")]
            for row, source in zip(rows, sources, strict=True):
                np.testing.assert_allclose(
                    selected_tensor(row.value["tensors"])[0],
                    expected(source.outputs["image"]),
                    atol=2e-6,
                )
                assert ticket_key(row.payload["ticket"]) == ticket_key(source.outputs["ticket"])
        else:
            source = graph.blocks["source"].last_result
            image = source.outputs["image"]
            np.testing.assert_allclose(
                selected_tensor(result.outputs["tensors"])[0],
                expected(image, 2 if case == "multi_input" else 1),
                atol=2e-6,
            )
            assert ticket_key(result.payloads["profile"]["ticket"]) == ticket_key(
                source.outputs["ticket"]
            )
            if case != "multi_input":
                np.testing.assert_array_equal(result.outputs["image"], image)
                assert result.outputs["image"].dtype == image.dtype
        chart = profile == "debug" and case not in ("disabled", "plot_off", "optional_budget")
        assert bool(result.graphics) == chart
        for call in data["calls"]:
            assert call["selected_execution"]["target"] == "cpu"
            assert call["native_placement_verified"] is False
            if call["status"] == "available":
                assert call["total_ticks"] > 0 and call["layers"]
                assert call["native_total_ms"] > 0 and call["forward_wall_ms"] > 0
        if case.startswith("classic_"):
            call = data["calls"][0]
            assert call["selected_execution"]["fusion"] == (case == "classic_fused")
            assert call["native_layer_id_base"] == 1
            assert call["zero_tick_layers"] == (1 if case == "classic_fused" else 0)
        if case == "disabled":
            assert data["native_call_count"] is None and data["observed_native_call_count"] == 0
            assert data["native_total_ms"] is None and data["forward_wall_ms"] is None
    finally:
        graph.close()


def test_exact_asset_pins_authoring_regeneration_and_no_overwrite(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError):
        build(tmp_path)


def test_sealed_graph_save_as_xml_both_bundle_files_and_no_persisted_runtime(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "sealed.json", registry)
    try:
        graph.run()
        expected_tensors = graph.blocks["profile"].last_result.outputs["tensors"]
        assert block_entry(graph.blocks["profile"]).path[:2] == (
            "OpenCV",
            "Deep Neural Networks (dnn)",
        )
        graph.save(tmp_path / "moved.json")
        export_project_xml(graph.to_dict(), tmp_path / "graph.xml")
        for name in ("model.onnx", "contract.json"):
            assert (tmp_path / "bundle" / name).read_bytes() == (
                BASE / "bundle" / name
            ).read_bytes()
    finally:
        graph.close()
    for graph in (
        Graph.load(tmp_path / "moved.json", registry),
        Graph.from_dict(read_project_xml(tmp_path / "graph.xml"), registry, project_dir=tmp_path),
    ):
        try:
            b = graph.blocks["profile"]
            assert b._net is None and not b._profile_calls
            graph.run()
            assert b.last_result.outputs["tensors"] == expected_tensors
            assert b.last_result.outputs["profile"]["status"] == "available"
        finally:
            graph.close()


def test_failed_admission_keeps_prior_input_cache_and_profile_without_event(monkeypatch):
    graph = Graph.load(BASE / "new_cpu.json", builtin_registry())
    try:
        graph.run()
        b = graph.blocks["profile"]
        previous, last_inputs, native = b.last_result, b.last_inputs, b._net
        b.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed)
        graph.run()
        assert b.input_rejected and not b.current_emissions
        assert b.last_result is previous and b.last_inputs is last_inputs and b._net is native
    finally:
        graph.close()


@pytest.mark.parametrize("case,mode", [("new_cpu", "once"), ("batch_tail", "all")])
def test_cold_offscreen_runner_native_cleanup(case, mode):
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,sys.argv[2],registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 assert graph.blocks['profile'].last_result.outputs['profile']['status']=='available'
finally:graph.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(BASE / (case + ".json")), mode],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
