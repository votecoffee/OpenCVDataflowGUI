"""Image-derived rank references, saved project portability and cold native execution."""

import base64
import copy
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
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/native_embedding_index"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_projects_image_metrics_and_declared_failures(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        for step in range(6 if case.startswith("train_") else 1 if case == "reject" else 2):
            graph.run()
            block = graph.blocks["index" if case.startswith("train_") else "search"]
            result = block.last_result
            if case.startswith("train_"):
                if step < 5:
                    assert not block.current_emissions
                else:
                    assert result is not None and result.outputs["count"] == 6
                    assert result.outputs["model"]["item_ids"] == TRUTH["training"]
            elif case == "wrong_contract":
                assert (
                    result is None
                    and not block.current_emissions
                    and "preprocessing" in block.error
                )
            else:
                assert result is not None
                assert result.passed == (case != "reject")
                model = graph.blocks["model"].last_result.outputs["model"]
                descriptor = graph.blocks["descriptor"].last_result.outputs[
                    "hash" if case == "hamming" else "descriptor"
                ]
                if case == "hamming":
                    q = np.frombuffer(bytes.fromhex(descriptor["hex"]), np.uint8)
                    measurements = np.array(
                        [
                            sum((int(a) ^ int(b)).bit_count() for a, b in zip(row, q))
                            for row in model["samples"]
                        ]
                    )
                else:
                    q = np.array(descriptor, np.float64)
                    values = np.array(model["samples"], np.float64)
                    if case == "normalized":
                        q /= np.linalg.norm(q)
                    if case in ("euclidean", "normalized", "reject"):
                        measurements = np.sqrt(((values - q) ** 2).sum(axis=1))
                    elif case == "manhattan":
                        measurements = np.abs(values - q).sum(axis=1)
                    elif case == "dot":
                        measurements = values @ q
                    else:
                        cosine = (values @ q) / (np.linalg.norm(values, axis=1) * np.linalg.norm(q))
                        measurements = np.sqrt(np.maximum(0, 2 - 2 * np.clip(cosine, -1, 1)))
                if case == "reject":
                    assert result.outputs["count"] == 0 and result.outputs["item_ids"] == [None]
                    assert measurements.min() > 0
                else:
                    expected = sorted(
                        range(6),
                        key=lambda i: (
                            -measurements[i] if case == "dot" else measurements[i],
                            model["item_ids"][i],
                        ),
                    )[:3]
                    neighbors = result.outputs["neighbors"][0]
                    assert [n["index"] for n in neighbors] == expected
                    np.testing.assert_allclose(
                        [n["similarity"] if case == "dot" else n["distance"] for n in neighbors],
                        measurements[expected],
                        atol=2e-7,
                    )
                    assert result.outputs["labels"] == ["vertical" if step == 0 else "horizontal"]
                source = graph.blocks["source"].last_result
                assert ticket_key(result.payloads["neighbors"]["ticket"]) == ticket_key(
                    source.outputs["ticket"]
                )
                np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
                assert bool(result.drawings) == (profile == "debug")
    finally:
        graph.close()


def test_native_bundle_bytes_pins_and_exact_authoring_regeneration(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError):
        build(tmp_path)
    model = json.loads((BASE / "assets/euclidean.json").read_bytes())
    assert (
        hashlib.sha256(base64.b64decode(model["native"]["data"])).hexdigest()
        == model["native"]["sha256"]
    )


def test_saved_project_xml_relocation_is_self_contained_and_model_handles_stay_private(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "euclidean.json", registry)
    try:
        graph.run()
        first = copy.deepcopy(graph.blocks["search"].last_result.outputs["neighbors"])
        assert block_entry(type(graph.blocks["search"])).path[:2] == (
            "OpenCV",
            "Features Framework (features)",
        )
        graph.save(tmp_path / "moved.json")
        export_project_xml(graph.to_dict(), tmp_path / "graph.xml")
    finally:
        graph.close()
    for graph in [
        Graph.load(tmp_path / "moved.json", registry),
        Graph.from_dict(read_project_xml(tmp_path / "graph.xml"), registry, project_dir=tmp_path),
    ]:
        try:
            assert graph.blocks["search"]._handle is None
            graph.run()
            assert graph.blocks["search"].last_result.outputs["neighbors"] == first
        finally:
            graph.close()


def test_failed_source_admission_keeps_prior_input_and_emits_nothing(monkeypatch):
    graph = Graph.load(BASE / "euclidean.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["search"]
        previous = block.last_inputs["image"].copy()
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed)
        graph.run()
        assert block.input_rejected and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()


def test_cold_run_all_releases_native_mmaps():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,'all',registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==2
 assert graph.blocks['search'].last_result.outputs['labels']==['horizontal']
finally:graph.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(BASE / "hamming.json")],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
