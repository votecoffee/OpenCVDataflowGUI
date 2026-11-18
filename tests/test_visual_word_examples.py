"""Saved visual-word image workflows and independent held-out retrieval references."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/visual_words"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_visual_word_workflows(case, profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        assert validate_graph(graph)["valid"]
        if "save" in graph.blocks:
            graph.blocks["save"].parameters["path"] = str(tmp_path / "model.json")
        for index, state in enumerate(expected["states"]):
            graph.run()
            target = graph.blocks[expected["target"]]
            assert target.state == state, (case, target.error)
            if state in ("inactive", "error"):
                assert not target.current_emissions
                continue
            if "save" in graph.blocks:
                source = graph.blocks[
                    "fit" if case.startswith("train") else "build"
                ].last_result.outputs["model"]
                assert json.loads((tmp_path / "model.json").read_bytes()) == source
                continue
            result = graph.blocks["rank"].last_result
            assert result is not None and result.score is None and result.final_judgment is None
            assert result.passed is (state == "pass")
            np.testing.assert_array_equal(
                result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
            )
            if state == "pass":
                expected_label = (
                    "washers" if case == "spatial_rearrangement" else TRUTH["labels"][index]
                )
                assert result.outputs["label"] == expected_label
                assert len(result.outputs["results"]) == 3
                assert result.outputs["distance"] < 0.1
            else:
                assert result.outputs["count"] == 0 and result.outputs["distance"] is NO_VALUE
            assert (
                graph.blocks["encode"].last_result.payloads["histogram"]["ticket"]["value"]
                == expected["images"][index]
            )
            if profile == "debug":
                assert result.drawings and graph.blocks["encode"].last_result.graphics
                displayed = graph.blocks["overlay"].last_result.outputs["image"]
                assert np.any(displayed != np.repeat(result.outputs["image"][..., None], 3, axis=2))
            else:
                assert not result.drawings and not graph.blocks["encode"].last_result.graphics
    finally:
        graph.close()


def test_regeneration_asset_hashes_disjoint_queries_and_preserved_existing_folder(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }
    assert set(TRUTH["training"]).isdisjoint(TRUTH["queries"])
    hashes = TRUTH["asset_sha256"]
    assert {hashes[n] for n in TRUTH["training"]}.isdisjoint(hashes[n] for n in TRUTH["queries"])
    for name, digest in hashes.items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build.__globals__["HERE"] = tmp_path
    build()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build()
    assert before == {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file()
    }


def test_independent_native_and_numpy_image_retrieval_references():
    benchmark = runpy.run_path(str(BASE / "benchmark.py"))["measure"]()
    registry = builtin_registry()
    for mode, rows in benchmark["cases"].items():
        method, normalization = mode.split("-")
        graph = Graph.load(
            BASE
            / ("hellinger.json" if normalization == "hellinger" else f"retrieve_{method}.json"),
            registry,
        )
        try:
            for index, row in enumerate(rows[:3]):
                graph.run()
                actual = graph.blocks["rank"].last_result.outputs
                assert row["assignment_disagreements"] == 0
                assert row["maximum_distance_error"] < 1e-4
                assert row["ranking"][0]["label"] == TRUTH["labels"][index]
                np.testing.assert_allclose(
                    graph.blocks["encode"].last_result.outputs["histogram"],
                    row["histogram"],
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    [v["distance"] for v in actual["results"]],
                    [v["distance"] for v in row["ranking"]],
                    atol=1e-14,
                )
            assert rows[-1]["ranking"][0]["label"] == "washers"
        finally:
            graph.close()


def test_json_xml_save_as_keep_codebook_and_gallery_bytes_and_results(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "retrieve_sift.json", registry)
    try:
        graph.save(tmp_path / "program.json")
    finally:
        graph.close()
    reopened = Graph.load(tmp_path / "program.json", registry)
    try:
        for block, asset in (
            ("codebook", "codebook-sift.json"),
            ("gallery", "gallery-sift-l1.json"),
        ):
            assert (tmp_path / reopened.blocks[block].parameters["path"]).read_bytes() == (
                BASE / "assets" / asset
            ).read_bytes()
        export_project_xml(reopened.to_dict(), tmp_path / "program.xml")
    finally:
        reopened.close()
    restored = Graph.from_dict(
        read_project_xml(tmp_path / "program.xml"), registry, project_dir=tmp_path
    )
    try:
        for label in TRUTH["labels"]:
            restored.run()
            assert restored.blocks["rank"].last_result.outputs["label"] == label
    finally:
        restored.close()


def test_missing_and_failed_inputs_do_not_replace_last_input_image_or_emit(monkeypatch):
    graph = Graph.load(BASE / "retrieve_orb.json", builtin_registry())
    try:
        graph.run()
        encode = graph.blocks["encode"]
        previous = encode.last_inputs["image"].copy()
        codebook = graph.blocks["codebook"].last_result.outputs["model"]
        assert encode.execute({"model": codebook, "features": NO_VALUE, "image": previous}) is None
        assert not encode.current_emissions
        np.testing.assert_array_equal(encode.last_inputs["image"], previous)
        encode.input_failure_policies = {"features": "reject_failed"}
        original = graph.blocks["features"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["features"], "process", failed)
        count = encode.timing.count
        graph.run()
        assert encode.input_rejected and encode.timing.count == count
        assert not encode.current_emissions
        np.testing.assert_array_equal(encode.last_inputs["image"], previous)
    finally:
        graph.close()


def test_cold_background_run_all_completes_three_image_queries_and_tears_down():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication, Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([])
registry=builtin_registry()
graph=Graph.load(Path(sys.argv[1])/'retrieve_sift.json',registry)
worker=Runner(graph,'all',registry)
errors=[];frames=[]
worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection)
try:
    worker.start()
    assert worker.wait(25000),'worker timeout'
    assert not errors,errors
    assert worker.iterations==3 and len(frames)==3
    assert graph.blocks['rank'].last_result.outputs['label']=='tabs'
finally:
    worker.stop();worker.ack.set();worker.wait(10000);graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
