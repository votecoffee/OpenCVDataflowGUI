"""Native graph contracts, independent head/geometry references and bounded failures."""

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.native_lightglue import NativeALIKEDFeatures, NativeLightGlueMatch
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/native_lightglue"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.fixture
def graph():
    value = Graph.load(BASE / "translation.json", builtin_registry())
    yield value
    value.close()


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("verbosity", ["debug", "lean"])
def test_saved_projects_have_explicit_positive_negative_and_error_outcomes(case, verbosity):
    value = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        value.set_run_profile(verbosity)
        value.run()
        block = value.blocks["match"]
        result = block.last_result
        if case == "pair_budget":
            assert result is None and not block.current_emissions and "attention" in block.error
        else:
            assert result is not None
            assert result.passed == (case not in ("empty", "score_rejection"))
            assert result.outputs["count"] == (
                0
                if case == "empty"
                else 1
                if case == "score_rejection"
                else 6
                if case == "selected"
                else 7
            )
            assert (result.outputs["pose"] is NO_VALUE) != result.passed
            if result.passed:
                matrix = np.array(result.outputs["pose"]["source_to_target"])
                np.testing.assert_allclose(
                    matrix, [[1, 0, TRUTH["translation_x"]], [0, 1, 0], [0, 0, 1]], atol=2e-5
                )
            if case == "high_depth":
                assert result.outputs["image"].dtype == np.uint16
                np.testing.assert_array_equal(
                    result.outputs["image"], value.blocks["query"].last_result.outputs["image"]
                )
    finally:
        value.close()


def test_direct_onnx_and_native_matcher_agree_on_scores_pairs_and_pose(graph):
    graph.run()
    r = graph.blocks["reference_features"].last_result.outputs["features"]
    q = graph.blocks["query_features"].last_result.outputs["features"]
    result = graph.blocks["match"].last_result
    net = cv2.dnn.readNetFromONNX(str(BASE / "assets/matcher.onnx"), cv2.dnn.ENGINE_NEW)
    for index, features in enumerate((r, q)):
        # Independent preparation, without the application's validation/matching helpers.
        xy = np.float32(features["points"])
        normalized = xy / np.float32(features["image_size"]) * 2 - 1
        net.setInput(normalized[None], f"kpts{index}")
        net.setInput(np.float32(features["descriptors"])[None], f"desc{index}")
    pairs, scores = net.forward(["matches0", "mscores0"])
    selected = scores.ravel() >= 0.1
    report = result.outputs["report"]
    np.testing.assert_array_equal(report["candidate_pairs"], pairs)
    np.testing.assert_allclose(report["candidate_scores"], scores.ravel(), atol=6e-8)
    np.testing.assert_array_equal(report["rejected_pairs"], pairs[~selected])
    p0 = np.float32(r["points"])[pairs[selected, 0]]
    p1 = np.float32(q["points"])[pairs[selected, 1]]
    matrix, flags = cv2.findHomography(p0, p1, cv2.RANSAC, 3, maxIters=2000, confidence=0.995)
    np.testing.assert_allclose(result.outputs["pose"]["source_to_target"], matrix, atol=1e-8)
    assert result.outputs["inliers"] == int(flags.sum()) == 7
    assert report["rejected_scores"][0] < 0.1


def test_source_depth_ticket_graphics_cache_and_empty_context_lifecycle(graph):
    graph.run()
    block = graph.blocks["match"]
    assert block.last_result.drawings
    assert block.last_result.outputs["report"]["native_model_reused"] is False
    first = block._matcher
    graph.run()
    assert block._matcher is first and block.last_result.outputs["report"]["native_model_reused"]
    assert graph.blocks["query_features"].last_result.outputs["report"]["native_model_reused"]
    result = block.last_result
    assert ticket_key(result.payloads["matches"]["ticket"]) == ticket_key(
        graph.blocks["query"].last_result.outputs["ticket"]
    )
    np.testing.assert_array_equal(
        result.outputs["image"], graph.blocks["query"].last_result.outputs["image"]
    )
    old = copy.deepcopy(block.last_inputs)
    block.run_profile = profile("lean")
    lean = block.execute(old)
    assert lean is not None and not lean.drawings and not lean.graphics
    block.reset_runtime()
    assert block._matcher is None
    block.close()
    assert block._matcher is None


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (NativeALIKEDFeatures, {"model_sha256": "invalid"}),
        (NativeALIKEDFeatures, {"normalize_descriptors": 1}),
        (NativeALIKEDFeatures, {"width": 33.0}),
        (NativeALIKEDFeatures, {"maximum_network_pixels": 1}),
        (NativeALIKEDFeatures, {"minimum_count": 10, "maximum_features": 8}),
        (NativeLightGlueMatch, {"maximum_attention_pairs": True}),
        (NativeLightGlueMatch, {"feature_family": "sift"}),
        (NativeLightGlueMatch, {"model_path": 7}),
        (NativeLightGlueMatch, {"ransac_confidence": 1}),
    ],
)
def test_parameter_fields_validate_without_loading_models(cls, parameters):
    with pytest.raises(ValidationError):
        cls(parameters=parameters).validate_parameters()


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "model_budget",
        "row_budget",
        "pixel_budget",
        "family",
        "nonunit",
        "ticket",
        "grid",
        "recipe",
    ],
)
def test_invalid_input_is_gated_before_native_loading_and_recovers(graph, problem):
    graph.run()
    source = graph.blocks["match"]
    inputs = copy.deepcopy(source.last_inputs)
    block = NativeLightGlueMatch(parameters=copy.deepcopy(source.parameters))
    block.project_dir = source.project_dir
    if problem == "hash":
        block.parameters["model_sha256"] = "0" * 64
    elif problem == "model_budget":
        block.parameters["maximum_model_bytes"] = 1
    elif problem == "row_budget":
        block.parameters["maximum_features"] = 4
    elif problem == "pixel_budget":
        block.parameters["maximum_image_pixels"] = 1
    elif problem == "family":
        block.parameters["feature_family"] = "disk"
    elif problem == "nonunit":
        inputs["query_features"]["descriptors"][0] = [0] * 128
    else:
        body = inputs["query_features"]["coordinate_context"]
        if problem == "ticket":
            body["ticket"] = TicketClock().issue("other")
        elif problem == "grid":
            body["spatial"] = frame([64, 48], "other")
        else:
            body["recipe_ref"] = {"id": "p", "revision": 1, "digest": "1" * 64}
            inputs["image"] = ValuePacket(
                inputs["image"], {"recipe_ref": {"id": "p", "revision": 2, "digest": "1" * 64}}
            )
    assert block.execute(inputs) is None and not block.current_emissions
    assert block._matcher is None
    block.parameters = copy.deepcopy(source.parameters)
    assert block.execute(copy.deepcopy(source.last_inputs)).passed
    block.close()


def test_source_roi_threshold_raw_descriptor_and_depth_rules(graph):
    source = graph.blocks["query"]
    source.execute({})
    image = source.last_result.outputs["image"]
    settings = graph.blocks["query_features"].parameters
    block = NativeALIKEDFeatures(parameters=settings)
    block.project_dir = graph.project_dir
    full = block.execute({"image": image})
    assert full.passed and full.outputs["count"] == 8
    block.parameters["normalize_descriptors"] = False
    assert block.execute({"image": image}).outputs["features"]["model"] == "native_aliked_raw_v1"
    block.parameters["normalize_descriptors"] = True
    transparent = ImageArray(
        np.dstack((image[..., ::-1], np.zeros(image.shape[:2], np.uint8))), "rgba"
    )
    result = block.execute({"image": transparent})
    assert result.outputs["count"] == 0
    np.testing.assert_array_equal(result.outputs["image"], transparent)
    block.parameters["alpha_policy"] = "ignore"
    assert block.execute({"image": transparent}).outputs["count"] == 8
    block.parameters["alpha_policy"] = "exclude"
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:30] = 255
    assert block.execute({"image": image, "mask": mask}).outputs["count"] == 6
    block.parameters["maximum_features"] = 7
    assert block.execute({"image": image}) is None and not block.current_emissions
    block.parameters["maximum_features"] = 1024
    block.parameters["maximum_image_pixels"] = 1
    assert block.execute({"image": image}) is None
    block.parameters["maximum_image_pixels"] = 4_194_304
    assert block.execute({"image": image.astype(np.float32)}) is None
    block.close()


def test_project_xml_save_as_regeneration_and_toolbox(tmp_path, graph):
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path / "generated")
    for path in (tmp_path / "generated").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes() == (BASE / path.relative_to(tmp_path / "generated")).read_bytes()
            )
    with pytest.raises(FileExistsError):
        build(tmp_path / "generated")
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    xml = tmp_path / "graph.xml"
    export_project_xml(graph.to_dict(), xml)
    restored = Graph.from_dict(
        read_project_xml(xml), builtin_registry(), project_dir=graph.project_dir
    )
    try:
        restored.run()
        assert restored.blocks["match"].last_result.passed
        for block in (NativeLightGlueMatch, NativeALIKEDFeatures):
            assert block_entry(block).path[:2] == ("OpenCV", "Features Framework (features)")
    finally:
        restored.close()
    graph.save(tmp_path / "copied.json")
    reopened = Graph.load(tmp_path / "copied.json", builtin_registry())
    try:
        reopened.run()
        assert reopened.blocks["match"].last_result.passed
    finally:
        reopened.close()


def test_import_uses_a_hashed_private_snapshot_not_the_mutable_source(tmp_path, monkeypatch, graph):
    graph.run()
    block = graph.blocks["match"]
    path = tmp_path / "caller.onnx"
    original = (BASE / "assets/matcher.onnx").read_bytes()
    path.write_bytes(original)
    block.parameters["model_path"] = str(path)
    block.reset_runtime()
    create = cv2.LightGlueMatcher.create

    def load(filename, *args):
        assert Path(filename) != path
        assert Path(filename).read_bytes() == original
        path.write_bytes(b"caller changed after snapshot")
        return create(filename, *args)

    monkeypatch.setattr(cv2, "LightGlueMatcher", SimpleNamespace(create=load))
    assert block.execute(copy.deepcopy(block.last_inputs)).passed
    assert block.execute(copy.deepcopy(block.last_inputs)) is None
    assert not block.current_emissions and "SHA-256" in block.error


def test_native_failure_discards_pair_context_and_recovers(monkeypatch, graph):
    graph.run()
    block = graph.blocks["match"]
    inputs = copy.deepcopy(block.last_inputs)
    create = cv2.LightGlueMatcher.create

    def broken(*args):
        raise RuntimeError("native import failure")

    block.reset_runtime()
    monkeypatch.setattr(cv2, "LightGlueMatcher", SimpleNamespace(create=broken))
    assert block.execute(inputs) is None and not block.current_emissions
    assert "native import failure" in block.error and block._matcher is None
    monkeypatch.setattr(cv2, "LightGlueMatcher", SimpleNamespace(create=create))
    assert block.execute(inputs).passed


def test_failed_input_does_not_replace_last_processed_image(graph, monkeypatch):
    graph.run()
    block = graph.blocks["match"]
    previous = block.last_inputs["image"].copy()
    block.input_failure_policies = {"image": "reject_failed"}
    original = graph.blocks["query"].process

    def failed(inputs):
        result = original(inputs)
        result.passed = False
        return result

    monkeypatch.setattr(graph.blocks["query"], "process", failed)
    graph.run()
    assert block.input_rejected and not block.current_emissions
    np.testing.assert_array_equal(block.last_inputs["image"], previous)


def test_cold_desktop_worker_executes_native_pair_and_tears_down():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,'once',registry);errors=[]
 worker.frame.connect(lambda:worker.ack.set(),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==1 and graph.blocks['match'].last_result.passed
 assert graph.blocks['match'].last_result.outputs['count']==7
finally:graph.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(BASE / "translation.json")],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
