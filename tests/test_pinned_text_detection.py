"""Pinned native detector geometry, grouping controls, failure cases and project persistence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.ocr import TextRegions
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/pinned_text_detection"
MODEL = BASE / ".local/ppocr_en/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
native = pytest.mark.skipif(
    not MODEL.exists(), reason="Run the example's explicit prepare_model.py first"
)
COUNTS = {
    "labels": 3,
    "rotated": 3,
    "selected_roi": 1,
    "empty": 0,
    "count_failure": 3,
    "truncated": 1,
    "joined_words": 1,
    "split_words": 2,
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def overlap(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    intersection = cv2.intersectConvexConvex(a, b)[0]
    return intersection / (cv2.contourArea(a) + cv2.contourArea(b) - intersection)


@native
@pytest.mark.parametrize("name", list(COUNTS))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_saved_projects_match_known_text_geometry_and_current_failures(name, run_profile):
    with ProjectSession.open(BASE / (name + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(run_profile)
        session.graph.run()
        assert not any(b.state == "error" for b in session.graph.blocks.values()), {
            b.id: b.error for b in session.graph.blocks.values() if b.state == "error"
        }
        result = session.graph.blocks["decode"].last_result
        assert result is not None
        assert result.outputs["count"] == COUNTS[name]
        assert result.passed == TRUTH["cases"][name].get("passed", True)
        image_name = TRUTH["cases"][name].get("image", "labels")
        annotations = TRUTH["annotations"][image_name]
        found = result.outputs["regions"]["items"]
        if name not in ("selected_roi", "truncated"):
            assert len(found) == len(annotations)
            matched = set()
            for annotation in annotations:
                ranked = sorted(
                    (
                        (overlap(annotation["corners"], region["corners"]), i)
                        for i, region in enumerate(found)
                    ),
                    reverse=True,
                )
                assert ranked[0][0] > 0.55, (name, annotation, ranked)
                assert ranked[0][1] not in matched
                matched.add(ranked[0][1])
        if name == "rotated":
            for region in found:
                delta = np.subtract(region["corners"][1], region["corners"][0])
                assert abs(np.degrees(np.arctan2(delta[1], delta[0])) + 12) < 2
        if name == "truncated":
            assert result.outputs["report"]["accepted_before_limit"] == 3
            assert result.fail_reasons[0]["code"] == "text_regions.truncated"
        assert result.outputs["report"]["context_checks"] == {
            "frame_checked": True,
            "ticket_checked": True,
        }
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        assert result.payloads["regions"]["ticket"] == source.payloads["image"]["ticket"]
        assert result.payloads["regions"]["spatial"] == source.payloads["image"]["spatial"]
        assert bool(result.drawings) == (run_profile == "debug" and name != "empty")
        assert bool(result.graphics) == (run_profile == "debug")
        np.testing.assert_array_equal(
            session.graph.blocks["filmstrip"].last_result.outputs["image"],
            session.graph.blocks["overlay"].last_result.outputs["image"],
        )
        receipt = session.graph.blocks["infer"].last_result.outputs["receipt"]
        assert receipt["model"] == "onnx_inference_receipt_v1"


@native
def test_new_cpu_engine_matches_pinned_classic_vertices_and_confidence():
    with ProjectSession.open(BASE / "labels.json") as session:
        session.graph.run()
        original = session.graph.blocks["decode"].last_result.outputs["regions"]["items"]
        tool = session.graph.blocks["infer"]
        # An explicit unsealed alternative engine experiment; never mutate the pinned bundle.
        tool.parameters.update(contract_path="", model_path=str(MODEL), engine="new")
        session.graph.run()
        assert tool.state == "pass", tool.error
        changed = session.graph.blocks["decode"].last_result.outputs["regions"]["items"]
        assert len(original) == len(changed) == 3
        for left, right in zip(original, changed, strict=True):
            np.testing.assert_allclose(left["corners"], right["corners"], atol=1.5)
            assert left["confidence"] == pytest.approx(right["confidence"], abs=1e-5)


@native
def test_db_rectangle_expansion_agrees_with_native_on_an_identity_grid():
    image = cv2.imread(str(BASE / "assets/labels.png"))
    canvas = np.full((736, 736, 3), 255, np.uint8)
    canvas[184:552] = cv2.resize(image, (736, 368))
    preparation = module("text_reference_prepare", BASE / "prepare_model.py")
    block = DNNInference(parameters={**preparation.PARAMETERS, "model_path": str(MODEL)})
    try:
        result = block.execute({"image": canvas})
        assert result is not None, block.error
        decoder = TextRegions(
            parameters={"expansion": "db_unclip", "unclip_ratio": 1.5, "minimum_side": 3}
        )
        decoded = decoder.execute(
            {
                "image": canvas,
                "tensors": result.outputs["tensors"],
                "preprocessing": result.outputs["preprocessing"],
            }
        )
        assert decoded is not None, decoder.error
        net = cv2.dnn.readNetFromONNX(str(MODEL), cv2.dnn.ENGINE_CLASSIC)
        native_decoder = cv2.dnn_TextDetectionModel_DB(net)
        native_decoder.setInputSize((736, 736))
        native_decoder.setInputMean(tuple(preparation.PARAMETERS["mean"]))
        native_decoder.setInputScale(1 / 255 / np.asarray(preparation.PARAMETERS["std"]))
        native_decoder.setBinaryThreshold(0.3)
        native_decoder.setPolygonThreshold(0.5)
        native_decoder.setUnclipRatio(1.5)
        native_decoder.setMaxCandidates(2000)
        quads, _ = native_decoder.detect(canvas)
        assert len(quads) == decoded.outputs["count"] == 3
        for region in decoded.outputs["regions"]["items"]:
            assert max(overlap(region["corners"], quad) for quad in quads) > 0.95
    finally:
        block.close()


def test_regeneration_is_exact_and_differing_assets_are_preserved(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("pinned_text_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in [*BASE.glob("*.json"), *(BASE / "assets").glob("*.png")]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    target = tmp_path / "assets/labels.png"
    target.write_bytes(b"preserve image")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_wrong_weights_reject_without_network_or_output(tmp_path, monkeypatch):
    prepare = module("pinned_text_prepare_bad", BASE / "prepare_model.py")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    raw = tmp_path / "wrong.onnx"
    raw.write_bytes(b"keep")
    with pytest.raises(ValueError, match="pinned"):
        prepare.prepare(raw, directory=tmp_path / "bundle")
    assert raw.read_bytes() == b"keep" and not (tmp_path / "bundle").exists()


@native
def test_preparation_is_offline_idempotent_and_model_context_survives_save_as(
    tmp_path, monkeypatch
):
    prepare = module("pinned_text_prepare", BASE / "prepare_model.py")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    paths = [MODEL, MODEL.with_name("contract.json")]
    before = [p.stat().st_mtime_ns for p in paths]
    assert prepare.prepare() == MODEL.parent
    assert [p.stat().st_mtime_ns for p in paths] == before
    with ProjectSession.open(BASE / "labels.json") as session:
        session.save(tmp_path / "program.json")
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
        assert read_project_xml(tmp_path / "program.xml") == definition
    assert (tmp_path / ".local/ppocr_en/model.onnx").read_bytes() == MODEL.read_bytes()
    for name in ("MODEL-LICENSE.txt", "SOURCES.md"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(definition, builtin_registry(), project_dir=tmp_path),
    ):
        try:
            graph.run()
            assert graph.blocks["decode"].last_result.outputs["count"] == 3
            assert graph.blocks["decode"].last_result.passed
        finally:
            graph.close()
