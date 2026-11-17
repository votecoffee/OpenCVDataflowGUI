"""Native ONNX image-dependent controls and a pinned optional pretrained photo example."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.tickets import ticket_key

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/opencv tools/dnn_detection"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
MODEL = ROOT / ".local/models/yolox_s.onnx"


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_heads_source_boxes_tickets_and_rendered_results(case, profile):
    options = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["decode"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed is options.get("passed", True)
        assert result.outputs["count"] == options["count"]
        assert result.outputs["report"]["context_checks"] == {
            "ticket_checked": True,
            "frame_checked": True,
        }
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        context = result.payloads["collection"]
        assert context["spatial"]["image_size"] == [160, 80]
        assert ticket_key(context["ticket"]) == ticket_key(
            graph.blocks["source"].last_result.outputs["ticket"]
        )
        for item in result.outputs["collection"]["items"]:
            network = np.array(TRUTH["network_xyxy"][item["id"]], float)
            if case == "letterbox":
                expected = (network - [0, 16, 0, 16]) / 0.4
            else:
                expected = network / [0.4, 0.8, 0.4, 0.8]
            expected = expected.clip(0, [160, 80, 160, 80])
            wanted = [*expected[:2], *(expected[2:] - expected[:2])]
            np.testing.assert_allclose(item["geometry"]["bbox"], wanted)
        if profile == "debug" and options["count"]:
            assert result.drawings
            overlay = graph.blocks["overlay"].last_result.outputs["image"]
            assert np.any(overlay != result.outputs["image"])
        if profile == "lean":
            assert not result.drawings
        json.dumps(result.outputs["report"], allow_nan=False)
    finally:
        graph.close()


@pytest.mark.skipif(not MODEL.is_file(), reason="Prepare the pinned optional YOLOX-S model first")
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_pretrained_coffee_matches_independent_yolox_grid_and_native_batched_nms(profile):
    assert (
        hashlib.sha256(MODEL.read_bytes()).hexdigest()
        == "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
    )
    graph = Graph.load(BASE / "yolox_coffee.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["decode"]
        assert block.last_result is not None, block.error
        assert block.last_result.passed
        (cup,) = block.last_result.outputs["collection"]["items"]
        assert cup["class_id"] == 41 and cup["label"] == "cup" and cup["score"] > 0.7
        values = selected_tensor(graph.blocks["infer"].last_result.outputs["tensors"], "output")[
            0
        ].copy()
        grid, strides = [], []
        for stride in (8, 16, 32):
            for y in range(640 // stride):
                for x in range(640 // stride):
                    grid.append([x, y])
                    strides.append(stride)
        values[:, :2] = (values[:, :2] + np.asarray(grid)) * np.asarray(strides)[:, None]
        values[:, 2:4] = np.exp(values[:, 2:4]) * np.asarray(strides)[:, None]
        values[:, :2] -= values[:, 2:4] / 2
        scores = values[:, 4:5] * values[:, 5:]
        classes, confidences = scores.argmax(axis=1), scores.max(axis=1)
        keep = cv2.dnn.NMSBoxesBatched(
            values[:, :4].tolist(), confidences.tolist(), classes.tolist(), 0.35, 0.45
        )
        wanted = next(
            values[int(i), :4].copy() for i in np.asarray(keep).ravel() if classes[int(i)] == 41
        )
        # Independent fixed-photo reference: 600x400 -> 640x427, 106px top padding.
        wanted[0] *= 600 / 640
        wanted[1] = (wanted[1] - 106) * 400 / 427
        wanted[2] *= 600 / 640
        wanted[3] *= 400 / 427
        np.testing.assert_allclose(cup["geometry"]["bbox"], wanted, atol=1e-4)
    finally:
        graph.close()


def test_decoder_rejects_a_later_image_ticket_in_the_native_inference_chain():
    graph = Graph.load(BASE / "stretch.json", builtin_registry())
    try:
        graph.run()
        source = graph.blocks["source"].execute({})
        inference = graph.blocks["infer"].last_result
        image = ValuePacket(source.outputs["image"], source.payloads["image"])
        inputs = {
            "image": image,
            **{
                name: ValuePacket(inference.outputs[name], inference.payloads[name])
                for name in ("tensors", "preprocessing")
            },
        }
        assert graph.blocks["decode"].execute(inputs) is None
        assert "different acquisition tickets" in graph.blocks["decode"].error
    finally:
        graph.close()


def test_model_path_rebases_on_save_as_and_reopens(tmp_path):
    graph = Graph.load(BASE / "stretch.json", builtin_registry())
    path = tmp_path / "saved.json"
    graph.save(path)
    graph.close()
    reopened = Graph.load(path, builtin_registry())
    try:
        reopened.run()
        assert reopened.blocks["decode"].last_result.passed
        assert reopened.blocks["decode"].last_result.outputs["count"] == 3
    finally:
        reopened.close()


def test_asset_model_hashes_and_full_regeneration_keep_original_collection(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    spec = importlib.util.spec_from_file_location("detection_example", BASE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        expected = json.loads(path.read_bytes())
        if "blocks" in expected:
            # Old projects acquire newly declared optional defaults when reopened.
            restored = Graph.load(path, builtin_registry())
            expected = restored.to_dict()
            restored.close()
        assert json.loads((tmp_path / path.name).read_bytes()) == expected
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
