"""Verify learned SuperPoint heads, source mapping, and example registrations."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.superpoint_features import (
    SuperPointONNXFeatures,
    heatmap,
    network_input,
    peak_scores,
    sampled_descriptors,
)
from opencv_dataflow.tickets import TicketClock

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/custom tools/superpoint_v6"
MODELS = ROOT / ".local/models"
CASES = ("extract", "translation", "blank", "resized_region")


def test_detector_heatmap_uses_dustbin_and_row_major_subpixel_channels():
    logits = np.full((65, 2, 3), -10, np.float32)
    logits[13, 1, 2] = logits[64, 1, 2] = 10
    result = heatmap(logits)
    assert result.shape == (16, 24)
    assert np.unravel_index(result.argmax(), result.shape) == (9, 21)
    assert result[9, 21] == pytest.approx(0.5, abs=1e-6)


def test_suppression_keeps_nonadjacent_secondary_peak_and_can_be_disabled():
    scores = np.float32([[0, 0.9, 0.8, 0.7, 0]])
    np.testing.assert_array_equal(peak_scores(scores, 1), np.float32([[0, 0.9, 0, 0.7, 0]]))
    np.testing.assert_array_equal(peak_scores(scores, 0), scores)


def test_descriptor_rows_use_center_aligned_bilinear_sampling_and_unit_norm():
    dense = np.zeros((256, 2, 2), np.float32)
    dense[:4] = np.eye(4, dtype=np.float32).reshape((4, 2, 2))
    desc = sampled_descriptors(dense, np.array([0, 7]), np.array([0, 7]))
    expected = np.array([81, 63, 63, 49], np.float32)
    np.testing.assert_allclose(desc[0, :4], [1, 0, 0, 0])
    np.testing.assert_allclose(desc[1, :4], expected / np.linalg.norm(expected), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(desc, axis=1), [1, 1])
    assert not np.any(desc[:, 4:])


def test_area_resize_preserves_normalized_pixel_values_and_exposes_content_grid():
    image = np.full((240, 320), 65535, np.uint16)
    blob, (width, height) = network_input(image, 193)
    assert (width, height) == (193, 145)
    assert blob.shape == (1, 1, 152, 200)
    np.testing.assert_allclose(blob[0, 0, :height, :width], 1, atol=1e-6)
    assert not blob[0, 0, height:, :].any()
    assert not blob[0, 0, :, width:].any()


def test_wrong_checkpoint_reports_error_without_emission(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"wrong")
    block = SuperPointONNXFeatures(parameters={"model_path": str(model), "model_sha256": "0" * 64})
    assert block.execute({"image": np.zeros((32, 32), np.uint8)}) is None
    assert block.state == "error" and "SHA-256" in block.error and not block.current_emissions


@pytest.fixture(scope="module")
def detector():
    model = MODELS / "superpoint-v6-dense.onnx"
    if not model.is_file():
        pytest.skip("Optional pinned MIT SuperPoint v6 model not prepared")
    return model


@pytest.fixture
def matcher(monkeypatch):
    if not (MODELS / "superpoint_lightglue.pth").is_file():
        pytest.skip("Optional pinned Apache-2.0 matcher not prepared")
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightglue")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)

    def fail(*args, **kwargs):
        pytest.fail("A saved project must not download model weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fail)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("case", CASES)
def test_example_validates_without_loading_model(case):
    graph = Graph.load(EXAMPLES / f"{case}.json", builtin_registry())
    try:
        report = validate_graph(graph)
        assert report["valid"], report["issues"]
    finally:
        graph.close()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_real_examples_and_match_geometry(case, profile, detector, request):
    if case == "translation":
        request.getfixturevalue("matcher")
    graph = Graph.load(EXAMPLES / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        assert not {b.id: b.error for b in graph.blocks.values() if b.error}
        source_id = "blank" if case == "blank" else "reference"
        block = graph.blocks[f"{source_id}_features"]
        result = block.last_result
        features = result.outputs["features"]
        points = np.array(features["points"])
        report = result.outputs["report"]
        assert result.passed is (case != "blank")
        assert report["detector_executed"] and report["engine"] == "new"
        assert report["retained_count"] == result.outputs["count"] == len(points)
        count_gate = {"blank": (0, 3), "resized_region": (20, 150)}
        lower, upper = count_gate.get(case, (150, 250))
        assert lower <= len(points) <= upper
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks[source_id].last_result.outputs["image"]
        )
        ticket = graph.blocks[source_id].last_result.payloads["image"]["ticket"]
        assert features["coordinate_context"]["ticket"] == ticket
        assert bool(result.drawings) is (profile == "debug")
        if case == "resized_region":
            assert report["network_size"] == [200, 152]
            assert report["content_size"] == [193, 145]
            xy = np.rint(points).astype(int)
            assert np.all(xy >= [40, 35]) and np.all(xy < [170, 195])
            raw_indices = np.asarray(features["raw_indices"])
            x, y = raw_indices % 200, raw_indices // 200
            np.testing.assert_allclose(points[:, 0], (x + 0.5) * 320 / 193 - 0.5)
            np.testing.assert_allclose(points[:, 1], (y + 0.5) * 240 / 145 - 0.5)
        if case == "translation":
            match = graph.blocks["match"].last_result
            assert match.passed and match.outputs["inliers"] >= 20
            pose = match.outputs["pose"]
            np.testing.assert_allclose(pose["origin"], [8, 6], atol=1.5)
            corners = np.array([[[29, 27], [288, 27], [288, 210], [29, 210]]], np.float64)
            mapped = cv2.perspectiveTransform(corners, np.array(pose["source_to_target"]))
            assert np.linalg.norm(mapped - corners - [8, 6], axis=2).max() <= 1.5
            query = graph.blocks["query"].last_result.outputs["image"]
            film = graph.blocks["filmstrip"].last_result.outputs["image"]
            assert bool(np.any(film != query)) is (profile == "debug")
    finally:
        graph.close()


def test_repeatability_and_model_reuse_across_network_sizes(detector):
    image = cv2.imread(str(EXAMPLES / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    context = {"ticket": TicketClock().issue("feature_test"), "spatial": frame([320, 240], "a")}
    block = SuperPointONNXFeatures(parameters={"model_path": str(detector), "maximum_features": 48})
    try:
        first = block.execute({"image": ValuePacket(image, context)})
        assert first.passed and first.outputs["count"] == 48
        block.parameters["maximum_edge"] = 193
        small = block.execute({"image": ValuePacket(image, context)})
        assert small.passed and small.outputs["report"]["native_model_reused"]
        assert small.outputs["report"]["network_size"] == [200, 152]
        block.parameters["maximum_edge"] = 640
        repeat = block.execute({"image": ValuePacket(image, context)})
        assert repeat.passed and repeat.outputs["report"]["native_model_reused"]
        assert repeat.outputs["features"] == first.outputs["features"]
    finally:
        block.close()
        assert block._model is None
