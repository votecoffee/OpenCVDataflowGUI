"""Qualify native balance, palette guards, and held-out synthetic casts."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.learned_white_balance import LearnedWhiteBalance
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/learned_white_balance"


def chroma_spread(image):
    patch = image[16:97, 14:98].astype(float)
    return float(np.mean(np.ptp(patch, axis=2) / patch.mean(axis=2)))


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_charts_document_both_correction_and_regression(profile, workers):
    graph = Graph.load(EXAMPLE / "learned_white_balance.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        block = graph.blocks["learned"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Computational Photography (xphoto)")
        for filename, populated, before, after in (
            ("warm.png", 3342, 0.59217, 0.11539),
            ("cool.png", 3510, 0.49049, 0.19960),
            ("green_deficit.png", 4642, 0.22904, 0.60068),
        ):
            graph.run()
            result = block.last_result
            assert result and result.passed, block.error
            input_image = graph.blocks["source"].last_result.outputs["image"]
            assert chroma_spread(input_image) == pytest.approx(before, abs=1e-5)
            assert chroma_spread(result.outputs["image"]) == pytest.approx(after, abs=0.005)
            assert result.outputs["report"]["populated_bins"] == populated
            assert result.outputs["report"]["model"]["reference_header_sha256"].startswith("c18df1a6")
            assert result.payloads["image"]["ticket"]["value"] == filename
            assert result.payloads["image"]["spatial"]["root_id"] == "source:image"
            assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


@pytest.fixture
def bgr():
    return np.random.default_rng(67).integers(11, 250, (54, 68, 3), dtype=np.uint8)


@pytest.mark.parametrize(("depth", "range_max"), [(np.uint8, 255), (np.uint16, 4095)])
def test_native_output_features_and_alpha_are_preserved(bgr, depth, range_max):
    bgr = bgr.astype(depth) * (1 if depth == np.uint8 else 16)
    bgr[0, 0] = range_max
    native = cv2.xphoto.createLearningBasedWB()
    native.setRangeMaxVal(range_max)
    native.setHistBinNum(56)
    native.setSaturationThreshold(0.9)
    expected = native.balanceWhite(bgr)
    direct_features = native.extractSimpleFeatures(bgr).reshape(4, 2)
    alpha = np.broadcast_to(np.arange(bgr.shape[1], dtype=depth), bgr.shape[:2])
    rgba = ImageArray(np.dstack((bgr[..., ::-1], alpha)), "rgba")
    before = rgba.copy()
    block = LearnedWhiteBalance(parameters={
        "range_max": range_max, "histogram_bins": 56, "saturation_threshold": 0.9,
    })
    block.exposed_outputs.append("features")
    frame_id = frame([68, 54], "camera")
    mask = np.zeros(bgr.shape[:2], np.uint8)
    mask[1:30, 5:62] = 255
    result = block.execute({
        "image": ValuePacket(rgba, {"spatial": frame_id}),
        "mask": ValuePacket(mask, {"spatial": frame_id}),
    })
    assert result and result.passed, block.error
    out = result.outputs["image"]
    assert out.dtype == depth and out.color_format == "rgba"
    np.testing.assert_array_equal(out[..., :3][mask > 0], expected[..., ::-1][mask > 0])
    np.testing.assert_array_equal(out[mask == 0], before[mask == 0])
    np.testing.assert_array_equal(out[..., 3], alpha)
    np.testing.assert_array_equal(rgba, before)
    np.testing.assert_array_equal(list(result.outputs["features"].values()), direct_features)
    assert result.payloads["image"]["spatial"] == frame_id
    cutoff = int(np.float32(0.9) * np.float32(range_max))
    assert result.outputs["report"]["eligible_pixels"] == np.count_nonzero(bgr.max(axis=2) < cutoff)


@pytest.mark.parametrize("sample", [0, 72, 255])
def test_low_diversity_fails_before_native_palette_evaluation(monkeypatch, sample):
    def unexpected():
        pytest.fail("Unsafe palette reached the native estimator")

    monkeypatch.setattr(cv2.xphoto, "createLearningBasedWB", unexpected)
    image = np.full((42, 48, 3), sample, np.uint8)
    block = LearnedWhiteBalance()
    result = block.execute({"image": image})
    assert result and not result.passed, block.error
    assert result.outputs["report"]["populated_bins"] < 300
    assert result.outputs["image"] is result.outputs["features"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "learned_white_balance_palette_too_small"


def write_constant_model(path, *, selector=0, leaves=16):
    model = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        model.write("num_trees", 1)
        model.write("num_tree_nodes", 2)
        model.write("feature_idx", np.full((8, 1), selector, np.uint8))
        model.write("thresh_vals", np.full((8, 1), 0.5, np.float32))
        leaf_values = np.tile(np.float32([0.4, 0.4, 0.35, 0.35]), 4)[:leaves]
        model.write("leaf_vals", leaf_values.reshape(-1, 1))
    finally:
        model.release()


def test_external_model_is_loaded_and_hash_is_verified(tmp_path, bgr):
    path = tmp_path / "synthetic.yml"
    write_constant_model(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    native = cv2.xphoto.createLearningBasedWB(str(path))
    direct = native.balanceWhite(bgr)
    assert np.any(direct != cv2.xphoto.createLearningBasedWB().balanceWhite(bgr))
    block = LearnedWhiteBalance(parameters={"model_path": path.name, "model_sha256": digest})
    block.project_dir = tmp_path
    result = block.execute({"image": bgr})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], direct)
    assert result.outputs["report"]["model"]["sha256"] == digest
    write_constant_model(path, selector=2)
    assert block.execute({"image": bgr}) is None
    assert "SHA-256 differs" in block.error


@pytest.mark.parametrize(("selector", "leaves", "error"), [
    (2, 16, "indices or chromaticities"), (0, 15, "model matrix leaf_vals"),
])
def test_invalid_external_tree_structure_is_rejected(tmp_path, bgr, selector, leaves, error):
    path = tmp_path / "invalid.yml"
    write_constant_model(path, selector=selector, leaves=leaves)
    block = LearnedWhiteBalance(parameters={"model_path": str(path)})
    assert block.execute({"image": bgr}) is None
    assert error in block.error


def test_invalid_input_depth_and_sensor_range(bgr):
    block = LearnedWhiteBalance(parameters={"range_max": 200})
    assert block.execute({"image": bgr}) is None and "Sensor range" in block.error
    assert block.execute({"image": bgr.astype(np.float32)}) is None
    assert "Unsupported image depth" in block.error
