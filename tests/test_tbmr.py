"""Native TBMR parity, mapped internal levels and detector repeatability."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "inspect.tbmr"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/tbmr_repeatability"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture(scope="module")
def target():
    return cv2.imread(str(EXAMPLE / "assets/regions.png"), cv2.IMREAD_GRAYSCALE)


def test_single_level_matches_native_centers_and_axis(registry, target):
    expected = cv2.xfeatures2d.TBMR_create(32, 0.025, 1.25, 1).detect(target)
    block = registry.create(TYPE, parameters={"min_area": 32, "max_area_relative": 0.025})
    result = block.execute({"image": target})
    assert result and result.passed, block.error
    record = result.outputs["keypoints"]
    assert block_entry(block).path[0] == "OpenCV"
    assert len(record["keypoints"]) == len(expected) == 24
    assert record["levels"][0]["level_to_image"] == np.eye(3).tolist()
    for row, key in zip(record["keypoints"], expected, strict=True):
        assert row["point"] == list(key.pt)
        assert row["size"] == pytest.approx(2 * key.size)
        assert row["response"] == key.response
        assert row["angle"] == -1
        assert row["major_axis_angle_degrees"] == pytest.approx(math.degrees(key.angle) % 180)


def test_odd_raster_pyramid_maps_centers_and_respects_source_selection(registry, target):
    gray = target[:175, :239]
    block = registry.create(
        TYPE, parameters={"min_area": 18, "max_area_relative": 0.025, "n_scales": 3},
    )
    context = {
        "spatial": frame([239, 175], "test"),
        "ticket": TicketClock("test").issue("camera", "fixed", "plate"),
    }
    result = block.execute({"image": ValuePacket(gray, context)})
    assert result and result.passed, block.error
    record = result.outputs["keypoints"]
    assert set(row["octave"] for row in record["keypoints"]) == {0, 1, 2}
    by_octave = {info["octave"]: info for info in record["levels"]}
    assert by_octave[0]["detector_size"] == [240, 176]
    info = by_octave[1]
    w, h = info["image_size"]
    ww, hh = info["detector_size"]
    reduced = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(reduced, 0, hh - h, 0, ww - w, cv2.BORDER_REPLICATE)
    raw = cv2.xfeatures2d.TBMR_create(18, 0.025, 1.25, 1).detect(padded)
    mapped = [row for row in record["keypoints"] if row["octave"] == 1]
    assert len(mapped) == len(raw)
    for row, key in zip(mapped, raw, strict=True):
        x, y = key.pt
        sx, sy = 239 / w, 175 / h
        assert row["point"] == pytest.approx(((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5))
        assert row["size"] == pytest.approx(
            2 * key.size * math.hypot(sx * math.cos(key.angle), sy * math.sin(key.angle))
        )
    mask = np.zeros(gray.shape, np.uint8)
    mask[20:105, 12:153] = 255
    area = {
        "model": "image_region_v1", "shape": "rectangle",
        "image_size": [239, 175], "rectangle": [80, 0, 90, 102],
    }
    selected = block.execute(
        {"image": ValuePacket(gray, context), "mask": ValuePacket(mask, context), "region": area}
    )
    assert selected and selected.passed, block.error
    assert selected.outputs["keypoints"]["keypoints"] == [
        row for row in record["keypoints"]
        if 80 <= round(row["point"][0]) < 153 and 20 <= round(row["point"][1]) < 102
    ]
    assert selected.payloads["keypoints"]["spatial"] == context["spatial"]
    assert selected.payloads["points"]["ticket"] == context["ticket"]


@pytest.mark.parametrize(
    ("shape", "params", "message"),
    [
        ((8, 8), {"min_area": 60}, "Minimum area exceeds"),
        ((800, 800), {"max_area_relative": 0.1}, "overflow native region moments"),
        ((32, 32), {"n_scales": 0}, "Number of scales"),
    ],
)
def test_bad_bounds_rejected_before_native_factory(registry, monkeypatch, shape, params, message):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid settings reached the native component tree")

    monkeypatch.setattr(cv2.xfeatures2d, "TBMR_create", unexpected)
    block = registry.create(TYPE, parameters=params)
    assert block.execute({"image": np.zeros(shape, np.uint8)}) is None
    assert message in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_pair_compares_geometry_and_scale(registry, profile):
    graph = Graph.load(EXAMPLE / "tbmr_repeatability.json", registry)
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        for name, count, rate, error, ratio in (
            ("tbmr", 24, 1.0, 0.399, 1.516), ("mser", 8, 8 / 13, 0.243, 1.508),
        ):
            result = graph.blocks[f"{name}_evaluate"].last_result
            report = result.outputs["report"]
            assert report["repeatability"] == pytest.approx(rate)
            assert report["correspondence_count"] == count
            assert report["expected_similarity_scale"] == 1.5
            assert report["p95_center_error_px"] == pytest.approx(error, abs=0.01)
            assert report["median_size_ratio"] == pytest.approx(ratio, abs=0.001)
            assert graph.blocks[f"{name}_review"].last_result.outputs["info"]["pairs"] == count
    finally:
        graph.close()


def test_repeatability_distinguishes_no_support_and_nonuniform_scale(registry):
    block = registry.create("inspect.evaluate_keypoints", parameters={
        "reference_transform": [[1.5, 0, 0], [0, 2, 0], [0, 0, 1]],
    })
    query = np.zeros((20, 20), np.uint8)
    train = np.zeros((40, 30), np.uint8)
    inputs = {
        "query_image": query, "train_image": train,
        "query": {"image_size": [20, 20], "keypoints": [{"point": [6, 6], "size": 3}]},
        "train": {"image_size": [30, 40], "keypoints": [{"point": [9, 12], "size": 6}]},
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["correspondence_count"] == 1 and report["median_center_error_px"] == 0
    assert report["median_size_ratio"] == 2
    assert report["expected_similarity_scale"] is None and report["p95_relative_size_error"] is None
    result = block.execute({**inputs, "query_mask": np.zeros_like(query)})
    assert result and not result.passed and result.outputs["repeatability"] is NO_VALUE
    assert block.execute({**inputs, "query": {**inputs["query"], "truncated": True}}) is None
    assert "needs complete observation records" in block.error
