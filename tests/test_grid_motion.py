"""Verify GMS filtering, row alignment, and the downstream pose fit."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/grid_motion"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def control_pairs():
    rng = np.random.default_rng(909)
    source = rng.integers(30, 290, size=(500, 2))
    target = source + [15, -11]
    target[::4] = rng.integers(20, 300, size=target[::4].shape)
    return {
        "query_size": [320, 320],
        "train_size": [320, 320],
        "query_points": source.tolist(),
        "train_points": target.tolist(),
        "matches": [
            {"query_index": 1000 + i, "train_index": 3000 + i, "distance": i % 128}
            for i in range(len(source))
        ],
        "query_context": {"spatial": frame([320, 320], "left")},
        "train_context": {"spatial": frame([320, 320], "right")},
        "scores": [i / 500 for i in range(len(source))],
    }


def test_matches_native_and_keeps_aligned_metadata(registry):
    data = control_pairs()
    ticket = TicketClock("gms").issue("input", "fixed", "comparison")
    block = registry.create(
        "inspect.filter_matches_gms",
        parameters={
            "with_rotation": True,
            "with_scale": True,
            "threshold_factor": 4.5,
        },
    )
    result = block.execute({"matches": ValuePacket(data, {"ticket": ticket})})
    assert result and result.passed, block.error
    keys = [
        [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in data[f"{name}_points"]]
        for name in ("query", "train")
    ]
    native = cv2.xfeatures2d.matchGMS(
        data["query_size"],
        data["train_size"],
        *keys,
        [cv2.DMatch(i, i, 0.0) for i in range(500)],
        True,
        True,
        4.5,
    )
    indexes = {match.queryIdx for match in native}
    assert 0 < len(indexes) < 500
    mask = result.outputs["kept_mask"]
    assert mask == [index in indexes for index in range(500)]
    assert result.outputs["count"] == len(indexes)
    assert result.outputs["rejected_count"] == 500 - len(indexes)
    for port, accept in (("matches", True), ("rejected", False)):
        selected = result.outputs[port]
        for key in ("query_points", "train_points", "matches", "scores"):
            assert selected[key] == [
                row for row, keep in zip(data[key], mask, strict=True) if keep == accept
            ]
        assert selected["query_context"] == data["query_context"]
        assert selected["train_context"] == data["train_context"]
        assert result.payloads[port]["ticket"] == ticket


def test_empty_duplicate_and_misaligned_matches(registry):
    block = registry.create("inspect.filter_matches_gms", parameters={"min_matches": 1})
    empty = {"query_size": [20, 30], "train_size": [30, 20], "query_points": [], "train_points": []}
    result = block.execute({"matches": empty})
    assert result and not result.passed and result.outputs["kept_mask"] == []
    assert result.fail_reasons[0]["code"] == "too_few_grid_motion_matches"
    duplicates = control_pairs()
    duplicates["matches"][0]["query_index"] = duplicates["matches"][1]["query_index"]
    assert block.execute({"matches": duplicates}) is None
    assert "one candidate match per query" in block.error
    incomplete = {**control_pairs(), "scores": [0.1]}
    assert block.execute({"matches": incomplete}) is None
    assert "one entry per match" in block.error
    out_of_frame = control_pairs()
    out_of_frame["train_points"][0] = [320, 0]
    assert block.execute({"matches": out_of_frame}) is None
    assert "inside their image" in block.error


def test_example_rejects_outliers_before_pose_fit(registry):
    graph = Graph.load(EXAMPLE / "grid_motion.json", registry)
    try:
        graph.run()
        block = graph.blocks["filter"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")
        filtered = block.last_result
        assert filtered and filtered.passed, block.error
        raw = graph.blocks["raw"].last_result.outputs["matches"]
        kept = filtered.outputs["matches"]
        distinct = {row["query_index"] for row in raw["matches"]}
        assert len(distinct) == len(raw["matches"]) == 4200
        transform = cv2.getRotationMatrix2D((300, 220), 7, 0.92)
        transform[:, 2] += (19, -13)

        def correct_fraction(data):
            query = np.asarray(data["query_points"])
            train = np.asarray(data["train_points"])
            known = query @ transform[:, :2].T + transform[:, 2]
            return np.mean(np.linalg.norm(known - train, axis=1) < 3)

        assert correct_fraction(raw) < 0.1
        assert correct_fraction(kept) > 0.75
        assert filtered.outputs["count"] > 200
        markers = np.array([[64, 64, 1], [535, 64, 1], [64, 375, 1], [535, 375, 1], [300, 220, 1]])

        def fit_error(key):
            result = graph.blocks[key].last_result
            assert result and result.passed, graph.blocks[key].error
            matrix = np.asarray(result.outputs["transform"]["source_to_target"])[:2]
            delta = markers @ (matrix - transform).T
            return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))

        assert fit_error("filtered_fit") < 1.0
        assert fit_error("raw_fit") > 20
        assert graph.blocks["filtered_review"].last_result.outputs["info"]["pairs"] == len(
            kept["matches"]
        )
    finally:
        graph.close()
