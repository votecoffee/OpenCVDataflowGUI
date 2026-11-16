"""Independent small-graph/threshold oracles and image-feature evaluation contracts."""

import copy
import itertools
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.feature_benchmark import maximum_matching, precision_recall_curve
from opencv_dataflow.feature_evaluation import EvaluateImageFeatures
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key


def feature_set(points, values=None):
    if values is None:
        values = [0, 255, 170, 85][: len(points)]
    return {
        "algorithm": "ORB",
        "descriptor_size": 32,
        "wta_k": 2,
        "image_size": [100, 100],
        "keypoints": [{"point": list(p)} for p in points],
        "descriptors": [[v] * 32 for v in values],
    }


def sample():
    query = feature_set([[10, 10], [30, 10], [50, 10], [75, 75]])
    train = feature_set([[12, 13], [32, 13], [52, 13], [92, 92]])
    return {
        "query_image": np.zeros((100, 100), np.uint8),
        "train_image": np.zeros((100, 100), np.uint8),
        "query": query,
        "train": train,
        "transform": [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
    }


def match_record(values, rows):
    return {
        "query_size": [100, 100],
        "train_size": [100, 100],
        "matches": [{"query_index": q, "train_index": t, "distance": d} for q, t, d in rows],
        **{
            role + "_points": [
                values[role]["keypoints"][q if role == "query" else t]["point"] for q, t, _ in rows
            ]
            for role in ("query", "train")
        },
    }


def test_maximum_pairing_matches_exhaustive_oracle_on_all_three_by_three_graphs():
    for bits in range(512):
        adjacency = [[t for t in range(3) if bits & (1 << (q * 3 + t))] for q in range(3)]
        expected = max(
            sum(t >= 0 for t in choices)
            for choices in itertools.product(range(-1, 3), repeat=3)
            if len({t for t in choices if t >= 0}) == sum(t >= 0 for t in choices)
            and all(t < 0 or t in adjacency[q] for q, t in enumerate(choices))
        )
        pairing = maximum_matching(adjacency, 3)
        assert len(pairing) == expected
        assert len(set(pairing.values())) == len(pairing)
    assert maximum_matching([[0, 1], [0]], 2) == {0: 1, 1: 0}


def test_analytic_repeatability_tied_threshold_curve_and_duplicate_penalty():
    values = sample()
    rows = [(0, 0, 1), (1, 2, 2), (1, 1, 2), (2, 2, 3), (2, 2, 3), (3, 3, 4)]
    values["matches"] = match_record(values, rows)
    block = EvaluateImageFeatures(parameters={"match_threshold": 3, "reprojection_tolerance": 0.1})
    result = block.execute(values)
    assert result is not None, block.error
    assert result.passed and result.outputs["repeatability"] == 0.75
    assert result.outputs["precision"] == 0.6 and result.outputs["recall"] == 1
    assert [
        (p["threshold"], p["accepted"], p["true_positives"]) for p in result.outputs["curve"]
    ] == [(1, 1, 1), (2, 3, 2), (3, 5, 3), (4, 6, 3)]
    assert [p["precision"] for p in result.outputs["curve"]] == pytest.approx([1, 2 / 3, 0.6, 0.5])
    assert [p["recall"] for p in result.outputs["curve"]] == pytest.approx([1 / 3, 2 / 3, 1, 1])
    assert sum(row["unique_true_positive"] for row in result.outputs["matches"]) == 3
    values["matches"] = match_record(values, [rows[i] for i in (4, 2, 1, 0, 5, 3)])
    assert block.execute(values).outputs["curve"] == result.outputs["curve"]


def test_valid_alternative_pairing_is_not_rejected_by_one_canonical_truth_assignment():
    values = sample()
    values["query"] = feature_set([[10, 10], [10.9, 10]], [0, 255])
    values["train"] = feature_set([[10.7, 10], [9.3, 10]], [255, 0])
    values["transform"] = np.eye(3).tolist()
    values["matches"] = match_record(values, [(0, 1, 0), (1, 0, 0)])
    block = EvaluateImageFeatures(parameters={"reprojection_tolerance": 0.8})
    result = block.execute(values)
    assert result is not None, block.error
    assert result.outputs["metrics"]["correspondence_count"] == 2
    assert (
        result.outputs["repeatability"]
        == result.outputs["precision"]
        == result.outputs["recall"]
        == 1
    )


def test_forward_and_reverse_common_support_masks_regions_and_margin():
    values = sample()
    mask = np.zeros((100, 100), np.uint8)
    mask[:, :40] = 255
    values["train_mask"] = mask
    values["query_region"] = {
        "model": "image_region_v1",
        "image_size": [100, 100],
        "shape": "circle",
        "center": [20, 10],
        "radius": 20,
    }
    values["matches"] = match_record(values, [(0, 0, 1), (1, 1, 2), (2, 2, 3)])
    result = EvaluateImageFeatures().execute(values)
    metrics = result.outputs["metrics"]
    assert metrics["eligible_query"] == metrics["eligible_train"] == 2
    assert metrics["correspondence_count"] == 2
    assert metrics["outside_common_support"] == 1
    assert metrics["operating"]["accepted"] == 2
    block = EvaluateImageFeatures(parameters={"border_margin": 45})
    empty = block.execute(values)
    assert not empty.passed and empty.outputs["repeatability"] is NO_VALUE


def test_sampled_curve_preserves_operating_point_highest_threshold_and_ties():
    rows = [
        {
            "query_index": 0,
            "train_index": 0,
            "rank": rank,
            "eligible": True,
            "geometrically_correct": True,
        }
        for rank in [1, 2, 3, 4, 5, 5]
    ]
    curve, sampled = precision_recall_curve(rows, 2, 1, 1, 1, 2)
    assert sampled and [p["threshold"] for p in curve] == [2, 5]
    assert curve[-1]["accepted"] == 6 and curve[-1]["true_positives"] == 1
    assert curve[-1]["precision"] == 1 / 6


@pytest.mark.parametrize("kind", ["empty", "zero_truth", "no_predictions", "ambiguous_ratio"])
def test_undefined_metrics_remain_explicit_and_no_support_does_not_pass(kind):
    values = sample()
    p = {"reprojection_tolerance": 0.1, "match_threshold": 0}
    if kind == "empty":
        values["query"] = feature_set([], [])
    elif kind == "zero_truth":
        values["transform"][0][2] = 20
    elif kind == "no_predictions":
        values["train"]["descriptors"] = [[1] * 32 for _ in range(4)]
    else:
        values["query"]["descriptors"] = [[0] * 32 for _ in range(4)]
        values["train"]["descriptors"] = [[0] * 32 for _ in range(4)]
        p["ranking"] = "ratio"
    block = EvaluateImageFeatures(parameters=p)
    result = block.execute(values)
    assert result is not None, block.error
    if kind in ("empty", "zero_truth"):
        assert not result.passed and result.outputs["recall"] is NO_VALUE
        assert "feature_evaluation_no_correspondences" in result.outputs["metrics"]["reasons"]
    else:
        assert result.outputs["precision"] is NO_VALUE and result.outputs["recall"] == 0
        assert result.outputs["metrics"]["operating"]["accepted"] == 0
    if kind == "ambiguous_ratio":
        assert result.outputs["metrics"]["unranked_rows"] == 4


@pytest.mark.parametrize(
    "family,settings",
    [
        ("orb", {}),
        ("orb", {"wta_k": "4"}),
        ("brisk", {}),
        ("sift", {}),
        ("akaze", {}),
        ("kaze", {}),
    ],
)
def test_native_feature_families_use_compatible_descriptor_norms(family, settings):
    registry = builtin_registry()
    rng = np.random.default_rng(55)
    image = rng.integers(0, 256, (160, 180), np.uint8)
    detect = registry.create("inspect." + family, parameters=settings)
    features = detect.execute({"image": image}).outputs["features"]
    assert features["keypoints"]
    block = EvaluateImageFeatures(parameters={"maximum_curve_points": 4})
    result = block.execute(
        {"query_image": image, "train_image": image, "query": features, "train": features}
    )
    assert result is not None, block.error
    assert result.passed and result.outputs["repeatability"] == 1
    assert result.outputs["precision"] > 0.95


@pytest.mark.parametrize(
    "parameters,field",
    [
        ({"maximum_features": 1}, "maximum_features"),
        ({"maximum_pairs": 1}, "maximum_pairs"),
        ({"maximum_geometry_edges": 1}, "maximum_geometry_edges"),
        ({"maximum_matches": 1}, "maximum_matches"),
        ({"maximum_image_pixels": 1}, "maximum_image_pixels"),
        ({"maximum_curve_points": 1}, "maximum_curve_points"),
        ({"minimum_precision": 1.1}, "minimum_precision"),
        ({"ranking": "bad"}, "ranking"),
        ({"reprojection_tolerance": float("nan")}, "reprojection_tolerance"),
    ],
)
def test_limits_and_parameters_fail_without_output_events(parameters, field):
    block = EvaluateImageFeatures(parameters=parameters)
    assert block.execute(sample()) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())
    assert (
        field in block.error
        or field.replace("_", " ").capitalize() in block.error
        or any(
            x in block.error
            for x in ["Feature count", "Feature pairs", "Geometric", "matches", "image"]
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "singular",
        "horizon",
        "size",
        "bad_points",
        "bad_matches",
        "stale_matches",
        "bad_descriptor",
        "family",
    ],
)
def test_invalid_geometry_and_feature_records_are_contained(mutation):
    values = sample()
    if mutation == "singular":
        values["transform"] = [[0] * 3] * 3
    elif mutation == "horizon":
        values["transform"] = [[1, 0, 0], [0, 1, 0], [0.1, 0, -2]]
    elif mutation == "size":
        values["query"]["image_size"] = [99, 100]
    elif mutation == "bad_points":
        values["query"]["keypoints"][0]["point"] = [100, 10]
    elif mutation == "bad_descriptor":
        values["query"]["descriptors"][0][0] = -1
    elif mutation == "family":
        values["train"]["wta_k"] = 4
    else:
        values["matches"] = match_record(values, [(0, 0, 1)])
        if mutation == "bad_matches":
            values["matches"]["matches"][0]["query_index"] = 999
        else:
            values["matches"]["query_points"][0] = [11, 10]
    block = EvaluateImageFeatures()
    assert block.execute(values) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


@pytest.mark.parametrize("mismatch", ["frame", "ticket"])
def test_feature_to_image_snapshot_consistency(mismatch):
    values, clock = sample(), TicketClock()
    ticket = clock.issue("camera", "external", "part-42")
    payload = {"spatial": frame([100, 100], "query"), "ticket": ticket}
    values["query_image"] = ValuePacket(values["query_image"], payload)
    other = copy.deepcopy(payload)
    if mismatch == "frame":
        other["spatial"]["to_root"][0][2] = 1
    else:
        other["ticket"] = clock.issue("camera", "external", "part-43")
    values["query"] = ValuePacket(values["query"], other)
    block = EvaluateImageFeatures()
    assert block.execute(values) is None and block.error_kind == "invalid"


@pytest.mark.parametrize("preview_width", [128, 131])
def test_graphics_own_grid_scaled_endpoints_source_bytes_and_lean_numerical_parity(preview_width):
    values = sample()
    ticket = TicketClock().issue("camera", "external", "query-42")
    values["query_image"] = ValuePacket(
        values["query_image"], {"ticket": ticket, "spatial": frame([100, 100], "camera")}
    )
    block = EvaluateImageFeatures(
        parameters={"preview_width": preview_width, "preview_height": 128}
    )
    originals = values["train_image"].copy()
    result = block.execute(values)
    assert result is not None, block.error
    height, width = result.outputs["overlay"].shape[:2]
    assert width == preview_width and height <= 128
    assert result.payloads["overlay"]["spatial"]["root_id"] != "camera"
    assert ticket_key(result.payloads["overlay"]["ticket"]) == ticket_key(ticket)
    assert "spatial" not in result.payloads["metrics"]
    preview = result.outputs["metrics"]["preview"]
    left_width = round(100 * preview_width / 200)
    right_width = preview_width - left_width
    assert preview["train_to_canvas"][0][2] == left_width + (right_width / 100 - 1) / 2
    first = result.drawings["matched pairs"].objects[0]
    np.testing.assert_allclose(
        first.points,
        [
            [10.5 * left_width / 100 - 0.5, 10.5 * height / 100 - 0.5],
            [left_width + 12.5 * right_width / 100 - 0.5, 13.5 * height / 100 - 0.5],
        ],
    )
    np.testing.assert_array_equal(values["train_image"], originals)
    block.set_graphics_mode("off")
    lean = block.execute(values)
    assert lean.outputs["overlay"] is lean.outputs["canvas"] is NO_VALUE
    assert not lean.graphics and not lean.drawings
    assert lean.outputs["curve"] == result.outputs["curve"]
    assert lean.outputs["correspondences"] == result.outputs["correspondences"]


def test_parallel_instances_and_native_error_recovery(monkeypatch):
    def run(translation):
        values = sample()
        values["transform"][0][2] = translation
        block = EvaluateImageFeatures(parameters={"reprojection_tolerance": 0.1})
        result = block.execute(values)
        assert result is not None, block.error
        return result.outputs["repeatability"]

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(run, [2, 20])) == [0.75, 0]
    native = cv2.BFMatcher

    def broken(*args):
        raise cv2.error("Controlled feature matcher failure")

    monkeypatch.setattr(cv2, "BFMatcher", broken)
    block = EvaluateImageFeatures()
    assert block.execute(sample()) is None and block.error_kind == "exception"
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(cv2, "BFMatcher", native)
    assert block.execute(sample()).passed


def test_xml_project_and_parameter_dropdowns(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    registry, graph = builtin_registry(), Graph()
    block = graph.add(
        registry.create(
            "inspect.evaluate_features", parameters={"ranking": "ratio", "match_threshold": 0.7}
        )
    )
    xml = tmp_path / "benchmark.inspect.evaluate_features.xml"
    save_block_xml(block, xml)
    assert read_block_xml(xml)["parameters"] == block.parameters
    graph.save(tmp_path / "program.json")
    reopened = Graph.load(tmp_path / "program.json", registry)
    assert reopened.blocks[block.id].parameters == block.parameters
    reopened.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert isinstance(window.inspector.fields["ranking"], QComboBox)
        assert window.inspector.fields["reprojection_tolerance"].toolTip()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
