"""Known image transforms, independent matching certificates and saved-project outcomes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/feature_evaluation"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


def mapped(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def in_image(points):
    return np.all((points >= 0) & (points < [480, 320]), axis=1)


def maximum_certificate(adjacency, pairing):
    assert len(set(pairing.values())) == len(pairing)
    assert all(adjacency[q, t] for q, t in pairing.items())
    owners = {t: q for q, t in pairing.items()}
    seen = set(range(len(adjacency))) - pairing.keys()
    pending = list(seen)
    while pending:
        q = pending.pop()
        for t in np.flatnonzero(adjacency[q]):
            assert t in owners, "An augmenting path proves this matching is not maximum"
            owner = owners[t]
            if owner not in seen:
                seen.add(owner)
                pending.append(owner)


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_benchmarks_use_known_transform_and_exact_center_metric(case, profile):
    graph = Graph.load(HERE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    expected = TRUTH["cases"][case]
    try:
        graph.run()
        block = graph.blocks["evaluate"]
        assert block.state == expected["expected_state"], block.error
        result = block.last_result
        if expected["expected_state"] == "error":
            assert block.error_kind == "invalid" and result is None
            assert not any(block.current_emissions.values())
            return
        assert result is not None
        metrics = result.outputs["metrics"]
        assert metrics["transform_independence"] == "caller_supplied_not_verified"
        points = {
            role: np.asarray(
                [
                    k["point"]
                    for k in graph.blocks[role].last_result.outputs["features"]["keypoints"]
                ],
                np.float64,
            ).reshape(-1, 2)
            for role in ("query", "train")
        }
        matrix = np.asarray(expected["evaluation_transform"])
        projection = mapped(points["query"], matrix)
        backward = mapped(points["train"], np.linalg.inv(matrix))
        qkeep = in_image(points["query"]) & in_image(projection)
        tkeep = in_image(points["train"]) & in_image(backward)
        if case == "selected_support":
            qkeep &= np.rint(projection[:, 0]) < 250
            tkeep &= np.rint(points["train"][:, 0]) < 250
        assert metrics["eligible_query"] == sum(qkeep)
        assert metrics["eligible_train"] == sum(tkeep)
        errors = np.linalg.norm(projection[:, None] - points["train"][None], axis=2)
        adjacency = (errors <= 3) & qkeep[:, None] & tkeep[None]
        pairing = {p["query_index"]: p["train_index"] for p in result.outputs["correspondences"]}
        maximum_certificate(adjacency, pairing)
        assert metrics["correspondence_count"] == len(pairing)
        denominator = min(sum(qkeep), sum(tkeep))
        if denominator:
            assert result.outputs["repeatability"] == len(pairing) / denominator
        else:
            assert result.outputs["repeatability"] is NO_VALUE
        accepted = [r for r in result.outputs["matches"] if r["accepted"]]
        predicted = np.zeros_like(adjacency)
        for row in result.outputs["matches"]:
            q, t = row["query_index"], row["train_index"]
            assert row["error_pixels"] == pytest.approx(errors[q, t], abs=1e-9)
            assert row["geometrically_correct"] == bool(adjacency[q, t])
            if row["accepted"] and row["geometrically_correct"]:
                predicted[q, t] = True
        positives = {
            r["query_index"]: r["train_index"] for r in accepted if r["unique_true_positive"]
        }
        maximum_certificate(predicted, positives)
        assert metrics["operating"]["accepted"] == len(accepted)
        assert metrics["operating"]["true_positives"] == len(positives)
        assert result.outputs["recall"] == (len(positives) / len(pairing) if pairing else NO_VALUE)
        if case != "empty_features":
            assert result.payloads["metrics"]["ticket"]["value"] == expected["query"]
        if profile == "lean":
            assert result.outputs["canvas"] is result.outputs["overlay"] is NO_VALUE
            assert not result.graphics and not result.drawings
        else:
            assert result.outputs["overlay"].shape[:2] == (320, 960)
            assert result.graphics["precision recall"].shape == (360, 480, 3)
            assert (
                result.payloads["overlay"]["spatial"]["root_id"]
                != metrics["query_context"]["spatial"]["root_id"]
            )
    finally:
        graph.close()


def test_external_threshold_exposes_precision_loss_without_modifying_matcher():
    graph = Graph.load(HERE / "external_matches.json", builtin_registry())
    try:
        graph.run()
        result = graph.blocks["evaluate"].last_result
        assert result.passed and result.outputs["metrics"]["ranking"] == "external_distance"
        graph.blocks["evaluate"].parameters["match_threshold"] = 64
        graph.run()
        broad = graph.blocks["evaluate"].last_result
        assert not broad.passed
        assert broad.outputs["precision"] < result.outputs["precision"]
        assert broad.outputs["recall"] >= result.outputs["recall"]
    finally:
        graph.close()


def test_regeneration_matches_all_original_assets_and_saved_graphs(tmp_path):
    spec = importlib.util.spec_from_file_location("feature_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        # Git checkout settings can change JSON line endings on Windows.
        regenerated = (tmp_path / source.name).read_bytes().replace(b"\r\n", b"\n")
        checked_in = source.read_bytes().replace(b"\r\n", b"\n")
        assert regenerated == checked_in, source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
