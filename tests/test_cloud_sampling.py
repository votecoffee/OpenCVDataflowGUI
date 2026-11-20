"""Check deterministic sampling, distance diagnostics and source cloud attributes."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_sampling import SampleCloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

PROJECT = Path(__file__).resolve().parents[1] / "examples/custom tools/cloud_sampling"


def _covering_radius(points, samples):
    sq = ((points[:, None, :].astype(np.float64) - samples[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(sq.min(axis=1).max()))


def _independent_greedy(points, seed, count):
    generator = np.random.Generator(np.random.PCG64(seed))
    result = [int(generator.integers(len(points)))]
    values = points.astype(np.float64)
    while len(result) < count:
        sq = ((values[:, None, :] - values[result][None, :, :]) ** 2).sum(axis=2)
        distance = sq.min(axis=1)
        distance[result] = -1
        result.append(int(np.argmax(distance)))
    return result


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_equal_budget_project_matches_oracle_and_improves_coverage(profile):
    graph = Graph.load(PROJECT / "cloud_sampling.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        source = graph.blocks["source"].last_result.outputs["cloud"]
        random, far = (graph.blocks[key].last_result for key in ("random", "farthest"))
        assert random and random.passed and far and far.passed
        for result in (random, far):
            out = result.outputs
            indices = out["indices"]
            assert len(indices) == out["count"] == 20 and len(set(indices)) == 20
            np.testing.assert_array_equal(out["cloud"].points, source.points[indices])
            np.testing.assert_array_equal(out["cloud"].colors, source.colors[indices])
            np.testing.assert_array_equal(out["cloud"].normals, source.normals[indices])
            assert out["cloud"].frame == "sampling_fixture" and out["cloud"].units == "mm"
        assert far.outputs["indices"] == _independent_greedy(source.points, seed=71, count=20)
        random_radius = _covering_radius(source.points, random.outputs["cloud"].points)
        far_radius = _covering_radius(source.points, far.outputs["cloud"].points)
        assert far_radius < 0.7 * random_radius
        assert far.outputs["report"]["covering_radius"] == pytest.approx(far_radius)
        assert far.outputs["report"]["minimum_selected_spacing"] == pytest.approx(160.0)
    finally:
        graph.close()


def test_random_fraction_seed_and_attributes_follow_exact_source_rows():
    x = np.arange(20)
    pixels = np.column_stack((x, np.zeros_like(x)))
    cloud = PointCloud(
        np.column_stack((x, x, x)),
        "mm",
        "fixture",
        colors=np.column_stack((x, x, x)),
        normals=np.tile([0, 1, 0], (20, 1)),
        pixels=pixels,
        source_size=(20, 1),
    )
    ticket = TicketClock().issue("frame")
    packet = ValuePacket(cloud, {"spatial": frame([20, 1], "camera"), "ticket": ticket})
    block = SampleCloud(
        parameters={"seed": 17, "budget": "fraction", "sample_fraction": 0.21, "max_points": 4}
    )
    result = block.execute({"cloud": packet})
    assert result and result.passed, block.error
    expected = np.random.Generator(np.random.PCG64(17)).choice(20, 4, replace=False).tolist()
    assert result.outputs["indices"] == expected
    second = block.execute({"cloud": packet})
    assert second.outputs["indices"] == expected
    block.parameters["max_points"] = 8
    assert block.execute({"cloud": packet}).outputs["count"] == 5
    chosen = result.outputs["cloud"]
    np.testing.assert_array_equal(chosen.pixels, pixels[expected])
    np.testing.assert_array_equal(chosen.colors, cloud.colors[expected])
    np.testing.assert_array_equal(chosen.normals, cloud.normals[expected])
    assert result.payloads["cloud"]["ticket"] == ticket
    assert result.payloads["cloud"]["spatial"] == frame([20, 1], "camera")
    assert chosen.source_size == cloud.source_size


def test_inclusive_spacing_keeps_boundary_rows_and_stops_at_duplicate():
    source = PointCloud([[0, 0, 0], [2, 0, 0], [4, 0, 0], [4, 0, 0]], "mm", "fixture")
    sampler = SampleCloud(parameters={"method": "farthest", "max_points": 4, "minimum_spacing": 2})
    result = sampler.execute({"cloud": source})
    assert result and result.passed, sampler.error
    report = result.outputs["report"]
    assert report["selected_points"] == 3
    assert report["stop_reason"] == "minimum_spacing"
    assert report["minimum_selected_spacing"] == 2 and report["covering_radius"] == 0
    assert report["distance_evaluations"] == 12
    sampler.parameters["minimum_spacing"] = 0
    result = sampler.execute({"cloud": source})
    assert result and result.passed and sorted(result.outputs["indices"]) == [0, 1, 2, 3]
    assert result.outputs["report"]["minimum_selected_spacing"] == 0


def test_work_budget_and_cancel_keep_partial_rows_and_fail_the_result():
    source = PointCloud([[0, 0, 0], [2, 0, 0], [4, 0, 0]], "m", "fixture")
    block = SampleCloud(parameters={"method": "farthest", "max_distance_evaluations": 6})
    limited = block.execute({"cloud": source})
    assert limited is not None and not limited.passed
    assert limited.outputs["report"]["distance_evaluations"] == 6 and limited.outputs["count"] == 2
    assert limited.fail_reasons[0]["code"] == "cloud_sampling_work_limit"
    calls = iter([False, False, True])
    cancelled = block.execute({"cloud": source}, cancelled=lambda: next(calls, True))
    assert cancelled is not None and not cancelled.passed
    assert cancelled.outputs["count"] == 1
    assert cancelled.fail_reasons[0]["code"] == "cloud_sampling_cancelled"


def test_empty_cloud_and_tiny_fraction_are_explicit():
    block = SampleCloud(parameters={"budget": "fraction", "sample_fraction": 0.000001})
    result = block.execute({"cloud": PointCloud(np.empty((0, 3)))})
    assert result is not None and not result.passed and result.outputs["indices"] == []
    assert result.fail_reasons[0]["code"] == "cloud_sampling_empty_cloud"
    one = block.execute({"cloud": PointCloud([[0, 0, 1]], "unitless", "fixture")})
    assert one and one.passed and one.outputs["indices"] == [0]
