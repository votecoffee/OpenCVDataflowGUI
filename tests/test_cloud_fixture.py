"""Qualify seeded scan fixtures, exact row provenance and registration controls."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/perturbed_cloud_fixture"


def fixture_block(**settings):
    return builtin_registry().create("3d.cloud_perturb", parameters=settings)


@pytest.fixture
def source():
    points = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 0]], dtype=np.float32)
    return PointCloud(
        points,
        "mm",
        "bench",
        colors=[[126, 1, 4], [27, 5, 3], [61, 14, 12], [76, 36, 14]],
        normals=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pixels=[[0, 0], [1, 0], [0, 1], [1, 1]],
        source_size=(2, 2),
        metadata={"capture_id": "source"},
        attributes={"source_index": [21, 22, 29, 34], "confidence": [0.8, np.nan, 0.95, 0.93]},
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_fixture_drives_sampling_pose_recovery_and_tolerance(profile):
    graph = Graph.load(EXAMPLE / "perturbed_cloud_fixture.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        assert {
            key: graph.blocks[key].state for key in ("fixture", "fit", "sample", "wide", "tight")
        } == {
            "fixture": "pass",
            "fit": "pass",
            "sample": "pass",
            "wide": "pass",
            "tight": "fail",
        }
        outputs = graph.blocks["fixture"].last_result.outputs
        clean, perturbed = outputs["clean"], outputs["cloud"]
        report = outputs["report"]
        assert (report["input_points"], report["retained_points"], report["dropped_points"]) == (
            192,
            154,
            38,
        )
        assert 0.265 < report["displacement_rms"] < 0.28
        reference = graph.blocks["source"].last_result.outputs["cloud"]
        pairs = np.asarray(outputs["pairs"])
        np.testing.assert_array_equal(pairs[:, 1], np.arange(154))
        np.testing.assert_array_equal(
            perturbed.attributes["source_index"], reference.attributes["source_index"][pairs[:, 0]]
        )
        delta = perturbed.points.astype(float) - clean.points
        assert report["displacement_rms"] == pytest.approx(np.sqrt(np.mean((delta**2).sum(axis=1))))
        recovered = graph.blocks["fit"].last_result.outputs
        assert recovered["report"]["inlier_count"] == 154
        truth = np.array(outputs["pose"]["source_to_target"])
        actual = np.array(recovered["pose"]["source_to_target"])
        angle = (np.trace(truth[:3, :3].T @ actual[:3, :3]) - 1) / 2
        assert np.degrees(np.arccos(np.clip(angle, -1, 1))) < 0.04
        assert np.linalg.norm(truth[:3, 3] - actual[:3, 3]) < 0.23
        sample = graph.blocks["sample"].last_result.outputs["cloud"]
        assert len(sample.points) == 24
        assert set(sample.attributes["source_index"]) <= set(perturbed.attributes["source_index"])
        assert len(graph.blocks["tight"].last_result.outputs["measured_defects"].points) == 149
        assert bool(graph.blocks["fixture"].last_result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_fixed_pose_matches_analytic_rotation_and_preserves_row_provenance(source):
    block = fixture_block(
        rotation_vector_deg=[0, 0, 90], translation=[10, -5, 3], target_frame="scan"
    )
    result = block.execute({"cloud": source})
    assert result and result.passed, block.error
    out = result.outputs
    clean, noisy = out["clean"], out["cloud"]
    np.testing.assert_array_equal(clean.points, [[10, -5, 4], [10, -4, 5], [9, -5, 6], [9, -4, 3]])
    np.testing.assert_array_equal(noisy.points, clean.points)
    np.testing.assert_allclose(
        noisy.normals, [[0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-6
    )
    assert (noisy.units, noisy.frame, noisy.source_size) == ("mm", "scan", (2, 2))
    assert out["pose"]["source_frame"] == "bench" and out["pose"]["target_frame"] == "scan"
    assert out["pairs"] == [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert noisy.metadata["capture_id"] == "source"
    assert noisy.metadata["synthetic_fixture"]["role"] == "perturbed"
    for values in ("colors", "pixels"):
        np.testing.assert_array_equal(getattr(source, values), getattr(noisy, values))
        assert not np.shares_memory(getattr(clean, values), getattr(noisy, values))
    for name, values in source.attributes.items():
        np.testing.assert_array_equal(noisy.attributes[name], values)
    assert out["report"]["displacement_max"] == 0


def test_seeded_streams_are_independent_of_dropout_noise_and_parallel_invocation(source):
    settings = dict(
        pose_mode="seeded",
        translation=[7, 4, 5],
        max_rotation_deg=30,
        seed=71,
        noise_distribution="uniform",
        noise_scale=[0.35, 0.2, 0.0],
    )
    original_state = np.random.get_state()
    with ThreadPoolExecutor(3) as pool:
        jobs = [
            pool.submit(
                fixture_block(**(settings | patch)).execute, {"cloud": source, "seed": seed}
            )
            for patch, seed in (
                ({}, 31),
                ({"dropout_fraction": 0.5}, 31),
                ({"noise_scale": [0, 0, 0]}, 31),
                ({}, 31),
                ({}, 32),
            )
        ]
        results = [job.result() for job in jobs]
    assert all(result and result.passed for result in results)
    full, half, no_noise, repeated, new = (result.outputs for result in results)
    assert full["report"]["seed"] == 31
    assert full["pose"] == half["pose"] == no_noise["pose"] == repeated["pose"]
    assert full["pose"] != new["pose"]
    assert len(half["pairs"]) == 2 and half["report"]["dropped_points"] == 2
    kept = np.asarray(half["pairs"])[:, 0]
    np.testing.assert_array_equal(half["cloud"].points, full["cloud"].points[kept])
    np.testing.assert_array_equal(full["cloud"].points, repeated["cloud"].points)
    np.testing.assert_array_equal(no_noise["cloud"].points, full["clean"].points)
    offsets = full["cloud"].points - full["clean"].points
    assert (np.abs(offsets[:, :2]) <= [0.35001, 0.20001]).all()
    assert not offsets[:, 2].any()
    final_state = np.random.get_state()
    assert final_state[0] == original_state[0] and final_state[2:] == original_state[2:]
    np.testing.assert_array_equal(final_state[1], original_state[1])


def test_fixture_admission_rejects_unrepresentable_rows_and_noninteger_seed(source):
    block = fixture_block()
    for count in (0, 4097):
        assert block.execute({"cloud": PointCloud(np.zeros((count, 3)))}) is None
        assert "4,096" in block.error
    assert block.execute({"cloud": source, "seed": 3.5}) is None
    assert "whole number" in block.error
    assert block.execute({"cloud": source, "seed": 3}).passed
