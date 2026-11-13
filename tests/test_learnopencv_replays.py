"""Optional local tutorial studies, verified against independent original truth."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import image_format
from opencv_dataflow.payloads import NO_VALUE

STUDIES = Path(__file__).resolve().parents[1] / "examples/private/learnopencv"


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_local_planar_mosaic_reconstructs_independent_pixels_from_four_pairs(
    profile, tmp_path, monkeypatch
):
    folder = STUDIES / "73_planar_mosaic"
    if not (folder / "73_planar_mosaic.json").exists():
        pytest.skip("Private LearnOpenCV study collection is absent")
    monkeypatch.chdir(tmp_path)
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    expected = cv2.imread(str(folder / "assets/plane.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    graph = Graph.load(folder / "73_planar_mosaic.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        result = graph.blocks["mosaic"].last_result
        output = result.outputs["image"]
        assert output.dtype == np.uint16 and image_format(output) == truth["output_encoding"]
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_allclose(
            graph.blocks["fit"].last_result.outputs["transform"]["source_to_target"],
            truth["b_to_a"],
            atol=1e-9,
        )
        layout = result.outputs["layout"]
        assert layout["image_size"] == truth["image_size"]
        assert np.count_nonzero(result.outputs["coverage"]) == truth["covered_pixels"]
        assert np.count_nonzero(result.outputs["overlap"]) == truth["overlap_pixels"]
        assert [row["source_ticket"]["value"] for row in layout["views"]] == [
            "tile_a.png",
            "tile_b.png",
        ]
        assert result.payloads["image"]["ticket"]["value"] == "tile_a.png"
        assert "LearnOpenCV" in graph.blocks["attribution"].description
    finally:
        graph.close()


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_local_mosaic_missing_overlap_withholds_image(profile):
    project = STUDIES / "73_planar_mosaic/insufficient_overlap.json"
    if not project.exists():
        pytest.skip("Private LearnOpenCV study collection is absent")
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["mosaic"]
        assert block.state == "fail" and not block.error
        assert block.last_result.outputs["image"] is NO_VALUE
        assert block.current_emissions["image"] == []
        assert block.last_result.outputs["layout"]["overlap_pixels"] == 0
        assert not block.last_result.outputs["overlap"].any()
        assert block.last_result.fail_reasons[0]["code"] == "stitch_insufficient_overlap"
        assert graph.blocks["review"].state == "inactive"
    finally:
        graph.close()


def _known_smoothed_path(raw, radius, sigma):
    # Independent scalar formulation; no use of the block's reported poses.
    result = []
    for index in range(len(raw)):
        indices = range(max(0, index - radius), min(len(raw), index + radius + 1))
        weights = [np.exp(-0.5 * ((other - index) / sigma) ** 2) for other in indices]
        result.append(
            [
                sum(w * raw[other, axis] for other, w in zip(indices, weights)) / sum(weights)
                for axis in range(2)
            ]
        )
    return np.array(result)


def _acceleration_rms(path):
    acceleration = path[2:] - 2 * path[1:-1] + path[:-2]
    return np.sqrt(np.mean(np.sum(acceleration**2, axis=1)))


def test_local_stabilized_samples_and_coverage_are_identical_in_debug_and_lean():
    project = STUDIES / "74_camera_stabilization/74_camera_stabilization.json"
    if not project.exists():
        pytest.skip("Private LearnOpenCV study collection is absent")
    outputs = []
    for profile in ("debug", "lean"):
        graph = Graph.load(project, builtin_registry())
        graph.set_run_profile(profile)
        try:
            for _ in range(12):
                graph.run()
            block = graph.blocks["smooth"]
            assert block.state == "pass", block.error
            result = block.last_result.outputs
            outputs.append(
                tuple(
                    tuple(item.value.copy() for item in result[name].items)
                    for name in ("corrected", "coverage")
                )
            )
        finally:
            graph.close()
    for debug_group, lean_group in zip(*outputs, strict=True):
        for debug, lean in zip(debug_group, lean_group, strict=True):
            np.testing.assert_array_equal(debug, lean)


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_local_stabilization_matches_known_motion_landmarks_and_replays_each_ticket(
    profile, tmp_path, monkeypatch
):
    folder = STUDIES / "74_camera_stabilization"
    if not (folder / "74_camera_stabilization.json").exists():
        pytest.skip("Private LearnOpenCV study collection is absent")
    monkeypatch.chdir(tmp_path)
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    raw = np.array(truth["shifts"], dtype=np.float64)
    desired = _known_smoothed_path(raw, truth["radius"], truth["sigma"])
    graph = Graph.load(folder / "74_camera_stabilization.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        for index in range(truth["frame_count"]):
            graph.run()
            if index < truth["frame_count"] - 1:
                assert graph.blocks["smooth"].state == graph.blocks["replay"].state == "inactive"
        assert graph.sources_exhausted()
        block = graph.blocks["smooth"]
        assert block.state == "pass", block.error
        result = block.last_result.outputs
        report = result["report"]
        assert report["status"] == "complete" and report["segment_starts"] == [0]
        frames = report["frames"]
        assert frames[0]["window"] == [0, 4] and frames[-1]["window"] == [8, 12]
        observed = np.array([row["observed_pose"][:2] for row in frames])
        smoothed = np.array([row["smoothed_pose"][:2] for row in frames])
        np.testing.assert_allclose(observed, raw, atol=truth["observed_max_error"])
        np.testing.assert_allclose(smoothed, desired, atol=truth["observed_max_error"])
        assert _acceleration_rms(smoothed) < truth[
            "maximum_acceleration_ratio"
        ] * _acceleration_rms(raw)
        assert min(row["coverage_fraction"] for row in frames) > truth["minimum_coverage"]
        emitted = graph.blocks["replay"].current_emissions["value"]
        reviewed = graph.blocks["review"].current_emissions["image"]
        assert len(emitted) == len(reviewed) == len(raw)
        for index, (image, coverage, transform, packet, review) in enumerate(
            zip(
                result["corrected"].items,
                result["coverage"].items,
                result["transforms"].items,
                emitted,
                reviewed,
                strict=True,
            )
        ):
            assert image.value.dtype == np.uint8 and image_format(image.value) == "gray"
            ticket = image.payload["ticket"]
            assert ticket["value"] == f"frame-{index:02d}.png"
            assert ticket == coverage.payload["ticket"] == transform.payload["ticket"]
            assert ticket == packet.payload["ticket"] == review.payload["ticket"]
            np.testing.assert_array_equal(packet.value, image.value)
            np.testing.assert_array_equal(review.value, image.value)
            assert np.all(image.value[coverage.value == 0] == 0)
            yy, xx = np.nonzero(image.value >= 250)
            assert len(xx) == 81
            np.testing.assert_allclose(
                [xx.mean(), yy.mean()],
                np.array(truth["landmark"]) + desired[index],
                atol=truth["landmark_max_error"],
            )
            target = transform.value["target_spatial"]
            assert target == coverage.payload["spatial"] == image.payload["spatial"]
            raw_point = [*(np.array(truth["landmark"]) + raw[index]), 1]
            to_root = np.asarray(target["to_root"]) @ np.asarray(
                transform.value["source_to_target"]
            )
            np.testing.assert_allclose(to_root @ raw_point, [*truth["landmark"], 1], atol=0.2)
        assert "LearnOpenCV" in graph.blocks["attribution"].description
    finally:
        graph.close()
