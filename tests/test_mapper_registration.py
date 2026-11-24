"""Check native mapper constraints, coordinate provenance and measured overlap."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.mapper_registration import AlignMapper
from opencv_dataflow.payloads import ValuePacket, frame

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/mapper_registration"
POINTS = np.array([[30, 30], [141, 43], [148, 127], [35, 134], [90, 80]], np.float64)
MAPS = {
    "translation": [[1, 0, 9.5], [0, 1, -7.25], [0, 0, 1]],
    "euclidean": np.vstack((cv2.getRotationMatrix2D((96.5, 80.5), 3, 1), [0, 0, 1])),
    "similarity": np.vstack((cv2.getRotationMatrix2D((96.5, 80.5), -4, 1.05), [0, 0, 1])),
    "affine": [[0.99, 0.045, 2.1], [-0.026, 1.025, 2], [0, 0, 1]],
    "projective": [[1.02, 0.037, 2.1], [-0.019, 1.02, 3], [0.00034, -0.00027, 1]],
}


@pytest.fixture
def reference():
    return cv2.imread(str(EXAMPLE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)


def project(points, matrix):
    return cv2.perspectiveTransform(points[:, None], np.asarray(matrix, np.float64))[:, 0]


def known_moving(reference, motion):
    return cv2.warpPerspective(
        reference, np.asarray(MAPS[motion]), reference.shape[::-1], flags=cv2.INTER_CUBIC
    )


@pytest.mark.parametrize("motion", MAPS)
def test_recovers_all_five_models_and_round_trips_points(reference, motion):
    moving = known_moving(reference, motion)
    block = AlignMapper(parameters={"motion": motion})
    result = block.execute({"reference": reference, "image": moving})
    assert result and result.passed, block.error
    transform = result.outputs["transform"]
    mapped = project(POINTS, transform["target_to_source"])
    assert np.linalg.norm(mapped - project(POINTS, MAPS[motion]), axis=1).max() < 0.15
    np.testing.assert_allclose(project(mapped, transform["source_to_target"]), POINTS, atol=1e-10)
    report = result.outputs["report"]
    assert report["coverage_fraction"] > 0.88
    assert report["normalized_rmse"] < 0.006
    assert report["final_comparison_rmse"] < report["initial_comparison_rmse"] / 15


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_graph_refines_seed_in_correct_order_and_preserves_frames(profile, workers):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text())["reference_to_moving"]
    graph = Graph.load(EXAMPLE / "mapper_registration.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        node = graph.blocks["register"]
        result = node.last_result
        assert result and result.passed, node.error
        report = result.outputs["report"]
        assert report["seeded"] and report["final_comparison_rmse"] < 0.02
        assert report["final_comparison_rmse"] < report["initial_comparison_rmse"] / 10
        transform = result.outputs["transform"]
        expected = project(POINTS, truth)
        inverse = np.asarray(transform["target_to_source"])
        assert np.linalg.norm(project(POINTS, inverse) - expected, axis=1).max() < 0.55
        coarse = graph.blocks["coarse"].last_result.outputs["transform"]
        seed = np.linalg.inv(coarse["source_to_target"])
        swapped = np.linalg.inv(seed) @ inverse @ seed
        assert np.linalg.norm(project(POINTS, swapped) - expected, axis=1).max() > 0.65
        assert transform["source_context"]["root_id"] == "moving:image"
        assert transform["target_context"]["root_id"] == "reference:image"
        for name in ("image", "coverage", "residual"):
            payload = result.payloads[name]
            assert payload["spatial"]["root_id"] == "reference:image"
            assert payload["ticket"]["value"] == "moving.png"
        assert graph.blocks["filmstrip"].last_result.passed
    finally:
        graph.close()


def test_bilinear_coverage_excludes_invalid_donors_and_residuals(reference):
    source = known_moving(reference, "translation")
    mask = np.ones(source.shape, np.uint8) * 255
    mask[40:100, 61] = 0
    ref_mask = np.zeros(source.shape, np.uint8)
    ref_mask[24:-24, 20:-20] = 255
    height, width = reference.shape
    region = {
        "model": "image_region_v1", "image_size": [width, height],
        "shape": "rectangle", "rectangle": [30, 35, 90, 73],
    }
    block = AlignMapper(parameters={"motion": "translation"})
    result = block.execute({
        "reference": reference, "image": source, "mask": mask,
        "reference_mask": ref_mask, "reference_region": region,
    })
    assert result and result.passed, block.error
    covered = result.outputs["coverage"] != 0
    assert not covered[:, :30].any() and not covered[108:].any()
    assert not covered[60, 51:53].any() and covered[60, 53]
    assert not result.outputs["residual"][~covered].any()
    assert result.outputs["report"]["coverage_fraction"] == pytest.approx(covered.sum() / (90 * 73))


def test_output_keeps_uint16_rgb_and_independent_quality_failures(reference):
    source = known_moving(reference, "translation").astype(np.uint16) * 257
    target = reference.astype(np.uint16) * 257
    source = ImageArray(np.repeat(source[..., None], 3, axis=2), "rgb")
    block = AlignMapper(parameters={
        "motion": "translation", "minimum_coverage": 0.99, "maximum_normalized_rmse": 0.00001,
    })
    result = block.execute({"reference": target, "image": source})
    assert result is not None and not result.passed and not block.error
    assert {reason["code"] for reason in result.fail_reasons} == {
        "mapper_insufficient_coverage", "mapper_residual_too_high",
    }
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgb"


@pytest.mark.parametrize(
    ("motion", "seed", "message"),
    [
        ("translation", [[1, 0.1, 0], [0, 1, 0]], "can only shift"),
        ("euclidean", [[1.1, 0, 0], [0, 1.1, 0]], "needs rotation"),
        ("similarity", [[1, 0.1, 0], [0, 1, 0]], "without shear"),
        ("affine", [[1, 0, 0], [0, 1, 0], [0.001, 0, 1]], "affine-form"),
        ("projective", [[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]], "projective pole"),
    ],
)
def test_rejects_seeds_outside_the_selected_model(reference, motion, seed, message):
    block = AlignMapper(parameters={"motion": motion})
    inputs = {"reference": reference, "image": reference, "initial_transform": seed}
    assert block.execute(inputs) is None
    assert message in block.error


def test_rejects_mismatched_seed_grid_and_unobservable_input(reference):
    height, width = reference.shape
    source = ValuePacket(reference, {"spatial": frame([width, height], "sample")})
    seed = {
        "source_to_target": np.eye(3).tolist(),
        "source_context": frame([width, height], "wrong sample"),
    }
    block = AlignMapper()
    inputs = {"reference": reference, "image": source, "initial_transform": seed}
    assert block.execute(inputs) is None
    assert "seed's source frame differs" in block.error
    stripes = np.broadcast_to(np.arange(width, dtype=np.uint8), reference.shape).copy()
    assert block.execute({"reference": stripes, "image": stripes}) is None
    assert "gradients in two directions" in block.error


def test_rejects_a_pyramid_whose_coarsest_image_is_too_small(reference):
    block = AlignMapper(parameters={"pyramid_levels": 8})
    assert block.execute({"reference": reference, "image": reference}) is None
    assert "coarsest level" in block.error
