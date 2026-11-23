"""Check native shape-context settings and closed-contour inspection workflows."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/shape_context/shape_context.json"
REFERENCE = [[0, 0], [8, 0], [8, 5], [0, 5]]
SAMPLE = [[0, 0], [9, 0], [9, 4], [0, 4]]


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def rectangle_samples(width, height, count):
    """Independent analytic perimeter positions for an axis-aligned rectangle."""
    distance = np.arange(count) * (2 * (width + height) / count)
    points = np.column_stack(
        (
            np.clip(distance, 0, width) - np.clip(distance - width - height, 0, width),
            np.clip(distance - width, 0, height)
            - np.clip(distance - 2 * width - height, 0, height),
        )
    )
    return (points - points.mean(axis=0)).astype(np.float32).reshape(1, -1, 2)


@pytest.mark.parametrize(
    ("cost", "norm", "factory", "flag", "alignment"),
    [
        ("chi-square", "L2", cv2.createChiHistogramCostExtractor, (), "thin-plate spline"),
        ("norm", "L1", cv2.createNormHistogramCostExtractor, (cv2.DIST_L1,), "full affine"),
        ("norm", "L2", cv2.createNormHistogramCostExtractor, (cv2.DIST_L2,), "partial affine"),
        ("earth mover", "L1", cv2.createEMDHistogramCostExtractor, (cv2.DIST_L1,), "full affine"),
        ("earth mover L1", "L2", cv2.createEMDL1HistogramCostExtractor, (), "thin-plate spline"),
    ],
)
def test_native_distance_settings(registry, cost, norm, factory, flag, alignment):
    block = registry.create(
        "inspect.shape_context_distance",
        parameters={
            "samples": 32,
            "angular_bins": 8,
            "radial_bins": 3,
            "inner_radius": 0.15,
            "outer_radius": 2.2,
            "iterations": 2,
            "rotation_invariant": True,
            "histogram_cost": cost,
            "histogram_norm": norm,
            "dummy_count": 5,
            "dummy_cost": 0.3,
            "alignment": alignment,
            "shape_weight": 1.5,
            "bending_weight": 0.6,
        },
    )
    spline = alignment == "thin-plate spline"
    transformer = (
        cv2.createThinPlateSplineShapeTransformer()
        if spline
        else cv2.createAffineTransformer(alignment == "full affine")
    )
    native = cv2.createShapeContextDistanceExtractor(
        8, 3, 0.15, 2.2, 2, factory(*flag, 5, 0.3), transformer
    )
    native.setRotationInvariant(True)
    native.setShapeContextWeight(1.5)
    native.setBendingEnergyWeight(0.6 if spline else 0)
    expected = native.computeDistance(rectangle_samples(8, 5, 32), rectangle_samples(9, 4, 32))
    result = block.execute({"reference": REFERENCE, "sample": SAMPLE})
    assert result and result.passed, block.error
    assert result.outputs["distance"] == pytest.approx(expected, abs=1e-6)


def test_closed_contour_order_and_bad_geometry(registry):
    block = registry.create("inspect.shape_context_distance")
    original = block.execute({"reference": REFERENCE, "sample": SAMPLE})
    ring = np.roll(np.asarray(SAMPLE)[::-1], 2, axis=0) + [1234, -5678]
    closed = np.vstack((ring[:1], ring, ring[:1]))
    reordered = block.execute({"reference": REFERENCE, "sample": closed.tolist()})
    assert reordered and original
    assert reordered.outputs == original.outputs
    assert block.execute({"reference": REFERENCE, "sample": [[0, 0], [1, 1], [2, 2]]}) is None
    assert "nonzero area" in block.error


@pytest.mark.parametrize(
    ("settings", "fields"),
    [
        ({"inner_radius": 1, "outer_radius": 0.5}, {"inner_radius", "outer_radius"}),
        ({"alignment": "partial affine", "shape_weight": 0}, {"alignment", "shape_weight"}),
        ({"samples": 16, "dummy_count": 16}, {"dummy_count", "samples"}),
        (
            {"histogram_cost": "earth mover", "samples": 128, "iterations": 6},
            {"samples", "iterations"},
        ),
    ],
)
def test_incompatible_parameters_are_reported(registry, settings, fields):
    block = registry.create("inspect.shape_context_distance")
    with pytest.raises(ValidationError) as caught:
        block.validate_parameters({**block.parameters, **settings})
    assert fields <= set(caught.value.fields)


def test_saved_example_rank_and_optional_threshold(registry):
    graph = Graph.load(EXAMPLE, registry)
    try:
        graph.run()
        same, deformed, different = (
            graph.blocks[f"{name}_distance"] for name in ("reference", "deformed", "circle")
        )
        assert block_entry(same).path[:2] == ("OpenCV", "Shape Distance and Matching (shape)")
        for block in (same, deformed, different):
            assert block.last_result and block.last_result.passed, block.error
        a, b, c = (block.last_result.outputs["distance"] for block in (same, deformed, different))
        assert 0 <= a < b < 0.05 < c
        for block in (deformed, different):
            block.parameters["check_distance"] = True
        graph.run()
        assert deformed.last_result.passed
        assert not different.last_result.passed
        assert different.last_result.outputs["distance"] == c
        (reason,) = different.last_result.fail_reasons
        assert reason["code"] == "contour_distance_out_of_tolerance"
        assert "exceeds 0.05" in reason["message"]
    finally:
        graph.close()
