"""Independent pixel-center references for mask boundary and area inspection."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, frame, metadata
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE = "inspect.mask_boundary"


def compare(reference, sample, parameters=None, **extra):
    block = builtin_registry().create(TYPE, parameters=parameters or {})
    result = block.execute({"reference": reference, "sample": sample, **extra})
    assert result is not None, block.error
    return block, result


def edge_by_neighbors(mask, connectivity=8, exterior=False):
    points = []
    for y, x in np.argwhere(mask):
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        if connectivity == 8:
            neighbors += [(y + dy, x + dx) for dy in (-1, 1) for dx in (-1, 1)]
        if any(
            not (mask[ny, nx] if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] else exterior)
            for ny, nx in neighbors
        ):
            points.append((y, x))
    return np.asarray(points, int).reshape(-1, 2)


def pair_distances(query, target, metric):
    differences = np.abs(query[:, None, :] - target[None, :, :])
    if metric == "manhattan":
        return differences.sum(axis=2).min(axis=1)
    if metric == "chessboard":
        return differences.max(axis=2).min(axis=1)
    return np.sqrt(np.square(differences).sum(axis=2)).min(axis=1)


@pytest.mark.parametrize("metric", ("euclidean", "manhattan", "chessboard"))
@pytest.mark.parametrize("connectivity", (4, 8))
def test_directed_maps_statistics_and_area_match_brute_force(metric, connectivity):
    reference = np.zeros((14, 19), np.uint16)
    reference[1:12, 2:15] = 50000
    reference[4:7, 6:9] = 0
    sample = reference.copy()
    sample[1:4, 12:15] = 0
    sample[9:13, 15:18] = 40000
    _, result = compare(
        reference, sample, {"distance_type": metric, "connectivity": str(connectivity)}
    )
    info = result.outputs["measurements"]
    edges = [edge_by_neighbors(image != 0, connectivity) for image in (reference, sample)]
    combined = []
    for name, query, target, port in (
        ("sample_to_reference", edges[1], edges[0], "sample_distance"),
        ("reference_to_sample", edges[0], edges[1], "reference_distance"),
    ):
        expected = pair_distances(query, target, metric)
        combined.extend(expected)
        np.testing.assert_allclose(result.outputs[port][tuple(query.T)], expected, atol=1e-6)
        report = info["directed"][name]
        assert report["count"] == len(query)
        assert report["mean"] == pytest.approx(np.mean(expected), abs=1e-6)
        assert report["rms"] == pytest.approx(np.sqrt(np.mean(expected**2)), abs=1e-6)
        assert report["percentile"] == pytest.approx(np.percentile(expected, 95), abs=1e-6)
    assert result.outputs["distance"] == pytest.approx(max(combined), abs=1e-6)
    a, b = reference != 0, sample != 0
    assert info["missing_pixels"] == 9
    assert info["extra_pixels"] == 12
    assert info["iou"] == pytest.approx((a & b).sum() / (a | b).sum())
    np.testing.assert_array_equal(result.outputs["missing"] != 0, a & ~b)
    np.testing.assert_array_equal(result.outputs["extra"] != 0, b & ~a)
    assert result.outputs["sample_distance"].dtype == np.float32


def test_holes_and_disconnected_components_cannot_hide_in_a_one_way_comparison():
    reference = np.zeros((23, 31), np.uint8)
    reference[2:21, 2:22] = 255
    sample = reference.copy()
    sample[9:14, 10:15] = 0
    sample[4:7, 27:30] = 255
    _, both = compare(reference, sample, {"tolerance": 0})
    assert not both.passed
    assert both.outputs["measurements"]["missing_pixels"] == 25
    assert both.outputs["measurements"]["extra_pixels"] == 9
    _, one_way = compare(reference, sample, {"direction": "reference_to_sample", "tolerance": 0})
    assert one_way.passed and one_way.outputs["distance"] == 0
    assert one_way.outputs["measurements"]["directed"]["sample_to_reference"]["maximum"] > 0


@pytest.mark.parametrize("statistic", ("maximum", "percentile", "mean", "rms"))
def test_combined_statistics_are_pixel_weighted_and_judged_inclusively(statistic):
    a, b = np.zeros((9, 12), np.uint8), np.zeros((9, 12), np.uint8)
    a[2:5, 2:6], b[2:6, 3:8] = 255, 255
    _, first = compare(a, b, {"statistic": statistic, "percentile": 63.0})
    expected = first.outputs["measurements"]["combined"][statistic]
    _, inclusive = compare(
        a, b, {"statistic": statistic, "percentile": 63.0, "tolerance": expected}
    )
    assert inclusive.passed
    _, strict = compare(
        a, b, {"statistic": statistic, "percentile": 63.0, "tolerance": expected / 2}
    )
    assert not strict.passed


def test_irrational_pixel_distance_does_not_fail_at_the_inclusive_limit():
    a, b = np.zeros((6, 8), np.uint8), np.zeros((6, 8), np.uint8)
    a[2, 2], b[3, 4] = 255, 255
    _, result = compare(a, b, {"tolerance": float(np.sqrt(5))})
    assert result.passed
    assert result.outputs["measurements"]["extra_beyond_tolerance_pixels"] == 0
    assert result.outputs["tolerance_band"][3, 4] == 255


def test_selection_restricts_queries_not_target_boundaries_or_neighbor_support():
    a = np.zeros((12, 16), np.uint8)
    a[2:10, 3:13] = 255
    b = a.copy()
    b[2:5, 10:13] = 0
    select = np.zeros_like(a)
    select[2:8, 3:8] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [16, 12],
        "shape": "rectangle",
        "rectangle": [3, 2, 5, 6],
    }
    _, result = compare(a, b, {"tolerance": 0}, mask=select, region=region)
    assert result.passed and result.outputs["distance"] == 0
    assert result.outputs["measurements"]["missing_pixels"] == 0
    # The region's right edge lies inside foreground, not on the actual boundary.
    assert not result.outputs["sample_boundary"][4, 7]
    assert np.count_nonzero(result.outputs["sample_boundary"]) == 10


@pytest.mark.parametrize("basis,passed", (("raw", False), ("beyond_tolerance", True)))
def test_area_tolerance_can_ignore_a_permitted_boundary_shift(basis, passed):
    a, b = np.zeros((15, 20), np.uint8), np.zeros((15, 20), np.uint8)
    a[3:12, 4:15], b[3:12, 5:16] = 255, 255
    _, result = compare(a, b, {"tolerance": 1, "check_area": True, "area_basis": basis})
    assert result.passed == passed
    info = result.outputs["measurements"]
    assert info["missing_pixels"] == info["extra_pixels"] == 9
    assert info["missing_beyond_tolerance_pixels"] == info["extra_beyond_tolerance_pixels"] == 0


def test_area_limits_and_reasons_are_independent_of_boundary_judgment():
    a = np.zeros((8, 10), np.uint8)
    a[1:7, 1:9] = 255
    b = a.copy()
    b[3:5, 4:6] = 0
    _, result = compare(
        a,
        b,
        {
            "check_boundary": False,
            "check_area": True,
            "area_basis": "raw",
            "maximum_missing_pixels": 4,
        },
    )
    assert result.passed
    _, failed = compare(
        a,
        b,
        {
            "check_boundary": False,
            "check_area": True,
            "area_basis": "raw",
            "maximum_missing_pixels": 3,
        },
    )
    assert [reason["code"] for reason in failed.fail_reasons] == ["mask_missing_area_above_limit"]


@pytest.mark.parametrize("policy,passed", (("fail", False), ("both_empty_pass", True)))
def test_both_empty_is_explicit_and_never_invents_a_distance(policy, passed):
    empty = np.zeros((5, 7), np.uint8)
    _, result = compare(empty, empty, {"empty_policy": policy})
    assert result.passed == passed
    assert result.outputs["distance"] is NO_VALUE
    assert result.outputs["sample_distance"] is result.outputs["reference_distance"] is NO_VALUE
    assert result.outputs["measurements"]["combined"] is None
    assert result.outputs["measurements"]["iou"] == 1


@pytest.mark.parametrize("which", ("reference", "sample"))
def test_one_empty_always_fails_and_reports_all_missing_or_extra_area(which):
    empty, filled = np.zeros((7, 9), np.uint8), np.zeros((7, 9), np.uint8)
    filled[2:5, 2:7] = 255
    a, b = (empty, filled) if which == "reference" else (filled, empty)
    _, result = compare(a, b, {"empty_policy": "both_empty_pass", "tolerance": 999})
    assert not result.passed
    kind = "extra" if which == "reference" else "missing"
    assert result.outputs["measurements"][kind + "_beyond_tolerance_pixels"] == 15
    assert result.outputs["distance"] is NO_VALUE


@pytest.mark.parametrize("border,valid", (("background", True), ("ignore", False)))
def test_full_masks_have_a_boundary_only_with_an_exterior_assumption(border, valid):
    a = np.full((5, 7), 255, np.uint8)
    _, result = compare(a, a, {"image_border": border})
    assert result.passed == valid
    assert result.outputs["measurements"]["distance_valid"] == valid


def test_zero_foreground_inverts_both_masks():
    a = np.full((5, 7), 65535, np.uint16)
    a[1:4, 2:5] = 0
    _, result = compare(a, a, {"foreground": "zero", "tolerance": 0})
    assert result.passed and result.outputs["measurements"]["reference_pixels"] == 9


def test_frame_ticket_payload_source_pixels_and_pyramid_ownership():
    a = np.zeros((9, 13), np.uint8)
    a[2:7, 3:10] = 255
    clock = TicketClock()
    payload = {
        "spatial": frame([13, 9], "camera"),
        "ticket": clock.issue("part-4"),
        "fixture": {"lot": "L"},
        "pyramid": {
            "level": 1,
            "largest_size": [26, 18],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    sample = ImageArray(a, "gray", payload)
    original = ImageArray(
        np.full((9, 13, 4), 12345, np.uint16),
        "rgba",
        {**payload, "alpha_representation": "premultiplied"},
    )
    reference = ImageArray(
        a, "gray", {"spatial": frame([13, 9], "reference"), "ticket": clock.issue("master")}
    )
    _, result = compare(reference, sample, {"alignment": "pixel_grid"}, image=original)
    np.testing.assert_array_equal(result.outputs["image"], original)
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgba"
    assert result.payloads["image"]["alpha_representation"] == "premultiplied"
    assert "alpha_representation" not in result.payloads["missing"]
    for name in ("missing", "measurements", "sample_distance"):
        context = result.payloads[name]
        assert context["ticket"] == payload["ticket"]
        assert context["spatial"] == payload["spatial"]
        assert context["pyramid"] == payload["pyramid"]
        assert context["fixture"] == payload["fixture"]
    assert metadata(sample)["fixture"] == payload["fixture"]
    for parameters, image in (
        ({}, original),
        ({"alignment": "pixel_grid", "tickets": "require_match"}, original),
    ):
        block = builtin_registry().create(TYPE, parameters=parameters)
        assert block.execute({"reference": reference, "sample": sample, "image": image}) is None
        assert block.error_kind == "invalid"
    wrong = ImageArray(np.asarray(original), "rgba", {**payload, "ticket": clock.issue("other")})
    block = builtin_registry().create(TYPE, parameters={"alignment": "pixel_grid"})
    assert block.execute({"reference": reference, "sample": sample, "image": wrong}) is None


def test_invalid_inputs_and_empty_selections_are_rejected():
    a = np.ones((5, 7), np.uint8)
    for bad in (
        a.astype(float),
        np.ones((5, 7, 3), np.uint8),
        a[:, :-1],
        np.zeros((2049, 2049), np.uint8),
    ):
        block = builtin_registry().create(TYPE)
        assert block.execute({"reference": a, "sample": bad}) is None
        assert block.error_kind == "invalid"
    block = builtin_registry().create(TYPE)
    assert block.execute({"reference": a, "sample": a, "mask": np.zeros_like(a)}) is None
    other = ImageArray(a, "gray", {"spatial": frame([7, 5], "other")})
    sample = ImageArray(a, "gray", {"spatial": frame([7, 5], "camera")})
    assert block.execute({"reference": sample, "sample": sample, "mask": other}) is None


def test_graphics_off_skips_previews_and_geometry(monkeypatch):
    import opencv_dataflow.mask_boundary as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected graphics work")

    monkeypatch.setattr(implementation, "overlay_image", unexpected)
    monkeypatch.setattr(implementation, "outlines", unexpected)
    a = np.zeros((5, 7), np.uint8)
    a[1:4, 1:6] = 255
    block = builtin_registry().create(TYPE)
    block.set_graphics_mode("off")
    result = block.execute({"reference": a, "sample": a})
    assert result is not None and result.passed, block.error
    assert result.outputs["preview"] is NO_VALUE and not result.graphics and not result.drawings


@pytest.mark.parametrize("aperture", ("3", "5"))
def test_approximate_l2_is_explicit_and_other_metrics_ignore_its_aperture(aperture):
    a, b = np.zeros((8, 10), np.uint8), np.zeros((8, 10), np.uint8)
    a[2, 2], b[4, 5] = 255, 255
    _, result = compare(a, b, {"l2_mask": aperture})
    assert result.outputs["distance"] == pytest.approx(np.sqrt(13), rel=0.05)
    _, l1 = compare(a, b, {"l2_mask": aperture, "distance_type": "manhattan"})
    assert l1.outputs["distance"] == 5


def test_single_direction_can_query_a_full_target_outside_the_selected_window():
    a, b = np.zeros((10, 15), np.uint8), np.zeros((10, 15), np.uint8)
    a[1:9, 2:10], b[1:9, 3:11] = 255, 255
    selected = np.zeros_like(a)
    selected[3:7, 3] = 255
    _, result = compare(a, b, {"direction": "sample_to_reference", "tolerance": 1}, mask=selected)
    assert result.passed and result.outputs["distance"] == 1
    assert not np.any(result.outputs["reference_boundary"])
    _, both = compare(a, b, {"tolerance": 1}, mask=selected)
    assert not both.passed and both.outputs["distance"] is NO_VALUE


def test_graphics_are_bounded_and_keep_the_sample_coordinate_grid():
    a = np.zeros((60, 80), np.uint8)
    a[1::3, 1::3] = 255
    payload = {"spatial": frame([80, 60], "fixture")}
    _, result = compare(
        ImageArray(a, "gray", payload), ImageArray(a, "gray", payload), {"graphics_points": 64}
    )
    assert result.passed
    assert result.drawings and result.outputs["preview"].shape == (60, 80, 3)
    assert all(
        sum(len(obj.points) for obj in drawing.objects) <= 64
        for drawing in result.drawings.values()
    )
    assert result.payloads["preview"]["spatial"] == payload["spatial"]


@pytest.mark.parametrize(
    "parameters", ({"tolerance": -1}, {"percentile": 101}, {"connectivity": "3"})
)
def test_invalid_parameters_use_the_common_validation_contract(parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    a = np.ones((5, 7), np.uint8)
    assert block.execute({"reference": a, "sample": a}) is None
    assert block.error_kind == "invalid"


def test_extreme_finite_tolerance_does_not_overflow_native_comparisons():
    a, b = np.zeros((5, 7), np.uint8), np.zeros((5, 7), np.uint8)
    a[1, 1], b[3, 5] = 255, 255
    with np.errstate(over="raise", invalid="raise"):
        _, result = compare(a, b, {"tolerance": np.finfo(float).max})
    assert result.passed
    assert not np.any(result.outputs["extra_beyond_tolerance"])


def test_fields_and_documentation_round_trip_through_xml(tmp_path):
    graph, registry = Graph(), builtin_registry()
    try:
        graph.add(
            registry.create(
                TYPE,
                parameters={
                    "alignment": "pixel_grid",
                    "direction": "sample_to_reference",
                    "check_area": True,
                },
            )
        )
        expected = graph.to_dict()
        path = tmp_path / "mask.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_examples_match_independent_area_and_distance_truth():
    folder = Path(__file__).parents[1] / "examples/opencv tools/mask_boundary"
    for filename, passed, missing, extra in (
        ("project.json", True, 0, 16),
        ("defect.json", False, 35, 16),
    ):
        graph = Graph.load(folder / filename, builtin_registry())
        try:
            graph.run()
            block = graph.blocks["compare"]
            assert block.state == ("pass" if passed else "fail"), block.error
            outputs = block.last_result.outputs
            assert outputs["measurements"]["missing_pixels"] == missing
            assert outputs["measurements"]["extra_pixels"] == extra
            assert outputs["distance"] == (4.0 if passed else 5.0)
        finally:
            graph.close()
