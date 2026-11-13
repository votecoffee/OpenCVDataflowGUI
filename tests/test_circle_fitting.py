"""Analytic circle/arc truth, robust support, conditioning and graph interoperability."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import collection
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata, moved_payload
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE_ID = "inspect.fit_circle_arc"
PROJECT = Path(__file__).parents[1] / "examples/opencv tools/circle_arc_fitting/project.json"


def arc_points(center=(90, 80), radius=40, start=350, sweep=210, count=160, noise=0):
    angles = np.radians(np.linspace(start, start + sweep, count))
    radial = radius + np.random.default_rng(134).normal(0, noise, count)
    return np.asarray(center) + radial[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))


def fit(points=None, *, parameters=None, extra=None):
    block = builtin_registry().create(TYPE_ID, parameters=parameters or {})
    value = arc_points() if points is None else points
    value = value.tolist() if isinstance(value, np.ndarray) else value
    result = block.execute({"points": value, **(extra or {})})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("method", ("ransac", "least_squares"))
@pytest.mark.parametrize("sweep", (210, -210))
def test_exact_arcs_fit_circle_geometry_and_cross_the_angular_branch_cut(method, sweep):
    _, result = fit(arc_points(sweep=sweep), parameters={"method": method})
    assert result.passed
    np.testing.assert_allclose(result.outputs["center"], [90, 80], atol=1e-10)
    assert result.outputs["radius"] == pytest.approx(40, abs=1e-10)
    assert result.outputs["rmse"] < 1e-10
    arc = result.outputs["arc"]
    assert arc["sweep_degrees"] == pytest.approx(sweep)
    assert arc["support_span_degrees"] == pytest.approx(abs(sweep))
    assert arc["start_angle_degrees"] == pytest.approx(350)
    assert arc["end_angle_degrees"] == pytest.approx((350 + sweep) % 360)
    assert arc["backtracking_fraction"] < 1e-12


def test_closed_path_has_consistent_endpoint_and_does_not_claim_complete_observation():
    points = arc_points(start=15, sweep=360, count=121)
    points[-1] = points[0]
    _, result = fit(points)
    arc = result.outputs["arc"]
    assert result.passed and arc["closed_path"]
    assert arc["sweep_degrees"] == pytest.approx(360)
    assert arc["end_angle_degrees"] == pytest.approx(arc["start_angle_degrees"])
    assert arc["support_span_degrees"] == pytest.approx(357)
    assert result.outputs["measurements"]["eligible_count"] == 120


def test_ransac_rejects_outliers_and_refits_the_final_consensus():
    points = arc_points(noise=0.15)
    points[::9] += np.random.default_rng(5).normal(0, 12, (len(points[::9]), 2))
    _, robust = fit(points)
    _, ordinary = fit(points, parameters={"method": "least_squares", "check_rmse": False})
    assert robust.passed
    np.testing.assert_allclose(robust.outputs["center"], [90, 80], atol=0.05)
    assert robust.outputs["radius"] == pytest.approx(40, abs=0.05)
    assert robust.outputs["rmse"] < 0.2
    assert ordinary.outputs["rmse"] > 2
    report = robust.outputs["measurements"]
    assert 0.85 < report["inlier_fraction"] < 0.95
    assert report["solver"]["trials"] <= 256
    for row in robust.outputs["residuals"]["rows"]:
        assert row["used_for_fit"] == (abs(row["signed_radial_error"]) <= 1.0)
    again = fit(points)[1]
    assert robust.outputs["measurements"] == again.outputs["measurements"]


@pytest.mark.parametrize(
    ("center", "radius", "tolerance"),
    [((1e7 - 10, -1e7 + 10), 2, 1e-8), ((0.01, -0.02), 1e-5, 1e-12)],
)
def test_local_conditioning_handles_small_features_and_large_coordinates(center, radius, tolerance):
    _, result = fit(arc_points(center=center, radius=radius), parameters={"minimum_radius": 1e-9})
    assert result.passed
    np.testing.assert_allclose(result.outputs["center"], center, atol=tolerance, rtol=0)
    assert result.outputs["radius"] == pytest.approx(radius, abs=tolerance)


def test_uncertainty_agrees_with_full_circle_radial_noise_model_and_scales_with_pixels():
    points = arc_points(start=0, sweep=360, count=241, noise=0.1)
    _, result = fit(points, parameters={"path": "closed"})
    assert result.passed
    uncertainty = result.outputs["measurements"]["uncertainty"]
    n = result.outputs["measurements"]["inlier_count"]
    sigma = uncertainty["radial_noise_std_px"]
    expected = np.array([np.sqrt(2), np.sqrt(2), 1]) * sigma / np.sqrt(n)
    np.testing.assert_allclose(uncertainty["std_px"], expected, rtol=0.02)
    covariance = np.asarray(uncertainty["covariance_px2"])
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15
    _, scaled = fit(
        points * 10 + [1000, -2000], parameters={"path": "closed", "inlier_tolerance": 10}
    )
    np.testing.assert_allclose(
        scaled.outputs["measurements"]["uncertainty"]["covariance_px2"],
        covariance * 100,
        rtol=1e-6,
        atol=1e-10,
    )


def test_three_point_fit_can_be_requested_without_inventing_uncertainty():
    _, result = fit(arc_points(start=0, sweep=180, count=3), parameters={"minimum_inliers": 3})
    assert result.passed and result.outputs["measurements"]["uncertainty"] is None


def test_duplicate_coordinates_vote_once_and_do_not_shrink_uncertainty():
    points = arc_points(noise=0.1)
    _, one = fit(points)
    _, repeated = fit(np.vstack((points, points)))
    assert repeated.outputs["measurements"]["input_count"] == 320
    assert repeated.outputs["measurements"]["unique_count"] == 160
    assert (
        repeated.outputs["measurements"]["uncertainty"]
        == one.outputs["measurements"]["uncertainty"]
    )


@pytest.mark.parametrize("case", ("short", "ill_conditioned", "collinear", "identical"))
def test_unqualified_geometry_is_withheld_as_a_failed_inspection(case):
    parameters = {}
    if case in ("short", "ill_conditioned"):
        points = arc_points(sweep=5)
        if case == "ill_conditioned":
            parameters = {"minimum_span_degrees": 0, "maximum_condition": 100}
    elif case == "collinear":
        points = np.column_stack((np.arange(20), np.arange(20) * 2))
    else:
        points = np.full((20, 2), 5.0)
    block, result = fit(points, parameters=parameters)
    assert not result.passed and block.state == "fail" and not block.error
    assert not result.outputs["measurements"]["circle_valid"]
    assert result.outputs["fit"] is result.outputs["center"] is result.outputs["radius"] is NO_VALUE
    assert result.fail_reasons


def test_arc_order_and_rmse_judgments_are_separate_from_circle_validity():
    points = arc_points(noise=0.1)
    shuffled = points[np.random.default_rng(3).permutation(len(points))]
    _, result = fit(shuffled)
    assert not result.passed and result.outputs["measurements"]["circle_valid"]
    assert not result.outputs["measurements"]["arc_valid"]
    assert result.outputs["fit"] is not NO_VALUE and result.outputs["arc"] is NO_VALUE
    _, noisy = fit(points, parameters={"maximum_rmse": 0.01})
    assert not noisy.passed and noisy.outputs["measurements"]["circle_valid"]
    assert "circle_rmse_out_of_tolerance" in noisy.outputs["measurements"]["status"]


def test_radius_and_inlier_fraction_limits_gate_primary_geometry():
    points = arc_points(noise=0.1)
    _, result = fit(points, parameters={"method": "least_squares", "maximum_radius": 30})
    assert result.outputs["fit"] is NO_VALUE
    assert "circle_radius_out_of_range" in result.outputs["measurements"]["status"]
    points[::3] += [0, 25]
    _, result = fit(points, parameters={"minimum_inlier_fraction": 0.95})
    assert not result.passed and result.outputs["fit"] is NO_VALUE


def test_mask_and_region_exclude_points_before_fitting():
    points = arc_points(center=(64, 64), radius=32)
    image = np.zeros((128, 128), np.uint8)
    mask = np.zeros_like(image)
    mask[:, 55:] = 1
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [128, 128],
        "rectangle": [0, 0, 128, 95],
    }
    _, result = fit(points, extra={"image": image, "mask": mask, "region": region})
    assert result.passed
    assert 6 < result.outputs["measurements"]["eligible_count"] < len(points)
    assert result.outputs["radius"] == pytest.approx(32, abs=1e-9)
    for row in result.outputs["residuals"]["rows"]:
        x, y = np.rint(row["point"]).astype(int)
        assert row["eligible"] == (x >= 55 and y < 95)


def test_auto_closed_contour_trimmed_by_a_region_becomes_an_observed_open_arc():
    points = arc_points(center=(64, 64), radius=32, start=0, sweep=360, count=181)
    points[-1] = points[0]
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [128, 128],
        "rectangle": [64, 0, 64, 128],
    }
    _, result = fit(points, extra={"image": np.zeros((128, 128), np.uint8), "region": region})
    assert result.passed
    arc = result.outputs["arc"]
    assert not arc["closed_path"] and arc["sweep_degrees"] == pytest.approx(180)
    assert arc["maximum_ordered_gap_degrees"] == pytest.approx(2)


def test_subnormal_finite_points_fail_without_overflowing_the_solver():
    _, result = fit(arc_points(center=(0, 0), radius=1e-250))
    assert not result.passed and result.outputs["fit"] is NO_VALUE


def test_collection_origin_and_pyramid_mapping_preserve_the_image_and_context():
    points = arc_points(start=0, sweep=360, count=181)
    source_frame = frame([180, 160], "camera")
    source_frame["root_size"] = [300, 300]
    ticket = TicketClock("circle-context").issue("camera")
    source_payload = {"spatial": source_frame, "ticket": ticket, "fixture": {"value": 134}}
    offset = [100, 200]
    shifted = moved_payload(source_payload, [[1, 0, -100], [0, 1, -200], [0, 0, 1]], [180, 160])
    data = collection(
        "contours",
        [
            {
                "id": 4,
                "metrics": {},
                "geometry": {
                    "points": (points + offset).tolist(),
                },
            }
        ],
        np.zeros((160, 180), np.uint8),
        shifted,
        origin=offset,
    )
    target_frame = frame([90, 80], "camera")
    target_frame["root_size"] = [300, 300]
    target_frame["to_root"] = [[2, 0, 10], [0, 2, 20], [0, 0, 1]]
    target_payload = {
        "spatial": target_frame,
        "ticket": ticket,
        "pyramid": {"level": 1, "largest_size": [180, 160], "to_largest": target_frame["to_root"]},
    }
    image = ImageArray(np.full((80, 90, 3), 12345, np.uint16), "rgb", target_payload)
    original = copy.deepcopy(data)
    block, result = fit(ValuePacket(data, shifted), extra={"image": image})
    assert result.passed
    np.testing.assert_allclose(result.outputs["center"], [40, 30], atol=1e-9)
    assert result.outputs["radius"] == pytest.approx(20, abs=1e-9)
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert (
        result.outputs["image"].dtype == np.uint16 and result.outputs["image"].color_format == "rgb"
    )
    assert data == original
    for name in ("fit", "arc", "points", "image", "measurements"):
        assert result.payloads[name]["spatial"] == target_frame
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["pyramid"] == target_payload["pyramid"]
        assert result.payloads[name]["fixture"] == {"value": 134}
    assert metadata(result.outputs["image"]) == result.payloads["image"]
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"points": ValuePacket(data, shifted), "image": image})
    assert result.outputs["result_graphics"].layers[0].payload["spatial"] == target_frame


def test_blob_boundary_selection_can_fit_a_bore_and_feed_the_region_tool():
    registry = builtin_registry()
    image = np.zeros((100, 100), np.uint8)
    cv2.circle(image, (50, 50), 32, 255, -1)
    cv2.circle(image, (50, 50), 10, 0, -1)
    detector = registry.create("inspect.blobs", parameters={"min_area": 1})
    detected = detector.execute({"image": image})
    packet = ValuePacket(detected.outputs["collection"], detected.payloads["collection"])
    _, outer = fit(packet, extra={"image": image})
    _, inner = fit(packet, parameters={"boundary_index": 1}, extra={"image": image})
    assert outer.passed and inner.passed
    assert inner.outputs["radius"] < outer.outputs["radius"] / 2
    region = registry.create("region.fitted_shape")
    result = region.execute(
        {"image": image, "fit": ValuePacket(inner.outputs["fit"], inner.payloads["fit"])}
    )
    assert result is not None, region.error
    assert result.outputs["mask"][50, 50] == 255


def test_default_blob_boundary_selection_preserves_small_features_at_large_origins():
    offset = np.array([9_000_000, 9_000_000])
    inner = arc_points(center=(32, 32), radius=0.01, start=0, sweep=360, count=61)
    outer = arc_points(center=(32, 32), radius=0.02, start=0, sweep=360, count=61)
    item = {
        "id": 1,
        "metrics": {},
        "geometry": {
            "contours": [(inner + offset).tolist(), (outer + offset).tolist()],
        },
    }
    data = collection("blobs", [item], np.zeros((64, 64), np.uint8), {}, origin=offset)
    _, result = fit(data, parameters={"minimum_radius": 1e-4, "method": "least_squares"})
    assert result.passed
    assert result.outputs["radius"] == pytest.approx(0.02, abs=1e-8)


@pytest.mark.parametrize("kind", ("frame", "ticket", "mask_frame"))
def test_incompatible_frames_tickets_and_selection_fail_validation(kind):
    payload = {"spatial": frame([180, 160], "camera")}
    target = copy.deepcopy(payload)
    extra = {}
    if kind == "frame":
        target["spatial"]["root_id"] = "other"
    elif kind == "ticket":
        clock = TicketClock("circle-tickets")
        payload["ticket"], target["ticket"] = clock.issue("a"), clock.issue("b")
    else:
        extra["mask"] = ImageArray(
            np.ones((160, 180), np.uint8), "gray", {"spatial": frame([180, 160], "other")}
        )
    block = builtin_registry().create(TYPE_ID)
    assert (
        block.execute(
            {
                "points": ValuePacket(arc_points().tolist(), payload),
                "image": ImageArray(np.zeros((160, 180), np.uint8), "gray", target),
                **extra,
            }
        )
        is None
    )
    assert block.error_kind == "invalid"


@pytest.mark.parametrize("points", ([], [[1, 2, 3]] * 6, [[0, float("nan")]] * 6, [["bad", 0]] * 6))
def test_malformed_point_data_is_a_validation_error(points):
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"points": points}) is None
    assert block.error_kind == "invalid"


def test_iteration_and_configuration_budgets_are_checked_before_fitting():
    points = arc_points(count=9000).tolist()
    block = builtin_registry().create(TYPE_ID, parameters={"maximum_trials": 4096})
    assert block.execute({"points": points}) is None
    assert "32M point trials" in block.error
    for update in (
        {"confidence": 1},
        {"inlier_tolerance": 0},
        {"minimum_inliers": 2},
        {"minimum_radius": 20, "maximum_radius": 10},
        {"maximum_trials": -1},
    ):
        block = builtin_registry().create(TYPE_ID, parameters=update)
        assert block.execute({"points": arc_points().tolist()}) is None
        assert block.error_kind == "invalid"


def test_nonconvergent_refinement_withholds_primary_geometry():
    points = arc_points(noise=0.5)
    _, result = fit(
        points,
        parameters={
            "method": "least_squares",
            "refinement_iterations": 1,
            "position_tolerance": 1e-12,
        },
    )
    assert not result.passed and result.outputs["fit"] is NO_VALUE
    assert "circle_refinement_not_converged" in result.outputs["measurements"]["status"]


def test_graphics_and_residual_retention_can_be_disabled_without_changing_fit(monkeypatch):
    import opencv_dataflow.circle_fitting as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected preview allocation")

    monkeypatch.setattr(implementation, "overlay_image", unexpected)
    block = builtin_registry().create(TYPE_ID, parameters={"sample_limit": 0})
    block.set_graphics_mode("off")
    result = block.execute(
        {"points": arc_points().tolist(), "image": np.zeros((160, 180), np.uint8)}
    )
    assert result is not None and result.passed, block.error
    assert not result.graphics and not result.drawings and result.outputs["overlay"] is NO_VALUE
    assert result.outputs["residuals"]["rows"] == [] and len(result.outputs["points"]) == 160


def test_parameter_and_documentation_xml_round_trip(tmp_path):
    graph, registry = Graph(), builtin_registry()
    try:
        block = graph.add(
            registry.create(TYPE_ID, parameters={"method": "least_squares", "path": "open"})
        )
        block.description, block.documentation = "Circle inspection", "<p>Arc setup.</p>"
        expected = graph.to_dict()
        path = tmp_path / "circle.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_example_matches_independent_generating_truth():
    graph = Graph.load(PROJECT, builtin_registry())
    try:
        graph.run()
        assert graph.blocks["robust"].state == graph.blocks["circle"].state == "pass"
        robust = graph.blocks["robust"].last_result
        assert robust.outputs["image"].dtype == np.uint16
        np.testing.assert_allclose(robust.outputs["center"], [128, 96], atol=0.1)
        assert robust.outputs["radius"] == pytest.approx(60, abs=0.1)
        assert all(block.state == "pass" for block in graph.processing_blocks())
    finally:
        graph.close()
    negative = Graph.load(PROJECT.with_name("short_arc.json"), builtin_registry())
    try:
        negative.run()
        assert negative.blocks["short"].state == "fail"
        assert negative.blocks["short"].last_result.outputs["fit"] is NO_VALUE
        assert negative.blocks["short"].last_result.outputs["arc"] is NO_VALUE
    finally:
        negative.close()
