"""Check fit quality, support, and independent workplane validation."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import map_points, validate_profile
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

_PIXELS = np.float64([[10, 10], [95, 12], [90, 75], [14, 77], [56, 24], [55, 66]])
_MATRIX = np.array([[0.26, 0.03, -1], [-0.02, 0.23, 2], [0.0008, -0.0006, 1]])


def points_on_plane(points):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[None], _MATRIX)[0]


def fixture(**parameters):
    registry = builtin_registry()
    payload = {"spatial": frame([110, 90], "camera"), "ticket": TicketClock().issue("camera")}
    image = ValuePacket(np.zeros((90, 110), np.uint8), payload)
    fit = registry.create("calibration.fit_planar", parameters={
        "physical_points": points_on_plane(_PIXELS).tolist(), **parameters,
    })
    return registry, fit, image, payload


def test_direct_fit_preserves_origin_units_frame_and_ticket_for_validation():
    registry, fit, image, payload = fixture(method="direct")
    result = fit.execute({"image": image, "points": ValuePacket(_PIXELS.tolist(), payload)})
    assert result is not None and result.passed, fit.error
    assert result.outputs["inliers"] == 6
    assert result.outputs["rms"] < 1e-5
    mapping = result.outputs["mapping"]
    validate_profile(mapping)
    assert mapping["units"] == "mm"
    np.testing.assert_allclose(map_points(mapping, [mapping["origin_pixel"]]), [[0, 0]], atol=1e-8)
    held_out = [[22, 31], [74, 38], [44, 58]]
    check = registry.create("calibration.validate_planar", parameters={
        "physical_points": points_on_plane(held_out).tolist(), "maximum_rms": 1e-4,
    })
    review = check.execute({
        "image": image, "mapping": ValuePacket(mapping, result.payloads["mapping"]),
        "points": ValuePacket(held_out, payload),
    })
    assert review is not None and review.passed, check.error
    assert review.payloads["checked_mapping"]["ticket"] == payload["ticket"]
    assert review.payloads["checked_mapping"]["spatial"] == payload["spatial"]


def test_ransac_rejects_displaced_reference_and_can_require_more_support():
    _, fit, image, payload = fixture(inlier_tolerance=0.08, minimum_inliers=5)
    observed = _PIXELS.copy()
    observed[-1] += [14, -9]
    wires = {"image": image, "points": ValuePacket(observed.tolist(), payload)}
    result = fit.execute(wires)
    assert result is not None and result.passed, fit.error
    assert result.outputs["report"]["inlier_mask"] == [True] * 5 + [False]
    assert result.outputs["rms"] < 1e-5
    fit.parameters["minimum_inliers"] = 6
    result = fit.execute(wires)
    assert result.outputs["mapping"] is NO_VALUE and not result.passed
    assert result.fail_reasons[0]["code"] == "planar_fit_inliers_below_minimum"


@pytest.mark.parametrize("points", [
    [[10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10]],
    [[10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [45, 70]],
])
def test_degenerate_image_points_do_not_release_a_mapping(points):
    _, fit, image, payload = fixture(method="direct")
    result = fit.execute({"image": image, "points": ValuePacket(points, payload)})
    assert result is not None and not result.passed, fit.error
    assert result.outputs["mapping"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "planar_fit_geometry_invalid"


def test_planar_fit_rejects_points_from_another_capture():
    _, fit, image, payload = fixture()
    stale = {**payload, "ticket": TicketClock().issue("camera")}
    assert fit.execute({"image": image, "points": ValuePacket(_PIXELS.tolist(), stale)}) is None
    assert "current capture" in fit.error
