"""Native contour sampling, spectral coefficients, and alignment of known silhouettes."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/fourier_contours"
DESCRIBE = "inspect.fourier_contour"
FIT = "inspect.fit_fourier_contours"
CONTOUR = np.array([
    [35, 40], [110, 40], [124, 56], [123, 83], [96, 83], [96, 108],
    [117, 108], [110, 132], [44, 132], [32, 108], [47, 85], [35, 72],
], dtype=float)


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def test_periodic_sampling_and_compact_spectrum_match_native(registry):
    square = np.array([[4, 5], [12, 5], [12, 13], [4, 13], [4, 5]], float)
    full = registry.create(DESCRIBE, parameters={"samples": 32})
    compact = registry.create(DESCRIBE, parameters={"samples": 32, "coefficients": 8})
    context = {"spatial": frame([20, 20], "square")}
    result = full.execute({"contour": ValuePacket(square.tolist(), context)})
    assert result and result.passed, full.error
    points = np.asarray(result.outputs["samples"])
    np.testing.assert_allclose(points[0], [4, 5], atol=1e-7)
    assert np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) == pytest.approx(np.ones(32))
    assert result.payloads["descriptor"]["spatial"] == context["spatial"]
    descriptor = result.outputs["descriptor"]
    normalized = ((points - descriptor["center"]) / descriptor["rms_radius"]).reshape(-1, 1, 2)
    native = cv2.ximgproc.fourierDescriptor(normalized, nbElt=32, nbFD=8).reshape(-1, 2)
    comp = compact.execute({"contour": square.tolist()})
    assert comp and comp.passed, compact.error
    reported = comp.outputs["descriptor"]
    assert reported["frequencies"] == [1, 2, 3, 4, -4, -3, -2, -1]
    np.testing.assert_allclose(reported["coefficients"], native, atol=1e-10)
    invalid = registry.create(FIT, parameters={"samples": 32, "harmonics": 8})
    assert invalid.execute({"sample": reported, "reference": descriptor}) is None
    assert "requires a full spectrum" in invalid.error


@pytest.mark.parametrize(("degrees", "scale"), ((23.0, 1.24), (-48.0, 0.72)))
def test_analytic_transform_with_reversed_and_shifted_traversal(registry, degrees, scale):
    reference = CONTOUR + [600_000, 800_000]
    center = reference.mean(axis=0)
    angle = math.radians(degrees)
    linear = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    moved = (reference - center) @ linear.T + center + [9, -17]
    reordered = np.roll(moved, 2, axis=0)[::-1]
    reordered = np.vstack((reordered, reordered[-1], reordered[0]))
    block = registry.create(FIT)
    result = block.execute({"reference": reference.tolist(), "sample": reordered.tolist()})
    assert result and result.passed, block.error
    info = result.outputs["report"]
    assert info["rms_pixels"] < 0.2
    assert info["scale"] == pytest.approx(1 / scale, abs=0.001)
    assert info["angle_degrees"] == pytest.approx(-degrees, abs=0.1)
    matrix = np.asarray(result.outputs["transform"]["source_to_target"])
    restored = moved @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_allclose(restored, reference, atol=0.2)


def test_invalid_contours_and_settings_produce_diagnostics(registry):
    block = registry.create(DESCRIBE)
    assert block.execute({"contour": [[0, 0], [1, 1], [2, 2]]}) is None
    assert "nonzero area" in block.error
    block = registry.create(DESCRIBE, parameters={"coefficients": 7})
    assert block.execute({"contour": CONTOUR.tolist()}) is None
    assert "must be even" in block.error
    block = registry.create(FIT, parameters={"samples": 16, "harmonics": 8})
    assert block.execute({"reference": CONTOUR.tolist(), "sample": CONTOUR.tolist()}) is None
    assert "less than half" in block.error


def test_reference_overlay_respects_frame_and_sample_ticket(registry):
    image = cv2.imread(str(EXAMPLE / "assets/reference.png"), 0)
    ref_context = {"spatial": frame([224, 204], "reference")}
    clock = TicketClock("contours")
    ref_context["ticket"] = clock.issue("camera", "fixed", "ref")
    sample_context = {"ticket": clock.issue("camera", "fixed", "sample")}
    block = registry.create(FIT)
    inputs = {
        "reference": ValuePacket(CONTOUR.tolist(), ref_context),
        "sample": ValuePacket(CONTOUR.tolist(), sample_context),
        "reference_image": ValuePacket(image, ref_context),
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["rms_pixels"] < 1e-5
    np.testing.assert_allclose(result.outputs["transform"]["source_to_target"], np.eye(3), atol=1e-5)
    assert result.payloads["aligned"]["spatial"] == ref_context["spatial"]
    assert result.payloads["image"]["ticket"] == sample_context["ticket"]
    assert result.drawings["contour alignment"].from_output
    inputs["reference_image"] = ValuePacket(image, {**ref_context, "ticket": sample_context["ticket"]})
    assert block.execute(inputs) is None
    assert "different tickets" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_success_warps_the_sample_and_failure_withholds_matrix(registry, profile):
    for filename, accepted in (("align_fourier_contours", True), ("reject_altered_contour", False)):
        graph = Graph.load(EXAMPLE / f"{filename}.json", registry)
        graph.set_run_profile(profile)
        graph.set_branch_workers(3)
        try:
            graph.run()
            block = graph.blocks["fit"]
            assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
            result = block.last_result
            assert result and result.passed == accepted, block.error
            row = result.outputs["report"]
            assert bool(result.drawings) == (profile == "debug")
            if accepted:
                assert row["rms_pixels"] == pytest.approx(1.33, abs=0.02)
                assert row["angle_degrees"] == pytest.approx(-23.62, abs=0.05)
                assert row["scale"] == pytest.approx(0.8078, abs=0.001)
                warp = graph.blocks["warp"].last_result.outputs["image"] > 127
                ref = result.outputs["image"] > 127
                assert (warp & ref).sum() / (warp | ref).sum() > 0.97
            else:
                assert row["rms_pixels"] == pytest.approx(6.08, abs=0.05)
                assert result.outputs["transform"] is NO_VALUE
                assert result.fail_reasons[0]["code"] == "fourier_contour_residual_above_limit"
                assert len(result.outputs["aligned"]) == 128
        finally:
            graph.close()
