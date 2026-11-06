"""Fit and validate batches with different identified points in each view."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, packet_for

_PROJECT = (
    Path(__file__).parents[1] / "examples/charuco_lens_calibration/charuco_lens_calibration.json"
)


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_batch_fits_only_when_all_views_arrive(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    fit = graph.blocks["fit"]
    for _ in range(7):
        graph.run()
        assert graph.blocks["observe"].state == "pass"
        assert fit.state == "inactive" and fit.last_result is None
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    outputs = fit.last_result.outputs
    calibration = outputs["calibration"]
    assert calibration["views"] == 8 and calibration["units"] == "m"
    assert calibration["image_size"] == [640, 480]
    np.testing.assert_allclose(
        calibration["camera_matrix"], [[720, 0, 319.5], [0, 726, 239.5], [0, 0, 1]], atol=5
    )
    assert calibration["distortion"][0] == pytest.approx(-0.08, abs=0.03)
    report = outputs["fit_report"]
    views = report["views"]
    assert [view["image_name"] for view in views] == [f"view-{i:02d}.png" for i in range(1, 9)]
    counts = [view["point_count"] for view in views]
    assert min(counts) < max(counts) == 24
    weighted = np.average([view["rms_pixels"] ** 2 for view in views], weights=counts)
    assert outputs["rms"] == pytest.approx(np.sqrt(weighted))
    assert outputs["rms"] < 0.25
    assert graph.blocks["maps"].last_result.outputs["map_x"].shape == (480, 640)


@pytest.fixture(scope="module")
def observations():
    detector = builtin_registry().create("calibration.charuco")
    packets = []
    for index in range(1, 4):
        image = cv2.imdecode(
            np.frombuffer(
                (_PROJECT.parent / "assets" / f"view-{index:02d}.png").read_bytes(), np.uint8
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        result = detector.execute({"image": image})
        assert result is not None and result.passed, detector.error
        packets.append(packet_for(result, "observation"))
    return tuple(packets)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("board", "same board"),
        ("size", "same image dimensions"),
        ("object_points", "invalid matched points"),
        ("duplicate", "duplicates"),
    ],
)
def test_inconsistent_view_lists_are_rejected(observations, problem, message):
    views = [copy.deepcopy(packet.value) for packet in observations]
    if problem == "board":
        views[2]["board"]["units"] = "mm"
    elif problem == "size":
        views[2]["image_size"][0] -= 1
    elif problem == "object_points":
        views[2]["object_points"][0][0] += 0.03
    else:
        views[2] = copy.deepcopy(views[0])
    fit = builtin_registry().create("calibration.lens")
    assert fit.execute({"observations": views}) is None
    assert message in fit.error


def test_group_rejects_views_with_a_different_pixel_transform(observations):
    entries = copy.deepcopy(observations)
    entries[2].payload["spatial"]["to_root"][0][2] = 3.0
    fit = builtin_registry().create("calibration.lens")
    group = FlowGroup(entries, ("first", "second", "third"))
    assert fit.execute({"observations": group}) is None
    assert "same camera pixel grid" in fit.error


def test_group_rejects_a_failed_detection_even_when_it_contains_enough_points(observations):
    entries = copy.deepcopy(observations)
    entries[2].payload["outcome"] = {
        "model": "tool_outcome_v1",
        "source": {"id": "detector"},
        "passed": False,
        "chain_passed": False,
    }
    fit = builtin_registry().create("calibration.lens")
    group = FlowGroup(entries, ("first", "second", "third"))
    assert fit.execute({"observations": group}) is None
    assert "must pass detection" in fit.error
