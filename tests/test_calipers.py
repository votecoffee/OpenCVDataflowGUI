"""Verify paired-edge dimensions, pose projection, and continuity checks."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.poses import local_offset, make_pose


@pytest.mark.parametrize("polarity", ["bright", "dark"])
def test_caliper_reports_width_and_gap_separately(polarity):
    image = np.zeros((100, 100), np.uint8)
    image[:, 40:60] = 200
    image[47:54] = 0
    if polarity == "dark":
        image = 200 - image
    block = builtin_registry().create(
        "inspect.bead_calipers",
        parameters={
            "polarity": polarity,
            "scan_count": 5,
            "scan_span": 40,
            "min_width": 19,
            "max_width": 21,
            "min_pass_fraction": 0.75,
            "maximum_gap": 0,
        },
    )
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs["mean_width"] == pytest.approx(20)
    assert result.outputs["pass_fraction"] == 0.8
    assert result.outputs["longest_gap"] == 1
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "bead_gap_above_maximum"


def test_pose_rotates_scan_and_selection_withholds_measurement():
    image = np.zeros((100, 100), np.uint8)
    image[40:60, :] = 200
    pose = make_pose(local_offset(50, 50, 90), [100, 100], {})
    block = builtin_registry().create(
        "inspect.bead_calipers",
        parameters={"start": [-30, 0], "end": [30, 0], "scan_count": 1, "draw_overlay": False},
    )
    result = block.execute({"image": image, "pose": pose})
    assert result is not None, block.error
    assert result.passed
    assert result.outputs["mean_width"] == pytest.approx(20)
    a, b = result.outputs["measurements"][0]["edges"]
    assert a[0] == b[0] == pytest.approx(50)
    assert a[1] > b[1]
    mask = np.full_like(image, 255)
    mask[50, 50] = 0
    clipped = block.execute({"image": image, "pose": pose, "mask": mask})
    assert not clipped.passed
    assert clipped.outputs["mean_width"] is NO_VALUE
    assert clipped.outputs["measurements"][0]["reason"] == "scan_outside_selection"


def test_caliper_project_measures_demo_rectangle():
    path = Path(__file__).resolve().parents[1] / "examples/bead_calipers/bead_calipers.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        result = graph.blocks["calipers"].last_result
        assert result is not None, graph.blocks["calipers"].error
        assert result.passed
        assert result.outputs["mean_width"] == pytest.approx(126, abs=0.5)
    finally:
        graph.close()
