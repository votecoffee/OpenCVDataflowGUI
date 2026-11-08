"""Check ordered gap lengths, closed seams, unavailable samples, and profile wiring."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def run_profile(values, *, closed=False, distances=None, length=None, **parameters):
    if distances is None:
        distances = list(range(len(values)))
    if length is None:
        length = distances[-1] + int(closed)
    tool = builtin_registry().create("inspect.path_continuity", parameters=parameters)
    return tool.execute({"profile": {
        "closed": closed, "path_length": length, "distances": distances,
        "points": [[d, 0] for d in distances], "channels": [{"channel": "Gray", "values": values}],
    }})


def test_continuity_distinguishes_scattered_samples_and_gaps_of_equal_count():
    good = run_profile([255, 0, 255, 0, 255, 0, 255], maximum_gap=1)
    bad = run_profile([255, 255, 0, 0, 0, 255, 255], maximum_gap=1)
    assert good.passed and not bad.passed
    assert good.outputs["longest_gap"] == 1 and good.outputs["gap_count"] == 3
    assert bad.outputs["longest_gap"] == 3 and bad.outputs["gap_count"] == 1
    assert good.score == bad.score
    assert bad.fail_reasons[0]["code"] == "path_gap_above_maximum"


def test_closed_path_merges_first_and_last_runs_without_merging_other_gaps():
    result = run_profile([0, 0, 255, 0, 255, 0, 0, 0], closed=True, maximum_gap=3)
    short, wrapped = result.outputs["gaps"]
    assert short["length"] == 1 and not short["wraps_seam"]
    assert wrapped == {
        "start_distance": 4.5, "end_distance": 1.5, "length": 5.0, "sample_count": 5,
        "wraps_seam": True, "first_point": [5, 0], "last_point": [1, 0],
    }
    assert result.outputs["longest_gap"] == 5 and not result.passed


@pytest.mark.parametrize("sample_count", [1, 3])
def test_closed_path_full_gap_has_exact_length(sample_count):
    result = run_profile([0] * sample_count, closed=True, length=6.5)
    gap, = result.outputs["gaps"]
    assert gap["start_distance"] == 0 and gap["end_distance"] == 6.5
    assert gap["length"] == 6.5 and not gap["wraps_seam"]


def test_open_path_clips_endpoint_cells_and_separates_unavailable_samples():
    result = run_profile(
        [0, None, 0, 255], distances=[0, 2, 3, 4], maximum_gap=1,
    )
    assert result.outputs["longest_gap"] == 1
    assert result.outputs["gap_count"] == 2
    assert result.outputs["sampled_fraction"] == 0.75
    assert [reason["code"] for reason in result.fail_reasons] == ["path_sampling_below_minimum"]
    assert [gap["start_distance"] for gap in result.outputs["gaps"]] == [0, 2.5]


def test_continuity_checks_selected_channel_and_retains_profile_frame_and_ticket():
    registry = builtin_registry()
    profile = registry.create("cv.path_intensity_profile", parameters={
        "vertices": [[0, 3], [9, 3]], "interpolation": "nearest",
    })
    bgr = np.zeros((8, 10, 3), np.uint8)
    bgr[:, :, 1] = 100
    bgr[:, 4:7, 1] = 210
    spatial = frame([10, 8], "part")
    ticket = TicketClock().issue("camera")
    result = profile.execute({"image": ValuePacket(bgr, {"spatial": spatial, "ticket": ticket})})
    tool = registry.create("inspect.path_continuity", parameters={
        "channel": "G", "range_mode": "between", "minimum_value": 100, "maximum_value": 200,
    })
    tool.exposed_outputs.append("result_graphics")
    result = tool.execute({
        "profile": ValuePacket(result.outputs["profile"], result.payloads["profile"]),
    })
    assert result is not None, tool.error
    assert result.outputs["longest_gap"] == 3
    assert result.payloads["longest_gap"]["ticket"] == ticket
    layer, = result.outputs["result_graphics"].layers
    assert layer.payload["spatial"] == spatial
    assert layer.objects[0].points == ((4, 3), (5, 3), (6, 3))


def test_continuity_rejects_nonmonotonic_or_nonfinite_profile_data():
    assert run_profile([0, 0, 1], distances=[0, 2, 2]) is None
    assert run_profile([0, float("nan"), 1]) is None
