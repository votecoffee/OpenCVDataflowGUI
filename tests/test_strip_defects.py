"""Check defect grouping, source-path lengths, filtering, and seam graphics."""

import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def profile(status, widths, **more):
    return {"center_row": 2, "units": "strip px", "status": status, "widths": widths, **more}


def group(**parameters):
    tool = builtin_registry().create("inspect.strip_defects", parameters=parameters)
    tool.exposed_outputs.append("result_graphics")
    return tool


def test_defect_runs_keep_different_failure_kinds_separate_and_filter_by_length():
    data = profile(
        ["pass", "out_of_range", "out_of_range", "out_of_range", "out_of_range", "missing",
         "missing", "pass", "unmeasurable", "unmeasurable"],
        [8, 4, 3, 13, 14, None, None, 8, None, None],
        width_limits=[5, 10],
    )
    tool = group(minimum_length=2)
    result = tool.execute({"profile": data})
    assert result is not None and result.passed, tool.error
    runs = result.outputs["defects"]
    assert result.outputs["defect_count"] == 4
    assert result.outputs["longest_defect"] == 2
    assert [(r["kind"], r["first_column"], r["last_column"]) for r in runs] == [
        ("narrow", 1, 2), ("wide", 3, 4), ("missing", 5, 6), ("unmeasurable", 8, 9),
    ]
    assert runs[0]["minimum_width"] == 3 and runs[0]["maximum_width"] == 4
    assert runs[-1]["minimum_width"] is None
    assert runs[-1]["units"] == "strip px"
    assert runs[-1]["start_distance"] == 8 and runs[-1]["end_distance"] == 10
    tool.parameters["minimum_length"] = 2.1
    assert tool.execute({"profile": data}).outputs["defects"] == []


def test_closed_strip_merges_one_defect_across_seam_but_draws_two_intervals():
    data = profile(
        ["out_of_range", "pass", "missing", "pass", "pass", "pass", "out_of_range", "out_of_range"],
        [3, 8, None, 8, 8, 8, 4, 2], width_limits=[5, None],
    )
    spatial = frame([8, 5], "closed strip")
    ticket = TicketClock().issue("camera")
    payload = {"spatial": spatial, "ticket": ticket}
    sampling = {"model": "path_strip_v1", "strip_spatial": spatial,
                "path_length": 16.0, "closed": True}
    result = group().execute({
        "profile": ValuePacket(data, payload), "sampling": ValuePacket(sampling, payload),
    })
    missing, narrow = result.outputs["defects"]
    assert missing["length"] == 2 and not missing["wraps_seam"]
    assert (narrow["first_column"], narrow["last_column"], narrow["column_count"]) == (6, 0, 3)
    assert narrow["length"] == 6 and narrow["wraps_seam"]
    assert narrow["start_distance"] == 11 and narrow["end_distance"] == 1
    assert narrow["units"] == "source px" and narrow["width_units"] == "strip px"
    assert result.payloads["longest_defect"]["ticket"] == ticket
    layer, = result.outputs["result_graphics"].layers
    assert layer.payload["spatial"] == spatial
    assert [obj.points for obj in layer.objects[:2]] == [((6, 2), (7, 2)), ((0, 2),)]


def test_open_sampling_clips_the_endpoint_cell_and_accepts_legacy_profiles():
    spatial = frame([6, 5], "open strip")
    data = profile(["out_of_range"] + ["pass"] * 5, [4] + [8] * 5)
    sampling = {"model": "path_strip_v1", "strip_spatial": spatial,
                "path_length": 15.0, "closed": False}
    result = group().execute({
        "profile": ValuePacket(data, {"spatial": spatial}), "sampling": sampling,
    })
    first, = result.outputs["defects"]
    assert first["kind"] == "out_of_range"
    assert first["start_distance"] == 0 and first["length"] == 1.5
    assert group().execute({"profile": profile(["missing"], [None])}).outputs["longest_defect"] == 1


def test_group_rejects_malformed_profile_or_foreign_sampling_grid_or_ticket():
    tool = group()
    assert tool.execute({"profile": profile(["pass"], [None])}) is None
    assert "one width and status" in tool.error
    spatial = frame([2, 5], "open strip")
    clock = TicketClock()
    data = ValuePacket(profile(["pass", "missing"], [8, None]),
                       {"spatial": spatial, "ticket": clock.issue("camera")})
    sampling = {"model": "path_strip_v1", "strip_spatial": frame([2, 5], "other strip"),
                "path_length": 5.0, "closed": False}
    assert tool.execute({"profile": data, "sampling": sampling}) is None
    assert "current strip grid" in tool.error
    sampling["strip_spatial"] = spatial
    assert tool.execute({
        "profile": data, "sampling": ValuePacket(sampling, {"ticket": clock.issue("camera")}),
    }) is None
    assert "share one ticket" in tool.error


@pytest.mark.parametrize("closed, length", [(True, 7.5), (False, 6.0)])
def test_whole_strip_missing_yields_one_run_with_the_full_path_length(closed, length):
    spatial = frame([4, 5], "strip")
    data = profile(["missing"] * 4, [None] * 4)
    sampling = {"model": "path_strip_v1", "strip_spatial": spatial,
                "path_length": length, "closed": closed}
    result = group().execute({
        "profile": ValuePacket(data, {"spatial": spatial}), "sampling": sampling,
    })
    run, = result.outputs["defects"]
    assert run["start_distance"] == 0 and run["end_distance"] == length
    assert run["length"] == length and not run["wraps_seam"]
