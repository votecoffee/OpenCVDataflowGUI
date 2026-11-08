"""Verify complete-report judging and recipe overrides without losing measurements."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.recipes import reference
from opencv_dataflow.tickets import TicketClock


def group_report(status, widths, minimum_length=0):
    tool = builtin_registry().create(
        "inspect.strip_defects", parameters={"minimum_length": minimum_length}
    )
    return tool.execute({"profile": {
        "status": status, "widths": widths, "width_limits": [5, 10],
        "center_row": 2, "units": "strip px",
    }}).outputs


def judge(**parameters):
    return builtin_registry().create(
        "inspect.judge_strip_defects", parameters={"length_units": "strip px", **parameters}
    )


def test_judge_uses_complete_report_even_when_overlay_filter_hides_every_run():
    grouped = group_report(
        ["out_of_range", "pass", "out_of_range"], [3, 8, 3], minimum_length=2
    )
    assert grouped["defect_count"] == 0
    report = grouped["inspection_report"]
    tool = judge(limits={"narrow": {"max_count": 1, "max_length": 1}})
    result = tool.execute({"inspection_report": report})
    assert not result.passed
    assert result.outputs["summary"]["by_kind"]["narrow"] == {"count": 2, "longest_length": 1}
    assert [reason["code"] for reason in result.fail_reasons] == [
        "strip_defect_count_above_maximum"
    ]
    tool.parameters["limits"]["narrow"]["max_count"] = 2
    assert tool.execute({"inspection_report": report}).outputs["passed"] is True
    tool.parameters["limits"]["narrow"]["max_length"] = 0.75
    result = tool.execute({"inspection_report": report})
    assert [reason["code"] for reason in result.fail_reasons] == [
        "strip_defect_length_above_maximum"
    ]


def test_judge_requires_a_measurement_and_rejects_unlisted_kinds():
    unknown = group_report(["missing"] * 3, [None] * 3)["inspection_report"]
    result = judge(limits={"missing": {"max_count": 1}}).execute({"inspection_report": unknown})
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "strip_defect_no_measurable_width"
    mixed = group_report(["pass", "missing"], [8, None])["inspection_report"]
    result = judge().execute({"inspection_report": mixed})
    assert result.fail_reasons[0]["code"] == "strip_defect_count_above_maximum"


def test_judge_validates_report_units_and_missing_column_accounting():
    report = group_report(["pass", "missing"], [8, None])["inspection_report"]
    tool = judge()
    assert tool.execute({"inspection_report": {**report, "units": "source px"}}) is None
    assert "length units" in tool.error
    assert tool.execute({"inspection_report": {**report, "defects": []}}) is None
    assert "account for" in tool.error


@pytest.mark.parametrize("limits", [
    {"bad_kind": {"max_count": 2}},
    {"narrow": {}},
    {"narrow": {"max_count": -1}},
    {"wide": {"max_length": True}},
])
def test_judge_rejects_malformed_rules(limits):
    report = group_report(["pass"], [8])["inspection_report"]
    tool = judge(limits=limits)
    assert tool.execute({"inspection_report": report}) is None
    assert tool.error


def test_pinned_recipe_controls_final_verdict_for_diagnostic_width_measurements():
    registry = builtin_registry()
    graph = Graph()
    width = graph.add(registry.create("inspect.strip_width", parameters={
        "minimum_width": 4, "enforce_limits": False,
    }))
    group = graph.add(registry.create("inspect.strip_defects", parameters={"minimum_length": 2}))
    policy = graph.add(judge())
    final = graph.add(registry.create("inspect.final_judgment"))
    policy.recipe_bindings = {"limits": "settings.defect_limits"}
    group.exposed_outputs = ["inspection_report"]
    graph.connect(width.id, "profile", group.id, "profile")
    graph.connect(group.id, "inspection_report", policy.id, "inspection_report")
    graph.connect(policy.id, "summary", final.id, "value")
    strict = graph.recipes.put("strict", {"defect_limits": {}})
    lenient = graph.recipes.put("lenient", {
        "defect_limits": {"narrow": {"max_length": 1.0, "max_count": 1}},
    })
    graph.recipes.activate(strict, project_default=True)
    image = np.zeros((9, 6), np.uint8)
    image[2:7] = 255
    image[2:7, 2] = 0
    image[3:6, 2] = 255
    clock = TicketClock()
    try:
        for recipe, expected in [(lenient, True), (strict, False)]:
            payload = {
                "ticket": clock.issue("camera"), "spatial": frame([6, 9], "strip"),
                "recipe_ref": reference(recipe),
            }
            graph.run(injected={(width.id, "image"): ValuePacket(image, payload)})
            assert width.last_result.passed
            assert width.last_result.outputs["profile"]["status"][2] == "out_of_range"
            assert group.last_result.outputs["defect_count"] == 0
            assert policy.last_recipe_ref == reference(recipe)
            assert policy.parameters["limits"] == {}
            assert final.last_result.final_judgment is expected
            assert graph.ticket_results.find(payload["ticket"])["status"] == (
                "final_pass" if expected else "final_fail"
            )
    finally:
        graph.close()
