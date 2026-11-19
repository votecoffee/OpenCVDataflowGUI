"""Qualify native grouping parity, stable member identity, and saved graphs."""

import copy
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.detection_votes import voted_members
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/opencv tools/spatial_vote_grouping"


def make(**parameters):
    return make_registry().create("ml.group_detection_votes", parameters=parameters)


def row(identity, box, cid=0, score=0.75, kind=None):
    return {
        "id": identity,
        "geometry": {"bbox": box},
        "metrics": {},
        "class_id": cid,
        "label": f"class {cid}",
        "score": score,
        "score_kind": kind,
    }


def collection(items):
    return {
        "model": "vision_collection_v1",
        "kind": "boxes",
        "image_size": [112, 80],
        "origin": [0, 0],
        "context": {},
        "count": len(items),
        "items": items,
    }


def test_native_keeps_representative_vote_counts_across_random_cluster_settings():
    rng = np.random.default_rng(1471)
    for iteration in range(60):
        bases = rng.integers(0, 40, (5, 2))
        boxes = [
            [int(x) + dx, int(y) + dy, 12 + dw, 12 + dh]
            for x, y in bases
            for dx, dy, dw, dh in rng.integers(0, 6, (8, 4)).tolist()
        ]
        eps = (0, 0.12, 0.2, 0.6)[iteration % 4]
        minimum = (2, 3, 5)[iteration % 3]
        native, weights = cv2.groupRectangles(boxes, minimum - 1, eps)
        actual = list(voted_members(boxes, eps, minimum, "drop"))
        assert Counter(
            (tuple(box), len(members)) for box, members, status in actual if status == "grouped"
        ) == Counter((tuple(box), int(weight)) for box, weight in zip(native, weights, strict=True))
        assert sorted(i for _, members, _ in actual for i in members) == list(range(len(boxes)))


def test_transitive_edge_tolerance_and_containment_keep_native_semantics():
    chain = [[x, 0, 10, 10] for x in (0, 2, 4)]
    assert list(voted_members(chain, 0.2, 2, "drop")) == [([2, 0, 10, 10], [0, 1, 2], "grouped")]
    assert all(
        status == "below_minimum_votes" for _, _, status in voted_members(chain, 0.19, 2, "drop")
    )
    contained = [[0, 0, 30, 30]] * 4 + [[5, 5, 10, 10]] * 2
    assert list(voted_members(contained, 0.2, 2, "drop")) == [
        ([0, 0, 30, 30], [0, 1, 2, 3], "grouped"),
        ([5, 5, 10, 10], [4, 5], "native_suppressed"),
    ]


def test_membership_and_output_ids_are_stable_after_reorder_and_are_class_aware():
    items = [
        row("left-b", [10, 11, 20, 20], score=0.9, kind="margin"),
        row(10, [11, 11, 20, 20], score=0.6, kind="margin"),
        row("right-a", [32, 10, 20, 20]),
        row("right-b", [33, 10, 20, 20]),
        row("fixture-a", [10, 11, 20, 20], cid=1),
        row("fixture-b", [10, 11, 20, 20], cid=1),
    ]
    block = make()
    result = block.execute({"collection": collection(items)})
    assert block.state == "pass", block.error
    found = result.outputs["collection"]["items"]
    assert [r["source_ids"] for r in found] == [
        [10, "left-b"],
        ["right-a", "right-b"],
        ["fixture-a", "fixture-b"],
    ]
    assert [(r["id"], r["class_id"]) for r in found] == [(10, 0), ("right-a", 0), ("fixture-a", 1)]
    assert found[0]["score"] == 0.9 and found[0]["score_kind"] == "margin"
    assert all("score" not in group and group["metrics"]["score"] is None for group in found[1:])
    assert found[0]["metrics"]["strength"] == 2
    reordered = block.execute({"collection": collection(items[::-1])})
    assert reordered.outputs["collection"] == result.outputs["collection"]
    assert reordered.outputs["report"] == result.outputs["report"]


def test_keep_single_only_does_not_override_vote_requirement_or_blend_scores():
    data = collection(
        [
            row("single", [60, 60, 10, 10]),
            row("low-a", [0, 0, 10, 10]),
            row("low-b", [0, 0, 10, 10]),
            row("group-a", [24, 24, 12, 12], kind="probability"),
            row("group-b", [24, 24, 12, 12], kind="margin"),
            row("group-c", [24, 24, 12, 12], kind="margin"),
        ]
    )
    block = make(minimum_votes=3, singleton_policy="keep")
    out = block.execute({"collection": data})
    items = out.outputs["collection"]["items"]
    assert [r["id"] for r in items] == ["group-a", "single"]
    assert "score" not in items[0] and items[0]["metrics"]["score"] is None
    assert items[1]["score"] == 0.75 and "score_kind" not in items[1]
    assert out.outputs["report"]["group_totals"] == {
        "below_minimum_votes": 1,
        "grouped": 1,
        "singleton_kept": 1,
    }


@pytest.mark.parametrize("rounding,extent", [("cover", 11), ("nearest", 10)])
def test_fractional_source_boxes_use_explicit_integer_edge_rule(rounding, extent):
    data = collection([row(0, [0.2, 0.2, 10, 10]), row(1, [0.4, 0.4, 10, 10])])
    old = copy.deepcopy(data)
    out = make(epsilon=0, rounding=rounding).execute({"collection": data})
    assert out.outputs["collection"]["items"][0]["geometry"]["bbox"] == [0, 0, extent, extent]
    assert data == old


def test_image_must_match_collection_ticket_and_frame():
    clock = TicketClock()
    a, b = clock.issue("camera"), clock.issue("camera")
    spatial = frame([112, 80], "view")
    data = collection([row(0, [0, 0, 10, 10]), row(1, [0, 0, 10, 10])])
    packet = ValuePacket(data, {"ticket": a, "spatial": spatial})
    image = np.zeros((80, 112), np.uint8)
    block = make()
    block.execute(
        {
            "collection": packet,
            "image": ValuePacket(
                image,
                {
                    "ticket": b,
                    "spatial": spatial,
                },
            ),
        }
    )
    assert block.state == "error" and "different acquisition tickets" in block.error
    block.execute(
        {
            "collection": packet,
            "image": ValuePacket(
                image,
                {
                    "ticket": a,
                    "spatial": frame([112, 80], "other view"),
                },
            ),
        }
    )
    assert block.state == "error" and "different image frames" in block.error
    assert not block.current_emissions


def test_input_limit_precedes_native_quadratic_work(monkeypatch):
    monkeypatch.setattr(cv2, "groupRectangles", lambda *a: pytest.fail("called native grouping"))
    block = make(maximum_candidates=1)
    block.execute({"collection": collection([row(0, [0, 0, 10, 10]), row(1, [0, 0, 10, 10])])})
    assert block.state == "error" and "candidate allowance" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "case",
    [
        "grouped",
        "keep_singletons",
        "exact_duplicates",
        "minimum_three",
        "data_only",
        "count_failure",
    ],
)
def test_saved_examples_qualify_grouping_and_overlay_without_changing_pixels(case, profile):
    truth = json.loads((EXAMPLES / "truth.json").read_bytes())["results"][case]
    with ProjectSession.open(EXAMPLES / f"{case}.json") as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        receipt = session.run()
        block = session.graph.blocks["vote"]
        assert block.state == truth["state"]
        assert receipt["passed"] == (block.state == "pass")
        result = block.last_result
        items = result.outputs["collection"]["items"]
        assert [r["source_ids"] for r in items] == truth["members"]
        assert [r["vote_count"] for r in items] == truth["vote_counts"]
        assert [r["geometry"]["bbox"] for r in items] == truth["bbox"]
        if case == "data_only":
            assert result.outputs["image"] is NO_VALUE
        else:
            np.testing.assert_array_equal(
                result.outputs["image"], session.graph.blocks["source"].last_result.outputs["image"]
            )
        assert bool(result.drawings) == (profile == "debug" and bool(items) and case != "data_only")
        if block.state == "fail":
            assert result.fail_reasons[0]["code"] == "detection_votes.count"


def test_native_border_rounding_is_visible_in_report_and_clipped_in_collection():
    data = collection([row(0, [107, 0, 4, 10]), row(1, [108, 0, 3, 10])])
    data["image_size"] = [111, 80]
    result = make().execute({"collection": data})
    item = result.outputs["collection"]["items"][0]
    assert item["geometry"]["bbox"] == [108, 0, 3, 10]
    assert item["geometry"]["unclipped_xyxy"] == [108, 0, 112, 10]
    assert item["geometry"]["clipped"]
    assert result.outputs["report"]["groups"][0]["native_bbox"] == [108, 0, 4, 10]
