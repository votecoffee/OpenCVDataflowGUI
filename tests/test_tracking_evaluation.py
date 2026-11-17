import copy

import pytest

from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.tracking_evaluation import evaluate_tracking


def record(identity, x=0):
    return {"id": identity, "bbox": [x, 0, 10, 10], "label": "part"}


def frame(expected, actual):
    return {"truth": expected, "predictions": actual}


def test_known_switch_fragment_false_positive_miss_and_mota_are_counted_independently():
    frames = [
        frame([record("A")], [record("1")]),
        frame([record("A")], []),
        frame([record("A")], [record("2"), record("extra", 50)]),
        frame([], []),
        frame([record("A")], [record("2")]),
    ]
    before = copy.deepcopy(frames)
    report = evaluate_tracking(frames)
    summary = report["summary"]
    assert (summary["tp"], summary["fp"], summary["fn"]) == (3, 1, 1)
    assert (summary["id_switches"], summary["fragmentations"]) == (1, 1)
    assert summary["mota"] == 0.25 and summary["precision"] == summary["recall"] == 0.75
    assert summary["mean_matched_iou"] == 1
    assert frames == before


def test_truth_absence_does_not_create_fragment_but_later_identity_change_counts_switch():
    report = evaluate_tracking(
        [frame([record("A")], [record("1")]), frame([], []), frame([record("A")], [record("2")])]
    )
    assert report["summary"]["id_switches"] == 1
    assert report["summary"]["fragmentations"] == 0
    assert evaluate_tracking([frame([], [])])["summary"]["mota"] is None


def test_global_assignment_maximizes_cardinality_and_class_mismatch_does_not_match():
    # A can match either; B can only match prediction 1 at threshold .3.
    report = evaluate_tracking(
        [frame([record("A", 4), record("B", 0)], [record("1", 2), record("2", 8)])],
        iou_threshold=0.3,
    )
    assert report["summary"]["tp"] == 2
    wrong = record("1")
    wrong["label"] = "wrong"
    assert evaluate_tracking([frame([record("A")], [wrong])])["summary"]["tp"] == 0


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [frame([record("A"), record("A")], [])],
        [frame([{"id": True, "bbox": [0, 0, 10, 10]}], [])],
        [frame([{"id": "x", "bbox": [0, 0, -1, 2]}], [])],
        [frame([], [{"id": "x", "bbox": [0, 0, float("nan"), 2]}])],
        [{"truth": []}],
    ],
)
def test_invalid_labels_boxes_and_frames_reject(frames):
    with pytest.raises(ValidationError):
        evaluate_tracking(frames)


def test_record_comparison_budgets_reject_before_assignment(monkeypatch):
    import opencv_dataflow.tracking_evaluation as module

    monkeypatch.setattr(module, "associate_costs", lambda *a: pytest.fail("assignment"))
    rows = [record(str(i), i) for i in range(128)]
    with pytest.raises(ValidationError, match="budget"):
        evaluate_tracking([frame(rows, rows)] * 123)
