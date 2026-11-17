"""Independent separate-head SSD contracts and shared decoder behavior."""

import copy

import numpy as np
import pytest

from opencv_dataflow.detection_decode import DecodeDetections
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def arguments(*, batched=True, count=None, order="xyxy", length=3):
    image = np.zeros((80, 160, 3), np.uint8)
    _, geometry = preprocess({"image": image}, {**DNNInference.defaults, "width": 64, "height": 64})
    boxes = np.array(
        [[0.1, 0.2, 0.4, 0.6], [0.11, 0.2, 0.41, 0.6], [0.6, 0.25, 0.8, 0.75]], np.float32
    )[:length]
    if order == "yxyx":
        boxes = boxes[:, [1, 0, 3, 2]]
    values = {
        "bboxes": boxes,
        "labels": np.array([1, 1, 2])[:length],
        "scores": np.array([0.9, 0.8, 0.7])[:length],
    }
    if batched:
        values = {key: value[None] for key, value in values.items()}
    if count is not None:
        values["count"] = np.array([count])
    return {
        "image": image,
        "preprocessing": geometry,
        "tensors": {
            key: tensor_record(value, 100, allow_empty=True) for key, value in values.items()
        },
    }


def block(**parameters):
    return DecodeDetections(
        parameters={
            "format": "ssd_heads",
            "coordinate_units": "normalized",
            "class_id_offset": -1,
            "labels": ["part", "fixture"],
            **parameters,
        }
    )


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("order", ["xyxy", "yxyx"])
def test_heads_map_to_source_and_share_class_aware_nms(batched, order):
    tool = block(box_coordinate_order=order)
    result = tool.execute(arguments(batched=batched, order=order))
    assert result is not None, tool.error
    assert result.outputs["count"] == 2
    items = result.outputs["collection"]["items"]
    assert [(item["id"], item["label"]) for item in items] == [(0, "part"), (2, "fixture")]
    np.testing.assert_allclose(items[0]["geometry"]["bbox"], [16, 16, 48, 32], atol=1e-5)
    assert (
        result.drawings and result.outputs["report"]["ssd_heads"]["box_coordinate_order"] == order
    )


@pytest.mark.parametrize("count,expected", [(0, 0), (1, 1), (2, 1), (3, 2)])
def test_explicit_count_excludes_padded_rows(count, expected):
    args = arguments(count=count)
    # Inactive padding may have invalid classes, scores and geometry.
    for name in ("labels", "scores"):
        args["tensors"][name]["values"][count:] = [-99] * (3 - count)
    tool = block(count_output_name="count", minimum_count=0)
    result = tool.execute(args)
    assert result is not None, tool.error
    assert result.outputs["count"] == expected
    assert result.outputs["report"]["raw_rows"] == count


def test_empty_dynamic_heads_are_a_valid_inspection_failure():
    tool = block()
    result = tool.execute(arguments(length=0))
    assert result is not None and not result.passed
    assert result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "detection.count"


@pytest.mark.parametrize(
    "head,shape",
    [
        ("bboxes", [3, 1, 4]),
        ("bboxes", [1, 1, 3, 4]),
        ("bboxes", [4, 3]),
        ("labels", [3, 1]),
        ("labels", [1, 1, 3]),
        ("scores", [3, 1]),
    ],
)
def test_ambiguous_or_multi_image_shapes_reject(head, shape):
    args = arguments()
    args["tensors"][head]["shape"] = shape
    tool = block()
    assert tool.execute(args) is None
    assert "SSD" in tool.error


@pytest.mark.parametrize("head", ["labels", "scores", "bboxes"])
def test_missing_and_mismatched_heads_reject(head):
    args = arguments()
    del args["tensors"][head]
    tool = block()
    assert tool.execute(args) is None
    assert "existing" in tool.error


@pytest.mark.parametrize("count", [-1, 4, 0.5])
def test_invalid_count_rejects_without_emission(count):
    tool = block(count_output_name="count")
    assert tool.execute(arguments(count=count)) is None
    assert "integer in 0..N" in tool.error and not tool.current_emissions


@pytest.mark.parametrize(
    "head,value",
    [
        ("labels", 1.5),
        ("labels", 99),
        ("labels", -1),
        ("scores", 1.1),
        ("scores", -0.1),
        ("bboxes", 2),
    ],
)
def test_scored_invalid_labels_probabilities_and_boxes_are_contained(head, value):
    args = arguments()
    args["tensors"][head]["values"][0] = value
    tool = block()
    assert tool.execute(args) is None and tool.state == "error"
    assert not tool.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"boxes_output_name": ""},
        {"scores_output_name": "labels"},
        {"count_output_name": "bboxes"},
        {"classes_output_name": 3},
        {"box_coordinate_order": "guess"},
        {"grid_decode": "yolox"},
    ],
)
def test_configuration_errors_are_field_bound(parameters):
    tool = block(**parameters)
    with pytest.raises(ValidationError):
        tool.validate_parameters()


def test_storage_row_budget_is_checked_before_count_truncation():
    tool = block(count_output_name="count", maximum_raw_rows=2)
    assert tool.execute(arguments(count=1)) is None
    assert "raw-row budget" in tool.error


def test_explicit_named_heads_ignore_dictionary_order_and_packed_name():
    args = arguments()
    args["tensors"] = dict(reversed(list(args["tensors"].items())))
    tool = block(output_name="not-used")
    assert tool.execute(args).outputs["count"] == 2


def test_failed_later_activation_cannot_emit_retained_detections():
    tool = block()
    assert tool.execute(arguments()) is not None
    args = arguments()
    args["tensors"]["scores"]["shape"] = [1, 2]
    assert tool.execute(args) is None
    assert not tool.current_emissions


def test_known_wrong_ticket_rejects_before_head_decoding():
    args = arguments()
    clock = TicketClock()
    context = {"ticket": clock.issue("one"), "spatial": frame((160, 80), root_id="camera")}
    args = {key: ValuePacket(value, copy.deepcopy(context)) for key, value in args.items()}
    args["tensors"].payload["ticket"] = clock.issue("two")
    tool = block()
    assert tool.execute(args) is None
    assert "different acquisition tickets" in tool.error
