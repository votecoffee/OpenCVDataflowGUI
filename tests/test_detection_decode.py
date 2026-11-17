"""Independent box geometry, score/NMS semantics, source identity and failure contracts."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.detection_decode import DecodeDetections
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock


def arguments(rows=None, *, letterbox=False):
    image = np.full((80, 160, 3), 100, np.uint8)
    _, geometry = preprocess(
        {"image": image},
        {
            **DNNInference.defaults,
            "width": 64,
            "height": 64,
            "resize_mode": "letterbox" if letterbox else "stretch",
        },
    )
    if rows is None:
        rows = [
            [
                [20, 24, 20, 16, 0.9, 0.1],
                [21, 24, 20, 16, 0.8, 0.1],
                [20, 24, 20, 16, 0.1, 0.85],
                [46, 44, 12, 12, 0.7, 0.1],
            ]
        ]
    return {
        "image": image,
        "tensors": {"head": tensor_record(np.asarray(rows), 100000)},
        "preprocessing": geometry,
    }


def run(p=None, args=None):
    block = DecodeDetections(parameters=p)
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("agnostic,ids", [(False, [0, 2, 3]), (True, [0, 3])])
def test_class_aware_native_nms_stable_scores_and_sortable_box_collection(agnostic, ids):
    _, result = run({"labels": ["part", "fixture"], "class_agnostic_nms": agnostic})
    data = validate_collection(result.outputs["collection"])
    assert [item["id"] for item in data["items"]] == ids
    assert result.outputs["count"] == len(ids)
    assert data["items"][0]["label"] == "part"
    assert data["items"][0]["metrics"]["angle"] is None
    assert not data["items"][0]["orientation_measured"]
    np.testing.assert_allclose(data["items"][0]["geometry"]["bbox"], [25, 20, 50, 20])
    assert result.drawings["Object detections"].objects[1].text.startswith("part:")


def test_anisotropic_stretch_and_rounded_letterbox_edges_have_independent_references():
    args = arguments(letterbox=True)
    _, result = run(args=args)
    items = result.outputs["collection"]["items"]
    np.testing.assert_allclose(items[0]["geometry"]["bbox"], [25, 0, 50, 40])
    np.testing.assert_allclose(items[-1]["geometry"]["bbox"], [100, 55, 30, 25])
    assert items[-1]["geometry"]["clipped"]


@pytest.mark.parametrize(
    "policy,count,last_height", [("clip", 3, 25), ("discard", 2, 40), ("keep", 3, 30)]
)
def test_border_policy_is_separate_from_unclipped_nms(policy, count, last_height):
    _, result = run({"border_policy": policy}, arguments(letterbox=True))
    assert result.outputs["count"] == count
    assert result.outputs["collection"]["items"][-1]["metrics"]["height"] == last_height


@pytest.mark.parametrize("format", ["xywh_scores", "xywh_objectness", "xyxy_score_class", "ssd"])
@pytest.mark.parametrize("layout", ["rows", "channels_first"])
def test_explicit_row_formats_and_normalized_units_agree(format, layout):
    p = {"format": format, "layout": layout, "coordinate_units": "normalized"}
    if format.startswith("xywh"):
        row = [0.3125, 0.375, 0.3125, 0.25, 0.9, 0.1]
        if format == "xywh_objectness":
            row.insert(4, 1)
    elif format == "xyxy_score_class":
        row = [0.15625, 0.25, 0.46875, 0.5, 0.9, 1]
    else:
        row = [0, 1, 0.9, 0.15625, 0.25, 0.46875, 0.5]
    rows = np.asarray([row], np.float32)
    if layout == "channels_first" and format != "ssd":
        rows = rows.T
    _, result = run(p, arguments(rows[None]))
    item = result.outputs["collection"]["items"][0]
    np.testing.assert_allclose(item["geometry"]["bbox"], [25, 20, 50, 20])
    assert result.passed


def test_ssd_unused_rows_and_one_based_class_labels():
    args = arguments([[[[0, 1, 0.9, 0.15625, 0.25, 0.46875, 0.5], [-1] * 7]]])
    _, result = run(
        {
            "format": "ssd",
            "coordinate_units": "normalized",
            "class_id_offset": -1,
            "labels": ["part"],
        },
        args,
    )
    assert result.outputs["count"] == 1
    assert result.outputs["collection"]["items"][0]["label"] == "part"


def test_yolox_stride_expansion_matches_analytic_grid_and_sizes():
    rows = np.zeros((1, 84, 7), np.float32)  # 64x64: 8*8 + 4*4 + 2*2.
    rows[0, 9] = [0.5, 0.5, np.log(2), np.log(3), 0.8, 0.9, 0.1]
    _, result = run({"format": "xywh_objectness", "grid_decode": "yolox"}, arguments(rows))
    item = result.outputs["collection"]["items"][0]
    np.testing.assert_allclose(item["geometry"]["network_xyxy"], [4, 0, 20, 24], atol=1e-6)
    assert item["score"] == pytest.approx(0.72)


def test_explicit_sigmoid_and_objectness_combination():
    _, result = run(
        {"format": "xywh_objectness", "score_activation": "sigmoid"},
        arguments([[[20, 24, 20, 16, 0, np.log(3), -1000]]]),
    )
    assert result.outputs["collection"]["items"][0]["score"] == pytest.approx(0.375)


def test_selection_filter_and_empty_valid_inspection_failure_keep_source_pixels():
    args = arguments()
    before = args["image"].copy()
    mask = np.zeros(args["image"].shape[:2], np.uint8)
    mask[45:75, 90:150] = 255
    _, result = run(args={**args, "mask": mask})
    assert [item["id"] for item in result.outputs["collection"]["items"]] == [3]
    _, failed = run({"accepted_class_ids": [99]}, args)
    assert not failed.passed and failed.outputs["count"] == 0
    assert failed.fail_reasons[0]["code"] == "detection.count"
    np.testing.assert_array_equal(args["image"], before)


@pytest.mark.parametrize("limit", ["maximum_raw_rows", "maximum_candidates", "maximum_results"])
def test_resource_limits_reject_incomplete_results_instead_of_truncating(limit):
    block = DecodeDetections(parameters={limit: 1})
    assert block.execute(arguments()) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


@pytest.mark.parametrize(
    "parameters",
    [
        {"format": "guess"},
        {"nms_eta": 0},
        {"nms_iou": 1.1},
        {"nms_top_k": 1.0},
        {"class_agnostic_nms": 1},
        {"labels": [3]},
        {"accepted_class_ids": [True]},
        {"strides": [16, 8]},
        {"grid_decode": "yolox"},
        {"class_id_offset": 1},
        {"minimum_count": 4, "maximum_count": 2},
        {"confidence_threshold": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected_before_processing(parameters):
    block = DecodeDetections(parameters=parameters)
    with pytest.raises(ValidationError):
        block.validate_parameters()
    assert block.execute(arguments()) is None


@pytest.mark.parametrize("change", ["frame", "ticket", "recipe"])
def test_known_mismatched_inputs_are_rejected_without_new_outputs(change):
    args = arguments()
    clock = TicketClock()
    context = {
        "spatial": frame([160, 80], "camera"),
        "ticket": clock.issue("source"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "0" * 64},
    }
    contexts = [copy.deepcopy(context) for _ in range(3)]
    if change == "frame":
        contexts[1]["spatial"]["root_id"] = "different-camera"
    elif change == "ticket":
        contexts[1]["ticket"] = clock.issue("source")
    else:
        contexts[1]["recipe_ref"]["revision"] = 2
    inputs = {
        name: ValuePacket(args[name], c)
        for name, c in zip(("image", "tensors", "preprocessing"), contexts, strict=True)
    }
    block = DecodeDetections()
    assert block.execute(inputs) is None
    assert not any(block.current_emissions.values())


def test_tensor_and_geometry_validation_and_native_exception_recovery(monkeypatch):
    block = DecodeDetections()
    args = arguments()
    broken = copy.deepcopy(args)
    broken["preprocessing"]["network_to_input"][0][0] = 1
    assert block.execute(broken) is None
    native = cv2.dnn.NMSBoxes
    monkeypatch.setattr(
        cv2.dnn, "NMSBoxes", lambda *a, **k: (_ for _ in ()).throw(cv2.error("fixture NMS"))
    )
    assert block.execute(args) is None and "fixture NMS" in block.error
    monkeypatch.setattr(cv2.dnn, "NMSBoxes", native)
    assert block.execute(args).passed


def test_native_nms_controls_and_graphics_off_leave_detection_values_unchanged():
    _, ordinary = run()
    block = DecodeDetections(parameters={"nms_eta": 0.8, "nms_top_k": 1})
    block.set_graphics_mode("off")
    result = block.execute(arguments())
    assert result.outputs["count"] == 2 and not result.drawings
    _, all_rows = run({"nms_enabled": False})
    assert all_rows.outputs["count"] == 4
    assert ordinary.outputs["count"] == 3


def test_negative_coordinate_boxes_do_not_allow_cross_class_suppression():
    rows = [[[4, 4, 20, 20, 0.9, 0.1], [4, 4, 20, 20, 0.1, 0.85]]]
    _, result = run({"nms_iou": 0.01}, arguments(rows))
    assert result.outputs["count"] == 2


def test_score_boundary_and_equal_score_ties_are_explicit_and_stable():
    rows = [[[20, 24, 20, 16, 0.5], [20, 24, 20, 16, 0.5], [46, 44, 12, 12, 0.25]]]
    _, result = run({"confidence_threshold": 0.25}, arguments(rows))
    assert [item["id"] for item in result.outputs["collection"]["items"]] == [0]


def test_selected_center_uses_the_containing_pixel_cell_at_image_edges():
    args = arguments([[[63.9, 20, 0.08, 10, 0.9]]])
    mask = np.zeros((80, 160), np.uint8)
    mask[:, -1] = 255
    _, result = run(args={**args, "mask": mask})
    assert result.outputs["count"] == 1


def test_quantized_declared_tensor_cannot_be_interpreted_as_probabilities():
    args = arguments()
    args["tensors"]["head"]["dtype"] = "uint8"
    block = DecodeDetections()
    assert block.execute(args) is None
    assert block.error_kind == "invalid" and "dequantized" in block.error


def test_yolox_mismatched_large_grid_is_rejected_before_allocation(monkeypatch):
    args = arguments([[[0, 0, 1, 1, 1, 1]]])
    matrix = np.diag([4096 / 160, 4096 / 80, 1])
    args["preprocessing"].update(
        network_size=[4096, 4096],
        input_to_network=matrix.tolist(),
        network_to_input=np.linalg.inv(matrix).tolist(),
    )
    monkeypatch.setattr(
        np, "indices", lambda *a, **k: pytest.fail("Grid allocated before row check")
    )
    block = DecodeDetections(
        parameters={"format": "xywh_objectness", "grid_decode": "yolox", "strides": [1]}
    )
    assert block.execute(args) is None and "row count" in block.error


@pytest.mark.parametrize(
    "rows",
    [
        [[[20, 24, -1, 16, 0.9]]],
        [[[0, 0, 1, 1, 2]]],
        np.ones((2, 4, 6)),
    ],
)
def test_malformed_or_batched_predictions_emit_no_partial_results(rows):
    block = DecodeDetections()
    assert block.execute(arguments(rows)) is None
    assert not any(block.current_emissions.values())


def test_persistence_metadata_and_existing_collection_sorting(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create("ml.decode_detections"))
    path = tmp_path / "decode.xml"
    export_project_xml(graph.to_dict(), path)
    restored = Graph.from_dict(read_project_xml(path), registry)
    try:
        assert restored.blocks[block.id].parameters == block.parameters
        assert block.parameter_choices.keys() <= block.defaults.keys()
        assert block.parameter_ranges.keys() <= block.defaults.keys()
        assert block.parameter_help.keys() == block.defaults.keys()
        _, result = run()
        sort = registry.create(
            "collection.sort", parameters={"metric": "width", "direction": "descending"}
        )
        out = sort.execute({"collection": result.outputs["collection"]})
        assert out is not None, sort.error
        assert out.outputs["collection"]["items"][0]["metrics"]["width"] == 50
    finally:
        restored.close()
