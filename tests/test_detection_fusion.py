"""Independent tile geometry, native suppression, provenance and bounded failure checks."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.detection_fusion import FuseTileDetections
from opencv_dataflow.objects import collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, moved_payload
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock


def item(bounds, score=0.9, class_id=0, identity="part"):
    x, y, w, h = bounds
    return {
        "id": identity,
        "class_id": class_id,
        "label": "part" if class_id == 0 else "fixture",
        "score": score,
        "orientation_measured": False,
        "geometry": {"bbox": list(bounds), "unclipped_xyxy": [x, y, x + w, y + h]},
        "metrics": {"area": w * h},
    }


def arguments():
    image = np.zeros((80, 160, 3), np.uint8)
    context = {"spatial": frame([160, 80], "camera"), "ticket": TicketClock().issue("camera")}
    packets = []
    for x, bounds, score in ((0, [70, 20, 20, 20], 0.9), (60, [10, 20, 20, 20], 0.8)):
        payload = moved_payload(context, [[1, 0, x], [0, 1, 0], [0, 0, 1]], [100, 80])
        packets.append(
            ValuePacket(
                collection("boxes", [item(bounds, score)], np.zeros((80, 100)), payload), payload
            )
        )
    return {
        "image": ValuePacket(image, context),
        "detections": ValuePacket(
            FlowGroup(tuple(packets), ("left", "right")), {"ticket": context["ticket"]}
        ),
    }


def tiles(args):
    return args["detections"].value.items


def run(args=None, **parameters):
    block = FuseTileDetections(parameters=parameters)
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


def set_frame(packet, matrix, size):
    packet.payload["spatial"] = {**packet.payload["spatial"], "to_root": matrix, "image_size": size}
    packet.value["image_size"] = size
    packet.value["context"] = {"spatial": copy.deepcopy(packet.payload["spatial"])}


def test_duplicate_observations_become_one_source_box_with_original_winner_evidence():
    args = arguments()
    block, result = run(args, expected_tiles=2, include_candidate_details=True)
    assert result.passed and result.outputs["count"] == 1
    (found,) = result.outputs["collection"]["items"]
    assert found["id"] == 0 and found["provenance"]["tile_label"] == "left"
    assert found["provenance"]["source_detection"] == tiles(args)[0].value["items"][0]
    np.testing.assert_allclose(found["geometry"]["bbox"], [70, 20, 20, 20])
    assert found["metrics"]["area"] == 400 and found["metrics"]["angle"] is None
    assert result.outputs["report"]["filtered_counts"]["nms_or_top_k"] == 1
    assert [row["status"] for row in result.outputs["report"]["candidates"]] == [
        "retained",
        "nms_or_top_k",
    ]
    assert result.payloads["collection"]["spatial"] == args["image"].payload["spatial"]
    assert result.payloads["collection"]["ticket"] == args["image"].payload["ticket"]
    assert result.drawings and block.process_inputs_readonly
    assert "candidates" not in run()[1].outputs["report"]


@pytest.mark.parametrize("agnostic,top_k", [(False, 0), (True, 0), (False, 1), (True, 1)])
def test_class_aware_selection_matches_independent_native_batched_reference(agnostic, top_k):
    args = arguments()
    second = tiles(args)[1].value
    second["items"].extend(
        [item([10, 20, 20, 20], 0.85, 1, "fixture"), item([70, 5, 10, 10], 0.6, 0, "other")]
    )
    second["count"] = 3
    _, result = run(args, class_agnostic_nms=agnostic, nms_top_k=top_k)
    # Native top_k is per class in our class-aware contract.
    boxes = [[70, 20, 20, 20], [70, 20, 20, 20], [70, 20, 20, 20], [130, 5, 10, 10]]
    scores, classes = [0.9, 0.8, 0.85, 0.6], [0, 0, 1, 0]
    expected = []
    for group in [list(range(4))] if agnostic else [[0, 1, 3], [2]]:
        keep = cv2.dnn.NMSBoxes(
            [boxes[i] for i in group], [scores[i] for i in group], 0.25, 0.5, top_k=top_k
        )
        expected += [group[int(i)] for i in np.asarray(keep).ravel()]
    expected.sort(key=lambda i: -scores[i])
    found = result.outputs["collection"]["items"]
    assert [i["class_id"] for i in found] == [classes[i] for i in expected]
    np.testing.assert_allclose([i["geometry"]["bbox"] for i in found], [boxes[i] for i in expected])


@pytest.mark.parametrize(
    "matrix,size,box",
    [
        ([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], [50, 40], [35, 10, 10, 10]),
        ([[-1, 0, 99], [0, 1, 0], [0, 0, 1]], [100, 80], [10, 20, 20, 20]),
        ([[0, 1, 0], [-1, 0, 79], [0, 0, 1]], [80, 100], [40, 70, 20, 20]),
    ],
)
def test_pixel_center_maps_convert_to_pixel_edges_for_scale_flip_and_right_angle(matrix, size, box):
    args = arguments()
    first = tiles(args)[0]
    first.value["items"] = [item(box)]
    set_frame(first, matrix, size)
    _, result = run(args)
    assert result.outputs["count"] == 1
    np.testing.assert_allclose(
        result.outputs["collection"]["items"][0]["geometry"]["bbox"], [70, 20, 20, 20]
    )


def partial_arguments():
    args = arguments()
    first, second = tiles(args)
    first.value["items"] = [item([80, 20, 20, 20], 0.9)]
    first.value["items"][0]["geometry"]["unclipped_xyxy"] = [80, 20, 110, 40]
    second.value["items"] = [item([20, 20, 30, 20], 0.8)]
    return args


def test_internal_edge_filter_keeps_the_complete_observation_and_exempts_reference_outer_edges():
    args = partial_arguments()
    _, baseline = run(args)
    assert baseline.outputs["collection"]["items"][0]["metrics"]["width"] == 20
    _, filtered = run(args, edge_policy="drop_internal")
    assert filtered.outputs["count"] == 1
    assert filtered.outputs["collection"]["items"][0]["metrics"]["width"] == 30
    assert filtered.outputs["report"]["filtered_counts"]["internal_edge"] == 1
    first = tiles(args)[0]
    first.value["items"] = [item([0, 0, 10, 10])]
    args["detections"].value = FlowGroup((first,), ("outer",))
    assert run(args, edge_policy="drop_internal", edge_margin=1)[1].outputs["count"] == 1


def test_explicit_unclipped_source_uses_model_rectangles_without_claiming_reconstruction():
    _, result = run(partial_arguments(), box_source="unclipped")
    assert result.outputs["count"] == 1
    np.testing.assert_allclose(
        result.outputs["collection"]["items"][0]["geometry"]["bbox"], [80, 20, 30, 20]
    )
    assert result.outputs["report"]["partial_object_reconstruction"] is False


@pytest.mark.parametrize(
    "border,count,width", [("clip", 1, 10), ("keep", 1, 20), ("discard", 0, None)]
)
def test_reference_border_is_independent_of_unclipped_suppression(border, count, width):
    args = arguments()
    first = tiles(args)[0]
    first.value["items"] = [item([-10, 20, 20, 10])]
    args["detections"].value = FlowGroup((first,), ("one",))
    _, result = run(args, border_policy=border, minimum_count=0)
    assert result.outputs["count"] == count
    if count:
        assert result.outputs["collection"]["items"][0]["metrics"]["width"] == width


def test_reference_mask_and_region_select_mapped_box_centers_without_changing_image():
    args = arguments()
    args["mask"] = np.zeros((80, 160), np.uint8)
    _, result = run(args)
    assert result.outputs["count"] == 0 and not result.passed
    np.testing.assert_array_equal(result.outputs["image"], args["image"].value)
    args["mask"][:, 70:100] = 255
    assert run(args)[1].outputs["count"] == 1


@pytest.mark.parametrize("nms", [True, False])
def test_float32_native_score_threshold_boundary_is_consistent_with_nms_disabled(nms):
    args = arguments()
    for packet in tiles(args):
        packet.value["items"][0]["score"] = 0.25 + 1e-10
    assert run(args, nms_enabled=nms, minimum_count=0)[1].outputs["count"] == 0
    tiles(args)[0].value["items"][0]["score"] = float(np.nextafter(np.float32(0.25), np.float32(1)))
    assert run(args, nms_enabled=nms)[1].outputs["count"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        "ticket",
        "root",
        "root_size",
        "wire_frame",
        "unknown_frame",
        "recipe",
        "origin",
        "angle",
        "class",
        "bbox",
        "score",
        "labels",
    ],
)
def test_foreign_or_malformed_tiles_are_contained_without_current_results(bad):
    args = arguments()
    first, second = tiles(args)
    if bad == "ticket":
        second.payload["ticket"] = TicketClock().issue("other")
    elif bad in ("root", "root_size", "unknown_frame"):
        for target in (second.payload["spatial"], second.value["context"]["spatial"]):
            target[
                {"root": "root_id", "root_size": "root_size", "unknown_frame": "root_id"}[bad]
            ] = {"root": "other", "root_size": [200, 200], "unknown_frame": None}[bad]
    elif bad == "wire_frame":
        second.payload["spatial"]["to_root"][0][2] += 1
    elif bad == "recipe":
        first.payload["recipe_ref"] = {"id": "a", "revision": 1, "digest": "0" * 64}
        second.payload["recipe_ref"] = {"id": "b", "revision": 1, "digest": "1" * 64}
    elif bad == "origin":
        second.value["origin"] = [1, 0]
    elif bad == "angle":
        second.value["items"][0]["orientation_measured"] = True
    elif bad == "class":
        second.value["items"][0]["class_id"] = 1.5
    elif bad == "bbox":
        second.value["items"][0]["geometry"]["bbox"][2] = 0
    elif bad == "score":
        second.value["items"][0]["score"] = 1.1
    else:
        second.value["items"][0]["label"] = "conflicting label"
    block = FuseTileDetections()
    assert block.execute(args) is None
    assert block.state == "error" and not block.current_emissions


@pytest.mark.parametrize(
    "matrix", [[[1, 0.1, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0.01, 0, 1]]]
)
def test_relative_shear_and_perspective_are_rejected_without_axis_aligned_envelope_claim(matrix):
    args = arguments()
    set_frame(tiles(args)[0], matrix, [100, 80])
    block = FuseTileDetections()
    assert block.execute(args) is None
    assert "mapping" in block.error or "shear" in block.error


@pytest.mark.parametrize(
    "parameters",
    [
        {"expected_tiles": 3},
        {"maximum_raw_detections": 1},
        {"maximum_candidates": 1},
        {"maximum_results": 1, "nms_enabled": False},
        {"maximum_image_pixels": 100},
        {"edge_policy": "drop_internal", "edge_margin": 40},
    ],
)
def test_work_and_completeness_budgets_reject_without_truncation(parameters):
    block = FuseTileDetections(parameters=parameters)
    assert block.execute(arguments()) is None
    assert not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"nms_enabled": 1},
        {"expected_tiles": 1.5},
        {"accepted_class_ids": [True]},
        {"accepted_class_ids": [1, 1]},
        {"box_source": "guess"},
        {"edge_policy": "guess"},
        {"nms_iou": 2},
        {"minimum_count": 3, "maximum_count": 2},
    ],
)
def test_invalid_configuration_is_field_bound(parameters):
    with pytest.raises(ValidationError):
        FuseTileDetections(parameters=parameters).validate_parameters()


def test_failed_tile_and_count_are_distinct_inspection_failures_with_available_geometry():
    args = arguments()
    tiles(args)[1].payload["outcome"] = {
        "model": "tool_outcome_v1",
        "passed": False,
        "chain_passed": False,
        "source": {"id": "detector"},
    }
    _, result = run(args, minimum_count=2)
    assert not result.passed and result.outputs["count"] == 1
    assert {reason["code"] for reason in result.fail_reasons} == {
        "tile_detections.count",
        "tile_detections.failed_tile",
    }


def test_native_exception_then_valid_input_recovers_and_inputs_stay_unmodified(monkeypatch):
    args = arguments()
    before = copy.deepcopy(args)
    block = FuseTileDetections()
    native = cv2.dnn.NMSBoxes
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda *a, **k: (_ for _ in ()).throw(cv2.error("native fusion failure")),
    )
    assert block.execute(args) is None and not block.current_emissions
    monkeypatch.setattr(cv2.dnn, "NMSBoxes", native)
    assert block.execute(args).passed
    for old, new in zip(tiles(before), tiles(args), strict=True):
        assert old.value == new.value and old.payload == new.payload


def test_registration_choices_xml_and_lean_graphics_contract(tmp_path):
    registry = builtin_registry()
    block = registry.create("ml.fuse_tile_detections")
    assert "unclipped" in block.parameter_choices["box_source"]
    block.run_profile = PROFILES["lean"]
    result = block.execute(arguments())
    assert result.outputs["count"] == 1 and not result.drawings
    graph = Graph()
    graph.add(block)
    path = tmp_path / "program.xml"
    export_project_xml(graph.to_dict(), path)
    assert read_project_xml(path) == graph.to_dict()
    graph.close()
