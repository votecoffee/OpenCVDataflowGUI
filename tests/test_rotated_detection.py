"""Independent rotated geometry, nested IoU, native parity, poses and containment."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.poses import PoseFromFeature, local_offset
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.rotated_detection import DecodeRotatedDetections, rotated_iou
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/rotated_detection"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def inputs(rows=None, *, letterbox=False):
    image = np.full((96, 160, 3), [20, 100, 200], np.uint8)
    _, geometry = preprocess(
        {"image": image},
        {
            **DNNInference.defaults,
            "width": 64,
            "height": 64,
            "resize_mode": "letterbox" if letterbox else "stretch",
        },
    )
    rows = (
        rows
        if rows is not None
        else [
            [20, 28, 24, 10, 30, 0.9, 0],
            [20, 28, 6, 2, 30, 0.8, 0],
            [48, 44, 12, 6, -35, 0.7, 1],
        ]
    )
    return {
        "image": image,
        "tensors": {"head": tensor_record(np.array([rows], np.float32), 100000)},
        "preprocessing": geometry,
    }


def decode(args=None, **parameters):
    block = DecodeRotatedDetections(parameters={"format": "xywha_score_class", **parameters})
    result = block.execute(args or inputs())
    assert result is not None, block.error
    return block, result


def corners(x, y, w, h, angle):
    a = math.radians(angle)
    return np.array(
        [
            [x + dx * math.cos(a) - dy * math.sin(a), y + dx * math.sin(a) + dy * math.cos(a)]
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
        ]
    )


def test_nested_iou_default_differs_from_native_full_containment_shortcut():
    a, b = ((20.0, 28.0), (24.0, 10.0), 30.0), ((20.0, 28.0), (6.0, 2.0), 30.0)
    assert rotated_iou(a, b) == pytest.approx(0.05)
    _, result = decode()
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 1, 2]
    _, native = decode(nms_method="opencv")
    expected = cv2.dnn.NMSBoxesRotated([a, b], [0.9, 0.8], 0.25, 0.45).ravel().tolist()
    assert [i["id"] for i in native.outputs["collection"]["items"] if i["id"] < 2] == expected
    assert result.outputs["report"]["nms_grid"] == "unclipped_network_rectangles"


@pytest.mark.parametrize("letterbox", [False, True])
def test_exact_affine_corners_and_pose_preserve_anisotropy_without_fitted_rectangles(letterbox):
    args = inputs(letterbox=letterbox)
    _, result = decode(args)
    data = validate_collection(result.outputs["collection"])
    item = data["items"][0]
    original = corners(20, 28, 24, 10, 30)
    if letterbox:
        # 160x96 -> 64x38, rounded centered vertical padding 13.
        expected = (original - [0, 13]) * [2.5, 96 / 38]
    else:
        expected = original * [2.5, 1.5]
    np.testing.assert_allclose(item["geometry"]["unclipped_points"], expected, atol=1e-10)
    matrix = np.asarray(item["pose"]["source_to_target"])
    local = np.c_[item["geometry"]["local_corners"], np.ones(4)]
    np.testing.assert_allclose((local @ matrix.T)[:, :2], expected, atol=1e-10)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])) % 180
    assert item["metrics"]["angle"] == pytest.approx(angle)
    assert item["orientation_predicted"] and not item["orientation_measured"]
    assert item["metrics"]["major_size"] is None
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    assert result.outputs["matches"]["items"][0]["pose"] == item["pose"]


def test_supplied_collection_pose_survives_sort_and_feature_offset():
    args = inputs()
    _, result = decode(args)
    collection = result.outputs["collection"]
    pose = PoseFromFeature(
        parameters={
            "angle_mode": "feature",
            "pose_offset_x": 3,
            "pose_offset_y": -2,
            "pose_offset_angle": 7,
        }
    )
    answer = pose.execute({"image": args["image"], "features": collection})
    assert answer is not None, pose.error
    expected = np.asarray(collection["items"][0]["pose"]["source_to_target"]) @ local_offset(
        3, -2, 7
    )
    np.testing.assert_allclose(answer.outputs["pose"]["source_to_target"], expected)
    fixed = PoseFromFeature(parameters={"angle_mode": "fixed", "angle_degrees": 12})
    answer = fixed.execute({"image": args["image"], "features": collection})
    assert answer.outputs["angle_degrees"] == pytest.approx(12)


@pytest.mark.parametrize(
    "format", ["xywh_scores_angle", "xywha_scores", "xywha_objectness", "xywha_score_class"]
)
@pytest.mark.parametrize("layout", ["rows", "channels_first"])
@pytest.mark.parametrize("unit", ["degrees", "radians"])
def test_explicit_formats_layouts_units_and_class_offsets_agree(format, layout, unit):
    angle = math.radians(30) if unit == "radians" else 30
    dims = [20, 28, 24, 10]
    row = (
        dims + [0.9, 0.1, angle]
        if format == "xywh_scores_angle"
        else dims + [angle, 1, 0.9, 0.1]
        if format == "xywha_objectness"
        else dims + [angle, 0.9, 1]
        if format == "xywha_score_class"
        else dims + [angle, 0.9, 0.1]
    )
    args = inputs([row])
    if layout == "channels_first":
        args["tensors"]["head"] = tensor_record(np.asarray([row], np.float32).T[None], 1000)
    _, result = decode(
        args,
        format=format,
        layout=layout,
        angle_unit=unit,
        class_id_offset=-1 if format == "xywha_score_class" else 0,
        labels=["part", "fixture"],
    )
    item = result.outputs["collection"]["items"][0]
    np.testing.assert_allclose(
        item["geometry"]["unclipped_points"], corners(*dims, 30) * [2.5, 1.5], atol=2e-6
    )
    assert item["class_id"] == 0 and item["label"] == "part"


def test_normalized_xywh_decode_precedes_rotation_and_direction_period_are_explicit():
    args = inputs([[20 / 64, 28 / 64, 24 / 64, 10 / 64, -30, 0.9, 0]])
    _, result = decode(
        args, coordinate_units="normalized", angle_direction="counterclockwise", angle_period="360"
    )
    item = result.outputs["collection"]["items"][0]
    np.testing.assert_allclose(
        item["geometry"]["unclipped_points"], corners(20, 28, 24, 10, 30) * [2.5, 1.5], atol=1e-6
    )
    assert item["pose"]["orientation_period"] == 360


def test_class_aware_suppression_ties_disable_and_native_eta_top_k():
    rows = [
        [20, 28, 24, 10, 30, 0.9, 0],
        [20, 28, 24, 10, 30, 0.9, 1],
        [20, 28, 24, 10, 31, 0.8, 0],
        [48, 44, 12, 6, 70, 0.7, 0],
    ]
    _, result = decode(inputs(rows))
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 1, 3]
    _, result = decode(inputs(rows), class_agnostic_nms=True)
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 3]
    _, result = decode(inputs(rows), nms_enabled=False)
    assert result.outputs["count"] == 4
    _, result = decode(inputs(rows), nms_method="opencv", nms_eta=0.7, nms_top_k=2, nms_iou=0.7)
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 1]


@pytest.mark.parametrize("policy,expected_area", [("clip", 90.0), ("keep", 150.0)])
def test_polygon_clipping_does_not_move_model_center_or_pose(policy, expected_area):
    _, result = decode(inputs([[1, 28, 10, 4, 0, 0.9, 0]]), border_policy=policy)
    item = result.outputs["collection"]["items"][0]
    assert item["metrics"]["area"] == pytest.approx(expected_area)
    assert item["metrics"]["center_x"] == pytest.approx(2.5)
    assert item["pose"]["origin"][0] == pytest.approx(2.5)
    assert item["metrics"]["visible_fraction"] == pytest.approx(0.6)
    _, discarded = decode(inputs([[1, 28, 10, 4, 0, 0.9, 0]]), border_policy="discard")
    assert discarded.outputs["count"] == 0 and not discarded.passed


def test_mask_region_filter_class_and_strict_score_gate_are_on_original_center():
    args = inputs()
    args["mask"] = np.pad(np.full((96, 80), 255, np.uint8), ((0, 0), (80, 0)))
    _, result = decode(args)
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [2]
    args = inputs()
    args["region"] = {
        "model": "image_region_v1",
        "shape": "circle",
        "image_size": [160, 96],
        "center": [50, 42],
        "radius": 8,
    }
    _, result = decode(args)
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 1]
    _, result = decode(accepted_class_ids=[1])
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [2]
    _, empty = decode(inputs([[20, 28, 24, 10, 30, 0.5, 0]]), confidence_threshold=0.5)
    assert empty.outputs["count"] == 0 and empty.fail_reasons[0]["code"] == "detection.count"


def test_known_frame_and_ticket_conflicts_emit_nothing_and_recover():
    args = inputs()
    ticket = TicketClock().issue("camera")
    raw = copy.deepcopy(args)
    for name, value in args.items():
        args[name] = ValuePacket(value, {"ticket": ticket, "spatial": frame([160, 96], "camera")})
    block, result = decode(args)
    assert result.outputs["report"]["context_checks"] == {
        "ticket_checked": True,
        "frame_checked": True,
    }
    args["tensors"].payload["ticket"] = TicketClock().issue("other")
    assert (
        block.execute(args) is None
        and "tickets" in block.error
        and not any(block.current_emissions.values())
    )
    args["tensors"].payload["ticket"] = ticket
    args["preprocessing"].payload["spatial"]["root_id"] = "wrong"
    assert block.execute(args) is None and "frame" in block.error
    assert block.execute(raw).passed


def test_tagged_depth_recipe_pyramid_and_processing_start_survive_full_pose_rescaling():
    args = inputs()
    context = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([160, 96], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [320, 192],
            "to_largest": np.diag([2, 2, 1]).tolist(),
        },
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
        "processing_time": advance_timing(None, mark(), mark(), source=True),
    }
    context["spatial"].update(root_size=[320, 192], to_root=np.diag([2, 2, 1]).tolist())
    pixels = ImageArray(args["image"].astype(np.uint16) * 257, "rgb")
    args["image"] = pixels
    args = {k: ValuePacket(v, copy.deepcopy(context)) for k, v in args.items()}
    _, result = decode(args)
    assert (
        result.outputs["image"].dtype == np.uint16 and result.outputs["image"].color_format == "rgb"
    )
    np.testing.assert_array_equal(result.outputs["image"], pixels)
    for name in ("image", "collection", "matches", "report"):
        assert result.payloads[name]["pyramid"] == context["pyramid"]
        assert result.payloads[name]["recipe_ref"] == context["recipe_ref"]
        assert (
            result.payloads[name]["processing_time"]["started_ns"]
            == context["processing_time"]["started_ns"]
        )
    largest = ImageArray(
        np.zeros((192, 320, 3), np.uint16), "rgb", metadata={"spatial": frame([320, 192], "camera")}
    )
    pose = PoseFromFeature(parameters={"angle_mode": "feature"})
    answer = pose.execute({"image": largest, "features": result.outputs["collection"]})
    assert answer is not None, pose.error
    expected = np.diag([2, 2, 1]) @ np.asarray(
        result.outputs["collection"]["items"][0]["pose"]["source_to_target"]
    )
    np.testing.assert_allclose(answer.outputs["pose"]["source_to_target"], expected)


def test_native_exception_budgets_and_graphics_off_are_contained(monkeypatch):
    original = cv2.rotatedRectangleIntersection
    monkeypatch.setattr(
        cv2,
        "rotatedRectangleIntersection",
        lambda *a: (_ for _ in ()).throw(cv2.error("controlled native error")),
    )
    block = DecodeRotatedDetections(parameters={"format": "xywha_score_class"})
    assert block.execute(inputs()) is None and "controlled native" in block.error
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(cv2, "rotatedRectangleIntersection", original)
    assert block.execute(inputs()).passed
    for p in (
        {"maximum_candidates": 1},
        {"maximum_pair_comparisons": 1},
        {"maximum_results": 1},
        {"maximum_raw_rows": 1},
        {"maximum_image_pixels": 1},
    ):
        small = DecodeRotatedDetections(
            parameters={"format": "xywha_score_class", "class_agnostic_nms": True, **p}
        )
        assert small.execute(inputs()) is None and not any(small.current_emissions.values())
    block.graphics_mode = "off"
    assert not block.execute(inputs()).drawings


@pytest.mark.parametrize(
    "settings",
    [
        {"angle_unit": "guess"},
        {"nms_method": "guess"},
        {"maximum_pair_comparisons": True},
        {"nms_iou": 1.1},
        {"class_id_offset": 1},
        {"minimum_count": 3, "maximum_count": 1},
        {"labels": [""]},
    ],
)
def test_invalid_settings_validate_without_processing(settings):
    block = DecodeRotatedDetections(parameters=settings)
    with pytest.raises((ValidationError, ValueError, TypeError)):
        block.validate_parameters()
    assert not block.last_inputs


def test_native_precision_and_degenerate_scored_geometry_are_invalid():
    for row in (
        [20, 28, -4, 10, 30, 0.9, 0],
        [1e6, 28, 0.0001, 10, 30, 0.9, 0],
        [20, 28, 4, 10, 36001, 0.9, 0],
    ):
        block = DecodeRotatedDetections(parameters={"format": "xywha_score_class"})
        assert block.execute(inputs([row])) is None and not any(block.current_emissions.values())


def test_catalog_defaults_help_and_ports_remain_complete():
    block = builtin_registry().create("ml.decode_rotated_detections")
    block.validate_parameters()
    assert block.parameter_help.keys() == block.defaults.keys()
    assert "grid_decode" not in block.defaults and "strides" not in block.defaults


def test_crossing_thin_rectangles_survive_and_affine_source_polygon_iou_agrees():
    a, b = ((20.0, 28.0), (24.0, 4.0), 45.0), ((20.0, 28.0), (24.0, 4.0), 135.0)
    assert rotated_iou(a, b) == pytest.approx(16 / (96 + 96 - 16), abs=1e-6)
    args = inputs([[20, 28, 24, 4, 45, 0.9, 0], [20, 28, 24, 4, 135, 0.8, 0]])
    _, result = decode(args, nms_iou=0.3)
    assert result.outputs["count"] == 2
    pa = corners(20, 28, 24, 4, 45) * [2.5, 1.5]
    pb = corners(20, 28, 24, 4, 135) * [2.5, 1.5]
    intersection = cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))[0]
    area_a = cv2.contourArea(pa.astype(np.float32))
    area_b = cv2.contourArea(pb.astype(np.float32))
    assert rotated_iou(a, b) == pytest.approx(
        intersection / (area_a + area_b - intersection), abs=1e-6
    )


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_native_examples_geometry_counts_affine_roi_and_filmstrip_graphics(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        result = graph.blocks["decode"].last_result
        assert result is not None, graph.blocks["decode"].error
        expected = TRUTH["cases"][case]
        assert result.passed is expected["passed"]
        items = result.outputs["collection"]["items"]
        assert {i["id"] for i in items} == set(expected["ids"])
        assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)
        for item in items:
            wanted = np.array(expected["source_unclipped_points"][str(item["id"])])
            actual = np.array(item["geometry"]["unclipped_points"])
            # Unoriented axes permit the rectangle's cyclic corner origin to change.
            error = min(np.max(np.abs(actual - np.roll(wanted, i, axis=0))) for i in range(4))
            assert error < 3e-5
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        assert (
            result.payloads["collection"]["ticket"]
            == graph.blocks["source"].last_result.outputs["ticket"]
        )
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        film = graph.blocks["filmstrip"].last_result.outputs["image"]
        np.testing.assert_array_equal(overlay, film)
        if profile == "debug" and items:
            assert result.drawings and np.any(overlay != result.outputs["image"])
        if profile == "lean":
            assert not result.drawings
        if case == "pose_region":
            chosen = graph.blocks["sort"].last_result.outputs["collection"]["items"][0]
            pose = graph.blocks["pose"].last_result.outputs["pose"]
            np.testing.assert_allclose(pose["source_to_target"], chosen["pose"]["source_to_target"])
            assert np.count_nonzero(graph.blocks["roi"].last_result.outputs["mask"]) > 0
        assert not [b.error for b in graph.blocks.values() if b.error]
    finally:
        graph.close()


def test_project_json_xml_and_exact_asset_regeneration_preserve_original_collection(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "pose_region.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        restored = Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            restored.run()
            assert restored.blocks["decode"].last_result.passed
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("rotated_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "regenerated"
    module.HERE.mkdir()
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((module.HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((module.HERE / path.name).read_bytes()) == json.loads(path.read_bytes())
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
