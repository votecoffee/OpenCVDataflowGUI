"""Independent mask pixels, overlap identity, NMS association and bounded failures."""

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
from opencv_dataflow.collection_tools import FilterCollection, SortCollection
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.instance_segmentation import (
    DecodeInstances,
    PackDetectionHeads,
    SelectInstanceMask,
)
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.multi_inference import read_tensor
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.selection import rasterize
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/instance_segmentation"
TRUTH = json.loads((BASE / "truth.json").read_bytes())

ROWS = np.array(
    [
        [5, 6, 8, 8, 0.9, 0.1],
        [5, 6, 8, 8, 0.8, 0.1],
        [8, 6, 8, 8, 0.1, 0.85],
        [1, 13, 6, 6, 0.75, 0.1],
    ],
    np.float32,
)


def arguments(*, mode="roi", layout="NCHW", letterbox=False):
    image = np.full((12, 16, 3), [30, 100, 180], np.uint8)
    _, geometry = preprocess(
        {"image": image},
        {
            **DNNInference.defaults,
            "width": 16,
            "height": 16,
            "resize_mode": "letterbox" if letterbox else "stretch",
        },
    )
    rows = ROWS.copy()
    if mode == "prototypes":
        masks = np.stack((np.tile(np.linspace(-1, 1, 8), (8, 1)), np.ones((8, 8)))).astype(
            np.float32
        )
        coeff = np.array([[1, 0.5], [-1, 0], [-1, 0.8], [0, 1]], np.float32)
        rows = np.concatenate((rows, coeff), axis=1)
        if layout in ("HWC", "NHWC"):
            masks = masks.transpose(1, 2, 0)
        if layout.startswith("N"):
            masks = masks[None]
    else:
        masks = np.full((4, 1 if mode == "roi" else 2, 8, 8), 0.1, np.float32)
        masks[0, 0, :, :6] = 0.9
        masks[0, 0, 3:5, 2:4] = 0.1
        masks[1] = 1
        masks[2, 0 if mode == "roi" else 1] = 0.8
        masks[2, 0 if mode == "roi" else 1, 2:5, 2:5] = 0.1
        masks[3, 0, :, 4:] = 0.9
        if layout == "NHWC":
            masks = masks.transpose(0, 2, 3, 1)
        elif layout == "NHW":
            masks = masks[:, 0]
    return {
        "image": image,
        "tensors": {
            "boxes": tensor_record(rows[None], 2_000_000),
            "masks": tensor_record(masks, 8_000_000),
        },
        "preprocessing": geometry,
    }


def decode(args=None, **parameters):
    block = DecodeInstances(
        parameters={"output_name": "boxes", "mask_interpolation": "nearest", **parameters}
    )
    result = block.execute(args or arguments())
    assert result is not None, block.error
    return block, result


def scalar_sample(plane, x, y, linear):
    if not linear:
        return float(plane[min(int(y), plane.shape[0] - 1), min(int(x), plane.shape[1] - 1)])
    # This CPU's float64 linear sampler uses 1/32 phases with replicated borders.
    x = round(float(np.float32(x - 0.5)) * 32) / 32
    y = round(float(np.float32(y - 0.5)) * 32) / 32
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    return sum(
        weight
        * float(plane[min(max(yy, 0), plane.shape[0] - 1), min(max(xx, 0), plane.shape[1] - 1)])
        for xx, yy, weight in (
            (ix, iy, (1 - fx) * (1 - fy)),
            (ix + 1, iy, fx * (1 - fy)),
            (ix, iy + 1, (1 - fx) * fy),
            (ix + 1, iy + 1, fx * fy),
        )
    )


def roi_reference(identity, args, *, channel=0, linear=False, threshold=0.5):
    raw = args["tensors"]["masks"]
    planes = np.array(raw["values"]).reshape(raw["shape"])
    if planes.ndim == 3:
        planes = planes[:, None]
    plane = planes[identity, channel]
    cx, cy, bw, bh = ROWS[identity, :4]
    left, top = float(cx - bw / 2), float(cy - bh / 2)
    forward = np.array(args["preprocessing"]["input_to_network"])
    result = np.zeros(args["image"].shape[:2], np.uint8)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            xx = (x + 0.5) * forward[0, 0] + forward[0, 2]
            yy = (y + 0.5) * forward[1, 1] + forward[1, 2]
            if left <= xx < left + bw and top <= yy < top + bh:
                value = scalar_sample(
                    plane,
                    (xx - left) * plane.shape[1] / bw,
                    (yy - top) * plane.shape[0] / bh,
                    linear,
                )
                result[y, x] = 255 if value >= threshold else 0
    return result


@pytest.mark.parametrize("letterbox", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_roi_masks_match_independent_source_pixels_and_clipping_never_stretches(letterbox, linear):
    args = arguments(letterbox=letterbox)
    block, result = decode(args, mask_interpolation="linear" if linear else "nearest")
    data = validate_collection(result.outputs["collection"])
    assert [i["id"] for i in data["items"]] == [0, 2, 3]
    expected = []
    for item in data["items"]:
        mask = rasterize(item["mask_region"], args["image"].shape)
        wanted = roi_reference(item["id"], args, linear=linear)
        np.testing.assert_array_equal(mask, wanted)
        assert item["metrics"]["mask_area"] == np.count_nonzero(wanted)
        assert item["label_id"] == item["id"] + 1
        expected.append(wanted != 0)
    np.testing.assert_array_equal(result.outputs["mask"], np.logical_or.reduce(expected) * 255)
    np.testing.assert_array_equal(result.outputs["overlap_count"], np.array(expected).sum(axis=0))
    assert np.any(result.outputs["overlap_count"] > 1)
    assert block.timing.count == 1
    assert data["instance_masks"]["overlap_preserved"]


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "NHW"])
def test_roi_layout_and_class_channel_selection_preserve_raw_row_identity(layout):
    args = arguments(layout=layout)
    _, result = decode(args, mask_layout=layout)
    _, reference = decode()
    for actual, wanted in zip(
        result.outputs["collection"]["items"], reference.outputs["collection"]["items"], strict=True
    ):
        assert actual["mask_region"] == wanted["mask_region"]
    args = arguments(mode="roi_per_class", layout="NHWC" if layout == "NHWC" else "NCHW")
    _, result = decode(
        args, mask_mode="roi_per_class", mask_layout="NHWC" if layout == "NHWC" else "NCHW"
    )
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 2, 3]
    item = result.outputs["collection"]["items"][1]
    assert item["class_id"] == 1 and item["metrics"]["mask_area"] > 0


@pytest.mark.parametrize("layout", ["CHW", "HWC", "NCHW", "NHWC"])
@pytest.mark.parametrize("channels_first", [False, True])
def test_prototype_mixing_and_crop_match_independent_coefficients_and_grid(layout, channels_first):
    args = arguments(mode="prototypes", layout=layout)
    record = args["tensors"]["boxes"]
    rows = np.array(record["values"]).reshape(record["shape"])[0]
    if channels_first:
        args["tensors"]["boxes"] = tensor_record(rows.T[None], 2_000_000)
    _, result = decode(
        args,
        mask_mode="prototypes",
        mask_layout=layout,
        mask_activation="sigmoid",
        layout="channels_first" if channels_first else "rows",
    )
    assert result.outputs["count"] == 3
    prototypes = arguments(mode="prototypes", layout="CHW")["tensors"]["masks"]
    basis = np.array(prototypes["values"]).reshape(prototypes["shape"])
    for item in result.outputs["collection"]["items"]:
        coeff = rows[item["id"], -2:]
        grid = sum(float(coeff[k]) * basis[k] for k in range(2))
        actual = rasterize(item["mask_region"], (12, 16))
        expected = np.zeros((12, 16), np.uint8)
        cx, cy, bw, bh = ROWS[item["id"], :4]
        for y in range(12):
            ny = (y + 0.5) * 16 / 12
            for x in range(16):
                nx = x + 0.5
                if cx - bw / 2 <= nx < cx + bw / 2 and cy - bh / 2 <= ny < cy + bh / 2:
                    expected[y, x] = 255 if grid[int(ny / 2), int(nx / 2)] >= 0 else 0
        np.testing.assert_array_equal(actual, expected)
    assert result.outputs["report"]["prototype_operations"] == 3 * 2 * 8 * 8


def test_prototype_crop_can_be_disabled_and_extreme_logits_stay_finite():
    args = arguments(mode="prototypes", layout="CHW")
    _, cropped = decode(args, mask_mode="prototypes", mask_layout="CHW", mask_activation="sigmoid")
    _, full = decode(
        args,
        mask_mode="prototypes",
        mask_layout="CHW",
        mask_activation="sigmoid",
        crop_to_box=False,
    )
    assert np.count_nonzero(full.outputs["mask"]) > np.count_nonzero(cropped.outputs["mask"])
    args["tensors"]["masks"]["values"] = [float(np.finfo(np.float32).max)] * 128
    _, extreme = decode(
        args,
        mask_mode="prototypes",
        mask_layout="CHW",
        mask_activation="sigmoid",
        activation_order="after_resampling",
        mask_interpolation="linear",
    )
    assert all(
        math.isfinite(i["metrics"]["mask_mean_confidence"])
        for i in extreme.outputs["collection"]["items"]
    )


@pytest.mark.parametrize("mode", ["pixels", "centers", "centers_and_pixels"])
def test_output_selection_policy_is_explicit_and_pixel_roi_does_not_reject_outside_centers(mode):
    args = arguments()
    args["region"] = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [16, 12],
        "rectangle": [1, 2, 2, 4],
    }
    _, result = decode(args, selection_mode=mode, minimum_count=0)
    if mode == "pixels":
        assert result.outputs["count"] > 0  # Box center is outside this small selection.
        assert not np.any(result.outputs["mask"][:, 3:])
    else:
        assert result.outputs["count"] == 0


@pytest.mark.parametrize(
    "overlap,expected_id", [("detection_score", 1), ("mask_confidence", 3), ("score_times_mask", 3)]
)
def test_overlap_summary_policy_never_erases_either_independent_mask(overlap, expected_id):
    args = arguments()
    record = args["tensors"]["masks"]
    values = np.array(record["values"]).reshape(record["shape"])
    values[0] = 0.6
    values[2] = 0.95
    args["tensors"]["masks"] = tensor_record(values, 8_000_000)
    _, result = decode(args, label_overlap=overlap)
    masks = [
        rasterize(i["mask_region"], (12, 16)) != 0
        for i in result.outputs["collection"]["items"][:2]
    ]
    common = masks[0] & masks[1]
    assert common.any()
    assert np.all(result.outputs["labels"][common] == expected_id)
    assert np.all(result.outputs["overlap_count"][common] == 2)


def test_mask_area_filters_precede_final_count_and_valid_empty_outputs_are_current():
    _, result = decode(minimum_mask_pixels=1000)
    assert not result.passed and result.fail_reasons[0]["code"] == "instance.count"
    assert result.outputs["count"] == 0 and not result.outputs["collection"]["items"]
    assert len(result.outputs["report"]["area_filtered"]) == 3
    assert not np.any(result.outputs["mask"])
    _, empty_allowed = decode(minimum_mask_pixels=1000, minimum_count=0)
    assert empty_allowed.passed
    args = arguments()
    args["tensors"]["masks"]["values"] = [0.0] * 256
    _, retained_empty = decode(args, minimum_mask_pixels=0)
    assert retained_empty.outputs["count"] == 3
    assert all(not i["mask_region"]["runs"] for i in retained_empty.outputs["collection"]["items"])


def test_sorted_filtered_collection_and_selected_mask_keep_geometry_including_overlap():
    args = arguments()
    _, result = decode(args)
    sorter = SortCollection(parameters={"metric": "custom", "custom_metric": "mask_area"})
    sorted_result = sorter.execute({"collection": result.outputs["collection"]})
    assert sorted_result is not None, sorter.error
    selector = SelectInstanceMask()
    chosen = selector.execute(
        {"image": args["image"], "collection": sorted_result.outputs["collection"]}
    )
    assert chosen is not None, selector.error
    wanted = sorted_result.outputs["collection"]["items"][0]
    assert chosen.outputs["item"]["id"] == wanted["id"]
    np.testing.assert_array_equal(
        chosen.outputs["mask"], rasterize(wanted["mask_region"], (12, 16))
    )
    filt = FilterCollection(
        parameters={"metric": "custom", "custom_metric": "mask_area", "minimum": 20}
    )
    filtered = filt.execute({"collection": result.outputs["collection"]})
    assert filtered is not None, filt.error
    assert filtered.outputs["collection"]["instance_masks"]["overlap_preserved"]


def test_selector_missing_policy_and_raster_budget_have_clear_failure_semantics():
    args = arguments()
    _, result = decode(args)
    selector = SelectInstanceMask(
        parameters={"selection": "id", "instance_id": 1}
    )  # Suppressed row.
    inputs = {"image": args["image"], "collection": result.outputs["collection"]}
    missing = selector.execute(inputs)
    assert missing is not None and not missing.passed
    assert missing.outputs["item"] is NO_VALUE and missing.outputs["found"] is False
    assert not np.any(missing.outputs["mask"])
    selector.parameters["missing"] = "error"
    assert selector.execute(inputs) is None and not selector.current_emissions
    selector.parameters.update(instance_id=0, missing="empty", maximum_region_runs=1)
    assert selector.execute(inputs) is None and "budget" in selector.error


@pytest.mark.parametrize(
    "params,field",
    [
        ({"mask_output_name": ""}, "mask_output_name"),
        ({"output_name": "masks"}, "output_name"),
        ({"mask_threshold": 1.1}, "mask_threshold"),
        ({"crop_to_box": 1}, "crop_to_box"),
        ({"mask_mode": "prototypes", "mask_layout": "NHW"}, "mask_layout"),
        ({"mask_mode": "prototypes", "format": "ssd"}, "format"),
        ({"mask_layout": "CHW"}, "mask_layout"),
        ({"mask_class_offset": 1}, "mask_class_offset"),
        ({"minimum_mask_pixels": 3, "maximum_mask_pixels": 2}, "maximum_mask_pixels"),
    ],
)
def test_invalid_settings_identify_fields_before_execution(params, field):
    block = DecodeInstances(parameters=params)
    with pytest.raises(ValidationError) as invalid:
        block.validate_parameters()
    assert field in invalid.value.fields
    assert not block.last_inputs and block.last_result is None


def test_nms_options_and_packed_objectness_do_not_change_mask_row_association():
    _, all_rows = decode(nms_enabled=False)
    assert [i["id"] for i in all_rows.outputs["collection"]["items"]] == [0, 2, 1, 3]
    _, agnostic = decode(class_agnostic_nms=True)
    assert [i["id"] for i in agnostic.outputs["collection"]["items"]] == [0, 3]
    args = arguments(mode="prototypes", layout="CHW")
    record = args["tensors"]["boxes"]
    rows = np.array(record["values"]).reshape(record["shape"])[0]
    rows = np.concatenate((rows[:, :4], np.ones((4, 1)), rows[:, 4:]), axis=1)
    args["tensors"]["boxes"] = tensor_record(rows[None], 2_000_000)
    _, result = decode(
        args,
        format="xywh_objectness",
        mask_mode="prototypes",
        mask_layout="CHW",
        mask_activation="sigmoid",
    )
    assert [i["id"] for i in result.outputs["collection"]["items"]] == [0, 2, 3]


def test_explicit_class_offsets_and_ssd_rows_choose_the_correct_roi_channel():
    args = arguments(mode="roi_per_class")
    rows = []
    for row in ROWS:
        cx, cy, w, h = row[:4]
        cls = int(np.argmax(row[4:]))
        rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(row[4 + cls]), cls])
    args["tensors"]["boxes"] = tensor_record(np.array(rows)[None], 2_000_000)
    _, shifted = decode(
        args,
        format="xyxy_score_class",
        class_id_offset=1,
        mask_mode="roi_per_class",
        mask_class_offset=-1,
    )
    assert [i["class_id"] for i in shifted.outputs["collection"]["items"]] == [1, 2, 1]
    ssd = [[0, r[5], r[4], *r[:4]] for r in rows]
    args["tensors"]["boxes"] = tensor_record(np.array(ssd)[None, None], 2_000_000)
    _, result = decode(args, format="ssd", mask_mode="roi_per_class")
    assert result.outputs["count"] == 3


@pytest.mark.parametrize(
    "limit,proto",
    [
        ("maximum_mask_elements", False),
        ("maximum_decoded_pixels", False),
        ("maximum_region_runs", False),
        ("maximum_results", False),
        ("maximum_prototype_operations", True),
    ],
)
def test_work_budgets_reject_atomically_without_partial_output(limit, proto):
    p = {limit: 1, "output_name": "boxes"}
    if proto:
        p.update(mask_mode="prototypes", mask_layout="CHW", mask_activation="sigmoid")
    block = DecodeInstances(parameters=p)
    assert (
        block.execute(arguments(mode="prototypes", layout="CHW") if proto else arguments()) is None
    )
    assert not block.current_emissions and block.error


@pytest.mark.parametrize(
    "mutation",
    [
        {"dtype": "uint8"},
        {"values": [False] * 256},
        {"values": [[0]] * 256},
        {"shape": [3, 1, 8, 8], "values": [0.5] * 192},
        {"values": [2.0] * 256},
        {"shape": [4, 1, 8, 9]},
    ],
)
def test_bad_mask_heads_do_not_emit_current_masks(mutation):
    args = arguments()
    args["tensors"]["masks"].update(mutation)
    block = DecodeInstances(parameters={"output_name": "boxes"})
    assert block.execute(args) is None and block.error
    assert not block.current_emissions


def test_activation_order_is_explicit_and_threshold_comparison_is_inclusive():
    args = arguments()
    rows = tensor_record(np.array([[[8, 8, 16, 16, 0.9]]]), 2_000_000)
    args["tensors"] = {"boxes": rows, "masks": tensor_record(np.array([[[[-1, 8]]]]), 8_000_000)}
    _, before = decode(
        args, mask_activation="sigmoid", mask_threshold=0.7, mask_interpolation="linear"
    )
    _, after = decode(
        args,
        mask_activation="sigmoid",
        mask_threshold=0.7,
        mask_interpolation="linear",
        activation_order="after_resampling",
    )
    assert np.count_nonzero(after.outputs["mask"]) > np.count_nonzero(before.outputs["mask"])
    args["tensors"]["masks"] = tensor_record(np.full((1, 1, 1, 1), 0.5), 8_000_000)
    _, threshold = decode(args)
    assert np.all(threshold.outputs["mask"] == 255)


def test_native_errors_are_contained_and_graphics_off_skips_optional_work(monkeypatch):
    block, first = decode(mask_interpolation="linear")
    assert first.drawings and first.graphics
    native = cv2.remap

    def fail(*args, **kwargs):
        raise cv2.error("controlled mask remap error")

    monkeypatch.setattr(cv2, "remap", fail)
    assert block.execute(arguments()) is None and "controlled mask remap" in block.error
    assert not block.current_emissions
    monkeypatch.setattr(cv2, "remap", native)
    assert block.execute(arguments()) is not None
    assert block.timing.count == 3
    block.run_profile = profile("lean")
    result = block.execute(arguments())
    assert result is not None and not result.drawings and not result.graphics


def test_graphics_limits_preserve_complete_masks_and_all_block_metadata():
    _, full = decode()
    _, limited = decode(graphics_maximum_points=1)
    assert limited.outputs["report"]["graphics_truncated"]
    np.testing.assert_array_equal(full.outputs["mask"], limited.outputs["mask"])
    for kind in ("ml.decode_instances", "ml.select_instance_mask"):
        block = builtin_registry().create(kind)
        block.validate_parameters()
        assert block.parameter_help.keys() == block.defaults.keys()
        assert block.preview_safe and block.parallel_safe and block.process_inputs_readonly


def context_args():
    args = arguments()
    args["image"] = ImageArray(args["image"].astype(np.uint16) * 257, "rgb")
    spatial = frame([16, 12], "camera")
    spatial.update(root_size=[32, 24], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    payload = {
        "ticket": TicketClock().issue("camera"),
        "spatial": spatial,
        "recipe_ref": {"id": "part", "revision": 1, "digest": "1" * 64},
        "pyramid": {
            "level": 1,
            "largest_size": [32, 24],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "processing_time": advance_timing(None, mark(), mark(), source=True),
    }
    return args, payload, {k: ValuePacket(v, copy.deepcopy(payload)) for k, v in args.items()}


def test_ticket_frame_recipe_depth_pyramid_and_start_timing_survive_decoder_and_selector():
    args, payload, wrapped = context_args()
    _, result = decode(wrapped)
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["image"], args["image"])
    for output in result.payloads.values():
        for key in ("ticket", "spatial", "recipe_ref", "pyramid"):
            assert output[key] == payload[key]
        assert output["processing_time"]["started_at"] == payload["processing_time"]["started_at"]
    selector = SelectInstanceMask(parameters={"selection": "id", "instance_id": 2})
    selected = selector.execute(
        {
            "image": wrapped["image"],
            "collection": ValuePacket(result.outputs["collection"], result.payloads["collection"]),
        }
    )
    assert selected is not None, selector.error
    assert selected.outputs["found"]
    assert selected.payloads["mask"]["ticket"] == payload["ticket"]


@pytest.mark.parametrize("conflict", ["ticket", "frame", "recipe"])
def test_known_context_conflicts_reject_and_valid_followup_recovers(conflict):
    args, payload, wrapped = context_args()
    bad = wrapped["tensors"].payload
    if conflict == "ticket":
        bad["ticket"] = TicketClock().issue("other")
    elif conflict == "frame":
        bad["spatial"]["root_id"] = "other"
    else:
        bad["recipe_ref"]["revision"] = 2
    block = DecodeInstances(parameters={"output_name": "boxes"})
    assert block.execute(wrapped) is None and not block.current_emissions
    result = block.execute(args)
    assert result is not None
    selector = SelectInstanceMask()
    assert (
        selector.execute(
            {
                "image": ValuePacket(args["image"], bad),
                "collection": ValuePacket(result.outputs["collection"], payload),
            }
        )
        is None
    )
    assert not selector.current_emissions


def separate_heads():
    args = arguments()
    rows = []
    for row in ROWS:
        cx, cy, w, h = row[:4]
        cls = int(np.argmax(row[4:]))
        rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(row[4 + cls]), cls])
    rows = np.array(rows)
    tensors = {
        "boxes": tensor_record(rows[:, :4], 2_000_000),
        "scores": tensor_record(rows[:, 4], 2_000_000),
        "labels": tensor_record(rows[:, 5, None], 2_000_000),
        "masks": args["tensors"]["masks"],
    }
    return args, tensors, rows


def test_separate_heads_pack_without_losing_masks_and_feed_ordinary_and_instance_decoders():
    args, tensors, rows = separate_heads()
    before = copy.deepcopy(tensors)
    ticket = TicketClock().issue("source")
    pack = PackDetectionHeads()
    result = pack.execute({"tensors": ValuePacket(tensors, {"ticket": ticket})})
    assert result is not None, pack.error
    assert tensors == before
    assert result.outputs["count"] == 4
    output = result.outputs["tensors"]
    np.testing.assert_allclose(np.array(output["packed_boxes"]["values"]).reshape(4, 6), rows)
    assert output["masks"] == tensors["masks"]
    assert result.payloads["tensors"]["ticket"] == ticket
    args["tensors"] = output
    _, decoded = decode(args, output_name="packed_boxes", format="xyxy_score_class")
    assert [i["id"] for i in decoded.outputs["collection"]["items"]] == [0, 2, 3]
    ordinary = builtin_registry().create(
        "ml.decode_detections",
        parameters={"output_name": "packed_boxes", "format": "xyxy_score_class"},
    )
    assert ordinary.execute(args).outputs["count"] == 3


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_name",
        "existing_output",
        "score_shape",
        "score_range",
        "class_fraction",
        "invalid_box",
        "total_budget",
        "row_budget",
    ],
)
def test_packing_rejects_invalid_names_shapes_values_and_budgets(problem):
    _, tensors, _ = separate_heads()
    p = {}
    if problem == "duplicate_name":
        p["scores_output"] = "boxes"
    elif problem == "existing_output":
        tensors["packed_boxes"] = copy.deepcopy(tensors["boxes"])
    elif problem == "score_shape":
        tensors["scores"]["shape"] = [2, 2]
    elif problem == "score_range":
        tensors["scores"]["values"][0] = 1.1
    elif problem == "class_fraction":
        tensors["labels"]["values"][0] = 0.5
    elif problem == "invalid_box":
        tensors["boxes"]["values"][2] = -10
    elif problem == "total_budget":
        p["maximum_total_elements"] = 300
    elif problem == "row_budget":
        p["maximum_rows"] = 1
    block = PackDetectionHeads(parameters=p)
    assert block.execute({"tensors": tensors}) is None and block.error
    assert not block.current_emissions


def expected_masks(reference):
    w, h = reference["image_size"]
    return {
        int(key): np.unpackbits(
            np.frombuffer(bytes.fromhex(value["packed_bits_hex"]), np.uint8), count=w * h
        ).reshape(h, w)
        != 0
        for key, value in reference["instances"].items()
    }


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_native_projects_preserve_each_mask_overlap_order_source_pixels_and_filmstrip(
    case, run_profile
):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        graph.run()
        result = graph.blocks["decode"].last_result
        assert result is not None, graph.blocks["decode"].error
        reference = TRUTH["cases"][case]
        masks = expected_masks(reference)
        assert result.passed is reference["passed"]
        assert result.outputs["count"] == reference["count"]
        items = result.outputs["collection"]["items"]
        assert {i["id"] for i in items} == set(masks)
        overlaps, labels = np.zeros((48, 80), np.int32), np.zeros((48, 80), np.int32)
        order = sorted(masks, key=lambda key: -reference["instances"][str(key)]["score"])
        assert [i["id"] for i in items] == order
        for item in items:
            expected = masks[item["id"]]
            np.testing.assert_array_equal(rasterize(item["mask_region"], (48, 80)) != 0, expected)
            assert item["metrics"]["mask_area"] == reference["instances"][str(item["id"])]["pixels"]
            overlaps += expected
            labels[expected & (labels == 0)] = item["id"] + 1
        np.testing.assert_array_equal(result.outputs["overlap_count"], overlaps)
        np.testing.assert_array_equal(result.outputs["labels"], labels)
        np.testing.assert_array_equal(result.outputs["mask"], (overlaps > 0) * 255)
        assert result.outputs["report"]["overlap_pixels"] == np.count_nonzero(overlaps > 1)
        source = graph.blocks["source"].last_result.outputs["image"]
        np.testing.assert_array_equal(result.outputs["image"], source)
        assert (
            result.payloads["collection"]["ticket"]
            == graph.blocks["source"].last_result.outputs["ticket"]
        )
        overlay = graph.blocks["overlay"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            overlay, graph.blocks["filmstrip"].last_result.outputs["image"]
        )
        if run_profile == "debug" and items:
            assert result.drawings and np.any(overlay != source)
        elif run_profile == "lean":
            assert not result.drawings and not result.graphics
        if case == "selected_roi":
            selected = graph.blocks["select"].last_result
            assert selected is not None and selected.outputs["found"]
            chosen = max(items, key=lambda item: item["metrics"]["mask_area"])
            assert selected.outputs["item"]["id"] == chosen["id"]
            np.testing.assert_array_equal(selected.outputs["mask"] != 0, masks[chosen["id"]])
            assert graph.blocks["blobs"].last_result.outputs["count"] > 0
        if case == "separate_heads":
            packed = graph.blocks["pack"].last_result
            assert packed is not None and packed.outputs["count"] == 4
    finally:
        graph.close()


def test_json_xml_save_as_asset_integrity_and_exact_example_regeneration(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "separate_heads.json", builtin_registry())
    try:
        graph.save(tmp_path / "saved.json")
        restored = Graph.load(tmp_path / "saved.json", builtin_registry())
        try:
            restored.run()
            assert restored.blocks["decode"].last_result.outputs["count"] == 3
        finally:
            restored.close()
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
    spec = importlib.util.spec_from_file_location("instance_examples", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "generated"
    module.HERE.mkdir()
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((module.HERE / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((module.HERE / path.name).read_bytes()) == json.loads(path.read_bytes())
    assert all(
        hashlib.sha256((BASE / path).read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )
    (module.HERE / "assets/instance_heads.onnx").write_bytes(b"existing bytes")
    with pytest.raises(FileExistsError):
        module.build()


@pytest.mark.parametrize("proto", [False, True])
def test_zero_row_heads_are_valid_empty_inspections_and_generic_readers_stay_strict(proto):
    args = arguments(mode="prototypes", layout="CHW") if proto else arguments()
    args["tensors"]["boxes"] = tensor_record(
        np.empty((1, 0, 8 if proto else 6), np.float32), 100, allow_empty=True
    )
    if not proto:
        args["tensors"]["masks"] = tensor_record(
            np.empty((0, 1, 8, 8), np.float32), 100, allow_empty=True
        )
    p = (
        {"mask_mode": "prototypes", "mask_layout": "CHW", "mask_activation": "sigmoid"}
        if proto
        else {}
    )
    _, result = decode(args, **p, **({} if proto else {"maximum_mask_elements": 1}))
    assert result.outputs["count"] == 0 and not result.passed
    assert result.fail_reasons[0]["code"] == "instance.count"
    assert not np.any(result.outputs["mask"])
    with pytest.raises(ValueError):
        tensor_record(np.empty((0, 4)), 100)
    with pytest.raises(ValidationError):
        read_tensor(args["tensors"]["boxes"], "", 100, "input")
    assert read_tensor(args["tensors"]["boxes"], "", 100, "output", allow_empty=True).shape == (
        1,
        0,
        8 if proto else 6,
    )


def test_zero_row_separate_heads_pack_and_axis_rotated_decoders_emit_current_empty_results():
    tensors = {
        name: tensor_record(np.empty(shape), 100, allow_empty=True)
        for name, shape in (
            ("boxes", (0, 4)),
            ("scores", (0,)),
            ("labels", (0,)),
            ("masks", (0, 1, 8, 8)),
        )
    }
    pack = PackDetectionHeads()
    packed = pack.execute({"tensors": tensors})
    assert packed is not None and packed.outputs["count"] == 0, pack.error
    args = arguments()
    args["tensors"] = packed.outputs["tensors"]
    _, result = decode(args, format="xyxy_score_class", output_name="packed_boxes")
    assert result.outputs["count"] == 0
    for kind, width in (("ml.decode_detections", 6), ("ml.decode_rotated_detections", 7)):
        block = builtin_registry().create(kind)
        args["tensors"] = {"boxes": tensor_record(np.empty((1, 0, width)), 100, allow_empty=True)}
        empty = block.execute(args)
        assert empty is not None and not empty.passed and empty.outputs["count"] == 0, block.error


def test_inference_output_serialization_preserves_zero_shape_and_detailed_receipt(monkeypatch):
    # Only native-array handoff is stubbed. No claim that this model/engine emits N=0.
    class EmptyOutputs:
        def setPreferableBackend(self, *args):
            pass

        def setPreferableTarget(self, *args):
            pass

        def enableFusion(self, *args):
            pass

        def setInput(self, *args):
            pass

        def getUnconnectedOutLayersNames(self):
            return ["boxes", "masks"]

        def forward(self, names):
            return [np.empty((1, 0, 6), np.float32), np.empty((0, 1, 8, 8), np.float32)]

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: EmptyOutputs())
    block = DNNInference(
        parameters={
            "model_path": str(BASE / "assets/instance_heads.onnx"),
            "width": 32,
            "height": 32,
            "engine": "classic",
            "receipt_mode": "detailed",
        }
    )
    result = block.execute({"image": arguments()["image"]})
    assert result is not None, block.error
    assert result.outputs["tensors"]["boxes"]["shape"] == [1, 0, 6]
    assert result.outputs["tensors"]["masks"]["values"] == []
    assert result.outputs["receipt"]["calls"]


def test_batched_zero_instances_unpack_with_image_and_ticket_ownership(monkeypatch):
    from opencv_dataflow.batch_inference import BatchImageInference, UnpackImageInference
    from opencv_dataflow.payloads import FlowGroup

    class EmptyBatch:
        def setPreferableBackend(self, *args):
            pass

        def setPreferableTarget(self, *args):
            pass

        def enableFusion(self, *args):
            pass

        def setInput(self, value, *args):
            self.n = value.shape[0]

        def getUnconnectedOutLayersNames(self):
            return ["boxes", "masks"]

        def forward(self, names):
            return [
                np.empty((self.n, 0, 6), np.float32),
                np.empty((self.n, 0, 1, 8, 8), np.float32),
            ]

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: EmptyBatch())
    image = arguments()["image"]
    clock = TicketClock()
    group = FlowGroup(
        tuple(ValuePacket(image.copy(), {"ticket": clock.issue("camera")}) for _ in range(2)),
        ("a", "b"),
    )
    batch = BatchImageInference(
        parameters={
            "model_path": str(BASE / "assets/instance_heads.onnx"),
            "width": 32,
            "height": 32,
            "engine": "classic",
        }
    )
    result = batch.execute({"images": group})
    assert result is not None, batch.error
    # Inspect the portable row through the normal unpacker, then the instance block.
    row = result.outputs["results"].items[0]
    unpack = UnpackImageInference()
    selected = unpack.execute({"result": row})
    assert selected is not None, unpack.error
    _, decoded = decode(
        {
            name: ValuePacket(selected.outputs[name], selected.payloads[name])
            for name in ("image", "tensors", "preprocessing")
        },
        output_name="boxes",
    )
    assert (
        decoded.outputs["count"] == 0
        and decoded.payloads["collection"]["ticket"] == group.items[0].payload["ticket"]
    )


def test_connected_missing_region_never_falls_back_to_an_unmasked_inspection(monkeypatch):
    graph = Graph.load(BASE / "selected_roi.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["decode"]
        count = block.timing.count
        previous = block.last_result
        assert previous is not None
        monkeypatch.setattr(
            graph.blocks["roi"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count
        assert not block.current_emissions
        assert block.last_result is None  # Run reset clears stale publications.
    finally:
        graph.close()
