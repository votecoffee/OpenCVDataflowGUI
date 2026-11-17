"""Independent map geometry, DB rectangle offset, gates and error recovery."""

import copy

import numpy as np
import pytest

from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.ocr import TextRegions
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock


def arguments(probability=None):
    image = ImageArray(np.full((40, 80, 3), 12000, np.uint16), "rgb")
    _, geometry = preprocess({"image": image}, {**DNNInference.defaults, "width": 16, "height": 8})
    if probability is None:
        probability = np.zeros((8, 16), np.float32)
        probability[2:5, 3:10] = 0.75
    context = {
        "spatial": frame([80, 40], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "label", "revision": 1, "digest": "2" * 64},
        "custom_context": {"part": "A"},
    }
    return {
        key: ValuePacket(value, copy.deepcopy(context))
        for key, value in {
            "image": image,
            "tensors": {"probability": tensor_record(probability, 10000)},
            "preprocessing": geometry,
        }.items()
    }


def run(args=None, **parameters):
    block = TextRegions(parameters={"padding_ratio": 1, **parameters})
    result = block.execute(arguments() if args is None else args)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("expansion", ["rectangle_scale", "db_unclip"])
def test_source_centers_and_db_rectangle_offset_match_independent_equations(expansion):
    args = arguments()
    _, result = run(args, expansion=expansion, unclip_ratio=2)
    (region,) = result.outputs["regions"]["items"]
    np.testing.assert_allclose(region["center"], [32, 17], atol=1e-5)
    # Head rectangle x3..9,y2..4; source pixel centers x=(head+.5)*5-.5.
    offset = (6 * 2 * 2) / (2 * (6 + 2)) if expansion == "db_unclip" else 0
    corners = np.array(
        [
            [3 - offset, 2 - offset],
            [9 + offset, 2 - offset],
            [9 + offset, 4 + offset],
            [3 - offset, 4 + offset],
        ]
    )
    np.testing.assert_allclose(region["corners"], (corners + 0.5) * 5 - 0.5, atol=1e-5)
    assert region["confidence"] == 0.75
    assert result.outputs["report"]["context_checks"] == {
        "frame_checked": True,
        "ticket_checked": True,
    }
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    for payload in result.payloads.values():
        for key in ("spatial", "ticket", "recipe_ref", "custom_context"):
            assert payload[key] == args["image"].payload[key]


def test_letterbox_padding_maps_back_to_source_edges_without_scale_offset():
    args = arguments()
    _, geometry = preprocess(
        {"image": args["image"].value},
        {**DNNInference.defaults, "width": 16, "height": 16, "resize_mode": "letterbox"},
    )
    values = np.zeros((16, 16), np.float32)
    values[6:9, 3:10] = 0.75
    args["tensors"].value = {"probability": tensor_record(values, 10000)}
    args["preprocessing"].value = geometry
    _, result = run(args)
    np.testing.assert_allclose(result.outputs["regions"]["items"][0]["center"], [32, 17], atol=1e-5)


def test_split_and_bridged_probability_maps_have_explicit_region_grouping():
    probability = np.zeros((8, 16), np.float32)
    probability[2:6, 2:6] = 0.75
    probability[2:6, 9:13] = 0.75
    _, split = run(arguments(probability), minimum_area=1)
    assert split.outputs["count"] == 2
    probability[3, 6:9] = 0.75
    _, merged = run(arguments(probability), minimum_area=1)
    assert merged.outputs["count"] == 1
    assert merged.outputs["report"]["raw_contours"] == 1


def test_threshold_boundaries_and_area_side_confidence_rejections():
    _, exact = run(pixel_threshold=0.75, minimum_count=1)
    assert exact.outputs["count"] == 0 and not exact.passed
    _, accepted = run(minimum_confidence=0.75, minimum_area=12, minimum_side=2, minimum_count=1)
    assert accepted.outputs["count"] == 1 and accepted.passed
    for key, value, rejected in (
        ("minimum_confidence", 0.8, "confidence"),
        ("minimum_area", 13, "area"),
        ("minimum_side", 3, "side"),
    ):
        _, result = run(**{key: value})
        assert result.outputs["count"] == 0
        assert result.outputs["report"]["rejected"][rejected] == 1


def test_roi_border_and_count_rules_return_current_failures():
    args = arguments()
    mask = np.zeros((40, 80), np.uint8)
    mask[10:25, 20:50] = 255
    args["mask"] = ValuePacket(mask, copy.deepcopy(args["image"].payload))
    _, good = run(args, minimum_count=1)
    assert good.passed
    args["mask"].value[:] = 0
    _, bad = run(args, minimum_count=1)
    assert not bad.passed and bad.outputs["count"] == 0
    assert bad.outputs["report"]["rejected"]["selection"] == 1
    probability = np.zeros((8, 16), np.float32)
    probability[1:4, :4] = 0.75
    _, keep = run(arguments(probability), expansion="db_unclip", unclip_ratio=4, minimum_area=1)
    assert keep.outputs["count"] == 1
    _, discard = run(
        arguments(probability),
        expansion="db_unclip",
        unclip_ratio=4,
        border_policy="discard",
        minimum_area=1,
    )
    assert discard.outputs["count"] == 0 and discard.outputs["report"]["rejected"]["border"] == 1


def test_truncation_is_reported_and_can_fail_independently_of_retained_count():
    probability = np.zeros((8, 16), np.float32)
    probability[1:4, 1:5] = 0.9
    probability[4:7, 10:14] = 0.75
    _, result = run(arguments(probability), minimum_area=1, maximum_regions=1, minimum_count=1)
    assert not result.passed and result.outputs["count"] == 1
    assert result.outputs["regions"]["items"][0]["confidence"] == pytest.approx(0.9)
    assert result.fail_reasons[0]["code"] == "text_regions.truncated"
    assert result.outputs["report"]["accepted_before_limit"] == 2
    _, allowed = run(
        arguments(probability), minimum_area=1, maximum_regions=1, fail_on_truncation=False
    )
    assert allowed.passed and allowed.outputs["report"]["truncated"]
    _, too_many = run(arguments(probability), minimum_area=1, maximum_count=1)
    assert not too_many.passed and too_many.outputs["count"] == 2


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_graphics_follow_profile_without_changing_results_or_inputs(run_profile):
    args = arguments()
    before = copy.deepcopy(args)
    block = TextRegions(parameters={"padding_ratio": 1})
    block.run_profile = PROFILES[run_profile]
    result = block.execute(args)
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert bool(result.drawings) == (run_profile == "debug")
    assert (result.outputs["overlay"] is NO_VALUE) == (run_profile == "lean")
    np.testing.assert_array_equal(args["image"].value, before["image"].value)
    assert args["tensors"].value == before["tensors"].value


@pytest.mark.parametrize(
    "change",
    [
        "root_size",
        "flat_values",
        "ticket",
        "frame",
        "recipe",
        "geometry",
        "map_budget",
        "image_budget",
        "candidates",
        "nan",
        "range",
        "channels",
        "mask_frame",
        "dtype",
    ],
)
def test_invalid_inputs_emit_nothing_and_a_valid_followup_recovers(change):
    args = arguments()
    parameters = {}
    if change == "root_size":
        args["tensors"].payload["spatial"]["root_size"] = [999, 40]
    elif change == "flat_values":
        args["tensors"].value["probability"]["values"][0] = True
    elif change == "ticket":
        args["tensors"].payload["ticket"] = TicketClock().issue("other")
    elif change == "frame":
        args["tensors"].payload["spatial"]["root_id"] = "other"
    elif change == "recipe":
        args["preprocessing"].payload["recipe_ref"]["revision"] = 2
    elif change == "geometry":
        args["preprocessing"].value["input_to_network"][0][0] = 4
    elif change == "map_budget":
        parameters["maximum_tensor_elements"] = 4
    elif change == "image_budget":
        parameters["maximum_image_pixels"] = 1
    elif change == "candidates":
        probability = np.zeros((8, 16), np.float32)
        probability[1:4, 1:5] = 0.9
        probability[4:7, 10:14] = 0.75
        args = arguments(probability)
        parameters["maximum_candidates"] = 1
    elif change in ("nan", "range"):
        args["tensors"].value["probability"]["values"][0] = float("nan") if change == "nan" else 2
    elif change == "channels":
        args["tensors"].value["probability"]["shape"] = [2, 4, 16]
    elif change == "mask_frame":
        args["mask"] = ValuePacket(
            np.ones((40, 80), np.uint8), copy.deepcopy(args["image"].payload)
        )
        args["mask"].payload["spatial"]["root_id"] = "other"
    else:
        args["tensors"].value["probability"]["dtype"] = "int8"
    block = TextRegions(parameters=parameters)
    assert block.execute(args) is None, change
    assert block.state == "error" and not block.current_emissions
    block.parameters = copy.deepcopy(block.defaults)
    assert block.execute(arguments()) is not None, block.error
