"""Observed mask continuity, bounded association, stream lifecycle and current provenance."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.instance_tracking import cropped, mask_iou
from opencv_dataflow.objects import collection
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.raster_regions import region_from_mask
from opencv_dataflow.selection import rasterize
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []

    def create(**parameters):
        block = builtin_registry().create(
            "sequence.track_instances",
            parameters={
                "identity_policy": "configured",
                "configured_model_revision": "unit-mask-model",
                "appearance": "off",
                **parameters,
            },
        )
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def inputs(positions, *, ids=None, colors=None, payload=None, timestamp=None):
    image = np.zeros((24, 40, 3), np.uint8)
    items = []
    for index, position in enumerate(positions):
        x, y = position
        mask = np.zeros((24, 40), np.uint8)
        mask[y : y + 6, x : x + 6] = 255
        image[mask != 0] = (colors or [(50, 50, 50)] * len(positions))[index]
        items.append(
            {
                "id": ids[index] if ids else index,
                "class_id": 0,
                "label": "part",
                "score": 0.9,
                "geometry": {"bbox": [x, y, 6, 6]},
                "metrics": {},
                "mask_region": region_from_mask(mask),
            }
        )
    data = collection("boxes", items, image, payload or {})
    data["instance_masks"] = {"model": "instance_masks_v1", "encoding": "image_region_v1"}
    result = {
        "image": ValuePacket(image, payload or {}),
        "instances": ValuePacket(data, payload or {}),
    }
    if timestamp is not None:
        result["timestamp_ms"] = timestamp
    return result


def test_independent_mask_iou_matches_analytic_intersection_and_budget():
    first = np.zeros((10, 10), bool)
    second = np.zeros_like(first)
    first[1:5, 1:5], second[3:7, 3:7] = True, True
    assert mask_iou(cropped(first), cropped(second), [100]) == pytest.approx(4 / 28)
    with pytest.raises(ValueError):
        mask_iou(cropped(first), cropped(second), [1])


def test_row_ids_do_not_define_track_identity_and_masks_are_current(make):
    block = make()
    first = block.execute(inputs([(2, 3), (25, 12)], ids=[99, 7]))
    assert [i["id"] for i in first.outputs["tracks"]["items"]] == [1, 2]
    second = block.execute(inputs([(23, 12), (4, 3)], ids=[100, 400]))
    assert [i["source_instance_id"] for i in second.outputs["tracks"]["items"]] == [400, 100]
    for item in second.outputs["tracks"]["items"]:
        mask = rasterize(item["mask_region"], (24, 40))
        assert mask.sum() == 36 * 255 and item["observed"]
        assert np.all(second.outputs["labels"][mask != 0] == item["id"])


def test_missing_masks_have_report_predictions_only_then_reacquire_or_expire(make):
    block = make(maximum_missed_frames=2)
    block.execute(inputs([(2, 3)]))
    block.execute(inputs([(4, 3)]))
    missing = block.execute(inputs([]))
    assert missing.outputs["count"] == 0 and not missing.outputs["tracks"]["items"]
    assert np.all(missing.outputs["labels"] == -1)
    assert missing.outputs["report"]["active"][0]["status"] == "predicted"
    assert missing.outputs["report"]["active"][0]["center"][0] == pytest.approx(8.5)
    result = block.execute(inputs([(8, 3)], ids=[55]))
    assert result.outputs["tracks"]["items"][0]["id"] == 1
    for _ in range(3):
        last = block.execute(inputs([]))
    assert last.outputs["report"]["expired_ids"] == [1]
    assert block.execute(inputs([(14, 3)])).outputs["tracks"]["items"][0]["id"] == 2


def test_overlapping_instance_masks_remain_independent(make):
    result = make().execute(inputs([(2, 3), (5, 3)]))
    assert result.outputs["report"]["overlap_pixels"] == 18
    assert np.count_nonzero(result.outputs["overlap_count"] == 2) == 18
    masks = [rasterize(i["mask_region"], (24, 40)) != 0 for i in result.outputs["tracks"]["items"]]
    assert np.count_nonzero(masks[0] & masks[1]) == 18


def test_masked_appearance_prevents_identity_swap_when_paths_cross(make):
    block = make(appearance="hsv_hs", maximum_appearance_distance=0.2)
    red, blue = (0, 0, 220), (220, 0, 0)
    block.execute(inputs([(2, 3), (26, 12)], colors=[red, blue]))
    block.execute(inputs([(8, 3), (20, 12)], colors=[red, blue]))
    result = block.execute(inputs([(14, 12), (14, 3)], ids=[90, 8], colors=[blue, red]))
    assert [i["source_instance_id"] for i in result.outputs["tracks"]["items"]] == [8, 90]
    assert all(
        a["appearance_distance"] == pytest.approx(0)
        for a in result.outputs["report"]["associations"]
    )


def test_ambiguous_matches_are_withheld_without_duplicate_births(make):
    block = make(maximum_center_distance=50, mask_weight=0, minimum_cost_margin=0.1)
    block.execute(inputs([(2, 3), (12, 3)]))
    result = block.execute(inputs([(7, 3)], ids=[50]))
    assert result.outputs["count"] == 0
    assert result.outputs["report"]["ambiguous_instance_ids"] == [50]
    assert len(result.outputs["report"]["active"]) == 2


def test_warmup_confirmation_reset_and_copy_are_independent(make):
    block = make(warmup=2, minimum_observations=2)
    first = block.execute(inputs([(2, 3)]))
    assert not first.outputs["ready"] and first.outputs["tracks"] is NO_VALUE
    result = block.execute(inputs([(3, 3)]))
    assert result.outputs["ready"] and result.outputs["count"] == 1
    other = make(warmup=2, minimum_observations=2)
    other.copy_runtime_from(block)
    other.execute(inputs([(4, 3)]))
    assert block.snapshot_runtime()["streams"] != {} and other.memory is not block.memory
    assert next(iter(block.memory["streams"].values()))["activations"] == 2
    data = inputs([(5, 3)])
    data["reset"] = True
    result = block.execute(data)
    assert result.outputs["activations"] == 1 and result.outputs["tracks"] is NO_VALUE


def test_explicit_clock_stale_timestamp_recovers_with_a_new_generation(make):
    block = make(clock="explicit")
    first = block.execute(inputs([(2, 3)], timestamp=100))
    generation = first.outputs["status"]["generation"]
    assert block.execute(inputs([(3, 3)], timestamp=100)) is None
    assert not block.current_emissions and not block.memory["streams"]
    recovered = block.execute(inputs([(3, 3)], timestamp=200))
    assert recovered.outputs["status"]["generation"] != generation


def test_known_ticket_and_frame_conflicts_are_rejected_before_association(make):
    clock = TicketClock()
    payload = {"ticket": clock.issue("camera"), "spatial": frame([40, 24], root_id="camera")}
    data = inputs([(2, 3)], payload=payload)
    data["instances"].payload = {**payload, "ticket": clock.issue("other")}
    block = make()
    assert block.execute(data) is None and "ticket" in block.error
    data = inputs([(2, 3)], payload=payload)
    data["instances"].payload = {**payload, "spatial": frame([40, 24], root_id="other")}
    assert block.execute(data) is None


def test_model_receipt_preprocessing_change_resets_context(make):
    block = make(identity_policy="receipt")
    data = inputs([(2, 3)])
    receipt = {
        "model": "onnx_inference_receipt_v1",
        "schema_version": 1,
        "calls": [
            {
                "artifact": {"sha256": "a" * 64},
                "contract_revision": "b" * 64,
                "preprocessing": {"image": {"scale": 1 / 255}},
            }
        ],
    }
    data["receipt"] = receipt
    first = block.execute(data)
    assert first is not None
    receipt["calls"][0]["preprocessing"]["image"]["scale"] = 1
    second = block.execute(data)
    assert second.outputs["status"]["generation"] != first.outputs["status"]["generation"]
    data.pop("receipt")
    assert block.execute(data) is None


def test_selection_and_graphics_off_keep_original_high_depth_pixels(make):
    block = make(appearance="gray")
    data = inputs([(2, 3), (25, 12)])
    data["image"].value = data["image"].value.astype(np.uint16) * 257
    mask = np.zeros((24, 40), np.uint8)
    mask[:10] = 255
    data["mask"] = mask
    block.set_graphics_mode("off")
    result = block.execute(data)
    assert result.outputs["count"] == 1 and result.outputs["image"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["image"], data["image"].value)
    assert not result.graphics and not result.drawings


@pytest.mark.parametrize(
    "parameters",
    [
        {"maximum_instances": 1},
        {"maximum_decoded_pixels": 100},
        {"maximum_tracks": 1},
        {"maximum_region_runs": 2},
        {"max_bytes_mb": 0.0001},
    ],
)
def test_memory_geometry_and_capacity_limits_fail_without_partial_masks(make, parameters):
    block = make(**parameters)
    assert block.execute(inputs([(2, 3), (25, 12)])) is None
    assert not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"maximum_missed_frames": True},
        {"maximum_tracks": 2.5},
        {"minimum_interval_ms": 10, "maximum_interval_ms": 1},
        {"identity_policy": "configured", "configured_model_revision": ""},
        {"hue_bins": 32, "saturation_bins": 32},
    ],
)
def test_parameter_validation_is_exposed(make, parameters):
    with pytest.raises(ValueError):
        make(**parameters).validate_parameters()


def test_current_box_measurements_follow_selected_mask_and_preserve_source_metrics(make):
    data = inputs([(2, 3)])
    item = data["instances"].value["items"][0]
    item["geometry"]["bbox"] = [0, 0, 40, 24]
    item["metrics"] = {
        "area": 960,
        "aspect_ratio": 40 / 24,
        "mask_coverage": 36 / 960,
        "mask_mean_confidence": 0.88,
        "custom_quality": 0.7,
    }
    block = make()
    result = block.execute(data)
    current = result.outputs["tracks"]["items"][0]
    assert current["geometry"]["bbox"] == [2, 3, 6, 6]
    assert current["metrics"]["area"] == 36
    assert current["metrics"]["aspect_ratio"] == 1
    assert current["metrics"]["mask_coverage"] == 1
    assert current["metrics"]["mask_mean_confidence"] == 0.88
    assert current["source_metrics"]["area"] == 960
    assert current["source_geometry"]["bbox"] == [0, 0, 40, 24]
    assert current["metrics"]["custom_quality"] == 0.7
    selected = np.zeros((24, 40), np.uint8)
    selected[3:6, 2:8] = 255
    data["mask"] = selected
    current = block.execute(data).outputs["tracks"]["items"][0]
    assert current["metrics"]["area"] == current["metrics"]["mask_area"] == 18
    assert current["metrics"]["aspect_ratio"] == 2
    assert current["metrics"]["mask_mean_confidence"] is None
