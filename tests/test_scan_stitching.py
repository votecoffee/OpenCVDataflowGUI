"""Exercise ordered scan registration, failure gates, and mosaic coverage."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mosaic_photometry import FeatherMosaic, corrected_view
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, metadata

EXAMPLE = Path(__file__).parents[1] / "examples/image_processing/stitch_scan_sequence"
EXPOSURE_EXAMPLE = EXAMPLE.parent / "stitch_scan_exposure/stitch_scan_exposure.json"
TYPE = "cv.stitch_scan_sequence"


def _tile(index):
    return cv2.imread(str(EXAMPLE / "assets" / f"tile-{index:02d}.png"), cv2.IMREAD_UNCHANGED)


def _group(*images):
    return FlowGroup(
        tuple(ValuePacket(image, {}) for image in images), tuple(map(str, range(len(images))))
    )


def test_saved_scan_waits_for_batch_and_reconstructs_all_covered_pixels():
    graph = Graph.load(EXAMPLE / "stitch_scan_sequence.json", builtin_registry())
    try:
        graph.set_run_profile("lean")
        for _ in range(2):
            graph.run()
            assert graph.blocks["batch"].state == "waiting"
            assert graph.blocks["stitch"].state == "inactive"
        graph.run()
        stitch = graph.blocks["stitch"]
        assert stitch.state == "pass", stitch.error
        result = stitch.last_result
        layout = result.outputs["layout"]
        assert layout["view_count"] == 3
        assert [p["overlap_pixels"] for p in layout["pairs"]] == [11088, 11088]
        assert [p["status"] for p in layout["pairs"]] == ["registered", "registered"]
        assert [v["source_ticket"]["value"] for v in layout["views"]] == [
            "tile-01.png",
            "tile-02.png",
            "tile-03.png",
        ]
        assert metadata(result.outputs["image"])["ticket"]["source"] == "batch"
        assert (
            metadata(result.outputs["image"])["spatial"]
            == metadata(result.outputs["coverage"])["spatial"]
        )
        expected = np.concatenate([_tile(1), _tile(2)[:, 84:], _tile(3)[:, 84:]], axis=1)
        x, y = map(int, np.asarray(layout["views"][0]["source_to_mosaic"])[:2, 2])
        np.testing.assert_array_equal(result.outputs["image"][y : y + 132, x : x + 372], expected)
        assert layout["covered_pixels"] == expected.size
        assert graph.blocks["measure"].last_result.outputs["selected_count"] == expected.size
    finally:
        graph.close()


def test_similarity_motion_recovers_rotated_16_bit_view():
    first = _tile(1).astype(np.uint16) * 257
    forward = cv2.getRotationMatrix2D((89.5, 65.5), 2.0, 1.01)
    forward[:, 2] += [5, -3]
    second = cv2.warpAffine(first, forward, first.shape[1::-1])
    block = builtin_registry().create(TYPE, parameters={"motion_model": "similarity"})
    result = block.execute({"views": _group(first, second)})
    assert result is not None and result.passed, block.error
    assert result.outputs["image"].dtype == np.uint16
    fit = np.asarray(result.outputs["layout"]["pairs"][0]["source_to_previous"])
    points = np.array([[26, 25, 1], [95, 58, 1], [142, 107, 1]], np.float64)
    np.testing.assert_allclose(points @ np.vstack((forward, [0, 0, 1])).T @ fit.T, points, atol=0.4)


def test_masks_keep_excluded_later_pixels_from_overwriting_valid_earlier_pixels():
    first, second = _tile(1), _tile(2) + 20
    mask = np.full_like(second, 255)
    mask[40:61, 35:56] = 0
    block = builtin_registry().create(
        TYPE, parameters={"interpolation": "nearest", "blend": "prefer_last"}
    )
    result = block.execute(
        {"views": _group(first, second), "masks": _group(np.full_like(first, 255), mask)}
    )
    assert result is not None and result.passed, block.error
    layout = result.outputs["layout"]
    x, y = map(int, np.asarray(layout["views"][0]["source_to_mosaic"])[:2, 2])
    assert result.outputs["image"][y + 50, x + 140] == first[50, 140]
    assert result.outputs["image"][y + 50, x + 115] == second[50, 19]
    assert result.outputs["overlap"][y + 50, x + 140] == 0


def test_unregistrable_pair_reports_which_join_failed_without_partial_mosaic():
    first = _tile(1)
    block = builtin_registry().create(TYPE)
    result = block.execute({"views": _group(first, _tile(2), np.zeros_like(first))})
    assert result is not None and not result.passed, block.error
    assert result.outputs["image"] is result.outputs["coverage"] is NO_VALUE
    first_pair, failed = result.outputs["layout"]["pairs"]
    assert first_pair["status"] == "registered"
    assert failed == {
        "view_index": 2,
        "previous_index": 1,
        "matches": 0,
        "status": "insufficient_matches",
    }
    assert result.fail_reasons[0]["code"] == "scan_registration_failed"


def test_overlap_and_mask_group_bounds_are_checked():
    first = _tile(1)
    block = builtin_registry().create(TYPE, parameters={"min_overlap_pixels": 12000})
    result = block.execute({"views": _group(first, _tile(2))})
    assert result is not None and not result.passed
    assert result.outputs["image"] is NO_VALUE
    assert result.outputs["layout"]["pairs"][0]["status"] == "insufficient_overlap"
    assert result.fail_reasons[0]["code"] == "scan_insufficient_overlap"
    assert block.execute({"views": _group(first, first), "masks": _group(first)}) is None
    assert block.error_kind == "invalid" and "one mask per view" in block.error
    assert block.execute({"views": _group(*([first] * 9))}) is None
    assert block.error_kind == "invalid" and "two to eight" in block.error


def test_saved_scan_corrects_cumulative_gain_without_changing_coverage():
    graph = Graph.load(EXPOSURE_EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile("lean")
        for _ in range(3):
            graph.run()
        raw = graph.blocks["stitch"].last_result.outputs
        corrected = graph.blocks["corrected"]
        assert corrected.state == "pass", corrected.error
        result = corrected.last_result.outputs
        layout = result["layout"]
        assert [pair["exposure"]["samples"] for pair in layout["pairs"]] == [11088, 11088]
        assert [pair["exposure"]["median_error_after"] for pair in layout["pairs"]] == [0, 0]
        assert [view["exposure"]["gain"] for view in layout["views"]] == [1, 1.25, 1.6]
        for name in ("coverage", "overlap"):
            np.testing.assert_array_equal(raw[name], result[name])
        assert metadata(raw["image"])["spatial"] == metadata(result["image"])["spatial"]
        expected = np.concatenate([_tile(1), _tile(2)[:, 84:], _tile(3)[:, 84:]], axis=1)
        x, y = map(int, np.asarray(layout["views"][0]["source_to_mosaic"])[:2, 2])
        expected = expected.astype(np.uint16) * 200
        np.testing.assert_array_equal(result["image"][y : y + 132, x : x + 372], expected)
        assert result["image"].dtype == expected.dtype
        assert not result["image"][result["coverage"] == 0].any()
        assert graph.blocks["measure"].last_result.outputs["selected_count"] == 49104
    finally:
        graph.close()


def test_offset_changes_color_but_leaves_alpha_and_masked_pixels_alone():
    first = cv2.cvtColor(_tile(1), cv2.COLOR_GRAY2BGRA)
    second = cv2.cvtColor(_tile(2), cv2.COLOR_GRAY2BGRA)
    second[..., :3] += 20
    second[..., 3] = 75
    mask = np.full(second.shape[:2], 255, dtype=np.uint8)
    mask[40:61, 35:56] = 0
    block = builtin_registry().create(
        TYPE, parameters={"interpolation": "nearest", "blend": "prefer_last", "exposure": "offset"}
    )
    result = block.execute(
        {"views": _group(first, second), "masks": _group(np.full_like(mask, 255), mask)}
    )
    assert result is not None and result.passed, block.error
    layout = result.outputs["layout"]
    fit = layout["pairs"][0]["exposure"]
    assert fit["offset"] == -20 and fit["median_error_after"] == 0
    assert fit["samples"] == 11088 - 21 * 21
    x, y = map(int, np.asarray(layout["views"][0]["source_to_mosaic"])[:2, 2])
    np.testing.assert_array_equal(result.outputs["image"][y + 50, x + 140], first[50, 140])
    np.testing.assert_array_equal(
        result.outputs["image"][y + 50, x + 115], [*first[50, 115, :3], 75]
    )


@pytest.mark.parametrize(
    ("settings", "status"),
    [
        ({"min_exposure_samples": 32768}, "insufficient_exposure_samples"),
        ({"max_exposure_gain": 1.25}, "exposure_out_of_range"),
    ],
)
def test_exposure_failure_reports_pair_and_withholds_image(settings, status):
    block = builtin_registry().create(
        TYPE, parameters={"exposure": "gain", "interpolation": "nearest", **settings}
    )
    first = _tile(1).astype(np.uint16) * 200
    second = _tile(2).astype(np.uint16) * 125
    result = block.execute({"views": _group(first, second)})
    assert result is not None and not result.passed, block.error
    assert result.outputs["image"] is result.outputs["coverage"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "scan_exposure_failed"
    pair = result.outputs["layout"]["pairs"][0]
    assert pair["status"] == pair["exposure"]["status"] == status
    assert pair["exposure"]["samples"] == 11088


def test_feather_normalizes_weights_and_excludes_invalid_pixels():
    shape = (11, 11)
    first_valid, second_valid = np.zeros(shape, bool), np.zeros(shape, bool)
    first_valid[:, :8] = True
    second_valid[:, 3:10] = True
    second_valid[5, 5] = False
    blender = FeatherMosaic(shape, 8)
    blender.add(np.full(shape, 20.0, np.float32), first_valid)
    blender.add(np.full(shape, 100.0, np.float32), second_valid)
    result = blender.finish(np.dtype(np.float32))
    assert result.dtype == np.float32
    assert result[5, 0] == result[5, 5] == 20
    assert result[5, 9] == 100
    assert result[5, 10] == 0
    assert 20 < result[4, 4] < result[4, 7] < 100


def test_exposure_correction_ignores_masked_high_dynamic_range_samples():
    image = np.array([[0.2, np.finfo(np.float64).max]])
    corrected = corrected_view(image, np.array([[True, False]]), (2.0, 0.0))
    assert corrected[0, 0] == 0.4
    assert np.isfinite(corrected).all()
