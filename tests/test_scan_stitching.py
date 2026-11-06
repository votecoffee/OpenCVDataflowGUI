"""Exercise ordered scan registration, failure gates, and mosaic coverage."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, metadata

EXAMPLE = Path(__file__).parents[1] / "examples/image_processing/stitch_scan_sequence"
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
