"""Check native marker metadata, spatial provenance and physical point identity."""

from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/marked_checkerboard"


@pytest.fixture
def images():
    return run_path(str(EXAMPLE / "build_example.py"))["fixture"]()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_recovers_same_board_coordinates_in_overlapping_crops(profile, images):
    views, board_origin, pitch = images
    graph = Graph.load(EXAMPLE / "marked_checkerboard.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        known_points = {}
        for name in ("full_board", "crop_a", "crop_b"):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            out = result.outputs
            xy, object_points, ids, meta = (
                np.asarray(out[key])
                for key in ("image_points", "object_points", "corner_ids", "native_metadata")
            )
            expected = cv2.perspectiveTransform(
                (board_origin + ids * pitch)[None, ...], views[name][1]
            )[0]
            expected_origin = cv2.perspectiveTransform(board_origin[None, None, :], views[name][1])[
                0, 0
            ]
            tolerance_pixels = 0.7 if name == "crop_b" else 0.25
            assert np.linalg.norm(xy - expected, axis=1).max() < tolerance_pixels
            np.testing.assert_allclose(out["report"]["origin_pixel"], expected_origin, atol=0.3)
            np.testing.assert_array_equal(object_points[:, :2], ids * 12)
            assert not object_points[:, 2].any()
            assert ids.tolist() == sorted(ids.tolist(), key=lambda item: (item[1], item[0]))
            assert (meta == 3).sum() == 2 and (meta == 4).sum() == 1
            assert out["corner_count"] == meta.size == len(ids)
            assert out["report"]["native_inner_size"] == list(meta.shape[::-1])
            assert result.payloads["object_points"]["object_point_units"] == "mm"
            assert (
                result.payloads["image_points"]["spatial"]
                == block.input_payloads["image"]["spatial"]
            )
            assert ("Marked origin and axes" in result.drawings) == (profile == "debug")
            known_points[name] = {
                tuple(key): tuple(point) for key, point in zip(ids, object_points)
            }
        assert len(known_points["full_board"]) == 17 * 13
        overlap = known_points["crop_a"].keys() & known_points["crop_b"].keys()
        assert len(overlap) >= 100
        for point_id in overlap:
            assert (
                known_points["crop_a"][point_id]
                == known_points["crop_b"][point_id]
                == known_points["full_board"][point_id]
            )
        rejected = graph.blocks["no_marker"].last_result
        assert rejected and not rejected.passed
        assert rejected.outputs["report"]["status"] == "not_found"
        assert rejected.outputs["corner_ids"] is NO_VALUE
        assert rejected.outputs["object_points"] is NO_VALUE
        # Verify the control contains enough chessboard corners without its origin marker.
        assert cv2.findChessboardCornersSB(
            views["no_marker"][0], (5, 5), flags=cv2.CALIB_CB_LARGER
        )[0]
    finally:
        graph.close()


@pytest.mark.local_examples
def test_scoped_switches_pass_native_flags_and_keep_raw_cell_order(images, monkeypatch):
    views, _, _ = images
    base = cv2.findChessboardCornersSBWithMeta
    captured = []

    def native(gray, size, flags):
        result = base(gray, size, flags)
        captured.append((size, flags, result[2].copy()))
        return result

    monkeypatch.setattr(cv2, "findChessboardCornersSBWithMeta", native)
    block = builtin_registry().create(
        "calibration.marked_checkerboard",
        parameters={
            "allow_larger": True,
            "normalize": False,
            "exhaustive": True,
            "accuracy": True,
        },
    )
    source = views["crop_a"][0].astype(np.uint16) * 257
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    size, flags, meta = captured[0]
    assert size == (7, 5)
    assert flags == (
        cv2.CALIB_CB_MARKER | cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    np.testing.assert_array_equal(result.outputs["native_metadata"], meta)


@pytest.mark.local_examples
def test_unresolved_marker_has_no_coordinate_outputs_and_keeps_diagnostics(images, monkeypatch):
    views, _, _ = images
    native = cv2.findChessboardCornersSBWithMeta

    def remove_marker(gray, size, flags):
        found, points, meta = native(gray, size, flags)
        assert found
        meta = np.where(meta >= 3, meta - 2, meta).astype(np.uint8)
        return found, points, meta

    monkeypatch.setattr(cv2, "findChessboardCornersSBWithMeta", remove_marker)
    block = builtin_registry().create("calibration.marked_checkerboard")
    result = block.execute({"image": views["crop_a"][0]})
    assert result and not result.passed
    assert result.outputs["report"]["status"] == "unresolved_marker"
    assert result.outputs["corner_count"] > 50
    assert result.outputs["native_metadata"] is not NO_VALUE
    for key in ("observation", "corner_ids", "image_points", "object_points"):
        assert result.outputs[key] is NO_VALUE


@pytest.mark.local_examples
def test_invalid_native_grid_is_contained_and_next_run_recovers(images, monkeypatch):
    views, _, _ = images
    native = cv2.findChessboardCornersSBWithMeta

    def truncated(gray, size, flags):
        found, points, meta = native(gray, size, flags)
        return found, points[:-1], meta

    block = builtin_registry().create("calibration.marked_checkerboard")
    with monkeypatch.context() as patch:
        patch.setattr(cv2, "findChessboardCornersSBWithMeta", truncated)
        assert block.execute({"image": views["crop_a"][0]}) is None
        assert "invalid or oversized checkerboard metadata" in block.error
    result = block.execute({"image": views["crop_a"][0]})
    assert result and result.passed and not block.error
