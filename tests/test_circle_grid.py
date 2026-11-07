"""Verify detected circle grids, physical coordinates, and the lens solver."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import map_points
from opencv_dataflow.core import Graph


def _board_xy(columns, rows, asymmetric):
    return np.float32(
        [
            [2 * column + row % 2 if asymmetric else column, row]
            for row in range(rows)
            for column in range(columns)
        ]
    )


@pytest.mark.parametrize(
    ("detector", "rows", "light", "clustering", "reverse"),
    [
        ("circles_symmetric", 5, True, True, False),
        ("circles_asymmetric", 6, False, False, False),
        ("circles_asymmetric", 5, False, False, True),
    ],
)
def test_circle_board_fit_preserves_pitch_and_anchors_selected_center(
    detector, rows, light, clustering, reverse
):
    registry = builtin_registry()
    block = registry.create(
        "calibration.grid",
        parameters={
            "detector": detector,
            "columns": 4,
            "rows": rows,
            "spacing": 12.5,
            "origin_column": 1,
            "origin_row": 1,
            "reverse_order": reverse,
            "max_error": 0.01,
            "circle_polarity": "light" if light else "dark",
            "circle_clustering": clustering,
            "circle_min_circularity": 0.6,
            "circle_min_convexity": 0.0,
            "circle_min_inertia_ratio": 0.0,
        },
    )
    board = _board_xy(4, rows, detector == "circles_asymmetric")
    image = np.full((120 + rows * 32, 376), 0 if light else 255, np.uint8)
    for x, y in (60 + 32 * board).astype(int):
        cv2.circle(image, (int(x), int(y)), 9, 255 if light else 0, -1)
    result = block.execute({"image": image})
    assert result is not None and result.passed, (block.error, result.message if result else None)
    mapping = result.outputs["mapping"]
    centers = result.outputs["corners"]
    assert mapping["origin_pixel"] == centers[5]
    if reverse:
        board = -board[::-1]
    expected = (board - board[5]) * 12.5
    np.testing.assert_allclose(map_points(mapping, centers), expected, atol=0.001)
    assert len(result.drawings["Grid features"].objects) == 4 * rows


def test_staggered_circle_lens_fit_recovers_synthetic_camera_from_rendered_views(tmp_path):
    registry = builtin_registry()
    camera = np.array([[620.0, 0, 270], [0, 625, 210], [0, 0, 1]])
    distortion = np.array([0.025, -0.008, 0.001, -0.0008, 0])
    world = np.column_stack((_board_xy(4, 6, True) * 10, np.zeros(24, np.float32)))
    poses = (
        ((0.28, -0.22, 0.12), (-32.0, -30, 215)),
        ((-0.20, 0.33, 0.20), (-38.0, -18, 225)),
        ((0.34, 0.10, -0.31), (-50.0, -12, 240)),
        ((-0.30, -0.20, -0.18), (-18.0, -45, 240)),
        ((0.15, 0.38, 0.35), (-20.0, -15, 205)),
        ((-0.18, 0.14, -0.25), (-48.0, -27, 270)),
    )
    paths = []
    for index, (rotation, translation) in enumerate(poses):
        image = np.full((420, 540), 255, np.uint8)
        pixels, _ = cv2.projectPoints(
            world, np.asarray(rotation), np.asarray(translation), camera, distortion
        )
        for center in pixels.reshape(-1, 2):
            point = tuple(np.rint(center * 256).astype(int))
            cv2.circle(image, point, 5 * 256, 0, -1, cv2.LINE_AA, shift=8)
        path = tmp_path / f"view-{index}.png"
        ok, png = cv2.imencode(".png", image)
        assert ok
        path.write_bytes(png.tobytes())
        paths.append(str(path))
    block = registry.create(
        "calibration.lens",
        parameters={
            "detector": "circles_asymmetric",
            "columns": 4,
            "rows": 6,
            "max_rms_pixels": 0.1,
            "image_paths": paths,
        },
    )
    result = block.execute({})
    assert result is not None and result.passed, (block.error, result.message if result else None)
    model = result.outputs["calibration"]
    assert model["views"] == len(poses) and model["rms_pixels"] < 0.1
    np.testing.assert_allclose(model["camera_matrix"], camera, atol=2.5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"circle_min_area": 80, "circle_max_area": 80}, "Circle area minimum"),
        ({"circle_min_threshold": 220, "circle_max_threshold": 50}, "Circle threshold minimum"),
        ({}, "circle centers"),
    ],
)
def test_circle_detector_reports_invalid_ranges_and_a_missing_target(changes, message):
    block = builtin_registry().create(
        "calibration.grid",
        parameters={"detector": "circles_symmetric", "columns": 4, "rows": 5, **changes},
    )
    assert block.execute({"image": np.full((200, 200), 255, np.uint8)}) is None
    assert message in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_circle_grid_example_maps_staggered_centers(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools/circle_grid_calibration/circle_grid_calibration.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    data = graph.blocks["physical"].last_result.outputs["points"]
    world = _board_xy(4, 6, True) * 12.5
    assert data["units"] == "mm"
    np.testing.assert_allclose(data["physical"], world - world[5], atol=0.001)
