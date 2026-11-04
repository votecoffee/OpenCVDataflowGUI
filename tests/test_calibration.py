import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import calibration_templates, grid_settings, map_points
from opencv_dataflow.toolblocks import ToolBlock


def chessboard(columns=7, rows=5, square=40, margin=40):
    image = np.full(
        ((rows + 1) * square + margin * 2, (columns + 1) * square + margin * 2, 3), 200, np.uint8
    )
    for row in range(rows + 1):
        for col in range(columns + 1):
            y, x = margin + row * square, margin + col * square
            image[y : y + square, x : x + square] = 255 if (row + col) % 2 else 0
    return image


def test_real_grid_detection_origin_and_physical_mapping():
    registry = builtin_registry()
    block = registry.create(
        "calibration.grid",
        parameters={
            "columns": 7,
            "rows": 5,
            "spacing": 12.5,
            "origin_column": 2,
            "origin_row": 1,
            "max_error": 0.1,
        },
    )
    block.execute({"image": chessboard()})
    assert block.state == "pass", block.error
    mapping = block.last_result.outputs["mapping"]
    corners = np.array(block.last_result.outputs["corners"])
    origin_index = 1 * 7 + 2
    measure = registry.create("calibration.pixel_to_physical")
    measure.execute(
        {
            "mapping": mapping,
            "points": corners[[origin_index, origin_index + 1, origin_index + 7]].tolist(),
        }
    )
    assert measure.state == "pass", measure.error
    np.testing.assert_allclose(
        measure.last_result.outputs["points"]["physical"], [[0, 0], [12.5, 0], [0, 12.5]], atol=0.03
    )
    assert mapping["units"] == "mm" and mapping["plane_only"]
    assert mapping["rms_error"] < 0.1
    assert "grid origin" in block.last_graphics


def test_perspective_grid_map_and_reverse_order():
    image = chessboard()
    height, width = image.shape[:2]
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
        np.float32([[25, 20], [width - 35, 5], [width - 5, height - 20], [5, height - 5]]),
    )
    warped = cv2.warpPerspective(image, transform, (width, height), borderValue=(200, 200, 200))
    block = builtin_registry().create(
        "calibration.grid", parameters={"columns": 7, "rows": 5, "reverse_order": True}
    )
    block.execute({"image": warped})
    assert block.state == "pass", block.error
    mapping = block.last_result.outputs["mapping"]
    np.testing.assert_allclose(map_points(mapping, [mapping["origin_pixel"]]), [[0, 0]], atol=0.02)
    assert mapping["grid_orientation"] == "reversed"


def test_grid_not_found_and_mapping_quality_rejection():
    registry = builtin_registry()
    block = registry.create("calibration.grid", parameters={"columns": 7, "rows": 5})
    block.execute({"image": np.full((200, 200, 3), 100, np.uint8)})
    assert block.state == "error" and "not found" in block.error
    measure = registry.create("calibration.pixel_to_physical")
    mapping = {
        "model": "planar_homography_v1",
        "pixel_to_physical": np.eye(3).tolist(),
        "rms_error": 2.0,
        "units": "mm",
    }
    measure.execute({"mapping": mapping})
    assert "exceeds" in measure.error
    mapping["rms_error"] = 0.0
    mapping["pixel_to_physical"] = [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    with pytest.raises(ValueError, match="Invalid"):
        map_points(mapping, [[0, 0]])


def test_lens_solver_against_known_synthetic_camera(tmp_path, monkeypatch):
    import opencv_dataflow.calibration as calibration

    registry = builtin_registry()
    block = registry.create("calibration.lens", parameters={"columns": 7, "rows": 5})
    _, world = grid_settings(block.parameters)
    camera = np.array([[750.0, 0, 320], [0, 760, 240], [0, 0, 1]])
    distortion = np.array([0.03, -0.02, 0.001, -0.001, 0.0])
    views = []
    paths = []
    for index in range(6):
        rotation = np.array([0.04 * index - 0.12, 0.05 * index - 0.14, 0.03 * index])
        translation = np.array([-25.0 + index * 4, -20.0 + index * 3, 400.0 + index * 22])
        corners, _ = cv2.projectPoints(world, rotation, translation, camera, distortion)
        views.append(corners.reshape(-1, 2))
        path = tmp_path / f"view{index}.png"
        ok, pixels = cv2.imencode(".png", np.full((480, 640, 3), index, np.uint8))
        assert ok
        path.write_bytes(pixels.tobytes())
        paths.append(str(path))
    counter = iter(views)
    monkeypatch.setattr(calibration, "find_grid", lambda image, size: (next(counter), image))
    block.parameters["image_paths"] = paths
    block.execute({})
    assert block.state == "pass", block.error
    result = block.last_result.outputs["calibration"]
    np.testing.assert_allclose(result["camera_matrix"], camera, atol=0.3)
    assert result["rms_pixels"] < 0.001
    undistort = registry.create("calibration.undistort")
    image = np.random.default_rng(23).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    undistort.execute({"image": image, "calibration": result})
    assert undistort.state == "pass", undistort.error
    np.testing.assert_array_equal(
        undistort.last_output_images["image"],
        cv2.undistort(image, np.array(result["camera_matrix"]), np.array(result["distortion"])),
    )
    undistort.execute({"image": image[:100], "calibration": result})
    assert "dimensions" in undistort.error
    block.parameters["image_paths"] = paths[:1]
    block.execute({})
    assert "three distinct" in block.error


def test_white_balance_roi_manual_and_alpha():
    registry = builtin_registry()
    image = np.full((8, 8, 4), [30, 60, 90, 127], np.uint8)
    block = registry.create("calibration.color_balance", parameters={"roi": [1, 1, 4, 4]})
    block.execute({"image": image})
    assert block.state == "pass", block.error
    np.testing.assert_array_equal(block.last_output_images["image"][0, 0], [60, 60, 60, 127])
    block.parameters.update(mode="manual", blue_gain=2.0, green_gain=1.0, red_gain=0.5)
    block.execute({"image": image})
    np.testing.assert_array_equal(block.last_output_images["image"][0, 0], [60, 60, 45, 127])
    block.parameters["roi"] = [-1, 0, 2, 2]
    block.execute({"image": image})
    assert "inside" in block.error


def test_calibration_toolblock_templates_are_editable_and_classified():
    registry = builtin_registry()
    templates = calibration_templates(registry)
    assert len(templates) == 5
    for template in templates:
        block = ToolBlock.from_definition(template, registry)
        assert block.block_type == "Calibration"
        assert block.subtype != "General"
        assert len(block.graph.blocks) == 1
        assert block.definition() == template


def test_mapping_profile_save_load_and_file_preservation(tmp_path):
    registry = builtin_registry()
    grid = registry.create("calibration.grid", parameters={"columns": 7, "rows": 5})
    grid.execute({"image": chessboard()})
    assert grid.state == "pass", grid.error
    profile = grid.last_result.outputs["mapping"]
    path = tmp_path / "mapping.json"
    save = registry.create("output.calibration", parameters={"path": str(path)})
    save.execute({"profile": profile})
    assert save.state == "pass", save.error
    before = path.read_bytes()
    save.execute({"profile": profile})
    assert save.state == "error" and path.read_bytes() == before
    load = registry.create("input.calibration", parameters={"path": str(path)})
    load.execute({})
    assert load.state == "pass", load.error
    assert load.last_result.outputs["profile"] == profile
    measure = registry.create("calibration.pixel_to_physical")
    measure.execute(
        {"mapping": load.last_result.outputs["profile"], "points": [profile["origin_pixel"]]}
    )
    np.testing.assert_allclose(
        measure.last_result.outputs["points"]["physical"], [[0, 0]], atol=1e-6
    )
