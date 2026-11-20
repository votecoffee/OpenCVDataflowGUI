"""Verify digital fiducial generation, detector round trips, and print geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import image_format

TYPE_ID = "calibration.generate_fiducial_target"
EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/fiducial_targets/fiducial_targets.json"
)


def expected_pixels(target, xyz):
    origin = np.array(target["target_rect_px"][:2])
    return np.asarray(xyz)[..., :2] * target["pixels_per_unit"] + origin - 0.5


def assert_image_geometry(image, target):
    assert image.dtype == np.uint8 and image_format(image) == "gray"
    assert image.shape[::-1] == tuple(target["image_size"])
    assert set(np.unique(np.asarray(image))) == {0, 255}
    x, y, w, h = target["target_rect_px"]
    assert image.shape == (h + 2 * y, w + 2 * x)
    for margin in (image[:y], image[y + h :], image[:, :x], image[:, x + w :]):
        assert not margin.size or np.all(margin == 255)
    np.testing.assert_allclose(
        np.asarray(target["print_size"]) * target["pixels_per_unit"], (w + 2 * x, h + 2 * y)
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_decodes_all_three_generated_layouts(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for kind in ("marker", "grid", "charuco"):
            maker = graph.blocks[f"make_{kind}"]
            detector = graph.blocks[f"read_{kind}"]
            source, found = maker.last_result, detector.last_result
            assert source and source.passed, maker.error
            assert found and found.passed, (detector.error, detector.last_result)
            target = source.outputs["target"]
            assert_image_geometry(source.outputs["image"], target)

            if kind == "charuco":
                assert len(found.outputs["corner_ids"]) == 15
                assert target["legacy_pattern"] is True
                assert target["marker_ids"] == list(range(17, 29))
                np.testing.assert_allclose(
                    found.outputs["object_points"], target["charuco_corners_xyz"], atol=1e-6
                )
                points = expected_pixels(target, target["charuco_corners_xyz"])
                np.testing.assert_allclose(found.outputs["image_points"], points, atol=0.3)
            else:
                reading = {marker["id"]: marker["corners"] for marker in found.outputs["markers"]}
                assert set(reading) == set(target["marker_ids"])
                points = expected_pixels(target, target["marker_corners_xyz"])
                np.testing.assert_allclose(
                    [reading[key] for key in target["marker_ids"]], points, atol=0.3
                )
    finally:
        graph.close()


def test_fractional_chess_pitch_and_marker_insets_survive_png_and_rotation(tmp_path):
    registry = builtin_registry()
    config = {
        "layout": "charuco",
        "columns": 6,
        "rows": 10,
        "square_length": 92.40079041989269 / 5,
        "marker_length": 55.69078554544611 / 5,
        "dpi": 127,
    }
    make = registry.create(TYPE_ID, parameters=config)
    result = make.execute({})
    assert result and result.passed, make.error
    output, target = result.outputs["image"], result.outputs["target"]
    assert_image_geometry(output, target)
    assert target["square_length"] == pytest.approx(config["square_length"])
    assert abs(target["marker_length"] - config["marker_length"]) <= 1 / target["pixels_per_unit"]

    path = tmp_path / "printed-target.png"
    save = registry.create("output.image", parameters={"path": str(path)})
    assert save.execute({"image": output}).passed
    raw = np.frombuffer(path.read_bytes(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(image, output)
    detector = registry.create(
        "calibration.charuco",
        parameters={
            key: target[key]
            for key in (
                "columns",
                "rows",
                "square_length",
                "marker_length",
                "dictionary",
                "marker_ids",
                "border_bits",
                "legacy_pattern",
                "units",
            )
        },
    )
    reading = detector.execute({"image": cv2.rotate(image, cv2.ROTATE_180)})
    assert reading and reading.passed, detector.error
    assert reading.outputs["corner_ids"] == list(range(45))
    np.testing.assert_allclose(
        reading.outputs["object_points"], target["charuco_corners_xyz"], atol=2e-5
    )
    expected = (
        np.array(image.shape[::-1]) - 1 - expected_pixels(target, target["charuco_corners_xyz"])
    )
    np.testing.assert_allclose(reading.outputs["image_points"], expected, atol=0.3)


def test_fractional_grid_physics_preserve_marker_length_and_meter_conversion():
    registry = builtin_registry()
    config = {
        "layout": "aruco_grid",
        "marker_length": 31.485946419618518 / 5,
        "marker_separation": 15.7814460216055 / 5,
        "margin": 5.0,
        "columns": 5,
        "rows": 3,
        "dpi": 127,
    }
    mm = registry.create(TYPE_ID, parameters=config).execute({}).outputs
    m_config = {
        **config,
        "units": "m",
        **{key: config[key] * 0.001 for key in ("marker_length", "marker_separation", "margin")},
    }
    meters = registry.create(TYPE_ID, parameters=m_config).execute({}).outputs
    np.testing.assert_array_equal(mm["image"], meters["image"])
    a, b = mm["target"], meters["target"]
    assert a["marker_length"] == pytest.approx(config["marker_length"])
    assert abs(a["marker_separation"] - config["marker_separation"]) <= 0.5 / a["pixels_per_unit"]
    assert b["units"] == "m"
    assert b["effective_dpi"] == pytest.approx(a["effective_dpi"])
    for key in ("marker_corners_xyz", "target_size", "print_size"):
        np.testing.assert_allclose(np.asarray(b[key]) * 1000, a[key], rtol=1e-8)
    edge = np.asarray(a["marker_corners_xyz"])
    pitches = edge[1, 0] - edge[0, 0]
    np.testing.assert_allclose(pitches, [a["marker_length"] + a["marker_separation"], 0, 0])


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"layout": "aruco_marker", "marker_ids": [50]}, "unique marker IDs"),
        ({"layout": "aruco_grid", "columns": 11, "rows": 5}, "fewer IDs"),
        ({"layout": "aruco_grid", "marker_separation": 1e-4}, "between markers"),
        ({"layout": "charuco", "columns": 2}, "three squares per axis"),
        ({"layout": "charuco", "marker_length": 30}, "less than square length"),
        ({"layout": "charuco", "marker_length": 29.99}, "white inset"),
        ({"layout": "aruco_marker", "marker_length": 0.001}, "code and border"),
        ({"dpi": 2400, "margin": 500}, "image pixels"),
    ],
)
def test_rejects_invalid_or_oversized_layout_before_generation(settings, error, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid target reached a native rasterizer")

    for name in ("generateImageMarker", "GridBoard", "CharucoBoard"):
        monkeypatch.setattr(cv2.aruco, name, unexpected)
    block = builtin_registry().create(TYPE_ID, parameters=settings)
    assert block.execute({}) is None
    assert error in block.error
