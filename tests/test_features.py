import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format


def run(type_id, inputs, parameters=None):
    block = builtin_registry().create(type_id, parameters=parameters)
    block.execute(inputs)
    assert block.state == "pass", block.error
    return block


def test_equalization_and_uint16_clahe():
    image = np.full((32, 32), 40, np.uint8)
    image[:, 16:] = 60
    output = run("cv.equalize", {"image": image}).last_output_images["image"]
    assert output.min() == 0 and output.max() == 255
    high = image.astype(np.uint16) * 257
    block = run("cv.clahe", {"image": high})
    assert block.last_output_images["image"].dtype == np.uint16
    np.testing.assert_array_equal(
        block.last_output_images["image"],
        cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(high),
    )


@pytest.mark.parametrize("type_id", ["cv.equalize", "cv.clahe"])
def test_contrast_preserves_rgba_depth_alpha_and_unselected_pixels(type_id):
    rng = np.random.default_rng(42)
    pixels = rng.integers(10000, 40000, (24, 32, 4), dtype=np.uint16)
    image = ImageArray(pixels, "rgba")
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, :16] = 255
    output = run(type_id, {"image": image, "mask": mask}).last_output_images["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgba"
    np.testing.assert_array_equal(output[..., 3], pixels[..., 3])
    np.testing.assert_array_equal(output[mask == 0], pixels[mask == 0])
    assert np.any(output[mask != 0, :3] != pixels[mask != 0, :3])

    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create(type_id))
    old_project = graph.to_dict()
    old_project["blocks"][0]["parameters"] = {
        name: value for name, value in block.parameters.items() if name != "output_format"
    }
    legacy = Graph.from_dict(old_project, registry).blocks[block.id]
    legacy.execute({"image": ImageArray((pixels // 257).astype(np.uint8), "rgba")})
    assert legacy.parameters["output_format"] == "gray"
    assert image_format(legacy.last_output_images["image"]) == "gray"


def test_normalize_and_gamma_preserve_alpha():
    output = run(
        "cv.normalize",
        {"image": np.array([[50, 100, 150]], np.uint16)},
        {"minimum": 0.0, "maximum": 1.0, "pixel_depth": "float32"},
    ).last_output_images["image"]
    np.testing.assert_allclose(output, [[0, 0.5, 1]], atol=1e-6)
    image = np.full((3, 3, 4), [64, 128, 255, 90], np.uint8)
    corrected = run("cv.gamma", {"image": image}, {"gamma": 2.0}).last_output_images["image"]
    np.testing.assert_array_equal(corrected[0, 0], [128, 181, 255, 90])
    np.testing.assert_array_equal(
        run("cv.gamma", {"image": image}).last_output_images["image"], image
    )


def test_split_merge_preserve_channel_values_and_encoding():
    image = ImageArray(np.random.default_rng(8).integers(0, 65536, (12, 15, 3), np.uint16), "rgb")
    split = run("cv.split_channels", {"image": image})
    assert all(v.ndim == 2 for v in split.last_output_images.values())
    merged = run("cv.merge_channels", split.last_result.outputs, {"color_format": "rgb"})
    np.testing.assert_array_equal(merged.last_output_images["image"], image)
    assert image_format(merged.last_output_images["image"]) == "rgb"


def test_mask_nonzero_and_unsigned_difference():
    image = ImageArray(np.full((4, 6, 3), [10, 20, 30], np.uint16), "rgb")
    mask = np.zeros((4, 6), np.uint8)
    mask[:, 2:] = 7
    result = run("cv.apply_mask", {"image": image, "mask": mask}).last_output_images["image"]
    np.testing.assert_array_equal(result[:, :2], 0)
    np.testing.assert_array_equal(result[:, 2:], image[:, 2:])
    assert image_format(result) == "rgb"
    a, b = np.array([[0, 255]], np.uint8), np.array([[255, 0]], np.uint8)
    difference = run("cv.absdiff", {"image_a": a, "image_b": b}).last_output_images["image"]
    np.testing.assert_array_equal(difference, [[255, 255]])


def test_corners_find_known_rectangle_and_empty_image():
    image = np.zeros((100, 120), np.uint8)
    image[20:71, 30:91] = 255
    block = run("inspect.corners", {"image": image})
    assert block.last_result.outputs["count"] == 4
    actual = sorted(block.last_result.outputs["points"])
    np.testing.assert_allclose(actual, [[30, 20], [30, 70], [90, 20], [90, 70]], atol=1)
    block.execute({"image": np.zeros_like(image)})
    assert block.state == "pass" and block.last_result.outputs["points"] == []


def test_hough_lines_known_edge_and_empty_result():
    edges = np.zeros((100, 120), np.uint8)
    cv2.line(edges, (10, 30), (100, 30), 255, 1)
    block = run("inspect.hough_lines", {"image": edges}, {"votes": 20, "min_length": 40.0})
    assert block.last_result.outputs["count"] >= 1
    assert any(
        abs(y1 - 30) <= 1 and abs(y2 - 30) <= 1 and abs(x2 - x1) >= 40
        for x1, y1, x2, y2 in block.last_result.outputs["segments"]
    )
    block.execute({"image": np.zeros_like(edges)})
    assert block.last_result.outputs["count"] == 0


def test_hough_circle_known_center_and_radius():
    image = np.zeros((128, 128), np.uint8)
    cv2.circle(image, (64, 64), 25, 255, 2)
    image = cv2.GaussianBlur(image, (5, 5), 1)
    block = run(
        "inspect.hough_circles",
        {"image": image},
        {"dp": 1.0, "center_threshold": 15.0, "min_radius": 20, "max_radius": 30},
    )
    assert any(
        np.linalg.norm(np.array(c[:2]) - [64, 64]) < 3 and abs(c[2] - 25) < 3
        for c in block.last_result.outputs["circles"]
    )


def test_precise_distance_transform_is_pixel_metric_not_preview():
    mask = np.zeros((9, 9), np.uint8)
    mask[2:7, 2:7] = 255
    block = run("cv.distance_transform", {"image": mask}, {"mask_size": "precise"})
    distance = block.last_output_images["distance"]
    assert distance.dtype == np.float32 and distance[4, 4] == pytest.approx(3)
    assert distance[2, 2] == pytest.approx(1) and distance[0, 0] == 0
    assert block.last_output_images["preview"].dtype == np.uint8


@pytest.mark.parametrize(
    "type_id,parameters",
    [
        ("cv.gamma", {"gamma": 0.0}),
        ("cv.normalize", {"minimum": 1.0, "maximum": 0.0}),
        ("cv.clahe", {"tiles_x": 0}),
        ("inspect.corners", {"quality": 0.0}),
        ("inspect.hough_lines", {"rho": 0.0}),
        ("inspect.hough_circles", {"min_radius": 40, "max_radius": 20}),
    ],
)
def test_invalid_parameters_clear_previous_results(type_id, parameters):
    block = run(type_id, {"image": np.zeros((80, 80), np.uint8)})
    block.parameters.update(parameters)
    block.execute({"image": np.zeros((80, 80), np.uint8)})
    assert block.state == "error" and block.last_result is None


def test_channel_mask_and_distance_contract_errors():
    registry = builtin_registry()
    cases = [
        ("cv.split_channels", {"image": np.zeros((5, 5), np.uint8)}),
        (
            "cv.apply_mask",
            {"image": np.zeros((5, 5, 3), np.uint8), "mask": np.zeros((4, 5), np.uint8)},
        ),
        (
            "cv.merge_channels",
            {
                "channel_0": np.zeros((5, 5), np.uint8),
                "channel_1": np.zeros((4, 5), np.uint8),
                "channel_2": np.zeros((5, 5), np.uint8),
            },
        ),
        ("cv.distance_transform", {"image": np.ones((5, 5), np.uint8)}),
    ]
    for type_id, inputs in cases:
        block = registry.create(type_id)
        block.execute(inputs)
        assert block.state == "error", type_id
