import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format, preview_rgb


@pytest.fixture
def registry():
    return builtin_registry()


def execute(registry, type_id, inputs, parameters=None):
    block = registry.create(type_id, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block


def test_convert_rgb_uint16_grayscale_and_unchanged_encoding(registry):
    rgb = ImageArray(np.array([[[65535, 0, 0], [0, 0, 65535]]], np.uint16), "rgb")
    converted = execute(
        registry, "cv.convert", {"image": rgb}, {"color_format": "gray", "pixel_depth": "uint8"}
    )
    np.testing.assert_array_equal(converted.last_output_images["image"], [[76, 29]])
    hsv = ImageArray(np.array([[[0, 255, 255]]], np.uint8), "hsv")
    preserved = execute(registry, "cv.convert", {"image": hsv}, {"color_format": "unchanged"})
    assert image_format(preserved.last_output_images["image"]) == "hsv"
    np.testing.assert_array_equal(
        preview_rgb(preserved.last_output_images["image"]), [[[255, 0, 0]]]
    )
    preserved.parameters["pixel_depth"] = "float32"
    preserved.execute({"image": hsv})
    assert preserved.state == "error" and "explicit color format" in preserved.error


def test_crop_pixels_metadata_and_bounds(registry):
    image = ImageArray(np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3), "rgb")
    block = execute(
        registry, "cv.crop", {"image": image}, {"x": 2, "y": 1, "width": 3, "height": 4}
    )
    output = block.last_output_images["image"]
    np.testing.assert_array_equal(output, image[1:5, 2:5])
    assert image_format(output) == "rgb"
    geometry = block.last_result.outputs["geometry"]
    np.testing.assert_array_equal(np.array(geometry["output_to_input"]) @ [0, 0, 1], [2, 1, 1])
    block.parameters["width"] = 7
    block.execute({"image": image})
    assert block.state == "error" and not block.last_output_images


@pytest.mark.parametrize("rotation", ["clockwise", "counterclockwise", "180"])
def test_rotation_pixel_coordinate_transforms(registry, rotation):
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    block = execute(registry, "cv.rotate90", {"image": image}, {"rotation": rotation})
    output = block.last_output_images["image"]
    transform = np.array(block.last_result.outputs["geometry"]["output_to_input"])
    for y, x in [(0, 0), (output.shape[0] - 1, output.shape[1] - 1), (1, 2)]:
        source_x, source_y, _ = (transform @ [x, y, 1]).astype(int)
        assert output[y, x] == image[source_y, source_x]


@pytest.mark.parametrize("direction", ["horizontal", "vertical", "both"])
def test_flip_pixel_coordinate_transforms(registry, direction):
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    block = execute(registry, "cv.flip", {"image": image}, {"direction": direction})
    output = block.last_output_images["image"]
    transform = np.array(block.last_result.outputs["geometry"]["output_to_input"])
    for y, x in [(0, 0), (4, 5), (1, 2)]:
        source_x, source_y, _ = (transform @ [x, y, 1]).astype(int)
        assert output[y, x] == image[source_y, source_x]


def test_resize_dimensions_interpolation_and_pixel_center_geometry(registry):
    image = np.array([[10, 20], [30, 40]], np.uint16)
    block = execute(
        registry,
        "cv.resize",
        {"image": image},
        {"width": 4, "height": 4, "interpolation": "nearest"},
    )
    np.testing.assert_array_equal(
        block.last_output_images["image"], np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    )
    matrix = np.array(block.last_result.outputs["geometry"]["output_to_input"])
    np.testing.assert_allclose(matrix @ [0.5, 0.5, 1], [0, 0, 1])


@pytest.mark.parametrize(
    "type_id,reference",
    [
        ("cv.box_blur", lambda a: cv2.blur(a, (5, 5))),
        ("cv.median_blur", lambda a: cv2.medianBlur(a, 3)),
        ("cv.bilateral", lambda a: cv2.bilateralFilter(a, 5, 50, 5)),
    ],
)
def test_filter_wrapper_numeric_results_and_input_preservation(registry, type_id, reference):
    image = np.random.default_rng(7).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    original = image.copy()
    tagged = ImageArray(image, "rgb")
    block = execute(registry, type_id, {"image": tagged})
    np.testing.assert_array_equal(block.last_output_images["image"], reference(image))
    np.testing.assert_array_equal(image, original)
    assert image_format(block.last_output_images["image"]) == "rgb"


def test_auto_threshold_uint16_and_adaptive_threshold(registry):
    image = np.full((30, 40), 1000, np.uint16)
    image[:, 20:] = 50000
    block = execute(registry, "cv.auto_threshold", {"image": image})
    assert block.last_result.outputs["threshold"] == 1000
    np.testing.assert_array_equal(block.last_output_images["image"][:, 20:], 65535)
    np.testing.assert_array_equal(block.last_output_images["image"][:, :20], 0)
    source = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40).astype(np.uint8)
    adaptive = execute(registry, "cv.adaptive_threshold", {"image": source})
    expected = cv2.adaptiveThreshold(
        source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    np.testing.assert_array_equal(adaptive.last_output_images["image"], expected)


def test_in_range_honors_hsv_channels_and_single_bound_broadcast(registry):
    hsv = ImageArray(np.array([[[0, 255, 255], [60, 255, 255], [120, 255, 255]]], np.uint8), "hsv")
    block = execute(
        registry, "cv.in_range", {"image": hsv}, {"lower": [0, 200, 200], "upper": [10, 255, 255]}
    )
    np.testing.assert_array_equal(block.last_output_images["image"], [[255, 0, 0]])
    broadcast = execute(registry, "cv.in_range", {"image": hsv})
    np.testing.assert_array_equal(broadcast.last_output_images["image"], [[255, 255, 255]])


@pytest.mark.parametrize(
    "operation,code",
    [
        ("erode", cv2.MORPH_ERODE),
        ("dilate", cv2.MORPH_DILATE),
        ("open", cv2.MORPH_OPEN),
        ("close", cv2.MORPH_CLOSE),
        ("gradient", cv2.MORPH_GRADIENT),
        ("tophat", cv2.MORPH_TOPHAT),
        ("blackhat", cv2.MORPH_BLACKHAT),
    ],
)
def test_morphology_operations(registry, operation, code):
    mask = np.zeros((30, 40), np.uint8)
    mask[5:20, 5:20] = 255
    mask[10, 10] = 0
    mask[25, 30] = 255
    block = execute(registry, "cv.morphology", {"image": mask}, {"operation": operation})
    expected = cv2.morphologyEx(mask, code, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    np.testing.assert_array_equal(block.last_output_images["image"], expected)


def test_image_math_masks_blend_and_missing_inputs(registry):
    a = np.array([[0, 255], [255, 0]], np.uint8)
    b = np.array([[255, 255], [0, 0]], np.uint8)
    for operation, reference in (
        ("and", np.bitwise_and),
        ("or", np.bitwise_or),
        ("xor", np.bitwise_xor),
    ):
        block = execute(
            registry, "cv.bitwise", {"image_a": a, "image_b": b}, {"operation": operation}
        )
        np.testing.assert_array_equal(block.last_output_images["image"], reference(a, b))
    inverted = execute(registry, "cv.bitwise", {"image_a": a}, {"operation": "not"})
    np.testing.assert_array_equal(inverted.last_output_images["image"], 255 - a)
    blended = execute(
        registry,
        "cv.blend",
        {"image_a": np.full((4, 4), 100, np.uint8), "image_b": np.full((4, 4), 200, np.uint8)},
    )
    assert np.all(blended.last_output_images["image"] == 150)
    missing = registry.create("cv.bitwise")
    missing.execute({"image_a": a})
    assert "requires image_b" in missing.error


@pytest.mark.parametrize("operator", ["sobel", "scharr", "laplacian"])
def test_gradient_retains_signed_float_and_separate_preview(registry, operator):
    image = np.zeros((16, 24), np.uint16)
    image[4:12, 8:16] = 65535
    block = execute(registry, "cv.gradient", {"image": image}, {"operator": operator})
    gradient = block.last_output_images["gradient"]
    assert gradient.dtype == np.float32
    assert gradient.min() < 0 < gradient.max()
    preview = block.last_output_images["preview"]
    assert preview.dtype == np.uint8 and preview.max() == 255


def test_components_measurements_filtering_and_valid_failure(registry):
    image = np.zeros((40, 60), np.uint8)
    image[5:15, 5:25] = 255
    image[20:30, 35:45] = 255
    image[1, 1] = 255
    block = execute(registry, "inspect.components", {"image": image}, {"min_area": 20})
    outputs = block.last_result.outputs
    assert outputs["count"] == 2
    assert [c["area"] for c in outputs["data"]["components"]] == [200, 100]
    np.testing.assert_allclose(outputs["centroids"], [[14.5, 9.5], [39.5, 24.5]])
    assert outputs["mask"][1, 1] == 0
    block.parameters["expected_count"] = 3
    block.execute({"image": image})
    assert block.state == "fail" and block.last_result.outputs["count"] == 2


@pytest.mark.parametrize("method", ["ccoeff_normed", "ccorr_normed", "sqdiff_normed"])
def test_template_match_finds_known_location_and_quality(registry, method):
    template = np.random.default_rng(5).integers(30, 230, (9, 12), dtype=np.uint8)
    image = np.zeros((60, 80), np.uint8)
    image[23:32, 17:29] = template
    block = execute(
        registry,
        "inspect.match_template",
        {"image": image, "template": template},
        {"method": method, "min_score": 0.99},
    )
    assert block.state == "pass"
    assert block.last_result.outputs["score"] > 0.999
    assert block.last_result.outputs["location"]["top_left"] == [17, 23]
    assert block.last_result.outputs["center"] == [[22.5, 27.0]]
    assert set(block.last_graphics) == {"match", "response"}


@pytest.mark.parametrize(
    "type_id,parameters",
    [
        ("cv.median_blur", {"kernel": 4}),
        ("cv.resize", {"width": 0}),
        ("cv.morphology", {"iterations": 0}),
        ("cv.adaptive_threshold", {"block_size": 2}),
        ("cv.in_range", {"lower": [255], "upper": [0]}),
        ("cv.gradient", {"dx": 0, "dy": 0}),
        ("cv.box_blur", {"kernel": 2}),
    ],
)
def test_invalid_settings_fail_without_stale_outputs(registry, type_id, parameters):
    block = execute(registry, type_id, {"image": np.zeros((20, 30), np.uint8)})
    block.parameters.update(parameters)
    block.execute({"image": np.zeros((20, 30), np.uint8)})
    assert block.state == "error" and block.last_result is None and not block.last_output_images


def test_template_rejects_constant_or_oversize_reference(registry):
    block = registry.create("inspect.match_template")
    block.execute({"image": np.zeros((10, 10), np.uint8), "template": np.ones((3, 3), np.uint8)})
    assert "nonconstant" in block.error
    block.execute({"image": np.zeros((10, 10), np.uint8), "template": np.ones((13, 13), np.uint8)})
    assert "fit inside" in block.error
