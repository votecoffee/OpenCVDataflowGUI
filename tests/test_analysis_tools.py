import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


def run(kind, inputs, parameters=None):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block


def test_box_and_separable_filter_controls():
    image = np.arange(80, dtype=np.uint8).reshape(8, 10)
    box = run(
        "cv.box_filter",
        {"image": image},
        {
            "width": 3,
            "height": 5,
            "normalize": False,
            "output_depth": "float32",
            "border": "replicate",
            "anchor": [0, 1],
        },
    )
    expected = cv2.boxFilter(
        image, cv2.CV_32F, (3, 5), anchor=(0, 1), normalize=False, borderType=cv2.BORDER_REPLICATE
    )
    np.testing.assert_array_equal(box.last_output_images["image"], expected)
    separable = run(
        "cv.separable_filter",
        {"image": image},
        {
            "kernel_x": [-1.0, 0.0, 1.0],
            "kernel_y": [1.0, 2.0, 1.0],
            "output_depth": "int16",
            "delta": 2.0,
            "border": "replicate",
        },
    )
    expected = cv2.sepFilter2D(
        image,
        cv2.CV_16S,
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
        delta=2,
        borderType=cv2.BORDER_REPLICATE,
    )
    np.testing.assert_array_equal(separable.last_output_images["image"], expected)


@pytest.mark.parametrize("method", ["harris", "min_eigen"])
def test_corner_response_and_subpixel_refinement(method):
    image = np.zeros((100, 120), np.uint8)
    image[20:71, 30:91] = 255
    response = run("inspect.corner_response", {"image": image}, {"method": method, "aperture": "5"})
    assert response.last_output_images["response"].dtype == np.float32
    assert response.last_output_images["preview"].max() == 255
    refined = run(
        "inspect.refine_corners",
        {"image": image, "points": [[31.0, 21.0], [89.0, 69.0]]},
        {"termination": "both", "epsilon": 0.001, "window_x": 5, "window_y": 5},
    )
    np.testing.assert_allclose(
        refined.last_result.outputs["points"], [[29.56, 19.56], [90.44, 70.44]], atol=0.1
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_phase_correlation_recovers_known_shift(precision):
    image = np.random.default_rng(4).normal(size=(100, 120)).astype(precision)
    moved = np.roll(np.roll(image, 7, axis=0), -11, axis=1)
    result = run(
        "inspect.phase_correlation", {"image_a": image, "image_b": moved}, {"precision": precision}
    )
    assert result.state == "pass"
    shift = result.last_result.outputs["shift"]
    assert shift["x"] == pytest.approx(-11, abs=0.01)
    assert shift["y"] == pytest.approx(7, abs=0.01)


@pytest.mark.parametrize("model", ["full", "partial"])
def test_affine_models_direction_and_warp_connection(model):
    source = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0], [40.0, 60.0], [70.0, 30.0]]
    )
    matrix = cv2.getRotationMatrix2D((50.0, 50.0), 8.0, 1.1)
    target = cv2.transform(source[:, None, :], matrix).reshape(-1, 2)
    result = run(
        "inspect.affine",
        {"matches": {"query_points": source.tolist(), "train_points": target.tolist()}},
        {"model": model, "method": "ransac", "refine_iterations": 15},
    )
    assert result.state == "pass"
    transform = result.last_result.outputs["transform"]
    np.testing.assert_allclose(np.array(transform["source_to_target"])[:2], matrix, atol=1e-4)
    warped = run(
        "cv.warp_affine", {"image": np.zeros((100, 100), np.uint8), "transform": transform}
    )
    assert warped.state == "pass"


def test_remap_native_coordinate_fields_and_polar_units():
    image = np.arange(80, dtype=np.uint8).reshape(8, 10)
    y, x = np.indices(image.shape, dtype=np.float32)
    mapped = run(
        "cv.remap",
        {"image": image, "map_x": x + 1, "map_y": y},
        {"interpolation": "nearest", "border": "constant", "border_value": 55.0},
    )
    np.testing.assert_array_equal(mapped.last_output_images["image"][:, :-1], image[:, 1:])
    np.testing.assert_array_equal(mapped.last_output_images["image"][:, -1], 55)
    polar = run(
        "cv.cart_to_polar",
        {"x": np.array([[3.0, 0.0]], np.float32), "y": np.array([[4.0, 2.0]], np.float32)},
        {"angle_units": "degrees"},
    )
    np.testing.assert_allclose(polar.last_output_images["magnitude"], [[5, 2]], atol=1e-5)
    np.testing.assert_allclose(polar.last_output_images["angle"], [[53.13, 90]], atol=0.02)


def test_new_tools_reject_invalid_contracts_and_clear_results():
    registry = builtin_registry()
    for kind, inputs, parameters in [
        ("cv.box_filter", {"image": np.zeros((8, 8), np.uint8)}, {"width": 0}),
        ("cv.separable_filter", {"image": np.zeros((8, 8), np.uint8)}, {"kernel_x": []}),
        (
            "inspect.phase_correlation",
            {"image_a": np.zeros((8, 8), np.uint8), "image_b": np.zeros((9, 8), np.uint8)},
            {},
        ),
        (
            "cv.remap",
            {
                "image": np.zeros((8, 8), np.uint8),
                "map_x": np.zeros((8, 8), np.float32),
                "map_y": np.zeros((9, 8), np.float32),
            },
            {},
        ),
        (
            "inspect.refine_corners",
            {"image": np.zeros((8, 8), np.uint8), "points": [[-1.0, 0.0]]},
            {},
        ),
    ]:
        block = registry.create(kind, parameters=parameters)
        block.execute(inputs)
        assert block.state == "error" and block.last_result is None, kind
