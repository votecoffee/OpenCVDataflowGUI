import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


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


@pytest.mark.parametrize("selection_mode", ["samples_and_centers", "centers"])
@pytest.mark.parametrize("border", ["crop", "valid"])
def test_local_window_statistics_match_direct_masked_color_windows(selection_mode, border):
    mono = np.arange(30, dtype=np.uint16).reshape(5, 6)
    image = np.stack([mono, mono * 4, 1000 - mono], axis=2)
    mask = np.full(mono.shape, 255, np.uint8)
    mask[1, 3] = mask[3, 2] = 0
    block = run(
        "cv.local_statistics",
        {"image": image, "mask": mask},
        {
            "width": 4,
            "height": 3,
            "anchor": [2, 0],
            "border": border,
            "selection_mode": selection_mode,
        },
    )
    result = block.last_result
    for y, x in np.ndindex(mono.shape):
        x0, x1, y0, y1 = x - 2, x + 2, y, y + 3
        complete = x0 >= 0 and x1 <= 6 and y1 <= 5
        sample = image[y0:y1, max(0, x0) : x1]
        if selection_mode == "samples_and_centers":
            sample = sample[mask[y0:y1, max(0, x0) : x1] != 0]
        if mask[y, x] == 0 or (border == "valid" and not complete):
            sample = sample[:0]
        sample = sample.reshape(-1, 3)
        assert result.outputs["sample_count"][y, x] == len(sample)
        np.testing.assert_allclose(
            result.outputs["mean"][y, x],
            sample.mean(axis=0) if len(sample) else 0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.outputs["standard_deviation"][y, x],
            sample.std(axis=0) if len(sample) else 0,
            atol=1e-10,
        )
    assert result.graphics["local contrast"].shape == mono.shape


def test_local_window_statistics_preserve_low_contrast_at_large_float_offsets():
    x = np.arange(36).reshape(6, 6) % 3
    image = 1e12 + x * 0.25
    block = run("cv.local_statistics", {"image": image}, {"width": 3, "height": 3})
    result = block.last_result.outputs
    assert result["mean"][3, 3] == 1e12 + 0.25
    assert result["standard_deviation"][3, 3] == pytest.approx(np.std(x[2:5, 2:5] * 0.25))
    empty = block.execute({"image": image, "mask": np.zeros(image.shape, np.uint8)})
    assert empty is not None, block.error
    assert all(np.count_nonzero(output) == 0 for output in empty.outputs.values())


@pytest.mark.parametrize(
    "operation,values,parameters,expected",
    [
        ("log", [1, 2, 4, 8], {}, np.log([1, 2, 4, 8])),
        ("exp", [-2, 0, 1, 2], {}, np.exp([-2, 0, 1, 2])),
        ("sqrt", [0, 1, 4, 9], {}, [0, 1, 2, 3]),
        ("power", [-4, -2, 2, 4], {"exponent": 3}, [-64, -8, 8, 64]),
        ("power", [0, 1, 4, 9], {"exponent": 0.5}, [0, 1, 2, 3]),
        ("clip", [-2, 2, 4, 6], {"minimum": 1, "maximum": 4}, [1, 2, 4, 4]),
    ],
)
def test_intensity_transform_applies_raw_sample_math(operation, values, parameters, expected):
    image = np.array([values], dtype=np.float32)
    block = run("cv.intensity_transform", {"image": image}, {"operation": operation, **parameters})
    result = block.last_result
    assert result.outputs["image"].dtype == np.float64
    np.testing.assert_allclose(result.outputs["image"], [expected], rtol=1e-9)
    assert result.graphics["intensity"].shape == image.shape


def test_intensity_transform_checks_only_selected_samples_and_preserves_alpha():
    image = np.array([[[1, 2, 4, 0], [0, 4, 8, 16]]], dtype=np.uint16)
    mask = np.array([[255, 0]], dtype=np.uint8)
    block = run("cv.intensity_transform", {"image": image, "mask": mask}, {"operation": "log"})
    np.testing.assert_allclose(
        block.last_result.outputs["image"][0, 0], [0, np.log(2), np.log(4), 0]
    )
    np.testing.assert_array_equal(block.last_result.outputs["image"][0, 1], image[0, 1])
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid" and "positive samples" in block.error


def test_intensity_transform_saturates_integer_output_and_rejects_invalid_domains():
    image = np.array([[2, 5, 20]], dtype=np.uint8)
    block = run("cv.intensity_transform", {"image": image}, {"exponent": 2, "output_depth": "same"})
    output = block.last_result.outputs["image"]
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, [[4, 25, 255]])
    block.parameters["exponent"] = -1
    assert block.execute({"image": np.array([[0]], np.uint8)}) is None
    assert block.error_kind == "invalid" and "nonzero samples" in block.error
    block.parameters["exponent"] = 0.5
    assert block.execute({"image": np.array([[-1]], np.int16)}) is None
    assert block.error_kind == "invalid" and "whole exponent" in block.error
    block.parameters.update(operation="exp", output_depth="float32")
    assert block.execute({"image": np.array([[100]], np.uint8)}) is None
    assert block.error_kind == "invalid" and "float32 range" in block.error
    assert block.execute({"image": np.array([[1000]], np.uint16)}) is None
    assert block.error_kind == "invalid" and "overflowed" in block.error


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("add", [6, 6, 6, 6]),
        ("subtract", [-2, 2, 6, -6]),
        ("multiply", [16, 16, 0, 0]),
        ("divide", [1, 4, 0, 0]),
        ("minimum", [2, 2, 0, 0]),
        ("maximum", [4, 4, 6, 6]),
        ("absolute_difference", [2, 2, 6, 6]),
    ],
)
def test_image_arithmetic_applies_two_images_in_raw_sample_units(operation, expected):
    a = np.array([[2, 4, 6, 0]], dtype=np.uint8)
    b = np.array([[4, 2, 0, 6]], dtype=np.uint16)
    block = run(
        "cv.image_arithmetic",
        {"image_a": a, "image_b": b},
        {"operation": operation, "scale": 2, "zero_divisor": "zero"},
    )
    assert block.last_result.outputs["image"].dtype == np.float64
    np.testing.assert_allclose(block.last_result.outputs["image"], [expected])


def test_image_arithmetic_preserves_alpha_selection_and_channel_encoding():
    a = ImageArray(np.array([[[100, 80, 60, 32], [4, 3, 2, 1]]], np.uint8), "rgba")
    b = ImageArray(np.array([[[5, 10, 20], [20, 50, 100]]], np.uint16), "bgr")
    mask = np.array([[255, 0]], np.uint8)
    block = run(
        "cv.image_arithmetic",
        {"image_a": a, "image_b": b, "mask": mask},
        {"operation": "add", "output_depth": "same"},
    )
    output = block.last_result.outputs["image"]
    assert output.dtype == np.uint8 and image_format(output) == "rgba"
    np.testing.assert_array_equal(output, [[[120, 90, 65, 32], [4, 3, 2, 1]]])
    mono = np.array([[200, 0]], dtype=np.uint16)
    result = block.execute({"image_a": a, "image_b": mono})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[[255, 255, 255, 32], [4, 3, 2, 1]]])


def test_image_arithmetic_checks_frames_zero_divisors_and_operand_conflicts():
    a, b = (np.array([values], np.uint8) for values in ([12, 36], [2, 0]))
    ctx = frame([2, 1], "a")
    inputs = {
        "image_a": ValuePacket(a, {"spatial": ctx}),
        "image_b": ValuePacket(b, {"spatial": frame([2, 1], "b")}),
    }
    block = builtin_registry().create("cv.image_arithmetic", parameters={"operation": "divide"})
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and "pixel frame" in block.error
    block.parameters["alignment"] = "pixel_grid"
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and "nonzero divisors" in block.error
    selected = block.execute({**inputs, "mask": np.array([[255, 0]], np.uint8)})
    assert selected is not None, block.error
    np.testing.assert_array_equal(selected.outputs["image"], [[6, 36]])
    block.parameters["zero_divisor"] = "keep_a"
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.payloads["image"]["spatial"] == ctx
    np.testing.assert_array_equal(result.outputs["image"], [[6, 36]])
    result = block.execute({"image_a": a, "scalar_b": 4})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[3, 9]])
    assert block.execute({**inputs, "scalar_b": 4}) is None
    assert "either Image B or Scalar B" in block.error
