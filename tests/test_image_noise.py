"""Exercise deterministic sources and selected noise through the graph runtime."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import frame, metadata


def test_random_image_uses_wired_seed_and_inclusive_integer_bounds():
    block = builtin_registry().create(
        "input.random_image",
        parameters={
            "output_depth": "uint16",
            "color_format": "rgb",
            "minimum": 65534,
            "maximum": 65535,
            "width": 19,
            "height": 17,
        },
    )
    seed = 2**32 - 1
    first = block.execute({"seed": seed})
    assert first is not None, block.error
    image = first.outputs["image"]
    assert first.outputs["seed"] == seed
    assert image.shape == (17, 19, 3) and image.dtype == np.uint16
    assert image_format(image) == "rgb"
    assert set(np.unique(image)) == {65534, 65535}
    np.testing.assert_array_equal(block.execute({"seed": seed}).outputs["image"], image)
    assert np.any(block.execute({"seed": 0}).outputs["image"] != image)


def test_random_float_image_draws_signed_gaussian_samples_in_stored_units():
    block = builtin_registry().create(
        "input.random_image",
        parameters={
            "output_depth": "float64",
            "distribution": "gaussian",
            "mean": -2.0,
            "standard_deviation": 3.0,
            "width": 125,
            "height": 80,
            "seed": 61,
        },
    )
    result = block.execute({})
    assert result is not None, block.error
    image = result.outputs["image"]
    assert image.dtype == np.float64
    assert float(image.mean()) == pytest.approx(-2.0, abs=0.15)
    assert float(image.std()) == pytest.approx(3.0, abs=0.15)


@pytest.mark.parametrize(
    ("parameters", "inputs", "error"),
    [
        ({"minimum": 1.1, "maximum": 1.9}, {}, "include an integer"),
        ({"minimum": 0, "maximum": 256}, {}, "fit the output depth"),
        ({"minimum": 5, "maximum": 2}, {}, "ordered range"),
        ({"width": 8192, "height": 8192}, {}, "limited to"),
        ({}, {"seed": 0.5}, "whole number"),
    ],
)
def test_random_source_rejects_invalid_settings_before_draw(parameters, inputs, error):
    block = builtin_registry().create("input.random_image", parameters=parameters)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and error in block.error


def test_selected_noise_preserves_encoding_alpha_unselected_pixels_and_frame():
    context = frame([3, 2], "camera")
    values = np.full((2, 3, 4), (248, 8, 3, 100), np.uint8)
    image = ImageArray(values, "rgba", {"spatial": context})
    mask = np.array([[0, 0, 255], [255, 255, 255]], dtype=np.uint8)
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [3, 2],
        "rectangle": [1, 0, 2, 2],
    }
    block = builtin_registry().create(
        "cv.add_image_noise",
        parameters={"distribution": "uniform", "minimum": 12, "maximum": 12},
    )
    result = block.execute({"image": image, "mask": mask, "region": region})
    assert result is not None, block.error
    output = result.outputs["image"]
    expected = values.copy()
    expected[0, 2] = expected[1, 1] = expected[1, 2] = (255, 20, 15, 100)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(image, values)
    assert output.dtype == np.uint8 and image_format(output) == "rgba"
    assert metadata(output)["spatial"] == context


def test_noise_example_is_repeatable_across_serial_and_parallel_branches():
    path = Path(__file__).parents[1] / "examples/image_processing/seeded_noise/seeded_noise.json"
    graph = Graph.load(path, builtin_registry())
    try:
        assert graph.branch_workers == 2
        graph.run()
        first = graph.blocks["noise_a"].last_result.outputs["image"]
        assert graph.blocks["compare"].last_result.outputs["value"] == 0
        graph.set_branch_workers(1)
        graph.run()
        np.testing.assert_array_equal(graph.blocks["noise_a"].last_result.outputs["image"], first)
        np.testing.assert_array_equal(graph.blocks["noise_b"].last_result.outputs["image"], first)
        graph.blocks["noise_b"].parameters["seed"] += 1
        graph.run()
        assert graph.blocks["compare"].last_result.outputs["value"] > 0
        assert graph.blocks["compare"].state == "fail"
    finally:
        graph.close()
