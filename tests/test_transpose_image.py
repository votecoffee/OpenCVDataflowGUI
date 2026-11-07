"""Check exact diagonal transforms and composed pixel coordinates."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import frame, metadata


@pytest.mark.parametrize("diagonal", ("main", "anti"))
@pytest.mark.parametrize(
    ("shape", "dtype", "encoding"),
    (
        ((3, 5), np.uint16, "gray"),
        ((5, 1), np.int32, "gray"),
        ((3, 5, 3), np.float64, "rgb"),
        ((1, 5, 4), np.uint8, "rgba"),
    ),
)
def test_transpose_copies_samples_and_maps_every_output_pixel(diagonal, shape, dtype, encoding):
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    block = builtin_registry().create("cv.transpose_image", parameters={"diagonal": diagonal})
    result = block.execute({"image": ImageArray(pixels, encoding)})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    expected = pixels.swapaxes(0, 1)
    if diagonal == "anti":
        expected = expected[::-1, ::-1]
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == dtype and output.color_format == encoding
    geometry = result.outputs["geometry"]
    assert geometry["input_size"] == list(shape[:2][::-1])
    assert geometry["output_size"] == list(shape[:2])
    matrix = np.asarray(geometry["output_to_input"], dtype=int)
    for y, x in np.ndindex(output.shape[:2]):
        sx, sy, _ = matrix @ [x, y, 1]
        np.testing.assert_array_equal(output[y, x], pixels[sy, sx])


def test_physical_depth_keeps_missing_samples_units_and_display_range():
    source = DepthArray([[np.nan, 1, 2], [0, 4, np.nan]], units="m", display_min=1, display_max=9)
    block = builtin_registry().create("cv.transpose_image")
    result = block.execute({"image": source})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    assert isinstance(output, DepthArray)
    assert (output.units, output.display_min, output.display_max) == ("m", 1, 9)
    np.testing.assert_array_equal(output, source.T)


def test_transposing_twice_restores_source_and_pyramid_coordinates():
    spatial = frame([4, 3], "camera")
    spatial["root_size"] = [8, 6]
    spatial["to_root"] = np.diag([2, 2, 1]).tolist()
    pyramid = {"level": 1, "largest_size": [8, 6], "to_largest": spatial["to_root"]}
    source = ImageArray(
        np.arange(12, dtype=np.uint8).reshape(3, 4),
        "gray",
        {"spatial": spatial, "pyramid": pyramid},
    )
    block = builtin_registry().create("cv.transpose_image", parameters={"diagonal": "anti"})
    result = block.execute({"image": source})
    assert result is not None and result.passed, block.error
    first = result.outputs["image"]
    first_payload = metadata(first)
    transform = np.asarray(result.outputs["geometry"]["output_to_input"])
    np.testing.assert_array_equal(
        first_payload["pyramid"]["to_largest"], np.asarray(pyramid["to_largest"]) @ transform
    )
    result = block.execute({"image": first})
    assert result is not None and result.passed, block.error
    restored = result.outputs["image"]
    np.testing.assert_array_equal(restored, source)
    assert metadata(restored)["spatial"] == spatial
    assert metadata(restored)["pyramid"] == pyramid


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_maps_transposed_crop_points_back_to_full_source(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "transpose_image"
        / "transpose_image.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    cropped = graph.blocks["crop"].last_result.outputs["image"]
    output = graph.blocks["transpose"].last_result.outputs["image"]
    np.testing.assert_array_equal(output, cropped.swapaxes(0, 1)[::-1, ::-1])
    points = graph.blocks["points"].last_result.outputs["points"]
    np.testing.assert_array_equal(points, [[414, 279], [255, 80], [334, 248]])
    source = graph.blocks["source"].last_result.outputs["image"]
    for (x, y), (sx, sy) in zip(
        graph.blocks["points"].parameters["pixel_points"], points, strict=True
    ):
        np.testing.assert_array_equal(output[y, x], source[int(sy), int(sx)])
