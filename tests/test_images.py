import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import (
    ImageArray,
    convert_depth,
    format_image,
    image_format,
    opencv_image,
    preview_rgb,
)


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(encoded.tobytes())


@pytest.mark.parametrize("color", ["bgr", "rgb", "bgra", "rgba", "gray", "hsv", "lab"])
@pytest.mark.parametrize("depth", ["uint8", "uint16", "float32"])
def test_file_formats_and_depths(tmp_path, color, depth):
    # Deliberately unequal channels make RGB/BGR and preview errors visible.
    source = np.array([[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [31, 87, 191]]], np.uint8)
    path = tmp_path / "source.png"
    save_image(path, source)
    block = builtin_registry().create(
        "input.file", parameters={"path": str(path), "color_format": color, "pixel_depth": depth}
    )
    block.execute({})
    if depth == "uint16" and color in ("hsv", "lab"):
        assert block.state == "error" and "require uint8 or float32" in block.error
        return
    assert block.state == "pass", block.error
    output = block.last_output_images["image"]
    assert output.dtype == np.dtype(depth)
    assert image_format(output) == color
    assert image_format(copy.deepcopy(output)) == color
    info = block.last_result.outputs["info"]
    assert info["width"] == 2 and info["height"] == 2
    assert info["bits_per_channel"] == output.dtype.itemsize * 8
    if color == "gray":
        expected = cv2.cvtColor(convert_depth(source, depth), cv2.COLOR_BGR2GRAY)
        np.testing.assert_allclose(output, expected)
        assert output.ndim == 2
    else:
        # Lab8 is quantized and not exactly invertible, especially at extreme colors.
        tolerance = (
            (10 if color == "lab" else 3) if color in ("lab", "hsv") and depth == "uint8" else 1
        )
        np.testing.assert_allclose(preview_rgb(output), source[..., ::-1], atol=tolerance)
    np.testing.assert_array_equal(block.last_graphics["decoded source"], source)


def test_uint16_decode_preservation_and_fixed_range_scaling(tmp_path):
    source = np.array([[0, 1, 255, 256, 32768, 65535]], np.uint16)
    path = tmp_path / "depth16.png"
    save_image(path, source)
    block = builtin_registry().create(
        "input.file",
        parameters={"path": str(path), "color_format": "unchanged", "pixel_depth": "unchanged"},
    )
    block.execute({})
    np.testing.assert_array_equal(block.last_output_images["image"], source)
    assert block.last_output_images["image"].dtype == np.uint16
    assert image_format(block.last_output_images["image"]) == "gray"
    block.parameters["pixel_depth"] = "uint8"
    block.execute({})
    np.testing.assert_array_equal(block.last_output_images["image"], [[0, 0, 1, 1, 128, 255]])
    block.parameters["pixel_depth"] = "float32"
    block.execute({})
    np.testing.assert_allclose(block.last_output_images["image"], source.astype(np.float32) / 65535)
    constant = np.full((2, 2), 32768, np.uint16)
    assert np.all(preview_rgb(constant) == 128)  # no contrast stretch of constant images


def test_alpha_preservation_and_channel_reordering(tmp_path):
    source = np.array([[[10, 20, 30, 0], [40, 50, 60, 127]]], np.uint8)
    path = tmp_path / "alpha.png"
    save_image(path, source)
    block = builtin_registry().create(
        "input.file",
        parameters={"path": str(path), "color_format": "rgba", "pixel_depth": "uint16"},
    )
    block.execute({})
    assert block.state == "pass", block.error
    np.testing.assert_array_equal(
        block.last_output_images["image"], source[..., [2, 1, 0, 3]].astype(np.uint16) * 257
    )
    block.parameters.update(color_format="unchanged", pixel_depth="unchanged")
    block.execute({})
    np.testing.assert_array_equal(block.last_output_images["image"], source)
    assert image_format(block.last_output_images["image"]) == "bgra"
    # Adding alpha uses the full range of the selected pixel depth.
    converted = format_image(source[..., :3], "rgba", "float32")
    assert np.all(converted[..., 3] == 1)


def test_float_tiff_and_saturating_integer_conversion(tmp_path):
    source = np.array([[-0.1, 0.0, 0.5, 1.0, 1.5]], np.float32)
    path = tmp_path / "floating.tiff"
    save_image(path, source)
    block = builtin_registry().create(
        "input.file",
        parameters={"path": str(path), "color_format": "gray", "pixel_depth": "unchanged"},
    )
    block.execute({})
    assert block.state == "pass", block.error
    np.testing.assert_array_equal(block.last_output_images["image"], source)
    block.parameters["pixel_depth"] = "uint8"
    block.execute({})
    np.testing.assert_array_equal(block.last_output_images["image"], [[0, 0, 128, 255, 255]])


@pytest.mark.parametrize("encoding", ["rgb", "hsv", "lab"])
def test_encoding_survives_graph_snapshots_and_is_interpreted_by_tools(tmp_path, encoding):
    source = np.zeros((10, 10, 3), np.uint8)
    source[:, :5] = (10, 30, 240)
    source[:, 5:] = (230, 40, 20)
    path = tmp_path / "source.png"
    save_image(path, source)
    registry = builtin_registry()
    graph = Graph()
    load = graph.add(
        registry.create(
            "input.file",
            parameters={"path": str(path), "color_format": encoding, "pixel_depth": "uint8"},
        )
    )
    gray = graph.add(registry.create("gapi.gray"))
    graph.connect(load.id, "image", gray.id, "image")
    graph.run()
    assert gray.state == "pass", gray.error
    expected = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    np.testing.assert_allclose(gray.last_output_images["image"], expected, atol=3)
    assert image_format(gray.last_inputs["image"]) == encoding
    restored = Graph.from_dict(graph.to_dict(), registry)
    restored.run()
    assert restored.blocks[load.id].parameters["color_format"] == encoding
    # Image saving also honors the output encoding rather than swapping channels.
    saved_path = tmp_path / "saved.png"
    save = registry.create("output.image", parameters={"path": str(saved_path)})
    save.execute({"image": load.last_output_images["image"]})
    assert save.state == "pass", save.error
    np.testing.assert_array_equal(
        cv2.imdecode(np.frombuffer(saved_path.read_bytes(), np.uint8), 1),
        opencv_image(load.last_output_images["image"]),
    )


def test_gapi_depth_errors_are_explicit():
    registry = builtin_registry()
    block = registry.create("gapi.gray")
    block.execute({"image": np.ones((2, 2, 3), np.float32)})
    assert block.state == "error" and "requires pixel depth uint8" in block.error


def test_invalid_path_settings_and_old_project_defaults(tmp_path):
    registry = builtin_registry()
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    block = registry.create("input.file", parameters={"path": str(path)})
    block.execute({})
    assert block.state == "error" and "decode" in block.error
    save_image(path, np.full((2, 2), 128, np.uint8))
    block.parameters["color_format"] = "invalid"
    block.execute({})
    assert "Color format must" in block.error
    graph = Graph()
    graph.add(registry.create("input.file"))
    data = graph.to_dict()
    data["blocks"][0]["parameters"] = {"path": str(path)}
    data["blocks"][0]["exposed_outputs"] = ["image"]
    restored = Graph.from_dict(data, registry)
    restored.run()
    result = next(iter(restored.blocks.values())).last_output_images["image"]
    assert result.dtype == np.uint8 and result.shape == (2, 2, 3)
    with pytest.raises(ValueError, match="nonfinite"):
        convert_depth(np.full((2, 2), np.nan, np.float32), "uint8")
    assert image_format(ImageArray(np.zeros((2, 2)), "gray")) == "gray"
