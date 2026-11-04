import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray, preview_rgb
from opencv_dataflow.toolblocks import encapsulate


@pytest.mark.parametrize(
    "encoding,labels", [("bgr", ["B", "G", "R"]), ("rgb", ["R", "G", "B"]), ("gray", ["Gray"])]
)
def test_histogram_channel_counts(encoding, labels):
    pixels = np.array([[0, 0, 255], [255, 128, 128]], np.uint8)
    image = pixels if encoding == "gray" else np.stack((pixels, pixels // 2, pixels // 3), axis=2)
    block = builtin_registry().create("inspect.histogram")
    block.execute({"image": ImageArray(image, encoding)})
    assert block.state == "pass", block.error
    result = block.last_result.outputs["histogram"]
    assert [c["channel"] for c in result["channels"]] == labels
    for index, channel in enumerate(result["channels"]):
        expected = pixels if encoding == "gray" else image[..., index]
        np.testing.assert_array_equal(
            channel["counts"], np.bincount(expected.ravel(), minlength=256)
        )
        assert sum(channel["counts"]) == 6
    assert block.last_output_images["plot"].shape == (300, 600, 3)


def test_histogram_float_nan_manual_range_and_alpha():
    block = builtin_registry().create(
        "inspect.histogram",
        parameters={"range_mode": "manual", "range_min": 0.0, "range_max": 1.0, "bins": 2},
    )
    block.execute({"image": np.array([[np.nan, -1, 0, 0.5, 1, 2]], np.float32)})
    assert block.state == "pass", block.error
    channel = block.last_result.outputs["histogram"]["channels"][0]
    assert channel["counts"] == [1, 2] and channel["outside_range"] == 2
    block.parameters.update(range_mode="automatic", include_alpha=True)
    block.execute({"image": np.zeros((2, 2, 4), np.uint16)})
    assert len(block.last_result.outputs["histogram"]["channels"]) == 4


def test_depth_units_pairing_sampling_and_nested_snapshots(tmp_path):
    path = tmp_path / "depth.png"
    raw = np.array([[0, 1000], [2000, 3000]], np.uint16)
    ok, encoded = cv2.imencode(".png", raw)
    assert ok
    path.write_bytes(encoded.tobytes())
    registry = builtin_registry()
    graph = Graph()
    load = graph.add(registry.create("input.depth_file", parameters={"path": str(path)}))
    sample = graph.add(
        registry.create("inspect.depth_sample", parameters={"pixel_points": [[0, 0], [1, 0]]})
    )
    graph.connect(load.id, "depth", sample.id, "depth")
    nested = encapsulate(graph, [sample.id], "Depth samples", registry)
    graph.run()
    assert load.state == "pass", load.error
    assert nested.state == "pass", nested.error
    depth = load.last_output_images["depth"]
    assert isinstance(depth, DepthArray) and depth.units == "m"
    assert np.isnan(depth[0, 0]) and depth[1, 1] == pytest.approx(3)
    assert copy.deepcopy(depth).units == "m"
    assert np.all(preview_rgb(depth)[0, 0] == 0)
    result = next(iter(nested.last_result.outputs.values()))
    assert result["samples"] == [{"pixel": [0, 0], "depth": None}, {"pixel": [1, 0], "depth": 1.0}]
    paired = registry.create("image.rgbd")
    paired.execute({"image": np.zeros((2, 2, 3), np.uint8), "depth": depth})
    assert paired.state == "pass" and isinstance(paired.last_output_images["depth"], DepthArray)
    paired.execute({"image": np.zeros((3, 2, 3), np.uint8), "depth": depth})
    assert "identical dimensions" in paired.error
    gray = graph.add(registry.create("gapi.gray"))
    with pytest.raises(ValueError, match="depth to image"):
        graph.connect(load.id, "depth", gray.id, "image")
    restored = Graph.from_dict(graph.to_dict(), registry)
    restored.run()
    assert restored.blocks[nested.id].state == "pass"
