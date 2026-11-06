"""Verify packed map contracts and equivalence to native fixed-point remapping."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame, metadata, packet_for
from opencv_dataflow.remap_maps import PackedRemapMap


def _run(block, inputs):
    result = block.execute(inputs)
    assert result and result.passed, block.error
    return result


@pytest.mark.parametrize(
    ("method", "native_method"),
    (
        ("nearest", cv2.INTER_NEAREST),
        ("linear", cv2.INTER_LINEAR),
        ("cubic", cv2.INTER_CUBIC),
        ("lanczos4", cv2.INTER_LANCZOS4),
    ),
)
def test_packed_maps_sample_rgb16_with_native_coordinates_and_border(method, native_method):
    registry = builtin_registry()
    nearest = method == "nearest"
    mode = "nearest" if nearest else "subpixel"
    pack = registry.create("cv.pack_remap_map", parameters={"sampling": mode})
    y, x = np.indices((11, 15), np.float32)
    x += np.float32(0.304)
    y -= np.float32(1.596)
    packed = _run(pack, {"map_x": x, "map_y": y}).outputs["packed_map"]
    assert packed.describe() == {
        "model": "fixed_remap_map_v1",
        "size": [15, 11],
        "sampling": mode,
        "bytes": 11 * 15 * (4 if nearest else 6),
    }
    native = cv2.convertMaps(x, y, cv2.CV_16SC2, nninterpolation=nearest)
    coordinates, fractions = packed.native_maps()
    assert not coordinates.flags.writeable
    np.testing.assert_array_equal(coordinates, native[0])
    if nearest:
        assert fractions is None
        np.testing.assert_array_equal(coordinates[..., 0], np.rint(x))
        np.testing.assert_array_equal(coordinates[..., 1], np.rint(y))
    else:
        assert not fractions.flags.writeable
        np.testing.assert_array_equal(fractions, native[1])
        xx, yy = cv2.convertMaps(coordinates, fractions, cv2.CV_32FC1)
        np.testing.assert_allclose(xx, x, atol=1 / 64)
        np.testing.assert_allclose(yy, y, atol=1 / 64)

    values = np.random.default_rng(5).integers(0, 65536, (12, 16, 3), dtype=np.uint16)
    remap = registry.create(
        "cv.remap_packed", parameters={"interpolation": method, "border": "wrap"}
    )
    result = _run(remap, {"image": ImageArray(values, "rgb"), "packed_map": packed})
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgb"
    expected = cv2.remap(values, *native, native_method, borderMode=cv2.BORDER_WRAP)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("sampling", ("nearest", "subpixel"))
def test_sampling_mismatch_withholds_output(sampling):
    registry = builtin_registry()
    pack = registry.create("cv.pack_remap_map", parameters={"sampling": sampling})
    y, x = np.indices((6, 8), np.float32)
    packet = _run(pack, {"map_x": x, "map_y": y}).outputs["packed_map"]
    method = "linear" if sampling == "nearest" else "nearest"
    remap = registry.create("cv.remap_packed", parameters={"interpolation": method})
    assert remap.execute({"image": x, "packed_map": packet}) is None
    assert "Nearest maps" in remap.error


@pytest.mark.parametrize("coordinate", (-32769, 32768))
def test_packing_rejects_coordinates_that_would_saturate(coordinate):
    y, x = np.indices((6, 8), np.float32)
    x[0, 0] = coordinate
    pack = builtin_registry().create("cv.pack_remap_map")
    assert pack.execute({"map_x": x, "map_y": y}) is None
    assert "fixed-point range" in pack.error


def test_wrong_packet_type_and_invalid_native_indices_are_rejected():
    remap = builtin_registry().create("cv.remap_packed")
    assert (
        remap.execute({"image": np.zeros((3, 3), np.uint8), "packed_map": {"size": [1, 1]}}) is None
    )
    assert "requires packed remap coordinates" in remap.error
    with pytest.raises(ValueError, match="interpolation indices"):
        PackedRemapMap((1, 1), bytes(4), np.array([cv2.INTER_TAB_SIZE2], np.uint16).tobytes())


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_preserves_rectified_frame_and_ticket(profile):
    project = Path(__file__).parents[1] / "examples/packed_remap/packed_remap.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    floating = graph.blocks["float"].last_result.outputs["image"]
    fixed = graph.blocks["fixed"].last_result.outputs["image"]
    assert metadata(fixed)["spatial"] == metadata(floating)["spatial"]
    assert metadata(fixed)["spatial"] != metadata(source)["spatial"]
    assert metadata(fixed)["ticket"] == metadata(source)["ticket"]
    assert graph.blocks["compare"].last_result.outputs["value"] < 0.5

    pack = graph.blocks["pack"]
    maps = graph.blocks["maps"].last_result
    assert (
        pack.execute({"map_x": packet_for(maps, "map_y"), "map_y": packet_for(maps, "map_x")})
        is None
    )
    assert "matching X and Y" in pack.error

    fresh = _run(pack, {axis: packet_for(maps, axis) for axis in ("map_x", "map_y")})
    packed = packet_for(fresh, "packed_map")
    remap = graph.blocks["fixed"]
    assert remap.execute({"image": np.asarray(source)[:-1], "packed_map": packed}) is None
    assert "Source image dimensions" in remap.error
    bad_grid = {**packed.payload, "spatial": frame([192, 144], "different-map")}
    assert (
        remap.execute({"image": source, "packed_map": ValuePacket(packed.value, bad_grid)}) is None
    )
    assert "original pixel grid" in remap.error
