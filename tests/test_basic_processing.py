import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray, convert_depth, image_format, preview_rgb
from opencv_dataflow.toolblocks import ToolBlock


def execute(kind, image, parameters=None, **extra):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute({"image": image, **extra})
    assert block.state == "pass", block.error
    return block.last_output_images["image"]


@pytest.mark.parametrize("depth", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "gray"])
def test_grayscale_preserves_depth_and_interprets_encoding(depth, encoding):
    image = np.random.default_rng(16).integers(0, 255, (17, 23, 3), dtype=np.uint8).astype(depth)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if encoding == "rgb":
        image = image[:, :, ::-1].copy()
    elif encoding == "bgra":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        expected = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif encoding == "gray":
        image = expected.copy()
    image = ImageArray(image, encoding, metadata={"custom": {"part": 12}})
    output = execute("cv.gray", image)
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == depth and image_format(output) == "gray"
    assert output.metadata["custom"] == {"part": 12}


@pytest.mark.parametrize("depth", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("automatic", [False, True])
def test_gaussian_matches_opencv_with_rectangular_or_automatic_kernel(depth, automatic):
    image = np.random.default_rng(11).integers(0, 255, (23, 31, 3), dtype=np.uint8).astype(depth)
    image = ImageArray(image, "rgb")
    kx, ky = (0, 0) if automatic else (3, 7)
    output = execute(
        "cv.gaussian_blur",
        image,
        {"kernel": kx, "kernel_y": ky, "sigma": 0.8, "sigma_y": 2.1, "border": "replicate"},
    )
    expected = cv2.GaussianBlur(
        np.asarray(image), (kx, ky), 0.8, sigmaY=2.1, borderType=cv2.BORDER_REPLICATE
    )
    np.testing.assert_array_equal(output, expected)
    assert image_format(output) == "rgb"


def test_gaussian_constant_border_and_mask():
    image = np.zeros((19, 25, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:10, :12] = 255
    output = execute(
        "cv.gaussian_blur",
        image,
        {"kernel": 3, "kernel_y": 7, "border": "constant", "border_value": [20, 80, 180, 0]},
        mask=mask,
    )
    padded = cv2.copyMakeBorder(image, 3, 3, 1, 1, cv2.BORDER_CONSTANT, value=(20, 80, 180))
    expected = cv2.GaussianBlur(padded, (3, 7), 1.2)[3:-3, 1:-1]
    expected[mask == 0] = image[mask == 0]
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("aperture", ["3", "5", "7"])
@pytest.mark.parametrize("l2", [False, True])
def test_canny_parameters_and_region_selection(aperture, l2):
    image = np.random.default_rng(15).integers(0, 255, (40, 60), dtype=np.uint8)
    # Use a mask for exact pixel inclusion and retain a separate region integration
    # check in test_regions.py shared with the legacy implementation.
    mask = np.zeros_like(image)
    mask[:, :25] = 255
    output = execute("cv.canny", image, {"aperture": aperture, "l2_gradient": l2}, mask=mask)
    expected = cv2.Canny(image, 60, 150, apertureSize=int(aperture), L2gradient=l2)
    expected[mask == 0] = 0
    np.testing.assert_array_equal(output, expected)
    assert image_format(output) == "gray"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_canny_and_adaptive_threshold_convert_tagged_color_to_byte_masks(dtype):
    color = np.random.default_rng(9).integers(0, 255, (24, 32, 4), dtype=np.uint8)
    source = color.astype(dtype)
    if dtype == np.uint16:
        source *= 257
    elif np.issubdtype(dtype, np.floating):
        source /= 255
    image = ImageArray(source, "rgba", metadata={"part": "fixture"})
    np.testing.assert_array_equal(preview_rgb(image), color[..., :3])
    working = source.astype(np.float32) if dtype == np.float64 else source
    gray = convert_depth(cv2.cvtColor(working, cv2.COLOR_RGBA2GRAY), "uint8")
    canny = execute("cv.canny", image)
    adaptive = execute("cv.adaptive_threshold", image)
    np.testing.assert_array_equal(canny, cv2.Canny(gray, 60, 150))
    np.testing.assert_array_equal(
        adaptive, cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
        )
    )
    assert canny.dtype == adaptive.dtype == np.uint8
    assert image_format(canny) == image_format(adaptive) == "gray"
    assert canny.metadata["part"] == adaptive.metadata["part"] == "fixture"


@pytest.mark.parametrize(
    "kind, parameters, image",
    [
        ("cv.gaussian_blur", {"kernel": 2}, np.zeros((10, 10), np.uint8)),
        ("cv.gaussian_blur", {"kernel": 3.5}, np.zeros((10, 10), np.uint8)),
        ("cv.gaussian_blur", {"sigma": float("nan")}, np.zeros((10, 10), np.uint8)),
        ("cv.gaussian_blur", {"kernel": 0, "sigma": 0}, np.zeros((10, 10), np.uint8)),
        ("cv.canny", {"high": 50}, np.zeros((10, 10), np.uint8)),
        ("cv.canny", {"low": float("inf")}, np.zeros((10, 10), np.uint8)),
        ("cv.canny", {"aperture": "9"}, np.zeros((10, 10), np.uint8)),
        ("cv.canny", {}, np.zeros((10, 10), np.int16)),
        ("cv.canny", {}, DepthArray(np.ones((10, 10), np.float32))),
    ],
)
def test_invalid_inputs_fail_visibly(kind, parameters, image):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute({"image": image})
    assert block.state == "error" and block.error
    assert not block.last_output_images


def test_default_graph_runs_without_gapi_and_legacy_projects_remain_loadable(monkeypatch):
    registry = builtin_registry()
    graph = demo_graph(registry)
    data = copy.deepcopy(graph.to_dict())
    mapping = {"cv.gray": "gapi.gray", "cv.gaussian_blur": "gapi.blur"}
    for block in data["blocks"]:
        block["type"] = mapping.get(block["type"], block["type"])
        if block["type"].startswith("gapi."):
            block["parameters"].pop("algorithm_hint", None)
    legacy = Graph.from_dict(data, registry)
    legacy.run()
    assert all(b.state == "pass" for b in legacy.blocks.values())
    assert Graph.from_dict(legacy.to_dict(), registry).to_dict() == legacy.to_dict()

    def no_gapi(*args, **kwargs):
        raise AssertionError("Ordinary graph tried to construct a GComputation")

    monkeypatch.setattr(cv2, "GComputation", no_gapi)
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values())
    assert list(graph.blocks.values())[-1].last_result.outputs["count"] == 2
    for path in Path("examples").glob("*/*.json"):

        def check(value):
            if isinstance(value, dict):
                assert not str(value.get("type", "")).startswith("gapi."), path
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(json.loads(path.read_text(encoding="utf-8")))
    definition = ToolBlock.read("examples/inspect_shapes/inspect_shapes.toolblock.json", registry)
    assert definition is not None


def test_toolbox_separates_advanced_gapi_and_eager_enum_controls(app):
    window = MainWindow()
    found = {}
    tree = window.library
    for i in range(tree.topLevelItemCount()):
        category = tree.topLevelItem(i)
        for j in range(category.childCount()):
            subtype = category.child(j)
            for k in range(subtype.childCount()):
                item = subtype.child(k)
                found[item.data(0, Qt.ItemDataRole.UserRole)] = (category.text(0), subtype.text(0))
    assert found["gapi.gray"] == found["gapi.blur"] == found["gapi.canny"] == ("Advanced", "G-API")
    assert found["cv.gray"] == ("Image processing", "Color and depth")
    assert found["cv.gaussian_blur"] == ("Image processing", "Filtering")
    assert found["cv.canny"] == ("Image processing", "Edges")
    for kind, parameter in (
        ("cv.gaussian_blur", "border"),
        ("cv.canny", "aperture"),
        ("cv.gaussian_blur", "algorithm_hint"),
        ("cv.gray", "algorithm_hint"),
    ):
        window.inspector.select(window.registry.create(kind))
        field = window.inspector.fields[parameter]
        assert isinstance(field, QComboBox)
        assert field.count() > 1
    window.close()


@pytest.mark.parametrize(
    "hint_name, hint",
    [
        ("default", cv2.ALGO_HINT_DEFAULT),
        ("accurate", cv2.ALGO_HINT_ACCURATE),
        ("approximate", cv2.ALGO_HINT_APPROX),
    ],
)
def test_algorithm_hints_match_native_execution(hint_name, hint):
    image = np.random.default_rng(67).integers(0, 255, (32, 48, 3), dtype=np.uint8)
    gray = execute("cv.gray", image, {"algorithm_hint": hint_name})
    np.testing.assert_array_equal(gray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, hint=hint))
    blur = execute("cv.gaussian_blur", image, {"algorithm_hint": hint_name})
    np.testing.assert_array_equal(blur, cv2.GaussianBlur(image, (5, 5), 1.2, hint=hint))


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize(
    "algorithm,dtype,maximum",
    [("bilinear", np.uint16, 65535), ("edge_aware", np.uint16, 65535), ("vng", np.uint8, 255)],
)
def test_bayer_demosaicing_recovers_logical_color_channels(pattern, algorithm, dtype, maximum):
    values = {"R": maximum * 3 // 4, "G": maximum // 2, "B": maximum // 8}
    tile = np.array([values[color] for color in pattern], dtype=dtype).reshape(2, 2)
    raw = np.tile(tile, (8, 9))
    output = execute(
        "cv.demosaic_bayer",
        raw,
        {"pattern": pattern, "algorithm": algorithm, "color_format": "rgb"},
    )
    assert output.dtype == dtype and output.shape == (*raw.shape, 3)
    assert image_format(output) == "rgb"
    assert output[8, 9].tolist() == [values["R"], values["G"], values["B"]]


@pytest.mark.parametrize("row,column", [(0, 1), (1, 0), (1, 1)])
def test_bayer_demosaicing_uses_sensor_offset_and_opaque_alpha(row, column):
    raw = np.tile(np.array([[50000, 20000], [20000, 9000]], np.uint16), (8, 8))
    output = execute(
        "cv.demosaic_bayer",
        raw[row:, column:],
        {
            "color_format": "bgra",
            "row_offset": "odd" if row else "even",
            "column_offset": "odd" if column else "even",
        },
    )
    assert image_format(output) == "bgra"
    assert output[7, 7].tolist() == [9000, 20000, 50000, 65535]


def test_bayer_demosaicing_validates_layout_algorithm_and_output_format():
    block = builtin_registry().create("cv.demosaic_bayer", parameters={"algorithm": "vng"})
    for raw in (np.zeros((8, 8), np.uint16), np.zeros((7, 8), np.uint8)):
        assert block.execute({"image": raw}) is None
        assert block.error_kind == "invalid" and "VNG requires" in block.error
    block.parameters.update(algorithm="edge_aware", color_format="gray")
    raw = np.tile(np.array([[200, 100], [100, 20]], np.uint8), (4, 4))
    assert block.execute({"image": raw}) is None
    assert block.error_kind == "invalid" and "require bilinear" in block.error
    block.parameters["algorithm"] = "bilinear"
    result = block.execute({"image": raw})
    assert result is not None, block.error
    assert result.outputs["image"].ndim == 2
    assert image_format(result.outputs["image"]) == "gray"
    assert block.execute({"image": np.zeros((8, 8, 3), np.uint8)}) is None
    assert "single-channel" in block.error
