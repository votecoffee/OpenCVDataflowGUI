"""Independent triangular references and observable stack-blur wrapper contracts."""

import copy

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.stack_blur import StackBlur
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def successful(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def triangular_reference(image, kernel):
    result = image.astype(np.float64)
    for axis, size in ((1, kernel[0]), (0, kernel[1])):
        radius = size // 2
        weights = (radius + 1 - np.abs(np.arange(-radius, radius + 1))) / (radius + 1) ** 2
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        result = np.apply_along_axis(
            lambda row: np.convolve(row, weights, mode="valid"), axis, padded
        )
    return result


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba"])
def test_safe_native_float_work_preserves_depth_layout_and_alpha(dtype, encoding):
    shape = (37, 43) if encoding == "gray" else (37, 43, 4 if encoding.endswith("a") else 3)
    pixels = np.random.default_rng(85).normal(120, 30, shape)
    if dtype == np.uint16:
        pixels *= 210
    elif dtype == np.int16:
        pixels = (pixels - 120) * 200
    pixels = pixels.astype(dtype)
    source = ImageArray(pixels, encoding)
    before = source.copy()
    block = StackBlur(parameters={"kernel": 7, "kernel_y": 11})
    result = successful(block, {"image": source})
    work = pixels[..., :3] if encoding.endswith("a") else pixels
    expected = cv2.stackBlur(work.astype(np.float32), (7, 11))
    if dtype != np.float32:
        limits = np.iinfo(dtype)
        expected = np.clip(np.rint(expected), limits.min, limits.max).astype(dtype)
    actual = result.outputs["image"]
    assert actual.dtype == dtype and image_format(actual) == encoding
    np.testing.assert_array_equal(actual[..., :3] if encoding.endswith("a") else actual, expected)
    if encoding.endswith("a"):
        np.testing.assert_array_equal(actual[..., 3], pixels[..., 3])
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("kernel", [(3, 3), (31, 7), (1, 9)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_explicit_native_matches_the_qualified_binding(kernel, dtype):
    source = np.random.default_rng(3).integers(0, 256, (43, 53, 3)).astype(dtype)
    result = successful(
        StackBlur(parameters={"kernel": kernel[0], "kernel_y": kernel[1], "arithmetic": "native"}),
        {"image": source},
    )
    np.testing.assert_array_equal(result.outputs["image"], cv2.stackBlur(source, kernel))


@pytest.mark.parametrize("kernel", [(5, 3), (31, 17), (127, 63)])
def test_independent_triangular_impulse_and_replicated_edges(kernel):
    source = np.zeros((129, 151), np.float32)
    source[64, 75] = 9
    source[0, 0] = 5
    result = successful(
        StackBlur(parameters={"kernel": kernel[0], "kernel_y": kernel[1]}), {"image": source}
    )
    np.testing.assert_allclose(
        result.outputs["image"], triangular_reference(source, kernel), rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize("dtype,value", [(np.uint16, 65535), (np.int16, -32768)])
@pytest.mark.parametrize("kernel", [31, 255, 511])
def test_high_depth_constant_remains_constant_without_native_integer_artifacts(
    dtype, value, kernel
):
    source = np.full((513, 519), value, dtype)
    result = successful(StackBlur(parameters={"kernel": kernel}), {"image": source})
    np.testing.assert_array_equal(result.outputs["image"], source)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (6, 8)])
def test_explicit_oversize_clamp_and_identity(shape):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    block = StackBlur(parameters={"kernel": 31})
    result = successful(block, {"image": source})
    kernel = (
        shape[1] if shape[1] % 2 else shape[1] - 1,
        shape[0] if shape[0] % 2 else shape[0] - 1,
    )
    assert result.outputs["info"]["effective_kernel"] == list(kernel)
    assert result.outputs["info"]["kernel_clamped"]
    np.testing.assert_allclose(
        result.outputs["image"], triangular_reference(source, kernel), atol=2e-5
    )
    block.parameters["oversize_policy"] = "reject"
    assert block.execute({"image": source}) is None and block.error_kind == "invalid"


@pytest.mark.parametrize(
    "shape,fields",
    [
        ("circle", {"center": [17, 18], "radius": 12}),
        ("rectangle", {"rectangle": [6, 5, 26, 21]}),
        ("polygon", {"vertices": [[4, 4], [33, 6], [28, 31]]}),
    ],
)
def test_mask_region_intersection_uses_full_neighborhood_and_retains_context(shape, fields):
    from opencv_dataflow.selection import selection_mask

    data = np.random.default_rng(3).integers(0, 255, (37, 43, 3), dtype=np.uint8)
    payload = {
        "ticket": TicketClock().issue("PLC", "fixed", "part-A"),
        "spatial": frame([43, 37], "camera"),
        "acquisition": {"encoder": 77},
        "pyramid": {
            "level": 1,
            "largest_size": [86, 74],
            "to_largest": np.diag([2, 2, 1]).tolist(),
        },
    }
    source = ImageArray(data, "rgb", payload)
    mask = np.full((37, 43), 255, np.uint8)
    mask[10:16, 12:17] = 0
    region = {"model": "image_region_v1", "image_size": [43, 37], "shape": shape, **fields}
    inputs = {"image": ValuePacket(source, payload), "mask": mask, "region": region}
    selection = selection_mask({"image": source, "mask": mask, "region": region}, source) != 0
    block = StackBlur(parameters={"kernel": 15})
    result = successful(block, inputs)
    all_pixels = np.rint(cv2.stackBlur(data.astype(np.float32), (15, 15))).astype(np.uint8)
    np.testing.assert_array_equal(result.outputs["image"][selection], all_pixels[selection])
    np.testing.assert_array_equal(result.outputs["image"][~selection], data[~selection])
    assert result.outputs["info"]["selected_pixels"] == np.count_nonzero(selection)
    context = result.payloads["image"]
    assert ticket_key(context["ticket"]) == ticket_key(payload["ticket"])
    assert context["spatial"] == payload["spatial"] and context["pyramid"] == payload["pyramid"]
    assert context["acquisition"] == {"encoder": 77}
    assert result.drawings["write selection"].objects
    block.graphics_mode = "off"
    lean = successful(block, inputs)
    assert not lean.drawings
    np.testing.assert_array_equal(lean.outputs["image"], result.outputs["image"])


def test_empty_selection_skips_native_work_and_mismatched_grid_is_rejected(monkeypatch):
    source = ImageArray(np.zeros((9, 11), np.uint8), "gray", {"spatial": frame([11, 9], "A")})
    monkeypatch.setattr(cv2, "stackBlur", lambda *a, **k: pytest.fail("Unexpected native call"))
    block = StackBlur()
    result = successful(block, {"image": source, "mask": np.zeros((9, 11), np.uint8)})
    assert result.outputs["info"]["status"] == "empty_selection"
    other = ImageArray(np.ones((9, 11), np.uint8), "gray", {"spatial": frame([11, 9], "B")})
    assert block.execute({"image": source, "mask": other}) is None
    assert "frame differs" in block.error


def test_alpha_filter_and_association_rules():
    source = np.random.default_rng(12).random((17, 19, 4), dtype=np.float32)
    source[..., :3] *= source[..., 3:4]
    image = ImageArray(source, "rgba", {"alpha_representation": "premultiplied"})
    block = StackBlur(parameters={"kernel": 5})
    assert block.execute({"image": image}) is None and "Unassociate" in block.error
    block.parameters["alpha_mode"] = "filter"
    result = successful(block, {"image": image})
    np.testing.assert_array_equal(result.outputs["image"], cv2.stackBlur(source, (5, 5)))
    assert result.payloads["image"]["alpha_representation"] == "premultiplied"


def test_maximum_horizontal_kernel_matches_independent_impulse_reference():
    image = np.zeros((3, 2051), np.float32)
    image[1, 1025] = 1
    result = successful(StackBlur(parameters={"kernel": 2047, "kernel_y": 1}), {"image": image})
    np.testing.assert_allclose(
        result.outputs["image"], triangular_reference(image, (2047, 1)), atol=1e-7
    )


def test_scratch_limit_and_native_exception_are_contained(monkeypatch):
    import opencv_dataflow.stack_blur as module

    image = np.zeros((9, 11), np.uint8)
    block = StackBlur()
    with monkeypatch.context() as patch:
        patch.setattr(module, "MAX_SCRATCH_BYTES", 1)
        patch.setattr(cv2, "stackBlur", lambda *a, **k: pytest.fail("Unexpected native call"))
        assert block.execute({"image": image}) is None and "scratch" in block.error
        empty = successful(block, {"image": image, "mask": np.zeros_like(image)})
        assert empty.outputs["info"]["scratch_estimate_bytes"] == 0

    def broken(*args, **kwargs):
        raise cv2.error("Controlled native failure")

    monkeypatch.setattr(cv2, "stackBlur", broken)
    assert block.execute({"image": image}) is None
    assert "Controlled native failure" in block.error and not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"kernel": 0},
        {"kernel": 4},
        {"kernel_y": 4},
        {"kernel": 2049},
        {"kernel": True},
        {"arithmetic": "invalid"},
        {"maximum_pixels": 1},
    ],
)
def test_invalid_parameters_emit_nothing(parameters, monkeypatch):
    monkeypatch.setattr(cv2, "stackBlur", lambda *a, **k: pytest.fail("Unexpected native call"))
    block = StackBlur(parameters=parameters)
    assert block.execute({"image": np.zeros((9, 11), np.uint8)}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize(
    "source,parameters",
    [
        (np.zeros((11, 13), np.float64), {}),
        (np.full((11, 13), np.nan, np.float32), {}),
        (np.zeros((11, 13), np.uint16), {"arithmetic": "native"}),
        (np.zeros((259, 261), np.uint8), {"arithmetic": "native", "kernel": 257}),
        (np.full((11, 13), 1e37, np.float32), {}),
    ],
)
def test_unsupported_or_unsafe_work_is_rejected_before_native(source, parameters, monkeypatch):
    monkeypatch.setattr(cv2, "stackBlur", lambda *a, **k: pytest.fail("Unexpected native call"))
    block = StackBlur(parameters=parameters)
    assert block.execute({"image": source}) is None and not block.current_emissions


def test_parameter_enums_persistence_and_inspector_validation(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    registry = builtin_registry()
    block = registry.create("cv.stack_blur", parameters={"kernel": 127, "kernel_y": 63})
    assert block_entry(type(block)).path == ("OpenCV", "Image Processing (imgproc)", "Filtering")
    graph = Graph()
    graph.add(block)
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "stack.cv.stack_blur.xml"
    save_block_xml(block, path)
    assert read_block_xml(path)["parameters"] == block.parameters
    graph.save(tmp_path / "stack.json")
    restored = Graph.load(tmp_path / "stack.json", registry)
    assert restored.to_dict() == snapshot
    restored.close()
    window = MainWindow()
    try:
        window.replace_graph(graph, None)
        window.inspector.select(block)
        for name in ("arithmetic", "oversize_policy", "alpha_mode"):
            assert isinstance(window.inspector.fields[name], QComboBox)
        with pytest.raises(ValueError):
            block.validate_parameters(dict(block.parameters, kernel=4))
    finally:
        window.dirty = False
        window.close()
