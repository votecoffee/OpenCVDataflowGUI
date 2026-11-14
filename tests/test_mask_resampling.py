"""Independent cell-overlap references for coverage-mask resizing (S1-118)."""

import copy
from fractions import Fraction

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray
from opencv_dataflow.mask_resampling import ResampleMask
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry


def run(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def overlap_reference(source, size):
    """Direct exact rational rectangle intersections, without prefix sums or resize."""
    sh, sw = source.shape
    width, height = size
    output = np.zeros((height, width))
    for y in range(height):
        top, bottom = Fraction(y * sh, height), Fraction((y + 1) * sh, height)
        for x in range(width):
            left, right = Fraction(x * sw, width), Fraction((x + 1) * sw, width)
            total = Fraction(0)
            for sy in range(sh):
                dy = max(0, min(bottom, sy + 1) - max(top, sy))
                for sx in range(sw):
                    dx = max(0, min(right, sx + 1) - max(left, sx))
                    total += dx * dy * Fraction(float(source[sy, sx]))
            output[y, x] = float(total / ((right - left) * (bottom - top)))
    return output


@pytest.mark.parametrize(
    "size", [(3, 2), (13, 9), (13, 3), (3, 9), (7, 5), (1, 1), (1, 11), (11, 1)]
)
@pytest.mark.parametrize("mode", ["binary", "coverage"])
def test_arbitrary_resize_ratios_match_exact_area_intersections(size, mode):
    source = np.random.default_rng(118).integers(0, 5, (5, 7))
    if mode == "binary":
        source = (source > 1).astype(np.uint8) * 37  # Every nonzero value, not only 255.
        fractions = (source != 0).astype(float)
    else:
        source = source.astype(np.float64) / 4
        fractions = source
    original = source.copy()
    expected = overlap_reference(fractions, size)
    for threshold in (0.0, 0.5, 1.0):
        block = ResampleMask(
            parameters={
                "width": size[0],
                "height": size[1],
                "input_mode": mode,
                "minimum_coverage": threshold,
            }
        )
        result = run(block, mask=source)
        np.testing.assert_allclose(result.outputs["coverage"], expected, atol=1e-14, rtol=0)
        accepted = (expected > 0) & (expected >= threshold)
        np.testing.assert_array_equal(result.outputs["mask"], accepted.astype(np.uint8) * 255)
        assert result.outputs["mask"].dtype == np.uint8
        assert result.outputs["coverage"].dtype == np.float64
        assert result.outputs["info"]["mean_coverage"] == pytest.approx(fractions.mean(), abs=1e-14)
        assert result.passed  # Empty masks are valid processing results, not failed inspections.
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("size", [(17, 11), (32, 24), (1, 1)])
def test_downsampling_agrees_with_native_area_and_preserves_total_coverage(size):
    source = np.random.default_rng(7).integers(0, 2, (48, 64), dtype=np.uint8)
    result = run(ResampleMask(parameters={"width": size[0], "height": size[1]}), mask=source)
    native = cv2.resize(source.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(result.outputs["coverage"], native, atol=2e-7, rtol=0)
    assert result.outputs["coverage"].mean() == pytest.approx(source.mean(), abs=1e-14)


def test_checkerboard_thin_valid_stripes_holes_and_threshold_endpoints():
    source = np.full((4, 8), 255, np.uint8)
    source[:, 1::2] = 0
    tool = ResampleMask(parameters={"width": 4, "height": 2})
    strict = run(tool, mask=source)
    np.testing.assert_array_equal(strict.outputs["coverage"], np.full((2, 4), 0.5))
    assert not strict.outputs["mask"].any()
    tool.parameters["minimum_coverage"] = 0.5
    assert run(tool, mask=source).outputs["mask"].all()
    source[:] = 0
    tool.parameters["minimum_coverage"] = 0
    empty = run(tool, mask=source)
    assert not empty.outputs["mask"].any() and empty.passed
    source[0, 0] = 255
    positive = run(tool, mask=source)
    assert np.count_nonzero(positive.outputs["mask"]) == 1
    assert positive.outputs["coverage"][0, 0] == 0.25


def test_fractional_near_one_is_not_rounded_to_full_validity_and_zero_is_not_invented():
    source = np.ones((5, 7), np.float64)
    source[2, 3] = np.nextafter(1.0, 0.0)
    tool = ResampleMask(parameters={"width": 1, "height": 1, "input_mode": "coverage"})
    result = run(tool, mask=source)
    assert result.outputs["coverage"][0, 0] < 1 and result.outputs["mask"][0, 0] == 0
    source[:] = 0
    source[2, 3] = 1e-300
    tool.parameters.update(width=14, height=10, minimum_coverage=0)
    result = run(tool, mask=source)
    assert np.count_nonzero(result.outputs["mask"]) == 4
    assert np.all(result.outputs["coverage"][result.outputs["mask"] != 0] > 0)


def test_chunked_sampling_and_noncontiguous_inputs(monkeypatch):
    import opencv_dataflow.mask_resampling as module

    source = np.random.default_rng(31).integers(0, 2, (159, 201), dtype=np.uint8)[:, ::-1]
    tool = ResampleMask(parameters={"width": 71, "height": 94})
    first = run(tool, mask=source)
    monkeypatch.setattr(module, "ROW_CHUNK", 3)
    second = run(tool, mask=source)
    np.testing.assert_array_equal(first.outputs["coverage"], second.outputs["coverage"])
    assert second.outputs["coverage"].mean() == pytest.approx(source.mean(), abs=1e-14)


def test_target_grid_geometry_tickets_recipes_and_pyramid_context():
    source = np.full((7, 9), 255, np.uint8)
    spatial = frame([9, 7], "camera-A")
    spatial["to_root"] = [[2, 0, 11], [0, 2, 5], [0, 0, 1]]
    payload = {
        "spatial": spatial,
        "ticket": TicketClock().issue("camera-A"),
        "recipe_ref": {"id": "part", "revision": 1, "digest": "a" * 64},
        "pyramid": {
            "level": 1,
            "largest_size": [18, 14],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    original = copy.deepcopy(payload)
    resized = run(
        builtin_registry().create("cv.resize", parameters={"width": 4, "height": 3}),
        image=ValuePacket(source, payload),
    )
    tool = ResampleMask()  # Target size overrides 640x480 defaults.
    result = run(tool, mask=ValuePacket(source, payload), target=packet_for(resized, "image"))
    assert result.outputs["mask"].shape == (3, 4)
    assert result.outputs["info"]["target_frame_verified"]
    for name in result.outputs:
        output = result.payloads[name]
        assert output["spatial"] == resized.payloads["image"]["spatial"]
        assert output["pyramid"] == resized.payloads["image"]["pyramid"]
        assert (
            output["ticket"] == payload["ticket"] and output["recipe_ref"] == payload["recipe_ref"]
        )
    assert payload == original
    for change in ("root", "offset", "root_size"):
        bad = copy.deepcopy(resized.payloads["image"])
        if change == "root":
            bad["spatial"]["root_id"] = "camera-B"
        elif change == "offset":
            bad["spatial"]["to_root"][0][2] += 1
        else:
            bad["spatial"]["root_size"][0] += 1
        assert (
            tool.execute(
                {
                    "mask": ValuePacket(source, payload),
                    "target": ValuePacket(resized.outputs["image"], bad),
                }
            )
            is None
        )
        assert tool.error_kind == "invalid" and not tool.current_emissions


@pytest.mark.parametrize(
    "source,mode",
    [
        (np.zeros((4, 6, 3), np.uint8), "binary"),
        (np.zeros((4, 6), np.uint16), "binary"),
        (np.zeros((4, 6), np.float32), "binary"),
        (np.zeros((4, 6), np.uint8), "coverage"),
        (np.full((4, 6), np.nan), "coverage"),
        (np.full((4, 6), np.inf), "coverage"),
        (np.full((4, 6), -0.01), "coverage"),
        (np.full((4, 6), 1.01), "coverage"),
        (DepthArray(np.zeros((4, 6), np.float32), "m"), "coverage"),
    ],
)
def test_invalid_source_is_contained(source, mode):
    tool = ResampleMask(parameters={"input_mode": mode})
    assert tool.execute({"mask": source}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"width": 0},
        {"height": 1.5},
        {"width": True},
        {"width": 16385},
        {"minimum_coverage": -0.1},
        {"minimum_coverage": 1.1},
        {"minimum_coverage": True},
        {"minimum_coverage": "0.5"},
        {"input_mode": "invalid"},
    ],
)
def test_invalid_configuration_emits_nothing(parameters):
    tool = ResampleMask(parameters=parameters)
    assert tool.execute({"mask": np.zeros((4, 6), np.uint8)}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions


def test_native_errors_resource_bounds_and_graphics_off(monkeypatch):
    import opencv_dataflow.mask_resampling as module

    source = np.ones((4, 6), np.uint8)
    tool = ResampleMask(parameters={"width": 4, "height": 3})
    assert run(tool, mask=source).graphics
    tool.graphics_mode = "off"
    assert not run(tool, mask=source).graphics
    with monkeypatch.context() as patch:
        patch.setattr(module, "MAX_PIXELS", 20)
        assert tool.execute({"mask": source}) is None and tool.error_kind == "invalid"
        tool.parameters.update(width=6, height=4)
        assert tool.execute({"mask": source[:2]}) is None and tool.error_kind == "invalid"

    def broken(*args, **kwargs):
        raise cv2.error("Controlled integral failure")

    monkeypatch.setattr(cv2, "integral", broken)
    assert tool.execute({"mask": source}) is None
    assert "Controlled integral failure" in tool.error and not tool.current_emissions


def test_registry_json_xml_and_offscreen_editor(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.assistant_api import block_schema
    from opencv_dataflow.opencv_reference import opencv_reference

    registry = builtin_registry()
    graph = Graph()
    tool = graph.add(
        registry.create(
            "cv.resample_mask",
            parameters={
                "width": 7,
                "height": 5,
                "input_mode": "coverage",
                "minimum_coverage": 0.75,
            },
        )
    )
    assert block_entry(tool).path == ("OpenCV", "Image Processing (imgproc)", "Geometry")
    assert block_schema(tool.type_id, registry)["opencv_functions"] == ["integral"]
    assert "cv2.integral" in opencv_reference(tool)
    for extension in ("json", "xml"):
        path = tmp_path / f"mask.{extension}"
        graph.save(path)
        restored = Graph.load(path, registry)
        assert restored.to_dict() == graph.to_dict()
        restored.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        app.processEvents()
        window.inspector.select(tool)
        assert isinstance(window.inspector.fields["input_mode"], QComboBox)
        result = run(tool, mask=np.ones((4, 6), np.float32))
        assert result.outputs["mask"].all() and window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
