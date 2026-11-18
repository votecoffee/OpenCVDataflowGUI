"""Independent quantities, stencil support, typed units and source-frame preservation."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.monocular_depth import MonocularDepth, linear_support, plane
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.stereo3d import DepthToCloud
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/monocular_depth"


def tool(case="relative", **parameters):
    return MonocularDepth(
        parameters={
            "model_path": str(BASE / "models" / case / "model.onnx"),
            "width": 31,
            "height": 23,
            "input_interpolation": "linear",
            **parameters,
        }
    )


def source(encoding="bgr", depth="uint8"):
    yy, xx = np.indices((23, 31))
    red = (40 + xx * 4 + yy * 2).astype(np.uint8)
    image = np.dstack((30 + yy * 5, 220 - xx * 3, red)).astype(np.uint8)
    if encoding in ("rgb", "rgba"):
        image = image[..., ::-1]
    if encoding in ("rgba", "bgra"):
        image = np.dstack((image, (xx * 7).astype(np.uint8)))
    if encoding == "gray":
        image = red
    if depth == "uint16":
        image = image.astype(np.uint16) * 257
    elif depth == "float32":
        image = image.astype(np.float32) / 255
    return ImageArray(image, encoding), red.astype(np.float64) / 255


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "gray"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_known_metric_planes_preserve_source_encoding_and_original_pixels(encoding, dtype):
    image, red = source(encoding, dtype)
    before = image.copy()
    block = tool(convention="metric_depth")
    try:
        result = block.execute({"image": image})
        assert result is not None and result.passed, block.error
        np.testing.assert_allclose(result.outputs["depth"], 1 + 2 * red, atol=6e-7)
        np.testing.assert_array_equal(result.outputs["image"], before)
        np.testing.assert_array_equal(image, before)
        assert result.outputs["image"].color_format == encoding
        assert (
            isinstance(result.outputs["depth"], DepthArray) and result.outputs["depth"].units == "m"
        )
        assert (
            result.outputs["estimate"]["metric_declared"]
            and not result.outputs["estimate"]["metric_validated"]
        )
        assert result.payloads["depth"]["depth_estimation"]["metric_validated"] is False
    finally:
        block.close()


@pytest.mark.parametrize(
    "convention,scale,offset,units",
    [
        ("relative_depth", 2.0, 0.5, "m"),
        ("relative_inverse_depth", 1.0, 0.0, "m"),
        ("metric_depth", 1000.0, 0.0, "mm"),
        ("metric_inverse_depth", 1.0, 0.0, "m"),
        ("metric_inverse_depth", 0.001, 0.0, "mm"),
    ],
)
def test_affine_quantities_units_inverse_order_and_relative_port_guard(
    convention, scale, offset, units
):
    image, red = source()
    block = tool(
        convention=convention,
        head_scale=scale,
        head_offset=offset,
        units=units,
        maximum_depth=10000,
    )
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        q = (1 + 2 * red) * scale + offset
        np.testing.assert_allclose(
            np.array(result.outputs["estimate"]["values"]).reshape(23, 31), q, rtol=2e-7
        )
        if convention.startswith("relative"):
            assert result.outputs["depth"] is NO_VALUE
            assert result.outputs["estimate"]["units"] == "arbitrary"
            graph = Graph()
            a = graph.add(block)
            target = graph.add(DepthToCloud())
            with pytest.raises(ValueError):
                graph.connect(a.id, "estimate", target.id, "depth")
            graph.close()
        else:
            expected = 1 / q if convention.endswith("inverse_depth") else q
            np.testing.assert_allclose(result.outputs["depth"], expected, rtol=3e-7)
            assert result.outputs["depth"].units == units
    finally:
        block.close()


@pytest.mark.parametrize("layout", ["HW", "NHW", "CHW", "HWC", "NCHW", "NHWC"])
def test_declared_plane_layouts_are_exact_and_wrong_shapes_reject(layout):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    wrapped = {
        "HW": values,
        "NHW": values[None],
        "CHW": values[None],
        "HWC": values[..., None],
        "NCHW": values[None, None],
        "NHWC": values[None, ..., None],
    }[layout]
    np.testing.assert_array_equal(plane(wrapped, layout), values)
    with pytest.raises(ValueError):
        plane(np.zeros((2, 2, 2, 2), np.float32), layout)


def test_all_nonzero_interpolation_contributors_required_even_tiny_weights():
    supported = np.array([[True, False], [True, True]])
    x = np.array([0, 1e-9, 0.5, 1.0], np.float32)
    y = np.array([0, 0.5, 1], np.float32)
    actual = linear_support(supported, x, y)
    expected = np.array(
        [[True, False, False, False], [True, False, False, False], [True, True, True, True]]
    )
    np.testing.assert_array_equal(actual, expected)


def test_invalid_holes_and_metric_range_produce_current_masks_and_nan_depth():
    image = cv2.imread(str(BASE / "assets/hole.png"))
    block = tool("invalid_hole", convention="metric_inverse_depth", maximum_depth=2.0)
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        q = image[..., 2].astype(float) / 255
        valid = (q > 0) & (q >= 0.5)
        np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, valid)
        assert np.isnan(result.outputs["depth"][~valid]).all()
        np.testing.assert_allclose(result.outputs["depth"][valid], 1 / q[valid], atol=2e-7)
        values = np.array(result.outputs["estimate"]["values"]).reshape(q.shape)
        assert np.all(values[~valid] == 0)
    finally:
        block.close()


def test_selections_and_same_grid_ticket_recipe_pyramid_context():
    image, _ = source()
    context = {
        "spatial": frame([31, 23], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        "aux": {"lot": "A"},
        "pyramid": {
            "level": 1,
            "largest_size": [62, 46],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    mask = np.zeros((23, 31), np.uint8)
    mask[2:18, 4:25] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [31, 23],
        "shape": "rectangle",
        "rectangle": [8, 6, 21, 15],
    }
    block = tool(convention="metric_depth")
    try:
        result = block.execute(
            {"image": ValuePacket(image, context), "mask": mask, "region": region}
        )
        assert result is not None, block.error
        selected = np.zeros((23, 31), bool)
        selected[6:18, 8:25] = True
        np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, selected)
        for payload in result.payloads.values():
            for key, value in context.items():
                assert payload[key] == value
        assert "depth_estimation" not in result.payloads["image"]
    finally:
        block.close()


def test_metric_depth_projects_to_known_camera_rays_with_estimation_provenance():
    image, red = source()
    block = tool(convention="metric_depth")
    cloud = DepthToCloud(parameters={"sample_step": 1})
    camera = camera_record([[80, 0, 15], [0, 90, 11], [0, 0, 1]], [0] * 5, [31, 23])
    try:
        result = block.execute({"image": image})
        assert result is not None, block.error
        projected = cloud.execute({"depth": result.outputs["depth"], "camera": camera})
        assert projected is not None and projected.passed, cloud.error
        yy, xx = np.indices((23, 31))
        z = 1 + 2 * red
        xyz = np.stack(((xx - 15) * z / 80, (yy - 11) * z / 90, z), axis=-1).reshape(-1, 3)
        np.testing.assert_allclose(projected.outputs["cloud"].points, xyz, atol=6e-7)
    finally:
        block.close()
        cloud.close()


def test_empty_selection_skips_file_io_and_emits_current_failed_coverage(monkeypatch):
    import opencv_dataflow.monocular_depth as module

    monkeypatch.setattr(module, "model_bytes", lambda *a, **k: pytest.fail("model I/O"))
    block = MonocularDepth()
    image, _ = source()
    result = block.execute({"image": image, "mask": np.zeros((23, 31), np.uint8)})
    assert result is not None and not result.passed and result.outputs["info"]["native_calls"] == 0
    assert result.outputs["depth"] is NO_VALUE and result.outputs["receipt"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "monocular_depth.coverage"
    block.close()


def test_lean_and_display_controls_do_not_change_numbers_or_scale():
    image, _ = source()
    blocks = [tool(), tool(preview_mode="fixed", display_min=100, display_max=200), tool()]
    blocks[-1].run_profile = PROFILES["lean"]
    try:
        results = [b.execute({"image": image}) for b in blocks]
        assert all(r is not None for r in results)
        assert [r.outputs["estimate"]["values"] for r in results].count(
            results[0].outputs["estimate"]["values"]
        ) == 3
        assert results[-1].outputs["preview"] is NO_VALUE and not results[-1].graphics
        assert not np.array_equal(results[0].outputs["preview"], results[1].outputs["preview"])
    finally:
        for b in blocks:
            b.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"head_scale": 0},
        {"head_scale": True},
        {"mean": [0]},
        {"std": [1, 0, 1]},
        {"maximum_model_value": 0.1, "minimum_model_value": 1},
        {"display_min": 1, "display_max": 0},
        {"minimum_depth": 5, "maximum_depth": 4},
        {"minimum_valid_fraction": 1.1},
        {"export_estimate": "yes"},
        {"width": 0},
        {"layout": "bad"},
        {"backend": "cuda", "target": "cpu"},
        {"target": "opencl", "engine": "auto"},
    ],
)
def test_invalid_parameters_contain_errors_and_recover(changes):
    block = tool(**changes)
    image, _ = source()
    try:
        assert block.execute({"image": image}) is None and block.error
        block.parameters.update(copy.deepcopy(tool().parameters))
        assert block.execute({"image": image}) is not None, block.error
    finally:
        block.close()


@pytest.mark.parametrize(
    "changes",
    [{"maximum_source_pixels": 100}, {"maximum_input_elements": 100}, {"max_output_elements": 100}],
)
def test_resource_bounds_reject_without_output(changes):
    block = tool(**changes)
    image, _ = source()
    try:
        assert block.execute({"image": image}) is None and block.error
    finally:
        block.close()
