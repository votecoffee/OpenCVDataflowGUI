"""Native/analytic sampling, ROI holes, frame ownership and bounded raster selections."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, packet_for
from opencv_dataflow.raster_regions import region_from_mask
from opencv_dataflow.selection import rasterize

PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[110, 0, 79.5], [0, 115, 59.5], [0, 0, 1]],
    "distortion": [-0.3, 0.05, 0.003, -0.002, 0],
    "image_size": [160, 120],
    "rms_pixels": 0.1,
    "units": "mm",
}


def run(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def tool(packed=False, **parameters):
    return builtin_registry().create(
        "cv.remap_selection_packed" if packed else "cv.remap_selection", parameters=parameters
    )


def fields(w=13, h=11, dx=0.25, dy=0.5):
    y, x = np.indices((h, w), np.float32)
    return {"map_x": x + dx, "map_y": y + dy}


def linear_oracle(source, x, y):
    # Independent scalar four-tap accumulation for the floating-map path.
    output = np.zeros(x.shape, np.float64)
    for row in range(x.shape[0]):
        for column in range(x.shape[1]):
            xx, yy = float(x[row, column]), float(y[row, column])
            a, b = int(np.floor(xx)), int(np.floor(yy))
            for sy, wy in ((b, 1 - (yy - b)), (b + 1, yy - b)):
                for sx, wx in ((a, 1 - (xx - a)), (a + 1, xx - a)):
                    if 0 <= sy < source.shape[0] and 0 <= sx < source.shape[1]:
                        output[row, column] += source[sy, sx] * wx * wy
    return output


@pytest.mark.parametrize(
    "rule,cutoff",
    [
        ("complete_footprint", 0.5),
        ("coverage_threshold", 0),
        ("coverage_threshold", 0.5),
        ("coverage_threshold", 1),
    ],
)
def test_fractional_masks_and_holes_match_independent_bilinear_oracle(rule, cutoff):
    source = np.arange(143, dtype=np.uint16).reshape(11, 13) * 100
    selected = np.zeros((11, 13), np.uint8)
    selected[2:9, 2:11] = 255
    selected[4:7, 5:8] = 0
    maps = fields()
    block = tool(selection_rule=rule, minimum_coverage=cutoff)
    result = run(block, {"image": source, "mask": selected, **maps})
    coverage = linear_oracle(selected / 255, maps["map_x"], maps["map_y"])
    support = linear_oracle(np.ones(selected.shape), maps["map_x"], maps["map_y"])
    np.testing.assert_allclose(result.outputs["coverage"], coverage)
    accepted = (
        coverage >= 1 - 1e-6
        if rule == "complete_footprint"
        else coverage > 0
        if cutoff == 0
        else coverage >= cutoff
    )
    expected = (accepted & (support >= 1 - 1e-6)).astype(np.uint8) * 255
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    np.testing.assert_array_equal(rasterize(result.outputs["region"], expected.shape), expected)
    np.testing.assert_array_equal(
        result.outputs["image"], cv2.remap(source, maps["map_x"], maps["map_y"], cv2.INTER_LINEAR)
    )
    assert set(np.unique(result.outputs["mask"])) <= {0, 255}


@pytest.mark.parametrize(
    "method,halo", [("nearest", 0), ("linear", 0), ("cubic", 2), ("lanczos4", 4)]
)
@pytest.mark.parametrize("border", ["replicate", "wrap", "reflect_101"])
def test_filled_borders_never_add_real_or_selected_support(method, halo, border):
    source = np.full((15, 17), 200, np.uint16)
    maps = fields(21, 19, -2, -2)
    result = run(tool(interpolation=method, border=border), {"image": source, **maps})
    assert np.all(result.outputs["image"] == 200)
    valid = result.outputs["valid_mask"]
    assert not valid[0].any() and not valid[:, 0].any()
    assert result.outputs["report"]["conservative_halo"] == halo
    assert np.count_nonzero(valid) <= source.size
    np.testing.assert_array_equal(result.outputs["mask"], valid)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "lanczos4"])
def test_packed_sampling_matches_phase_aligned_float_maps_and_preserves_rgb16_alpha(method):
    maps = fields(13, 11)
    values = np.random.default_rng(3).integers(0, 65536, (11, 13, 4), np.uint16)
    source = ImageArray(values, "rgba")
    mask = np.zeros((11, 13), np.uint8)
    mask[1:-1, 1:-1] = 255
    selected = {"image": source, "mask": mask, **maps}
    sampling = "nearest" if method == "nearest" else "linear"
    split = run(
        tool(interpolation=method, selection_sampling=sampling, minimum_selected_pixels=0), selected
    )
    packer = builtin_registry().create(
        "cv.pack_remap_map",
        parameters={
            "sampling": "nearest" if method == "nearest" else "subpixel",
        },
    )
    packed = run(packer, maps)
    result = run(
        tool(True, interpolation=method, selection_sampling=sampling, minimum_selected_pixels=0),
        {
            "image": source,
            "mask": mask,
            "packed_map": packet_for(packed, "packed_map"),
        },
    )
    for name in ("image", "mask", "coverage", "valid_mask"):
        np.testing.assert_array_equal(result.outputs[name], split.outputs[name])
    assert (
        image_format(result.outputs["image"]) == "rgba"
        and result.outputs["image"].dtype == np.uint16
    )


def analytic_lens_maps(profile, size, target):
    y, x = np.indices(size[::-1], np.float64)
    nx, ny = (x - target[0][2]) / target[0][0], (y - target[1][2]) / target[1][1]
    k1, k2, p1, p2, k3 = profile["distortion"]
    r2 = nx * nx + ny * ny
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    dx = nx * radial + 2 * p1 * nx * ny + p2 * (r2 + 2 * nx * nx)
    dy = ny * radial + p1 * (r2 + 2 * ny * ny) + 2 * p2 * nx * ny
    camera = profile["camera_matrix"]
    return (dx * camera[0][0] + camera[0][2]).astype(np.float32), (
        dy * camera[1][1] + camera[1][2]
    ).astype(np.float32)


def test_whole_nonlinear_lens_roi_uses_rectified_grid_and_native_downstream_measurements():
    registry = builtin_registry()
    maps = run(
        registry.create("calibration.lens_maps", parameters={"width": 128, "height": 96}),
        {"calibration": PROFILE},
    )
    target = maps.outputs["rectified_calibration"]["camera_matrix"]
    x, y = analytic_lens_maps(PROFILE, [128, 96], target)
    np.testing.assert_allclose(maps.outputs["map_x"], x, atol=1e-5)
    np.testing.assert_allclose(maps.outputs["map_y"], y, atol=1e-5)
    yy, xx = np.indices((120, 160))
    image = ImageArray((1000 + xx * 100 + yy * 10).astype(np.uint16), "gray")
    spatial = frame([160, 120], "camera")
    original = ValuePacket(
        image, {"spatial": spatial, "sample": "retain", "video_frame": {"frame_index": 4}}
    )
    region = {
        "model": "image_region_v1",
        "image_size": [160, 120],
        "shape": "composite",
        "operation": "difference",
        "children": [
            {
                "model": "image_region_v1",
                "image_size": [160, 120],
                "shape": "circle",
                "center": [50, 55],
                "radius": 32,
            },
            {
                "model": "image_region_v1",
                "image_size": [160, 120],
                "shape": "rectangle",
                "rectangle": [44, 45, 18, 22],
            },
        ],
    }
    inputs = {
        "image": original,
        "region": region,
        "map_x": packet_for(maps, "map_x"),
        "map_y": packet_for(maps, "map_y"),
    }
    output = run(tool(), inputs)
    legacy = run(registry.create("cv.remap"), {k: v for k, v in inputs.items() if k != "region"})
    np.testing.assert_array_equal(output.outputs["image"], legacy.outputs["image"])
    assert output.payloads["image"]["spatial"] == legacy.payloads["image"]["spatial"]
    assert output.payloads["image"]["spatial"]["root_id"] != spatial["root_id"]
    assert output.payloads["image"]["sample"] == "retain"
    assert output.payloads["mask"]["video_frame"] == {"frame_index": 4}
    measured = run(
        registry.create("region.measure"),
        {"image": packet_for(output, "image"), "region": packet_for(output, "region")},
    )
    assert measured.outputs["area"] == np.count_nonzero(output.outputs["mask"])
    assert output.outputs["region"]["coordinate_context"] == output.payloads["image"]["spatial"]
    assert measured.payloads["data"]["spatial"] == output.payloads["image"]["spatial"]


def test_raster_regions_survive_composition_inversion_and_affine_transform_without_losing_holes():
    mask = np.zeros((15, 19), np.uint8)
    mask[2:13, 2:17] = 255
    mask[5:9, 6:12] = 0
    region = region_from_mask(mask)
    np.testing.assert_array_equal(rasterize(region, mask.shape), mask)
    inverted = {
        "model": "image_region_v1",
        "image_size": [19, 15],
        "shape": "composite",
        "operation": "invert",
        "children": [region],
    }
    np.testing.assert_array_equal(rasterize(inverted, mask.shape), 255 - mask)
    block = builtin_registry().create(
        "region.transform",
        parameters={
            "matrix": [[1, 0, 1], [0, 1, 2], [0, 0, 1]],
        },
    )
    result = run(block, {"image": np.zeros((20, 22), np.uint8), "region": region})
    expected = np.zeros((20, 22), np.uint8)
    expected[2:17, 1:20] = mask
    np.testing.assert_array_equal(result.outputs["mask"], expected)


@pytest.mark.parametrize(
    "runs",
    [
        [[0, -1, 2]],
        [[0, 0, 20]],
        [[15, 0, 1]],
        [[0, 0, 0]],
        [[1, 3, 2], [1, 4, 2]],
        [[2, 0, 1], [1, 0, 1]],
        [[0, True, 1]],
        [[0, 0, 1.0]],
    ],
)
def test_bad_raster_run_data_is_rejected(runs):
    region = {
        "model": "image_region_v1",
        "image_size": [19, 15],
        "shape": "raster",
        "encoding": "scanline_runs_v1",
        "runs": runs,
    }
    with pytest.raises(ValueError):
        rasterize(region, (15, 19))


def test_region_run_budget_is_shared_across_nested_children():
    region = region_from_mask(np.ones((5, 6), np.uint8))
    combined = {
        "model": "image_region_v1",
        "image_size": [6, 5],
        "shape": "composite",
        "operation": "union",
        "children": [region, copy.deepcopy(region)],
    }
    with pytest.raises(ValueError, match="budget"):
        rasterize(combined, (5, 6), _run_budget=[9])


def test_empty_selection_is_a_valid_failed_inspection_and_mask_only_skips_encoding():
    result = run(
        tool(emit_region=False),
        {"image": np.zeros((11, 13), np.uint8), "mask": np.zeros((11, 13), np.uint8), **fields()},
    )
    assert not result.passed and result.fail_reasons
    assert result.outputs["region"] is NO_VALUE and result.outputs["report"]["selected_pixels"] == 0
    assert result.outputs["image"].shape == (11, 13)


def test_frame_mismatch_and_limits_emit_no_stale_result_or_modified_input():
    image = ValuePacket(np.full((11, 13), 20, np.uint8), {"spatial": frame([13, 11], "camera")})
    mask = ValuePacket(np.ones((11, 13), np.uint8), {"spatial": frame([13, 11], "other")})
    block = tool()
    assert block.execute({"image": image, "mask": mask, **fields()}) is None
    assert "source image frame" in block.error
    for parameters in (
        {"maximum_pixels": 10},
        {"maximum_working_bytes": 1},
        {"maximum_region_runs": 1},
    ):
        block = tool(**parameters)
        assert block.execute({"image": image, **fields()}) is None
        assert not any(block.current_emissions.values())
    assert np.all(image.value == 20)


def test_graphics_off_and_invalid_float_maps_are_contained(monkeypatch):
    from opencv_dataflow import selection_remap

    block = tool()
    block.graphics_mode = "off"
    monkeypatch.setattr(
        selection_remap, "boundaries", lambda *args: pytest.fail("Graphics off drew")
    )
    result = run(block, {"image": np.zeros((11, 13), np.uint8), **fields()})
    assert not result.drawings
    invalid = fields()
    invalid["map_x"][2, 3] = np.nan
    assert block.execute({"image": np.zeros((11, 13), np.uint8), **invalid}) is None
    assert "finite" in block.error and not any(block.current_emissions.values())


def test_xml_and_inspector_offer_dropdowns_without_processing(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    registry, graph = builtin_registry(), Graph()
    block = graph.add(tool(selection_rule="coverage_threshold", minimum_coverage=0.75))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "selection.xml"
    export_project_xml(snapshot, path)
    assert Graph.from_dict(read_project_xml(path), registry).to_dict() == snapshot
    assert "imgproc" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert all(
            isinstance(window.inspector.fields[name], QComboBox)
            for name in ("selection_rule", "selection_sampling", "interpolation", "border")
        )
        window.inspector.fields["border_values"].setText("[20,40,60]")
        assert window.inspector.read_parameters()["border_values"] == [20, 40, 60]
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


@pytest.mark.parametrize("kind", ["cv.remap_selection", "cv.remap_selection_packed"])
def test_remap_supports_finite_border_values_per_stored_channel(kind):
    registry = builtin_registry()
    block = registry.create(kind, parameters={"border_values": [20, 40, 60]})
    source = ImageArray(np.zeros((11, 13, 3), np.uint16), "rgb")
    maps = fields(dx=-2, dy=-2)
    if kind.endswith("packed"):
        maps = registry.create("cv.pack_remap_map").execute(maps).outputs
    result = run(block, {"image": source, **maps})
    np.testing.assert_array_equal(result.outputs["image"][0, 0], [20, 40, 60])
    for invalid in ([1, 2], [1, float("nan"), 3]):
        block.parameters["border_values"] = invalid
        assert block.execute({"image": source, **maps}) is None
        assert not any(block.current_emissions.values())


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_coverage": 1.1},
        {"minimum_coverage": float("nan")},
        {"maximum_region_runs": 1.5},
        {"maximum_pixels": True},
        {"emit_region": "yes"},
    ],
)
def test_invalid_selection_sampling_parameters_are_contained(parameters):
    block = tool(**parameters)
    assert block.execute({"image": np.zeros((11, 13), np.uint8), **fields()}) is None
    assert block.error and not any(block.current_emissions.values())
