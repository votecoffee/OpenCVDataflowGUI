"""Native-property parity, analytic geometry and ordinary collection interoperability."""

import json

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.skimage_regions import PROPERTIES
from opencv_dataflow.tickets import TicketClock

pytest.importorskip("skimage.measure")


def run(inputs, **parameters):
    block = builtin_registry().create("skimage.region_properties", parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def labels():
    image = np.zeros((28, 40), np.int32)
    image[3:9, 4:14] = 7
    image[13:25, 22:36] = 9
    image[16:21, 26:31] = 0
    return image


def assert_native(value, expected):
    if isinstance(expected, tuple) and isinstance(expected[0], slice):
        assert value == [{"start": s.start, "stop": s.stop, "step": s.step} for s in expected]
    else:
        np.testing.assert_allclose(
            np.asarray(value, float), np.asarray(expected, float), equal_nan=True, atol=1e-10
        )


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_entire_standard_2d_catalog_matches_native_measurements(color, cache):
    measure = pytest.importorskip("skimage.measure")
    from skimage.measure._regionprops import COL_DTYPES

    assert set(PROPERTIES) == set(COL_DTYPES)
    plane = labels()
    yy, xx = np.indices(plane.shape)
    values = (100 + xx * 3 + yy * 7).astype(np.uint16)
    intensity = (
        ImageArray(np.stack((values, values * 2, values * 3), axis=2), "rgb") if color else values
    )
    result = run(
        {"labels": plane, "intensity_image": intensity},
        property_set="comprehensive",
        include_coordinates=True,
        include_images=True,
        cache=cache,
    )
    rows = result.outputs["measurements"]["rows"]
    for row, native in zip(rows, measure.regionprops(plane, intensity, cache=cache), strict=True):
        assert set(row["properties"]) == set(PROPERTIES)
        for name, value in row["properties"].items():
            assert_native(value, getattr(native, name))
    assert result.outputs["measurements"]["intensity_channels"] == (
        ["R", "G", "B"] if color else ["Gray"]
    )
    json.dumps(result.outputs["collection"], allow_nan=False)
    json.dumps(result.outputs["measurements"], allow_nan=False)


def test_analytic_area_hole_centroid_and_pixel_metric_units_with_spacing():
    plane = labels()
    result = run(
        {"labels": plane},
        property_set="comprehensive",
        spacing_yx=[2, 3],
        offset_yx=[10, 20],
        units="mm",
    )
    first, ring = result.outputs["collection"]["items"]
    assert first["metrics"]["area"] == 60
    assert first["metrics"]["center_x"] == 8.5 and first["metrics"]["center_y"] == 5.5
    assert first["geometry"]["bounds"] == [4, 3, 10, 6]
    assert first["properties"]["area"] == 360
    np.testing.assert_array_equal(first["properties"]["centroid"], [21, 45.5])
    assert first["properties"]["perimeter"] is None and "perimeter" in first["unavailable"]
    assert first["metrics"]["perimeter"] == 28
    assert first["metrics"]["hu_1"] is not None
    assert ring["properties"]["euler_number"] == 0
    assert ring["properties"]["area_filled"] == 12 * 14 * 6
    assert ring["metrics"]["area"] == 12 * 14 - 25


def test_sparse_signed_and_zero_label_ids_do_not_allocate_by_maximum_id():
    plane = labels()
    plane[plane == 7] = 2_000_000_000
    plane[plane == 9] = -4
    result = run({"labels": plane}, ignored_labels=[])
    assert [row["id"] for row in result.outputs["collection"]["items"]] == [-4, 0, 2_000_000_000]
    assert [row["label"] for row in result.outputs["measurements"]["rows"]] == [
        -4,
        0,
        2_000_000_000,
    ]


@pytest.mark.parametrize("connectivity,count", [(4, 2), (8, 1)])
def test_binary_connected_components_connectivity_and_boolean_masks(connectivity, count):
    image = np.zeros((12, 12), bool)
    image[3:5, 3:5] = True
    image[5:7, 5:7] = True
    result = run({"labels": image}, input_mode="binary", connectivity=connectivity)
    assert result.outputs["count"] == count and result.outputs["labels"].dtype == np.int32
    assert sum(item["metrics"]["area"] for item in result.outputs["collection"]["items"]) == 8


def test_selection_filter_sort_and_mask_reconstruction_use_original_ids_and_pixels():
    registry = builtin_registry()
    plane = labels()
    scope = np.full(plane.shape, 255, np.uint8)
    scope[3, :] = 0
    context = {"spatial": frame([40, 28], "camera"), "ticket": TicketClock().issue("camera")}
    result = run(
        {"labels": ValuePacket(plane, context), "mask": ValuePacket(scope, context)},
        property_set="comprehensive",
    )
    assert result.outputs["collection"]["items"][0]["metrics"]["area"] == 50
    filter_ = registry.create(
        "collection.filter",
        parameters={
            "metric": "custom",
            "custom_metric": "regionprops.euler_number",
            "use_minimum": True,
            "minimum": 0,
            "use_maximum": True,
            "maximum": 0,
        },
    )
    filtered = filter_.execute({"collection": ValuePacket(result.outputs["collection"], context)})
    assert filtered is not None, filter_.error
    assert [item["id"] for item in filtered.outputs["collection"]["items"]] == [9]
    sort = registry.create(
        "collection.sort",
        parameters={
            "metric": "custom",
            "custom_metric": "regionprops.area",
            "direction": "descending",
            "secondary_metric": "custom",
            "secondary_custom_metric": "regionprops.eccentricity",
        },
    )
    sorted_ = sort.execute({"collection": result.outputs["collection"]})
    assert sorted_ is not None, sort.error
    assert [item["id"] for item in sorted_.outputs["collection"]["items"]] == [9, 7]
    restore = registry.create("inspect.mask_from_labels")
    mask = restore.execute(
        {
            "labels": ValuePacket(result.outputs["labels"], context),
            "collection": ValuePacket(filtered.outputs["collection"], context),
            "mask": ValuePacket(scope, context),
        }
    )
    assert mask is not None, restore.error
    np.testing.assert_array_equal(
        mask.outputs["mask"], np.where((plane == 9) & (scope != 0), 255, 0)
    )
    assert mask.payloads["mask"]["ticket"] == context["ticket"]


def test_missing_intensity_zero_weights_and_strict_unavailable_policy():
    plane = labels()
    result = run({"labels": plane}, property_set="custom", properties=["intensity_mean"])
    assert result.outputs["measurements"]["rows"][0]["unavailable"] == {
        "intensity_mean": "requires_intensity_image"
    }
    result = run(
        {"labels": plane, "intensity_image": np.zeros(plane.shape)},
        property_set="custom",
        properties=["centroid_weighted"],
    )
    assert result.outputs["measurements"]["rows"][0]["properties"]["centroid_weighted"] == [
        None,
        None,
    ]
    block = builtin_registry().create(
        "skimage.region_properties",
        parameters={
            "property_set": "custom",
            "properties": ["perimeter"],
            "spacing_yx": [1, 2],
            "unavailable": "error",
        },
    )
    assert block.execute({"labels": plane}) is None
    assert "perimeter" in block.error and not block.current_emissions


@pytest.mark.parametrize("include_alpha", [False, True])
def test_raw_uint16_color_channel_statistics_and_alpha(include_alpha):
    plane = labels()
    intensity = ImageArray(np.tile([10000, 20000, 40000, 12345], (*plane.shape, 1)), "rgba")
    intensity = intensity.astype(np.uint16)
    result = run(
        {"labels": plane, "intensity_image": intensity},
        include_alpha=include_alpha,
        property_set="custom",
        properties=["intensity_mean", "intensity_std"],
    )
    row = result.outputs["measurements"]["rows"][0]["properties"]
    assert row["intensity_mean"] == [10000, 20000, 40000] + ([12345] if include_alpha else [])
    assert row["intensity_std"] == [0] * (4 if include_alpha else 3)
    assert result.outputs["collection"]["items"][0]["metrics"]["intensity"] == pytest.approx(19290)


def test_frame_and_ticket_checks_and_explicit_pixel_grid_override():
    plane = labels()
    clock = TicketClock()
    a = {"spatial": frame([40, 28], "a"), "ticket": clock.issue("a")}
    b = {"spatial": frame([40, 28], "b"), "ticket": clock.issue("b")}
    inputs = {
        "labels": ValuePacket(plane, a),
        "intensity_image": ValuePacket(plane.astype(float), b),
    }
    block = builtin_registry().create("skimage.region_properties")
    assert block.execute(inputs) is None and "align" in block.error
    block.parameters["intensity_alignment"] = "pixel_grid"
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.payloads["collection"]["ticket"] == a["ticket"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"spacing_yx": [0, 1]},
        {"offset_yx": [np.nan, 0]},
        {"properties": ["not_a_property"]},
        {"ignored_labels": [True]},
        {"cache": "yes"},
    ],
)
def test_parameter_validation(parameters):
    block = builtin_registry().create("skimage.region_properties", parameters=parameters)
    assert block.execute({"labels": labels()}) is None
    assert block.error_kind == "invalid"


def test_output_and_region_count_budgets_fail_before_publication():
    block = builtin_registry().create(
        "skimage.region_properties", parameters={"maximum_regions": 1}
    )
    assert block.execute({"labels": labels()}) is None
    assert "Too many measured regions" in block.error
    block.parameters.update(maximum_regions=10, maximum_values=100, include_coordinates=True)
    assert block.execute({"labels": labels()}) is None
    assert "budget" in block.error and not block.current_emissions


@pytest.mark.parametrize("shape", ["point", "line", "empty"])
def test_degenerate_regions_are_json_safe_and_valid_collections(shape):
    from opencv_dataflow.objects import validate_collection

    plane = np.zeros((15, 20), np.int32)
    if shape == "point":
        plane[5, 6] = 1
    elif shape == "line":
        plane[5, 6:15] = 1
    result = run({"labels": plane}, property_set="comprehensive")
    validate_collection(result.outputs["collection"])
    json.dumps(result.outputs["collection"], allow_nan=False)


def test_new_blocks_roundtrip_parameters_as_xml(tmp_path):
    from opencv_dataflow.block_xml import save_block_xml
    from opencv_dataflow.project_xml import read_block_xml

    registry = builtin_registry()
    for kind, parameters in (
        (
            "skimage.region_properties",
            {"properties": ["moments_hu"], "property_set": "custom", "spacing_yx": [2, 3]},
        ),
        ("skimage.multi_otsu", {"classes": 4, "selected_classes": [0, 3]}),
        ("skimage.global_threshold", {"method": "li", "li_initial": "quantile"}),
        ("skimage.local_threshold", {"window_yx": [9, 15], "method": "niblack"}),
    ):
        block = registry.create(kind, parameters=parameters)
        path = tmp_path / ("preset." + kind + ".xml")
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
