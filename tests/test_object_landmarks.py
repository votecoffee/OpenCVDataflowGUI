"""Explicit landmark columns, independent score/coordinate references and safe skeletons."""

import copy

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.dnn_blocks import DNNInference, preprocess
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE


@pytest.fixture
def make():
    blocks = []
    registry = builtin_registry()

    def create(**settings):
        block = registry.create(
            "ml.decode_landmarks",
            parameters={
                "format": "xy_score",
                "output_name": "head",
                "coordinate_columns": 5,
                "coordinate_score_columns": [3, 4],
                "names": ["a", "b", "c"],
                "connections": [[0, 1], [1, 2]],
                **settings,
            },
        )
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def inputs(rows=None, layout="rows", score=None):
    image = np.zeros((80, 120, 3), np.uint16)
    _, geometry = preprocess({"image": image}, {**DNNInference.defaults, "width": 24, "height": 16})
    rows = np.array(
        [[2, 3, -500, 0.9, 0.8], [10, 5, 900, 0.95, 0.1], [18, 11, 0, 0.6, 0.75]]
        if rows is None
        else rows,
        np.float32,
    )
    if layout == "channels_first":
        rows = rows.T
    elif layout == "flat_rows":
        rows = rows.reshape(1, -1)
    tensors = {"head": tensor_record(rows, 10000, allow_empty=True)}
    if score is not None:
        tensors["object"] = tensor_record(np.array(score, np.float32), 10000)
    return {"image": image, "preprocessing": geometry, "tensors": tensors}


@pytest.mark.parametrize("layout", ["rows", "channels_first", "flat_rows"])
@pytest.mark.parametrize(
    "origin,shift", [("pixel_centers", 2), ("pixel_edges", -0.5), ("scaled_indices", 0)]
)
@pytest.mark.parametrize("normalized", [False, True])
def test_direct_columns_and_mapping_match_independent_source_coordinate_formula(
    make, layout, origin, shift, normalized
):
    rows = np.array(
        [[2, 3, -500, 0.9, 0.8], [10, 5, 900, 0.95, 0.1], [18, 11, 0, 0.6, 0.75]], np.float32
    )
    if normalized:
        rows[:, :2] /= [24, 16]
    args = inputs(rows, layout)
    original = copy.deepcopy(args["tensors"])
    block = make(
        coordinate_layout=layout,
        coordinate_origin=origin,
        coordinate_units="normalized" if normalized else "network_pixels",
        minimum_confidence=0.5,
        require_all=False,
        minimum_count=2,
    )
    result = block.execute(args)
    assert result is not None and result.passed, block.error
    expected = np.array([[2, 3], [10, 5], [18, 11]]) * 5 + shift
    np.testing.assert_allclose(
        [row["position"] for row in result.outputs["landmarks"]["items"]], expected, atol=1e-5
    )
    assert result.outputs["indices"] == [0, 2] and result.outputs["points"] is NO_VALUE
    assert result.outputs["object_confidence"] is NO_VALUE
    assert result.outputs["report"]["score_columns"] == [3, 4]
    assert args["tensors"] == original
    np.testing.assert_array_equal(result.outputs["image"], args["image"])


@pytest.mark.parametrize("activation", ["probability", "sigmoid"])
@pytest.mark.parametrize("combination", ["minimum", "product", "mean"])
def test_confidence_components_are_activated_before_combining_and_threshold_is_inclusive(
    make, activation, combination
):
    rows = np.array(
        [[2, 3, -500, 0.9, 0.8], [10, 5, 900, 0.95, 0.1], [18, 11, 0, 0.6, 0.75]], np.float32
    )
    if activation == "sigmoid":
        rows[:, 3:] = np.log(rows[:, 3:] / (1 - rows[:, 3:]))
    components = rows[:, 3:].astype(float)
    if activation == "sigmoid":
        components = 1 / (1 + np.exp(-components))
    expected = {"minimum": np.min, "product": np.prod, "mean": np.mean}[combination](
        components, axis=1
    )
    block = make(
        score_activation=activation,
        coordinate_score_combination=combination,
        minimum_confidence=float(expected[0]),
        minimum_count=1,
        require_all=False,
    )
    result = block.execute(inputs(rows))
    assert result is not None and result.passed, block.error
    items = result.outputs["landmarks"]["items"]
    np.testing.assert_allclose([item["confidence"] for item in items], expected, atol=1e-15)
    np.testing.assert_allclose(
        [item["confidence_components"] for item in items], components, atol=1e-15
    )
    assert items[0]["accepted"]


@pytest.mark.parametrize("graphics", ["show", "hide"])
def test_skeleton_has_no_links_to_missing_endpoints_and_optional_markers_are_suppressed(
    make, graphics
):
    block = make(
        rejected_graphics=graphics,
        require_all=False,
        minimum_count=2,
        connections=[[0, 1], [1, 2], [0, 2]],
    )
    result = block.execute(inputs())
    assert result is not None and result.passed
    report = result.outputs["report"]
    assert report["drawn_links"] == [[0, 2]]
    assert report["drawn_points"] == ([0, 1, 2] if graphics == "show" else [0, 2])
    assert result.outputs["landmarks"]["items"][1]["name"] == "b"
    assert not result.outputs["landmarks"]["items"][1]["accepted"]
    block.set_graphics_mode("off")
    lean = block.execute(inputs())
    assert lean.outputs["landmarks"] == result.outputs["landmarks"]
    assert not lean.drawings and lean.outputs["report"]["drawn_links"] == []


@pytest.mark.parametrize("activation,value", [("probability", 0.5), ("sigmoid", 0.0)])
def test_object_gate_is_independent_inclusive_and_cannot_pass_without_object(
    make, activation, value
):
    block = make(
        object_score_name="object",
        object_score_activation=activation,
        require_all=False,
        minimum_count=0,
        rejected_graphics="hide",
    )
    result = block.execute(inputs(score=[value]))
    assert result is not None and result.passed and result.outputs["object_confidence"] == 0.5
    block.parameters["minimum_object_confidence"] = 0.51
    result = block.execute(inputs(score=[value]))
    assert result is not None and not result.passed and result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "landmark.object_confidence"
    assert "0.51" in result.fail_reasons[0]["message"]
    assert result.outputs["points"] is NO_VALUE
    assert (
        result.outputs["report"]["drawn_points"] == []
        and result.outputs["report"]["drawn_links"] == []
    )
    assert all(
        item["reason"] == "low_object_confidence" for item in result.outputs["landmarks"]["items"]
    )


@pytest.mark.parametrize("change", ["shape", "missing", "nonfinite", "budget"])
def test_invalid_object_head_has_no_current_emission_and_recovers(make, change):
    block = make(object_score_name="object")
    valid = inputs(score=[0.9])
    assert block.execute(valid) is not None
    invalid = inputs(score=[0.9])
    if change == "shape":
        invalid["tensors"]["object"] = tensor_record(np.array([0.8, 0.9], np.float32), 100)
    elif change == "missing":
        del invalid["tensors"]["object"]
    elif change == "nonfinite":
        invalid["tensors"]["object"]["values"] = [float("nan")]
    else:
        block.parameters["maximum_tensor_elements"] = 15
    assert block.execute(invalid) is None and not block.current_emissions
    block.parameters["maximum_tensor_elements"] = 8_000_000
    assert block.execute(valid) is not None, block.error


@pytest.mark.parametrize(
    "settings",
    [
        {"coordinate_xy_columns": [0, 0]},
        {"coordinate_xy_columns": [0, "1"]},
        {"coordinate_score_columns": [0, 4]},
        {"coordinate_score_columns": [3, 3]},
        {"coordinate_score_columns": [5]},
        {"coordinate_columns": 2},
        {"coordinate_score_combination": "unknown"},
        {"minimum_object_confidence": 2},
        {"object_score_name": 123},
        {"rejected_graphics": "unknown"},
    ],
)
def test_new_controls_have_configuration_validation(make, settings):
    with pytest.raises(ValidationError):
        make(**settings).validate_parameters()


@pytest.mark.parametrize("shape", [(2, 15), (14,), (3, 5)])
def test_flat_rows_refuse_multiple_images_incomplete_rows_and_implicit_reshape(make, shape):
    block = make(coordinate_layout="flat_rows")
    args = inputs()
    args["tensors"]["head"] = tensor_record(np.zeros(shape, np.float32), 1000)
    assert block.execute(args) is None and not block.current_emissions


def test_scaled_indices_respect_letterbox_offset_and_nonuniform_rounded_resize(make):
    args = inputs()
    args["image"] = np.zeros((80, 160, 3), np.uint16)
    _, args["preprocessing"] = preprocess(
        {"image": args["image"]},
        {**DNNInference.defaults, "width": 24, "height": 16, "resize_mode": "letterbox"},
    )
    result = make(coordinate_origin="scaled_indices").execute(args)
    assert result is not None
    np.testing.assert_allclose(
        [item["position"] for item in result.outputs["landmarks"]["items"]],
        [[2 / 0.15, (3 - 2) / 0.15], [10 / 0.15, (5 - 2) / 0.15], [18 / 0.15, (11 - 2) / 0.15]],
        atol=1e-12,
    )


def test_reordered_xy_and_confidence_columns_preserve_landmark_names(make):
    args = inputs()
    original = args["tensors"]["head"]
    rows = np.array(original["values"], np.float32).reshape(3, 5)
    args["tensors"]["head"] = tensor_record(rows[:, [3, 2, 4, 1, 0]], 100)
    block = make(coordinate_xy_columns=[4, 3], coordinate_score_columns=[0, 2])
    result = block.execute(args)
    assert result is not None and result.outputs["indices"] == [0, 2]
    assert [item["name"] for item in result.outputs["landmarks"]["items"]] == ["a", "b", "c"]
    np.testing.assert_allclose(
        [item["position"] for item in result.outputs["landmarks"]["items"]],
        [[12, 17], [52, 27], [92, 57]],
        atol=1e-12,
    )
