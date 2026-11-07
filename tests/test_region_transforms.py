import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.selection import rasterize


def run(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state == "pass", block.error
    return block.last_result.outputs


def source():
    image = np.zeros((60, 80), np.uint8)
    shape = run("region.shape", {"image": image}, rectangle=[20, 12, 16, 18])
    return image, shape


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("cv.crop", {"x": 7, "y": 9, "width": 45, "height": 35}),
        ("cv.resize", {"width": 120, "height": 90}),
        ("cv.flip", {"direction": "both"}),
        ("cv.rotate90", {"rotation": "clockwise"}),
        ("cv.copy_border", {"top": 3, "bottom": 7, "left": 5, "right": 11}),
    ],
)
def test_image_geometry_preserves_selection_coordinates(kind, parameters):
    image, shape = source()
    geometry = run(kind, {"image": image}, **parameters)
    output = run(
        "region.transform",
        {"image": geometry["image"], "region": shape["region"], "transform": geometry["geometry"]},
    )
    if kind == "cv.crop":
        expected = shape["mask"][9:44, 7:52]
    elif kind == "cv.resize":
        expected = cv2.resize(shape["mask"], (120, 90), interpolation=cv2.INTER_NEAREST_EXACT)
    elif kind == "cv.flip":
        expected = cv2.flip(shape["mask"], -1)
    elif kind == "cv.rotate90":
        expected = cv2.rotate(shape["mask"], cv2.ROTATE_90_CLOCKWISE)
    else:
        expected = cv2.copyMakeBorder(shape["mask"], 3, 7, 5, 11, cv2.BORDER_CONSTANT, value=0)
    np.testing.assert_array_equal(output["mask"], expected)
    np.testing.assert_allclose(
        np.asarray(output["transform"]["source_to_target"])
        @ np.asarray(geometry["geometry"]["output_to_input"]),
        np.eye(3),
    )
    assert output["transform"]["source_size"] == [80, 60]
    assert output["transform"]["target_size"] == [expected.shape[1], expected.shape[0]]
    restored = json.loads(json.dumps(output["region"]))
    np.testing.assert_array_equal(rasterize(restored, expected.shape), expected)
    measured = run("region.measure", {"image": geometry["image"], "region": restored})
    assert measured["area"] == np.count_nonzero(expected)


@pytest.mark.parametrize(
    "payload",
    [
        [[1, 0, 7], [0, 1, 5]],
        {"model": "image_affine_v1", "source_to_target": [[1, 0, 7], [0, 1, 5], [0, 0, 1]]},
    ],
)
def test_affine_registration_forward_inverse_and_point_compatibility(payload):
    image, shape = source()
    output = run(
        "region.transform", {"image": image, "region": shape["region"], "transform": payload}
    )
    assert run("region.measure", {"image": image, "region": output["region"]})["bounds"] == [
        27,
        17,
        16,
        18,
    ]
    points = run("cv.transform_points", {"points": [[20, 12]], "transform": output["transform"]})
    assert points["points"] == [[27.0, 17.0]]
    inverse = run(
        "region.transform",
        {"image": image, "region": output["region"], "transform": payload},
        direction="inverse",
    )
    np.testing.assert_array_equal(inverse["mask"], shape["mask"])


def test_arbitrary_rotation_and_projective_matrices_match_native_projection():
    image, shape = source()
    rotated = run("cv.rotate", {"image": image}, angle_degrees=17.0)
    output = run(
        "region.transform",
        {"image": rotated["image"], "region": shape["region"], "transform": rotated["geometry"]},
    )
    matrix = np.linalg.inv(rotated["geometry"]["output_to_input"])
    expected = cv2.warpPerspective(shape["mask"], matrix, (80, 60), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(output["mask"], expected)
    matrix = [[1, 0.1, 3], [0.03, 1, 4], [0.001, 0.002, 1]]
    projected = run("region.transform", {"image": image, "region": shape["region"]}, matrix=matrix)
    np.testing.assert_array_equal(
        projected["mask"],
        cv2.warpPerspective(shape["mask"], np.array(matrix), (80, 60), flags=cv2.INTER_NEAREST),
    )


def test_inverse_geometry_uses_output_region_and_original_image_dimensions():
    image, shape = source()
    crop = run("cv.crop", {"image": image}, x=7, y=9, width=45, height=35)
    forward = run(
        "region.transform",
        {"image": crop["image"], "region": shape["region"], "transform": crop["geometry"]},
    )
    inverse = run(
        "region.transform",
        {"image": image, "region": forward["region"], "transform": crop["geometry"]},
        direction="inverse",
    )
    np.testing.assert_array_equal(inverse["mask"], shape["mask"])


def test_nested_composite_holes_survive_translation_and_chained_crop_flip():
    image, shape = source()
    hole = run("region.shape", {"image": image}, rectangle=[24, 16, 4, 4])
    ring = run(
        "region.combine",
        {"image": image, "region_a": shape["region"], "region_b": hole["region"]},
        operation="difference",
    )
    crop = run("cv.crop", {"image": image}, x=10, y=10, width=50, height=40)
    first = run(
        "region.transform",
        {"image": crop["image"], "region": ring["region"], "transform": crop["geometry"]},
    )
    flipped = run("cv.flip", {"image": crop["image"]}, direction="horizontal")
    second = run(
        "region.transform",
        {"image": flipped["image"], "region": first["region"], "transform": flipped["geometry"]},
    )
    np.testing.assert_array_equal(second["mask"], cv2.flip(ring["mask"][10:50, 10:60], 1))


def test_linear_sampling_threshold_stays_binary_and_controls_coverage():
    image, shape = source()
    matrix = [[1, 0, 0.4], [0, 1, 0.4], [0, 0, 1]]
    samples = cv2.warpPerspective(
        shape["mask"].astype(np.float32) / 255, np.array(matrix), (80, 60), flags=cv2.INTER_LINEAR
    )
    for threshold in (0.25, 0.5, 0.9):
        output = run(
            "region.transform",
            {"image": image, "region": shape["region"]},
            matrix=matrix,
            interpolation="linear",
            threshold=threshold,
        )
        np.testing.assert_array_equal(output["mask"], np.where(samples >= threshold, 255, 0))


def test_inverted_source_does_not_select_new_border_pixels():
    image, _ = source()
    region = run("region.shape", {"image": image}, rectangle=[20, 12, 16, 18], invert=True)
    transformed = run(
        "region.transform",
        {"image": image, "region": region["region"]},
        matrix=[[1, 0, 10], [0, 1, 0]],
    )
    assert not transformed["mask"][:, :10].any()
    assert transformed["mask"][0, 10] == 255


@pytest.mark.parametrize(
    "parameters",
    [
        {"matrix": [[1, 0], [0, 1]]},
        {"matrix": [[0, 0, 0], [0, 0, 0]]},
        {"threshold": 0},
        {"threshold": 1.1},
        {"interpolation": "cubic"},
    ],
)
def test_invalid_transform_controls_clear_results(parameters):
    image, shape = source()
    block = builtin_registry().create("region.transform", parameters=parameters)
    block.execute({"image": image, "region": shape["region"]})
    assert block.state == "error" and block.last_result is None


@pytest.mark.parametrize(
    "payload",
    [
        {"output_to_input": np.eye(3).tolist()},
        {"output_to_input": np.eye(3).tolist(), "input_size": [5, 5], "output_size": [80, 60]},
        {"source_to_target": np.eye(3).tolist(), "target_size": [10, 10]},
        {"source_to_target": np.eye(3).tolist(), "output_to_input": np.eye(3).tolist()},
        {"pixel_to_world": np.eye(3).tolist()},
    ],
)
def test_ambiguous_or_inconsistent_mapping_metadata_is_rejected(payload):
    image, shape = source()
    block = builtin_registry().create("region.transform")
    block.execute({"image": image, "region": shape["region"], "transform": payload})
    assert block.state == "error" and block.last_result is None


def test_transformed_expression_validates_source_dimensions_before_allocation():
    _, shape = source()
    expression = {
        "model": "image_region_v1",
        "image_size": [80, 60],
        "shape": "transformed",
        "source": {**shape["region"], "image_size": [1000000, 1000000]},
        "source_to_target": np.eye(3).tolist(),
    }
    with pytest.raises(ValueError, match="dimension"):
        rasterize(expression, (60, 80))


@pytest.mark.local_examples
def test_saved_transform_project_repeats_and_survives_toolblock_reuse(tmp_path):
    from opencv_dataflow.core import Graph
    from opencv_dataflow.examples import transformed_regions_graph
    from opencv_dataflow.toolblocks import encapsulate

    registry = builtin_registry()
    graph = Graph.load(
        "examples/custom tools/transformed_regions/transformed_regions.json", registry
    )
    assert graph.to_dict() == transformed_regions_graph(registry).to_dict()
    previous = None
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        assert graph.blocks["tracked_components"].last_result.outputs["count"] == 1
        mask = graph.blocks["tracked_rotate_region"].last_output_images["mask"]
        if previous is not None:
            np.testing.assert_array_equal(mask, previous)
        previous = mask.copy()
    block = encapsulate(
        graph,
        [
            "tracked_crop",
            "tracked_crop_region",
            "tracked_resize",
            "tracked_resize_region",
            "tracked_rotate",
            "tracked_rotate_region",
        ],
        "Image and region transforms",
        registry,
    )
    block.position = [460.0, 200.0]
    path = tmp_path / "transformed.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    loaded.run()
    assert all(b.state == "pass" for b in loaded.blocks.values())
    assert loaded.blocks[block.id].position == [460.0, 200.0]
    assert loaded.blocks["tracked_components"].last_result.outputs["count"] == 1


def test_inverse_mapping_can_consume_transform_output_size_metadata():
    image, shape = source()
    larger = np.zeros((90, 120), np.uint8)
    forward = run(
        "region.transform",
        {"image": larger, "region": shape["region"]},
        matrix=[[1, 0, 7], [0, 1, 5]],
    )
    inverse = run(
        "region.transform",
        {"image": image, "region": forward["region"], "transform": forward["transform"]},
        direction="inverse",
    )
    np.testing.assert_array_equal(inverse["mask"], shape["mask"])
