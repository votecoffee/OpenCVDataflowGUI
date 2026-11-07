import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import collection, line_item
from opencv_dataflow.payloads import packet_for
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED


def run(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block


def parts():
    image = np.zeros((160, 240), np.uint8)
    image[10:30, 10:50] = 255
    image[50:100, 70:90] = 255
    cv2.circle(image, (160, 70), 25, 255, -1)
    cv2.circle(image, (160, 70), 9, 0, -1)
    return image


def test_blob_pixel_geometry_holes_hu_and_exact_rendered_mask():
    image = parts()
    block = run("inspect.blobs", {"image": image}, min_area=1)
    data = block.last_result.outputs["collection"]
    assert block.state == "pass" and data["kind"] == "blobs" and data["count"] == 3
    items = sorted(data["items"], key=lambda item: item["metrics"]["center_x"])
    first = items[0]["metrics"]
    assert first["area"] == 800 and first["width"] == 40 and first["height"] == 20
    assert first["center_x"] == 29.5 and first["center_y"] == 19.5
    assert first["aspect_ratio"] == 2 and first["angle"] == pytest.approx(0)
    assert first["solidity"] == pytest.approx(1)
    expected = cv2.HuMoments(cv2.moments((image[10:30, 10:50] > 0).astype(np.uint8), True)).reshape(
        -1
    )
    np.testing.assert_allclose([first[f"hu_{i}"] for i in range(1, 8)], expected)
    assert items[1]["metrics"]["angle"] == pytest.approx(90)
    assert len(items[2]["geometry"]["contours"]) == 2
    drawn = run(
        "collection.draw",
        {"image": image, "collection": packet_for(block.last_result, "collection")},
    )
    np.testing.assert_array_equal(drawn.last_output_images["mask"], image)


def test_contour_geometric_area_offset_and_measurements_match_native():
    image = ImageArray(
        parts(),
        "gray",
        {
            "spatial": {
                "root_id": "collection-test",
                "root_size": [240, 160],
                "image_size": [240, 160],
                "to_root": np.eye(3).tolist(),
            }
        },
    )
    block = run(
        "inspect.contours", {"image": image}, min_area=1, expected_count=3, offset=[100, 200]
    )
    data = block.last_result.outputs["collection"]
    assert data["kind"] == "contours" and data["origin"] == [100, 200]
    native, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(100, 200)
    )
    for item in data["items"]:
        contour = native[item["id"]]
        assert item["metrics"]["area"] == cv2.contourArea(contour)
        assert item["metrics"]["perimeter"] == cv2.arcLength(contour, True)
        hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
        np.testing.assert_allclose([item["metrics"][f"hu_{i}"] for i in range(1, 8)], hu)
    drawn = run(
        "collection.draw",
        {
            "image": block.last_inputs["image"],
            "collection": packet_for(block.last_result, "collection"),
        },
    )
    assert drawn.last_result.outputs["count"] == 3
    assert drawn.last_output_images["mask"][15, 15] == 255
    assert drawn.last_output_images["mask"][0, 0] == 0
    np.testing.assert_array_equal(
        block.last_result.payloads["collection"]["spatial"]["to_root"],
        [[1, 0, -100], [0, 1, -200], [0, 0, 1]],
    )


def test_numeric_filter_chaining_size_aspect_angle_and_exclusive_bounds():
    image = parts()
    blobs = run("inspect.blobs", {"image": image}, min_area=1).last_result.outputs["collection"]
    width = run("collection.filter", {"collection": blobs}, metric="width", minimum=30.0)
    assert (
        width.last_result.outputs["count"] == 2 and width.last_result.outputs["rejected_count"] == 1
    )
    graphic = width.last_graphics["collection filter"][40:]
    assert np.any(np.all(graphic[:, 60:100] == REJECTED, axis=-1))
    assert np.any(np.all(graphic[:, 120:200] == ACCEPTED, axis=-1))
    aspect = run(
        "collection.filter",
        {"collection": width.last_result.outputs["collection"]},
        metric="aspect_ratio",
        minimum=1.5,
    )
    assert aspect.last_result.outputs["count"] == 1
    assert aspect.last_result.outputs["points"] == [[29.5, 19.5]]
    angle = run(
        "collection.filter",
        {"collection": blobs},
        metric="angle",
        minimum=80.0,
        use_maximum=True,
        maximum=100.0,
    )
    assert angle.last_result.outputs["count"] == 1
    exclusive = run(
        "collection.filter",
        {"collection": blobs},
        metric="area",
        minimum=800.0,
        include_minimum=False,
        use_maximum=True,
        maximum=1000.0,
    )
    assert exclusive.last_result.outputs["count"] == 1
    rejected = run(
        "collection.filter",
        {"collection": blobs},
        metric="area",
        minimum=800.0,
        include_minimum=False,
        use_maximum=True,
        maximum=1000.0,
        invert_range=True,
    )
    assert rejected.last_result.outputs["count"] == 2


def test_hu_and_signed_log_filters_match_selected_native_descriptor():
    image = parts()
    blobs = run("inspect.blobs", {"image": image}, min_area=1).last_result.outputs["collection"]
    metric = blobs["items"][0]["metrics"]["hu_log_1"]
    output = run(
        "collection.filter",
        {"collection": blobs},
        metric="hu_log_1",
        minimum=metric - 1e-8,
        maximum=metric + 1e-8,
        use_maximum=True,
    )
    assert output.last_result.outputs["count"] >= 1
    assert blobs["items"][0]["id"] in [
        item["id"] for item in output.last_result.outputs["collection"]["items"]
    ]


def lines():
    image = np.zeros((100, 120), np.uint8)
    return collection(
        "lines",
        [
            line_item(s, i)
            for i, s in enumerate(
                [[10, 10, 50, 10], [10, 30, 50, 30], [60, 10, 60, 70], [20, 80, 60, 80]]
            )
        ],
        image,
        {},
    )


def test_stable_sort_secondary_keys_distance_and_ids_are_preserved():
    data = lines()
    original = copy.deepcopy(data)
    output = run("collection.sort", {"collection": data}, metric="length", direction="descending")
    assert [item["id"] for item in output.last_result.outputs["collection"]["items"]] == [
        2,
        0,
        1,
        3,
    ]
    output = run(
        "collection.sort",
        {"collection": data},
        metric="length",
        direction="descending",
        secondary_metric="center_y",
        secondary_direction="descending",
    )
    assert [item["id"] for item in output.last_result.outputs["collection"]["items"]] == [
        2,
        3,
        1,
        0,
    ]
    output = run(
        "collection.sort",
        {"collection": data},
        metric="distance",
        direction="ascending",
        reference_point=[30.0, 30.0],
    )
    assert output.last_result.outputs["collection"]["items"][0]["id"] == 1
    assert data == original


def test_undefined_metrics_use_explicit_filter_and_sort_policies():
    data = lines()
    assert all(item["metrics"]["area"] is None for item in data["items"])
    assert (
        run("collection.filter", {"collection": data}, metric="hu_1").last_result.outputs["count"]
        == 0
    )
    assert (
        run(
            "collection.filter", {"collection": data}, metric="hu_1", missing="keep"
        ).last_result.outputs["count"]
        == 4
    )
    data["items"][0]["metrics"]["area"] = 10.0
    for direction in ("ascending", "descending"):
        output = run("collection.sort", {"collection": data}, metric="area", direction=direction)
        assert output.last_result.outputs["collection"]["items"][0]["id"] == 0
        output = run(
            "collection.sort",
            {"collection": data},
            metric="area",
            direction=direction,
            missing="first",
        )
        assert output.last_result.outputs["collection"]["items"][-1]["id"] == 0
    for kind in ("collection.filter", "collection.sort"):
        block = builtin_registry().create(kind, parameters={"metric": "hu_1", "missing": "error"})
        block.execute({"collection": data})
        assert block.state == "error" and block.last_result is None


def test_collection_region_filter_preserves_reported_origin_coordinates():
    image = parts()
    contours = run("inspect.contours", {"image": image}, min_area=1, offset=[100, 200])
    shape = run("region.shape", {"image": contours.last_inputs["image"]}, rectangle=[0, 0, 60, 40])
    output = run(
        "collection.filter",
        {
            "collection": packet_for(contours.last_result, "collection"),
            "region": shape.last_result.outputs["region"],
        },
    )
    assert output.last_result.outputs["count"] == 1
    assert output.last_result.outputs["points"] == [[129.5, 219.5]]


def test_lsd_controls_match_native_and_hough_also_exposes_collection():
    image = np.zeros((120, 160), np.uint8)
    cv2.line(image, (15, 30), (140, 30), 255, 3)
    params = dict(
        refinement="advanced",
        scale=0.8,
        sigma_scale=0.6,
        quantization=2.0,
        angle_tolerance=22.5,
        log_epsilon=-1.0,
        density_threshold=0.7,
        bins=512,
    )
    output = run("inspect.line_segments", {"image": image}, **params)
    native = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_ADV, 0.8, 0.6, 2, 22.5, -1, 0.7, 512
    ).detect(image)
    np.testing.assert_allclose(output.last_result.outputs["segments"], native[0].reshape(-1, 4))
    assert output.last_result.outputs["count"] > 0
    for index, item in enumerate(output.last_result.outputs["collection"]["items"]):
        assert item["metrics"]["length"] > 0 and item["metrics"]["line_width"] > 0
        assert item["metrics"]["hu_1"] is None and item["metrics"]["nfa"] is not None
        for metric, values in zip(("line_width", "precision", "nfa"), native[1:]):
            assert item["metrics"][metric] == pytest.approx(values.reshape(-1)[index])
    edges = cv2.Canny(image, 30, 100)
    hough = run("inspect.hough_lines", {"image": edges}, votes=15, min_length=20.0)
    assert (
        hough.last_result.outputs["collection"]["count"] == hough.last_result.outputs["count"] > 0
    )


def test_empty_collections_and_invalid_ranges_or_schema():
    empty = run("inspect.blobs", {"image": np.zeros((40, 60), np.uint8)}).last_result.outputs[
        "collection"
    ]
    for kind in ("collection.filter", "collection.sort"):
        assert run(kind, {"collection": empty}).last_result.outputs["count"] == 0
    draw = run("collection.draw", {"image": np.zeros((40, 60), np.uint8), "collection": empty})
    assert not draw.last_output_images["mask"].any()
    invalid = builtin_registry().create(
        "collection.filter", parameters={"minimum": 10.0, "maximum": 1.0, "use_maximum": True}
    )
    invalid.execute({"collection": empty})
    assert invalid.state == "error"
    duplicate = lines()
    duplicate["items"][1]["id"] = duplicate["items"][0]["id"]
    invalid = builtin_registry().create("collection.sort")
    invalid.execute({"collection": duplicate})
    assert invalid.state == "error" and "unique IDs" in invalid.error


def test_collection_pipeline_preserves_pyramid_frames_and_refuses_wrong_render_level():
    registry = builtin_registry()
    source = run("input.demo", {}).last_output_images["image"]
    images = run("cv.image_pyramid", {"image": source}).last_result.outputs["images"]
    threshold = registry.create("cv.threshold")
    threshold.input_levels = {"image": "2"}
    threshold.execute({"image": images})
    blobs = run("inspect.blobs", {"image": threshold.last_output_images["image"]}, min_area=1)
    filtered = run(
        "collection.filter",
        {"collection": packet_for(blobs.last_result, "collection")},
        metric="height",
        minimum=20.0,
    )
    sorted_objects = run(
        "collection.sort",
        {"collection": packet_for(filtered.last_result, "collection")},
        metric="center_x",
        direction="ascending",
    )
    assert sorted_objects.last_result.outputs["count"] == 2
    assert sorted_objects.last_result.payloads["collection"]["pyramid"]["level"] == 2
    renderer = registry.create("collection.draw")
    renderer.execute(
        {"image": images, "collection": packet_for(sorted_objects.last_result, "collection")}
    )
    assert renderer.state == "error" and "dimensions" in renderer.error
    renderer.input_levels = {"image": "2"}
    renderer.execute(
        {"image": images, "collection": packet_for(sorted_objects.last_result, "collection")}
    )
    assert renderer.state == "pass", renderer.error
    assert renderer.last_output_images["overlay"].shape[:2] == (90, 120)


def test_saved_example_repeats_nested_and_headless_receipts_preserve_descriptors(tmp_path):
    import json

    from opencv_dataflow.core import Graph
    from opencv_dataflow.examples import object_collections_graph
    from opencv_dataflow.runner import main
    from opencv_dataflow.toolblocks import encapsulate

    registry = builtin_registry()
    graph = Graph.load("examples/custom tools/object_collections/object_collections.json", registry)
    assert graph.to_dict() == object_collections_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        assert graph.blocks["objects_blob_draw"].last_result.outputs["count"] == 2
        assert graph.blocks["objects_contour_draw"].last_result.outputs["count"] == 1
        assert graph.blocks["objects_line_draw"].last_result.outputs["count"] == 2
        objects = graph.blocks["objects_blob_sort"].last_result.outputs["collection"]["items"]
        assert objects[0]["metrics"]["center_x"] < objects[1]["metrics"]["center_x"]
    nested = encapsulate(
        graph,
        ["objects_blobs", "objects_blob_filter", "objects_blob_sort"],
        "Reusable ranked parts",
        registry,
    )
    path = tmp_path / "nested-collections.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    restored.run()
    assert restored.blocks[nested.id].state == "pass"
    assert restored.blocks["objects_blob_draw"].last_result.outputs["count"] == 2
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "examples/custom tools/object_collections/object_collections.json",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    records = json.loads(report.read_text())["runs"][0]["blocks"]
    selected = next(row for row in records if row["id"] == "objects_blob_sort")
    assert "hu_7" in selected["outputs"]["collection"]["items"][0]["metrics"]
    assert "spatial" in selected["payloads"]["collection"]


def test_hu_translation_invariance_and_draw_rejects_malformed_geometry():
    image = np.zeros((100, 140), np.uint8)
    image[10:30, 10:50] = 255
    image[45:65, 70:110] = 255
    items = run("inspect.blobs", {"image": image}, min_area=1).last_result.outputs["collection"][
        "items"
    ]
    np.testing.assert_allclose(
        [items[0]["metrics"][f"hu_{i}"] for i in range(1, 8)],
        [items[1]["metrics"][f"hu_{i}"] for i in range(1, 8)],
    )
    data = lines()
    data["items"][0]["geometry"]["points"] = [[1e100, 0], [1, 1]]
    renderer = builtin_registry().create("collection.draw")
    renderer.execute({"image": np.zeros((100, 120), np.uint8), "collection": data})
    assert renderer.state == "error" and "drawing coordinate limits" in renderer.error
