import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.selection import rasterize


def run(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block.last_result.outputs


@pytest.mark.parametrize("shape", ["rectangle", "circle", "polygon"])
def test_shapes_match_native_and_intersect_masks(shape):
    image = np.zeros((30, 40, 3), np.uint8)
    p = dict(
        shape=shape,
        rectangle=[3, 4, 10, 8],
        center=[15, 12],
        radius=7,
        vertices=[[3, 4], [20, 4], [14, 23]],
    )
    output = run("region.shape", {"image": image}, **p)
    expected = np.zeros(image.shape[:2], np.uint8)
    if shape == "rectangle":
        expected[4:12, 3:13] = 255
    elif shape == "circle":
        cv2.circle(expected, (15, 12), 7, 255, -1)
    else:
        cv2.fillPoly(expected, [np.array(p["vertices"], np.int32)], 255)
    np.testing.assert_array_equal(output["mask"], expected)
    inverse = run("region.shape", {"image": image}, **p, invert=True)
    np.testing.assert_array_equal(inverse["mask"], 255 - expected)
    gate = np.zeros_like(expected)
    gate[:, :10] = 1  # Every nonzero byte selects, not just 255.
    histogram = run("inspect.histogram", {"image": image, "mask": gate, "region": output["region"]})
    assert all(
        c["finite_samples"] == np.count_nonzero((expected > 0) & (gate > 0))
        for c in histogram["histogram"]["channels"]
    )


def test_filters_preserve_background_and_segmentation_zeroes_it():
    image = np.random.default_rng(9).integers(0, 256, (30, 40), np.uint8)
    mask = np.zeros_like(image)
    mask[8:20, 10:30] = 255
    blurred = run("cv.box_blur", {"image": image, "mask": mask}, kernel=3)["image"]
    expected = cv2.blur(image, (3, 3))
    expected[mask == 0] = image[mask == 0]
    np.testing.assert_array_equal(blurred, expected)
    for kind in (
        "cv.threshold",
        "cv.canny",
        "gapi.canny",
        "cv.gradient",
        "inspect.corner_response",
    ):
        output = run(kind, {"image": image, "mask": mask})
        key = {"cv.gradient": "gradient", "inspect.corner_response": "response"}.get(kind, "image")
        assert np.all(output[key][mask == 0] == 0)
    np.testing.assert_array_equal(
        image, np.random.default_rng(9).integers(0, 256, (30, 40), np.uint8)
    )


def test_auto_threshold_learns_only_selected_pixels():
    image = np.full((40, 40), 250, np.uint8)
    image[10:20, 10:20] = np.tile([20, 80], (10, 5))
    mask = np.zeros_like(image)
    mask[10:20, 10:20] = 255
    output = run("cv.auto_threshold", {"image": image, "mask": mask})
    assert output["threshold"] == 20
    assert np.count_nonzero(output["image"]) == 50


def test_detection_filters_keep_whole_objects_and_global_coordinates():
    image = np.zeros((80, 120), np.uint8)
    image[10:30, 10:40] = 255
    image[40:65, 75:105] = 255
    mask = np.zeros_like(image)
    mask[19:21, 24:26] = 255  # Tiny center selection must not crop the object.
    output = run("inspect.components", {"image": image, "mask": mask}, expected_count=1)
    assert output["count"] == 1
    assert output["data"]["components"][0]["area"] == 600
    assert output["centroids"] == [[24.5, 19.5]]
    contours = run(
        "inspect.contours", {"image": image, "mask": mask}, expected_count=1, offset=[100, 200]
    )
    assert contours["count"] == 1 and contours["data"]["areas"] == [551.0]
    assert min(p[0] for p in contours["data"]["contours"][0]) == 110


@pytest.mark.parametrize("kind", ["inspect.corners", "inspect.fast", "inspect.orb"])
def test_detectors_accept_regions_and_empty_masks(kind):
    image = np.random.default_rng(7).integers(0, 256, (180, 200), np.uint8)
    region = run("region.shape", {"image": image}, rectangle=[40, 40, 80, 90])["region"]
    output = run(kind, {"image": image, "region": region})
    assert output["count"] > 0
    assert all(39 <= x <= 120 and 39 <= y <= 130 for x, y in output["points"])
    empty = run(kind, {"image": image, "mask": np.zeros_like(image)})
    assert empty["count"] == 0


def test_template_search_region_is_separate_from_template_mask():
    rng = np.random.default_rng(2)
    template = rng.integers(0, 256, (9, 11), np.uint8)
    image = np.zeros((60, 80), np.uint8)
    image[5:14, 5:16] = template
    image[35:44, 50:61] = template
    region = run("region.shape", {"image": image}, rectangle=[45, 30, 25, 25])["region"]
    result = run("inspect.match_template", {"image": image, "template": template, "region": region})
    assert result["location"]["top_left"] == [50, 35]
    assert result["score"] > 0.99


def test_region_roundtrip_invalid_dimensions_and_point_filter():
    image = np.zeros((20, 30), np.uint8)
    output = run("region.shape", {"image": image}, rectangle=[-3, -2, 10, 10])
    assert np.count_nonzero(output["mask"]) == 56
    filtered = run(
        "region.filter_points",
        {"image": image, "region": output["region"], "points": [[2, 3], [15, 10], [-2, 1]]},
    )
    assert filtered == {"points": [[2.0, 3.0]], "count": 1}
    with pytest.raises(ValueError, match="dimensions"):
        rasterize(output["region"], (30, 30))
    registry = builtin_registry()
    graph = Graph()
    graph.add(registry.create("region.shape", parameters={"shape": "polygon"}))
    assert Graph.from_dict(graph.to_dict(), registry).to_dict() == graph.to_dict()


@pytest.mark.parametrize(
    "params",
    [
        {"radius": -1, "shape": "circle"},
        {"rectangle": [0, 0, 0, 10]},
        {"vertices": [[0, 0], [1, 1]], "shape": "polygon"},
        {"vertices": [[0, 0], [1, 1], [2, 2]], "shape": "polygon"},
    ],
)
def test_invalid_regions_fail_without_results(params):
    block = builtin_registry().create("region.shape", parameters=params)
    block.execute({"image": np.zeros((30, 30), np.uint8)})
    assert block.state == "error" and block.last_result is None


def test_clip_mode_measures_intersection_instead_of_whole_object():
    image = np.zeros((40, 60), np.uint8)
    image[10:30, 10:40] = 255
    mask = np.zeros_like(image)
    mask[15:25, 15:25] = 255
    output = run(
        "inspect.components",
        {"image": image, "mask": mask},
        region_mode="clip",
        expected_count=1,
        min_area=1,
    )
    assert output["data"]["components"][0]["area"] == 100
    assert output["centroids"] == [[19.5, 19.5]]
    output = run(
        "inspect.contours",
        {"image": image, "mask": mask},
        region_mode="clip",
        expected_count=1,
        min_area=1,
    )
    assert output["data"]["areas"] == [81.0]


def test_saved_region_project_runs_and_roundtrips_nested(tmp_path):
    from opencv_dataflow.examples import regions_graph
    from opencv_dataflow.toolblocks import encapsulate

    registry = builtin_registry()
    graph = Graph.load("examples/regions.json", registry)
    assert graph.to_dict() == regions_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        assert graph.blocks["roi_left"].last_result.outputs["count"] == 1
        assert graph.blocks["roi_right"].last_result.outputs["count"] == 1
    block = encapsulate(graph, ["roi_circle", "roi_left"], "Selected inspection", registry)
    block.position = [850.0, 250.0]
    path = tmp_path / "regions.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    restored.run()
    assert restored.blocks[block.id].state == "pass"
    assert restored.blocks[block.id].position == [850.0, 250.0]


def test_hough_result_filters_use_centers_and_line_endpoints(monkeypatch):
    image = np.zeros((80, 100), np.uint8)
    region = run("region.shape", {"image": image}, rectangle=[10, 10, 40, 40])["region"]
    monkeypatch.setattr(
        cv2,
        "HoughLinesP",
        lambda *a, **k: np.array(
            [[[12, 12, 45, 45]], [[12, 12, 80, 45]], [[60, 60, 75, 75]]], np.int32
        ),
    )
    output = run("inspect.hough_lines", {"image": image, "region": region})
    assert output["segments"] == [[12, 12, 45, 45]]
    monkeypatch.setattr(
        cv2, "HoughCircles", lambda *a, **k: np.array([[[25, 25, 10], [70, 60, 15]]], np.float32)
    )
    output = run("inspect.hough_circles", {"image": image, "region": region})
    assert output["circles"] == [[25, 25, 10]]


def test_empty_and_invalid_selection_contracts():
    image = np.ones((20, 30), np.uint8)
    empty = np.zeros_like(image)
    assert not run("cv.threshold", {"image": image, "mask": empty})["image"].any()
    hist = run("inspect.histogram", {"image": image, "mask": empty})
    assert hist["histogram"]["channels"][0]["finite_samples"] == 0
    for mask in (np.ones((20, 30), np.float32), np.ones((10, 10), np.uint8)):
        block = builtin_registry().create("cv.box_blur")
        block.execute({"image": image, "mask": mask})
        assert block.state == "error" and block.last_result is None
    block = builtin_registry().create("cv.auto_threshold")
    block.execute({"image": image, "mask": empty})
    assert block.state == "error" and "nonempty" in block.error


def test_selected_contrast_math_and_mask_composition():
    image = np.array([[10, 20, 50, 90], [10, 20, 50, 90]], np.uint8)
    left = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], np.uint8)
    top = np.array([[255, 255, 255, 255], [0, 0, 0, 0]], np.uint8)
    combined = run("cv.bitwise", {"image_a": left, "image_b": top})["image"]
    np.testing.assert_array_equal(combined, [[1, 1, 0, 0], [0, 0, 0, 0]])
    equalized = run("cv.equalize", {"image": image, "mask": left})["image"]
    np.testing.assert_array_equal(equalized, [[0, 255, 50, 90], [0, 255, 50, 90]])
    blended = run(
        "cv.blend", {"image_a": image, "image_b": image, "mask": left}, alpha=1.0, beta=1.0
    )["image"]
    np.testing.assert_array_equal(blended, [[20, 40, 50, 90], [20, 40, 50, 90]])
    difference = run(
        "cv.absdiff", {"image_a": image, "image_b": np.zeros_like(image), "mask": left}
    )["image"]
    np.testing.assert_array_equal(difference, [[10, 20, 0, 0], [10, 20, 0, 0]])
