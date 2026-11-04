import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


def run(image, **parameters):
    block = builtin_registry().create("inspect.oriented_edges", parameters=parameters)
    block.execute({"image": image})
    assert block.state == "pass", block.error
    return block


def stripe():
    image = np.zeros((140, 200), np.uint8)
    image[:, 60:120] = 255
    return image


def test_vertical_edge_polarity_band_thickness_intensity_and_contrast():
    image = stripe()
    block = run(
        image,
        min_thickness=2.0,
        max_thickness=6.0,
        min_length=100.0,
        min_continuous_length=100.0,
        min_contrast=200.0,
    )
    items = block.last_result.outputs["collection"]["items"]
    assert len(items) == 2
    assert {i["polarity"] for i in items} == {"rising", "falling"}
    for item in items:
        m = item["metrics"]
        assert m["angle"] == pytest.approx(90, abs=0.1)
        assert m["thickness"] == pytest.approx(4)
        assert m["continuous_length"] > 130 and m["coverage"] == pytest.approx(1)
        assert 100 < m["intensity"] < 150 and m["contrast"] == pytest.approx(255)
    assert run(image, polarity="rising").last_result.outputs["count"] == 1
    assert run(image, polarity="falling").last_result.outputs["count"] == 1
    assert run(image, max_thickness=2.0).last_result.outputs["count"] == 0
    assert run(image, min_intensity=200.0).last_result.outputs["count"] == 0


def test_horizontal_and_angled_edges_select_expected_orientation():
    image = stripe().T.copy()
    horizontal = run(image, orientation="horizontal")
    assert horizontal.last_result.outputs["count"] == 2
    assert all(
        abs(i["metrics"]["angle"]) < 0.1
        for i in horizontal.last_result.outputs["collection"]["items"]
    )
    assert run(image, orientation="vertical").last_result.outputs["count"] == 0
    yy, xx = np.indices((160, 200))
    diagonal = np.where(xx - yy > 20, 255, 0).astype(np.uint8)
    result = run(
        diagonal,
        orientation="angled",
        angle_degrees=45.0,
        angle_tolerance=8.0,
        min_continuous_length=60.0,
    )
    assert result.last_result.outputs["count"] == 1
    angle = result.last_result.outputs["collection"]["items"][0]["metrics"]["angle"]
    assert angle == pytest.approx(45, abs=1)


def test_weak_gradient_filtering_and_continuity_are_independent_of_gap_joining():
    image = np.zeros((140, 220), np.uint8)
    image[:, 30:80] = 40
    image[:, 130:180] = 255
    detected = run(image, response_threshold=5.0, min_strength=30.0)
    assert detected.last_result.outputs["count"] == 2
    # A discontinuous line has two real 50-pixel runs; joining must not invent continuity.
    broken = np.zeros((140, 140), np.uint8)
    broken[10:60, 70:] = 255
    broken[66:116, 70:] = 255
    separated = run(broken, max_gap=0, min_length=30.0, min_continuous_length=30.0)
    assert separated.last_result.outputs["count"] == 2
    joined = run(
        broken, max_gap=10, gap_normal_distance=3.0, min_length=90.0, min_continuous_length=30.0
    )
    assert joined.last_result.outputs["count"] == 1
    metric = joined.last_result.outputs["collection"]["items"][0]["metrics"]
    assert metric["length"] > 90 and metric["continuous_length"] < 60 and metric["coverage"] < 1
    rejected = run(broken, max_gap=10, min_length=90.0, min_continuous_length=90.0)
    assert rejected.last_result.outputs["count"] == 0


def test_oriented_edge_mask_region_and_collection_filter_integration():
    image = stripe()
    registry = builtin_registry()
    shape = registry.create("region.shape", parameters={"rectangle": [40, 20, 45, 90]})
    shape.execute({"image": image})
    edge = registry.create(
        "inspect.oriented_edges", parameters={"min_length": 50.0, "min_continuous_length": 50.0}
    )
    edge.execute({"image": image, "region": shape.last_result.outputs["region"]})
    assert edge.state == "pass" and edge.last_result.outputs["count"] == 1
    assert not edge.last_output_images["mask"][:, 100:].any()
    filtered = registry.create(
        "collection.filter", parameters={"metric": "thickness", "minimum": 3.0}
    )
    filtered.execute({"collection": edge.last_result.outputs["collection"]})
    assert filtered.state == "pass" and filtered.last_result.outputs["count"] == 1
    blank = run(np.zeros_like(image))
    assert blank.last_result.outputs["count"] == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"angle_tolerance": 90.0},
        {"response_threshold": 0.0},
        {"min_thickness": 5.0, "max_thickness": 2.0},
        {"min_intensity": 260.0},
        {"min_coverage": 1.1},
    ],
)
def test_invalid_controls_clear_edge_results(parameters):
    block = builtin_registry().create("inspect.oriented_edges", parameters=parameters)
    block.execute({"image": stripe()})
    assert block.state == "error" and block.last_result is None
