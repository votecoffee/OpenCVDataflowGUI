"""Qualify orientation, measurements and downstream use of contour ellipses."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/contour_ellipses"
TRUTH = (((90, 85), (106, 62), 27), ((248, 167), (132, 74), 117), ((345, 248), (98, 48), 153))


def test_saved_example_geometry_and_collection_workflow():
    graph = Graph.load(EXAMPLE / "contour_ellipses.json", builtin_registry())
    try:
        graph.run()
        detector = graph.blocks["detect"]
        result = detector.last_result
        assert result and result.passed, detector.error
        assert block_entry(detector).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        ellipses = result.outputs["ellipses"]
        assert len(ellipses) == 3 and result.outputs["count"] == 3
        assert [fit["score"] for fit in ellipses] == sorted(
            (fit["score"] for fit in ellipses), reverse=True
        )
        for fit, (center, diameters, angle) in zip(
            sorted(ellipses, key=lambda fit: fit["center"][0]), TRUTH, strict=True
        ):
            assert np.linalg.norm(np.subtract(fit["center"], center)) < 2
            np.testing.assert_allclose(fit["axis_lengths"], diameters, rtol=0, atol=4)
            assert abs((fit["angle_degrees"] - angle + 90) % 180 - 90) < 2
        drawing = result.drawings["ellipses"]
        assert drawing.source == "image" and len(drawing.objects) == 6
        assert sum(obj.closed for obj in drawing.objects) == 3

        kept = graph.blocks["filter"].last_result.outputs["collection"]
        assert kept["count"] == 1 and kept["kind"] == "ellipses"
        item = kept["items"][0]
        assert item["metrics"]["rotated_aspect_ratio"] >= 1.85
        assert item["metrics"]["strength"] == ellipses[0]["score"]
        drawn = graph.blocks["draw"].last_result.outputs["mask"]
        assert drawn[248, 345] == 255 and drawn[85, 90] == 0
        region = graph.blocks["region"].last_result.outputs["region"]
        assert region["center"] == result.outputs["best_fit"]["center"]
        np.testing.assert_allclose(region["axis_lengths"], np.add(ellipses[0]["axis_lengths"], 4))
    finally:
        graph.close()


def test_center_selection_preserves_typed_image_and_ticket():
    gray = cv2.imread(str(EXAMPLE / "assets/ellipses.png"), cv2.IMREAD_GRAYSCALE)
    wide = gray.astype(np.uint16) * 257
    source = ImageArray(np.stack([wide, wide, wide, np.zeros_like(wide)], axis=-1), "rgba")
    spatial = frame([460, 360], "camera")
    ticket = TicketClock("ellipses").issue("camera", "fixed", "part-3")
    selected = np.zeros(gray.shape, np.uint8)
    selected[80:94, 80:103] = 255
    selected[156:175, 240:263] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [460, 360],
        "rectangle": [240, 156, 23, 19],
    }
    block = builtin_registry().create("inspect.find_ellipses")
    result = block.execute(
        {
            "image": ValuePacket(source, {"ticket": ticket, "spatial": spatial}),
            "mask": ValuePacket(selected, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    assert result.outputs["count"] == 1
    assert np.linalg.norm(np.subtract(result.outputs["best_fit"]["center"], [248, 167])) < 2
    assert result.outputs["image"].color_format == "rgba"
    assert result.outputs["image"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["image"], source)
    for key in ("image", "best_fit", "collection"):
        assert result.payloads[key]["ticket"] == ticket
        assert result.payloads[key]["spatial"] == spatial
    assert result.outputs["collection"]["context"]["spatial"] == spatial

    other_frame = frame([460, 360], "other-camera")
    assert (
        block.execute(
            {
                "image": ValuePacket(source, {"spatial": spatial}),
                "mask": ValuePacket(selected, {"spatial": other_frame}),
            }
        )
        is None
    )
    assert "Selection coordinate frame differs" in block.error


def test_count_limit_and_empty_fit_emission():
    registry = builtin_registry()
    gray = cv2.imread(str(EXAMPLE / "assets/ellipses.png"), cv2.IMREAD_GRAYSCALE)
    capped = registry.create("inspect.find_ellipses", parameters={"maximum_results": 1})
    result = capped.execute({"image": gray})
    assert result and result.passed and result.outputs["count"] == 1
    assert result.outputs["best_fit"]["center"] == pytest.approx([346.1754, 249.0473], abs=1)

    blank = np.zeros_like(gray)
    result = capped.execute({"image": blank})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "too_few_ellipses"
    assert result.outputs["best_fit"] is NO_VALUE and result.emissions["best_fit"] == []
    assert result.outputs["collection"]["count"] == 0

    permissive = registry.create("inspect.find_ellipses", parameters={"minimum_count": 0})
    result = permissive.execute({"image": gray, "mask": blank})
    assert result and result.passed and result.outputs["count"] == 0

    invalid = registry.create(
        "inspect.find_ellipses", parameters={"minimum_count": 2, "maximum_results": 1}
    )
    assert invalid.execute({"image": gray}) is None
    assert "Minimum count must be at most maximum" in invalid.error
