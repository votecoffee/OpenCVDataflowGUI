"""Qualify ximgproc segment coordinates, selection, and edge-map ownership."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "inspect.fast_line_segments"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/fast_line_detection"


def tool(**settings):
    return builtin_registry().create(TYPE, parameters=settings)


@pytest.mark.parametrize("aperture", ["0", "3", "5", "7"])
def test_native_parity_and_unmodified_source(aperture):
    gray = np.zeros((76, 88), np.uint8)
    cv2.rectangle(gray, (13, 17), (72, 58), 210, -1)
    work = cv2.Canny(gray, 50, 85) if aperture == "0" else gray
    expected = cv2.ximgproc.createFastLineDetector(12, 1.5, 50, 85, int(aperture), True)
    expected = expected.detect(work.copy()).reshape(-1, 4)
    if aperture == "0":
        source = work
    else:
        source = ImageArray(np.repeat((work.astype(np.uint16) * 257)[..., None], 3, axis=2), "rgb")
    saved = source.copy()
    node = tool(
        length_threshold=12,
        distance_threshold=1.5,
        canny_low=50,
        canny_high=85,
        canny_aperture=aperture,
        merge_segments=True,
    )
    node.set_graphics_mode("off")
    result = node.execute({"image": source})
    assert result and result.passed, node.error
    np.testing.assert_array_equal(result.outputs["segments"], expected)
    np.testing.assert_array_equal(node.last_inputs["image"], saved)
    np.testing.assert_array_equal(result.outputs["image"], saved)
    np.testing.assert_array_equal(source, saved)
    assert not result.drawings


def test_endpoint_selection_preserves_pyramid_frame_ticket_and_drawings():
    edges = np.zeros((64, 96), np.uint8)
    edges[18, 12:52] = 255
    edges[37, 12:52] = 255
    edges[49, 12:71] = 255
    size = [96, 64]
    scale = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    spatial = {**frame([192, 128], "source"), "image_size": size, "to_root": scale}
    pyramid = {"level": 1, "largest_size": [192, 128], "to_largest": scale}
    ticket = TicketClock("fast_lines").issue("camera", "fixed", "plate")
    payload = {"spatial": spatial, "pyramid": pyramid, "ticket": ticket}
    mask = np.zeros_like(edges)
    mask[30:, :60] = 255
    region = {
        "model": "image_region_v1",
        "image_size": size,
        "shape": "rectangle",
        "rectangle": [10, 12, 72, 45],
        "coordinate_context": spatial,
    }
    node = tool(canny_aperture="0")
    result = node.execute(
        {
            "image": ValuePacket(edges, payload),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, node.error
    assert result.outputs["count"] == 1
    objects = result.outputs["collection"]
    validate_collection(objects)
    assert objects["context"] == {"spatial": spatial, "pyramid": pyramid}
    assert objects["image_size"] == size
    points = np.asarray(objects["items"][0]["geometry"]["points"])
    np.testing.assert_array_equal(points[:, 1], [37, 37])
    np.testing.assert_array_equal(np.sort(points[:, 0]), [12, 51])
    assert objects["items"][0]["metrics"]["length"] == 39
    for key, value in payload.items():
        assert result.payloads["collection"][key] == value
    (arrow,) = result.drawings["fast line segments"].objects
    assert arrow.kind == "arrow"
    np.testing.assert_array_equal(arrow.points, points)
    other = ValuePacket(mask, {"spatial": frame(size, "other")})
    assert node.execute({"image": ValuePacket(edges, payload), "mask": other}) is None
    assert "coordinate frame" in node.error


def test_validation_and_empty_detections():
    for settings, fields in (
        ({"canny_aperture": "1"}, ("canny_aperture",)),
        ({"distance_threshold": 0}, ("distance_threshold",)),
        ({"canny_low": 80, "canny_high": 70}, ("canny_low", "canny_high")),
    ):
        with pytest.raises(ValidationError) as error:
            tool(**settings).validate_parameters()
        assert error.value.fields == fields
    node = tool(canny_aperture="0", merge_segments=True)
    assert node.execute({"image": np.zeros((5, 8), np.uint8)}) is None
    assert "6 × 6" in node.error
    assert node.execute({"image": np.zeros((12, 12, 3), np.uint8)}) is None
    assert "monochrome uint8 edge map" in node.error
    result = node.execute({"image": np.zeros((12, 12), np.uint8)})
    assert result and result.passed
    assert result.outputs["count"] == 0 and not result.drawings


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_measures_known_edges_and_sorts_by_length(profile):
    graph = Graph.load(EXAMPLE / "fast_line_detection.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        for key in ("intensity", "edge_map"):
            node = graph.blocks[key]
            result = node.last_result
            assert result and result.passed, node.error
            items = result.outputs["collection"]["items"]
            measurements = [
                (np.array(item["geometry"]["points"]), item["metrics"]) for item in items
            ]
            assert any(
                max(abs(p[:, 1] - 25)) <= 1.1 and m["angle"] < 0.1 and m["length"] > 100
                for p, m in measurements
            )
            assert any(
                max(abs(p[:, 0] - 141)) <= 1 and abs(m["angle"] - 90) < 0.1 and m["length"] > 90
                for p, m in measurements
            )
            assert any(
                max(abs(p[:, 1] - p[:, 0] - 53)) <= 1.5
                and abs(m["angle"] - 45) < 0.3
                and m["length"] > 60
                for p, m in measurements
            )
        rows = graph.blocks["sorted"].last_result.outputs["collection"]["items"]
        lengths = [row["metrics"]["length"] for row in rows]
        assert lengths == sorted(lengths, reverse=True)
        assert len(lengths) == graph.blocks["intensity"].last_result.outputs["count"]
    finally:
        graph.close()
