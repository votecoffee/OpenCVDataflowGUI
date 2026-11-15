"""Native run endpoints, independent sparse geometry, flat topology and resource guards."""

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import Contours, builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.payloads import NO_VALUE, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.run_contours import RunContours, count_runs
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def successful(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def geometry(items):
    return sorted(
        (
            tuple(cv2.boundingRect(np.array(i["geometry"]["points"], np.int32))),
            i["metrics"]["area"],
            i["metrics"]["perimeter"],
        )
        for i in items
    )


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_large_sparse_solid_polygons_match_dense_reference_and_independent_measurements(profile):
    image = np.zeros((3072, 4096), np.uint8)
    rectangles = [(0, 0, 18, 23), (350, 67, 131, 233), (2900, 2100, 251, 451)]
    for x, y, w, h in rectangles:
        image[y : y + h, x : x + w] = 17  # Every nonzero uint8 value is foreground.
    original = image.copy()
    graph = Graph()
    block = graph.add(RunContours(parameters={"min_area": 0, "expected_count": 3}))
    graph.set_run_profile(profile)
    result = successful(block, {"image": image})
    np.testing.assert_array_equal(image, original)
    collection = validate_collection(result.outputs["collection"])
    assert result.passed and result.outputs["count"] == 3
    measured = geometry(collection["items"])
    expected = sorted(
        ((x, y, w, h), float((w - 1) * (h - 1)), float(2 * (w + h - 2)))
        for x, y, w, h in rectangles
    )
    assert measured == expected
    dense = successful(
        Contours(parameters={"retrieval": "list", "min_area": 0, "expected_count": 3}),
        {"image": image},
    )
    assert measured == geometry(dense.outputs["collection"]["items"])
    assert result.outputs["data"]["horizontal_runs"] == sum(r[3] for r in rectangles)
    assert result.outputs["data"]["raw_point_count"] == 2 * sum(r[3] for r in rectangles)
    assert result.outputs["overlay"] is result.outputs["mask"] is NO_VALUE
    if profile == "lean":
        assert not result.drawings
    graph.close()


def test_nested_mask_exposes_native_polygon_difference_and_explicit_flat_links():
    image = np.zeros((19, 23), np.uint8)
    image[:15, :15] = 255
    image[3:10, 3:10] = 0
    image[5:8, 5:8] = 255
    result = successful(
        RunContours(parameters={"min_area": 0, "expected_count": 3, "draw_mask": True}),
        {"image": image},
    )
    native, links = cv2.findContoursLinkRuns(image)
    assert result.outputs["data"]["contours"] == [c.reshape(-1, 2).tolist() for c in native]
    assert sorted(result.outputs["data"]["areas"]) == [4, 48, 196]
    dense, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    assert sorted(cv2.contourArea(c) for c in dense) == [4, 62, 196]
    assert np.count_nonzero(hierarchy[..., 3] >= 0) == 1
    assert not result.outputs["data"]["hole_parents_available"]
    assert result.outputs["data"]["hierarchy_semantics"] == "flat_list"
    for index, item in enumerate(result.outputs["collection"]["items"]):
        assert item["geometry"]["hierarchy"] == [index + 1 if index < 2 else -1, index - 1, -1, -1]
    expected_mask = np.zeros_like(image)
    cv2.drawContours(expected_mask, native, -1, 255, -1)
    np.testing.assert_array_equal(result.outputs["mask"], expected_mask)


@pytest.mark.parametrize("shape", ["empty", "point", "row", "diagonal", "strided"])
def test_native_edge_and_degenerate_masks_are_not_silently_simplified(shape):
    image = np.zeros((17, 23), np.uint8)
    if shape == "point":
        image[8, 12] = 255
    elif shape == "row":
        image[0, 3:19] = 255
    elif shape == "diagonal":
        np.fill_diagonal(image, 255)
    elif shape == "strided":
        image[2:14, 4:18] = 255
        image = image[::2, ::2]
    native, _ = cv2.findContoursLinkRuns(image)
    result = successful(
        RunContours(parameters={"min_area": 0, "expected_count": len(native)}), {"image": image}
    )
    assert result.outputs["data"]["contours"] == [c.reshape(-1, 2).tolist() for c in native]
    assert result.outputs["count"] == len(native)
    validate_collection(result.outputs["collection"])


@pytest.mark.parametrize("mode,expected_width", [("center", 9), ("clip", 5)])
def test_selection_offset_and_ticket_pyramid_frames(mode, expected_width):
    image = np.zeros((20, 24), np.uint8)
    image[4:13, 4:13] = 255
    spatial = frame([24, 20], "camera")
    payload = {
        "ticket": TicketClock().issue("camera"),
        "spatial": spatial,
        "pyramid": {
            "level": 1,
            "largest_size": [48, 40],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "acquisition": {"encoder": 771},
    }
    source = ImageArray(image, "gray", payload)
    mask = ImageArray(np.full_like(image, 255), "gray", {"spatial": spatial})
    region = {
        "model": "image_region_v1",
        "image_size": [24, 20],
        "shape": "rectangle",
        "rectangle": [0, 0, 9, 20],
        "coordinate_context": spatial,
    }
    block = RunContours(
        parameters={
            "region_mode": mode,
            "offset": [100, -40],
            "min_area": 0,
            "expected_count": 1,
            "draw_overlay": True,
            "draw_mask": True,
        }
    )
    result = successful(block, {"image": source, "mask": mask, "region": region})
    item = result.outputs["collection"]["items"][0]
    assert item["metrics"]["x"] == 104 and item["metrics"]["y"] == -36
    assert item["metrics"]["width"] == expected_width
    assert result.outputs["collection"]["origin"] == [100, -40]
    assert result.payloads["overlay"]["spatial"] == spatial
    shifted = result.payloads["collection"]
    assert shifted["spatial"]["to_root"] == [[1, 0, -100], [0, 1, 40], [0, 0, 1]]
    assert shifted["pyramid"]["to_largest"] == [[2, 0, -200], [0, 2, 80], [0, 0, 1]]
    assert ticket_key(shifted["ticket"]) == ticket_key(payload["ticket"])
    assert shifted["acquisition"] == payload["acquisition"]
    mask.metadata["spatial"] = frame([24, 20], "other")
    assert block.execute({"image": source, "mask": mask}) is None
    assert "coordinate frame" in block.error


def test_run_budget_precedes_native_and_counts_actual_clipped_search(monkeypatch):
    image = np.tile(np.arange(20) % 2, (10, 1)).astype(np.uint8)
    assert count_runs(image, 100) == 100
    calls, native = [], cv2.findContoursLinkRuns

    def observed(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(cv2, "findContoursLinkRuns", observed)
    block = RunContours(parameters={"max_runs": 2, "min_area": 0, "expected_count": 2})
    assert block.execute({"image": image}) is None and not calls
    block.parameters["region_mode"] = "clip"
    selection = np.zeros_like(image)
    selection[0, 0:4] = 255
    result = successful(block, {"image": image, "mask": selection})
    assert result.passed and result.outputs["data"]["horizontal_runs"] == 2 and calls


@pytest.mark.parametrize(
    "parameters,field", [({"max_contours": 1}, "Max contours"), ({"max_points": 3}, "Max points")]
)
def test_native_result_budget_applies_before_filtering_and_metadata(parameters, field):
    image = np.zeros((20, 30), np.uint8)
    image[2:8, 2:8] = 255
    image[12:18, 20:28] = 255
    block = RunContours(parameters={"min_area": 999999, **parameters})
    assert block.execute({"image": image}) is None and field in block.error
    assert not block.current_emissions and block.error_kind == "invalid"


@pytest.mark.parametrize(
    "parameters",
    [
        {"retrieval": "ccomp"},
        {"retrieval": "tree"},
        {"approximation": "none"},
        {"max_runs": 0},
        {"offset": [2**31, 0]},
        {"offset": [1.5, 0]},
        {"offset": [2**31 - 1, 0]},
        {"draw_mask": "yes"},
    ],
)
def test_invalid_modes_or_coordinates_do_not_call_native(parameters, monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "findContoursLinkRuns", lambda *a, **k: calls.append(1))
    block = RunContours(parameters=parameters)
    assert block.execute({"image": np.zeros((4, 6), np.uint8)}) is None
    assert not calls and not block.current_emissions


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 5), np.uint16),
        np.zeros((4, 5), bool),
        np.zeros((4, 5, 3), np.uint8),
        np.empty((0, 5), np.uint8),
        DepthArray(np.ones((4, 5))),
    ],
)
def test_native_input_restrictions_are_explicit(image):
    block = RunContours()
    assert block.execute({"image": image}) is None
    assert not block.current_emissions


def test_native_errors_are_contained(monkeypatch):
    def broken(*args):
        raise cv2.error("controlled link-runs error")

    monkeypatch.setattr(cv2, "findContoursLinkRuns", broken)
    block = RunContours()
    assert block.execute({"image": np.zeros((4, 5), np.uint8)}) is None
    assert "controlled link-runs" in block.error and not block.current_emissions


def test_ui_controls_old_defaults_and_xml_roundtrip(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    before = {
        "min_area": 100.0,
        "expected_count": 2,
        "retrieval": "external",
        "approximation": "simple",
        "offset": [0, 0],
        "region_mode": "center",
    }
    assert Contours.defaults == before
    block = RunContours(
        parameters={
            "offset": [-50, 100],
            "region_mode": "clip",
            "draw_mask": True,
            "max_points": 2000,
        }
    )
    assert block_entry(type(block)).path == ("OpenCV", "Image Processing (imgproc)", "Shapes")
    graph = Graph()
    graph.add(block)
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for field in ("retrieval", "approximation", "region_mode"):
            assert isinstance(window.inspector.fields[field], QComboBox)
        path = tmp_path / "settings.inspect.run_contours.xml"
        save_block_xml(block, path)
        assert read_block_xml(path)["parameters"] == block.parameters
        path = tmp_path / "graph.xml"
        graph.save(path)
        restored = Graph.load(path, builtin_registry())
        try:
            assert restored.to_dict() == graph.to_dict()
        finally:
            restored.close()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
