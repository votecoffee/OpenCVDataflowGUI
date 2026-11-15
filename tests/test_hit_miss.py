"""Binary pattern truth, native/independent neighborhoods, payloads and signed editor cells."""

import copy

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.hit_miss import HitOrMiss
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.kernel_editor import KERNEL_PARAMETERS, KernelEditor
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def successful(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def erode_reference(image, kernel, anchor, iterations, border, value):
    result = image.copy()
    x, y = anchor
    top, left = y, x
    bottom, right = kernel.shape[0] - 1 - y, kernel.shape[1] - 1 - x
    for _ in range(iterations):
        if border == "constant":
            padded = np.pad(
                result,
                ((top, bottom), (left, right)),
                constant_values=True if value is None else bool(value),
            )
        else:
            mode = {"replicate": "edge", "reflect_101": "reflect", "reflect_edge": "symmetric"}[
                border
            ]
            padded = np.pad(result, ((top, bottom), (left, right)), mode=mode)
        result = np.ones(image.shape, bool)
        for ky, kx in np.argwhere(kernel):
            result &= padded[ky : ky + image.shape[0], kx : kx + image.shape[1]]
    return result


@pytest.mark.parametrize(
    "border,value",
    [
        ("constant", None),
        ("constant", 0),
        ("constant", 255),
        ("replicate", None),
        ("reflect_101", None),
        ("reflect_edge", None),
    ],
)
@pytest.mark.parametrize("iterations,anchor", [(1, (1, 0)), (2, (0, 1))])
@pytest.mark.parametrize("support", ["full_kernel", "native_border"])
def test_asymmetric_neighborhood_matches_independent_boolean_reference(
    border, value, iterations, anchor, support
):
    image = (np.random.default_rng(401).random((11, 13)) > 0.35).astype(np.uint8) * 255
    kernel = np.array([[0, 1, -1], [-1, 0, 1]], np.int8)
    block = HitOrMiss(
        parameters={
            "custom_kernel": kernel.tolist(),
            "anchor": list(anchor),
            "border": border,
            "border_value": [] if value is None else [value],
            "iterations": iterations,
            "support": support,
        }
    )
    result = successful(block, {"image": image})
    expected = erode_reference(image != 0, kernel == 1, anchor, iterations, border, value)
    expected &= erode_reference(image == 0, kernel == -1, anchor, iterations, border, value)
    if support == "full_kernel":
        for y, x in np.ndindex(image.shape):
            if (
                x < anchor[0] * iterations
                or y < anchor[1] * iterations
                or x + (kernel.shape[1] - 1 - anchor[0]) * iterations >= image.shape[1]
                or y + (kernel.shape[0] - 1 - anchor[1]) * iterations >= image.shape[0]
            ):
                expected[y, x] = False
    np.testing.assert_array_equal(result.outputs["image"], expected.astype(np.uint8) * 255)
    assert result.outputs["count"] == int(expected.sum())
    assert result.outputs["points"] == np.argwhere(expected)[:, ::-1].tolist()
    assert result.outputs["measurements"]["anchor"] == list(anchor)


@pytest.mark.parametrize("kernel", [[[1]], [[-1]], [[0, 1]], [[-1, 0, -1]]])
def test_single_sided_constraints_and_even_centered_kernels_match_native(kernel):
    image = np.zeros((7, 11), np.uint8)
    image[3, 4] = 255
    result = successful(
        HitOrMiss(parameters={"custom_kernel": kernel, "support": "native_border"}),
        {"image": image},
    )
    expected = cv2.morphologyEx(image, cv2.MORPH_HITMISS, np.array(kernel, np.int8))
    np.testing.assert_array_equal(result.outputs["image"], expected)


@pytest.mark.parametrize("shape", ["rect", "ellipse", "cross", "diamond"])
def test_standard_structuring_elements_require_foreground_and_use_independent_anchor(shape):
    image = np.full((12, 17), 255, np.uint8)
    image[3:9, 8] = 0
    p = {
        "shape": shape,
        "kernel": 4,
        "kernel_y": 3,
        "shape_anchor": [0, 2],
        "anchor": [3, 0],
        "support": "native_border",
    }
    block = HitOrMiss(parameters=p)
    result = successful(block, {"image": image})
    kernel = block.kernel_matrix(block.parameters)
    np.testing.assert_array_equal(
        result.outputs["image"], cv2.erode(image, kernel.astype(np.uint8), anchor=(3, 0))
    )


@pytest.mark.parametrize("encoding,polarity", [(1, "light"), (255, "light"), (255, "dark")])
def test_binary_encodings_and_dark_polarity_have_exact_pixel_truth(encoding, polarity):
    image = np.zeros((12, 17), np.uint8)
    image[6, 7:10] = encoding
    image[5:8, 8] = encoding
    if polarity == "dark":
        image = 255 - image
    result = successful(HitOrMiss(parameters={"polarity": polarity}), {"image": image})
    assert result.outputs["points"] == [[8, 6]] and result.outputs["count"] == 1
    assert np.count_nonzero(result.outputs["image"]) == 1


def test_nonzero_is_explicit_and_strict_mode_rejects_unthresholded_grayscale():
    image = np.zeros((9, 11), np.uint8)
    image[4, 4:7], image[3:6, 5] = 17, 200
    block = HitOrMiss()
    assert block.execute({"image": image}) is None and "0/1 or 0/255" in block.error
    assert not block.current_emissions
    block.parameters["input_mode"] = "nonzero"
    assert successful(block, {"image": image}).outputs["points"] == [[5, 4]]


def test_selection_filters_centers_but_retains_neighbors_and_page_provenance():
    image = np.zeros((13, 17), np.uint8)
    image[6, 7:10], image[5:8, 8] = 255, 255
    payload = {
        "ticket": TicketClock().issue("binary-camera"),
        "spatial": frame([17, 13], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [34, 26],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "acquisition": {"encoder": 2048},
    }
    source = ImageArray(image, "gray", payload)
    mask = np.zeros_like(image)
    mask[6, 8] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [17, 13],
        "shape": "circle",
        "center": [8, 6],
        "radius": 2,
        "coordinate_context": payload["spatial"],
    }
    block = HitOrMiss()
    result = successful(block, {"image": source, "mask": mask, "region": region})
    assert result.outputs["points"] == [[8, 6]]
    for port in ("image", "points"):
        assert result.payloads[port]["spatial"] == payload["spatial"]
        assert result.payloads[port]["pyramid"] == payload["pyramid"]
        assert result.payloads[port]["acquisition"] == payload["acquisition"]
        assert ticket_key(result.payloads[port]["ticket"]) == ticket_key(payload["ticket"])
    assert len(result.drawings["binary matches"].objects) == 1
    wrong = ImageArray(mask, "gray", {"spatial": frame([17, 13], "other-camera")})
    assert block.execute({"image": source, "mask": wrong}) is None
    assert "coordinate frame" in block.error
    region["coordinate_context"] = wrong.metadata["spatial"]
    assert block.execute({"image": source, "region": region}) is None
    assert not block.current_emissions


def test_coordinate_cap_samples_without_truncating_mask_count_or_inspection():
    image = np.full((500, 700), 255, np.uint8)
    block = HitOrMiss(
        parameters={
            "custom_kernel": [[1]],
            "max_points": 3,
            "max_graphics": 2,
            "check_count": True,
            "minimum_count": image.size + 1,
        }
    )
    result = successful(block, {"image": image})
    assert not result.passed and result.fail_reasons
    assert result.outputs["count"] == image.size
    assert result.outputs["points"] == [[0, 0], [699, 249], [699, 499]]
    assert result.outputs["measurements"]["points_sampled"]
    assert result.outputs["measurements"]["drawn_count"] == 2
    assert np.all(result.outputs["image"] == 255)
    block.parameters["max_points"] = 0
    result = successful(block, {"image": image})
    assert result.outputs["points"] == [] and result.outputs["count"] == image.size
    assert not result.drawings


def test_empty_selection_oversized_footprint_and_graphics_off_are_valid_no_matches(monkeypatch):
    image = np.full((3, 4), 255, np.uint8)
    block = HitOrMiss(parameters={"shape": "rect", "kernel": 9, "iterations": 3})
    with monkeypatch.context() as guard:

        def forbidden(*args, **kwargs):
            raise AssertionError("An empty eligible footprint must not call native morphology")

        guard.setattr(cv2, "morphologyEx", forbidden)
        result = successful(block, {"image": image})
    assert result.passed and result.outputs["count"] == 0
    assert not result.outputs["image"].any()
    block.parameters.update(kernel=1, iterations=1)
    block.set_graphics_mode("off")
    result = successful(block, {"image": image, "mask": np.zeros_like(image)})
    assert result.outputs["points"] == [] and not result.drawings
    assert successful(block, {"image": image}).outputs["count"] == 12
    assert not block.last_result.drawings


@pytest.mark.parametrize(
    "parameters",
    [
        {"custom_kernel": [[0, 0]]},
        {"custom_kernel": [[1, -2]]},
        {"custom_kernel": [[1], [-1, 0]]},
        {"custom_kernel": [[True]]},
        {"custom_kernel": [[float("nan")]]},
        {"custom_kernel": [[1] * 256]},
        {"anchor": [3, 0]},
        {"anchor": [0.5, 0]},
        {"iterations": 0},
        {"iterations": 101},
        {"border_value": [17]},
        {"border_value": [0, 255]},
        {"border_value": "0"},
        {"max_points": -1},
        {"minimum_count": 2, "maximum_count": 1},
        {"check_count": "yes"},
    ],
)
def test_invalid_rules_are_reported_before_native_work(parameters, monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "morphologyEx", lambda *a, **k: calls.append(1))
    block = HitOrMiss(parameters=parameters)
    assert block.execute({"image": np.zeros((5, 5), np.uint8)}) is None
    assert block.error_kind == "invalid" and not calls and not block.current_emissions


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 3), np.uint16),
        np.ones((3, 3), np.float32),
        np.zeros((3, 3, 3), np.uint8),
        np.empty((0, 3), np.uint8),
        DepthArray(np.ones((3, 3))),
    ],
)
def test_incompatible_images_fail_without_an_output_event(image):
    block = HitOrMiss()
    assert block.execute({"image": image}) is None and not block.current_emissions


def test_native_exceptions_are_contained(monkeypatch):
    def broken(*args, **kwargs):
        raise cv2.error("controlled hit-or-miss failure")

    monkeypatch.setattr(cv2, "morphologyEx", broken)
    block = HitOrMiss()
    assert block.execute({"image": np.zeros((5, 5), np.uint8)}) is None
    assert "controlled hit-or-miss" in block.error and not block.current_emissions


def test_signed_grid_colors_tooltips_cycles_resizing_and_cancel_are_local(app):
    block = HitOrMiss()
    before = copy.deepcopy(block.parameters)
    parameters = {key: before[key] for key in KERNEL_PARAMETERS}
    dialog = KernelEditor(block, parameters)
    try:
        assert dialog.grid.cells.dtype == np.int8 and dialog.grid.cells[0, 0] == -1
        cell = dialog.grid.index(0, 0)
        assert "Background" in dialog.grid.data(cell, Qt.ItemDataRole.ToolTipRole)
        assert dialog.grid.data(cell, Qt.ItemDataRole.BackgroundRole).name() == "#8f3d32"
        dialog.toggle(cell)
        assert dialog.grid.cells[0, 0] == 0
        dialog.toggle(cell)
        assert dialog.grid.cells[0, 0] == 1
        dialog.toggle(cell)
        assert dialog.grid.cells[0, 0] == -1
        dialog.width.setValue(5)
        dialog.height.setValue(4)
        assert dialog.grid.cells[0, 0] == -1 and dialog.grid.cells.dtype == np.int8
        assert dialog.parameters["custom_kernel"][0] == [-1, 1, -1, 0, 0]
        dialog.grid.cells[:] = 0
        dialog.refresh_status()
        assert not dialog.ok.isEnabled()
        dialog.toggle(dialog.grid.index(0, 0))
        assert dialog.ok.isEnabled()
        dialog.reject()
        assert block.parameters == before and parameters == {
            k: before[k] for k in KERNEL_PARAMETERS
        }
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_signed_settings_editor_dropdowns_graph_and_block_xml_roundtrip(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    block = HitOrMiss(
        parameters={
            "custom_kernel": [[-1, 1, 0], [0, -1, 1]],
            "anchor": [0, 1],
            "polarity": "dark",
            "check_count": True,
            "maximum_count": 5,
        }
    )
    assert block_entry(type(block)).path == ("OpenCV", "Image Processing (imgproc)", "Morphology")
    graph = Graph()
    graph.add(block)
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for field in ("border", "support", "input_mode", "polarity"):
            assert isinstance(window.inspector.fields[field], QComboBox)
        assert "custom_kernel" not in window.inspector.fields
        assert window.inspector.kernel_button is not None
        with pytest.raises(ValidationError) as error:
            block.validate_parameters(dict(block.parameters, minimum_count=7))
        assert set(error.value.fields) == {"minimum_count", "maximum_count"}
        path = tmp_path / "settings.cv.hit_or_miss.xml"
        save_block_xml(block, path)
        assert read_block_xml(path)["parameters"] == block.parameters
        graph.save(tmp_path / "graph.xml")
        restored = Graph.load(tmp_path / "graph.xml", builtin_registry())
        try:
            assert restored.to_dict() == graph.to_dict()
        finally:
            restored.close()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_signed_kernel_draft_runs_once_without_applying_to_saved_block(app, monkeypatch):
    from PySide6.QtCore import QElapsedTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog

    from opencv_dataflow.app import MainWindow

    graph, window = Graph(), MainWindow()
    block = graph.add(HitOrMiss())
    source = np.zeros((13, 17), np.uint8)
    source[6, 7:10], source[5:8, 8] = 255, 255
    successful(block, {"image": source})
    before = copy.deepcopy(block.parameters)

    def edit(dialog):
        dialog.toggle(dialog.grid.index(0, 0))  # Background -> ignored.
        assert dialog.parameters["custom_kernel"][0] == [0, 1, -1]
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(KernelEditor, "exec", edit)
    try:
        window.replace_graph(graph)
        inspector = window.inspector
        inspector.select(block)
        assert not inspector.live_results.isChecked()
        inspector.edit_kernel()
        assert block.parameters == before
        assert inspector.kernel_parameters["custom_kernel"][0] == [0, 1, -1]
        inspector.run_block.click()
        timer = QElapsedTimer()
        timer.start()
        while inspector._live_result is None and timer.elapsed() < 5000:
            app.processEvents()
            QTest.qWait(5)
        preview = inspector._live_result
        assert preview is not None and preview.state == "pass"
        assert preview.parameters["custom_kernel"][0] == [0, 1, -1]
        assert preview.last_result.outputs["points"] == [[8, 6]]
        assert block.parameters == before
        assert inspector.apply()
        assert block.parameters["custom_kernel"][0] == [0, 1, -1]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
