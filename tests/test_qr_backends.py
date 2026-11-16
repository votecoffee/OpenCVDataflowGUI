"""Native backend controls, safe failure, geometry/payload and persistence contracts."""

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.code_readers import ReadQRCode
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.qr_detector import (
    ARUCO_FLAGS,
    ARUCO_OPTIONS,
    QR_OPTIONS,
    make_detector,
)
from opencv_dataflow.tickets import TicketClock, ticket_key


def label():
    core = cv2.QRCodeEncoder_create().encode("LOT-0042")[2:-2, 2:-2]
    return np.repeat(np.repeat(np.pad(core, 4, constant_values=255), 6, 0), 6, 1), core


@pytest.mark.parametrize("backend", ["standard", "aruco"])
@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgra"])
def test_native_read_preserves_geometry_modules_ticket_and_source(backend, mode, encoding):
    image, core = label()
    if encoding != "gray":
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB if encoding == "rgb" else cv2.COLOR_GRAY2BGRA
        )
    image = ImageArray(image, encoding)
    original = image.copy()
    spatial = frame(image.shape[1::-1], "camera")
    ticket = TicketClock().issue("camera", "external", "inspection-42")
    reader = ReadQRCode(parameters={"backend": backend, "mode": mode})
    result = reader.execute({"image": ValuePacket(image, {"spatial": spatial, "ticket": ticket})})
    assert result is not None, reader.error
    assert result.passed and result.outputs["count"] == 1
    (record,) = result.outputs["codes"]
    assert record["text"] == "LOT-0042" and record["symbology"] == "QR_CODE"
    np.testing.assert_allclose(
        record["corners"], [[24, 24], [149, 24], [149, 149], [24, 149]], atol=1
    )
    np.testing.assert_array_equal(result.outputs["rectified"], core)
    np.testing.assert_array_equal(image, original)
    for name in ("codes", "count", "overlay", "rectified"):
        assert ticket_key(result.payloads[name]["ticket"]) == ticket_key(ticket)
    assert result.payloads["codes"]["spatial"] == spatial
    assert result.payloads["rectified"]["spatial"]["root_id"] != spatial["root_id"]
    assert result.payloads["rectified"]["spatial"]["image_size"] == [21, 21]
    outline, x_axis, y_axis, text = result.drawings["decoded codes"].objects
    assert outline.closed and text.text == record["text"]
    assert x_axis.kind == y_axis.kind == "arrow"
    assert result.event_data["qr_backend"] == backend


def test_all_native_aruco_and_qr_fields_are_exposed_with_controls_and_help():
    block = ReadQRCode(parameters={"backend": "aruco"})
    detector = make_detector(block.parameters)
    for native, options, extra in (
        (detector.getDetectorParameters(), QR_OPTIONS, set()),
        (
            detector.getArucoParameters(),
            ARUCO_OPTIONS,
            {entry[0] for entry in ARUCO_FLAGS.values()} | {"cornerRefinementMethod"},
        ),
    ):
        attributes = {
            name
            for name in dir(native)
            if not name.startswith("_") and not callable(getattr(native, name))
        }
        assert attributes == {row[0] for row in options.values()} | extra
        assert options.keys() <= block.defaults.keys()
        assert options.keys() <= block.parameter_ranges.keys()
        assert all(block.parameter_help[name] for name in options)


def test_explicit_controls_are_forwarded_with_angles_converted_to_radians():
    block = ReadQRCode(
        parameters={
            "backend": "aruco",
            "qr_min_module_size": 2.5,
            "qr_max_rotation_degrees": 22,
            "qr_max_colors_mismatch": 0.1,
            "aruco_adaptive_min": 5,
            "aruco_adaptive_max": 35,
            "aruco_adaptive_step": 6,
            "aruco_cell_margin": 0.2,
            "aruco_corner_refinement": "contour",
            "aruco_apriltag_critical_degrees": 12,
            "aruco_detect_inverted": True,
            "aruco_valid_bit_threshold": 0.45,
            "aruco_apriltag_deglitch": True,
        }
    )
    block.validate_parameters()
    detector = make_detector(block.parameters)
    q, a = detector.getDetectorParameters(), detector.getArucoParameters()
    assert q.minModuleSizeInPyramid == 2.5
    assert q.maxRotation == pytest.approx(math.radians(22))
    assert q.maxColorsMismatch == pytest.approx(0.1)
    assert (
        a.adaptiveThreshWinSizeMin,
        a.adaptiveThreshWinSizeMax,
        a.adaptiveThreshWinSizeStep,
    ) == (5, 35, 6)
    assert a.perspectiveRemoveIgnoredMarginPerCell == 0.2
    assert a.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_CONTOUR
    assert a.aprilTagCriticalRad == pytest.approx(math.radians(12))
    assert a.detectInvertedMarker and a.aprilTagDeglitch == 1
    assert a.validBitIdThreshold == pytest.approx(0.45)


@pytest.mark.parametrize("refinement", ["none", "subpixel", "contour", "apriltag", "aruco3"])
def test_native_candidate_modes_read_simple_label_without_shared_state(refinement):
    image, _ = label()
    block = ReadQRCode(
        parameters={
            "backend": "aruco",
            "aruco_corner_refinement": "subpixel" if refinement == "aruco3" else refinement,
            "aruco_use_aruco3": refinement == "aruco3",
        }
    )
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.passed and result.outputs["codes"][0]["text"] == "LOT-0042"


@pytest.mark.parametrize(
    "parameters,field",
    [
        ({"backend": "aruco", "mode": "curved"}, "mode"),
        ({"backend": "other"}, "backend"),
        ({"qr_min_module_size": 0}, "qr_min_module_size"),
        ({"qr_max_penalties": 1.1}, "qr_max_penalties"),
        ({"qr_max_rotation_degrees": float("nan")}, "qr_max_rotation_degrees"),
        ({"aruco_adaptive_min": 25}, "aruco_adaptive_min"),
        ({"aruco_adaptive_min": 4}, "aruco_adaptive_min"),
        ({"aruco_adaptive_max": 1001, "aruco_adaptive_step": 1}, "aruco_adaptive_step"),
        ({"aruco_min_perimeter_rate": 5}, "aruco_min_perimeter_rate"),
        ({"aruco_cell_margin": 0.5}, "aruco_cell_margin"),
        ({"aruco_corner_refinement": "other"}, "aruco_corner_refinement"),
        ({"aruco_use_aruco3": True}, "aruco_corner_refinement"),
        ({"aruco_detect_inverted": 1}, "aruco_detect_inverted"),
        ({"maximum_pixels": 441}, "maximum_pixels"),
    ],
)
def test_invalid_controls_are_reported_before_native_work(parameters, field, monkeypatch):
    import opencv_dataflow.code_readers as readers

    monkeypatch.setattr(
        readers, "make_detector", lambda _: pytest.fail("Native work on invalid input")
    )
    block = ReadQRCode(parameters=parameters)
    assert block.execute({"image": label()[0]}) is None
    assert block.error_kind == "invalid"
    if field == "maximum_pixels":
        assert "QR pixel limit" in block.error
    else:
        with pytest.raises(ValidationError) as error:
            block.validate_parameters()
        assert field in error.value.fields
    assert not any(block.current_emissions.values())


def test_aruco_has_no_silent_standard_fallback_and_missing_binding_is_invalid(monkeypatch):
    image, _ = label()
    monkeypatch.setattr(cv2, "QRCodeDetector", lambda: pytest.fail("Unexpected fallback"))
    block = ReadQRCode(parameters={"backend": "aruco", "mode": "single"})
    assert block.execute({"image": image}).passed
    monkeypatch.delattr(cv2, "QRCodeDetectorAruco")
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid"
    with pytest.raises(ValidationError) as error:
        make_detector(block.parameters)
    assert "backend" in error.value.fields
    assert not any(block.current_emissions.values())


def test_empty_selection_skips_native_and_low_reads_are_inspection_failure(monkeypatch):
    import opencv_dataflow.code_readers as readers

    image, _ = label()
    monkeypatch.setattr(
        readers, "make_detector", lambda _: pytest.fail("Empty mask entered decoder")
    )
    block = ReadQRCode(parameters={"backend": "aruco"})
    result = block.execute({"image": image, "mask": np.zeros_like(image)})
    assert result is not None and not result.passed
    assert not block.error and result.outputs["codes"] == []
    assert result.event_data["qr_backend"] == "aruco"
    assert result.outputs["rectified"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "code_read_count_below_minimum"


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_unsupported_depth_is_invalid_without_hidden_conversion(dtype):
    block = ReadQRCode(parameters={"backend": "aruco"})
    assert block.execute({"image": label()[0].astype(dtype)}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_native_exception_is_contained_and_next_invocation_recovers(monkeypatch):
    import opencv_dataflow.code_readers as readers

    block = ReadQRCode(parameters={"backend": "aruco"})
    real = readers.make_detector

    def broken(_):
        raise cv2.error("Controlled native QR failure")

    monkeypatch.setattr(readers, "make_detector", broken)
    assert block.execute({"image": label()[0]}) is None
    assert block.state == "error" and "Controlled native" in block.error
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(readers, "make_detector", real)
    assert block.execute({"image": label()[0]}).passed


def test_overlay_and_optional_result_objects_can_both_be_disabled():
    block = ReadQRCode(parameters={"backend": "aruco", "draw_overlay": False})
    block.set_graphics_mode("off")
    result = block.execute({"image": label()[0]})
    assert result.passed and result.outputs["overlay"] is NO_VALUE
    assert not result.drawings and not result.graphics


def test_parallel_reader_instances_do_not_share_native_detector_results():
    def read(text):
        core = cv2.QRCodeEncoder_create().encode(text)[2:-2, 2:-2]
        image = np.repeat(np.repeat(np.pad(core, 4, constant_values=255), 6, 0), 6, 1)
        block = ReadQRCode(parameters={"backend": "aruco", "draw_overlay": False})
        result = block.execute({"image": image})
        assert result is not None, block.error
        return result.outputs["codes"][0]["text"]

    expected = ["BRANCH-A", "BRANCH-B", "BRANCH-C", "BRANCH-D"]
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(read, expected)) == expected


def test_parameter_xml_project_roundtrip_and_dropdowns(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    registry, graph = builtin_registry(), Graph()
    block = graph.add(
        registry.create(
            "inspect.read_qr",
            parameters={
                "backend": "aruco",
                "qr_max_rotation_degrees": 23,
                "aruco_corner_refinement": "subpixel",
            },
        )
    )
    xml = tmp_path / "reader.inspect.read_qr.xml"
    save_block_xml(block, xml)
    assert read_block_xml(xml)["parameters"] == block.parameters
    graph.save(tmp_path / "project.json")
    reopened = Graph.load(tmp_path / "project.json", registry)
    try:
        assert reopened.blocks[block.id].parameters == block.parameters
    finally:
        reopened.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name, choices in block.parameter_choices.items():
            widget = window.inspector.fields[name]
            assert isinstance(widget, QComboBox)
            assert [widget.itemText(i) for i in range(widget.count())] == list(choices)
        assert window.inspector.fields["qr_min_module_size"].toolTip()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
