"""Signature/count/encoder distinctions, immutable source bytes, failures and persistence."""

import hashlib
import os
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.codec_probe import COUNT_FLAGS, WRITERS, ProbeImageFile
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry


def executed(block, inputs=None):
    result = block.execute(inputs or {})
    assert result is not None, block.error
    return result


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "part.png"
    ok, data = cv2.imencode(".png", np.arange(99, dtype=np.uint8).reshape(9, 11))
    assert ok
    path.write_bytes(data.tobytes())
    return path


def block_for(path, **p):
    return ProbeImageFile(parameters={"path": str(path), **p})


@pytest.mark.parametrize("extension", [".png", ".jpg", ".bmp", ".tiff", ".webp"])
def test_real_stills_signature_count_and_input_unchanged(tmp_path, extension):
    image = np.full((31, 43, 3), [12, 93, 211], np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    # Deliberately unrelated suffix: reader selection uses the file signature.
    path = tmp_path / ("part" + extension + ".unknown")
    data = encoded.tobytes()
    path.write_bytes(data)
    result = executed(block_for(path))
    assert result.passed and result.outputs["reader_available"] is True
    assert result.outputs["writer_available"] is True and result.outputs["page_count"] == 1
    info = result.outputs["info"]
    assert info["pixel_decode_verified"] is False
    assert info["file"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert info["file"]["byte_count"] == len(data) and path.read_bytes() == data
    assert info["file"]["source_extension"] == ".unknown"


@pytest.mark.parametrize("flag", list(COUNT_FLAGS))
def test_multipage_tiff_native_flags_and_unicode_snapshot(tmp_path, flag):
    pages = [np.full((7 + i, 11), i * 1000, np.uint16) for i in range(3)]
    ok, data = cv2.imencodemulti(".tiff", pages, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    assert ok
    path = tmp_path / "\u90e8\u54c1-\u00e9.tiff"
    path.write_bytes(data.tobytes())
    result = executed(block_for(path, mode="file", count_flags=flag))
    assert result.passed and result.outputs["page_count"] > 0
    if flag != "gdal":
        assert result.outputs["page_count"] == 3
    assert result.outputs["info"]["file"]["count_flags_value"] == COUNT_FLAGS[flag]
    assert result.outputs["writer_available"] is NO_VALUE


@pytest.mark.parametrize("codec,extension", [("apng", ".png"), ("gif", ".gif"), ("webp", ".webp")])
def test_native_animation_frame_inventory(tmp_path, codec, extension):
    animation = cv2.Animation()
    frames = []
    for i in range(3):
        image = np.full((20, 30, 3), i * 40, np.uint8)
        image[4:12, 3 + i * 5 : 8 + i * 5] = 255
        frames.append(image)
    animation.frames, animation.durations = frames, [80, 120, 200]
    ok, data = cv2.imencodeanimation(extension, animation)
    assert ok
    path = tmp_path / (codec + extension)
    path.write_bytes(data.tobytes())
    result = executed(block_for(path, mode="file"))
    assert result.passed and result.outputs["page_count"] == 3


def test_native_calls_share_one_snapshot_and_do_not_load_pixels(image_file, monkeypatch):
    old_bytes = image_file.read_bytes()
    native_reader, native_count = cv2.haveImageReader, cv2.imcount
    paths = []

    def reader(path):
        paths.append(Path(path))
        assert Path(path).read_bytes() == old_bytes
        image_file.write_bytes(b"new external bytes")
        return native_reader(path)

    def count(path, flags):
        paths.append(Path(path))
        assert Path(path).read_bytes() == old_bytes
        return native_count(path, flags)

    def pixels(*args, **kwargs):
        raise AssertionError("Probe must not request pixel decoding")

    monkeypatch.setattr(cv2, "haveImageReader", reader)
    monkeypatch.setattr(cv2, "imcount", count)
    for name in (
        "imread",
        "imdecode",
        "imreadmulti",
        "imdecodemulti",
        "imreadanimation",
        "imdecodeanimation",
    ):
        monkeypatch.setattr(cv2, name, pixels)
    result = executed(block_for(image_file, mode="file"))
    assert result.passed and paths[0] == paths[1]
    assert not paths[0].exists() and not paths[0].parent.exists()
    assert result.outputs["info"]["file"]["sha256"] == hashlib.sha256(old_bytes).hexdigest()


def test_header_success_does_not_claim_pixel_integrity(image_file):
    raw = bytearray(image_file.read_bytes())
    start = raw.index(b"IDAT")
    length = int.from_bytes(raw[start - 4 : start], "big")
    raw[start + 4] = 0  # Invalid zlib stream with a valid PNG chunk CRC.
    struct.pack_into(">I", raw, start + 4 + length, zlib.crc32(raw[start : start + 4 + length]))
    image_file.write_bytes(raw)
    result = executed(block_for(image_file, mode="file"))
    assert result.passed and result.outputs["page_count"] == 1
    assert not result.outputs["info"]["pixel_decode_verified"]
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED) is None


def test_changing_file_during_read_is_reported_before_native(image_file, monkeypatch):
    native = os.fstat
    calls = []

    def changed(descriptor):
        value = native(descriptor)
        calls.append(1)
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + int(len(calls) > 1),
        )

    monkeypatch.setattr(os, "fstat", changed)

    def forbidden(*args):
        raise AssertionError("Unstable snapshots must not reach native probes")

    monkeypatch.setattr(cv2, "haveImageReader", forbidden)
    result = executed(block_for(image_file, mode="file"))
    assert not result.passed and result.fail_reasons[0]["code"] == "image_file_changed"
    assert result.outputs["reader_available"] is NO_VALUE


def test_permission_error_is_a_file_diagnostic(image_file, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("controlled access denial")

    monkeypatch.setattr(Path, "open", denied)
    result = executed(block_for(image_file))
    assert not result.passed and result.fail_reasons[0]["code"] == "image_file_io_error"
    assert result.outputs["writer_available"] is True


@pytest.mark.parametrize(
    "kind,code",
    [
        ("missing", "image_file_missing"),
        ("directory", "image_file_io_error"),
        ("unsupported", "image_reader_unavailable"),
        ("truncated", "image_header_unreadable"),
    ],
)
def test_invalid_external_files_report_diagnostics_without_exception(tmp_path, kind, code):
    path = tmp_path / "input.bin"
    if kind == "directory":
        path.mkdir()
    elif kind == "unsupported":
        path.write_bytes(b"not an image")
    elif kind == "truncated":
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    block = block_for(path, mode="file")
    result = executed(block)
    assert not result.passed and block.state == "fail" and not block.error
    assert result.fail_reasons[0]["code"] == code
    if kind == "truncated":
        assert result.outputs["reader_available"] is True and result.outputs["page_count"] == 0
    elif kind in ("missing", "directory"):
        assert result.outputs["reader_available"] is NO_VALUE


def test_byte_limit_is_checked_before_native_and_page_limit_preserves_count(
    image_file, monkeypatch
):
    calls = []
    native = cv2.haveImageReader
    monkeypatch.setattr(cv2, "haveImageReader", lambda *a: calls.append(a) or native(*a))
    result = executed(block_for(image_file, maximum_file_bytes=1))
    assert not result.passed and not calls and result.outputs["reader_available"] is NO_VALUE
    assert result.outputs["writer_available"] is True
    result = executed(block_for(image_file, minimum_pages=2))
    assert not result.passed and result.outputs["page_count"] == 1
    assert result.fail_reasons[0]["code"] == "image_page_count_out_of_range"


@pytest.mark.parametrize("format", list(WRITERS))
def test_writer_only_matches_registry_without_file_access(format, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Writer mode must not inspect files")

    monkeypatch.setattr(Path, "open", forbidden)
    extension = WRITERS[format] or ".ras"
    try:
        expected = cv2.haveImageWriter(extension)
    except cv2.error:
        expected = None
    result = executed(
        ProbeImageFile(
            parameters={"mode": "writer", "writer_format": format, "custom_extension": "RAS"}
        )
    )
    assert result.outputs["reader_available"] is result.outputs["page_count"] is NO_VALUE
    if expected is None:
        assert result.outputs["writer_available"] is NO_VALUE
    else:
        assert result.outputs["writer_available"] is bool(expected)
    assert result.passed == bool(expected)
    assert result.outputs["info"]["writer"]["extension"] == extension


def test_native_exceptions_are_visible_and_temporary_files_are_removed(image_file, monkeypatch):
    paths = []

    def broken(path, flags):
        paths.append(Path(path))
        raise cv2.error("controlled count error")

    monkeypatch.setattr(cv2, "imcount", broken)

    def encoder(*args):
        raise cv2.error("controlled encoder probe error")

    monkeypatch.setattr(cv2, "haveImageWriter", encoder)
    result = executed(block_for(image_file))
    assert not result.passed and result.outputs["reader_available"] is True
    assert result.outputs["page_count"] is result.outputs["writer_available"] is NO_VALUE
    assert {d["code"] for d in result.fail_reasons} == {
        "image_native_probe_error",
        "image_writer_probe_error",
    }
    assert not paths[0].exists() and not paths[0].parent.exists()


def test_wired_path_overrides_configured_path_and_keeps_ticket(image_file):
    ticket = TicketClock().issue("inspection-controller")
    result = executed(
        block_for("missing.png"),
        {"path": ValuePacket(str(image_file), {"ticket": ticket, "batch": {"id": 7}})},
    )
    assert result.passed
    assert ticket_key(result.payloads["info"]["ticket"]) == ticket_key(ticket)
    assert result.payloads["info"]["batch"] == {"id": 7}


@pytest.mark.parametrize(
    "p",
    [
        {"mode": "bad"},
        {"maximum_file_bytes": 0},
        {"minimum_pages": 4, "maximum_pages": 2},
        {"writer_format": "custom", "custom_extension": "folder/output.png"},
        {"path": ""},
    ],
)
def test_bad_configuration_has_no_output(p, image_file):
    block = ProbeImageFile(parameters={"path": str(image_file), **p})
    assert (
        block.execute({}) is None and block.error_kind == "invalid" and not block.current_emissions
    )


@pytest.mark.parametrize("path_base", ["project", "working_directory"])
def test_saved_project_paths_dropdowns_and_xml(app, tmp_path, image_file, path_base, monkeypatch):
    from opencv_dataflow.app import MainWindow

    monkeypatch.chdir(tmp_path)
    block = ProbeImageFile(
        parameters={
            "path": image_file.name,
            "path_base": path_base,
            "mode": "file",
            "minimum_pages": 1,
            "maximum_pages": 4,
        }
    )
    assert block_entry(type(block)).path == ("OpenCV", "Image Codecs (imgcodecs)", "Codecs")
    graph = Graph()
    graph.add(block)
    graph.save(tmp_path / "source.json")
    assert executed(block).passed
    (tmp_path / "new-location").mkdir()
    graph.save(tmp_path / "new-location" / "saved.json")
    assert executed(block).passed
    if path_base == "working_directory":
        assert block.parameters["path"] == image_file.name
    path = tmp_path / "settings.cv.probe_image_file.xml"
    save_block_xml(block, path)
    assert read_block_xml(path)["parameters"] == block.parameters
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name in block.parameter_choices:
            assert isinstance(window.inspector.fields[name], QComboBox)
        with pytest.raises(ValidationError) as exc:
            block.validate_parameters(dict(block.parameters, maximum_pages=0))
        assert exc.value.fields == ("maximum_pages",)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
