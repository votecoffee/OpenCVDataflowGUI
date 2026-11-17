"""Known media-time selection, independent sequential decode and guarded native failures."""

import copy
import hashlib

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock, ticket_key

TYPE = "input.video_at_time"


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "known.avi"
    writer = builtin_registry().create(
        "output.video",
        parameters={
            "path": str(path),
            "backend": "ffmpeg",
            "codec": "FFV1",
            "fps": 20,
        },
    )
    images = [np.full((40, 60, 3), 15 + i * 25, np.uint8) for i in range(8)]
    try:
        for image in images:
            assert writer.execute({"image": image}) is not None, writer.error
    finally:
        writer.close()
    return path, images


def source(clip, **parameters):
    return builtin_registry().create(TYPE, parameters={"path": str(clip[0]), **parameters})


@pytest.mark.parametrize("method", ["native_then_scan", "scan_from_start"])
@pytest.mark.parametrize(
    "requested,policy,index",
    [
        (0, "nearest", 0),
        (175, "nearest", 3),
        (175, "at_or_before", 3),
        (175, "at_or_after", 4),
        (210, "nearest", 4),
        (210, "at_or_after", 5),
        (350, "at_or_before", 7),
        (375, "nearest", 7),
        (375, "at_or_before", 7),
    ],
)
def test_known_lossless_times_bracket_policy_and_earlier_ties(
    clip, method, requested, policy, index
):
    block = source(clip, time_ms=requested, selection=policy, seek_method=method)
    result = block.execute({})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], clip[1][index])
    info = result.outputs["info"]
    assert info["frame_index"] == index and info["timestamp_ms"] == pytest.approx(index * 50)
    assert info["timestamp_source"] == "decoder"
    assert info["seek"]["error_ms"] == pytest.approx(index * 50 - requested)
    assert result.payloads["image"]["video_frame"] == info
    assert result.outputs["ticket"]["value"] == f"known.avi#{index:06d}"
    if method == "scan_from_start":
        assert info["frame_index_source"] == "decoded_ordinal"
        assert info["seek"]["native_set_accepted"] is None
    if method == "native_then_scan" and requested == 175:
        assert info["seek"]["fallback_reason"] == "native_seek_overshot"
        assert info["seek"]["native_first_timestamp_ms"] == 200
    block.close()


def test_native_exact_and_preroll_bracketing_do_not_require_start_scan(clip):
    for time, preroll, expected in [(250, 0, 5), (175, 75, 3)]:
        block = source(clip, time_ms=time, preroll_ms=preroll, seek_method="native_only")
        result = block.execute({})
        assert result is not None, block.error
        assert result.outputs["info"]["frame_index"] == expected
        seek = result.outputs["info"]["seek"]
        assert seek["method_used"] == "native_bracket" and seek["fallback_reason"] is None
        assert seek["decoded_frames"] <= 3


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"time_ms": 175, "seek_method": "native_only"}, "overshot"),
        ({"time_ms": 400, "selection": "at_or_after"}, "No frame"),
        ({"time_ms": 2000, "maximum_error_ms": 100}, "Maximum time error"),
        ({"time_ms": 175, "maximum_error_ms": 24}, "Maximum time error"),
        ({"time_ms": 175, "maximum_read_frames": 1}, "Maximum read frames"),
        ({"maximum_frame_pixels": 20}, "Maximum frame pixels"),
        ({"maximum_working_bytes": 20}, "Maximum working bytes"),
    ],
)
def test_no_eligible_frame_or_guard_never_assigns_ticket_or_emits_stale_data(
    clip, parameters, message
):
    block = source(clip, **parameters)
    assert block.execute({}) is None and message in block.error
    assert block.error_kind == "invalid" and block._source_ticket is None
    assert not any(block.current_emissions.values()) and block._prepared is None


def test_nonsequential_connected_requests_keep_fresh_tickets_grids_and_request_provenance(clip):
    block = source(clip, time_ms=999)
    request = TicketClock().issue("request")
    identities, grids = set(), []
    for time, index in [(300, 6), (50, 1), (200, 4), (50, 1)]:
        result = block.execute(
            {
                "time_ms": ValuePacket(
                    time,
                    {
                        "ticket": request,
                        "spatial": frame([20, 30], "request-grid"),
                    },
                )
            }
        )
        assert result is not None, block.error
        np.testing.assert_array_equal(result.outputs["image"], clip[1][index])
        assert result.outputs["info"]["seek"]["target_source"] == "input"
        assert result.payloads["image"]["video_request_ticket"] == request
        assert result.payloads["info"]["spatial"] == result.payloads["image"]["spatial"]
        assert result.payloads["info"]["spatial"]["root_id"] != "request-grid"
        identities.add(ticket_key(result.outputs["ticket"]))
        grids.append(result.payloads["image"]["spatial"])
    assert len(identities) == 4 and all(grids[i] != grids[i + 1] for i in range(3))
    block.ticket_settings = {**block.ticket_settings, "mode": "external"}
    result = block.execute({"ticket": "PLC-1", "time_ms": 0})
    assert result.outputs["ticket"]["value"] == "PLC-1"


def capture_wrapper(monkeypatch, update):
    from opencv_dataflow import video_time

    native, handles = cv2.VideoCapture, []

    class Capture:
        def __init__(self):
            self.real = native()
            self.closed, self.reads = False, 0
            handles.append(self)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def set(self, prop, value):
            return self.real.set(prop, value)

        def get(self, prop):
            return self.real.get(prop)

        def read(self):
            self.reads += 1
            return self.real.read()

        def release(self):
            self.closed = True
            self.real.release()

    update(Capture)
    monkeypatch.setattr(video_time.cv2, "VideoCapture", Capture)
    return handles


@pytest.mark.parametrize("policy", ["decoder_only", "decoder_or_fps"])
def test_missing_time_index_properties_need_explicit_estimates_and_scan_ordinals(
    clip, monkeypatch, policy
):
    def update(cls):
        def get(self, prop):
            if prop in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_POS_FRAMES):
                return float("nan")
            return self.real.get(prop)

        cls.get = get

    handles = capture_wrapper(monkeypatch, update)
    block = source(clip, time_ms=225, timestamp_policy=policy)
    result = block.execute({})
    if policy == "decoder_only":
        assert result is None and "increasing decoder time" in block.error
    else:
        assert result is not None, block.error
        info = result.outputs["info"]
        assert info["timestamp_ms"] == 200 and info["timestamp_source"] == "frame_rate_estimate"
        assert info["frame_index"] == 4 and info["decoder_frame_index"] is None
        assert info["frame_index_source"] == "decoded_ordinal"
        assert info["decoder_timestamp_ms"] is None
        np.testing.assert_array_equal(result.outputs["image"], clip[1][4])
    assert all(h.closed for h in handles)


def test_native_seek_rejection_reopens_from_start_and_closes_every_handle(clip, monkeypatch):
    def update(cls):
        def set(self, prop, value):
            return False if prop == cv2.CAP_PROP_POS_MSEC else self.real.set(prop, value)

        cls.set = set

    handles = capture_wrapper(monkeypatch, update)
    block = source(clip, time_ms=100)
    result = block.execute({})
    assert result is not None, block.error
    assert result.outputs["info"]["seek"]["fallback_reason"] == "native_seek_rejected"
    assert len(handles) == 2 and all(h.closed for h in handles)


def test_estimated_native_probe_time_retains_its_source_and_raw_bad_property(clip, monkeypatch):
    def update(cls):
        def get(self, prop):
            return 0 if prop == cv2.CAP_PROP_POS_MSEC else self.real.get(prop)

        cls.get = get

    handles = capture_wrapper(monkeypatch, update)
    block = source(clip, time_ms=225, timestamp_policy="decoder_or_fps")
    result = block.execute({})
    assert result is not None, block.error
    info = result.outputs["info"]
    assert info["timestamp_source"] == "frame_rate_estimate" and info["timestamp_ms"] == 200
    assert info["seek"]["native_first_timestamp_ms"] == 250
    assert info["seek"]["native_first_timestamp_source"] == "frame_rate_estimate"
    assert info["seek"]["native_first_decoder_timestamp_ms"] == 0
    assert all(h.closed for h in handles)


def test_cancelled_linear_scan_closes_decoder_without_ticket_or_output(clip, monkeypatch):
    handles = capture_wrapper(monkeypatch, lambda cls: None)
    block = source(clip, time_ms=350, seek_method="scan_from_start")
    result = block.execute({}, lambda: bool(handles) and handles[-1].reads >= 3)
    assert result is not None and result.outputs["image"] is NO_VALUE
    assert result.outputs["ticket"] is NO_VALUE and not any(block.current_emissions.values())
    assert handles[0].reads == 3 and all(h.closed for h in handles)


def test_changed_file_cannot_publish_an_old_frame(clip, monkeypatch):
    def update(cls):
        def read(self):
            answer = self.real.read()
            with clip[0].open("ab") as stream:
                stream.write(b"\0")
            return answer

        cls.read = read

    handles = capture_wrapper(monkeypatch, update)
    block = source(clip)
    assert block.execute({}) is None and "changed during" in block.error
    assert all(h.closed for h in handles) and block._source_ticket is None


def test_native_exception_closes_handle_and_records_traceback_before_ticketing(clip, monkeypatch):
    def update(cls):
        def read(self):
            raise RuntimeError("Controlled seek decoder failure")

        cls.read = read

    handles = capture_wrapper(monkeypatch, update)
    block = source(clip)
    assert block.execute({}) is None and block.error_kind == "exception"
    assert "Controlled seek decoder failure" in block.error
    assert all(h.closed for h in handles) and block._source_ticket is None
    assert not any(block.current_emissions.values())
    events = block.logger.snapshot()["events"]
    assert any(e.get("exception") and "RuntimeError" in e["exception"]["traceback"] for e in events)


def test_save_as_rebases_video_reference_without_copying_or_changing_original(clip, tmp_path):
    path = clip[0]
    before = path.read_bytes()
    graph = Graph()
    graph.bind_project_dir(path.parent)
    block = graph.add(source(clip, path="known.avi", path_base="project", time_ms=225))
    graph.save(tmp_path / "original.json")
    moved = tmp_path / "moved"
    moved.mkdir()
    graph.save(moved / "project.json")
    restored = Graph.load(moved / "project.json", builtin_registry())
    try:
        reopened = restored.blocks[block.id]
        result = reopened.execute({})
        assert result is not None, reopened.error
        np.testing.assert_array_equal(result.outputs["image"], clip[1][4])
        assert path.read_bytes() == before and not (moved / "known.avi").exists()
    finally:
        graph.close()
        restored.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"time_ms": -1},
        {"time_ms": True},
        {"time_ms": "0"},
        {"time_ms": float("inf")},
        {"selection": "anything"},
        {"seek_method": "unsafe"},
        {"frame_ticket": "false"},
        {"maximum_read_frames": 0},
        {"decoder_threads": 1.0},
        {"color_format": "hsv", "pixel_depth": "uint16"},
    ],
)
def test_invalid_parameters_do_not_open_native_handles(clip, parameters, monkeypatch):
    from opencv_dataflow import video_time

    monkeypatch.setattr(
        video_time.cv2, "VideoCapture", lambda: pytest.fail("Invalid source opened")
    )
    block = source(clip, **parameters)
    assert block.execute({}) is None and block.error_kind == "invalid"


def test_interframe_compressed_seek_matches_independent_full_decode(tmp_path):
    path = tmp_path / "gop.avi"
    writer = builtin_registry().create(
        "output.video",
        parameters={
            "path": str(path),
            "backend": "ffmpeg",
            "codec": "mp4v",
            "fps": 16,
        },
    )
    try:
        for index in range(32):
            image = np.full((64, 96, 3), index * 7, np.uint8)
            image[16:48, 4 + index : 28 + index] = [180, 240, 250]
            assert writer.execute({"image": image}) is not None, writer.error
    finally:
        writer.close()
    native = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    reference = []
    frame_types = []
    try:
        while True:
            ok, image = native.read()
            if not ok:
                break
            reference.append(hashlib.sha256(image.tobytes()).hexdigest())
            frame_types.append(native.get(cv2.CAP_PROP_FRAME_TYPE))
    finally:
        native.release()
    assert len(reference) == 32
    assert ord("P") in frame_types or ord("B") in frame_types
    block = builtin_registry().create(TYPE, parameters={"path": str(path)})
    for target in (62.5, 90, 593, 750, 1050, 1875, 200):
        result = block.execute({"time_ms": target})
        assert result is not None, block.error
        index = min(range(32), key=lambda i: (abs(i * 62.5 - target), i))
        assert result.outputs["info"]["frame_index"] == index
        assert result.outputs["info"]["timestamp_ms"] == pytest.approx(index * 62.5)
        assert hashlib.sha256(result.outputs["image"].tobytes()).hexdigest() == reference[index]


def test_xml_editor_and_runtime_copy_have_no_decoder_and_preserve_configuration(
    app, clip, tmp_path
):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.toolbox import block_entry

    block = source(clip, time_ms=175, selection="at_or_after")
    graph = Graph()
    graph.add(block)
    path = tmp_path / "time.xml"
    definition = copy.deepcopy(graph.to_dict())
    export_project_xml(definition, path)
    assert read_project_xml(path) == definition
    restored = Graph.from_dict(read_project_xml(path), builtin_registry())
    assert restored.to_dict() == definition and restored.blocks[block.id]._prepared is None
    assert block_entry(block).path[0] == "OpenCV" and "videoio" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert all(
            isinstance(window.inspector.fields[name], QComboBox)
            for name in ("selection", "seek_method", "timestamp_policy")
        )
        assert block._prepared is None and block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
