"""Controlled camera backends plus actual lossless file decoding, without physical devices."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import Camera, builtin_registry
from opencv_dataflow.capture_negotiation import PROPERTIES, fourcc_text
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.toolbox import block_entry


class ControlledCapture:
    """Driver replies are independent of the image geometry and setter success."""

    def __init__(self, *args):
        self.opened = True
        self.values = {PROPERTIES["width"]: 128, PROPERTIES["height"]: 96, PROPERTIES["fps"]: 24}
        self.ignored, self.rejected, self.unavailable = set(), set(), {}
        self.writes, self.queries = [], []
        self.reads, self.released = 0, False
        self.image = np.full((96, 128, 3), (20, 60, 180), np.uint8)
        self.read_error = None

    def set(self, code, value):
        self.writes.append((code, value))
        if code in self.rejected:
            return False
        if code not in self.ignored:
            self.values[code] = value
        return True

    def get(self, code):
        self.queries.append(code)
        value = self.unavailable.get(code, self.values.get(code, -1))
        if isinstance(value, Exception):
            raise value
        return value

    def getBackendName(self):
        return "CONTROLLED_FIXTURE"

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if isinstance(self.read_error, Exception):
            raise self.read_error
        return self.read_error is None, self.image.copy()

    def release(self):
        self.released = True


@pytest.fixture
def backend(monkeypatch):
    handles = []

    def open_capture(*args):
        handle = ControlledCapture(*args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    return handles


def report(result):
    return result.outputs["info"]["capture_negotiation"]


def test_camera_negotiates_once_reuses_and_reopens_after_parameter_change(backend):
    block = Camera(
        parameters={"width": 128, "height": 96, "fps": 24, "negotiation_policy": "require_readback"}
    )
    try:
        first, second = block.execute({}), block.execute({})
        assert first.passed and second.passed
        assert len(backend) == 1 and backend[0].reads == 2 and len(backend[0].writes) == 3
        assert report(first)["readback_fresh"] and not report(second)["readback_fresh"]
        assert report(second)["readback_frame_index"] == 0
        assert report(second)["frame_index"] == 1
        assert first.payloads["image"]["spatial"] == second.payloads["image"]["spatial"]
        assert first.outputs["ticket"] != second.outputs["ticket"]
        block.parameters["device"] = 1
        third = block.execute({})
        assert third.passed and backend[0].released and len(backend) == 2
        assert third.payloads["image"]["spatial"] != first.payloads["image"]["spatial"]
    finally:
        block.close()
    assert backend[-1].released


@pytest.mark.parametrize(
    "policy,passed", [("require_setter", True), ("require_readback", False), ("report", True)]
)
def test_true_setter_does_not_prove_frame_geometry(backend, policy, passed):
    block = Camera(parameters={"width": 640, "negotiation_policy": policy})
    try:
        result = block.execute({})
        assert result.passed is passed
        row = report(result)["properties"][0]
        assert row["reported_match"] and row["frame_match"] is False
        assert row["state"] == "frame_mismatch"
        assert report(result)["decoded_frame"]["width"] == 128
        assert (result.outputs["image"] is NO_VALUE) is (not passed)
        assert not report(result)["hardware_support_proven"]
        if not passed:
            assert backend[0].released
            assert result.fail_reasons[0]["code"] == "capture.negotiation_rejected"
    finally:
        block.close()


@pytest.mark.parametrize(
    "policy,passed", [("require_setter", False), ("require_readback", False), ("report", True)]
)
def test_rejected_setter_is_not_rescued_by_an_already_matching_value(monkeypatch, policy, passed):
    handle = ControlledCapture()
    handle.rejected.add(PROPERTIES["width"])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera(parameters={"width": 128, "negotiation_policy": policy})
    try:
        result = block.execute({})
        assert result.passed is passed and report(result)["properties"][0]["reported_match"]
        assert report(result)["properties"][0]["state"] == "setter_rejected"
        assert handle.reads == int(passed)
        assert (result.outputs["image"] is NO_VALUE) is (not passed)
    finally:
        block.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), RuntimeError("unavailable")])
def test_unavailable_getter_never_becomes_a_match(monkeypatch, value):
    handle = ControlledCapture()
    handle.unavailable[PROPERTIES["fps"]] = value
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera(parameters={"fps": 24, "negotiation_policy": "require_readback"})
    try:
        result = block.execute({})
        assert not result.passed
        assert report(result)["properties"][0]["state"] == "readback_unavailable"
        json.dumps(report(result), allow_nan=False)
    finally:
        block.close()


@pytest.mark.parametrize("value", [-1, 0])
def test_ambiguous_sentinels_are_preserved_not_called_hardware_support(backend, value):
    block = Camera(
        parameters={
            "gain_mode": "backend_value",
            "gain_value": value,
            "negotiation_policy": "require_readback",
        }
    )
    try:
        result = block.execute({})
        row = report(result)["properties"][0]
        assert result.passed and row["reported_match"]
        assert row["possible_unsupported_sentinel"]
        assert not report(result)["hardware_support_proven"]
    finally:
        block.close()


def test_native_units_manual_control_order_and_zero_values(backend):
    block = Camera(
        parameters={
            "fourcc": "YUY2",
            "auto_exposure_mode": "backend_value",
            "auto_exposure_value": 0.25,
            "exposure_mode": "backend_value",
            "exposure_value": -7,
            "focus_mode": "manual",
            "focus_value": 0,
            "white_balance_mode": "manual",
            "white_balance_temperature": 4200,
            "additional_controls": {"brightness": -2},
        }
    )
    try:
        result = block.execute({})
        assert result.passed
        writes = backend[0].writes
        assert writes.index((cv2.CAP_PROP_AUTOFOCUS, 0)) < writes.index((cv2.CAP_PROP_FOCUS, 0))
        assert writes.index((cv2.CAP_PROP_AUTO_WB, 0)) < writes.index(
            (cv2.CAP_PROP_WB_TEMPERATURE, 4200)
        )
        assert (cv2.CAP_PROP_EXPOSURE, -7) in writes
        assert report(result)["fourcc_text"] == "YUY2"
        assert report(result)["requested"]["auto_exposure"] == 0.25
    finally:
        block.close()


def test_every_frame_readback_and_cached_geometry_are_both_explicit(backend):
    block = Camera(parameters={"width": 128, "readback": "every_frame"})
    try:
        first = block.execute({})
        backend[0].values[PROPERTIES["width"]] = 120
        second = block.execute({})
        row = report(second)["properties"][0]
        assert row["frame_match"] and not row["reported_match"]
        assert report(second)["readback_fresh"] and report(second)["readback_frame_index"] == 1
        assert report(first)["properties"][0]["state"] == "reported_match"
        assert len(backend[0].writes) == 1
    finally:
        block.close()


def test_ignored_rate_retains_requested_and_actual_values(monkeypatch):
    handle = ControlledCapture()
    handle.ignored.add(PROPERTIES["fps"])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera(parameters={"fps": 30, "negotiation_policy": "require_readback"})
    try:
        result = block.execute({})
        assert not result.passed
        row = report(result)["properties"][0]
        assert row["requested"] == 30 and row["after_frame"]["value"] == 24
        assert row["state"] == "readback_mismatch" and row["setter"]["returned"]
    finally:
        block.close()


def test_cached_readback_still_rejects_later_frame_geometry_changes(backend):
    block = Camera(parameters={"width": 128, "negotiation_policy": "require_readback"})
    try:
        assert block.execute({}).passed
        backend[0].image = backend[0].image[:, :120]
        result = block.execute({})
        assert not result.passed and not report(result)["readback_fresh"]
        assert report(result)["properties"][0]["state"] == "frame_mismatch"
    finally:
        block.close()


def test_setter_exception_is_contained_with_explicit_report(monkeypatch):
    handle = ControlledCapture()

    def broken_set(*args):
        raise RuntimeError("driver refuses operation")

    handle.set = broken_set
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera(parameters={"fps": 24})
    result = block.execute({})
    assert result is not None and not result.passed and handle.released
    assert report(result)["properties"][0]["setter"]["exception"] == "driver refuses operation"


def test_open_failure_is_contained_and_released(monkeypatch):
    handle = ControlledCapture()
    handle.opened = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera()
    assert block.execute({}) is None and block.state == "error" and handle.released


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((4, 4, 2), np.uint8),
        np.zeros((4, 4), np.int16),
        np.full((4, 4), np.nan),
        np.zeros((0, 4), np.uint8),
    ],
)
def test_invalid_decoded_frames_release_the_handle(monkeypatch, pixels):
    handle = ControlledCapture()
    handle.image = pixels
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera()
    assert block.execute({}) is None and handle.released


@pytest.mark.parametrize("failure", [False, RuntimeError("disconnected")])
def test_read_failure_releases_and_next_activation_can_recover(monkeypatch, failure):
    handles = [ControlledCapture(), ControlledCapture()]
    handles[0].read_error = failure
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handles.pop(0))
    failed = handles[0]
    block = Camera()
    try:
        assert block.execute({}) is None and failed.released and block.state == "error"
        assert block.execute({}).passed
    finally:
        block.close()


@pytest.mark.parametrize(
    "params",
    [
        {"device": True},
        {"width": 128.0},
        {"fps": "24"},
        {"fps": float("nan")},
        {"fourcc": "abc"},
        {"fourcc": "a\nbc"},
        {"additional_controls": {"trigger": 1}},
        {"additional_controls": {"settings": 1}},
        {"additional_controls": {"brightness": True}},
        {"additional_controls": {"gain": 1}},
        {"readback": "silent"},
        {"relative_tolerance": 2},
        {"maximum_frame_bytes": 0},
    ],
)
def test_invalid_configuration_never_opens_a_device(backend, params):
    block = Camera(parameters=params)
    with pytest.raises(ValidationError):
        block.validate_parameters()
    assert block.execute({}) is None
    assert not backend


def test_decoded_resource_guard_and_lean_diagnostic_contract(backend):
    block = Camera(parameters={"maximum_frame_pixels": 100})
    assert block.execute({}) is None and backend[0].released
    block.parameters["maximum_frame_pixels"] = 128 * 96
    block.run_profile = PROFILES["lean"]
    try:
        result = block.execute({})
        assert result.passed and result.outputs["diagnostic"] is NO_VALUE
        assert report(result)["decoded_frame"]["bytes"] == 128 * 96 * 3
    finally:
        block.close()


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "reference.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), 24, (128, 96)
    )
    assert writer.isOpened()
    images = [np.full((96, 128, 3), (20 + i * 20, 70, 180), np.uint8) for i in range(3)]
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()
    return path, images


def recorded(path, **p):
    return builtin_registry().create(
        "input.capture_file", parameters={"path": str(path), "backend": "ffmpeg", **p}
    )


def test_native_lossless_capture_reports_exact_frames_rate_format_and_stops(clip):
    path, expected = clip
    source = recorded(path)
    try:
        for i, image in enumerate(expected):
            result = source.execute({})
            assert result.passed, source.error
            np.testing.assert_array_equal(result.outputs["image"], image)
            info = report(result)
            assert info["source_kind"] == "recorded_file"
            assert info["readback"]["fps"]["value"] == 24
            assert info["fourcc_text"].lower() == "ffv1"
            assert info["decoded_frame"]["width"] == 128
            assert info["decoded_frame"]["dtype"] == "uint8"
            assert info["output_frame"]["color_format"] == "bgr"
            assert result.outputs["ticket"]["value"] == f"reference.avi#{i:06d}"
            json.dumps(info, allow_nan=False)
        assert source.exhausted and source._capture is None
        empty = source.execute({})
        assert all(empty.outputs[name] is NO_VALUE for name in ("image", "info", "diagnostic"))
    finally:
        source.close()


@pytest.mark.parametrize(
    "policy,passed", [("require_setter", False), ("require_readback", False), ("report", True)]
)
def test_native_file_backend_rejects_camera_settings_without_silent_success(clip, policy, passed):
    source = recorded(
        clip[0],
        width=640,
        fps=30,
        exposure_mode="backend_value",
        exposure_value=-6,
        negotiation_policy=policy,
        end_frame=1,
    )
    try:
        result = source.execute({})
        assert result is not None, source.error
        assert result.passed is passed
        info = report(result)
        assert info["decoded_frame"]["width"] == 128
        assert info["readback"]["fps"]["value"] == 24
        assert info["requested"]["fps"] == 30
        assert {r["state"] for r in info["properties"]} == {"setter_rejected"}
        assert (result.outputs["image"] is NO_VALUE) is (not passed)
        assert source.exhausted and source._capture is None
    finally:
        source.close()


def test_recorded_capture_loop_reset_and_host_conversion_do_not_change_raw_report(clip):
    source = recorded(
        clip[0],
        start_frame=1,
        end_frame=2,
        repeat="loop",
        color_format="hsv",
        pixel_depth="float32",
    )
    try:
        first, second = source.execute({}), source.execute({})
        assert report(first)["decoded_frame"]["color_format"] == "bgr"
        assert report(first)["decoded_frame"]["dtype"] == "uint8"
        assert report(first)["output_frame"]["color_format"] == "hsv"
        assert report(first)["output_frame"]["dtype"] == "float32"
        assert report(first)["generation"] != report(second)["generation"]
        source.reset_runtime()
        assert source.execute({}).outputs["info"]["frame_index"] == 1
    finally:
        source.close()


def test_legacy_camera_and_new_source_json_xml_load_are_device_free(tmp_path, backend):
    registry = builtin_registry()
    graph = Graph()
    camera = graph.add(registry.create("input.camera", parameters={"device": 2, "width": 640}))
    camera.parameters["additional_controls"]["brightness"] = 2
    graph.add(recorded("missing.avi"))
    path = tmp_path / "capture.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    assert loaded.blocks[camera.id].parameters == camera.parameters
    xml = tmp_path / "capture.xml"
    export_project_xml(loaded.to_dict(), xml)
    assert read_project_xml(xml)["blocks"] == loaded.to_dict()["blocks"]
    legacy = loaded.to_dict()
    legacy_camera = next(b for b in legacy["blocks"] if b["id"] == camera.id)
    legacy_camera["parameters"] = {
        name: legacy_camera["parameters"][name]
        for name in ("device", "backend", "width", "height", "fps", "buffer_size")
    }
    restored = Graph.from_dict(legacy, registry)
    assert restored.blocks[camera.id].parameters["width"] == 640
    assert restored.blocks[camera.id].parameters["negotiation_policy"] == "require_setter"
    assert not backend


@pytest.mark.parametrize("actual,expected", [(24.01, True), (23.5, False)])
def test_clamped_rate_uses_declared_tolerances(monkeypatch, actual, expected):
    handle = ControlledCapture()
    handle.unavailable[PROPERTIES["fps"]] = actual
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: handle)
    block = Camera(
        parameters={
            "fps": 24,
            "negotiation_policy": "require_readback",
            "absolute_tolerance": 0.02,
            "relative_tolerance": 0,
        }
    )
    try:
        result = block.execute({})
        assert result.passed is expected
        assert report(result)["properties"][0]["after_frame"]["value"] == actual
    finally:
        block.close()


def test_recorded_seek_fallback_reapplies_settings_to_the_reopened_handle(monkeypatch, tmp_path):
    handles = []
    path = tmp_path / "fixture.avi"
    path.write_bytes(b"controlled backend fixture")

    class SeekFallback(ControlledCapture):
        def grab(self):
            return True

        def retrieve(self):
            return True, self.image.copy()

    def open_fixture(*args):
        handle = SeekFallback()
        handle.rejected.add(cv2.CAP_PROP_POS_FRAMES)
        handles.append(handle)
        return handle

    monkeypatch.setattr(cv2, "VideoCapture", open_fixture)
    block = recorded(path, start_frame=1, end_frame=2, fps=24)
    try:
        result = block.execute({})
        assert result.passed and report(result)["frame_index"] == 1
        assert len(handles) == 2 and all(h.released for h in handles)
        assert all((cv2.CAP_PROP_FPS, 24) in h.writes for h in handles)
    finally:
        block.close()


def test_graph_save_as_preserves_recorded_capture_reference(clip, tmp_path):
    path, _ = clip
    graph = Graph()
    block = graph.add(recorded(path.name, path_base="project"))
    graph.bind_project_dir(tmp_path)
    directory = tmp_path / "new"
    directory.mkdir()
    saved = directory / "saved.json"
    graph.save(saved)
    loaded = Graph.load(saved, builtin_registry())
    try:
        result = loaded.blocks[block.id].execute({})
        assert result is not None and result.passed
        assert report(result)["decoded_frame"]["width"] == 128
    finally:
        loaded.close()


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), 2**40, 1.5])
def test_non_fourcc_values_are_not_decoded_as_a_codec(value):
    assert fourcc_text(value) is None


def test_toolbox_and_independent_defaults():
    registry = builtin_registry()
    for name in ("input.camera", "input.capture_file"):
        assert "Video I/O (videoio)" in block_entry(registry.create(name)).path
    one, two = Camera(), Camera()
    before = copy.deepcopy(two.parameters)
    one.parameters["additional_controls"]["brightness"] = 1
    assert two.parameters == before


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_examples_have_expected_native_outcomes(profile):
    root = Path(__file__).resolve().parents[1] / "examples/opencv tools/capture_negotiation"
    paths = sorted(root.glob("*.json"))
    paths = [path for path in paths if path.name != "truth.json"]
    assert len(paths) == 5
    for path in paths:
        graph = Graph.load(path, builtin_registry())
        try:
            graph.set_run_profile(profile)
            graph.run()
            source = graph.blocks["capture"]
            assert source.last_result is not None, source.error
            expected = "reject" not in path.stem
            assert source.last_result.passed is expected
        finally:
            graph.close()


def test_original_reference_hashes_and_regeneration_do_not_modify_examples(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples/opencv tools/capture_negotiation"
    before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    truth = json.loads((root / "truth.json").read_bytes())
    assert (
        hashlib.sha256((root / "assets/reference.avi").read_bytes()).hexdigest()
        == truth["asset_sha256"]
    )
    source = recorded(root / "assets/reference.avi")
    try:
        for digest in truth["frame_sha256"]:
            result = source.execute({})
            assert hashlib.sha256(result.outputs["image"].tobytes()).hexdigest() == digest
    finally:
        source.close()
    spec = importlib.util.spec_from_file_location("capture_example", root / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    generated = json.loads((tmp_path / "truth.json").read_bytes())
    assert generated["frame_sha256"] == truth["frame_sha256"]
    assert generated["cases"] == truth["cases"]
    for name in truth["cases"]:
        assert json.loads((tmp_path / (name + ".json")).read_bytes()) == json.loads(
            (root / (name + ".json")).read_bytes()
        )
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
