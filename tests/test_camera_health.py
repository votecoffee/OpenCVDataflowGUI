"""Camera health against known image motion, parts, illumination and defocus."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.camera_health import (
    CameraHealth,
    load_reference,
    save_reference,
    train_reference,
)
from opencv_dataflow.camera_health.model import full_region
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def fixture():
    rng = np.random.default_rng(2026)
    image = cv2.GaussianBlur(rng.uniform(45, 205, (240, 320)).astype(np.float32), (3, 3), 0)
    return np.rint(image).astype(np.uint8)


def region(name, bounds):
    return {"name": name, "region": {**full_region([320, 240]), "rectangle": bounds}}


def zones():
    return {
        "lighting": [
            region("Left light", [20, 20, 80, 60]),
            region("Right light", [210, 20, 80, 60]),
        ],
        "focus": [
            region("Left texture", [25, 85, 85, 45]),
            region("Right texture", [210, 85, 85, 45]),
        ],
        "fiducials": [region("Datum A", [42, 42, 27, 29]), region("Datum B", [242, 42, 29, 27])],
    }


def make_tool(tmp_path, captures=None, regions=None, **parameters):
    captures = [fixture()] if captures is None else captures
    model = train_reference(captures, zones() if regions is None else regions, parameters)
    path = tmp_path / "reference.npz"
    digest = save_reference(model, path)
    tool = CameraHealth(
        parameters={"reference_path": str(path), "reference_sha256": digest, **parameters}
    )
    return tool, model


def execute(tool, image, **extra):
    result = tool.execute({"image": image, **extra})
    assert result is not None, tool.error
    return result, result.outputs["report"]


def test_frozen_baseline_healthy_and_multi_region_history(tmp_path):
    image = fixture()
    history = [image.copy() for _ in range(3)]
    history[0][175:205, 20:65] = 20
    history[1][175:205, 120:165] = 20
    history[2][175:205, 230:275] = 20
    tool, model = make_tool(tmp_path, history)
    current = image.copy()
    current[150:180, 170:215] = 230
    result, report = execute(tool, current)
    assert result.outputs["healthy"] and result.passed and report["reference_is_frozen"]
    assert report["capture_count"] == 3 and report["foreground_pixels"] > 500
    assert np.count_nonzero(model.stable == 0) >= 3 * 30 * 45
    assert len(model.info["baselines"]["lighting"]) == 2
    assert len(model.info["baselines"]["focus"][0]["values"]) == 3
    assert report["position"]["displacement_px"] < 0.15
    np.testing.assert_array_equal(result.outputs["image"], current)
    assert result.drawings


@pytest.mark.parametrize("shift,expected", [(0, "healthy"), (2, "warning"), (5, "alarm")])
def test_background_motion_with_moving_foreground(tmp_path, shift, expected):
    image = fixture()
    tool, _ = make_tool(tmp_path)
    current = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, -shift * 0.2]]), (320, 240))
    current[155:195, 120:180] = 15
    result, report = execute(tool, current)
    assert report["state"] == expected
    assert report["position"]["displacement_px"] == pytest.approx(
        np.hypot(shift, shift * 0.2), abs=0.2
    )
    assert report["foreground_pixels"] > 1200
    assert result.outputs["alarm"] == (expected == "alarm")
    assert result.outputs["warning"] == (expected == "warning")
    assert result.passed == (expected != "alarm")


@pytest.mark.parametrize("gain,expected", [(1.0, "healthy"), (1.15, "warning"), (1.30, "alarm")])
def test_lighting_and_focus_are_separate_and_gain_normalized(tmp_path, gain, expected):
    image = fixture()
    tool, _ = make_tool(tmp_path, check_position=False, auto_foreground=False)
    current = np.rint(image.astype(float) * gain).clip(0, 255).astype(np.uint8)
    result, report = execute(tool, current)
    lights = [row for row in report["checks"] if row["check"] == "lighting"]
    assert len(lights) == 2 and all(row["state"] == expected for row in lights)
    assert lights[0]["value"] == pytest.approx((gain - 1) * 100, abs=0.2)
    assert all(row["state"] == "healthy" for row in report["checks"] if row["check"] == "focus")
    assert result.outputs["state"] == expected


@pytest.mark.parametrize("method", ["tenengrad", "laplacian_variance"])
def test_defocus_alarms_with_two_focus_regions(tmp_path, method):
    tool, _ = make_tool(tmp_path, focus_method=method, check_position=False, auto_foreground=False)
    result, report = execute(tool, cv2.GaussianBlur(fixture(), (9, 9), 2))
    focus = [row for row in report["checks"] if row["check"] == "focus"]
    assert len(focus) == 2 and all(row["state"] == "alarm" for row in focus)
    assert all(row["value"] > 40 for row in focus)
    assert not result.passed


@pytest.mark.parametrize("count", [1, 2])
def test_trained_fiducials_measure_translation_and_do_not_invent_single_point_rotation(
    tmp_path, count
):
    regions = zones()
    regions["fiducials"] = regions["fiducials"][:count]
    tool, _ = make_tool(tmp_path, regions=regions, position_method="fiducials")
    current = cv2.warpAffine(fixture(), np.float32([[1, 0, 4], [0, 1, -2]]), (320, 240))
    _, report = execute(tool, current)
    assert report["position"]["displacement_px"] == pytest.approx(np.sqrt(20), abs=0.2)
    assert report["state"] == "alarm"
    if count == 1:
        assert report["position"]["rotation_deg"] is None
        assert report["position"]["scale_percent"] is None
    else:
        assert abs(report["position"]["rotation_deg"]) < 0.1


def test_missing_fiducial_and_too_little_visible_background_are_alarms(tmp_path):
    tool, _ = make_tool(tmp_path, position_method="fiducials")
    current = fixture()
    current[10:110, 10:110] = 100
    result, report = execute(tool, current)
    assert not result.passed and report["position"]["state"] == "unavailable"
    mask = np.full(current.shape, 255, np.uint8)
    result, report = execute(tool, fixture(), foreground_mask=mask)
    assert report["state"] == "alarm"
    assert all(row["state"] == "alarm" for row in report["checks"])


@pytest.mark.parametrize(
    "dtype,encoding", [(np.uint8, "gray"), (np.uint16, "rgb"), (np.float32, "rgba")]
)
def test_formats_tickets_recipes_pyramid_and_reference_persistence(tmp_path, dtype, encoding):
    image = fixture().astype(dtype)
    if dtype == np.uint16:
        image *= 257
    if dtype == np.float32:
        image /= 255
    if encoding != "gray":
        image = np.dstack([image] * (4 if encoding == "rgba" else 3))
    payload = {
        "spatial": frame([320, 240], "camera-grid"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "job", "revision": 1, "digest": "a" * 64},
        "pyramid": {
            "level": 1,
            "largest_size": [640, 480],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    packet = ValuePacket(ImageArray(image, encoding), payload)
    tool, model = make_tool(tmp_path, [packet])
    original = copy.deepcopy(model.info)
    result, _ = execute(tool, packet)
    assert result.passed
    for name in result.outputs:
        assert result.payloads[name]["ticket"] == payload["ticket"]
        assert result.payloads[name]["recipe_ref"] == payload["recipe_ref"]
        assert result.payloads[name]["spatial"] == payload["spatial"]
    loaded = load_reference(
        tool.parameters["reference_path"], tool.parameters["reference_sha256"], include_history=True
    )
    np.testing.assert_array_equal(loaded.history, model.history)
    assert model.info == original
    tool.parameters["focus_aperture"] = "5"
    assert tool.execute({"image": packet}) is None and tool.error_kind == "invalid"


def test_hash_tampering_archive_bounds_and_training_rejections(tmp_path, monkeypatch):
    tool, model = make_tool(tmp_path)
    path = Path(tool.parameters["reference_path"])
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert tool.execute({"image": fixture()}) is None and "SHA-256" in tool.error
    path.write_bytes(data)
    with pytest.raises(ValueError):
        train_reference([])
    with pytest.raises(ValueError):
        train_reference([fixture()], parameters={"float_min": 1, "float_max": 0})
    with pytest.raises(ValueError):
        train_reference([fixture(), fixture()[:100]])
    other = copy.deepcopy(model.info)
    assert other["capture_count"] == 1
    import opencv_dataflow.camera_health.model as module

    monkeypatch.setattr(module, "MAX_SAMPLES", 10)
    with pytest.raises(ValueError):
        train_reference([fixture()])


@pytest.mark.parametrize(
    "parameters",
    [
        {"lighting_warning": 30, "lighting_alarm": 20},
        {"focus_warning": 40, "focus_alarm": 40},
        {"tracking_window": 4},
        {"minimum_tracks": 1000},
        {"check_focus": 1},
        {"check_position": False, "check_lighting": False, "check_focus": False},
    ],
)
def test_parameter_validation_before_reference_loading(parameters):
    tool = CameraHealth(parameters=parameters)
    assert tool.execute({"image": fixture()}) is None and tool.error_kind == "invalid"


def test_graph_save_as_preserves_sealed_reference_and_xml(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    tool, _ = make_tool(first)
    tool.parameters["reference_path"] = "reference.npz"
    tool.project_dir = first
    graph = Graph()
    graph.add(tool)
    graph.save(first / "project.json")
    loaded = Graph.load(first / "project.json", builtin_registry())
    second = tmp_path / "second"
    second.mkdir()
    loaded.save(second / "project.json")
    assert (second / "reference.npz").read_bytes() == (first / "reference.npz").read_bytes()
    loaded.save(second / "project.xml")
    restored = Graph.load(second / "project.xml", builtin_registry())
    block = next(iter(restored.blocks.values()))
    execute(block, fixture())
    assert block.parameters["reference_sha256"] == tool.parameters["reference_sha256"]
    graph.close()
    loaded.close()
    restored.close()


@pytest.mark.parametrize("mode", ["background", "fiducials"])
@pytest.mark.parametrize(
    "angle,scale,expected",
    [(0.5, 1, "warning"), (1.5, 1, "alarm"), (0, 1.01, "warning"), (0, 1.03, "alarm")],
)
def test_rotation_and_scale_against_known_similarity(tmp_path, mode, angle, scale, expected):
    tool, _ = make_tool(tmp_path, position_method=mode, check_lighting=False, check_focus=False)
    matrix = cv2.getRotationMatrix2D((159.5, 119.5), angle, scale)
    current = cv2.warpAffine(fixture(), matrix, (320, 240))
    result, report = execute(tool, current)
    assert report["state"] == expected
    assert report["position"]["state"] == "measured"
    assert report["position"]["rotation_deg"] == pytest.approx(-angle, abs=0.05)
    assert report["position"]["scale_percent"] == pytest.approx((scale - 1) * 100, abs=0.05)
    assert result.outputs["transform"]["model"] == "image_affine_v1"


def test_input_grid_and_mask_mismatch_do_not_publish(tmp_path):
    image = fixture()
    first = frame([640, 480], "first")
    first["image_size"] = [320, 240]
    first["to_root"] = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    packet = ValuePacket(image, {"spatial": first})
    tool, _ = make_tool(tmp_path, [packet])
    second = copy.deepcopy(first)
    second["to_root"][0][2] = 10
    assert tool.execute({"image": ValuePacket(image, {"spatial": second})}) is None
    assert "crop/scale" in tool.error and not tool.current_emissions
    assert tool.execute({"image": packet, "foreground_mask": np.ones(image.shape, float)}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions
    wrong_mask = ValuePacket(
        np.zeros(image.shape, np.uint8), {"spatial": frame([320, 240], "other")}
    )
    assert tool.execute({"image": packet, "foreground_mask": wrong_mask}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions


def test_color_only_foreground_is_rejected_and_graphics_off_preserves_report(tmp_path):
    image = np.dstack([fixture()] * 3)
    tool, _ = make_tool(tmp_path, [image])
    current = image.copy()
    # BGR (80,130,52) has virtually the same gray as (100,100,100).
    image[150:190, 140:180] = 100
    current[150:190, 140:180] = [80, 130, 52]
    # Retrain this separate baseline, with the exact gray patch.
    model = train_reference([image], zones())
    path = tmp_path / "color-reference.npz"
    tool.parameters.update(reference_path=str(path), reference_sha256=save_reference(model, path))
    result, report = execute(tool, current)
    assert result.passed and report["foreground_pixels"] >= 1500
    assert result.outputs["foreground_mask"][160:180, 150:170].all()
    assert result.drawings
    tool.graphics_mode = "off"
    lean, other = execute(tool, current)
    assert not lean.drawings and other == report


def test_flat_reference_missing_fiducial_and_native_errors_are_contained(tmp_path, monkeypatch):
    flat = np.full((240, 320), 100, np.uint8)
    tool, _ = make_tool(tmp_path, [flat])
    result, report = execute(tool, flat)
    assert not result.passed and report["position"]["state"] == "unavailable"
    assert all(row["state"] == "alarm" for row in report["checks"] if row["check"] == "focus")
    tool.parameters.update(position_method="fiducials")
    _, report = execute(tool, flat)
    assert all(row["reason"] == "flat_template" for row in report["position"]["fiducials"])

    def broken(*args, **kwargs):
        raise cv2.error("Controlled camera-health tracking failure")

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", broken)
    tool.parameters.update(position_method="background")
    assert tool.execute({"image": flat}) is None
    assert "Controlled camera-health" in tool.error and not tool.current_emissions


def test_limits_are_inclusive_and_alarm_diagnostics_have_error_severity(tmp_path):
    from opencv_dataflow.camera_health.analysis import level

    assert level(9.999, 10, 20) == "healthy"
    assert level(10, 10, 20) == "warning"
    assert level(20, 10, 20) == "alarm"
    tool, _ = make_tool(tmp_path, check_position=False, check_focus=False, auto_foreground=False)
    _, report = execute(tool, np.rint(fixture() * 0.7).astype(np.uint8))
    assert report["state"] == "alarm"
    events = tool.logger.snapshot()["events"]
    assert any(e["severity"] == "ERROR" and "Camera health alarm" in e["message"] for e in events)


def test_built_in_package_toolbox_placement():
    from opencv_dataflow.toolbox import block_entry

    assert block_entry(CameraHealth()).path == ("Custom", "Inspection", "Camera monitoring")


def test_circle_polygon_regions_and_immutable_revision(tmp_path):
    regions = zones()
    regions["lighting"][0]["region"] = {
        **full_region([320, 240]),
        "shape": "circle",
        "center": [60, 50],
        "radius": 20,
    }
    regions["focus"][0]["region"] = {
        **full_region([320, 240]),
        "shape": "polygon",
        "vertices": [[25, 85], [110, 85], [105, 130], [30, 130]],
    }
    tool, _ = make_tool(tmp_path, regions=regions)
    result, report = execute(tool, fixture())
    assert result.passed and report["state"] == "healthy"
    original = Path(tool.parameters["reference_path"]).read_bytes()
    other = train_reference([255 - fixture()], regions)
    with pytest.raises(ValueError, match="different trained baseline"):
        save_reference(other, tool.parameters["reference_path"])
    assert Path(tool.parameters["reference_path"]).read_bytes() == original
