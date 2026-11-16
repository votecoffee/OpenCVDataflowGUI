"""Independent signal references and image-coordinate/calibration/invalid-support contracts."""

import copy

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.profile_resampling import read_profile
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.resample_image_profile"


def signal(values, step=1.0, start=0.0, closed=False, units="px"):
    values = np.asarray(values, dtype=object)
    if values.ndim == 1:
        values = values[:, None]
    count, channels = values.shape
    return {
        "model": "image_profile_v1",
        "closed": closed,
        "units": units,
        "path_length": step * (count if closed else count - 1),
        "distances": (start + np.arange(count) * step).tolist(),
        "points": np.column_stack((np.arange(count) * step, np.full(count, 4))).tolist(),
        "channels": [
            {"channel": f"C{i}", "values": values[:, i].tolist()} for i in range(channels)
        ],
    }


def run(value, **parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    result = block.execute({"profile": value})
    assert result is not None, block.error
    return result


def data(result):
    return np.array([c["values"] for c in result.outputs["profile"]["channels"]], float).T


@pytest.mark.parametrize("method", ["linear", "kaiser_sinc"])
def test_constant_identity_ramp_and_exact_unstretched_pitch(method):
    profile = signal(np.column_stack((np.full(129, 12.5), 2 + np.arange(129) * 3)))
    identical = run(profile, method=method, pitch=1)
    np.testing.assert_allclose(data(identical)[:, 1], 2 + np.arange(129) * 3, atol=1e-10)
    resized = run(profile, method=method, pitch=0.4)
    target = np.array(resized.outputs["profile"]["distances"])
    np.testing.assert_allclose(np.diff(target), 0.4, atol=1e-13)
    np.testing.assert_allclose(data(resized)[:, 0], 12.5, atol=1e-10)
    interior = (target > 15) & (target < 113)
    np.testing.assert_allclose(data(resized)[interior, 1], 2 + target[interior] * 3, atol=0.002)
    assert target[-1] == 128
    shortened = run(signal(np.ones(8)), method=method, pitch=2.5)
    assert shortened.outputs["profile"]["distances"] == [0, 2.5, 5]


@pytest.mark.parametrize("method", ["linear", "kaiser_sinc"])
def test_antialias_suppression_preserves_low_frequency_amplitude_and_peak_phase(method):
    x = np.arange(2049)
    low = np.cos(2 * np.pi * 0.02 * (x - 1024))
    high = np.sin(2 * np.pi * 0.30 * x)
    profile = signal(np.column_stack((low, high)))
    filtered = run(profile, pitch=4, method=method)
    aliased = run(profile, pitch=4, method=method, antialias=False)
    t = np.array(filtered.outputs["profile"]["distances"])
    keep = (t > 100) & (t < 1948)
    a, b = data(filtered), data(aliased)
    np.testing.assert_allclose(a[keep, 0], np.cos(2 * np.pi * 0.02 * (t[keep] - 1024)), atol=0.001)
    assert np.sqrt(np.mean(a[keep, 1] ** 2)) < 0.001
    assert np.sqrt(np.mean(b[keep, 1] ** 2)) > 0.65
    assert filtered.outputs["report"]["antialias_applied"]
    bell = run(signal(np.exp(-(((x - 1024) / 24) ** 2))), pitch=4, method=method)
    assert bell.outputs["profile"]["distances"][int(data(bell)[:, 0].argmax())] == 1024
    assert float(data(bell)[:, 0].max()) == pytest.approx(1, abs=0.002)


def test_fractional_reference_grid_uses_current_geometry_and_separate_reference_provenance():
    clock = TicketClock()
    current = {
        "ticket": clock.issue("current"),
        "spatial": frame([256, 32], "current"),
        "custom": {"batch": "A"},
    }
    other = {"ticket": clock.issue("reference"), "spatial": frame([256, 32], "reference")}
    profile = signal(np.arange(128) * 2)
    reference = signal(np.zeros(50), step=0.7, start=10.25)
    block = builtin_registry().create(TYPE, parameters={"grid": "reference", "method": "linear"})
    result = block.execute(
        {"profile": ValuePacket(profile, current), "reference": ValuePacket(reference, other)}
    )
    assert result is not None, block.error
    target = np.array(reference["distances"])
    np.testing.assert_allclose(data(result)[:, 0], target * 2)
    np.testing.assert_allclose(
        result.outputs["points"], np.column_stack((target, np.full(len(target), 4)))
    )
    assert result.payloads["profile"]["ticket"] == current["ticket"]
    assert result.outputs["report"]["reference_context"]["ticket"] == other["ticket"]
    assert result.payloads["points"]["custom"] == {"batch": "A"}
    assert result.drawings["Resampled sampling path"].source == "profile"
    assert block.last_inputs["profile"] == profile


def test_rectified_workplane_scale_converts_pitch_without_moving_source_points():
    clock = TicketClock()
    payload = {"spatial": frame([256, 32], "camera"), "ticket": clock.issue("camera")}
    scale = {
        "model": "rectified_workplane_v1",
        "spatial": payload["spatial"],
        "units": "mm",
        "units_per_pixel": 0.25,
        "origin": [0, 0],
    }
    block = builtin_registry().create(
        TYPE, parameters={"pitch": 0.5, "method": "linear", "distance_origin": 12}
    )
    result = block.execute(
        {
            "profile": ValuePacket(signal(np.arange(129)), payload),
            "scale": ValuePacket(scale, payload),
        }
    )
    assert result is not None, block.error
    assert result.outputs["profile"]["units"] == "mm"
    assert result.outputs["profile"]["distances"][:3] == [12, 12.5, 13]
    np.testing.assert_allclose(np.array(result.outputs["points"])[:3, 0], [0, 2, 4])
    assert result.outputs["report"]["source_pitch"] == 0.25
    assert result.payloads["profile"]["spatial"] == payload["spatial"]
    wrong = {**payload, "ticket": clock.issue("other")}
    assert (
        block.execute(
            {
                "profile": ValuePacket(signal(np.ones(129)), payload),
                "scale": ValuePacket(scale, wrong),
            }
        )
        is None
    )
    assert block.error_kind == "invalid"


@pytest.mark.parametrize("method", ["linear", "kaiser_sinc"])
def test_strict_null_gaps_are_channel_specific_and_not_bridged(method):
    samples = np.ones((129, 2), object)
    samples[64, 0] = None
    result = run(signal(samples), pitch=2, method=method)
    values = data(result)
    t = np.array(result.outputs["profile"]["distances"])
    assert np.isnan(values[t == 64, 0]).all()
    assert np.isnan(values[(t > 60) & (t < 68), 0]).all()
    assert np.isfinite(values[:, 1]).all()
    assert np.isfinite(values[t < 35, 0]).all()
    identity = run(signal(samples), pitch=1, method=method)
    assert np.isnan(data(identity)[64, 0]) and np.isfinite(data(identity)[63, 0])


def test_no_extrapolation_outer_boundaries_and_empty_support_are_explicit():
    result = run(signal(np.ones(65)), grid="explicit_extent", start=-4, end=70, pitch=2)
    t = np.array(result.outputs["profile"]["distances"])
    values = data(result)[:, 0]
    assert np.isnan(values[(t < 0) | (t > 64)]).all()
    assert all(
        p is None for p, v in zip(result.outputs["points"], t, strict=True) if v < 0 or v > 64
    )
    strict = run(signal(np.ones(129)), pitch=4, boundary="invalid")
    assert np.isnan(data(strict)[0, 0]) and np.isfinite(data(strict)[16, 0])
    assert strict.outputs["report"]["edge_extension_count"] == 0
    assert result.outputs["report"]["edge_extension_count"] > 0
    assert result.outputs["report"]["boundary_extended"][2]
    none = run(signal([None] * 129), pitch=4)
    assert none.passed is False and [r["code"] for r in none.fail_reasons] == [
        "profile_empty_support"
    ]
    assert none.outputs["selected_count"] == 0


def test_closed_wrap_and_period_grid_preserve_sine_phase_and_one_origin():
    x = np.arange(256)
    values = np.sin(2 * np.pi * x / 64)
    result = run(signal(values, closed=True), pitch=2)
    target = np.array(result.outputs["profile"]["distances"])
    assert len(target) == 128 and target[-1] == 254
    np.testing.assert_allclose(data(result)[:, 0], np.sin(2 * np.pi * target / 64), atol=0.0003)
    assert result.outputs["report"]["boundary"] == "wrap"
    read_profile(result.outputs["profile"])
    block = builtin_registry().create(TYPE, parameters={"pitch": 3})
    assert block.execute({"profile": signal(values, closed=True)}) is None
    assert block.error_kind == "invalid"


@pytest.mark.parametrize(
    "change",
    [
        {"distances": [0, 1, 3]},
        {"points": [[0, 0]]},
        {"closed": "no"},
        {"channels": [{"channel": "gray", "values": [1, float("nan"), 3]}]},
        {"channels": [{"channel": "gray", "values": [1, True, 3]}]},
        {"units": "seconds"},
        {"path_length": 0},
    ],
)
def test_invalid_profile_shapes_values_units_emit_no_event(change):
    value = signal([1, 2, 3])
    value.update(change)
    block = builtin_registry().create(TYPE)
    assert block.execute({"profile": value}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())


def test_resource_guards_precede_kernel_work_and_xml_controls_preserve_configuration(
    app, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QComboBox

    import opencv_dataflow.profile_resampling as module
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.core import Graph
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml

    block = builtin_registry().create(TYPE, parameters={"maximum_work": 1})
    monkeypatch.setattr(
        module, "sample_weights", lambda *a: pytest.fail("Work guard ran after kernel work")
    )
    assert block.execute({"profile": signal(np.ones(129))}) is None
    assert block.error_kind == "invalid"
    graph = Graph()
    block = graph.add(builtin_registry().create(TYPE))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "profile.xml"
    export_project_xml(snapshot, path)
    loaded = Graph.from_dict(read_project_xml(path), builtin_registry())
    assert loaded.to_dict() == snapshot
    loaded.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert all(
            isinstance(window.inspector.fields[n], QComboBox)
            for n in ("grid", "method", "boundary")
        )
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
