"""Native timing alignment, observation isolation and atomic inference contracts."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import dnn_blocks, dnn_profiling
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.dnn_profiling import (
    ProfileBatchDNN,
    ProfileDNN,
    ProfileMultiDNN,
    ProfileUnavailable,
    native_profile,
)
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.model_contracts import configuration, export_contract
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock, ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/dnn_profiling"


def block(**settings):
    return ProfileDNN(
        parameters={
            "model_path": str(BASE / "assets/single.onnx"),
            "width": 16,
            "height": 16,
            "input_name": "image",
            "engine": "new",
            "mean": [127.5] * 3,
            "scale": 1 / 127.5,
            **settings,
        }
    )


IMAGE = np.full((24, 32, 3), [20, 140, 240], np.uint8)


def execute(b, inputs):
    result = b.execute(inputs)
    assert result is not None, b.error
    return result


class Native:
    def __init__(self, base=1):
        self.names = ["second", "first"]
        self.ids = {"first": base, "second": base + 1}
        self.measurement = (100, np.float64([100, 0]))

    def getLayerNames(self):
        return self.names

    def getLayerId(self, name):
        return self.ids[name]

    def getPerfProfile(self):
        return self.measurement


@pytest.mark.parametrize("base", [0, 1])
def test_native_ids_align_zero_or_one_based_slots_and_zero_ticks_are_explicit(monkeypatch, base):
    monkeypatch.setattr(cv2, "getTickFrequency", lambda: 1000)
    result = native_profile(Native(base), 2)
    assert result["native_total_ms"] == 100
    assert result["native_layer_id_base"] == base
    assert [(r["name"], r["milliseconds"], r["timing_state"]) for r in result["layers"]] == [
        ("second", 0, "zero_ticks"),
        ("first", 100, "measured"),
    ]


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "duplicate_names",
        "long_name",
        "budget",
        "three_outputs",
        "negative",
        "nonfinite",
        "wrong_length",
        "duplicate_ids",
        "sparse_ids",
        "all_zero",
        "negative_total",
        "bool_total",
        "frequency",
    ],
)
def test_missing_invalid_or_ambiguous_native_timings_are_not_reported_as_zero(monkeypatch, problem):
    net = Native()
    budget = 10
    if problem == "empty":
        net.names = []
    elif problem == "duplicate_names":
        net.names = ["first", "first"]
    elif problem == "long_name":
        net.names = ["x" * 513, "first"]
    elif problem == "budget":
        budget = 1
    elif problem == "three_outputs":
        net.measurement = (["first"], ["1.0"], ["1"])
    elif problem == "negative":
        net.measurement = (100, [-1, 101])
    elif problem == "nonfinite":
        net.measurement = (100, [float("nan"), 0])
    elif problem == "wrong_length":
        net.measurement = (100, [100])
    elif problem == "duplicate_ids":
        net.ids = {"first": 1, "second": 1}
    elif problem == "sparse_ids":
        net.ids = {"first": 1, "second": 4}
    elif problem == "all_zero":
        net.measurement = (0, [0, 0])
    elif problem == "negative_total":
        net.measurement = (-1, [100, 0])
    elif problem == "bool_total":
        net.measurement = (True, [100, 0])
    else:
        monkeypatch.setattr(cv2, "getTickFrequency", lambda: 0)
    with pytest.raises(ProfileUnavailable):
        native_profile(net, budget)


@pytest.mark.parametrize("engine", ["classic", "new", "auto"])
def test_real_native_profile_matches_the_same_forward_ticks_names_and_identity(engine):
    b = block(engine=engine)
    context = {"ticket": TicketClock().issue("image"), "spatial": frame([32, 24], "camera")}
    try:
        result = execute(b, {"image": ValuePacket(IMAGE, context)})
        data = result.outputs["profile"]
        call = data["calls"][0]
        total, ticks = b._net.getPerfProfile()
        names = list(b._net.getLayerNames())
        assert data["status"] == "available" and data["native_call_count"] == 1
        assert call["total_ticks"] == total and [r["name"] for r in call["layers"]] == names
        for row in call["layers"]:
            assert row["ticks"] == ticks[row["timing_index"]]
            assert row["milliseconds"] == pytest.approx(
                row["ticks"] / cv2.getTickFrequency() * 1000
            )
        assert (
            call["artifact"]["sha256"]
            == hashlib.sha256((BASE / "assets/single.onnx").read_bytes()).hexdigest()
        )
        assert (
            call["selected_execution"]["engine"] == engine
            and call["selected_execution"]["target"] == "cpu"
        )
        assert call["native_placement_verified"] is False
        assert ticket_key(result.payloads["profile"]["ticket"]) == ticket_key(context["ticket"])
        np.testing.assert_array_equal(result.outputs["image"], IMAGE)
    finally:
        b.close()


def test_profile_disabled_and_ordinary_inference_do_not_call_timing_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled observation called a timing API")

    monkeypatch.setattr(dnn_profiling, "native_profile", forbidden)
    monkeypatch.setattr(dnn_profiling, "perf_counter_ns", forbidden)
    b = block(profile_mode="off", maximum_profile_layers=1)
    plain = DNNInference(
        parameters={k: v for k, v in b.parameters.items() if k in DNNInference.defaults}
    )
    try:
        observed = execute(b, {"image": IMAGE})
        ordinary = execute(plain, {"image": IMAGE})
        assert observed.outputs["profile"]["status"] == "disabled"
        assert (
            observed.outputs["profile"]["calls"] == []
            and observed.outputs["profile"]["native_total_ms"] is None
        )
        assert observed.outputs["tensors"] == ordinary.outputs["tensors"] and not observed.graphics
    finally:
        b.close()
        plain.close()


def test_exactly_one_forward_and_profile_is_fresh_after_cache_reuse(monkeypatch):
    original = dnn_blocks.load_onnx_snapshot
    counts = {"forward": 0, "profile": 0}

    class Proxy:
        def __init__(self, net):
            self.net = net

        def __getattr__(self, key):
            return getattr(self.net, key)

        def forward(self, *args):
            counts["forward"] += 1
            return self.net.forward(*args)

        def getPerfProfile(self):
            counts["profile"] += 1
            return self.net.getPerfProfile()

    def load(*args):
        net, info = original(*args)
        return Proxy(net), info

    monkeypatch.setattr(dnn_blocks, "load_onnx_snapshot", load)
    b = block()
    try:
        first = execute(b, {"image": IMAGE})
        handle = b._net
        second = execute(b, {"image": 255 - IMAGE})
        assert counts == {"forward": 2, "profile": 2} and b._net is handle
        assert second.outputs["profile"]["calls"][0]["cache_reused"]
        assert second.outputs["profile"]["calls"][0]["index"] == 0
        assert first.outputs["tensors"] != second.outputs["tensors"]
        b.parameters["profile_mode"] = "off"
        execute(b, {"image": IMAGE})
        assert counts == {"forward": 3, "profile": 2}
    finally:
        b.close()


def test_optional_unavailable_required_error_and_recovery_keep_distinct_behavior(monkeypatch):
    b = block()
    original = dnn_profiling.native_profile

    def missing(*a):
        raise ProfileUnavailable("unsupported test backend")

    monkeypatch.setattr(dnn_profiling, "native_profile", missing)
    try:
        r = execute(b, {"image": IMAGE})
        assert r.passed and r.outputs["profile"]["status"] == "unavailable"
        assert r.outputs["profile"]["native_total_ms"] is None and not r.graphics
        assert selected_tensor(r.outputs["tensors"]).shape == (1, 3)
        b.parameters["profile_mode"] = "required"
        assert b.execute({"image": IMAGE}) is None and not b.current_emissions
        assert b._net is None and not b._profile_calls
        monkeypatch.setattr(dnn_profiling, "native_profile", original)
        assert execute(b, {"image": IMAGE}).outputs["profile"]["status"] == "available"
    finally:
        b.close()


def group():
    return FlowGroup(
        tuple(ValuePacket(IMAGE + i, {"ticket": TicketClock().issue(str(i))}) for i in range(3)),
        ("a", "b", "c"),
    )


def test_batch_profiles_are_per_native_call_with_atomic_budget_and_row_ownership():
    b = ProfileBatchDNN(
        parameters={
            "model_path": str(BASE / "assets/single.onnx"),
            "width": 16,
            "height": 16,
            "input_name": "image",
            "batch_size": 2,
            "engine": "new",
            "maximum_profile_layers": 5,
        }
    )
    try:
        r = execute(b, {"images": group()})
        p = r.outputs["profile"]
        assert p["status"] == "partial" and p["native_total_ms"] is None
        assert p["measured_native_ms_sum"] > 0 and p["native_call_count"] == 2
        assert [c["batch"]["real_rows"] for c in p["calls"]] == [2, 1]
        assert len(r.outputs["results"].items) == 3 and p["layer_records"] == 5
        b.parameters["profile_mode"] = "required"
        assert b.execute({"images": group()}) is None and not b.current_emissions
        assert b._net is None and not b._profile_calls
        b.parameters.update(profile_mode="optional", maximum_profile_layers=20)
        assert execute(b, {"images": group()}).outputs["profile"]["status"] == "available"
    finally:
        b.close()


def test_observation_settings_do_not_change_sealed_algorithm_contract(tmp_path):
    b = block()
    export_contract(b, tmp_path / "bundle")
    b.project_dir = tmp_path
    b.parameters.update(model_path="", contract_path="bundle/contract.json")
    try:
        before = configuration(b)
        b.parameters.update(profile_mode="off", maximum_profile_layers=1, profile_plot_layers=0)
        assert configuration(b) == before
        assert execute(b, {"image": IMAGE}).outputs["profile"]["status"] == "disabled"
    finally:
        b.close()


@pytest.mark.parametrize("cls", [ProfileDNN, ProfileBatchDNN, ProfileMultiDNN])
@pytest.mark.parametrize(
    "settings",
    [
        {"profile_mode": "guess"},
        {"maximum_profile_layers": 0},
        {"profile_plot_layers": 51},
        {"profile_plot_layers": True},
    ],
)
def test_profile_control_validation_without_native_work(cls, settings):
    with pytest.raises(ValidationError):
        cls(parameters=settings).validate_parameters()


def test_receipts_graphics_lean_and_reset_are_independent_observations():
    b = block(receipt_mode="off")
    try:
        r = execute(b, {"image": IMAGE})
        assert r.outputs["receipt"] is NO_VALUE and r.outputs["profile"]["status"] == "available"
        assert r.graphics
        b.run_profile = profile("lean")
        lean = execute(b, {"image": IMAGE})
        assert not lean.graphics and lean.outputs["tensors"] == r.outputs["tensors"]
        assert lean.outputs["profile"]["status"] == "available"
        b.reset_runtime()
        assert b._net is None and not b._profile_calls
    finally:
        b.close()


def test_explicit_cpu_fallback_profile_retains_requested_and_selected_targets(monkeypatch):
    # Exercise selection without probing an accelerator or requiring one to exist.
    monkeypatch.setattr(cv2.dnn, "getAvailableTargets", lambda backend: [])
    b = block(
        engine="classic", backend="cuda", target="cuda", fallback_policy="cpu_if_unavailable"
    )
    try:
        call = execute(b, {"image": IMAGE}).outputs["profile"]["calls"][0]
        assert call["status"] == "available"
        assert call["requested_execution"]["target"] == "cuda"
        assert call["selected_execution"]["target"] == "cpu"
        assert call["selected_execution"]["backend"] == "opencv"
        assert call["native_placement_verified"] is False
    finally:
        b.close()
