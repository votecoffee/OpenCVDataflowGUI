"""Independent batch references for masked centered temporal moments and lifecycle."""

from decimal import Decimal, localcontext

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pixel_moments import PixelMoments
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock, ticket_key


def run(block, image, **inputs):
    result = block.execute({"image": image, **inputs})
    assert result is not None, block.error
    return result


def reference(x, y, method, normalization, alpha=0.25):
    weights = (
        np.ones(len(x))
        if method == "cumulative"
        else alpha * (1 - alpha) ** np.arange(len(x) - 1, -1, -1)
    )
    total = weights.sum()
    denominator = total if normalization == "population" else total - (weights**2).sum() / total
    # A separate, two-pass batch calculation with shifts, not the online recurrence.
    x, y = np.asarray(x), np.asarray(y)
    sx, sy = x - x[0], y - y[0]
    mx, my = np.average(sx, weights=weights, axis=0), np.average(sy, weights=weights, axis=0)
    dx, dy = sx - mx, sy - my
    w = weights.reshape((-1,) + (1,) * (x.ndim - 1))
    vx, vy = (np.sum(w * d * d, axis=0) / denominator for d in (dx, dy))
    return x[0] + mx, y[0] + my, vx, vy, np.sum(w * dx * dy, axis=0) / denominator


@pytest.mark.parametrize("method", ["cumulative", "exponential"])
@pytest.mark.parametrize("normalization", ["sample", "population"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_centered_results_match_independent_batch_reference(
    method, normalization, profile, monkeypatch
):
    rng = np.random.default_rng(160)
    x = rng.normal(200, 3, (8, 5, 7))
    y = 2 * x + rng.normal(0, 1, x.shape)
    observed = set()
    for name in ("accumulate", "accumulateSquare", "accumulateProduct", "accumulateWeighted"):
        native = getattr(cv2, name)

        def wrapped(*args, _native=native, _name=name, **kwargs):
            observed.add(_name)
            assert args[0].dtype == np.float64
            return _native(*args, **kwargs)

        monkeypatch.setattr(cv2, name, wrapped)
    block = PixelMoments(
        parameters={"method": method, "normalization": normalization, "alpha": 0.25}
    )
    block.run_profile = PROFILES[profile]
    for index in range(8):
        before = x[index].copy()
        result = run(block, x[index], other=y[index])
        np.testing.assert_array_equal(x[index], before)
        assert result.outputs["ready"] is (index > 0)
    expected = reference(x, y, method, normalization)
    for name, value in zip(
        ("mean_x", "mean_y", "variance_x", "variance_y", "covariance"), expected, strict=True
    ):
        np.testing.assert_allclose(result.outputs[name], value, rtol=1e-12, atol=2e-12)
    corr = expected[4] / np.sqrt(expected[2] * expected[3])
    np.testing.assert_allclose(result.outputs["correlation"], corr, atol=1e-12)
    assert np.all(result.outputs["count"] == 8)
    assert bool(result.graphics) is (profile == "debug")
    assert {"accumulate", "accumulateSquare", "accumulateProduct"} <= observed
    assert ("accumulateWeighted" in observed) is (method == "exponential")
    block.close()


@pytest.mark.parametrize("method", ["cumulative", "exponential"])
def test_changing_masks_have_per_pixel_counts_and_weights_not_zero_padding(method):
    block = PixelMoments(parameters={"method": method, "alpha": 0.25, "minimum_coverage": 0.5})
    observations = [[[] for _ in range(4)] for _ in range(3)]
    for index in range(6):
        image = np.full((3, 4), 10 + index, np.float64)
        mask = np.full((3, 4), 255, np.uint8)
        mask[0, 0] = 0
        mask[1, 1] = 255 if index % 2 else 0
        run(block, image, mask=mask)
        for y, x in np.argwhere(mask != 0):
            observations[y][x].append(10 + index)
    result = run(block, np.zeros((3, 4)), learn=False)
    assert result.outputs["valid"][0, 0] == 0 and result.outputs["count"][0, 0] == 0
    for y, x in np.argwhere(result.outputs["valid"] != 0):
        values = np.array(observations[y][x], np.float64)[:, None, None]
        expected = reference(values, values, method, "sample")
        assert result.outputs["mean_x"][y, x] == pytest.approx(expected[0].item())
        assert result.outputs["variance_x"][y, x] == pytest.approx(expected[2].item())
        assert result.outputs["count"][y, x] == len(values)
    assert result.outputs["status"]["phase"] == "frozen"
    empty = run(block, np.zeros((3, 4)), mask=np.zeros((3, 4), np.uint8))
    assert not empty.outputs["ready"] and empty.outputs["mean_x"] is NO_VALUE
    block.close()


def test_float64_high_baseline_retains_small_variance_against_decimal_oracle():
    x = [1e12 + value for value in (0.001, 0.002, -0.001, 0.003, -0.002, 0.004)]
    y = [1e12 + value for value in (-0.002, 0.003, -0.001, 0.004, -0.003, 0.005)]
    block = PixelMoments()
    for a, b in zip(x, y, strict=True):
        result = run(block, np.full((2, 3), a), other=np.full((2, 3), b))
    with localcontext() as ctx:
        ctx.prec = 60
        a, b = [list(map(Decimal.from_float, values)) for values in (x, y)]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        vx = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
        covariance = sum((u - ma) * (v - mb) for u, v in zip(a, b, strict=True)) / (len(a) - 1)
    np.testing.assert_allclose(result.outputs["variance_x"], float(vx), rtol=1e-12)
    np.testing.assert_allclose(result.outputs["covariance"], float(covariance), rtol=1e-12)
    block.close()


def test_large_finite_samples_do_not_overflow_and_native_errors_clear_partial_state(monkeypatch):
    block = PixelMoments()
    run(block, np.full((2, 3), -1e100))
    result = run(block, np.full((2, 3), 1e100))
    np.testing.assert_allclose(result.outputs["variance_x"], 2e200, rtol=1e-14)
    assert np.isfinite(result.outputs["covariance"]).all()
    block.reset_runtime()
    run(block, np.ones((2, 3)))
    native = cv2.accumulateProduct

    def broken(*args, **kwargs):
        raise cv2.error("Controlled native cross-product failure")

    monkeypatch.setattr(cv2, "accumulateProduct", broken)
    assert block.execute({"image": np.full((2, 3), 2.0)}) is None
    assert not block.current_emissions and not block.memory["streams"]
    monkeypatch.setattr(cv2, "accumulateProduct", native)
    assert not run(block, np.full((2, 3), 10.0)).outputs["ready"]
    result = run(block, np.full((2, 3), 12.0))
    np.testing.assert_allclose(result.outputs["mean_x"], 11)
    block.close()


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba"])
def test_named_channels_follow_encoding_and_do_not_narrow_hdr_values(encoding):
    block = PixelMoments(parameters={"channel_x": "blue", "channel_y": "red"})
    for value in (0.0, 1.0, 2.0, 3.0):
        raw = np.tile([1000 + value, 2000 + value, 3000 - 2 * value], (3, 4, 1)).astype(np.float64)
        if encoding in ("rgb", "rgba"):
            raw = raw[..., ::-1].copy()
        if encoding in ("bgra", "rgba"):
            raw = np.dstack((raw, np.ones((3, 4))))
        result = run(block, ImageArray(raw, encoding))
    np.testing.assert_allclose(result.outputs["variance_x"], 5 / 3)
    np.testing.assert_allclose(result.outputs["covariance"], -10 / 3)
    np.testing.assert_allclose(result.outputs["correlation"], -1)
    block.close()


def test_pause_sample_cap_constant_correlation_and_graphics_off():
    block = PixelMoments(parameters={"maximum_samples": 3})
    assert not run(block, np.ones((3, 4)), learn=False).outputs["ready"]
    for value in (1.0, 2.0, 3.0):
        result = run(block, np.full((3, 4), value))
    before = result.outputs["mean_x"].copy()
    block.set_graphics_mode("off")
    frozen = run(block, np.full((3, 4), 999.0))
    np.testing.assert_array_equal(frozen.outputs["mean_x"], before)
    assert np.all(frozen.outputs["count"] == 3) and not frozen.graphics
    constant = PixelMoments()
    run(constant, np.ones((3, 4)))
    result = run(constant, np.ones((3, 4)))
    assert np.all(result.outputs["valid"] == 255)
    assert not result.outputs["correlation_valid"].any()
    assert not result.outputs["correlation"].any()
    block.close()
    constant.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"alpha": 0},
        {"alpha": 1, "method": "exponential"},
        {"minimum_samples": 1},
        {"minimum_samples": 4, "maximum_samples": 3},
        {"maximum_samples": 2.0},
        {"minimum_coverage": 0},
        {"preview_min": 10, "preview_max": 5},
        {"method": "guess"},
        {"maximum_pixels": 1},
        {"maximum_workspace_bytes": 1},
        {"max_bytes_mb": 0.001},
        {"channel_x": "alpha"},
        {"channel_x": "red"},
    ],
)
def test_invalid_controls_and_preflight_bounds_do_not_enter_native_accumulation(
    parameters, monkeypatch
):
    calls = []
    monkeypatch.setattr(cv2, "accumulate", lambda *a, **k: calls.append(1))
    block = PixelMoments(parameters=parameters)
    assert block.execute({"image": np.ones((3, 4))}) is None
    assert not block.current_emissions and not calls and not block.memory["streams"]
    block.close()


@pytest.mark.parametrize(
    "case", ["ticket_mismatch", "missing_ticket", "grid", "shape", "huge", "nan"]
)
def test_paired_source_contract_rejections_and_native_exception_containment(case, monkeypatch):
    clock = TicketClock()
    a, b = np.ones((3, 4)), np.ones((3, 4))
    ta, tb = clock.issue("a", "external", "part"), clock.issue("b", "external", "part")
    pa, pb = (
        {"ticket": ta, "spatial": frame([4, 3], "grid")},
        {"ticket": tb, "spatial": frame([4, 3], "grid")},
    )
    if case == "ticket_mismatch":
        pb["ticket"] = clock.issue("b", "external", "wrong")
    elif case == "missing_ticket":
        pb.pop("ticket")
    elif case == "grid":
        pb["spatial"] = frame([4, 3], "other")
    elif case == "shape":
        b = np.ones((2, 4))
        pb["spatial"] = frame([4, 2], "grid")
    elif case == "huge":
        b *= 1e101
    else:
        b[0, 0] = np.nan
    block = PixelMoments(parameters={"ticket_policy": "require_match"})
    assert block.execute({"image": ValuePacket(a, pa), "other": ValuePacket(b, pb)}) is None
    assert not block.current_emissions and not block.memory["streams"]
    block.close()


def test_pixel_index_pairing_and_last_learning_ticket_are_explicit():
    clock = TicketClock()
    block = PixelMoments(
        parameters={"alignment": "pixel_indices", "ticket_policy": "require_match"}
    )
    for value in (1.0, 2.0):
        ta, tb = clock.issue("a", "external", value), clock.issue("b", "external", value)
        a = ValuePacket(
            np.full((3, 4), value), {"ticket": ta, "spatial": frame([4, 3], "a"), "lot": "A"}
        )
        b = ValuePacket(np.full((3, 4), value * 2), {"ticket": tb, "spatial": frame([4, 3], "b")})
        result = run(block, a, other=b)
    assert result.payloads["mean_x"]["spatial"]["root_id"] == "a"
    assert result.payloads["mean_x"]["lot"] == "A"
    assert ticket_key(result.payloads["mean_x"]["ticket"]) == ticket_key(ta)
    frozen = run(block, a, other=b, learn=False)
    assert frozen.outputs["status"]["last_observation_tickets"]["image"] == ta
    first_generation = frozen.outputs["status"]["generation"]
    reset = run(block, a, other=b, reset=True)
    assert reset.outputs["status"]["generation"] != first_generation and not reset.outputs["ready"]
    block.close()


def test_stream_isolation_copy_xml_reopen_and_dropdowns(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    block = PixelMoments()
    run(block, np.full((3, 4), 1.0), stream="A")
    run(block, np.full((3, 4), 9.0), stream="B")
    assert np.all(run(block, np.full((3, 4), 3.0), stream="A").outputs["mean_x"] == 2)
    graph, registry = Graph(), builtin_registry()
    source = graph.add(registry.create("input.sequence_demo"))
    stats = graph.add(registry.create("sequence.pixel_moments"))
    graph.connect(source.id, "image", stats.id, "image")
    try:
        for index in range(3):
            worker = Graph.from_dict(graph.to_dict(), registry)
            worker.copy_runtime_from(graph)
            worker.run()
            active = worker.blocks[stats.id]
            assert active.last_result is not None, active.error
            assert active.last_result.outputs["activations"] == index + 1
            graph.copy_runtime_from(worker)
            stats.copy_results_from(active)
            worker.close()
        xml = tmp_path / "moments.sequence.pixel_moments.xml"
        save_block_xml(stats, xml)
        assert read_block_xml(xml)["parameters"] == stats.parameters
        graph.save(tmp_path / "project.json")
        reopened = Graph.load(tmp_path / "project.json", registry)
        assert not reopened.blocks[stats.id].memory["streams"]
        reopened.close()
        window = MainWindow()
        try:
            window.replace_graph(graph)
            app.processEvents()
            window.inspector.select(stats)
            for name in stats.parameter_choices:
                assert isinstance(window.inspector.fields[name], QComboBox)
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    finally:
        block.close()
        graph.close()
