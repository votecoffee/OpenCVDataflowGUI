"""Check source-to-target match interpolation, edge fidelity, and native guards."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/edge_aware_matches"
TYPE = "cv.edge_aware_flow"


@pytest.fixture(autouse=True)
def thread_pool():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def images():
    return [cv2.imread(str(EXAMPLE / f"assets/{name}.png")) for name in ("source", "target")]


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_replay_preserves_motion_boundary(registry, profile):
    graph = Graph.load(EXAMPLE / "edge_aware_matches.json", registry)
    graph.set_run_profile(profile)
    try:
        node = graph.blocks["interpolate"]
        assert block_entry(node).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        truth = 4 + 0.003 * (np.arange(18, 94)[:, None] - 56)
        for weight, tolerance in ((999.0, 0.10), (0.0, 3.15)):
            node.parameters["geodesic_lambda"] = weight
            graph.run()
            result = node.last_result
            assert result and result.passed, node.error
            report = result.outputs["report"]
            assert report["seeds"] == 252 and report["method"] == "edge_aware"
            vectors = result.outputs["flow"].vectors
            error = np.hypot(vectors[18:-18, 80:88, 0], vectors[18:-18, 80:88, 1] - truth)
            assert error.mean() < tolerance
            if weight == 0:
                assert error.mean() > 2.65
            assert result.outputs["visualization"].shape == (112, 176, 3)
            assert bool(result.drawings) == (profile == "debug")
    finally:
        graph.close()


def test_match_record_cost_map_and_native_parity(registry):
    a, b = images()
    size = [176, 112]
    x, y = np.meshgrid(np.arange(8, 176, 8), np.arange(8, 104, 8))
    source = np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32)
    target = source.copy()
    target[:, 1] += np.where(source[:, 0] < 88, 4, -3) + 0.003 * (source[:, 1] - 56)
    cost = np.zeros((112, 176), np.float32)
    cost[:, 87:89] = 1
    expected = cv2.ximgproc.createEdgeAwareInterpolator()
    expected.setK(64)
    expected.setLambda(990.0)
    expected.setSigma(0.06)
    expected.setFGSLambda(40.0)
    expected.setFGSSigma(5.0)
    expected.setCostMap(cost)
    native = expected.interpolate(a, source, b, target)
    clock = TicketClock("matches")
    source_context = {"spatial": frame(size, "a"), "ticket": clock.issue("a", "fixed", "a")}
    target_context = {"spatial": frame(size, "b"), "ticket": clock.issue("b", "fixed", "b")}
    match = {
        "query_points": source.tolist(),
        "train_points": target.tolist(),
        "query_size": size,
        "train_size": size,
        "query_context": source_context,
        "train_context": target_context,
    }
    tool = registry.create(
        TYPE,
        parameters={
            "neighbors": 64,
            "geodesic_lambda": 990.0,
            "sigma": 0.06,
            "fgs_lambda": 40.0,
            "fgs_sigma": 5.0,
        },
    )
    mask = np.zeros(cost.shape, np.uint8)
    mask[32:56, 32:56] = 255
    inputs = {
        "image": ValuePacket(a, source_context),
        "target": ValuePacket(b, target_context),
        "matches": match,
        "mask": ValuePacket(mask, source_context),
        "cost_map": ValuePacket(cost, source_context),
    }
    result = tool.execute(inputs)
    assert result and result.passed, tool.error
    field = result.outputs["flow"]
    np.testing.assert_array_equal(field.vectors[mask != 0], native[mask != 0])
    assert not field.vectors[mask == 0].any()
    np.testing.assert_array_equal(result.outputs["support"], mask)
    assert field.metadata["ticket"] == source_context["ticket"]
    assert field.metadata["spatial"] == source_context["spatial"]
    assert field.metadata["flow_matches"] == {"source": source_context, "target": target_context}
    assert result.drawings["supplied matches"].objects
    match["train_context"] = {**target_context, "ticket": source_context["ticket"]}
    assert tool.execute(inputs) is None
    assert "different ticket" in tool.error


def test_exact_translation_and_repeated_identical_pairs(registry):
    a, b = images()
    points = np.array([[16, 16], [16, 50], [50, 16], [50, 50], [16, 16]], np.float32)
    tool = registry.create(TYPE, parameters={"neighbors": 4})
    result = tool.execute(
        {
            "image": a,
            "target": b,
            "source_points": points.tolist(),
            "target_points": (points + (3, -2)).tolist(),
        }
    )
    assert result and result.passed, tool.error
    report = result.outputs["report"]
    assert report["method"] == "uniform_translation"
    assert report["input_matches"] == 5 and report["seeds"] == 4
    vectors = result.outputs["flow"].vectors
    np.testing.assert_array_equal(vectors[..., 0], 3)
    np.testing.assert_array_equal(vectors[..., 1], -2)
    assert not result.outputs["support"][:2].any()
    assert not result.outputs["support"][:, -3:].any()


@pytest.mark.parametrize(
    "case,message",
    [
        ("duplicates", "share a source seed"),
        ("collinear", "span both image axes"),
        ("neighbors", "Only 4 distinct source seeds"),
        ("cost", "float32 in [0, 1]"),
        ("size", "raster sizes must match"),
    ],
)
def test_rejects_unsafe_associations(registry, case, message):
    a, b = images()
    source = [[8, 8], [16, 8], [8, 16], [16, 16]]
    target = [[8, 9], [16, 9], [8, 17], [16, 18]]
    if case == "duplicates":
        source[-1] = [8.1, 8.1]
    if case == "collinear":
        source = [[x, 8] for x in (8, 16, 24, 32)]
    tool = registry.create(
        TYPE,
        parameters={
            "source_points": source,
            "target_points": target,
            "neighbors": 5 if case == "neighbors" else 4,
        },
    )
    inputs = {"image": a, "target": b[:-1] if case == "size" else b}
    if case == "cost":
        inputs["cost_map"] = np.full(a.shape[:2], 2.0, np.float32)
    assert tool.execute(inputs) is None
    assert message in tool.error
