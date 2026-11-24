"""Sparse masks, dense OpenCV parity, and the saved inspection graph."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.processing import Morphology
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "examples/custom tools/run_length_morphology/compare_opening.json"
ENCODE, MORPH, DECODE = "mask.encode_runs", "mask.morphology_runs", "mask.decode_runs"
MODEL = "binary_mask_runs_v1"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("operation", Morphology.parameter_choices["operation"])
@pytest.mark.parametrize("erosion_border", ("foreground", "background"))
def test_all_operations_equal_dense_for_standard_and_custom_kernels(registry, operation, erosion_border):
    rng = np.random.default_rng(168)
    varied = rng.integers(0, 2, (22, 30), dtype=np.uint8) * 255
    varied[3:16, :10] = 255
    varied[-7:, -9:] = 255
    varied[7, 7] = 0
    cases = (
        (varied, {"shape": "rect", "kernel": 4, "kernel_y": 6}),
        (varied, {"shape": "ellipse", "kernel": 5}),
        (varied, {"shape": "cross", "kernel": 5, "shape_anchor": [1, 3], "anchor": [3, 1]}),
        (varied, {"shape": "diamond", "kernel": 5}),
        (varied, {"shape": "custom", "custom_kernel": [[0, 1], [0, 0], [1, 1]], "anchor": [0, 0]}),
        (np.zeros((3, 2), np.uint8), {
            "shape": "custom", "custom_kernel": [[0, 0, 1], [0, 0, 1]], "anchor": [0, 0],
        }),
        (np.full((2, 3), 255, np.uint8), {"shape": "rect", "kernel": 7}),
    )
    encode, decode = registry.create(ENCODE), registry.create(DECODE)
    native_op = getattr(cv2, "MORPH_" + operation.upper())
    for image, settings in cases:
        encoded = encode.execute({"image": image}).outputs["runs"]
        block = registry.create(MORPH, parameters={
            **settings, "erosion_border": erosion_border, "operation": operation, "iterations": 2,
        })
        result = block.execute({"runs": encoded})
        assert result and result.passed, block.error
        actual = decode.execute({"runs": result.outputs["runs"]})
        assert actual and actual.passed, decode.error
        kernel = Morphology.kernel_matrix(block.parameters)
        options = {} if erosion_border == "foreground" else {"borderValue": 0}
        expected = cv2.morphologyEx(
            image, native_op, kernel, anchor=tuple(block.parameters["anchor"]),
            iterations=2, borderType=cv2.BORDER_CONSTANT, **options,
        )
        np.testing.assert_array_equal(actual.outputs["image"], expected)


def test_threshold_selection_and_data_wires_preserve_ticket_and_geometry(registry):
    plane = np.array([[0, 1000, 1001, 3000], [0, 1, 10, 3000]], np.uint16)
    mask = np.array([[255, 255, 0, 255], [0, 255, 255, 0]], np.uint8)
    payload = {
        "ticket": TicketClock("mask test").issue("camera", "fixed", "ticket-2"),
        "spatial": frame([4, 2], "part"),
        "pyramid": {"level": 0, "largest_size": [4, 2], "to_largest": np.eye(3).tolist()},
    }
    encode = registry.create(ENCODE, parameters={"threshold": 1000, "polarity": "dark"})
    encoded = encode.execute({"image": ValuePacket(plane, payload), "mask": mask})
    assert encoded and encoded.outputs["runs"]["runs"] == [[0, 0, 2], [1, 1, 3]]
    sparse = registry.create(MORPH, parameters={"kernel": 1})
    carried = sparse.execute({"runs": ValuePacket(encoded.outputs["runs"], encoded.payloads["runs"])})
    assert carried and carried.passed, sparse.error
    decode = registry.create(DECODE, parameters={"foreground": 17})
    decoded = decode.execute({"runs": ValuePacket(carried.outputs["runs"], carried.payloads["runs"])})
    assert decoded and decoded.passed, decode.error
    assert {key: decoded.payloads["image"][key] for key in payload} == payload
    np.testing.assert_array_equal(decoded.outputs["image"], [[17, 17, 0, 0], [0, 17, 17, 0]])
    wrong = ValuePacket(encoded.outputs["runs"], {"spatial": frame([5, 2], "part")})
    assert sparse.execute({"runs": wrong}) is None
    assert "dimensions" in sparse.error


@pytest.mark.parametrize("runs", (
    [[0, 4, 4]], [[0, 0, 6]], [[2, 0, 1]], [[False, 0, 2]], [[1, 0, 1], [0, 1, 2]],
    [[0, 0, 2], [0, 2, 3]], [[0, 0, 2], [0, 1, 3]],
))
def test_invalid_records_are_rejected(registry, runs):
    decode = registry.create(DECODE)
    assert decode.execute({"runs": {"model": MODEL, "image_size": [5, 2], "runs": runs}}) is None
    assert "run" in decode.error.lower()


def test_sparse_tile_stays_sparse_and_dense_decode_is_bounded(registry):
    tile = {
        "model": MODEL, "image_size": [1_000_000, 1_000_000],
        "runs": [[y, 40, 60] for y in range(400, 420)],
    }
    sparse = registry.create(MORPH, parameters={"operation": "dilate", "shape": "rect"})
    result = sparse.execute({"runs": tile})
    assert result and result.passed, sparse.error
    assert result.outputs["runs"]["runs"] == [[y, 39, 61] for y in range(399, 421)]
    report = result.outputs["report"]
    assert report["foreground_pixels"] == 484
    assert report["packed_run_bytes"] == 264 and report["dense_uint8_bytes"] == 10**12
    assert report["work_items"] == 60
    decode = registry.create(DECODE)
    assert decode.execute({"runs": result.outputs["runs"]}) is None
    assert "dense decode limit" in decode.error


def test_kernel_validation_and_work_limit(registry):
    mask = {"model": MODEL, "image_size": [5000, 9000], "runs": [[y, 20, 500] for y in range(9000)]}
    sparse = registry.create(MORPH, parameters={"shape": "custom", "custom_kernel": [[1, 0, 1]]})
    assert sparse.execute({"runs": mask}) is None
    assert "contiguous" in sparse.error
    sparse = registry.create(MORPH, parameters={"operation": "dilate", "kernel": 255, "shape": "rect"})
    assert sparse.execute({"runs": mask}) is None
    assert "work limit" in sparse.error


@pytest.mark.parametrize("profile", ("lean", "debug"))
def test_saved_graph_dense_opening_agrees(registry, profile):
    graph = Graph.load(PROJECT, registry)
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(3)
        graph.run()
        compare = graph.blocks["compare"].last_result
        assert compare and compare.passed, graph.blocks["compare"].error
        assert compare.outputs["value"] == 0.0
        sparse = graph.blocks["opening"].last_result.outputs["report"]
        assert 0 < sparse["packed_run_bytes"] < sparse["dense_uint8_bytes"] / 100
        assert block_entry(graph.blocks["opening"]).path == ("Custom", "Image processing", "Morphology")
    finally:
        graph.close()
