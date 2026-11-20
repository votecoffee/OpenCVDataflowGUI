"""Verify depth connectivity, visible measurements and source-label reuse."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_components import MeasureDepthComponents
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/depth_components/depth_components.json"
)


def _camera(shape, **kwargs):
    return camera_record(
        [[200, 0, 0], [0, 250, 0], [0, 0, 1]],
        [],
        shape[::-1],
        frame="test_optical",
        **kwargs,
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_example_measures_known_sizes_and_reconstructs_exact_size_filtered_mask(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["measure"]
        result = block.last_result
        assert result and result.passed, block.error
        outputs = result.outputs
        expected = np.zeros((96, 160), np.int32)
        expected[20:48, 28:60] = 1
        expected[30:34, 39:44] = 0
        expected[24:50, 60:92] = 2
        expected[64:78, 112:130] = 3
        candidate = (expected > 0).astype(np.uint8) * 255
        candidate[84, 20:22] = 255
        np.testing.assert_array_equal(outputs["labels"], expected)
        np.testing.assert_array_equal(outputs["mask"], (expected != 0) * 255)
        np.testing.assert_array_equal(outputs["candidate_mask"], candidate)
        assert outputs["count"] == 3
        assert outputs["report"]["candidate_components"] == 4
        assert outputs["report"]["retained_pixels"] == 1960
        for row, z in zip(outputs["collection"]["items"], [1.0, 1.25, 0.9], strict=True):
            v, u = np.where(expected == row["id"])
            metrics, geometry = row["metrics"], row["geometry"]
            assert metrics["area"] == len(u)
            assert metrics["width_m"] == pytest.approx((u.max() - u.min() + 1) * z / 200)
            assert metrics["height_m"] == pytest.approx((v.max() - v.min() + 1) * z / 250)
            assert metrics["projected_area_m2"] == pytest.approx(len(u) * z**2 / (200 * 250))
            assert metrics["depth_span_m"] == 0
            assert geometry["camera_centroid_m"] == pytest.approx(
                [(u.mean() - 79.5) * z / 200, (v.mean() - 47.5) * z / 250, z]
            )
        selected = graph.blocks["selected_mask"].last_result
        assert selected and selected.passed, graph.blocks["selected_mask"].error
        assert selected.outputs["pixel_count"] == 832
        np.testing.assert_array_equal(selected.outputs["mask"], (expected == 2) * 255)
        source = graph.blocks["depth"].last_result
        for field in ("spatial", "ticket"):
            assert result.payloads["labels"][field] == source.payloads["depth"][field]
            assert selected.payloads["mask"][field] == source.payloads["depth"][field]
        if profile == "debug":
            assert len(result.drawings["Measured components"].objects) == 6
            assert not result.graphics["Depth components"][30, 39].any()
        else:
            assert not result.drawings and not result.graphics
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("connectivity", "step", "expected"),
    [
        ("4", 0.05, [[1, 2], [3, 4]]),
        ("8", 0.05, [[1, 2], [2, 1]]),
        ("8", 0, [[1, 1], [1, 1]]),
    ],
)
def test_crossing_diagonals_and_native_spatial_path_preserve_separate_edges(
    connectivity, step, expected
):
    z = DepthArray([[1, 1.25], [1.25, 1]], "m")
    block = MeasureDepthComponents(
        parameters={"connectivity": connectivity, "maximum_depth_step_m": step, "minimum_pixels": 1}
    )
    result = block.execute({"depth": z, "camera": _camera(z.shape)})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["labels"], expected)


def test_inclusive_raw_depth_interval_and_transitive_step_with_visible_z_extent():
    z = np.array([[7, 8, 9, 10, 11, 0]], np.uint16)
    block = MeasureDepthComponents(
        parameters={
            "input_units": "m",
            "units_per_sample": 0.125,
            "minimum_depth_m": 1,
            "maximum_depth_m": 1.25,
            "maximum_depth_step_m": 0.125,
            "minimum_pixels": 1,
        }
    )
    result = block.execute({"depth": z, "camera": _camera(z.shape)})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["labels"], [[0, 1, 1, 1, 0, 0]])
    metrics = result.outputs["collection"]["items"][0]["metrics"]
    assert metrics["depth_span_m"] == 0.25
    assert metrics["mean_depth_m"] == 1.125
    assert metrics["width_m"] == pytest.approx(3.5 * 1.25 / 200 - 0.5 / 200)
    np.testing.assert_array_equal(z, [[7, 8, 9, 10, 11, 0]])


def test_roi_and_pixel_count_filter_preserve_source_frame_and_exact_labels():
    data = DepthArray([[1] * 5, [1] * 5, [1] * 5], "m")
    spatial = frame((5, 3), "original-depth")
    ticket = TicketClock("depth-components").issue("depth", "fixed", "part-1")
    source = ValuePacket(data, {"spatial": spatial, "ticket": ticket})
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [0, 0, 5, 2],
        "image_size": [5, 3],
        "coordinate_context": spatial,
    }
    mask = np.full(data.shape, 255, np.uint8)
    mask[:, 2] = 0
    mask[1, 4] = 0
    registry = builtin_registry()
    block = MeasureDepthComponents(parameters={"minimum_pixels": 1, "maximum_pixels": 3})
    result = block.execute(
        {"depth": source, "camera": _camera(data.shape), "region": region, "mask": mask}
    )
    assert result and result.passed, block.error
    o = result.outputs
    assert o["count"] == 1
    assert o["report"]["candidate_components"] == 2
    expected = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(o["labels"], expected)
    assert o["collection"]["context"]["spatial"] == spatial
    assert result.payloads["mask"]["ticket"] == ticket
    reconstruct = registry.create("inspect.mask_from_labels")
    assert (
        reconstruct.execute(
            {
                "labels": ValuePacket(o["labels"], result.payloads["labels"]),
                "collection": ValuePacket(o["collection"], result.payloads["collection"]),
            }
        ).outputs["pixel_count"]
        == 3
    )
    altered = o["labels"].copy()
    altered[0, 0] = 1
    assert reconstruct.execute({"labels": altered, "collection": o["collection"]}) is None
    assert "original label plane" in reconstruct.error


def test_empty_selection_reports_empty_results_and_size_guard_rejects_fragmentation():
    z = DepthArray([[1, 2], [2, 1]], "m")
    block = MeasureDepthComponents(
        parameters={"connectivity": "4", "minimum_pixels": 1, "maximum_components": 2}
    )
    inputs = {"depth": z, "camera": _camera(z.shape)}
    empty = block.execute({**inputs, "mask": np.zeros(z.shape, np.uint8)})
    assert empty and not empty.passed and not block.error
    assert empty.fail_reasons[0]["code"] == "depth_components_empty"
    assert empty.outputs["collection"]["count"] == 0
    assert not empty.outputs["candidate_mask"].any()
    assert block.execute(inputs) is None and "Too many components" in block.error


def test_wrong_camera_and_foreign_selection_are_rejected():
    z = DepthArray(np.ones((3, 3)), "m")
    block = MeasureDepthComponents()
    assert block.execute({"depth": z, "camera": _camera((4, 4))}) is None
    assert "image size differs" in block.error
    camera = camera_record(
        [[200, 0, 0], [0, 250, 0], [0, 0, 1]], [0.1, 0, 0, 0], (3, 3), "m", "distorted"
    )
    assert (
        block.execute({"depth": z, "camera": camera}) is None
        and "undistorted pinhole" in block.error
    )
    foreign = ValuePacket(np.ones((3, 3), np.uint8), {"spatial": frame((3, 3), "other")})
    source = ValuePacket(z, {"spatial": frame((3, 3), "depth")})
    assert block.execute({"depth": source, "camera": _camera(z.shape), "mask": foreign}) is None
    assert "coordinate frame" in block.error
