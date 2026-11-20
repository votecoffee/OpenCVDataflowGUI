"""Qualify native colored fusion, input provenance and workspace admission."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.color_tsdf import FuseColorTSDF, PackPosedRGBD
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/colored_tsdf_fusion"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_colored_surface_matches_plane_and_paint(profile):
    graph = Graph.load(EXAMPLE / "colored_tsdf_fusion.json", builtin_registry())
    opencl = cv2.ocl.useOpenCL()
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["surface"]
        result = block.last_result
        assert result and result.passed, block.error
        report = result.outputs["report"]
        assert report["integrated_frames"] == 3
        assert all(row["confidence_rejected_pixels"] > 100 for row in report["views"])
        assert all(
            row["integrated_pixels"] + row["confidence_rejected_pixels"] == 96 * 72
            for row in report["views"]
        )
        assert report["workspace_estimate_bytes"] < 64 << 20
        cloud = result.outputs["cloud"]
        assert cloud.frame == "panel" and cloud.units == "m"
        assert len(cloud.points) > 2000
        x, y, z = cloud.points.T
        assert np.quantile(np.abs(z - (1.023 + 0.06 * x - 0.03 * y)), 0.99) < 0.0125
        np.testing.assert_allclose(np.linalg.norm(cloud.normals, axis=1), 1, atol=1e-6)
        for mask, rgb in (
            (x < -0.06, (225, 35, 18)),
            ((x > 0.06) & (y < 0.05), (22, 50, 221)),
            ((x > 0.17) & (y > 0.15), (0, 0, 0)),
        ):
            assert mask.sum() > 100
            np.testing.assert_array_equal(cloud.colors[mask], np.broadcast_to(rgb, (mask.sum(), 3)))
        assert not np.any(np.all(cloud.colors == [255, 0, 255], axis=1))
        assert "ticket" in result.payloads["cloud"] and "spatial" not in result.payloads["cloud"]
        assert ("point cloud" in result.graphics) == (profile == "debug")
        assert cv2.ocl.useOpenCL() == opencl
    finally:
        graph.close()


def _inputs(rgb=(25, 100, 205), confidence=None, depth_root=None):
    width, height = 64, 48
    camera = camera_record([[90, 0, 31.5], [0, 90, 23.5], [0, 0, 1]], [], (width, height))
    color = ImageArray(np.full((height, width, 3), rgb, dtype=np.uint8), "rgb")
    depth = DepthArray(np.full((height, width), 1.015), "m")
    values = dict(color=color, depth=depth, pose=pose_record(np.eye(4), "camera", "world"))
    if depth_root is not None:
        values["color"] = ValuePacket(color, {"spatial": frame((width, height), "color")})
        values["depth"] = ValuePacket(depth, {"spatial": frame((width, height), depth_root)})
    if confidence is not None:
        values["confidence"] = np.full((height, width), confidence, np.uint8)
    view = PackPosedRGBD().execute(values).outputs["view"]
    return {"views": view, "camera": camera}


def test_confidence_rejects_all_pixels_and_missing_plane_is_invalid():
    tool = FuseColorTSDF(parameters={"minimum_confidence": 0.5, "resolution": [32, 32, 32]})
    rejected = tool.execute(_inputs(confidence=100))
    assert rejected and not rejected.passed and rejected.outputs["cloud"] is NO_VALUE
    report = rejected.outputs["report"]
    assert report["integrated_frames"] == 0
    assert report["views"][0]["confidence_rejected_pixels"] == 64 * 48
    assert rejected.fail_reasons[0]["code"] == "color_tsdf_empty"
    assert tool.execute(_inputs()) is None and "confidence in every view" in tool.error


def test_grid_conflicts_and_workspace_limit_precede_integration(monkeypatch):
    def reject_native(*args, **kwargs):
        pytest.fail("Invalid inputs must be rejected before native volume allocation")

    monkeypatch.setattr("opencv_dataflow.color_tsdf.cpu_tsdf_volume", reject_native)
    tool = FuseColorTSDF()
    assert tool.execute(_inputs(depth_root="other")) is None
    assert "different pixel grids" in tool.error
    tool.parameters["maximum_workspace_mib"] = 1
    assert tool.execute(_inputs()) is None and "workspace budget" in tool.error


def test_pack_rejects_mixed_capture_tickets():
    view = _inputs()["views"]
    clock = TicketClock()
    fields = {
        key: ValuePacket(packet.value, {"ticket": clock.issue(key)})
        for key, packet in zip(view.labels, view.items, strict=True)
    }
    block = PackPosedRGBD()
    assert block.execute(fields) is None and "same capture" in block.error
