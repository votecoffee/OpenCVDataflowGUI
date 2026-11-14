"""Retained native motion review through the saved-project runtime, without ROS."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("opencv_simple_bin_picking.sim_dataflow")
from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/Simple Plugins/bin_picking/bin_to_grid"
MOTION = EXAMPLE / "motion_review"
CASES = [(case, cell) for case in ("baseline", "shifted_stacks") for cell in range(1, 7)]
CASES += [("held_transfer", 3)]


@pytest.mark.local_examples
@pytest.mark.parametrize("case,cell", CASES)
def test_recorded_motion_keeps_native_cloud_and_judgment_with_graphics_off(case, cell):
    registry = builtin_registry()
    register(registry)
    with patch("opencv_simple_bin_picking.sim_dataflow.SimClient", side_effect=AssertionError):
        with ProjectSession.open(MOTION / case / "project.json", registry) as session:
            session.apply([dict(op="update", id="review", fields={"parameters": {"cell": cell}})])
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["review"]
            assert block.state == ("fail" if case == "held_transfer" else "pass"), block.error
            outputs = block.last_result.outputs
            report = outputs["outcome"]
            assert block.preview_output == "motion"
            cloud = outputs["motion"]
            assert isinstance(cloud, PointCloud) and cloud.frame == "world" and cloud.units == "m"
            assert cloud.points.shape == cloud.colors.shape and np.isfinite(cloud.points).all()
            assert np.any(np.all(cloud.colors == [255, 70, 170], axis=1))
            assert outputs["motion_summary"].ndim == 3
            model = report["kinematic_model"]["sha256"]
            paths = [
                r["result"]["tcp_path"] for r in report["receipts"] if "tcp_path" in r["result"]
            ]
            assert paths and all(p["kinematic_sha256"] == model for p in paths)
            if case == "held_transfer":
                assert report["status"] == "held" and report["automatic_retry"] is False
                assert not any(r.get("motion_stage") == "transfer" for r in report["receipts"])
            else:
                assert report["observed_detached"] and report["settled"]
            session.graph.set_graphics_mode("off")
            session.run()
            assert block.last_result.outputs["outcome"] == report
            assert block.last_result.outputs["motion"] is NO_VALUE
            assert block.last_result.outputs["motion_summary"] is NO_VALUE


@pytest.mark.local_examples
def test_disabled_live_graph_accounts_for_every_cell_without_native_client():
    registry = builtin_registry()
    register(registry)
    with patch("opencv_simple_bin_picking.sim_dataflow.SimClient", side_effect=AssertionError):
        with ProjectSession.open(EXAMPLE / "project.json", registry) as session:
            assert session.inspect()["validation"]["valid"]
            assert session.graph.blocks["pick"].parameters["enabled"] is False
            session.run()
            summary = session.graph.blocks["summary"].last_result.outputs["report"]
            assert not summary["complete"] and summary["attempted_cells"] == 6
            assert all(row["status"] == "disabled" for row in summary["outcomes"])


@pytest.mark.local_examples
def test_motion_sources_are_hashed_and_separate_from_continuous_collision_claims():
    for relative in (
        "examples/Simple Plugins/bin_picking/bin_to_grid/motion_review/sha256.json",
        "integrations/ros2/pick_backend_sha256.json",
        "integrations/simple_plugins/bin_picking_sha256.json",
    ):
        path = ROOT / relative
        manifest = json.loads(path.read_bytes())
        files = manifest.get("files", manifest)
        base = path.parent
        if path.name == "pick_backend_sha256.json":
            base /= "packages/opencv_dataflow_pick_backend"
        elif path.name == "bin_picking_sha256.json":
            base /= "bin_picking"
        for name, digest in files.items():
            assert not any(p.startswith(".") and "cache" in p for p in Path(name).parts)
            assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
    acceptance = json.loads((MOTION / "acceptance.json").read_bytes())
    assert acceptance["independent_tcp_measurement"] is False
    assert acceptance["continuous_collision_certificate"] is False
    for row in acceptance["episodes"].values():
        assert row["placed"] == 6 and row["final_bin_candidates"] == 0
        assert row["final_all_detached"] and row["moveit_attachments_empty"]
