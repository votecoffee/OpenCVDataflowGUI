"""Dataset rights/geometry/ground-truth and offline app replay for mesh picking."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("opencv_simple_bin_picking.top_template")
from opencv_simple_bin_picking.sim_dataflow import register

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/Simple Plugins/bin_picking/tless_mesh_grid"


@pytest.mark.local_examples
@pytest.mark.parametrize("cell", range(1, 7))
def test_mesh_episode_keeps_perception_truth_and_native_review_separate(cell):
    registry = builtin_registry()
    register(registry)
    with patch("opencv_simple_bin_picking.sim_dataflow.SimClient", side_effect=AssertionError):
        with ProjectSession.open(CASE / "review.json", registry) as session:
            session.apply([dict(op="update", id="review", fields={"parameters": {"cell": cell}})])
            assert session.inspect()["validation"]["valid"]
            session.run()
            block = session.graph.blocks["review"]
            assert block.state == "pass", block.error
            result = block.last_result.outputs
            assert isinstance(result["motion"], PointCloud)
            report = result["outcome"]
            assert report["part_dataset"]["obj_id"] == 13
            assert report["detections"]["privileged_pose_input"] is False
            assert report["planning_geometry"] == "mesh_with_conservative_box_planning_bound"
            assert report["yaw_evaluation"].startswith("continuous")
            receipts = [r["operation"] for r in report["receipts"]]
            assert receipts.index("capture") < receipts.index("evaluation")
            assert report["estimated_center_error_m"] < 0.002
            assert report["placement_error_m"] < 0.002 and report["observed_detached"]
            candidates = report["detections"]["candidates"]
            assert candidates and all(p["support"] >= 0.6 for p in candidates)
            assert any(p["rectangular_fill_fraction"] < 0.6 for p in candidates)
            source = session.graph.blocks["source_ground_truth_cloud"].last_result.outputs["cloud"]
            assert isinstance(source, PointCloud) and source.frame == "tless_camera"
            session.graph.set_graphics_mode("off")
            session.run()
            assert block.last_result.outputs["outcome"] == report
            assert block.last_result.outputs["motion"] is NO_VALUE


@pytest.mark.local_examples
def test_original_dataset_members_and_converted_geometry_remain_source_bound():
    provenance = json.loads((CASE / "source/provenance.json").read_bytes())
    assert provenance["license"] == "CC-BY-4.0" and provenance["obj_id"] == 13
    for relative, digest in provenance["sha256"].items():
        assert hashlib.sha256((CASE / "source" / relative).read_bytes()).hexdigest() == digest
    model = json.loads((CASE / "provenance.json").read_bytes())
    assert (
        hashlib.sha256((CASE / "assets/obj_000013.ply").read_bytes()).hexdigest()
        == model["member_sha256"]
    )
    assert (
        model["visual_triangles"] == 12774
        and model["collision_triangles"] < model["visual_triangles"]
    )
    review = json.loads((CASE / "source-review.json").read_bytes())
    assert review["maximum_bbox_difference_px"] < 1.2
    assert not review["grasp_ground_truth_available"]
    scene = json.loads((CASE / "scene.json").read_bytes())
    assert all(p["mesh"]["visual"].endswith("tless13.stl") for p in scene["parts"])
    for row in scene["parts"]:
        for kind in ("visual", "collision"):
            assert (
                hashlib.sha256((CASE / row["mesh"][kind]).read_bytes()).hexdigest()
                == row["mesh"][kind + "_sha256"]
            )


@pytest.mark.local_examples
def test_template_save_as_preserves_training_and_live_example_stays_disabled(tmp_path):
    registry = builtin_registry()
    register(registry)
    with patch("opencv_simple_bin_picking.sim_dataflow.SimClient", side_effect=AssertionError):
        with ProjectSession.open(CASE / "project.json", registry) as session:
            original = session.graph.blocks["pick"].parameters.copy()
            session.save(tmp_path / "project.json")
            p = session.graph.blocks["pick"].parameters
            assert p["enabled"] is False
            copied = tmp_path / p["top_template_path"]
            assert (
                hashlib.sha256(copied.read_bytes()).hexdigest() == original["top_template_sha256"]
            )
            session.run()
            report = session.graph.blocks["summary"].last_result.outputs["report"]
            assert report["attempted_cells"] == 6 and report["placed"] == 0
