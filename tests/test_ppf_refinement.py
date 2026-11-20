"""Check native PPF fit quality and ranking of repeated hypotheses."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import read_cloud
from opencv_dataflow.cloud_tools import MatchPPF
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.ppf_refinement import rank_surface_candidates

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/ppf_surface_ranking"
TRUTH = (
    ([0.08, -0.17, 0.15], [-0.13, 0, 0.70]),
    ([-0.1, 0.1, 0.3], [0.15, 0.035, 0.76]),
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_ppf_graph_corrects_native_vote_order_and_rejects_clutter(profile):
    graph = Graph.load(EXAMPLE / "ppf_surface_ranking.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        result = graph.blocks["verify"].last_result
        votes = graph.blocks["votes"].last_result
        assert result.passed and votes.passed
        raw = votes.outputs["poses"]
        assert len(raw) == 5 and raw[0]["votes"] > raw[1]["votes"]
        np.testing.assert_allclose(raw[0]["translation"], TRUTH[1][1], atol=0.01)
        output = result.outputs
        assert output["count"] == 2 and output["poses"][0] == output["pose"]
        assert output["poses"][0]["votes"] < output["poses"][1]["votes"]
        model = graph.blocks["model"].last_result.outputs["cloud"]
        scene = graph.blocks["scene"].last_result.outputs["cloud"]
        for pose, (rotation, translation) in zip(output["poses"], TRUTH, strict=True):
            assert pose["refined"] and pose["source_frame"] == "panel"
            assert pose["target_frame"] == "station" and pose["units"] == "m"
            np.testing.assert_allclose(pose["translation"], translation, atol=0.0003)
            np.testing.assert_allclose(pose["rvec"], rotation, atol=0.003)
            matrix = np.asarray(pose["source_to_target"])
            xyz = model.points @ matrix[:3, :3].T + matrix[:3, 3]
            distance = np.linalg.norm(xyz[:, None, :] - scene.points[None, :, :], axis=2)
            peers = distance.argmin(axis=1)
            rows = np.arange(len(xyz))
            nearest = distance[rows, peers]
            overlap = nearest <= 0.004
            pairs = overlap & (distance.argmin(axis=0)[peers] == rows)
            quality = pose["quality"]
            assert quality["model_overlap_count"] == int(overlap.sum())
            assert quality["inlier_count"] == int(pairs.sum())
            assert quality["inlier_rmse"] == pytest.approx(np.sqrt(np.mean(nearest[pairs] ** 2)))

        evaluated = output["report"]["evaluated_candidates"]
        assert len(evaluated) == 8
        assert sum(c["status"] == "rejected" for c in evaluated) == 6
        for candidate in evaluated[:2]:
            assert candidate["selected_pass"] == 2
            assert candidate["quality"]["score"] > candidate["seed"]["score"]
        assert ("Ranked surface matches" in result.graphics) == (profile == "debug")
        assert result.payloads["pose"]["ticket"]["value"] == "ppf-ranking-fixture"
    finally:
        graph.close()


@pytest.mark.local_examples
def test_duplicate_loses_to_stronger_geometry_and_native_update_can_be_discarded():
    model = read_cloud(EXAMPLE / "assets/model.ply")
    scene = read_cloud(EXAMPLE / "assets/scene.ply")
    seeds = []
    for rotation, translation in TRUTH:
        matrix = np.eye(4)
        matrix[:3, :3] = cv2.Rodrigues(np.array(rotation))[0]
        matrix[:3, 3] = translation
        seeds.append(matrix)
    duplicate = seeds[0].copy()
    duplicate[0, 3] += 0.0015
    candidates = [
        {"candidate": 1, "votes": 900, "matrix": duplicate},
        {"candidate": 2, "votes": 800, "matrix": seeds[1]},
        {"candidate": 3, "votes": 700, "matrix": seeds[0]},
    ]

    class ShiftAway:
        def registerModelToScene(self, source, target):
            assert len(source) == len(target) >= 6
            transform = np.eye(4)
            transform[2, 3] = 0.1
            return 0, 0.0001, transform

    settings = MatchPPF(parameters={"ranking": "refined_geometry"}).parameters
    solved, report = rank_surface_candidates(
        candidates,
        model,
        scene,
        settings,
        ShiftAway(),
        lambda: False,
    )
    assert [pose["candidate"] for pose in solved] == [3, 2]
    assert report[0]["status"] == "duplicate" and report[0]["duplicate_of"] == 3
    assert all(row["selected_pass"] == 0 for row in report)
    assert all(row["refinement"][0]["quality"]["failed_checks"] for row in report)
    assert (
        len(
            rank_surface_candidates(
                candidates,
                model,
                scene,
                dict(settings, max_matches=1),
                ShiftAway(),
                lambda: False,
            )[0]
        )
        == 1
    )
    assert (
        rank_surface_candidates(candidates, model, scene, settings, ShiftAway(), lambda: True)
        is None
    )


def test_geometry_distances_have_coupled_limits():
    block = MatchPPF(parameters={"ranking": "refined_geometry", "refinement_distance": 0.003})
    with pytest.raises(ValidationError, match="inside the refinement distance"):
        block.validate_parameters()
