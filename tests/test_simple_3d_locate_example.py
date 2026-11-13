import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


@pytest.mark.local_examples
@pytest.mark.parametrize("variant", ["", "cad_surface"])
def test_simple_3d_locate_example_roundtrip(variant):
    pytest.importorskip("opencv_simple_3d_locate.dataflow")
    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate" / variant
    if variant:
        import opencv_simple_3d_locate as tool

        if not hasattr(tool, "compare_profiles"):
            pytest.skip("CAD example requires Simple3DLocate 0.1.0.dev2 or later")
    truth = [np.asarray(item) for item in json.loads((root / "assets/truth.json").read_text())]
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        assert block.last_result.outputs["count"] == 2
        assert block.last_result.outputs["overlay"].shape == (640, 960, 3)
        assert block.filmstrip_output == "overlay"
        if variant:
            report = json.loads((root / "comparison.json").read_text())
            assert report["qualification"] == "ground_truth"
            assert report["recommended"] is not None
            assert all(row["compatible"] for row in report["profiles"])
            assert (root / "model_analysis.png").is_file()
        cached = block._locator
        graph.run()
        assert block.state == "pass", block.error
        assert block._locator is cached
        assert block.last_result.outputs["count"] == 2
        for pose in block.last_result.outputs["poses"]:
            assert pose["source_frame"] == "part" and pose["target_frame"] == "camera"
            matrix = np.asarray(pose["source_to_target"])
            target = min(truth, key=lambda item: np.linalg.norm(matrix[:3, 3] - item[:3, 3]))
            assert np.linalg.norm(matrix[:3, 3] - target[:3, 3]) < 0.5
            rotation_error = cv2.Rodrigues(matrix[:3, :3] @ target[:3, :3].T)[0]
            assert np.degrees(np.linalg.norm(rotation_error)) < 1
        restored = Graph.from_dict(graph.to_dict(), builtin_registry())
        try:
            assert restored.blocks["locate"].parameters == block.parameters
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_experimental_plane_example_preserves_extra_pose_in_qualification():
    tool = pytest.importorskip("opencv_simple_3d_locate")
    if not hasattr(tool.Settings(), "candidate_backend"):
        pytest.skip("Plane example requires Simple3DLocate 0.1.0.dev4 or later")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/plane_candidates"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        output = block.last_result.outputs
        assert output["count"] == 2
        assert output["report"]["execution"]["candidate_backend"] == "ppf_planes"
        assert output["report"]["matches"][0]["candidate_source"] == "planes"
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        options = dict(
            position_tolerance=5, angle_degrees=5, symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),)
        )
        checked = pose_agreement(
            [np.asarray(p["source_to_target"]) for p in output["poses"]], [truth], **options
        )
        assert checked["matched"] == 1 and checked["extra"] == 1
        best = pose_agreement([np.asarray(output["pose"]["source_to_target"])], [truth], **options)
        assert best["matched"] == 1
        comparison = json.loads((root / "comparison.json").read_bytes())
        assert comparison["recommended"] is None
        assert all(not profile["compatible"] for profile in comparison["profiles"])
    finally:
        graph.close()


@pytest.mark.local_examples
def test_real_tless_first_surface_camera_example():
    tool = pytest.importorskip("opencv_simple_3d_locate")
    if not hasattr(tool, "Camera"):
        pytest.skip("Real T-LESS example requires Simple3DLocate 0.1.0.dev3 or later")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate/real_tless"
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        report = block.last_result.outputs["report"]
        assert report["execution"]["projection"] == "pinhole"
        assert report["execution"]["observation_mode"] == "surface"
        assert block.last_result.outputs["count"] == 1
        pose = block.last_result.outputs["pose"]
        assert 0 < pose["score"] <= 1
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        checked = pose_agreement(
            [np.asarray(pose["source_to_target"])],
            [truth],
            position_tolerance=5,
            angle_degrees=5,
            symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),),
        )
        assert checked["matched"] == 1 and checked["extra"] == 0
        colors = block.last_result.outputs["fit"].colors
        assert np.any(np.all(colors == [90, 115, 175], axis=1))
        benchmark = json.loads((root / "benchmark.json").read_bytes())
        assert len(benchmark["cases"]) == 8
        assert sum(case["segmentation_supplied"] for case in benchmark["cases"]) == 4
        restored = Graph.from_dict(graph.to_dict(), builtin_registry())
        try:
            assert {p.name for p in restored.blocks["locate"].inputs} == {"scene", "camera"}
        finally:
            restored.close()
    finally:
        graph.close()
