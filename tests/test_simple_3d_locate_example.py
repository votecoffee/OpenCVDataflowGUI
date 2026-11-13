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
def test_cropped_search_uses_separate_full_frame_depth():
    tool = pytest.importorskip("opencv_simple_3d_locate")
    if not hasattr(tool, "DepthScene"):
        pytest.skip("Separate depth example requires Simple3DLocate 0.1.0.dev6 or later")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/cropped_search_depth"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        report = block.last_result.outputs["report"]
        assert block.last_result.outputs["count"] == 1
        depth = report["execution"]["depth_verification"]
        assert depth["source"] == "depth_scene"
        assert depth["input_points"] > report["execution"]["input_points"]
        assert report["matches"][0]["depth_verification"]["applied"]
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        checked = pose_agreement(
            [np.asarray(report["matches"][0]["source_to_target"])],
            [truth],
            position_tolerance=5,
            angle_degrees=5,
            symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),),
        )
        assert checked["matched"] == 1 and checked["extra"] == checked["missed"] == 0
    finally:
        graph.close()


@pytest.mark.local_examples
def test_depth_verified_full_scene_has_one_correct_pose_and_records_rejection():
    tool = pytest.importorskip("opencv_simple_3d_locate")
    if not hasattr(tool.Settings(), "depth_verification"):
        pytest.skip("Depth example requires Simple3DLocate 0.1.0.dev5 or later")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/depth_verified"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        output = block.last_result.outputs
        assert output["count"] == 1
        report = output["report"]
        assert report["execution"]["depth_rejected"] > 0
        depth = output["pose"]["depth_verification"]
        assert depth["applied"] and depth["status"] == "checked"
        assert depth["free_space_fraction"] < depth["maximum_free_space"]
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        checked = pose_agreement(
            [np.asarray(output["pose"]["source_to_target"])],
            [truth],
            position_tolerance=5,
            angle_degrees=5,
            symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),),
        )
        assert checked["matched"] == 1 and checked["extra"] == checked["missed"] == 0
        for mode in ("auto", "off"):
            comparison = json.loads((root / f"comparison_{mode}.json").read_bytes())
            assert comparison["depth_verification"] == mode
            assert comparison["recommended"] is None
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
            expected = {"scene", "camera"}
            if hasattr(tool, "DepthScene"):
                expected.add("depth_scene")
            assert {p.name for p in restored.blocks["locate"].inputs} == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_feature_candidate_example_uses_full_scene_and_retains_failed_qualification():
    pytest.importorskip("opencv_simple_3d_locate")
    from opencv_simple_3d_locate.optimization import PROFILES, pose_agreement

    if "features" not in PROFILES:
        pytest.skip("Feature example requires Simple3DLocate 0.1.0.dev7 or later")
    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/feature_candidates"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        outputs = block.last_result.outputs
        assert outputs["count"] == 1
        report = outputs["report"]
        assert report["execution"]["candidate_backend"] == "ppf_features"
        assert report["matches"][0]["candidate_source"] == "features"
        feature = report["execution"]["feature_search"]
        assert 0 < feature["correspondences"] <= 2048
        assert feature["trial_limit"] == 4096
        assert 0 < feature["coarse_proposals"] <= 128
        assert feature["seconds"]["scene_descriptors"] > 0
        assert report["matches"][0]["depth_verification"]["applied"]
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        checked = pose_agreement(
            [np.asarray(outputs["pose"]["source_to_target"])],
            [truth],
            position_tolerance=5,
            angle_degrees=5,
            symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),),
        )
        assert checked["matched"] == 1 and checked["missed"] == checked["extra"] == 0
        comparison = json.loads((root / "comparison.json").read_bytes())
        assert comparison["recommended"] is None
        assert all(not row["compatible"] for row in comparison["profiles"])
        benchmark = json.loads((root / "benchmark.json").read_bytes())
        full = next(row for row in benchmark["cases"] if row["case"] == "full_000230")
        assert not full["segmentation_supplied"]
        assert len(benchmark["cases"]) == 8
    finally:
        graph.close()


@pytest.mark.local_examples
def test_real_tless_training_captures_completed_inputs_without_changing_recipe(app):
    pytest.importorskip("opencv_simple_3d_locate.preview_capture")
    from opencv_simple_3d_locate.qt_training import LocateTrainingDialog

    from opencv_dataflow.plugin_dialog import PluginDialogContext

    if not hasattr(PluginDialogContext, "snapshot_processed_inputs"):
        pytest.skip("Completed-input capture requires a compatible host")
    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/cropped_search_depth"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    context = dialog = None
    try:
        graph.set_run_profile("debug")
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        draft = dict(block.parameters)
        context = PluginDialogContext(graph, block)
        dialog = LocateTrainingDialog(None, draft, context)
        recipe = dialog.recipe
        original_samples = recipe.samples.copy()
        dialog.capture_preview_button.click()
        assert dialog.processed_capture is not None, dialog.status.text()
        assert dialog.processed_capture.bundle.passed
        assert dialog.processed_capture.bundle.execution_count == block.timing.count
        assert dialog.recipe is recipe and draft == block.parameters
        np.testing.assert_array_equal(recipe.samples, original_samples)
        np.testing.assert_array_equal(dialog.test_scene.points, block.last_inputs["scene"].points)
        np.testing.assert_array_equal(
            dialog.depth_scene.points, block.last_inputs["depth_scene"].points
        )
        assert dialog.test_scene is not block.last_inputs["scene"]
        assert dialog.camera.units == dialog.test_scene.units == dialog.depth_scene.units == "mm"
        assert dialog.mode.currentText() == "Test search cloud"
        assert not dialog.viewer.image_item.pixmap().isNull()
        assert "unverified" in dialog.capture_status.text()
        dialog._preview()
        assert len(dialog.result.matches) == block.last_result.outputs["count"] == 1
        assert draft == block.parameters
    finally:
        if dialog is not None:
            dialog.close()
        if context is not None:
            context._close()
        graph.close()


@pytest.mark.local_examples
def test_pose_alternatives_example_keeps_annotation_failure_and_reports_competitors():
    pytest.importorskip("opencv_simple_3d_locate.alternatives")
    from opencv_simple_3d_locate.optimization import pose_agreement

    root = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/3d_locate/real_tless/pose_alternatives"
    )
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        output = block.last_result.outputs
        assert output["count"] == 1
        report = output["report"]
        info = report["matches"][0]["pose_alternatives"]
        assert output["pose"]["pose_alternatives"] == info
        assert info["status"] == "alternatives_observed" and info["count"] >= 1
        assert 0 <= info["relative_score_margin"] < 0.2
        assert len(info["alternatives"]) <= 3 and not info["exhaustive"]
        assert report["execution"]["seconds"]["pose_alternatives"] >= 0
        truth = np.asarray(json.loads((root / "assets/truth.json").read_bytes()))
        options = dict(
            position_tolerance=5, angle_degrees=5, symmetries=(np.diag([-1.0, -1.0, 1.0, 1.0]),)
        )
        checked = pose_agreement(
            [np.asarray(output["pose"]["source_to_target"])], [truth], **options
        )
        assert checked["matched"] == 0 and checked["missed"] == checked["extra"] == 1
        alternative = pose_agreement(
            [np.asarray(row["source_to_target"]) for row in info["alternatives"]],
            [truth],
            **options,
        )
        assert alternative["matched"] == 1
        saved = json.loads((root / "isolated_000220.json").read_bytes())
        assert saved["validation"]["matched"] == 0
        assert saved["validation"]["missed"] == saved["validation"]["extra"] == 1
        assert saved["segmentation_supplied"]
        assert (root / "competing_result.png").is_file()
    finally:
        graph.close()
