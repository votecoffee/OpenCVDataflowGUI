"""Real retained source context survives reopening without changing measurement evidence."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

pytest.importorskip("opencv_simple_3d_measure.reference")
from opencv_simple_3d_measure import load_recipe

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugin_dialog import PluginDialogContext

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure"


@pytest.mark.local_examples
def test_source_context_rgb_pixels_and_calibration_match_pinned_real_capture():
    recipe = load_recipe(ROOT / "reference_context/assets/measurement.s3m")
    evidence = recipe.reference_evidence
    raw = ROOT / "real_tless/source/test_primesense/000002"
    image = cv2.imread(str(raw / "rgb/000003.png"))
    depth = cv2.imread(str(raw / "depth/000003.png"), cv2.IMREAD_UNCHANGED)
    y, x = np.nonzero(depth)
    np.testing.assert_array_equal(evidence.pixels, np.c_[x, y])
    np.testing.assert_array_equal(evidence.colors, image[y, x, ::-1])
    camera = json.loads((raw / "scene_camera.json").read_bytes())["3"]
    assert evidence.metadata["camera"] == camera
    k = np.array(camera["cam_K"]).reshape(3, 3)
    z = depth[y, x] * camera["depth_scale"]
    expected = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    fixture = np.asarray(evidence.metadata["reference_to_source"])
    measured = recipe.reference @ fixture[:3, :3].T + fixture[:3, 3]
    np.testing.assert_allclose(measured, expected, atol=1e-8)


@pytest.mark.local_examples
def test_context_project_preserves_previous_values_and_save_as_bytes(tmp_path):
    graph = Graph.load(ROOT / "reference_context/project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state == "pass", block.error
        report = block.last_result.outputs["report"]
        prior = json.loads((ROOT / "real_tless/frame000013/expected.json").read_bytes())
        assert report == prior
        assert graph.blocks["scene"].last_result.outputs["cloud"].colors is not None
        block.parameters["render_graphics"] = False
        graph.run()
        assert block.last_result.outputs["report"] == report
        path = tmp_path / "project.json"
        digest = block.parameters["sha256"]
        graph.save(path)
    finally:
        graph.close()
    reopened = Graph.load(path, builtin_registry())
    try:
        block = reopened.blocks["measure"]
        recipe_path = path.parent / block.parameters["recipe_path"]
        assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == digest
        assert load_recipe(recipe_path).reference_evidence.colors is not None
        reopened.run()
        assert block.last_result.outputs["report"] == report
    finally:
        reopened.close()


@pytest.mark.local_examples
def test_native_source_view_keeps_cached_result_and_frozen_context(app):
    graph = Graph.load(ROOT / "reference_context/project.json", builtin_registry())
    dialog = context = None
    try:
        graph.run()
        block = graph.blocks["measure"]
        active = dict(block.parameters)
        context = PluginDialogContext(graph, block)
        dialog = block.create_plugin_dialog(None, dict(active), context)
        assert dialog.session.trained and dialog.source_colors.isEnabled()
        evidence = dialog.session.reference_evidence
        dialog.source_colors.setChecked(True)
        np.testing.assert_array_equal(dialog.editor.source.cloud.colors, evidence.colors)
        assert dialog.editor._colors is None
        dialog._review()
        review = dialog._children[-1]
        cached = review.reference_result
        with patch("opencv_simple_3d_measure.qt_review.Measurer", side_effect=AssertionError):
            review.display.setCurrentIndex(1)
            assert review.reference_result is cached
            np.testing.assert_array_equal(review.viewer.cloud.pixels, evidence.pixels)
        review._capture()
        frozen = review.current_evidence
        assert frozen.metadata["capture"]["execution_count"] == block.timing.count
        context._outputs_published()
        review._outputs_changed()
        assert review.current_evidence is frozen
        dialog.reject()
        assert block.parameters == active
    finally:
        if dialog:
            dialog.done(0)
        if context:
            context._close()
        graph.close()
