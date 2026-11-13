"""Retained real-data teaching through the normal optional-plugin dialog route."""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opencv_simple_3d_measure.qt_training")
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugin_dialog import PluginDialogContext

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure"


@pytest.mark.local_examples
def test_real_recipe_teaches_with_completed_fixture_and_preserves_active_recipe(app):
    graph = Graph.load(ROOT / "real_tless/frame000013/project.json", builtin_registry())
    context = dialog = None
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state == "pass", block.error
        active = copy.deepcopy(block.parameters)
        draft = copy.deepcopy(active)
        context = PluginDialogContext(graph, block)
        dialog = block.create_plugin_dialog(None, draft, context)
        assert dialog.session.trained and dialog.session.frame == "object6"
        assert dialog.regions.count() == 2 and len(dialog.editor.source.cloud.points) > 100000
        before = dialog.session.reference.copy()
        # This uses the consumed scene+annotation fixture together, not newer upstream outputs.
        dialog._capture_reference()
        assert dialog.session.frame == "object6" and not dialog.session.trained
        assert len(dialog.session.planes) == 2
        assert not np.array_equal(dialog.session.reference, before)
        job = dialog.session.job()
        assert dialog.session.accept(job, job.run())
        dialog.refresh("part_surface")
        dialog.editor.focus_selected()
        assert dialog.editor.zoom > 1
        assert np.any(np.all(dialog.editor._colors == [245, 65, 65], axis=1))
        assert block.parameters == active and draft == active
        dialog.reject()
        assert block.parameters == active and draft == active
    finally:
        if dialog:
            dialog.done(0)
        if context:
            context._close()
        graph.close()
