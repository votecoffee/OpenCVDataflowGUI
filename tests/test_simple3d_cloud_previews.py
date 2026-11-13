import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"


def test_all_shipped_simple3d_projects_choose_native_fit_clouds():
    projects = 0
    for path in ROOT.rglob("project.json"):
        data = json.loads(path.read_bytes())
        for block in data.get("blocks", []):
            if block["type"] == "3d.simple_locate":
                assert block["preview_output"] == "fit", path
                projects += 1
    assert projects == 13


def test_locator_uses_native_colored_thumbnail_and_flyaround_popout_without_matching(
    app, monkeypatch
):
    pytest.importorskip("opencv_simple_3d_locate")
    from opencv_dataflow.app import MainWindow

    graph = Graph.load(ROOT / "project.json", builtin_registry())
    window = MainWindow()
    try:
        window.replace_graph(graph, ROOT / "project.json")
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        output = block.last_result.outputs
        fit = output["fit"]
        assert isinstance(fit, PointCloud)
        assert fit.units == "mm" and fit.frame == "camera"
        assert np.any(np.all(fit.colors == [45, 235, 100], axis=1))
        assert np.any(np.all(fit.colors == [245, 65, 65], axis=1))
        # An optional raster may coexist without taking over the cloud preview.
        assert output["overlay"].shape == (640, 960, 3)
        monkeypatch.setattr(
            block._locator, "locate", lambda *a, **k: pytest.fail("Preview reran matching")
        )
        window.show()
        app.processEvents()
        node = window.scene.nodes["locate"]
        window.view.centerOn(node)
        assert node.thumbnail_value() is fit
        before = node._cloud_yaw
        window.view.animate_cloud_thumbnails()
        assert node._cloud_yaw != before
        window.inspector.select(block)
        window.inspector.tabs.setCurrentWidget(window.inspector.results_page)
        app.processEvents()
        assert window.inspector.preview_choice.currentText() == "Cloud / fit"
        assert window.inspector.cloud_preview_timer.isActive()
        window.inspector.preview_choice.setCurrentText("Result / overlay")
        window.inspector.preview_choice.textActivated.emit("Result / overlay")
        window.inspector.refresh_result()
        assert window.inspector.preview_choice.currentText() == "Result / overlay"
        window.inspector.preview_choice.setCurrentText("Cloud / fit")
        window.inspector.preview_choice.textActivated.emit("Cloud / fit")
        window.inspector.open_image_popout()
        (viewer,) = window.inspector.image_popouts.values()
        assert viewer.output_name == "fit" and viewer.cloud is fit
        assert not viewer.image.pixmap().isNull()
        viewer.flyaround.setChecked(True)
        assert viewer.orbit_timer.isActive()
        yaw = viewer.yaw.value()
        viewer.advance_orbit()
        assert viewer.yaw.value() != yaw
        np.testing.assert_array_equal(viewer.cloud.colors, fit.colors)
        assert node.thumbnail_value() is fit
        viewer.close()
        assert not viewer.orbit_timer.isActive()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
