"""Seed construction, watershed visualization, and interactive seed editing."""

import json

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import preview_rgb
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.seed_editor import SeedEditor


def test_point_markers_and_watershed_regions_keep_coordinates_and_ids():
    registry = builtin_registry()
    image = np.zeros((32, 48), dtype=np.uint8)
    image[7:24, 6:23] = 210
    image[7:24, 23:42] = 120
    context = {"spatial": frame([48, 32], "camera")}
    packet = ValuePacket(image, context)
    seeds = registry.create("inspect.watershed_point_seeds", parameters={"radius": 1})
    result = seeds.execute({"image": packet, "foreground_points": [[14.1, 14], [32, 14]]})
    assert result is not None, seeds.error
    markers = result.outputs["markers"]
    assert markers.dtype == np.int32
    assert (markers[14, 14], markers[14, 32], markers[1, 1], markers[0, 0]) == (2, 3, 1, 0)
    assert result.outputs["count"] == 2
    assert np.count_nonzero(result.outputs["seeds"]) == 10
    assert result.payloads["markers"]["spatial"] == context["spatial"]
    assert preview_rgb(markers).shape == (*image.shape, 3)

    watershed = registry.create(
        "inspect.watershed", parameters={"marker_mode": "labels", "background_label": 1}
    )
    found = watershed.execute({"image": packet, "markers": markers})
    assert found is not None, watershed.error
    assert found.outputs["count"] == 2
    assert [row["label"] for row in found.outputs["regions"]] == [2, 3]
    assert sum(row["area_pixels"] for row in found.outputs["regions"]) == np.count_nonzero(
        found.outputs["mask"]
    )
    assert np.all(found.outputs["preview"][found.outputs["boundaries"] != 0] == 255)


def test_point_seeds_reject_overlap_selection_and_wrong_frame():
    image = np.zeros((22, 25), dtype=np.uint8)
    seeds = builtin_registry().create("inspect.watershed_point_seeds")
    assert seeds.execute({"image": image, "foreground_points": [[8, 8], [9, 8]]}) is None
    assert seeds.error_kind == "invalid" and "overlap" in seeds.error
    assert (
        seeds.execute({"image": image, "foreground_points": [[8, 8]], "mask": np.zeros_like(image)})
        is None
    )
    assert "inside the mask or region" in seeds.error
    assert (
        seeds.execute(
            {
                "image": ValuePacket(image, {"spatial": frame([25, 22], "camera")}),
                "foreground_points": ValuePacket([[8, 8]], {"spatial": frame([25, 22], "other")}),
            }
        )
        is None
    )
    assert "coordinate frame" in seeds.error


def test_mask_seed_components_and_background_are_distinct():
    foreground = np.zeros((14, 20), np.uint8)
    foreground[4:6, 4:6] = foreground[7:9, 12:14] = 255
    background = np.zeros_like(foreground)
    background[1, 2:17] = 255
    block = builtin_registry().create("inspect.watershed_mask_seeds")
    result = block.execute({"foreground": foreground, "background": background})
    assert result is not None, block.error
    markers = result.outputs["markers"]
    assert result.outputs["count"] == 2
    assert np.unique(markers).tolist() == [0, 1, 2, 3]
    assert (markers[4, 4], markers[7, 12], markers[1, 2], markers[6, 10]) == (2, 3, 1, 0)
    np.testing.assert_array_equal(result.outputs["seeds"], foreground)
    assert block.execute({"foreground": foreground, "background": foreground}) is None
    assert "overlap" in block.error


def test_seed_editor_clicks_stage_both_classes(app, monkeypatch):
    window = MainWindow()
    graph = Graph()
    seeds = graph.add(window.registry.create("inspect.watershed_point_seeds"))
    graph.run(injected={(seeds.id, "image"): np.zeros((40, 60), np.uint8)})
    window.replace_graph(graph)
    window.scene.nodes[seeds.id].setSelected(True)

    def pick(dialog):
        dialog.show()
        app.processEvents()
        for index, (x, y) in enumerate(((15, 12), (45, 25))):
            dialog.mode.setCurrentIndex(index)
            location = dialog.canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))
            QTest.mouseClick(dialog.canvas.viewport(), Qt.MouseButton.LeftButton, pos=location)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SeedEditor, "exec", pick)
    try:
        window.inspector.edit_seeds()
        fields = window.inspector.fields
        assert json.loads(fields["foreground_points"].text()) == [[15, 12]]
        assert json.loads(fields["background_points"].text()) == [[45, 25]]
        assert seeds.parameters["foreground_points"] == seeds.parameters["background_points"] == []
        assert not window.dirty
    finally:
        window.close()
