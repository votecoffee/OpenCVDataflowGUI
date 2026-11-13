"""Full-source picking, physical region edits and cached teaching overlays."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from opencv_dataflow.cloud_teaching import CloudTeachingWidget
from opencv_dataflow.cloud_teaching_geometry import CloudBox, CloudProjection, CloudTeachingSource
from opencv_dataflow.pointclouds import PointCloud


def test_first_visible_picking_keeps_original_rows_beyond_display_budget_and_freezes_source():
    rng = np.random.default_rng(7)
    xyz = rng.uniform([-10, -10, 15], [10, 10, 20], (25000, 3)).astype(np.float32)
    xyz[1234], xyz[24001], xyz[24002] = [0, 0, 5], [0, 0, 0], [0, 0, 0]
    cloud = PointCloud(xyz, "mm", "part")
    source = CloudTeachingSource(cloud)
    view = CloudProjection(source, 640, 480, yaw=0, pitch=0)
    xy, _ = view.project([[0, 0, 0]])
    picked = view.pick(*xy[0])
    assert picked["row"] == 24001 and picked["xyz"] == [0, 0, 0]
    assert picked["source_identity"] == source.identity
    cloud.points[:] = 1
    assert source.cloud.points[24001].tolist() == [0, 0, 0]
    assert not source.cloud.points.flags.writeable
    assert view.pick(-50, -50) is None
    other = CloudTeachingSource(PointCloud(source.cloud.points, "mm", "other"))
    assert source.identity != other.identity


def test_rectangle_seeds_visible_surface_and_face_drag_uses_physical_axes():
    xy = np.array([(x, y) for x in range(-5, 6) for y in range(-5, 6)], float)
    front = np.c_[xy, np.zeros(len(xy))]
    back = front + [0, 0, 10]
    source = CloudTeachingSource(PointCloud(np.r_[front, back], "mm", "part"))
    view = CloudProjection(source, 640, 480, yaw=0, pitch=0)
    rectangle, _ = view.project([[-6, -6, 0], [6, 6, 0]])
    box = view.box_from_rectangle(*rectangle, "patch")
    assert box.minimum[2] < 0 < box.maximum[2] < 1
    changed = view.drag_face(box, 1, [view.scale * 2, 0])
    assert changed.maximum[0] == pytest.approx(box.maximum[0] + 2)
    assert changed.minimum == box.minimum and changed.maximum[1:] == box.maximum[1:]
    with pytest.raises(ValueError, match="Rotate"):
        view.drag_face(box, 5, [10, 0])
    with pytest.raises(ValueError, match="maximum"):
        view.drag_face(box, 1, [-1000, 0])
    rotated = CloudProjection(source, 800, 600, yaw=41, pitch=32, zoom=3, pan=(21, -9))
    axis = rotated.rotation[:2, 2] * [rotated.scale, -rotated.scale]
    assert rotated.drag_face(box, 5, axis * 2).maximum[2] == pytest.approx(box.maximum[2] + 2)
    with pytest.raises(ValueError, match="three"):
        view.box_from_rectangle([0, 0], [1, 1], "empty")


def event(xy, button=Qt.MouseButton.LeftButton):
    return SimpleNamespace(position=lambda: QPointF(*xy), button=lambda: button)


def test_widget_point_and_region_proposals_preserve_camera_and_do_not_reproject_overlays(app):
    xy = np.array([(x, y) for x in range(-5, 6) for y in range(-5, 6)], float)
    widget = CloudTeachingWidget()
    widget.resize(900, 700)
    widget.set_cloud(PointCloud(np.c_[xy, np.zeros(len(xy))], "mm", "part"))
    widget.set_view(0, 0)
    widget.mode.setCurrentIndex(widget.mode.findData("pick"))
    got = []
    widget.point_picked.connect(got.append)
    view = widget.projection
    projected, _ = view.project([[0, 0, 0]])
    widget.canvas.mousePressEvent(event(projected[0]))
    widget.canvas.mouseReleaseEvent(event(projected[0]))
    assert got[0]["row"] == 60
    widget.set_boxes([CloudBox("region", (-6, -6, -1), (6, 6, 1))], "region")
    widget.set_datum([0, 0, 0], [0, 0, 1], 3)
    widget.set_point_colors(np.tile([255, 0, 0], (len(xy), 1)).astype(np.uint8))
    assert widget.projection is view  # Metadata/primitive overlays never rebuild the sensor buffer.
    assert len(widget.segments) == 3
    regions = []
    widget.box_created.connect(regions.append)
    widget.mode.setCurrentIndex(widget.mode.findData("draw"))
    rectangle, _ = view.project([[-5, -5, 0], [5, 5, 0]])
    widget.canvas.mousePressEvent(event(rectangle[0]))
    widget.canvas.mouseReleaseEvent(event(rectangle[1]))
    assert regions[0].minimum[2] < 0 < regions[0].maximum[2]
    widget.focus_selected()
    xy_center, _ = widget.projection.project([[0, 0, 0]])
    np.testing.assert_allclose(xy_center[0], [widget.projection.width / 2,
                                             widget.projection.height / 2])
    widget.show()  # Offscreen Qt surface: exercise paint/resize without desktop automation.
    app.processEvents()
    widget.set_cloud(PointCloud(np.c_[xy, np.ones(len(xy))], "mm", "part"))
    assert widget.last_pick is None and widget.boxes == () and widget.segments == ()
    region = CloudBox("region", (-6, -6, -1), (6, 6, 2))
    widget.set_boxes([region], "region")
    widget.mode.setCurrentIndex(widget.mode.findData("orbit"))
    handle, _ = widget.projection.project(region.faces)
    widget.canvas.mousePressEvent(event(handle[0]))
    widget.set_boxes(())  # A consumer can remove a region while its drag is active.
    widget.canvas.mouseReleaseEvent(event(handle[0]))
    widget.close()


@pytest.mark.parametrize("width,height,zoom", [(0, 480, 1), (640, 5000, 1),
                                              (1.5, 480, 1), (640, 480, 0)])
def test_invalid_projection_is_rejected(width, height, zoom):
    source = CloudTeachingSource(PointCloud(np.empty((0, 3)), "mm", "part"))
    with pytest.raises(ValueError):
        CloudProjection(source, width, height, zoom=zoom)
