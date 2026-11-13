"""Reusable frozen-cloud region teaching, separate from graph execution and recipes."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .canvas import pixmap
from .cloud_teaching_geometry import CloudBox, CloudProjection, CloudTeachingSource, vector


class _Canvas(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setMinimumSize(420, 330)
        self.setMouseTracking(True)
        self.start = self.last = self.face = self.original = None

    def position(self, event):
        projection = self.editor.projection
        return np.array([event.position().x() * projection.width / max(self.width(), 1),
                         event.position().y() * projection.height / max(self.height(), 1)])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.editor.render()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 25, 32))
        editor, view = self.editor, self.editor.projection
        if view is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a reference cloud")
            return
        painter.scale(self.width() / view.width, self.height() / view.height)
        painter.drawPixmap(0, 0, editor._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for box in editor.boxes:
            selected = box.name == editor.selected
            corners, _ = view.project(box.corners)
            painter.setPen(QPen(QColor(255, 220, 60) if selected else QColor(135, 125, 175),
                                2 if selected else 1))
            for i in range(8):
                for bit in (1, 2, 4):
                    if i < (j := i ^ bit):
                        painter.drawLine(QPointF(*corners[i]), QPointF(*corners[j]))
            painter.drawText(QPointF(*corners[0]), box.name)
            if selected:
                handles, _ = view.project(box.faces)
                for i, point in enumerate(handles):
                    color = (QColor(255, 100, 90), QColor(100, 240, 130),
                             QColor(95, 170, 255))[i // 2]
                    painter.setPen(QPen(color, 2))
                    painter.drawRect(QRectF(point[0] - 4, point[1] - 4, 8, 8))
                    painter.drawText(QPointF(*(point + [6, -6])), "XYZ"[i // 2] + "-+"[i % 2])
        for first, last, color, text in editor.segments:
            xy, _ = view.project([first, last])
            painter.setPen(QPen(QColor(*color), 2))
            painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))
            if text:
                painter.drawText(QPointF(*(xy[1] + [5, -5])), text)
        if editor.last_pick is not None:
            xy, _ = view.project(editor.last_pick["xyz"])
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(QPointF(*xy), 6, 6)
        if self.start is not None and self.last is not None and editor.mode.currentData() == "draw":
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(QPointF(*self.start), QPointF(*self.last)).normalized())

    def mousePressEvent(self, event):
        if self.editor.projection is None:
            return
        self.start = self.last = self.position(event)
        self.face = self.original = None
        box = next((b for b in self.editor.boxes if b.name == self.editor.selected), None)
        if (event.button() == Qt.MouseButton.LeftButton and box is not None
                and self.editor.mode.currentData() == "orbit"):
            handles, _ = self.editor.projection.project(box.faces)
            distances = np.linalg.norm(handles - self.start, axis=1)
            if distances.min() < 10:
                self.face, self.original = int(distances.argmin()), box
        self.update()

    def mouseMoveEvent(self, event):
        if self.start is None or self.editor.projection is None:
            return
        here = self.position(event)
        delta = here - self.last
        self.last = here
        editor = self.editor
        if self.face is not None:
            try:
                box = editor.projection.drag_face(self.original, self.face, here - self.start)
                editor.boxes = tuple(box if b.name == box.name else b for b in editor.boxes)
                editor.status.setText("Release to keep the region edit. Undo is in the trainer.")
            except ValueError as error:
                editor.status.setText(str(error))
        elif event.buttons() & (Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton):
            editor.pan += delta
            editor.render()
        elif editor.mode.currentData() == "orbit":
            editor.set_view(round((editor.yaw.value() + delta[0] * .5 + 180) % 360 - 180),
                            round(np.clip(editor.pitch.value() + delta[1] * .5, -90, 90)))
        self.update()

    def mouseReleaseEvent(self, event):
        editor, start = self.editor, self.start
        if start is None or editor.projection is None:
            return
        here = self.position(event)
        if self.face is not None:
            box = next((b for b in editor.boxes if b.name == self.original.name), None)
            if box is not None and box != self.original:
                editor.box_changed.emit(box)
        elif event.button() == Qt.MouseButton.LeftButton:
            if editor.mode.currentData() == "draw":
                try:
                    box = editor.projection.box_from_rectangle(start, here, editor.new_box_name)
                    editor.box_created.emit(box)
                except ValueError as error:
                    editor.status.setText(str(error))
            elif np.linalg.norm(here - start) < 4:
                pick = editor.projection.pick(*here)
                if pick is not None:
                    editor.last_pick = pick
                    editor.status.setText(f"Source row {pick['row']}: "
                                          + ", ".join(f"{v:.6g}" for v in pick["xyz"])
                                          + f" {pick['units']} · {pick['frame']}")
                    editor.point_picked.emit(pick)
        self.start = self.last = self.face = self.original = None
        self.update()

    def wheelEvent(self, event):
        editor = self.editor
        editor.zoom = float(np.clip(editor.zoom * 1.2**(event.angleDelta().y() / 120), .1, 100))
        editor.render()
        event.accept()


class CloudTeachingWidget(QWidget):
    """Point/region proposals in source coordinates; the consumer owns draft edits/undo."""

    point_picked = Signal(object)
    box_created = Signal(object)
    box_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = self.projection = self._image = self.last_pick = None
        self._colors = None
        self.boxes, self.segments, self.selected = (), (), None
        self.new_box_name = "Plane 1"
        self.zoom, self.pan = 1., np.zeros(2)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        for title, name in (("Orbit / edit region", "orbit"), ("Pick source point", "pick"),
                            ("Draw new region", "draw")):
            self.mode.addItem(title, name)
        controls.addWidget(self.mode)
        for title, yaw, pitch in (("XY", 0, 0), ("XZ", 0, 90), ("YZ", 90, 0)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, a=yaw, b=pitch: self.set_view(a, b))
            controls.addWidget(button)
        fit = QPushButton("Fit view")
        fit.clicked.connect(self.fit_view)
        controls.addWidget(fit)
        focus = QPushButton("Focus selected")
        focus.clicked.connect(self.focus_selected)
        controls.addWidget(focus)
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.yaw = QSlider(Qt.Orientation.Horizontal)
        self.pitch = QSlider(Qt.Orientation.Horizontal)
        for title, slider, low, high, value in (("Yaw", self.yaw, -180, 180, 25),
                                                ("Pitch", self.pitch, -90, 90, -20)):
            slider.setRange(low, high)
            slider.setValue(value)
            slider.valueChanged.connect(self.render)
            row.addWidget(QLabel(title))
            row.addWidget(slider)
        layout.addLayout(row)
        self.canvas = _Canvas(self)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("Wheel: zoom · Middle/right drag: pan · "
                             "Draw a visible patch, then drag colored XYZ face handles.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_cloud(self, cloud):
        self.source = CloudTeachingSource(cloud)
        self.last_pick = None
        self._colors = None
        self.canvas.start = self.canvas.last = self.canvas.face = self.canvas.original = None
        self.boxes, self.segments, self.selected = (), (), None
        self.fit_view()

    def set_boxes(self, boxes, selected=None):
        boxes = tuple(boxes)
        if (not all(isinstance(b, CloudBox) for b in boxes)
                or len({b.name for b in boxes}) != len(boxes)):
            raise ValueError("Expected uniquely named CloudBox regions")
        self.boxes, self.selected = boxes, selected
        self.canvas.update()

    def set_segments(self, segments):
        self.segments = tuple((vector(a), vector(b), tuple(color), str(text))
                              for a, b, color, text in segments)
        self.canvas.update()

    def set_point_colors(self, colors=None):
        if colors is not None:
            colors = np.asarray(colors)
            if (self.source is None or colors.shape != self.source.cloud.points.shape
                    or colors.dtype != np.uint8):
                raise ValueError("Display colors must be RGB bytes matching the frozen source rows")
            colors = colors.copy()
            colors.flags.writeable = False
        self._colors = colors
        if self.projection is not None:
            self._image = pixmap(self.projection.image(colors))
        self.canvas.update()

    def set_datum(self, origin, normal, length):
        origin, normal = vector(origin), vector(normal)
        size = np.linalg.norm(normal)
        if size < 1e-12 or not np.isfinite(length) or length <= 0:
            raise ValueError("Datum requires a nonzero normal and positive axis length")
        n = normal / size
        u = np.cross(n, np.eye(3)[np.argmin(abs(n))])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        self.set_segments((origin, origin + length * axis, color, title) for axis, color, title in
                          ((u, (255, 85, 70), "X"), (v, (80, 245, 100), "Y"),
                           (n, (75, 160, 255), "Datum normal")))

    def set_view(self, yaw, pitch):
        self.yaw.blockSignals(True)
        self.pitch.blockSignals(True)
        self.yaw.setValue(yaw)
        self.pitch.setValue(pitch)
        self.yaw.blockSignals(False)
        self.pitch.blockSignals(False)
        self.render()

    def fit_view(self):
        self.zoom, self.pan = 1., np.zeros(2)
        self.render()

    def focus_selected(self):
        box = next((b for b in self.boxes if b.name == self.selected), None)
        if box is None or self.projection is None:
            self.status.setText("Select a region to focus the view")
            return
        view = self.projection
        pixels, _ = view.project(box.corners)
        ratio = .75 * min(view.width / max(np.ptp(pixels[:, 0]), 1),
                           view.height / max(np.ptp(pixels[:, 1]), 1))
        zoom = float(np.clip(self.zoom * ratio, .1, 100))
        center = pixels.mean(0) - view.offset
        self.pan = -center * (zoom / self.zoom)
        self.zoom = zoom
        self.render()

    def render(self, *_):
        if self.source is None:
            return
        width, height = max(self.canvas.width(), 1), max(self.canvas.height(), 1)
        factor = min(1., 2048 / width, 1536 / height)
        self.projection = CloudProjection(self.source, max(1, round(width * factor)),
                                          max(1, round(height * factor)), yaw=self.yaw.value(),
                                          pitch=self.pitch.value(), zoom=self.zoom, pan=self.pan)
        self._image = pixmap(self.projection.image(self._colors))
        self.canvas.update()
