"""Image-space region drawing with transactional dialog acceptance."""

import copy
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .images import preview_rgb
from .selection import contains, rasterize


class RegionCanvas(QWidget):
    changed = Signal()

    def __init__(self, image, parameters, parent=None):
        super().__init__(parent)
        rgb = np.ascontiguousarray(preview_rgb(image))
        self.image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.parameters = copy.deepcopy(parameters)
        self.image_shape = image.shape[:2]
        rasterize(self.region(), self.image_shape)
        self.drawing = False
        self.drag = None
        self._backup = None
        self._start = None
        self._vertices = []
        self.setMinimumSize(400, 280)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def image_rect(self):
        h, w = self.image_shape
        scale = min((self.width() - 20) / w, (self.height() - 20) / h)
        return QRectF(
            (self.width() - w * scale) / 2, (self.height() - h * scale) / 2, w * scale, h * scale
        )

    def to_widget(self, point):
        rect = self.image_rect()
        scale = rect.width() / self.image_shape[1]
        # Draw pixel centers, including the half-pixel offset inside the image bounds.
        return QPointF(
            rect.left() + (point[0] + 0.5) * scale, rect.top() + (point[1] + 0.5) * scale
        )

    def to_image(self, point, clamp=False):
        rect = self.image_rect()
        if not clamp and not rect.contains(point):
            return None
        h, w = self.image_shape
        x = math.floor((point.x() - rect.left()) * w / rect.width())
        y = math.floor((point.y() - rect.top()) * h / rect.height())
        return [max(0, min(w - 1, x)), max(0, min(h - 1, y))]

    def region(self, invert=None):
        data = {
            "model": "image_region_v1",
            "image_size": list(self.image_shape[::-1]),
            **self.parameters,
        }
        if invert is not None:
            data["invert"] = invert
        return data

    def handles(self):
        p = self.parameters
        if p["shape"] == "rectangle":
            x, y, w, h = p["rectangle"]
            return [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
        if p["shape"] == "circle":
            x, y = p["center"]
            return [[x, y], [x + p["radius"], y]]
        return p["vertices"]

    def new_shape(self):
        self.cancel_gesture()
        self._backup = copy.deepcopy(self.parameters)
        self.drawing = True
        self._vertices = []
        self._start = None
        self.changed.emit()
        self.update()
        self.setFocus()

    def cancel_gesture(self):
        if self._backup is not None:
            self.parameters = self._backup
        self.drawing, self.drag = False, None
        self._backup, self._start = None, None
        self.changed.emit()
        self.update()

    def finish_polygon(self):
        if not self.drawing or self.parameters["shape"] != "polygon":
            return
        if len(self._vertices) < 3:
            return
        proposal = self.region()
        proposal["vertices"] = self._vertices
        try:
            rasterize(proposal, self.image_shape)
        except ValueError:
            return
        self.parameters["vertices"] = copy.deepcopy(self._vertices)
        self.drawing, self._backup = False, None
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            self.finish_polygon()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.to_image(event.position())
        if point is None:
            return
        if self.drawing:
            if self.parameters["shape"] == "polygon":
                if not self._vertices or point != self._vertices[-1]:
                    self._vertices.append(point)
            else:
                self._start = point
                self.update_draw(point)
        else:
            self._backup = copy.deepcopy(self.parameters)
            self._start = point
            self.drag = next(
                (
                    i
                    for i, handle in enumerate(self.handles())
                    if (self.to_widget(handle) - event.position()).manhattanLength() <= 12
                ),
                None,
            )
            if self.drag is None and contains(
                rasterize(self.region(False), self.image_shape), point
            ):
                self.drag = "move"
            if self.drag is None:
                self._backup = None
        self.changed.emit()
        self.update()

    def update_draw(self, point):
        a, b = self._start, point
        if self.parameters["shape"] == "rectangle":
            self.parameters["rectangle"] = [
                min(a[0], b[0]),
                min(a[1], b[1]),
                abs(a[0] - b[0]) + 1,
                abs(a[1] - b[1]) + 1,
            ]
        else:
            self.parameters["center"] = a[:]
            self.parameters["radius"] = max(1, round(math.dist(a, b)))

    def update_drag(self, point):
        p = self.parameters
        if self.drag == "move":
            delta = np.array(point) - self._start
            old = self._backup
            if p["shape"] == "rectangle":
                x, y, w, h = old["rectangle"]
                p["rectangle"] = [int(x + delta[0]), int(y + delta[1]), w, h]
            elif p["shape"] == "circle":
                p["center"] = (np.array(old["center"]) + delta).tolist()
            else:
                p["vertices"] = (np.array(old["vertices"]) + delta).tolist()
        elif p["shape"] == "rectangle":
            x, y, w, h = self._backup["rectangle"]
            opposite = [[x + w - 1, y + h - 1], [x, y + h - 1], [x, y], [x + w - 1, y]][self.drag]
            self._start = opposite
            self.update_draw(point)
        elif p["shape"] == "circle":
            if self.drag == 0:
                p["center"] = point
            else:
                p["radius"] = max(1, round(math.dist(p["center"], point)))
        else:
            p["vertices"][self.drag] = point

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        point = self.to_image(event.position(), clamp=True)
        if self.drawing and self._start is not None and self.parameters["shape"] != "polygon":
            self.update_draw(point)
        elif self.drag is not None:
            self.update_drag(point)
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.to_image(event.position(), clamp=True)
        if self.drawing and self._start is not None and self.parameters["shape"] != "polygon":
            self.update_draw(point)
            self.drawing = False
            self._backup = None
        elif self.drag is not None:
            self.update_drag(point)
            self.drag, self._backup = None, None
        self._start = None
        self.changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.finish_polygon()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.drawing:
            self.finish_polygon()
        elif event.key() == Qt.Key.Key_Escape and (self.drawing or self.drag is not None):
            self.cancel_gesture()
        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and self.drawing:
            if self._vertices:
                self._vertices.pop()
                self.changed.emit()
                self.update()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111b29"))
        painter.drawImage(self.image_rect(), self.image)
        try:
            region = self.region()
            if self.drawing and region["shape"] == "polygon":
                region["vertices"] = self._vertices
            mask = rasterize(region, self.image_shape)
            tint = np.zeros((*self.image_shape, 4), np.uint8)
            tint[..., :3] = [60, 220, 120]
            tint[..., 3] = np.where(mask != 0, 55, 0)
            overlay = QImage(
                tint.data,
                tint.shape[1],
                tint.shape[0],
                tint.strides[0],
                QImage.Format.Format_RGBA8888,
            )
            painter.drawImage(self.image_rect(), overlay)
        except (ValueError, TypeError):
            pass
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#52edab"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        points = (
            self._vertices
            if self.drawing and self.parameters["shape"] == "polygon"
            else self.handles()
        )
        if self.parameters["shape"] == "circle":
            center = self.to_widget(self.parameters["center"])
            radius = self.parameters["radius"] * self.image_rect().width() / self.image_shape[1]
            painter.drawEllipse(center, radius, radius)
        elif len(points) >= 3:
            painter.drawPolygon(QPolygonF([self.to_widget(p) for p in points]))
        elif len(points) > 1:
            painter.drawPolyline(QPolygonF([self.to_widget(p) for p in points]))
        painter.setBrush(QColor("#f5cd65"))
        for point in points:
            painter.drawEllipse(self.to_widget(point), 4, 4)


class RegionEditor(QDialog):
    def __init__(self, image, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw / edit region")
        self.resize(900, 680)
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.shape = QComboBox()
        self.shape.addItems(["rectangle", "circle", "polygon"])
        self.shape.setCurrentText(parameters["shape"])
        toolbar.addWidget(self.shape)
        self.invert = QCheckBox("Invert selection")
        self.invert.setChecked(parameters["invert"])
        toolbar.addWidget(self.invert)
        draw = QPushButton("Draw new")
        toolbar.addWidget(draw)
        layout.addLayout(toolbar)
        hint = QLabel(
            "Drag to draw rectangles/circles. Click polygon vertices; Enter, double-click or "
            "right-click to finish. Drag handles to resize, or the shape to move. "
            "Backspace removes a new vertex; Escape cancels a gesture."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(
            QLabel("Editing a captured frame. Run again after upstream image changes.")
        )
        self.canvas = RegionCanvas(image, parameters)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply region")
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        draw.clicked.connect(self.canvas.new_shape)
        self.shape.currentTextChanged.connect(self.change_shape)
        self.invert.toggled.connect(self.change_invert)
        self.canvas.changed.connect(self.refresh)
        self.refresh()

    def change_shape(self, shape):
        self.canvas.cancel_gesture()
        self.canvas.parameters["shape"] = shape
        self.canvas.new_shape()

    def change_invert(self, value):
        self.canvas.parameters["invert"] = value
        if self.canvas._backup is not None:
            self.canvas._backup["invert"] = value
        self.refresh()
        self.canvas.update()

    def refresh(self):
        try:
            if self.canvas.drawing:
                raise ValueError("Finish drawing the shape before applying it.")
            mask = rasterize(self.canvas.region(), self.canvas.image_shape)
            key = {"rectangle": "rectangle", "circle": "center", "polygon": "vertices"}[
                self.canvas.parameters["shape"]
            ]
            self.status.setText(
                f"{key}: {self.canvas.parameters[key]} · {np.count_nonzero(mask)} selected pixels"
            )
            self.ok.setEnabled(True)
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))
            self.ok.setEnabled(False)

    def accept(self):
        self.refresh()
        if self.ok.isEnabled():
            super().accept()

    @property
    def parameters(self):
        return copy.deepcopy(self.canvas.parameters)
