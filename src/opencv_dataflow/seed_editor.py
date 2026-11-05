"""Edit foreground and background watershed point seeds on a captured image."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .canvas import pixmap
from .watershed_seeds import seed_points

SEED_FIELDS = ("foreground_points", "background_points")


class SeedCanvas(QGraphicsView):
    picked = Signal(int, int)

    def __init__(self, image):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.scene().addPixmap(pixmap(image))
        self.height_pixels, self.width_pixels = image.shape[:2]
        self.setSceneRect(0, 0, self.width_pixels, self.height_pixels)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(400, 280)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            x, y = math.floor(point.x()), math.floor(point.y())
            if 0 < x < self.width_pixels - 1 and 0 < y < self.height_pixels - 1:
                self.picked.emit(x, y)
            event.accept()
        else:
            super().mousePressEvent(event)


class SeedEditor(QDialog):
    def __init__(self, image, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit watershed seeds")
        self.resize(720, 520)
        self.points = {
            key: seed_points(parameters[key], image.shape[:2], label)
            for key, label in zip(
                SEED_FIELDS, ("Foreground seeds", "Background seeds"), strict=True
            )
        }
        self.radius = max(2, int(parameters["radius"]))
        self.canvas = SeedCanvas(image)
        self.canvas.picked.connect(self.add_point)
        self.items = []

        layout = QVBoxLayout(self)
        hint = QLabel("Click inside the image to add a seed. Wired inputs override saved points.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Foreground", SEED_FIELDS[0])
        self.mode.addItem("Background", SEED_FIELDS[1])
        controls.addWidget(self.mode)
        for text, callback in (("Undo last", self.undo), ("Clear selected class", self.clear)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use points")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def add_point(self, x, y):
        current = self.points[self.mode.currentData()]
        point = [x, y]
        if point not in current and len(current) < 512:
            current.append(point)
            self.refresh()

    def undo(self):
        current = self.points[self.mode.currentData()]
        if current:
            current.pop()
            self.refresh()

    def clear(self):
        self.points[self.mode.currentData()].clear()
        self.refresh()

    def refresh(self):
        scene = self.canvas.scene()
        for item in self.items:
            scene.removeItem(item)
        self.items.clear()
        for key, color in zip(SEED_FIELDS, ("#71f69f", "#81cfff"), strict=True):
            pen = QPen(QColor(color), 2)
            pen.setCosmetic(True)
            for x, y in self.points[key]:
                r = self.radius
                self.items.append(scene.addEllipse(x + 0.5 - r, y + 0.5 - r, 2 * r, 2 * r, pen))
        foreground, background = (len(self.points[key]) for key in SEED_FIELDS)
        self.status.setText(f"{foreground} foreground · {background} background points")
