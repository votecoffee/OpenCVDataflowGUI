"""Zoomable sample and score review, shared by new and restored training."""

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QTransform
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class SampleView(QGraphicsView):
    def __init__(self, image, parent):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        raster = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self.image_item = scene.addPixmap(QPixmap.fromImage(raster.copy()))
        self.setSceneRect(self.image_item.boundingRect())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setToolTip("Scroll to zoom; drag to pan. 100% shows source pixels.")
        self.fitted = True

    def fit_image(self):
        self.fitted = True
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        self.fitted = False
        self.setTransform(QTransform())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.fitted = False
            current = self.transform().m11()
            wanted = min(32., max(.01, current * 1.25 ** (delta / 120)))
            self.scale(wanted / current, wanted / current)
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()


class SampleReview(QDialog):
    def __init__(self, parent, image, chart, description):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Classifier sample review")
        self.resize(1100, 650)
        layout = QVBoxLayout(self)
        self.description = QLabel(description)
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        images = QHBoxLayout()
        self.views = [SampleView(pixels, self) for pixels in (image, chart) if pixels is not None]
        for view in self.views:
            images.addWidget(view)
        layout.addLayout(images, 1)
        actions = QHBoxLayout()
        for label, method in (("Fit", "fit_image"), ("100%", "actual_size")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, name=method: [getattr(v, name)() for v in self.views]
            )
            actions.addWidget(button)
        actions.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        layout.addLayout(actions)
