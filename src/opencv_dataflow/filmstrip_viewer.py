"""Horizontal dockable results view, with verdict borders and ticket details."""

from pathlib import Path

import cv2
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from .canvas import pixmap
from .filmstrip import annotate_filmstrip_image, encode_thumbnail
from .image_metadata import decode_file_image
from .images import format_image, preview_rgb

COLORS = {"pass": "#2cb978", "fail": "#ef5350", "error": "#cc78e8", "pending": "#edba50"}


class VerdictDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLORS[index.data(Qt.ItemDataRole.UserRole)]), 3))
        painter.drawRoundedRect(option.rect.adjusted(3, 3, -3, -3), 4, 4)
        painter.restore()


class FilmstripImageView(QGraphicsView):
    def __init__(self, parent=None):
        scene = QGraphicsScene(parent)
        super().__init__(scene, parent)
        self.image_item = scene.addPixmap(QPixmap())
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setToolTip("Scroll to zoom · drag to pan")
        self._fitted = True

    def set_image(self, image):
        self.image_item.setPixmap(image)
        self.setSceneRect(self.image_item.boundingRect())
        self.fit_image()

    def fit_image(self):
        self._fitted = True
        if self.image_item.pixmap().isNull():
            return
        rect = self.image_item.boundingRect()
        scale = min(
            max(1, self.viewport().width() - 4) / rect.width(),
            max(1, self.viewport().height() - 4) / rect.height(),
        )
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(rect.center())

    def actual_size(self):
        self._fitted = False
        self.resetTransform()

    def zoom(self, factor):
        self._fitted = False
        current = self.transform().m11()
        low = min(0.02, current)
        high = max(16.0, current)
        scale = min(high, max(low, current * factor))
        self.scale(scale / current, scale / current)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.zoom(1.25 ** (delta / 120))
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit_image()


class FilmstripImageWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ticket = None
        self.image_revision = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.frame = QFrame()
        self.frame.setObjectName("filmstripImageFrame")
        frame_layout = QVBoxLayout(self.frame)
        self.view = FilmstripImageView(self.frame)
        frame_layout.addWidget(self.view, 1)
        self.empty = QLabel("No image was captured for this result.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.empty, 1)
        layout.addWidget(self.frame, 1)
        self.controls = QWidget()
        actions = QHBoxLayout(self.controls)
        actions.setContentsMargins(0, 0, 0, 0)
        for label, callback in (
            ("Zoom out", lambda: self.view.zoom(0.8)),
            ("Zoom in", lambda: self.view.zoom(1.25)),
            ("Fit", self.view.fit_image),
            ("100%", self.view.actual_size),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        actions.addStretch()
        layout.addWidget(self.controls)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

    def open_entry(self, row):
        same_ticket = self.ticket == row["ticket"]
        self.ticket = row["ticket"]
        self.image_revision = row["image_revision"]
        override = row["override"]
        encoded = override.detail if override is not None and override.detail else row["thumbnail"]
        image = QPixmap.fromImage(QImage.fromData(encoded))
        description = "Selected result image" if override is not None else "Captured thumbnail"
        source = row["image_file"]
        frame = row["frame"]
        layers = [row["layers"][label][0] for label in sorted(row["layers"])]
        if source:
            try:
                path = Path(source["path"])
                if path.is_file():
                    raw, _ = decode_file_image(path.read_bytes(), source["metadata_mode"])
                    if raw is not None:
                        restored = format_image(raw, source["color_format"], source["pixel_depth"])
                        if (
                            (not frame or list(restored.shape[:2][::-1]) == frame["image_size"])
                            and encode_thumbnail(restored) == row["source_thumbnail"]
                        ):
                            if layers:
                                bgr = cv2.cvtColor(preview_rgb(restored), cv2.COLOR_RGB2BGR)
                                restored = annotate_filmstrip_image(bgr, frame, layers)
                            image = pixmap(restored)
                            description = "Reloaded source image"
            except (OSError, ValueError, TypeError, cv2.error):
                pass
        has_image = not image.isNull()
        self.empty.setVisible(not has_image)
        self.view.setVisible(has_image)
        self.controls.setEnabled(has_image)
        if layers:
            description += " with result graphics"
        self.source_label.setText(
            f"{description} · {image.width()} × {image.height()} px" if has_image else ""
        )
        if same_ticket and image.size() == self.view.image_item.pixmap().size():
            self.view.image_item.setPixmap(image)
        else:
            self.view.set_image(image)
        self.update_verdict(row)

    def update_verdict(self, row):
        state = row["state"]
        self.setWindowTitle(f"{row['filename']} · Ticket {self.ticket['value']} · {state.upper()}")
        self.frame.setStyleSheet(
            f"QFrame#filmstripImageFrame {{ border: 3px solid {COLORS[state]}; }}"
        )


class FilmstripViewer(QWidget):
    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.provider, self.seen = provider, None
        self.rows = []
        self.popout = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.capture = QCheckBox("Capture result thumbnails")
        self.capture.setChecked(True)
        self.capture.setToolTip("Adds thumbnail encoding work. Off by default in Lean.")
        row.addWidget(self.capture)
        self.summary = QLabel("Green: pass · red: fail · purple: error · amber: pending")
        row.addWidget(self.summary, 1)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.provider().clear())
        row.addWidget(clear)
        layout.addLayout(row)
        self.items = QListWidget()
        self.items.setViewMode(QListView.ViewMode.IconMode)
        self.items.setFlow(QListView.Flow.LeftToRight)
        self.items.setWrapping(False)
        self.items.setMovement(QListView.Movement.Static)
        self.items.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.items.setIconSize(QSize(160, 100))
        self.items.setGridSize(QSize(180, 144))
        self.items.setMinimumHeight(160)
        self.items.setItemDelegate(VerdictDelegate(self.items))
        self.items.itemDoubleClicked.connect(self.open_image)
        layout.addWidget(self.items)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)

    def refresh(self):
        store = self.provider()
        revision = (id(store), store.revision)
        if revision == self.seen:
            return
        if self.popout is not None and self.seen[0] != revision[0]:
            self.popout.close()
        self.seen = revision
        selected = self.items.currentRow()
        scroll = self.items.horizontalScrollBar()
        following = scroll.value() == scroll.maximum()
        self.items.clear()
        self.rows = store.snapshot()
        for row in self.rows:
            item = QListWidgetItem(f"{row['filename']}\n{row['state'].upper()}")
            image = QImage.fromData(row["thumbnail"])
            if not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            item.setData(Qt.ItemDataRole.UserRole, row["state"])
            item.setToolTip(
                f"{row['path']}\nTicket: {row['ticket']['value']}\n{row['state']}\n"
                f"{row['reasons']}\nDouble-click to open image"
            )
            self.items.addItem(item)
            if self.popout is not None and row["ticket"] == self.popout.ticket:
                if row["image_revision"] != self.popout.image_revision:
                    self.popout.open_entry(row)
                else:
                    self.popout.update_verdict(row)
        if selected >= 0:
            self.items.setCurrentRow(min(selected, self.items.count() - 1))
        if following:
            scroll.setValue(scroll.maximum())

    def open_image(self, item):
        row = self.rows[self.items.row(item)]
        if self.popout is None:
            self.popout = FilmstripImageWindow(self)
            self.popout.finished.connect(self.popout_closed)
        if self.popout.ticket != row["ticket"]:
            self.popout.open_entry(row)
        self.popout.show()
        self.popout.raise_()
        self.popout.activateWindow()

    def popout_closed(self):
        self.popout = None
