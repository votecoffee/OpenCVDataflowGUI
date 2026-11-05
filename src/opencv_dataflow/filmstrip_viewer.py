"""Horizontal dockable results view, with verdict borders and ticket details."""

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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

COLORS = {"pass": "#2cb978", "fail": "#ef5350", "error": "#cc78e8", "pending": "#edba50"}


class VerdictDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLORS[index.data(Qt.ItemDataRole.UserRole)]), 3))
        painter.drawRoundedRect(option.rect.adjusted(3, 3, -3, -3), 4, 4)
        painter.restore()


class FilmstripViewer(QWidget):
    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.provider, self.seen = provider, None
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
        layout.addWidget(self.items)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)

    def refresh(self):
        store = self.provider()
        revision = (id(store), store.revision)
        if revision == self.seen:
            return
        self.seen = revision
        selected = self.items.currentRow()
        scroll = self.items.horizontalScrollBar()
        following = scroll.value() == scroll.maximum()
        self.items.clear()
        for row in store.snapshot():
            item = QListWidgetItem(f"{row['filename']}\n{row['state'].upper()}")
            image = QImage.fromData(row["thumbnail"])
            if not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))
            item.setData(Qt.ItemDataRole.UserRole, row["state"])
            item.setToolTip(
                f"{row['path']}\nTicket: {row['ticket']['value']}\n{row['state']}\n{row['reasons']}"
            )
            self.items.addItem(item)
        if selected >= 0:
            self.items.setCurrentRow(min(selected, self.items.count() - 1))
        if following:
            scroll.setValue(scroll.maximum())
