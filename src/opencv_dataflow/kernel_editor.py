"""Visual structuring-element editor for morphology blocks."""

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSignalBlocker, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableView,
    QVBoxLayout,
)

KERNEL_PARAMETERS = ("shape", "kernel", "kernel_y", "shape_anchor", "custom_kernel")
SHAPE_LABELS = {
    "rect": "Rectangle",
    "ellipse": "Ellipse",
    "cross": "Cross",
    "diamond": "Diamond",
    "custom": "Custom",
}


class KernelGrid(QAbstractTableModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.cells = np.empty((0, 0), dtype=np.uint8)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.cells.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.cells.shape[1]

    def data(self, index, role):
        if not index.isValid():
            return None
        active = self.cells[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(active)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if active and role == Qt.ItemDataRole.BackgroundRole:
            return QColor("#28557b")
        if active and role == Qt.ItemDataRole.ForegroundRole:
            return QColor("#ffffff")
        return None

    def replace(self, cells):
        self.beginResetModel()
        self.cells = cells
        self.endResetModel()

    def toggle(self, index):
        self.cells[index.row(), index.column()] ^= 1
        self.dataChanged.emit(index, index)


class KernelEditor(QDialog):
    def __init__(self, block, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Morphology kernel")
        self.resize(460, 480)
        self._original = parameters
        self._make_kernel = block.kernel_matrix
        try:
            self._custom = self._make_kernel({**parameters, "shape": "custom"})
        except (TypeError, ValueError):
            self._custom = np.empty((0, 0), dtype=np.uint8)
        if parameters["shape"] == "custom":
            initial = self._custom if self._custom.size else np.zeros((3, 3), dtype=np.uint8)
        else:
            initial = self._make_kernel(parameters)
        height, width = initial.shape
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.shape = QComboBox()
        for shape in block.parameter_choices["shape"]:
            self.shape.addItem(SHAPE_LABELS[shape], shape)
        self.shape.setCurrentIndex(self.shape.findData(parameters["shape"]))
        form.addRow("Shape", self.shape)
        self.width = self.spin(1, 255, width)
        self.height = self.spin(1, 255, height)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        x, y = parameters["shape_anchor"]
        self.anchor_x = self.spin(-1, width - 1, x)
        self.anchor_y = self.spin(-1, height - 1, y)
        anchors = QHBoxLayout()
        anchors.addWidget(self.anchor_x)
        anchors.addWidget(self.anchor_y)
        form.addRow("Cross origin (X, Y)", anchors)
        self.anchor_x.setSpecialValueText("Centered")
        self.anchor_y.setSpecialValueText("Centered")
        self.table = QTableView()
        self.grid = KernelGrid(self.table)
        self.table.setModel(self.grid)
        for header in (self.table.horizontalHeader(), self.table.verticalHeader()):
            header.hide()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            header.setDefaultSectionSize(27)
        self.table.clicked.connect(self.toggle)
        space = QShortcut(QKeySequence(Qt.Key.Key_Space), self.table)
        space.setContext(Qt.ShortcutContext.WidgetShortcut)
        space.activated.connect(lambda: self.toggle(self.table.currentIndex()))
        layout.addWidget(self.table)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText("Use kernel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.grid.replace(initial)
        self.shape.currentIndexChanged.connect(self.change_shape)
        for control in (self.width, self.height, self.anchor_x, self.anchor_y):
            control.valueChanged.connect(self.refresh)
        self.refresh()

    @staticmethod
    def spin(low, high, value):
        control = QSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        return control

    @property
    def parameters(self):
        shape = self.shape.currentData()
        values = {key: self._original[key] for key in KERNEL_PARAMETERS}
        values["shape"] = shape
        values["custom_kernel"] = self._custom.tolist()
        if shape != "custom":
            width, height = self.width.value(), self.height.value()
            values.update(
                kernel=width,
                kernel_y=0 if self._original["kernel_y"] == 0 and width == height else height,
                shape_anchor=[self.anchor_x.value(), self.anchor_y.value()],
            )
        return values

    def change_shape(self):
        if self.shape.currentData() == "custom" and self._custom.size:
            height, width = self._custom.shape
            with QSignalBlocker(self.width), QSignalBlocker(self.height):
                self.width.setValue(width)
                self.height.setValue(height)
        self.refresh()

    def refresh(self):
        width, height = self.width.value(), self.height.value()
        shape = self.shape.currentData()
        for control, size in ((self.anchor_x, width), (self.anchor_y, height)):
            with QSignalBlocker(control):
                control.setMaximum(size - 1)
            control.setEnabled(shape == "cross")
        if shape == "custom":
            source = self._custom if self._custom.size else self.grid.cells
            cells = np.zeros((height, width), dtype=np.uint8)
            rows, cols = min(height, source.shape[0]), min(width, source.shape[1])
            cells[:rows, :cols] = source[:rows, :cols]
            self._custom = cells
        else:
            cells = self._make_kernel(self.parameters)
        self.grid.replace(cells)
        self.refresh_status()

    def toggle(self, index):
        if index.isValid() and self.shape.currentData() == "custom":
            self.grid.toggle(index)
            self.refresh_status()

    def refresh_status(self):
        custom = self.shape.currentData() == "custom"
        valid = not custom or bool(self._custom.any())
        self.ok.setEnabled(valid)
        if not valid:
            self.status.setText("Turn on at least one cell.")
        elif custom:
            self.status.setText("Click or press Space to toggle. 1 = active, 0 = ignored.")
        else:
            self.status.setText("Select Custom to edit individual cells.")
