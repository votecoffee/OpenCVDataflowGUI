"""Desktop viewer for differences between two project XML snapshots."""

import json
from xml.etree.ElementTree import ParseError

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .project_xml import XML_FILTER, compare_project_xml


class ChangeTable(QAbstractTableModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.changes = []

    def replace(self, changes):
        self.beginResetModel()
        self.changes = changes
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.changes)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ("Change", "Field")[section]
        return None

    def data(self, index, role):
        if index.isValid() and role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            change = self.changes[index.row()]
            return (change.kind, change.path)[index.column()]
        return None


class ProjectCompare(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare project XML")
        self.resize(920, 620)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.before_path = self.file_row(form, "Before")
        self.after_path = self.file_row(form, "After")
        compare = QPushButton("Compare")
        compare.clicked.connect(self.compare)
        form.addRow(compare)
        self.status = QLabel("Choose two XML exports to compare.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(split)
        self.table = QTableView()
        self.model = ChangeTable(self.table)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.selectionModel().currentRowChanged.connect(self.show_change)
        split.addWidget(self.table)
        values = QSplitter()
        self.before_value = self.value_pane(values, "Before")
        self.after_value = self.value_pane(values, "After")
        split.addWidget(values)

    def file_row(self, form, title):
        field = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(field)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self.browse(field, title))
        row.addWidget(button)
        form.addRow(title, row)
        return field

    def browse(self, field, title):
        path, _ = QFileDialog.getOpenFileName(
            self, f"{title} project XML", field.text(), XML_FILTER
        )
        if path:
            field.setText(path)

    @staticmethod
    def value_pane(split, title):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel(title))
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        layout.addWidget(editor)
        split.addWidget(panel)
        return editor

    def compare(self):
        before, after = self.before_path.text(), self.after_path.text()
        self.model.replace([])
        self.show_change(QModelIndex())
        if not before or not after:
            self.status.setText("Choose both XML exports.")
            return
        try:
            changes = compare_project_xml(before, after)
        except (OSError, ParseError, ValueError, RecursionError) as exc:
            self.status.setText(f"Comparison failed. {exc}")
            return
        self.model.replace(changes)
        self.status.setText(f"Differences: {len(changes)}" if changes else "No differences found.")
        if changes:
            self.table.selectRow(0)

    def show_change(self, index, *_):
        if not index.isValid():
            self.before_value.clear()
            self.after_value.clear()
            return
        change = self.model.changes[index.row()]
        self.before_value.setPlainText(
            "Absent" if change.kind == "Added" else json.dumps(change.before, indent=2)
        )
        self.after_value.setPlainText(
            "Absent" if change.kind == "Removed" else json.dumps(change.after, indent=2)
        )
