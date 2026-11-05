"""Public terminal editor shared by nested ToolBlock instances."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .toolblocks import Binding


class InterfaceDialog(QDialog):
    def __init__(self, toolblock, parent_graph, parent=None, owner=None):
        super().__init__(parent)
        self.toolblock, self.parent_graph = toolblock, parent_graph
        self.owner = owner
        self.setWindowTitle(f"Public terminals · {toolblock.name}")
        self.resize(760, 440)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Choose the internal ports available on the outer node. "
            "Connected terminals may be renamed; disconnect wires before removing them."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Direction", "Public name", "Internal port"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 210)
        layout.addWidget(self.table)
        for direction, bindings in (
            ("inputs", toolblock.input_bindings),
            ("outputs", toolblock.output_bindings),
        ):
            for name, binding in bindings.items():
                self.add_terminal(direction, name, binding)
        actions = QHBoxLayout()
        for label, direction in (("Add input", "inputs"), ("Add output", "outputs")):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, d=direction: self.add_terminal(d))
            actions.addWidget(button)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        actions.addWidget(remove)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_terminal(self, direction, name="", binding=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        label = QTableWidgetItem(direction[:-1])
        label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
        label.setData(Qt.ItemDataRole.UserRole, direction)
        self.table.setItem(row, 0, label)
        self.table.setItem(row, 1, QTableWidgetItem(name))
        choices = QComboBox()
        for block in self.toolblock.graph.blocks.values():
            ports = block.inputs if direction == "inputs" else block.execution_outputs
            for port in ports:
                connected = any(
                    c.target == block.id and c.input == port.name
                    for c in self.toolblock.graph.connections
                )
                endpoint = Binding(block.id, port.name)
                if direction == "outputs" or not connected or endpoint == binding:
                    choices.addItem(
                        f"{block.name} [{block.id[:8]}] / {port.name} ({port.kind})", endpoint
                    )
        # QVariant comparison does not reliably compare Python dataclass values.
        index = next((i for i in range(choices.count()) if choices.itemData(i) == binding), -1)
        choices.setCurrentIndex(index if index >= 0 else 0)
        self.table.setCellWidget(row, 2, choices)

    def remove_selected(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def apply(self):
        bindings = {"inputs": {}, "outputs": {}}
        try:
            for row in range(self.table.rowCount()):
                direction = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                name = self.table.item(row, 1).text().strip()
                binding = self.table.cellWidget(row, 2).currentData()
                if not name or name in bindings[direction] or binding is None:
                    raise ValueError("Each terminal needs a unique name and an internal port")
                bindings[direction][name] = binding
            self.toolblock.update_interface(
                bindings["inputs"], bindings["outputs"], self.parent_graph, self.owner
            )
            self.accept()
        except ValueError as exc:
            QMessageBox.warning(self, "Public terminals", str(exc))
