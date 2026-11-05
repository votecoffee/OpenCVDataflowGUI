"""Validation image-set editor and background report runner."""

import copy
import json
import os
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .validation import EXPECTATIONS, run_set, validate_sets


class ValidationWorker(QThread):
    progress = Signal(object)

    def __init__(self, definition, manifest, registry, project_dir):
        super().__init__()
        self.arguments = definition, manifest, registry, project_dir
        self.stop = threading.Event()
        self.report, self.error = None, ""

    def run(self):
        try:
            self.report = run_set(
                *self.arguments, cancelled=self.stop.is_set, progress=self.progress.emit
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


class ValidationManager(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.catalog = copy.deepcopy(window.root_graph.validation_sets)
        self.current = -1
        self.worker = None
        self.report = None
        self.closing = False
        self.setWindowTitle("Validation image sets")
        self.resize(1150, 800)
        layout = QVBoxLayout(self)
        self.editor = QWidget()
        edit = QVBoxLayout(self.editor)
        row = QHBoxLayout()
        self.sets = QComboBox()
        self.sets.currentIndexChanged.connect(self.load_set)
        row.addWidget(self.sets, 1)
        for title, callback in (
            ("New set", self.new_set),
            ("Remove set", self.remove_set),
            ("Save set", self.save_set),
            ("Import set…", self.import_set),
            ("Export set…", self.export_set),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        edit.addLayout(row)
        form = QFormLayout()
        self.name = QLineEdit()
        self.source, self.judgment, self.state = QComboBox(), QComboBox(), QComboBox()
        for block in window.root_graph.blocks.values():
            if block.ticket_source and any(p.name == "image" for p in block.outputs):
                self.source.addItem(f"{block.name} [{block.id[:8]}]", block.id)
        self.judgment.addItem("Automatic terminal results", "")
        for block in window.root_graph.blocks.values():
            if block.type_id == "inspect.final_judgment":
                self.judgment.addItem(f"{block.name} [{block.id[:8]}]", block.id)
        self.state.addItems(["isolated", "sequence"])
        self.state.setToolTip(
            "Isolated starts fresh for every image; sequence preserves state in table order."
        )
        self.outputs = QCheckBox("Enable output blocks during validation")
        for label, field in (
            ("Name", self.name),
            ("Image source to replace", self.source),
            ("Decision", self.judgment),
            ("Image state", self.state),
        ):
            form.addRow(label, field)
        form.addRow(self.outputs)
        edit.addLayout(form)
        self.images = QTableWidget(0, 4)
        self.images.setHorizontalHeaderLabels(
            ["Image path (project relative)", "Expectation", "Preferred fail reason code", "Notes"]
        )
        self.images.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((380, 130, 240)):
            self.images.setColumnWidth(column, width)
        edit.addWidget(self.images)
        row = QHBoxLayout()
        for title, callback in (
            ("Add images…", self.add_images),
            ("Remove images", self.remove_images),
            ("Move up", lambda: self.move(-1)),
            ("Move down", lambda: self.move(1)),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        edit.addLayout(row)
        layout.addWidget(self.editor, 3)
        self.status = QLabel(
            "Save set stores definitions in the project; Save project retains them on disk."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.run_button, self.stop_button = QPushButton("Run image set"), QPushButton("Stop")
        self.run_button.clicked.connect(self.run_images)
        self.stop_button.clicked.connect(self.stop_run)
        self.stop_button.setEnabled(False)
        row.addWidget(self.run_button)
        row.addWidget(self.stop_button)
        export = QPushButton("Export report…")
        export.clicked.connect(self.export_report)
        row.addWidget(export)
        layout.addLayout(row)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results, 2)
        if not self.catalog:
            self.new_set()
        else:
            self.populate(0)

    def read_set(self):
        if self.current < 0:
            return None
        cases = []
        old = self.catalog[self.current]
        for index in range(self.images.rowCount()):
            cells = [self.images.item(index, c).text() for c in (0, 2, 3)]
            cases.append(
                {
                    "id": self.images.item(index, 0).data(256),
                    "path": cells[0],
                    "expectation": self.images.cellWidget(index, 1).currentText(),
                    "preferred_fail_reason": cells[1],
                    "notes": cells[2],
                }
            )
        return {
            "id": old["id"],
            "name": self.name.text(),
            "source": self.source.currentData() or "",
            "judgment": self.judgment.currentData() or "",
            "state_mode": self.state.currentText(),
            "allow_outputs": self.outputs.isChecked(),
            "cases": cases,
        }

    def stash(self):
        if 0 <= self.current < len(self.catalog):
            self.catalog[self.current] = self.read_set()

    def populate(self, index):
        self.sets.blockSignals(True)
        self.sets.clear()
        self.sets.addItems([s["name"] for s in self.catalog])
        self.sets.setCurrentIndex(index)
        self.sets.blockSignals(False)
        self.current = -1
        self.load_set(index)

    def load_set(self, index):
        self.stash()
        self.current = index
        self.images.setRowCount(0)
        if index < 0:
            return
        item = self.catalog[index]
        self.name.setText(item["name"])
        for field, key in ((self.source, "source"), (self.judgment, "judgment")):
            found = field.findData(item.get(key, ""))
            field.setCurrentIndex(found)
        self.state.setCurrentText(item.get("state_mode", "isolated"))
        self.outputs.setChecked(item.get("allow_outputs", False))
        for case in item["cases"]:
            self.append_case(case)

    def append_case(self, case):
        row = self.images.rowCount()
        self.images.insertRow(row)
        cell = QTableWidgetItem(case["path"])
        cell.setData(256, case["id"])
        self.images.setItem(row, 0, cell)
        expectation = QComboBox()
        expectation.addItems(EXPECTATIONS)
        expectation.setCurrentText(case["expectation"])
        self.images.setCellWidget(row, 1, expectation)
        self.images.setItem(row, 2, QTableWidgetItem(case.get("preferred_fail_reason", "")))
        self.images.setItem(row, 3, QTableWidgetItem(case.get("notes", "")))

    def new_set(self):
        self.stash()
        self.catalog.append(
            {
                "id": uuid.uuid4().hex,
                "name": f"Image set {len(self.catalog) + 1}",
                "source": self.source.currentData() or "",
                "judgment": "",
                "cases": [],
                "state_mode": "isolated",
                "allow_outputs": False,
            }
        )
        self.populate(len(self.catalog) - 1)

    def remove_set(self):
        if self.current >= 0:
            self.catalog.pop(self.current)
            self.populate(min(self.current, len(self.catalog) - 1))

    def save_set(self):
        try:
            self.stash()
            self.window.root_graph.validation_sets = validate_sets(self.catalog)
            for index, item in enumerate(self.catalog):
                self.sets.setItemText(index, item["name"])
            self.window.mark_dirty()
            self.status.setText(
                "Validation definitions updated; Save project to keep them on disk."
            )
            return True
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return False

    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add validation images",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        for path in paths:
            try:
                path = os.path.relpath(path, self.window.root_graph.project_dir or Path.cwd())
            except ValueError:
                pass
            self.append_case({"id": uuid.uuid4().hex, "path": path, "expectation": "unknown"})

    def remove_images(self):
        for row in sorted({i.row() for i in self.images.selectedIndexes()}, reverse=True):
            self.images.removeRow(row)

    def move(self, offset):
        row = self.images.currentRow()
        if row >= 0 and 0 <= row + offset < self.images.rowCount():
            item = self.read_set()
            item["cases"][row], item["cases"][row + offset] = (
                item["cases"][row + offset],
                item["cases"][row],
            )
            self.catalog[self.current] = item
            self.images.setRowCount(0)
            for case in item["cases"]:
                self.append_case(case)
            self.images.setCurrentCell(row + offset, 0)

    def import_set(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import validation set", filter="JSON (*.json)")
        if path:
            try:
                item = validate_sets([json.loads(Path(path).read_text("utf-8"))])[0]
                for case in item["cases"]:
                    image = Path(case["path"])
                    if not image.is_absolute():
                        case["path"] = str((Path(path).resolve().parent / image).resolve())
                self.stash()
                item["id"] = uuid.uuid4().hex
                self.catalog.append(item)
                self.populate(len(self.catalog) - 1)
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def export_set(self):
        if not self.save_set() or self.current < 0:
            return
        item = copy.deepcopy(self.catalog[self.current])
        for case in item["cases"]:
            path = Path(case["path"])
            if not path.is_absolute():
                case["path"] = str(
                    ((self.window.root_graph.project_dir or Path.cwd()) / path).resolve()
                )
        self.write_json("Export validation set", item)

    def write_json(self, title, value):
        path, _ = QFileDialog.getSaveFileName(self, title, filter="JSON (*.json)")
        if path:
            try:
                with Path(path).open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def run_images(self):
        if self.worker or not self.save_set() or self.current < 0:
            return
        item = self.catalog[self.current]
        if not item["cases"]:
            self.status.setText("Add images before running the set")
            return
        self.report = None
        self.results.clear()
        self.worker = ValidationWorker(
            self.window.root_graph.to_dict(),
            copy.deepcopy(item),
            self.window.registry,
            self.window.root_graph.project_dir,
        )
        self.worker.progress.connect(self.on_case)
        self.worker.finished.connect(self.finished_run)
        self.editor.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Running validation on a separate program instance…")
        self.worker.start()

    def on_case(self, row):
        actual = (
            "PASS" if row["actual"] is True else "FAIL" if row["actual"] is False else "UNRESOLVED"
        )
        matched = (
            "unscored" if row["matched"] is None else "match" if row["matched"] else "mismatch"
        )
        self.results.append(
            f"{row['path']} · expected {row['expectation'].replace('_', ' ')} · "
            f"{actual} · {matched}\n"
            + json.dumps({"reasons": row["fail_reasons"], "errors": row["errors"]})
        )

    def stop_run(self):
        if self.worker:
            self.worker.stop.set()
            self.status.setText("Stopping after the current tool returns…")

    def finished_run(self):
        worker = self.worker
        worker.wait()
        self.report = worker.report
        if worker.error:
            self.status.setText(worker.error)
        else:
            summary = self.report["summary"]
            score = (
                f"{summary['score_percent']:.1f}%"
                if summary["score_percent"] is not None
                else "unscored"
            )
            self.status.setText(
                f"{self.report['set']['name']} · Score {score} · "
                f"Required checks {'PASS' if summary['passed'] else 'FAIL'} · "
                f"{summary['matched']}/{summary['scored']} expected matches · "
                f"{summary['advisory_mismatches']} advisory misses · "
                f"{summary['unknown']} unknown · "
                f"{summary['execution_errors']} execution errors"
            )
        self.worker = None
        worker.deleteLater()
        self.editor.setEnabled(True)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.closing:
            self.close()

    def export_report(self):
        if self.report:
            self.write_json("Export validation report", self.report)

    def closeEvent(self, event):
        if self.worker:
            self.closing = True
            self.stop_run()
            event.ignore()
        else:
            super().closeEvent(event)
