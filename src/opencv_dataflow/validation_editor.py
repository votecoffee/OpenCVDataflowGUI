"""Validation image-set editor and background report runner."""

import copy
import json
import os
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .classification_evaluation import CLASSIFICATION_DEFAULTS
from .filmstrip import Filmstrip
from .filmstrip_viewer import FilmstripViewer
from .validation import EXPECTATIONS, run_set, validate_sets


class ValidationWorker(QThread):
    progress = Signal(object)

    def __init__(self, definition, manifest, registry, project_dir, filmstrip=None):
        super().__init__()
        self.arguments = definition, manifest, registry, project_dir
        self.filmstrip = filmstrip
        self.stop = threading.Event()
        self.report, self.error = None, ""

    def run(self):
        try:
            self.report = run_set(
                *self.arguments,
                cancelled=self.stop.is_set,
                progress=self.progress.emit,
                filmstrip=self.filmstrip,
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
        self.filmstrip = Filmstrip()
        self.filmstrip_view = FilmstripViewer(lambda: self.filmstrip, self)
        self.filmstrip_view.capture.setChecked(window.root_graph.run_profile.name != "lean")
        layout.addWidget(self.filmstrip_view)
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
        settings_tabs = QTabWidget()
        image_settings = QWidget()
        image_layout = QVBoxLayout(image_settings)
        image_layout.addLayout(form)
        settings_tabs.addTab(image_settings, "Image set")
        settings_tabs.addTab(self.classification_fields(), "Classifier evaluation")
        edit.addWidget(settings_tabs)
        self.images = QTableWidget(0, 7)
        self.images.setHorizontalHeaderLabels(
            [
                "Image path (project relative)",
                "Expectation",
                "Preferred fail reason code",
                "Notes",
                "Expected class",
                "Capture group",
                "Image SHA-256 (optional)",
            ]
        )
        self.images.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((380, 130, 240)):
            self.images.setColumnWidth(column, width)
        edit.addWidget(self.images)
        self.class_enabled.toggled.connect(self.classification_visibility)
        self.class_advanced.toggled.connect(self.classification_visibility)
        self.classification_visibility()
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
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self.results, "Events")
        class_results = QWidget()
        class_layout = QVBoxLayout(class_results)
        self.class_summary = QLabel()
        self.confusion = QTableWidget()
        self.class_metrics = QTableWidget()
        for table in (self.confusion, self.class_metrics):
            table.setEditTriggers(QTableWidget.NoEditTriggers)
        class_layout.addWidget(self.class_summary)
        class_layout.addWidget(self.confusion)
        class_layout.addWidget(self.class_metrics)
        self.result_tabs.addTab(class_results, "Classifier report")
        self.result_tabs.setTabVisible(1, False)
        layout.addWidget(self.result_tabs, 2)
        if not self.catalog:
            self.new_set()
        else:
            self.populate(0)

    def classification_fields(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.class_enabled = QCheckBox("Evaluate classifier labels")
        layout.addWidget(self.class_enabled)
        self.class_options = QWidget()
        form = QFormLayout(self.class_options)
        self.class_block = QComboBox()
        for block in self.window.root_graph.blocks.values():
            if any(p.kind in ("data", "text") for p in block.outputs):
                self.class_block.addItem(f"{block.name} [{block.id[:8]}]", block.id)
        self.class_output, self.class_path = QLineEdit("labels"), QLineEdit("0")
        self.class_path.setToolTip(
            "Dotted fields/indices; 0 selects the only label in a list. "
            "Blank uses the scalar output."
        )
        self.class_accuracy, self.class_coverage = QDoubleSpinBox(), QDoubleSpinBox()
        for field in (self.class_accuracy, self.class_coverage):
            field.setRange(0, 100)
            field.setSuffix(" %")
            field.setValue(100)
        self.class_accuracy.setToolTip(
            "Correct / all labeled images, including rejected, missing and errors."
        )
        self.class_coverage.setToolTip(
            "Recognized labels / all labeled images, including wrong recognized labels."
        )
        for label, field in (
            ("Classifier block", self.class_block),
            ("Label output", self.class_output),
            ("Value within output", self.class_path),
            ("Minimum accuracy", self.class_accuracy),
            ("Minimum coverage", self.class_coverage),
        ):
            form.addRow(label, field)
        self.class_advanced = QCheckBox("Show score and capture-group settings")
        form.addRow(self.class_advanced)
        self.class_extra = QWidget()
        extra = QFormLayout(self.class_extra)
        self.class_order, self.training_groups = QPlainTextEdit(), QPlainTextEdit()
        for field in (self.class_order, self.training_groups):
            field.setMaximumHeight(55)
            field.setPlaceholderText("One per line")
        self.training_groups.setToolTip(
            "Training groups must not occur in evaluation. "
            "Group labels do not prove model training provenance."
        )
        self.score_output, self.score_path, self.score_kind = (
            QLineEdit(),
            QLineEdit(),
            QLineEdit("score"),
        )
        self.score_path.setToolTip(
            "For example winning_vote_fractions.0 in a forest Report output. "
            "A data lookup, not an expression."
        )
        self.score_direction = QComboBox()
        self.score_direction.addItems(["higher", "lower"])
        self.thresholds = QLineEdit()
        self.thresholds.setPlaceholderText("For example: 0.5, 0.8, 0.9")
        for label, field in (
            ("Class display order", self.class_order),
            ("Training capture groups", self.training_groups),
            ("Score output (optional)", self.score_output),
            ("Value within score output", self.score_path),
            ("Score meaning", self.score_kind),
            ("Better scores are", self.score_direction),
            ("Score cutoffs", self.thresholds),
        ):
            extra.addRow(label, field)
        form.addRow(self.class_extra)
        layout.addWidget(self.class_options)
        return widget

    def classification_visibility(self):
        enabled = self.class_enabled.isChecked()
        self.class_options.setEnabled(enabled)
        self.class_extra.setVisible(enabled and self.class_advanced.isChecked())
        for column in (4, 5):
            self.images.setColumnHidden(column, not enabled)
        self.images.setColumnHidden(6, not (enabled and self.class_advanced.isChecked()))

    def read_set(self):
        if self.current < 0:
            return None
        cases = []
        old = self.catalog[self.current]
        for index in range(self.images.rowCount()):
            cells = [self.images.item(index, c).text() for c in (0, 2, 3)]
            case = copy.deepcopy(self.images.item(index, 0).data(257) or {})
            case.update(
                {
                    "id": self.images.item(index, 0).data(256),
                    "path": cells[0],
                    "expectation": self.images.cellWidget(index, 1).currentText(),
                    "preferred_fail_reason": cells[1],
                    "notes": cells[2],
                }
            )
            if (
                self.class_enabled.isChecked()
                or "class_label" in case
                or self.images.item(index, 4).text()
            ):
                case["class_label"] = self.images.item(index, 4).text() or None
            for column, key in ((5, "capture_group"), (6, "image_sha256")):
                value = self.images.item(index, column).text()
                if value:
                    case[key] = value
                else:
                    case.pop(key, None)
            cases.append(case)
        result = {
            "id": old["id"],
            "name": self.name.text(),
            "source": self.source.currentData() or "",
            "judgment": self.judgment.currentData() or "",
            "state_mode": self.state.currentText(),
            "allow_outputs": self.outputs.isChecked(),
            "cases": cases,
        }
        if self.class_enabled.isChecked():
            raw = [v.strip() for v in self.thresholds.text().split(",") if v.strip()]
            try:
                thresholds = [float(v) for v in raw]
            except ValueError:
                thresholds = raw  # Keep the draft; Save/Run displays its validation error.
            result["classification"] = {
                "block": self.class_block.currentData() or "",
                "output": self.class_output.text(),
                "value_path": self.class_path.text(),
                "classes": self.class_order.toPlainText().splitlines(),
                "training_groups": self.training_groups.toPlainText().splitlines(),
                "min_accuracy": self.class_accuracy.value() / 100,
                "min_coverage": self.class_coverage.value() / 100,
                "score_output": self.score_output.text(),
                "score_path": self.score_path.text(),
                "score_kind": self.score_kind.text(),
                "score_direction": self.score_direction.currentText(),
                "thresholds": thresholds,
            }
        return result

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
        config = {**copy.deepcopy(CLASSIFICATION_DEFAULTS), **item.get("classification", {})}
        self.class_enabled.setChecked("classification" in item)
        self.class_block.setCurrentIndex(self.class_block.findData(config["block"]))
        for field, key in (
            (self.class_output, "output"),
            (self.class_path, "value_path"),
            (self.score_output, "score_output"),
            (self.score_path, "score_path"),
            (self.score_kind, "score_kind"),
        ):
            field.setText(config[key])
        self.class_order.setPlainText("\n".join(config["classes"]))
        self.training_groups.setPlainText("\n".join(config["training_groups"]))
        self.class_accuracy.setValue(config["min_accuracy"] * 100)
        self.class_coverage.setValue(config["min_coverage"] * 100)
        self.score_direction.setCurrentText(config["score_direction"])
        self.thresholds.setText(", ".join(map(str, config["thresholds"])))
        self.classification_visibility()
        for case in item["cases"]:
            self.append_case(case)

    def append_case(self, case):
        row = self.images.rowCount()
        self.images.insertRow(row)
        cell = QTableWidgetItem(case["path"])
        cell.setData(256, case["id"])
        cell.setData(257, copy.deepcopy(case))
        self.images.setItem(row, 0, cell)
        expectation = QComboBox()
        expectation.addItems(EXPECTATIONS)
        expectation.setCurrentText(case["expectation"])
        self.images.setCellWidget(row, 1, expectation)
        self.images.setItem(row, 2, QTableWidgetItem(case.get("preferred_fail_reason", "")))
        self.images.setItem(row, 3, QTableWidgetItem(case.get("notes", "")))
        for column, key in ((4, "class_label"), (5, "capture_group"), (6, "image_sha256")):
            self.images.setItem(row, column, QTableWidgetItem(case.get(key) or ""))

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
        self.result_tabs.setTabVisible(1, False)
        self.results.clear()
        self.filmstrip.clear()
        self.worker = ValidationWorker(
            self.window.root_graph.to_dict(),
            copy.deepcopy(item),
            self.window.registry,
            self.window.root_graph.project_dir,
            self.filmstrip if self.filmstrip_view.capture.isChecked() else None,
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
            self.show_classification_report(self.report.get("classification"))
        self.worker = None
        worker.deleteLater()
        self.editor.setEnabled(True)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.closing:
            self.close()

    def show_classification_report(self, report):
        self.result_tabs.setTabVisible(1, report is not None)
        if report is None:
            return
        summary, classes = report["summary"], report["classes"]

        def percent(value):
            return f"{100 * value:.1f}%" if value is not None else "unscored"

        message = (
            f"Class accuracy {percent(summary['accuracy'])} · "
            f"coverage {percent(summary['coverage'])} · "
            f"{summary['correct']}/{summary['labeled']} correct · "
            f"{summary['unlabeled']} unlabeled"
        )
        self.class_summary.setText(message)
        self.status.setText(self.status.text() + " · " + message)
        self.confusion.setRowCount(len(classes))
        self.confusion.setColumnCount(len(classes) + 3)
        self.confusion.setVerticalHeaderLabels(classes)
        self.confusion.setHorizontalHeaderLabels(
            classes + ["Rejected (no label)", "Missing", "Error"]
        )
        for i, row in enumerate(report["confusion"]):
            for j, count in enumerate(row):
                cell = QTableWidgetItem(str(count))
                if count:
                    cell.setBackground(
                        QColor(
                            "#d5f2d6" if i == j else "#f5d6d6" if j >= len(classes) else "#fff0bb"
                        )
                    )
                self.confusion.setItem(i, j, cell)
        self.class_metrics.setRowCount(len(classes))
        self.class_metrics.setColumnCount(5)
        self.class_metrics.setHorizontalHeaderLabels(
            ["Class", "Support", "Precision", "Recall", "F1"]
        )
        for i, row in enumerate(report["per_class"]):
            values = [
                row["label"],
                str(row["support"]),
                *[percent(row[key]) for key in ("precision", "recall", "f1")],
            ]
            for j, value in enumerate(values):
                self.class_metrics.setItem(i, j, QTableWidgetItem(value))

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
