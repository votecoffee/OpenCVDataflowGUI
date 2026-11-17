"""Optional detector validation settings, box labels and report tables."""

import copy

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .detection_evaluation import DEFAULTS, truth


def decimal(low, high, value, decimals=6):
    field = QDoubleSpinBox()
    field.setDecimals(decimals)
    field.setRange(low, high)
    field.setValue(value)
    return field


def cutoffs(text):
    raw = [v.strip() for v in text.split(",") if v.strip()]
    try:
        return [float(v) for v in raw]
    except ValueError:
        return raw  # Save/Run owns draft validation, never a Qt signal callback.


class DetectionSettings(QWidget):
    changed = Signal()

    def __init__(self, graph):
        super().__init__()
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Evaluate detector object boxes")
        layout.addWidget(self.enabled)
        self.options = QWidget()
        form = QFormLayout(self.options)
        self.block, self.output = QComboBox(), QLineEdit("collection")
        for block in graph.blocks.values():
            if any(p.kind in ("data", "any") for p in block.outputs):
                self.block.addItem(f"{block.name} [{block.id[:8]}]", block.id)
        self.iou = decimal(0.000001, 1, 0.5)
        self.score = decimal(-1e12, 1e12, 0)
        self.precision, self.recall = decimal(0, 100, 100, 2), decimal(0, 100, 100, 2)
        self.precision.setSuffix(" %")
        self.recall.setSuffix(" %")
        self.iou.setToolTip(
            "Class-aware one-to-one matching of continuous source-pixel XYWH boxes."
        )
        self.score.setToolTip("Keep scores >= cutoff. AP uses all retained upstream predictions.")
        for label, field in (
            ("Detector block", self.block),
            ("Collection output", self.output),
            ("Matching IoU", self.iou),
            ("Operating score cutoff", self.score),
            ("Minimum precision", self.precision),
            ("Minimum recall", self.recall),
        ):
            form.addRow(label, field)
        self.advanced = QCheckBox("Show AP, score and capture-group settings")
        form.addRow(self.advanced)
        self.extra = QWidget()
        extra = QFormLayout(self.extra)
        self.classes, self.training_groups = QPlainTextEdit(), QPlainTextEdit()
        for field in (self.classes, self.training_groups):
            field.setMaximumHeight(50)
            field.setPlaceholderText("One per line")
        self.iou_extra, self.scores = QLineEdit("0.5, 0.75"), QLineEdit()
        self.interpolation = QComboBox()
        self.interpolation.addItem("All-point precision envelope", "all_points")
        self.interpolation.addItem("101 recall samples", "101_points")
        for label, field in (
            ("Class display order", self.classes),
            ("Training capture groups", self.training_groups),
            ("Additional IoU cutoffs", self.iou_extra),
            ("Score sweep cutoffs", self.scores),
            ("AP interpolation", self.interpolation),
        ):
            extra.addRow(label, field)
        form.addRow(self.extra)
        layout.addWidget(self.options)
        help_text = QLabel(
            "Object labels are source-image boxes. An empty labeled image is a "
            "negative; an unlabeled image is excluded from object metrics. Missing/errors fail."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.enabled.toggled.connect(self.visibility)
        self.advanced.toggled.connect(self.visibility)
        self.visibility()

    def visibility(self):
        self.options.setEnabled(self.enabled.isChecked())
        self.extra.setVisible(self.enabled.isChecked() and self.advanced.isChecked())
        self.changed.emit()

    def read(self):
        return {
            "block": self.block.currentData() or "",
            "output": self.output.text(),
            "classes": self.classes.toPlainText().splitlines(),
            "training_groups": self.training_groups.toPlainText().splitlines(),
            "iou_threshold": self.iou.value(),
            "score_threshold": self.score.value(),
            "min_precision": self.precision.value() / 100,
            "min_recall": self.recall.value() / 100,
            "iou_thresholds": cutoffs(self.iou_extra.text()),
            "score_thresholds": cutoffs(self.scores.text()),
            "ap_interpolation": self.interpolation.currentData(),
        }

    def load(self, value):
        p = {**copy.deepcopy(DEFAULTS), **(value or {})}
        self.enabled.setChecked(value is not None)
        self.block.setCurrentIndex(self.block.findData(p["block"]))
        self.output.setText(p["output"])
        for field, key in ((self.iou, "iou_threshold"), (self.score, "score_threshold")):
            field.setValue(p[key])
        self.precision.setValue(100 * p["min_precision"])
        self.recall.setValue(100 * p["min_recall"])
        self.classes.setPlainText("\n".join(p["classes"]))
        self.training_groups.setPlainText("\n".join(p["training_groups"]))
        self.iou_extra.setText(", ".join(map(str, p["iou_thresholds"])))
        self.scores.setText(", ".join(map(str, p["score_thresholds"])))
        self.interpolation.setCurrentIndex(self.interpolation.findData(p["ap_interpolation"]))
        self.visibility()


class DetectionTruthEditor(QDialog):
    def __init__(self, parent, value, size):
        super().__init__(parent)
        self.setWindowTitle("Image object labels — source pixel edges")
        self.resize(740, 440)
        layout = QVBoxLayout(self)
        self.unlabeled = QCheckBox("Unlabeled image — exclude from object metrics")
        self.unlabeled.setChecked(value is None)
        layout.addWidget(self.unlabeled)
        size_row = QHBoxLayout()
        self.width, self.height = QSpinBox(), QSpinBox()
        for name, field, default in zip(
            ("Image width", "Image height"),
            (self.width, self.height),
            (value["size"] if value else size),
            strict=True,
        ):
            field.setRange(1, 16384)
            field.setValue(default)
            size_row.addWidget(QLabel(name))
            size_row.addWidget(field)
        layout.addLayout(size_row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Class", "Left", "Top", "Width", "Height"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setToolTip(
            "Finite XYWH source-pixel edges. Positive sizes; truth stays in image bounds."
        )
        layout.addWidget(self.table)
        for item in value["objects"] if value else []:
            self.append(item)
        row = QHBoxLayout()
        add, remove = QPushButton("Add object"), QPushButton("Remove selected object")
        add.clicked.connect(lambda: self.append())
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)
        self.error = QLabel("No objects with Unlabeled off means a fully labeled negative image.")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.result_truth = copy.deepcopy(value)

    def append(self, item=None):
        index = self.table.rowCount()
        self.table.insertRow(index)
        existing = {self.table.item(i, 0).text() for i in range(index)}
        serial = index + 1
        while str(serial) in existing:
            serial += 1
        item = item or {"id": str(serial), "label": "part", "bbox": [0, 0, 1, 1]}
        for column, value in enumerate([item["id"], item["label"], *item["bbox"]]):
            self.table.setItem(index, column, QTableWidgetItem(str(value)))

    def accept(self):
        try:
            value = (
                None
                if self.unlabeled.isChecked()
                else {
                    "size": [self.width.value(), self.height.value()],
                    "objects": [
                        {
                            "id": self.table.item(i, 0).text(),
                            "label": self.table.item(i, 1).text(),
                            "bbox": [float(self.table.item(i, j).text()) for j in range(2, 6)],
                        }
                        for i in range(self.table.rowCount())
                    ],
                }
            )
            self.result_truth = truth(value)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()


def percent(value):
    return f"{100 * value:.1f}%" if value is not None else "unscored"


class DetectionReport(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.summary, self.classes, self.images = QLabel(), QTableWidget(), QTableWidget()
        self.scores, self.ious = QTableWidget(), QTableWidget()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        for name, table in (
            ("Classes", self.classes),
            ("Images", self.images),
            ("Score sweep", self.scores),
            ("IoU cutoffs", self.ious),
        ):
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            tabs.addTab(table, name)

    def show_report(self, report):
        summary = report["summary"]
        images = {row["id"]: row for row in report["images"]}
        self.summary.setText(
            f"{summary['tp']} TP · {summary['fp']} FP · {summary['fn']} misses · "
            f"precision {percent(summary['precision'])} · recall {percent(summary['recall'])} · "
            f"mAP {percent(summary['map'])} at IoU {report['settings']['iou_threshold']} · "
            f"{summary['unlabeled_images']} unlabeled images"
        )
        for table, headers, rows in (
            (
                self.classes,
                ["Class", "Truth", "TP", "FP", "Missed", "Precision", "Recall", "F1", "AP"],
                [
                    [
                        r["label"],
                        r["support"],
                        r["tp"],
                        r["fp"],
                        r["fn"],
                        *[percent(r[k]) for k in ("precision", "recall", "f1", "ap")],
                    ]
                    for r in report["per_class"]
                ],
            ),
            (
                self.images,
                ["Image", "Status", "Truth", "TP", "FP", "Missed"],
                [
                    [
                        r["id"],
                        r["status"],
                        len((r["truth"] or {}).get("objects", [])),
                        *[images.get(r["id"], {}).get(k, "unscored") for k in ("tp", "fp", "fn")],
                    ]
                    for r in report["records"]
                ],
            ),
            (
                self.scores,
                ["Score >=", "TP", "FP", "Missed", "Precision", "Recall"],
                [
                    [
                        r["score_threshold"],
                        r["tp"],
                        r["fp"],
                        r["fn"],
                        percent(r["precision"]),
                        percent(r["recall"]),
                    ]
                    for r in report["score_sweeps"]
                ],
            ),
            (
                self.ious,
                ["IoU >=", "TP", "FP", "Missed", "Precision", "Recall", "mAP"],
                [
                    [
                        r["iou_threshold"],
                        *[r["summary"][k] for k in ("tp", "fp", "fn")],
                        *[percent(r["summary"][k]) for k in ("precision", "recall", "map")],
                    ]
                    for r in report["iou_reports"]
                ],
            ),
        ):
            table.setRowCount(len(rows))
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    table.setItem(i, j, QTableWidgetItem(str(value)))
        for i, record in enumerate(report["records"]):
            self.images.item(i, 1).setToolTip(record.get("error", ""))
