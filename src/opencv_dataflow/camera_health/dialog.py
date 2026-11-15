"""Modeless capture-history and region trainer using host publication snapshots."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..images import ImageArray, image_format, preview_rgb
from ..payloads import ValuePacket
from ..plugin_image_picker import PluginImagePicker
from ..region_editor import RegionEditor
from .dataflow import CameraHealth
from .model import (
    MAX_SAMPLES,
    ROLES,
    full_region,
    load_reference,
    normalized,
    stage_reference,
    train_reference,
)

_TRAINERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-health-training")


class CameraHealthTrainer(QDialog):
    def __init__(self, parent, parameters, context):
        super().__init__(parent)
        self.setWindowTitle("Train camera health reference")
        self.resize(1100, 780)
        self.parameters, self.context = parameters, context
        self.frames, self.regions = [], {role: [] for role in ROLES}
        self.candidate = self.future = None
        self.closed = False
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Capture/import known-good images with the camera fixed. Draw stable check regions."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        captures, regions, thresholds = QWidget(), QWidget(), QWidget()
        tabs.addTab(captures, "Reference history")
        tabs.addTab(regions, "Regions / fiducials")
        tabs.addTab(thresholds, "Warning / alarm limits")
        capture_layout = QVBoxLayout(captures)
        self.picker = PluginImagePicker(context, self, input_port="image")
        self.picker.captured.connect(self.add_capture)
        capture_layout.addWidget(self.picker)
        row = QHBoxLayout()
        capture_layout.addLayout(row)
        self.import_button = self.button(row, "Import reference images…", self.import_images)
        self.remove_frame_button = self.button(row, "Remove selected capture", self.remove_capture)
        self.capture_list = QListWidget()
        self.capture_list.currentRowChanged.connect(self.refresh_preview)
        capture_layout.addWidget(self.capture_list, 1)
        region_layout = QVBoxLayout(regions)
        row = QHBoxLayout()
        region_layout.addLayout(row)
        self.role = QComboBox()
        self.role.addItems(ROLES)
        self.role.currentTextChanged.connect(self.refresh_regions)
        row.addWidget(self.role)
        self.add_region_button = self.button(row, "Add / draw…", self.add_region)
        self.edit_region_button = self.button(row, "Edit…", self.edit_region)
        self.remove_region_button = self.button(row, "Remove", self.remove_region)
        self.region_list = QListWidget()
        region_layout.addWidget(self.region_list)
        self.method = QComboBox()
        self.method.addItems(["background", "fiducials"])
        self.method.setCurrentText(parameters["position_method"])
        region_layout.addWidget(
            QLabel(
                "Ignore regions exclude expected part paths. Position regions restrict tracking. "
                "One fiducial measures translation; two or more also measure rotation/scale."
            )
        )
        region_layout.addWidget(self.method)
        self.preview = QLabel("Capture an image to draw reference regions.")
        self.preview.setMinimumSize(400, 280)
        region_layout.addWidget(self.preview, 1)
        self.limits = {}
        grid = QGridLayout(thresholds)
        for col, text in enumerate(("Check", "Warning", "Alarm", "Units")):
            grid.addWidget(QLabel(text), 0, col)
        for row, (key, units) in enumerate(
            (
                ("position", "pixels at image center"),
                ("rotation", "degrees"),
                ("scale", "% scale change"),
                ("lighting", "% mean change per region/channel"),
                ("focus", "% sharpness loss"),
            ),
            1,
        ):
            grid.addWidget(QLabel(key.title()), row, 0)
            for col, level in enumerate(("warning", "alarm"), 1):
                field = QDoubleSpinBox()
                field.setRange(0, 10000)
                field.setDecimals(3)
                field.setValue(parameters[f"{key}_{level}"])
                self.limits[f"{key}_{level}"] = field
                grid.addWidget(field, row, col)
            grid.addWidget(QLabel(units), row, 3)
        grid.setRowStretch(6, 1)
        self.report = QTableWidget(0, 4)
        self.report.setHorizontalHeaderLabels(["Check", "Region", "History mean", "History std"])
        self.report.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.report.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.report)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.train_button = self.button(actions, "Train reference", self.start_training)
        self.stage_button = self.button(actions, "Stage reference / limits", self.stage)
        self.button(actions, "Cancel", self.reject)
        self.finished.connect(self.finish)
        if parameters["reference_path"] and context.project_dir is not None:
            try:
                path = Path(context.project_dir) / parameters["reference_path"]
                model = load_reference(path, parameters["reference_sha256"], include_history=True)
                self.regions = copy.deepcopy(model.info["regions"])
                fmt, dtype = model.info["encoding"], np.dtype(model.info["dtype"])
                settings = model.info["settings"]
                lo, hi = (
                    (np.iinfo(dtype).min, np.iinfo(dtype).max)
                    if dtype.kind in "ui"
                    else (settings["float_min"], settings["float_max"])
                )
                for index, value in enumerate(model.history):
                    raw = value.astype(float) * (hi - lo) + lo
                    if dtype.kind in "ui":
                        raw = np.rint(raw)
                    raw = raw[..., 0] if fmt == "gray" else raw
                    if fmt in ("bgra", "rgba"):
                        raw = np.dstack([raw, np.full(raw.shape[:2], hi)])
                    self._append(
                        ValuePacket(
                            ImageArray(raw.astype(dtype), fmt), model.info["captures"][index]
                        ),
                        f"Saved capture {index + 1}",
                    )
                self.candidate = model
                self.show_report(model)
                self.status.setText(
                    f"Loaded {len(self.frames)} captures; baseline fixed until staged/applied."
                )
            except (ValueError, OSError, KeyError) as exc:
                self.status.setText(f"Could not load saved reference: {exc}")
        self.refresh_regions()
        self.controls()

    @staticmethod
    def button(layout, text, callback):
        widget = QPushButton(text)
        widget.clicked.connect(callback)
        layout.addWidget(widget)
        return widget

    def controls(self):
        idle = self.future is None
        for widget in (
            self.picker,
            self.import_button,
            self.remove_frame_button,
            self.add_region_button,
            self.edit_region_button,
            self.remove_region_button,
        ):
            widget.setEnabled(idle)
        self.train_button.setEnabled(idle and bool(self.frames))
        self.stage_button.setEnabled(
            idle and self.candidate is not None and self.context.project_dir is not None
        )

    def invalidate(self):
        self.candidate = None
        self.report.setRowCount(0)
        self.status.setText("Reference inputs changed. Train again before staging.")
        self.controls()

    def _append(self, packet, label):
        self.frames.append(packet)
        self.capture_list.addItem(label)
        self.capture_list.setCurrentRow(len(self.frames) - 1)

    def add_capture(self, packet):
        try:
            normalized(packet.value, self.parameters)
            if len(self.frames) >= 32:
                raise ValueError("Capture history is limited to thirty-two images")
            if self.frames and (
                packet.value.shape != self.frames[0].value.shape
                or packet.value.dtype != self.frames[0].value.dtype
                or image_format(packet.value) != image_format(self.frames[0].value)
            ):
                raise ValueError("All captures must share dimensions, encoding and depth")
            frames = [entry.value for entry in self.frames] + [packet.value]
            if (
                sum(a.shape[0] * a.shape[1] * (1 if a.ndim == 2 else 3) for a in frames)
                > MAX_SAMPLES
            ):
                raise ValueError("Capture history exceeds 32 million intensity samples")
            self._append(copy.deepcopy(packet), f"Capture {len(self.frames) + 1}")
            if len(self.frames) == 1 and not any(self.regions.values()):
                size = list(packet.value.shape[1::-1])
                for role in ("lighting", "focus"):
                    self.regions[role] = [{"name": role.title(), "region": full_region(size)}]
            self.invalidate()
            self.refresh_regions()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))

    def import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Reference captures", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        for path in paths:
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.status.setText(f"Could not read {Path(path).name}")
                continue
            fmt = "gray" if image.ndim == 2 else "bgra" if image.shape[2] == 4 else "bgr"
            self.add_capture(ValuePacket(ImageArray(image, fmt), {}))

    def remove_capture(self):
        row = self.capture_list.currentRow()
        if row >= 0:
            del self.frames[row]
            self.capture_list.takeItem(row)
            self.invalidate()

    def refresh_preview(self, *_):
        if not hasattr(self, "preview") or not self.frames:
            return
        image = self.frames[min(len(self.frames) - 1, max(0, self.capture_list.currentRow()))].value
        rgb = np.ascontiguousarray(preview_rgb(image))
        frame = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        from PySide6.QtCore import Qt

        self.preview.setPixmap(
            QPixmap.fromImage(frame).scaled(700, 330, Qt.AspectRatioMode.KeepAspectRatio)
        )

    def refresh_regions(self, *_):
        if not hasattr(self, "region_list"):
            return
        self.region_list.clear()
        self.region_list.addItems(row["name"] for row in self.regions[self.role.currentText()])
        self.refresh_preview()

    def draw_region(self, existing=None):
        if not self.frames:
            self.status.setText("Capture/import a reference image first.")
            return None
        image = self.frames[0].value
        h, w = image.shape[:2]
        p = {
            "shape": "rectangle",
            "rectangle": [w // 4, h // 4, max(2, w // 2), max(2, h // 2)],
            "center": [w // 2, h // 2],
            "radius": max(1, min(w, h) // 4),
            "vertices": [[0, 0], [w - 1, 0], [w - 1, h - 1]],
            "invert": False,
        }
        p.update(existing or {})
        editor = RegionEditor(image, p, self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return None
        p = editor.parameters
        key = {"rectangle": "rectangle", "circle": "center", "polygon": "vertices"}[p["shape"]]
        region = {
            "model": "image_region_v1",
            "image_size": [w, h],
            "shape": p["shape"],
            "invert": p["invert"],
            key: p[key],
        }
        if p["shape"] == "circle":
            region["radius"] = p["radius"]
        return region

    def add_region(self):
        region = self.draw_region()
        if region is None:
            return
        name, ok = QInputDialog.getText(self, "Region name", "Unique name within this category")
        if ok and name.strip():
            self.regions[self.role.currentText()].append({"name": name.strip(), "region": region})
            self.invalidate()
            self.refresh_regions()

    def edit_region(self):
        row = self.region_list.currentRow()
        if row >= 0:
            data = self.regions[self.role.currentText()][row]
            region = self.draw_region(data["region"])
            if region is not None:
                data["region"] = region
                self.invalidate()

    def remove_region(self):
        row = self.region_list.currentRow()
        if row >= 0:
            del self.regions[self.role.currentText()][row]
            self.invalidate()
            self.refresh_regions()

    def draft(self):
        return {
            **self.parameters,
            **{k: f.value() for k, f in self.limits.items()},
            "position_method": self.method.currentText(),
        }

    def start_training(self):
        try:
            p = self.draft()
            CameraHealth().validate_parameters(p)
            if p["position_method"] == "fiducials" and not self.regions["fiducials"]:
                raise ValueError("Draw at least one fiducial region")
            self.candidate = None
            self.future = _TRAINERS.submit(
                train_reference, copy.deepcopy(self.frames), copy.deepcopy(self.regions), p
            )
            self.status.setText("Training captured copies; the graph is not executed.")
            self.timer.start()
            self.controls()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))

    def poll(self):
        if self.future is None or not self.future.done() or self.closed:
            return
        future, self.future = self.future, None
        self.timer.stop()
        try:
            self.candidate = future.result()
            self.show_report(self.candidate)
            self.status.setText(
                f"Trained {len(self.frames)} captures; stage, then Apply configuration to activate."
            )
        except Exception as exc:
            self.candidate = None
            self.status.setText(f"Training failed: {exc}")
        self.controls()

    def show_report(self, reference):
        rows = [
            (role, row)
            for role in ("lighting", "focus")
            for row in reference.info["baselines"][role]
        ]
        self.report.setRowCount(len(rows))
        for i, (role, row) in enumerate(rows):
            stats = row["statistics"] or {}
            for j, value in enumerate((role, row["name"], stats.get("mean"), stats.get("std"))):
                self.report.setItem(
                    i,
                    j,
                    QTableWidgetItem(json.dumps(value) if not isinstance(value, str) else value),
                )

    def stage(self):
        try:
            if self.candidate is None:
                raise ValueError("Train a reference first")
            draft = self.draft()
            CameraHealth().validate_parameters(draft)
            if (
                draft["position_method"] == "fiducials"
                and not self.candidate.info["regions"]["fiducials"]
            ):
                raise ValueError("Train one or more fiducial regions before choosing Fiducials")
            patch = stage_reference(self.candidate, self.context.project_dir)
            self.parameters.update(draft)
            self.parameters.update(patch)
            self.accept()
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def finish(self, *_):
        self.closed = True
        self.timer.stop()
        if self.future is not None:
            self.future.cancel()
            self.future = None
        self.frames.clear()
