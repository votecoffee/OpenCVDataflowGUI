"""Native setup draft: retained reference, editable boxes, isolated preview and Stage."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPen
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
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from opencv_dataflow.canvas import pixmap
from opencv_dataflow.cloud_viewer import CloudViewer
from opencv_dataflow.filmstrip_viewer import FilmstripImageView
from opencv_dataflow.images import ImageArray
from opencv_dataflow.pointclouds import PointCloud

from .bridge import WorkerClient, configure_runtime, project_recipe_path, runtime_python


class BoxView(FilmstripImageView):
    box_drawn = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.draw_mode = False
        self.start = None
        self.rubber = None

    def mousePressEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self.start = self.mapToScene(event.position().toPoint())
            self.rubber = self.scene().addRect(
                QRectF(self.start, self.start), QPen(QColor(255, 220, 40), 2)
            )
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.rubber.setRect(
                QRectF(self.start, self.mapToScene(event.position().toPoint())).normalized()
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start is not None:
            rect = self.rubber.rect()
            self.scene().removeItem(self.rubber)
            self.start = self.rubber = None
            width, height = self.image_item.pixmap().width(), self.image_item.pixmap().height()
            box = (
                max(0, int(np.floor(rect.left()))),
                max(0, int(np.floor(rect.top()))),
                min(width, int(np.ceil(rect.right()))),
                min(height, int(np.ceil(rect.bottom()))),
            )
            if box[2] > box[0] and box[3] > box[1]:
                self.box_drawn.emit(box)
            return
        super().mouseReleaseEvent(event)


class GDRNPPSetupDialog(QDialog):
    def __init__(self, parent, parameters, context, *, client=None):
        super().__init__(parent)
        self.setWindowTitle("SimpleGDRNPP — object setup")
        self.resize(1180, 800)
        self.parameters = parameters
        self.context = context
        self.client = client or WorkerClient()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.base = None
        self.reference_override = None
        self.reference_arrays = {}
        self.reference = None
        self.summary = None
        self.revision = 0
        self.pending = None
        self.closed = False
        self.restoring = False
        self.token = None
        self.preview_arrays = None
        self.preview_report = None
        self.box_items = []
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        for text, callback in [
            ("Open recipe…", self._open_recipe),
            ("Import setup…", self._import),
            ("Replace reference…", self._reference_request),
        ]:
            actions.addWidget(self._button(text, callback))
        self.runtime_label = QLabel("Runtime not selected")
        self.runtime_label.setWordWrap(True)
        actions.addWidget(self.runtime_label, 1)
        actions.addWidget(self._button("Choose runtime…", self._runtime))
        layout.addLayout(actions)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model folder"))
        self.models = QLineEdit(parameters.get("models_dir", ""))
        model_row.addWidget(self.models, 1)
        model_row.addWidget(self._button("Browse…", self._models))
        layout.addLayout(model_row)
        self.models.textChanged.connect(self._invalidate)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        reference_tab = QWidget()
        ref_layout = QVBoxLayout(reference_tab)
        toolbar = QHBoxLayout()
        self.viewer = BoxView(self)
        for text, callback in [
            ("Fit", self.viewer.fit_image),
            ("100%", self.viewer.actual_size),
            ("Zoom +", lambda: self.viewer.zoom(1.25)),
            ("Zoom −", lambda: self.viewer.zoom(0.8)),
        ]:
            toolbar.addWidget(self._button(text, callback))
        self.draw = QCheckBox("Draw instance rectangle")
        self.draw.toggled.connect(self._draw_mode)
        toolbar.addWidget(self.draw)
        toolbar.addStretch()
        ref_layout.addLayout(toolbar)
        split = QSplitter()
        split.addWidget(self.viewer)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.name = QLineEdit()
        self.name.textChanged.connect(self._invalidate)
        side_layout.addWidget(QLabel("Setup name"))
        side_layout.addWidget(self.name)
        self.boxes = QTableWidget(0, 5)
        self.boxes.setHorizontalHeaderLabels(["Instance", "Left", "Top", "Right", "Bottom"])
        self.boxes.cellChanged.connect(self._box_changed)
        self.boxes.itemSelectionChanged.connect(self._paint_boxes)
        side_layout.addWidget(self.boxes, 1)
        side_layout.addWidget(self._button("Remove selected instance", self._remove))
        self.reference_info = QLabel("Open a recipe or import its setup definition.")
        self.reference_info.setWordWrap(True)
        side_layout.addWidget(self.reference_info)
        self.reference_cloud = self._button("Reference cloud…", self._reference_cloud)
        side_layout.addWidget(self.reference_cloud)
        split.addWidget(side)
        split.setSizes([780, 400])
        ref_layout.addWidget(split, 1)
        self.tabs.addTab(reference_tab, "Reference / instances")
        settings_tab = QWidget()
        form = QFormLayout(settings_tab)
        self.fields = {}
        self.dirty_fields = set()
        for key, text, low, high, decimals in [
            ("threads", "CPU threads", 1, 32, 0),
            ("batch_size", "Instance batch size", 1, 8, 0),
            ("distance_tolerance_m", "Depth tolerance", 0.01, 100, 2),
            ("minimum_coverage", "Minimum measured support", 10, 100, 1),
            ("maximum_contradiction", "Maximum contradiction", 0, 100, 1),
            ("maximum_correction_m", "Maximum translation correction", 0.01, 100, 2),
        ]:
            widget = QSpinBox() if decimals == 0 else QDoubleSpinBox()
            if decimals:
                widget.setDecimals(decimals)
            widget.setRange(low, high)
            if key.endswith("_m"):
                widget.setSuffix(" mm")
            elif decimals:
                widget.setSuffix(" %")
            widget.valueChanged.connect(lambda value, key=key: self._setting_changed(key))
            form.addRow(text, widget)
            self.fields[key] = widget
        self.depth_refine = QCheckBox("Optional depth refinement")
        self.depth_refine.toggled.connect(self._invalidate)
        form.addRow(self.depth_refine)
        self.refine_method = QComboBox()
        self.refine_method.addItem("Axial median translation", "median_translation")
        self.refine_method.addItem("Robust projective ICP (local 6D)", "projective_icp_v1")
        self.refine_method.currentIndexChanged.connect(self._invalidate)
        form.addRow("Depth alignment method", self.refine_method)
        hint = QLabel(
            "Setup imports a trained object model. It does not train a new object.\n"
            "Depth acceptance and pose/fit colors remain independent of detector confidence."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        self.tabs.addTab(settings_tab, "Execution / fit gates")
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_actions = QHBoxLayout()
        self.candidates = QComboBox()
        self.candidates.activated.connect(self._candidate)
        preview_actions.addWidget(self.candidates, 1)
        self.result_cloud_button = self._button("Flyaround result…", self._result_cloud)
        preview_actions.addWidget(self.result_cloud_button)
        preview_layout.addLayout(preview_actions)
        self.result_view = FilmstripImageView(self)
        preview_layout.addWidget(self.result_view, 1)
        self.result_info = QLabel("Preview the reference or a separate calibrated capture.")
        self.result_info.setWordWrap(True)
        preview_layout.addWidget(self.result_info)
        self.tabs.addTab(preview_tab, "Pose / depth review")
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bottom = QHBoxLayout()
        self.preview_button = self._button("Preview reference", self._preview_reference)
        self.test_button = self._button("Preview another capture…", self._preview_capture)
        self.stage_button = self._button("Stage setup", self._stage)
        self.cancel_button = self._button("Cancel preview", self._cancel_job)
        for button in [
            self.preview_button,
            self.test_button,
            self.stage_button,
            self.cancel_button,
        ]:
            bottom.addWidget(button)
        bottom.addStretch()
        if hasattr(context, "snapshot_connected_inputs"):
            bottom.addWidget(self._button("Use connected reference", self._use_connected))
            bottom.addWidget(self._button("Preview connected", self._preview_connected))
        bottom.addWidget(self._button("Close / discard", self.reject))
        layout.addLayout(bottom)
        self.viewer.box_drawn.connect(self._add_box)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self.finished.connect(self._finish)
        self._runtime_status()
        self._buttons()
        if parameters.get("recipe_path"):
            try:
                path = project_recipe_path(context.project_dir, parameters["recipe_path"])
                self.load_base(
                    dict(kind="recipe", path=str(path), sha256=parameters.get("sha256", ""))
                )
            except Exception as error:
                self.status.setText(str(error))

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda: self._perform(callback))
        return button

    def _perform(self, callback):
        try:
            callback()
        except Exception as error:
            self.status.setText(str(error))

    def _runtime_status(self):
        try:
            self.runtime_label.setText("Runtime: " + runtime_python())
        except (ValueError, OSError):
            self.runtime_label.setText("Choose the isolated GDRNPP runtime")

    def _runtime(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose installed GDRNPP Python",
            filter="Python (python.exe python);;All files (*)",
        )
        if path:
            if self.pending:
                raise ValueError("Cancel the current preview before changing runtime")
            self.client.close()
            configure_runtime(path)
            self.client = WorkerClient()
            self._runtime_status()
            if self.base:
                self._submit(
                    "inspect",
                    self._data() if self.summary else dict(base=self.base),
                    self._loaded,
                    self.reference_arrays,
                )

    def _models(self):
        path = QFileDialog.getExistingDirectory(self, "Choose verified checkpoint folder")
        if path:
            self.models.setText(path)

    def _open_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open retained object setup", filter="GDRNPP setup (*.gdrnpp)"
        )
        if path:
            self.load_base(dict(kind="recipe", path=path))

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import object setup definition", filter="Setup definition (*.json)"
        )
        if path:
            self.load_base(dict(kind="definition", path=path))

    def load_base(self, base):
        if self.pending:
            raise ValueError("Cancel the current task before opening another setup")
        self._invalidate()
        self.base = base
        self.reference_override = None
        self.reference_arrays = {}
        self.reference = None
        self.summary = None
        self._waiting_reference()
        self._submit("inspect", dict(base=base), self._loaded)

    def _reference_request(self):
        if not self.base:
            raise ValueError("Open a setup first")
        if self.pending:
            raise ValueError("Cancel the current task before replacing the reference")
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose calibrated RGB-D reference capture", filter="Capture request (*.json)"
        )
        if path:
            self._invalidate()
            self.reference_override = dict(kind="request", path=path)
            self.reference_arrays = {}
            self.reference = None
            self._waiting_reference()
            self._submit(
                "inspect", dict(base=self.base, reference=self.reference_override), self._loaded
            )

    def _submit(self, operation, data, callback, arrays=None):
        if self.pending:
            raise ValueError("Wait for the current task, or cancel it")
        cancellation = Event()
        revision = self.revision
        client = self.client
        future = self.pool.submit(
            client.call, operation, data, arrays, cancelled=cancellation.is_set
        )
        self.pending = (future, cancellation, revision, callback)
        self.status.setText("Working…")
        self._buttons()

    def _poll(self):
        if self.pending is None or not self.pending[0].done():
            return
        future, cancellation, revision, callback = self.pending
        self.pending = None
        try:
            result = future.result()
            if not self.closed and not cancellation.is_set() and revision == self.revision:
                callback(*result)
            elif not self.closed:
                self.status.setText("Discarded an older preview; use the current draft")
        except Exception as error:
            if not self.closed:
                self.status.setText(str(error))
        self._buttons()

    def _loaded(self, metadata, arrays):
        self.summary = metadata["setup"]
        self.reference = arrays
        self.restoring = True
        self.dirty_fields.clear()
        try:
            self.name.setText(self.summary["name"])
            self.boxes.setRowCount(0)
            for detection in self.summary["reference_detections"]:
                self._append_box(detection)
            for name, widget in self.fields.items():
                widget.setValue(self.summary["settings"][name] * self._display_factor(name))
            self.depth_refine.setChecked(self.summary["settings"]["depth_refine"])
            self.refine_method.setCurrentIndex(
                self.refine_method.findData(
                    self.summary["settings"].get("depth_refinement", "median_translation")
                )
            )
            self.viewer.set_image(pixmap(ImageArray(arrays["rgb"], "rgb")))
            self._paint_boxes()
            self.reference_info.setText(
                f"Retained {self.summary['reference_capture_id']}\n"
                f"{len(self.summary['reference_detections'])} instances · "
                f"{self.summary['symmetry_count']} discrete symmetries\n"
                f"Model: {self.summary['binding']['dataset']} / "
                f"{self.summary['binding']['object_id']}\n"
                f"Source: {self.summary['credit']['dataset']} · {self.summary['credit']['license']}"
            )
        finally:
            self.restoring = False
        self.status.setText("Retained reference restored. Edit instances, then preview or stage.")

    def _append_box(self, detection):
        row = self.boxes.rowCount()
        self.boxes.insertRow(row)
        for column, value in enumerate([detection["identifier"], *detection["box"]]):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, detection if column == 0 else None)
            self.boxes.setItem(row, column, item)

    def _draft(self):
        detections = []
        for row in range(self.boxes.rowCount()):
            original = self.boxes.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
            detections.append(
                dict(
                    identifier=self.boxes.item(row, 0).text(),
                    box=[float(self.boxes.item(row, c).text()) for c in range(1, 5)],
                    object_id=6,
                    source=original.get("source", "user_reference_rectangle"),
                    score=original.get("score"),
                )
            )
        return dict(
            name=self.name.text(),
            settings={
                **{
                    k: (
                        self.summary["settings"][k]
                        if self.summary and k not in self.dirty_fields
                        else w.value() / self._display_factor(k)
                        if self._display_factor(k) != 1
                        else w.value()
                    )
                    for k, w in self.fields.items()
                },
                "depth_refine": self.depth_refine.isChecked(),
                "depth_refinement": self.refine_method.currentData(),
            },
            detections=detections,
        )

    def _data(self):
        if not self.base:
            raise ValueError("Open a setup first")
        result = dict(base=self.base, draft=self._draft())
        if self.reference_override:
            result["reference"] = self.reference_override
        return result

    @staticmethod
    def _display_factor(name):
        return (
            1000
            if name.endswith("_m")
            else 100
            if name in ("minimum_coverage", "maximum_contradiction")
            else 1
        )

    def _model_folder(self):
        value = self.models.text()
        if not value:
            raise ValueError("Choose the verified external model folder")
        path = Path(value)
        return str(path if path.is_absolute() else Path(self.context.project_dir) / path)

    def _invalidate(self, *_):
        if self.restoring:
            return
        self.revision += 1
        self.token = None
        self.preview_arrays = None
        self.preview_report = None
        if hasattr(self, "result_view"):
            self.result_view.set_image(pixmap(np.zeros((2, 2, 3), np.uint8)))
        if self.pending:
            self.pending[1].set()
        if hasattr(self, "result_cloud_button"):
            self.result_cloud_button.setEnabled(False)
            self.result_info.setText("Draft changed — preview again to assess the current setup.")
            self.candidates.clear()

    def _setting_changed(self, name):
        if not self.restoring:
            self.dirty_fields.add(name)
        self._invalidate()

    def _waiting_reference(self):
        self.restoring = True
        try:
            self.boxes.setRowCount(0)
            self.viewer.set_image(pixmap(np.zeros((2, 2, 3), np.uint8)))
            self.reference_info.setText("Opening the selected reference…")
        finally:
            self.restoring = False

    def _box_changed(self, *_):
        self._invalidate()
        self._paint_boxes()

    def _paint_boxes(self):
        for item in self.box_items:
            self.viewer.scene().removeItem(item)
        self.box_items = []
        try:
            for row, detection in enumerate(self._draft()["detections"]):
                x0, y0, x1, y1 = detection["box"]
                color = (
                    QColor(255, 220, 40) if row == self.boxes.currentRow() else QColor(40, 240, 80)
                )
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                self.box_items.append(
                    self.viewer.scene().addRect(QRectF(x0, y0, x1 - x0, y1 - y0), pen)
                )
        except (ValueError, AttributeError):
            pass

    def _add_box(self, box):
        if self.reference is None:
            return
        self.restoring = True
        try:
            self._append_box(dict(identifier="instance_" + uuid.uuid4().hex[:8], box=box))
        finally:
            self.restoring = False
        self._box_changed()

    def _remove(self):
        if self.boxes.currentRow() >= 0:
            self.boxes.removeRow(self.boxes.currentRow())
            self._box_changed()

    def _draw_mode(self, value):
        self.viewer.draw_mode = value
        self.viewer.setDragMode(
            self.viewer.DragMode.NoDrag if value else self.viewer.DragMode.ScrollHandDrag
        )

    def _preview_reference(self):
        data = self._data()
        data.update(models_dir=self._model_folder(), graphics=True)
        self._submit("preview", data, self._previewed, self.reference_arrays)

    def _preview_capture(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Preview independent RGB-D capture", filter="Capture request (*.json)"
        )
        if path:
            data = self._data()
            data.update(
                models_dir=self._model_folder(),
                graphics=True,
                current=dict(kind="request", path=path),
            )
            self._submit("preview", data, self._previewed, self.reference_arrays)

    def _connected(self, prefix):
        from .dataflow import CAPTURE_PORTS, connected_capture

        bundle = self.context.snapshot_connected_inputs(CAPTURE_PORTS)
        current, arrays = connected_capture(
            {k: v.value for k, v in bundle.inputs.items()},
            {k: v.payload for k, v in bundle.inputs.items()},
            depth_scale_to_m=self.parameters.get("depth_scale_to_m", 1.0),
        )
        current["array_prefix"] = prefix
        return current, {prefix + k: v for k, v in arrays.items()}

    def _use_connected(self):
        if self.base is None:
            raise ValueError("Import the matching trained object setup first")
        if self.pending:
            raise ValueError("Cancel the current task before replacing the reference")
        current, arrays = self._connected("reference_")
        self._invalidate()
        self.reference_override = current
        self.reference_arrays = arrays
        self.reference = None
        self._waiting_reference()
        self._submit("inspect", dict(base=self.base, reference=current), self._loaded, arrays)

    def _preview_connected(self):
        current, arrays = self._connected("current_")
        data = self._data()
        data.update(current=current, models_dir=self._model_folder(), graphics=True)
        self._submit("preview", data, self._previewed, {**self.reference_arrays, **arrays})

    def _previewed(self, metadata, arrays):
        self.token = metadata["token"]
        self.preview_report = metadata["report"]
        self.preview_arrays = arrays
        self.result_view.set_image(pixmap(ImageArray(arrays["overlay"], "rgb")))
        self.candidates.blockSignals(True)
        self.candidates.clear()
        for candidate in self.preview_report["candidates"]:
            fit = candidate["fit"]
            self.candidates.addItem(
                f"{candidate['id']} · {fit['status']} · support {fit['coverage']:.1%}"
            )
        self.candidates.setCurrentIndex(metadata["candidate_index"])
        self.candidates.blockSignals(False)
        accepted = self.preview_report["accepted_candidates"]
        self.result_info.setText(
            f"{accepted} accepted / {len(self.preview_report['candidates'])} candidates. "
            "Green: supported; red: contradicted; purple: behind depth; orange: unknown/outside. "
            "Axes and predicted CAD surface are explanatory graphics, not camera measurements."
        )
        self.tabs.setCurrentIndex(2)
        self.status.setText(
            "Preview completed. Acceptance uses measured-depth gates. "
            "Refinement reuses these measurements when enabled."
        )
        self.result_cloud_button.setEnabled(True)

    def _candidate(self, index):
        if self.token:
            self._submit("render", dict(token=self.token, index=index), self._previewed)

    def _reference_cloud(self):
        if self.reference is None:
            raise ValueError("Open a reference first")
        f = self.reference
        y, x = np.where(f["depth"] > 0)
        ids = np.linspace(0, len(x) - 1, min(30_000, len(x)), dtype=int)
        y, x = y[ids], x[ids]
        z = f["depth"][y, x]
        k = f["camera"]
        points = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
        self._cloud(PointCloud(points, "m", self.summary["camera_frame"], colors=f["rgb"][y, x]))

    def _result_cloud(self):
        if self.preview_arrays is None:
            raise ValueError("Preview the current draft first")
        self._cloud(
            PointCloud(
                self.preview_arrays["points"],
                "m",
                self.preview_report["target_frame"],
                colors=self.preview_arrays["colors"],
            )
        )

    def _cloud(self, cloud):
        dialog = CloudViewer(None, self)
        dialog.set_cloud(cloud)
        dialog.flyaround.setChecked(True)
        dialog.show()

    def _stage(self):
        data = self._data()
        relative = "assets/gdrnpp/" + uuid.uuid4().hex + ".gdrnpp"
        project_recipe_path(self.context.project_dir, relative)
        data.update(project_dir=str(self.context.project_dir), relative_asset=relative)
        self._submit("stage", data, self._staged, self.reference_arrays)

    def _staged(self, metadata, arrays):
        self.parameters.update(
            recipe_path=metadata["recipe_path"],
            sha256=metadata["sha256"],
            models_dir=self.models.text(),
        )
        self.accept()

    def _cancel_job(self):
        if self.pending:
            self.pending[1].set()
        self._invalidate()
        self.status.setText("Cancelling preview; applied setup is unchanged")

    def _buttons(self):
        busy = self.pending is not None
        for button in [self.preview_button, self.test_button, self.stage_button]:
            button.setEnabled(not busy and self.reference is not None)
        self.cancel_button.setEnabled(busy)
        self.reference_cloud.setEnabled(self.reference is not None)
        self.result_cloud_button.setEnabled(self.preview_arrays is not None and not busy)

    def _finish(self, *_):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        if self.pending:
            self.pending[1].set()
        client = self.client
        self.pool.submit(client.close)
        self.pool.shutdown(wait=False, cancel_futures=False)
