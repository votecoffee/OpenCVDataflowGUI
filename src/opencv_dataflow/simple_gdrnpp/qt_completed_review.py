"""Read-only completed pose review. Capture/export are separate from training drafts."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from opencv_dataflow.canvas import pixmap
from opencv_dataflow.cloud_teaching import CloudTeachingWidget
from opencv_dataflow.cloud_viewer import CloudViewer
from opencv_dataflow.filmstrip_viewer import FilmstripImageView
from opencv_dataflow.images import ImageArray
from opencv_dataflow.pointclouds import PointCloud

from .selection import candidate_status


class SampleImageView(FilmstripImageView):
    pixel_chosen = Signal(int, int)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            point = self.mapToScene(event.position().toPoint())
            self.pixel_chosen.emit(int(np.floor(point.x())), int(np.floor(point.y())))
            return
        super().mousePressEvent(event)


class CompletedReviewDialog(QDialog):
    def __init__(self, parent, context, prepare):
        super().__init__(parent)
        self.setWindowTitle("SimpleGDRNPP — completed poses and depth support")
        self.resize(1280, 900)
        self.context, self.prepare = context, prepare
        self.frozen = self.pending = None
        self.pending_kind = None
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancel = Event()
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.capture_button = QPushButton("Capture latest completed run")
        self.capture_button.clicked.connect(self.capture_latest)
        self.cancel_button = QPushButton("Cancel capture")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel.set)
        self.export_button = QPushButton("Export frozen evidence…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        for button in (self.capture_button, self.cancel_button, self.export_button):
            actions.addWidget(button)
        self.status = QLabel(
            "Run in Debug, then capture a completed result. Training drafts are independent."
        )
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        self.identity = QLabel()
        self.identity.setWordWrap(True)
        self.identity.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.identity)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Proposal", "ID", "Status", "Support", "Contradiction"]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMaximumHeight(160)
        self.table.itemSelectionChanged.connect(self.show_candidate)
        layout.addWidget(self.table)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.image = SampleImageView(self)
        self.image.pixel_chosen.connect(self.select_pixel)
        split.addWidget(self.image)
        self.tabs = QTabWidget()
        split.addWidget(self.tabs)
        layout.addWidget(split, 1)
        self.cloud = CloudViewer(None, self)
        self.cloud.setWindowFlags(Qt.WindowType.Widget)
        self.cloud.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.tabs.addTab(self.cloud, "Native fly-around")
        self.pick = CloudTeachingWidget(self)
        # Only view/pick are meaningful here. There are no editable regions.
        self.pick.mode.removeItem(self.pick.mode.findData("draw"))
        self.pick.mode.setItemText(0, "Orbit stored fit samples")
        self.pick.mode.setCurrentIndex(self.pick.mode.findData("pick"))
        self.pick.point_picked.connect(self.select_cloud_sample)
        self.tabs.addTab(self.pick, "Pick verification sample")
        self.histogram = FilmstripImageView(self)
        self.tabs.addTab(self.histogram, "Recorded depth residuals")
        self.reference = FilmstripImageView(self)
        self.tabs.addTab(self.reference, "Executed training reference")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, "Capture / model / settings / pose")
        row = QHBoxLayout()
        self.sample = QSpinBox()
        self.sample.setEnabled(False)
        self.sample.valueChanged.connect(self.show_sample)
        row.addWidget(QLabel("Verification sample"))
        row.addWidget(self.sample)
        self.scene = QCheckBox("Include stored measured scene")
        self.scene.setChecked(True)
        self.scene.toggled.connect(self.show_candidate)
        row.addWidget(self.scene)
        fit = QPushButton("Fit images")
        fit.clicked.connect(self.fit_images)
        row.addWidget(fit)
        layout.addLayout(row)
        self.sample_info = QLabel()
        self.sample_info.setWordWrap(True)
        self.sample_info.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.sample_info)
        legend = QLabel(
            "Green: supported · red: contradiction · purple: behind depth · "
            "blue: outside target · orange: unknown. Flags can overlap. "
            "Fit fractions use all original verification samples, including unknowns. "
            "The image projects those stored samples; it is not a newly rendered surface. "
            "Measured scene and predicted CAD samples remain distinct. Wheel zoom / drag pan. "
            "Shift-click: select a stored sample. RGB pose axes are 20 mm display guides. "
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)
        self.unsubscribe = context.subscribe_outputs(self.published)

    def published(self, revision):
        if self.frozen is not None:
            self.status.setText(
                "New outputs available. This review stays frozen; capture explicitly to replace it."
            )

    def capture_latest(self):
        if self.pending is not None:
            return
        try:
            job = self.prepare()  # Host's paired copy must run on the GUI thread.
        except (ValueError, RuntimeError) as error:
            self.status.setText(str(error))
            return
        self.cancel = Event()
        self.status.setText(
            "Copying completed worker evidence; no inference, render or verification…"
        )
        self.capture_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.pending = self.pool.submit(job.fetch, cancelled=self.cancel.is_set)
        self.pending_kind = "capture"
        self.timer.start()

    def poll(self):
        if self.pending is None or not self.pending.done():
            return
        future, self.pending = self.pending, None
        self.timer.stop()
        self.capture_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        try:
            if self.pending_kind == "export":
                future.result()
                self.export_button.setEnabled(True)
                self.status.setText(
                    "Exported frozen sources, recipe, samples and display graphics."
                )
                return
            frozen = future.result()
            if self.closed:
                return
            self.set_capture(frozen)
            self.status.setText(
                "Frozen execution captured. Browsing and export use only this private copy."
                if frozen.bundle.output_revision == self.context.output_revision
                else "Captured an older completion; newer outputs available. Recapture explicitly."
            )
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            self.export_button.setEnabled(self.frozen is not None)
            self.status.setText(
                f"Capture unavailable: {error}. Any previous frozen review remains available."
            )

    def set_capture(self, frozen):
        self.frozen = frozen
        report = frozen.report
        identity = frozen.record.identity
        self.identity.setText(
            f"{report['capture_id']} · {report['target_frame']} · "
            f"session {identity['session'][:12]} / run {identity['token'][:12]} · "
            f"recipe {identity['archive_sha256'][:16]} · "
            f"{report['accepted_candidates']}/{len(report['candidates'])} retained poses"
        )
        self.export_button.setEnabled(True)
        self.table.blockSignals(True)
        self.table.setRowCount(len(report["candidates"]))
        for i, candidate in enumerate(report["candidates"]):
            fit = candidate.get("depth_fit") or candidate["fit"]
            for col, text in enumerate(
                (
                    str(i + 1),
                    candidate["id"],
                    candidate_status(candidate),
                    f"{fit['coverage']:.2%}",
                    f"{fit['contradiction']:.2%}",
                )
            ):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        self.scene.setEnabled("scene_points" in frozen.arrays)
        self.reference.image_item.setPixmap(
            pixmap(ImageArray(frozen.image(None, reference=True), "rgb"))
        )
        self.reference.fit_image()
        selection = frozen.bundle.outputs["review"].value.get("candidate_index")
        if report["candidates"]:
            self.table.selectRow(selection if selection is not None else 0)
        else:
            self.show_candidate()

    def index(self):
        row = self.table.currentRow()
        return row if row >= 0 else None

    def show_candidate(self, *args):
        if self.frozen is None:
            return
        f, index = self.frozen, self.index()
        self.image.image_item.setPixmap(pixmap(ImageArray(f.image(index), "rgb")))
        self.image.fit_image()
        self.cloud.set_cloud(f.cloud(index, scene=self.scene.isChecked()))
        evidence = f.candidates[index] if index is not None else {}
        count = len(evidence.get("points", ()))
        self.sample.blockSignals(True)
        self.sample.setRange(0, max(0, count - 1))
        self.sample.setValue(0)
        self.sample.setEnabled(count > 0)
        self.sample.blockSignals(False)
        if count:
            self.pick.set_cloud(
                PointCloud(
                    evidence["points"], "m", f.report["target_frame"], colors=f.colors(index)
                )
            )
        else:
            self.pick.set_cloud(
                PointCloud(np.empty((0, 3), np.float32), "m", f.report["target_frame"])
            )
        self.pick.status.setText(
            "Pick a stored verification sample; original source pixel and depth appear below."
        )
        self.show_sample()
        self.details.setPlainText(
            json.dumps(
                dict(
                    report=f.report,
                    setup=f.metadata["setup"],
                    capture=f.metadata["capture"],
                    scope=f.metadata["scope"],
                ),
                indent=2,
            )
        )
        self.show_histogram(evidence)

    def show_sample(self, *args):
        if self.frozen is None or not self.sample.isEnabled():
            self.sample_info.setText("No verification samples were retained for this candidate.")
            return
        detail = self.frozen.sample(self.index(), self.sample.value())
        self.pick.last_pick = dict(row=self.sample.value(), xyz=detail["points"])
        self.pick.canvas.update()
        self.sample_info.setText(json.dumps(detail, sort_keys=True))
        # Mark the source pixel without changing either the stored source or evidence.
        image = self.frozen.image(self.index())
        if detail.get("inside"):
            pixel = tuple(detail["pixels"])
            cv2.drawMarker(image, pixel, (255, 255, 255), cv2.MARKER_CROSS, 13, 1)
        self.image.image_item.setPixmap(pixmap(ImageArray(image, "rgb")))

    def select_cloud_sample(self, detail):
        self.sample.setValue(int(detail["row"]))

    def select_pixel(self, x, y):
        if self.frozen is None or self.index() is None:
            return
        evidence = self.frozen.candidates[self.index()]
        rows = np.flatnonzero(evidence.get("inside", np.empty(0, bool)))
        if not len(rows):
            return
        distances = np.linalg.norm(evidence["pixels"][rows] - [x, y], axis=1)
        if distances.min() > 3:
            self.sample_info.setText("No stored verification sample near that source pixel.")
            return
        self.sample.setValue(int(rows[distances.argmin()]))
        self.show_sample()

    def show_histogram(self, e):
        image = np.full((300, 520, 3), 22, np.uint8)
        known = e.get("known", np.empty(0, bool))
        values = e.get("delta_m", np.empty(0))[known] * 1000
        if len(values):
            counts, bounds = np.histogram(values, bins=40)
            for i, count in enumerate(counts):
                cv2.rectangle(
                    image,
                    (30 + i * 12, 250),
                    (40 + i * 12, 250 - int(180 * count / max(counts.max(), 1))),
                    (100, 180, 255),
                    -1,
                )
            cv2.putText(
                image,
                f"{bounds[0]:.2f} to {bounds[-1]:.2f} mm",
                (30, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
            )
        cv2.putText(
            image,
            f"Recorded predicted - observed depth; {len(values)} known samples",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
        )
        self.histogram.image_item.setPixmap(pixmap(ImageArray(image, "rgb")))
        self.histogram.fit_image()

    def fit_images(self):
        for view in (self.image, self.histogram, self.reference):
            view.fit_image()

    def choose_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "New frozen review folder")
        if path:
            self.start_export(Path(path))

    def start_export(self, path):
        if self.pending is not None or self.frozen is None:
            return
        self.capture_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.status.setText("Exporting the frozen copy; no worker requests…")
        self.pending_kind = "export"
        self.pending = self.pool.submit(self.frozen.export, path)
        self.timer.start()

    def done(self, code):
        if not self.closed:
            self.closed = True
            self.cancel.set()
            self.timer.stop()
            self.unsubscribe()
            # Do not terminate a session another graph owns. The in-flight read finishes;
            # cancellation is checked before the next page, and its lease is always released.
            self.pool.shutdown(wait=False, cancel_futures=False)
        super().done(code)
