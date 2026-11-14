"""Retained native teaching: floor picks, bin box, physical tool, frozen preview, Stage/Apply."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.cloud_teaching import CloudTeachingWidget
from opencv_dataflow.cloud_teaching_geometry import CloudBox
from opencv_dataflow.cloud_viewer import CloudViewer
from opencv_dataflow.pointclouds import PointCloud
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .dataflow import input_pose, plan_inputs, recipe_path
from .graphics import LAYERS, result_cloud
from .persistence import load_recipe, save_recipe, strict_json
from .planning import Picker
from .scene import Scene
from .specs import Camera, SuctionSettings, floor_frame
from .teaching import TeachingSession, initial_recipe, run_job


class BinPickingDialog(QDialog):
    def __init__(self, parent, parameters, context):
        super().__init__(parent)
        self.parameters, self.context = parameters, context
        self.session = TeachingSession()
        self.floor_picks = []
        self.form_dirty = False
        self._edited_fields = set()
        self.result = None
        self._future = self._job = self._capture = None
        self._reference_capture = self._job_capture = None
        self._training = self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="suction-preview")
        self.setWindowTitle("SimpleBinPicking · train and review")
        self.resize(1450, 960)
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.buttons = {}
        for name, callback in (
            ("Import recipe…", self._import_recipe),
            ("Import depth + camera…", self._import_depth),
            ("Capture connected reference", self.capture_reference),
            ("Preview reference", self.preview_reference),
            ("Preview completed run", self.preview_completed),
            ("Export review…", self._choose_export),
            ("Stage recipe", self.stage),
        ):
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, f=callback: self._run(f))
            actions.addWidget(button)
            self.buttons[name] = button
        layout.addLayout(actions)
        self.status = QLabel(
            "Capture or import a reference, pick the floor, draw the bin and preview."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        side = QWidget()
        controls = QVBoxLayout(side)
        self.identity = QLabel("No retained reference")
        self.identity.setWordWrap(True)
        controls.addWidget(self.identity)
        self.collect = QCheckBox("Collect floor picks (3 or more spread-out points)")
        self.collect.toggled.connect(
            lambda checked: self.editor.mode.setCurrentIndex(
                self.editor.mode.findData("pick" if checked else "orbit")
            )
        )
        controls.addWidget(self.collect)
        self.floor_status = QLabel("No floor picks")
        controls.addWidget(self.floor_status)
        row = QHBoxLayout()
        for name, callback in (
            ("Fit floor / align +Z", self.align_floor),
            ("Clear picks", self.clear_picks),
            ("Draw bin", self.draw_bin),
        ):
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, f=callback: self._run(f))
            row.addWidget(button)
        controls.addLayout(row)
        form = QFormLayout()
        self.fields = {}
        settings = SuctionSettings()
        for name in settings.__dataclass_fields__:
            if name in ("max_candidates", "max_picks", "workers"):
                spin = QSpinBox()
                spin.setRange(
                    1, 2048 if name == "max_candidates" else 100 if name == "max_picks" else 8
                )
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(8)
                spin.setRange(-1e6 if name == "tcp_offset" else 1e-8, 1e6)
                if name in ("min_contact_fraction", "overlap_threshold"):
                    spin.setRange(0.5 if name == "min_contact_fraction" else 0.0, 1.0)
                elif name == "max_tilt_degrees":
                    spin.setRange(1.0, 80.0)
            spin.setValue(getattr(settings, name))
            spin.valueChanged.connect(lambda value, n=name: self._dirty(n))
            self.fields[name] = spin
            form.addRow(name.replace("_", " ").capitalize(), spin)
        self.scale = QDoubleSpinBox()
        self.scale.setDecimals(12)
        self.scale.setRange(1e-12, 1e6)
        self.scale.setValue(1.0)
        self.scale.setToolTip(
            "Set before capture/import, unless calibration JSON supplies depth_scale. "
            "A changed depth scale needs a new reference/frame setup."
        )
        self.scale.valueChanged.connect(lambda value: self._dirty("depth_scale"))
        form.addRow("Depth units per sample", self.scale)
        controls.addLayout(form)
        self.health = QLabel(
            "Reference preview reports contact, observed collisions and unknowns separately."
        )
        self.health.setWordWrap(True)
        controls.addWidget(self.health)
        row = QHBoxLayout()
        undo, cancel = QPushButton("Undo committed edit"), QPushButton("Cancel")
        undo.clicked.connect(lambda: self._run(self.undo))
        cancel.clicked.connect(self.reject)
        row.addWidget(undo)
        row.addWidget(cancel)
        controls.addLayout(row)
        splitter.addWidget(side)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()
        self.editor = CloudTeachingWidget()
        self.editor.new_box_name = "Bin interior"
        self.editor.point_picked.connect(self._picked)
        self.editor.box_created.connect(lambda b: self._run(lambda: self._box(b)))
        self.editor.box_changed.connect(lambda b: self._run(lambda: self._box(b)))
        self.tabs.addTab(self.editor, "Reference / bin / floor")
        self.viewer = CloudViewer(None, self)
        self.viewer.setWindowFlags(Qt.WindowType.Widget)
        self.tabs.addTab(self.viewer, "Cached candidate review")
        right_layout.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.layers = {}
        for name in LAYERS:
            box = QCheckBox(name.capitalize())
            box.setChecked(True)
            box.toggled.connect(self.refresh_review)
            self.layers[name] = box
            row.addWidget(box)
        overview = QPushButton("Overview")
        overview.clicked.connect(self.overview)
        row.addWidget(overview)
        right_layout.addLayout(row)
        legend = QLabel(
            "Green: eligible · Red: rejected · Orange: unknown · Yellow: selected · "
            "Blue: configured swept envelope · Magenta: observed collision · RGB: TCP. "
            "+Z retracts; approach is -Z. Scores describe geometry, not pick probability."
        )
        legend.setWordWrap(True)
        right_layout.addWidget(legend)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ID / rank", "Status", "Score", "Contact", "Clearance", "Instance", "Reasons"]
        )
        self.table.setMaximumHeight(220)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda *_: self.refresh_review())
        right_layout.addWidget(self.table)
        splitter.addWidget(right)
        splitter.setSizes([430, 1000])
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)
        if parameters.get("recipe_path"):
            self._run(
                lambda: self.restore(
                    load_recipe(
                        recipe_path(context.project_dir, parameters["recipe_path"]),
                        sha256=parameters["sha256"],
                    )
                )
            )
        self.refresh()

    def _run(self, callback):
        try:
            callback()
        except (ValueError, RuntimeError, OSError, TypeError, KeyError) as error:
            self.status.setText(str(error))

    def _dirty(self, name):
        if self.session.recipe is not None:
            self._edited_fields.add(name)
            self.form_dirty = True
            self.buttons["Stage recipe"].setEnabled(False)
            self.status.setText("Settings edited. Preview this draft before Stage / Apply.")

    def _sync(self):
        if self.session.recipe is None:
            raise ValueError("Import or capture a reference first")
        if self.form_dirty:
            if "depth_scale" in self._edited_fields and not np.isclose(
                self.scale.value(), self.session.recipe.depth_scale, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    "Depth scale changed: capture/import again to rebuild bin geometry"
                )
            settings = replace(
                self.session.recipe.settings,
                **{
                    name: self.fields[name].value()
                    for name in self._edited_fields
                    if name in self.fields
                },
            )
            self.session.change(settings=settings)
            self.form_dirty = False
            self._edited_fields.clear()

    def restore(self, recipe):
        if not len(Scene(recipe, recipe.reference_depth).points):
            raise ValueError("Retained reference has no measured depth; import a valid reference")
        self._reference_capture = None
        self.session.load(recipe)
        self._source_changed()
        self.preview_reference()

    def _source_changed(self):
        recipe = self.session.recipe
        scene = Scene(recipe, recipe.reference_depth)
        colors = (
            None
            if recipe.reference_rgb is None
            else recipe.reference_rgb[scene.pixels[:, 1], scene.pixels[:, 0]]
        )
        if not len(scene.points):
            raise ValueError("Reference has no valid depth points")
        self.editor.set_cloud(PointCloud(scene.points, recipe.units, recipe.frame, colors=colors))
        self.editor.set_boxes(
            [CloudBox("Bin interior", recipe.minimum, recipe.maximum)], "Bin interior"
        )
        self.floor_picks.clear()
        self.floor_status.setText("Floor picks cleared after reference/frame replacement")
        for name, spin in self.fields.items():
            spin.blockSignals(True)
            spin.setValue(getattr(recipe.settings, name))
            spin.blockSignals(False)
            if name in (
                "cup_radius",
                "body_radius",
                "body_length",
                "pregrasp_distance",
                "tcp_offset",
                "surface_tolerance",
                "clearance",
                "min_pick_height",
            ):
                spin.setSuffix(" " + recipe.units)
        self.scale.blockSignals(True)
        self.scale.setValue(recipe.depth_scale)
        self.scale.blockSignals(False)
        self.form_dirty = False
        self._edited_fields.clear()
        self.result = None
        self.table.setRowCount(0)
        self.viewer.set_cloud(None)
        self.identity.setText(
            f"{len(scene.points):,} retained measured points · {recipe.units} · "
            f"{recipe.frame} → {recipe.camera.frame}\nFloor is minimum Z; top is open."
        )
        self.refresh()

    def _import_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import retained bin recipe", "", "Recipe (*.sbp)"
        )
        if path:
            self.restore(load_recipe(path))

    def _import_depth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import raw depth", "", "Depth (*.png *.tif *.npy)"
        )
        if not path:
            return
        if Path(path).suffix.lower() == ".npy":
            from .persistence import read_array

            depth = read_array(Path(path).read_bytes())
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        calibration, _ = QFileDialog.getOpenFileName(
            self, "Rectified camera_model_v1 JSON", "", "Camera (*.json)"
        )
        if not calibration:
            return
        if Path(calibration).stat().st_size > 1024 * 1024:
            raise ValueError("Camera JSON exceeds 1 MiB")
        record = strict_json(Path(calibration).read_bytes())
        self.load_reference(depth, record)

    def load_reference(self, depth, camera_record, *, rgb=None, instances=None, bin_pose=None):
        recipe = initial_recipe(
            Camera.from_record(camera_record),
            depth,
            units=camera_record.get("units", "mm"),
            depth_scale=camera_record.get("depth_scale", self.scale.value()),
            rgb=rgb,
        )
        if bin_pose is not None:
            recipe = replace(recipe, frame=bin_pose.get("source_frame", "bin"))
            transform = input_pose(bin_pose, recipe)
            data = Scene(recipe, depth, bin_to_camera=transform).points
            lo, hi = np.quantile(data, [0.01, 0.99], axis=0)
            lo[2] = 0.0
            hi = np.maximum(hi, lo + recipe.settings.cup_radius * 4)
            recipe = replace(recipe, bin_to_camera=transform, minimum=tuple(lo), maximum=tuple(hi))
        recipe = replace(recipe, reference_instances=instances)
        self.session.load(recipe)
        self._reference_capture = None
        self._source_changed()
        self.status.setText(
            "Frozen reference loaded. Initial bounds are a proposal: "
            "pick the floor, fit +Z, draw/resize the bin and preview."
        )

    def capture_reference(self):
        if not hasattr(self.context, "snapshot_connected_inputs"):
            raise ValueError(
                "Update the host for one-bundle connected capture; import depth/camera "
                "or a retained recipe on this host"
            )
        bundle = self.context.snapshot_connected_inputs(
            ["depth", "camera", "image", "instances", "bin_pose"]
        )
        if "camera" not in bundle.inputs:
            raise ValueError("Connect camera_model_v1 calibration before capturing a new reference")
        image = bundle.inputs.get("image")
        rgb = None
        if image is not None:
            from opencv_dataflow.images import image_format

            raw = image.value
            if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 3:
                rgb = raw if image_format(raw) == "rgb" else raw[..., ::-1]
        self.load_reference(
            bundle.inputs["depth"].value,
            bundle.inputs["camera"].value,
            rgb=rgb,
            instances=None
            if "instances" not in bundle.inputs
            else bundle.inputs["instances"].value,
            bin_pose=None if "bin_pose" not in bundle.inputs else bundle.inputs["bin_pose"].value,
        )
        self._reference_capture = bundle

    def _picked(self, pick):
        if self.collect.isChecked() and self.session.recipe is not None:
            if (
                pick.get("source_identity", self.editor.source.identity)
                != self.editor.source.identity
                or type(pick.get("row")) is not int
                or not 0 <= pick["row"] < len(self.editor.source.cloud.points)
            ):
                self.status.setText("Ignored a stale point pick after source replacement")
                return
            if len(self.floor_picks) >= 1024:
                self.status.setText("At most 1024 floor picks are supported")
                return
            point = self.editor.source.cloud.points[pick["row"]]
            transform = self.session.recipe.bin_to_camera
            self.floor_picks.append(point @ transform[:3, :3].T + transform[:3, 3])
            self.floor_status.setText(
                f"{len(self.floor_picks)} floor picks; spread them over the floor"
            )

    def clear_picks(self):
        self.floor_picks.clear()
        self.floor_status.setText("No floor picks")

    def align_floor(self):
        self._sync()
        if self._reference_capture is not None and "bin_pose" in self._reference_capture.inputs:
            raise ValueError(
                "Connected bin_pose owns this frame. Edit that fixture or disconnect "
                "and recapture before fitting a different floor frame."
            )
        transform, rms = floor_frame(self.floor_picks)
        recipe = replace(self.session.recipe, bin_to_camera=transform)
        data = Scene(recipe, recipe.reference_depth).points
        lo, hi = np.quantile(data, [0.01, 0.99], axis=0)
        lo[2] = 0.0
        hi = np.maximum(hi, lo + recipe.settings.cup_radius * 4)
        self.session.change(bin_to_camera=transform, minimum=tuple(lo), maximum=tuple(hi))
        self._source_changed()
        self.collect.setChecked(False)
        self.status.setText(
            f"Floor aligned from picked observations; residual RMS {rms:.6g} "
            f"{recipe.units}. Resize the bin and preview; this is not gauge truth."
        )

    def draw_bin(self):
        if self.session.recipe is None:
            raise ValueError("Capture or import a reference first")
        self.editor.new_box_name = "Bin interior"
        self.editor.mode.setCurrentIndex(self.editor.mode.findData("draw"))

    def _box(self, box):
        self._sync()
        self.session.change(minimum=box.minimum, maximum=box.maximum)
        self.editor.set_boxes([CloudBox("Bin interior", box.minimum, box.maximum)], "Bin interior")
        self.status.setText("Bin edited. Preview before Stage; prior results stay frozen.")
        self.refresh()

    def undo(self):
        if self.session.recipe is None:
            raise ValueError("Import or capture a reference first")
        self.session.undo()
        self._source_changed()

    def preview_reference(self):
        self._sync()
        self._start(self.session.job(), True)

    def preview_completed(self):
        self._sync()
        recipe = self.session.staged()
        bundle = self.context.snapshot_processed_inputs()
        camera = bundle.inputs.get("camera")
        if camera is not None and Camera.from_record(camera.value) != recipe.camera:
            raise ValueError("Completed camera differs from training")
        self._start(self.session.job(), False, bundle)

    def _start(self, job, training, bundle=None):
        if self._future is not None:
            raise ValueError("A bounded preview is already running")
        self._job, self._training = job, training
        self._job_capture = self._reference_capture if training else bundle
        if training:
            self._future = self._executor.submit(run_job, job)
        else:
            recipe = job[1]
            self._future = self._executor.submit(
                plan_inputs, Picker(recipe), {name: p.value for name, p in bundle.inputs.items()}
            )
        self.status.setText("Previewing one frozen draft/capture. Edits invalidate this result.")
        self.timer.start()
        self.refresh()

    def poll(self):
        if self._closed or self._future is None or not self._future.done():
            return
        future, job, training = self._future, self._job, self._training
        self._future = self._job = None
        self.timer.stop()
        try:
            result = future.result()
            if (
                self.form_dirty
                or job[0] != self.session.revision
                or job[1] is not self.session.recipe
            ):
                self.status.setText("Discarded stale preview; preview the current draft again")
            else:
                if training:
                    self.session.accept(job, result)
                self.result = result
                self._capture = self._job_capture
                self._populate()
                self.tabs.setCurrentIndex(1)
                self.status.setText(
                    "Frozen reference preview" if training else "Frozen completed run"
                )
        except Exception as error:
            self.status.setText(str(error))
        self.refresh()

    def _populate(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.result.candidates))
        for i, item in enumerate(self.result.candidates):
            values = (
                f"{item['id']} / {item.get('rank', '—')}",
                item["status"],
                f"{item['score']:.3f}",
                item["contact_status"],
                item["clearance_status"],
                str(item["instance_id"]),
                ", ".join(item["reasons"]),
            )
            for j, text in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.blockSignals(False)
        if self.result.candidates:
            self.table.setCurrentCell(0, 0)
        self.table.resizeColumnsToContents()
        self.health.setText(
            f"{len(self.result.picks)} eligible alternatives / "
            f"{len(self.result.candidates)} evaluated / "
            f"{self.result.searched_cells} occupied cells. "
            "Unknowns and rejections stay in the table; scores are geometric evidence."
        )
        self.refresh_review()

    def overview(self):
        self.table.setCurrentCell(-1, -1)
        self.refresh_review()

    def refresh_review(self, *_):
        if self.result is None:
            return
        row = self.table.currentRow()
        selected = (
            self.result.candidates[row]["id"] if 0 <= row < len(self.result.candidates) else None
        )
        cloud = result_cloud(
            self.result,
            selected=selected,
            layers=tuple(n for n, b in self.layers.items() if b.isChecked()),
        )
        self.viewer.set_cloud(
            PointCloud(
                cloud.points, cloud.units, cloud.frame, colors=cloud.colors, metadata=cloud.metadata
            )
        )

    def refresh(self):
        self.buttons["Stage recipe"].setEnabled(self.session.trained and not self.form_dirty)
        self.buttons["Preview reference"].setEnabled(self._future is None)
        self.buttons["Preview completed run"].setEnabled(
            self._future is None and self.session.trained
        )
        self.buttons["Export review…"].setEnabled(self.result is not None)

    def export(self, folder):
        if self.result is None:
            raise ValueError("Preview a reference or completed run before exporting")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        write_cloud(folder / "pick_review.ply", self.viewer.cloud, binary=True)
        save_recipe(folder / "recipe.sbp", self.result.scene.recipe)
        np.save(
            folder / "depth.npy",
            self.result.scene.depth / self.result.scene.recipe.depth_scale,
            allow_pickle=False,
        )
        report = {
            **self.result.report(),
            "capture_id": getattr(self._capture, "snapshot_id", None),
            "selected_candidate": self.viewer.cloud.metadata["selected_candidate"],
            "layers": self.viewer.cloud.metadata["layers"],
        }
        report["capture_identity"] = {
            name: getattr(self._capture, name)
            for name in (
                "snapshot_id",
                "publication_id",
                "block_id",
                "output_revision",
                "execution_count",
                "sources",
                "missing_inputs",
                "passed",
            )
            if hasattr(self._capture, name)
        }
        (folder / "bin_pose.json").write_bytes(
            (json.dumps(self.result.scene.pose.tolist(), indent=2) + "\n").encode("utf-8")
        )
        (folder / "report.json").write_bytes(
            (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        return folder

    def _choose_export(self):
        parent = QFileDialog.getExistingDirectory(self, "Parent for a fresh pick review export")
        if parent:
            self.export(Path(parent) / ("bin_picking_" + uuid.uuid4().hex[:12]))

    def stage(self):
        if self.form_dirty:
            raise ValueError("Preview edited settings before staging")
        recipe = self.session.staged()
        relative = "assets/bin_picking/" + uuid.uuid4().hex + ".sbp"
        path = recipe_path(self.context.project_dir, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = save_recipe(path, recipe)
        self.parameters.update(recipe_path=relative, sha256=digest)
        self.accept()

    def done(self, result):
        if not self._closed:
            self._closed = True
            self.timer.stop()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.viewer.orbit_timer.stop()
            self.viewer.timer.stop()
        super().done(result)
