"""Desktop application entry point and inspector."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .blocks import builtin_registry, demo_graph
from .calibration import calibration_templates
from .canvas import CanvasScene, CanvasView, Node, pixmap
from .core import Graph
from .examples import (
    channel_pyramid_graph,
    combined_regions_graph,
    control_flow_graph,
    input_coordination_graph,
    object_collections_graph,
    parallel_parts_graph,
    pose_inspection_graph,
    queued_input_graph,
    regions_graph,
    starter_graph,
    template_graph,
    ticket_results_graph,
    trained_patterns_graph,
    transformed_regions_graph,
)
from .image_queue import ImageQueue
from .outcomes import INPUT_FAILURE_POLICIES
from .parallel_inspection import ParallelInspect
from .payloads import describe_value, packet_for, prepare_inputs
from .plugins import load_plugin
from .queued_execution import QueuedExecution, background_queue
from .region_editor import RegionEditor
from .ticket_window import TicketResultsWindow
from .tickets import TICKET_MODES
from .toolblock_editor import InterfaceDialog
from .toolblocks import ToolBlock, encapsulate


class Runner(QThread):
    frame = Signal()
    failed = Signal(str)

    def __init__(self, graph, continuous, registry=None):
        super().__init__()
        self.graph, self.continuous = graph, continuous
        self.stop_requested = threading.Event()
        self.ack = threading.Event()
        self.duration = 0.0
        self.registry = registry or builtin_registry()
        self.final_queue_states = {}

    def stop(self):
        self.stop_requested.set()

    def run(self):
        queued = None
        try:
            if self.continuous and background_queue(self.graph) is not None:
                queued = QueuedExecution(self.graph, self.registry, self.stop_requested.is_set)
                queued.start()
            while not self.stop_requested.is_set():
                started = time.perf_counter()
                if queued is not None:
                    if not queued.run_one():
                        break
                else:
                    self.graph.run(self.stop_requested.is_set)
                self.duration = (time.perf_counter() - started) * 1000
                self.ack.clear()
                self.frame.emit()
                # UI reads a stable result before allowing the next iteration.
                self.ack.wait()
                if not self.continuous:
                    break
                self.stop_requested.wait(0.05)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if queued is not None:
                queued.close()
            self.final_queue_states = {
                key: b.snapshot_runtime()
                for key, b in self.graph.blocks.items()
                if isinstance(b, ImageQueue)
            }
            self.graph.close()


class Inspector(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.block = None
        self._region_image = None
        self._region_source = None
        self._region_inputs = {}
        self.fields, self.port_checks, self.level_fields, self.failure_fields = {}, [], {}, {}
        layout = QVBoxLayout(self)
        title = QLabel("BLOCK INSPECTOR")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.caption = QLabel("Select a block to configure its tools and ports.")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.editor = QGroupBox("Configuration")
        self.form = QFormLayout(self.editor)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, 3)
        self.preview_choice = QComboBox()
        self.preview_choice.currentTextChanged.connect(self.show_preview)
        layout.addWidget(self.preview_choice)
        self.image = QLabel("No image available")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(160)
        self.image.setStyleSheet("background: #111b29; border-radius: 6px;")
        layout.addWidget(self.image, 2)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 2)
        self.setMinimumWidth(310)

    def select(self, block):
        if block is not self.block:
            self._region_image = None
            self._region_source = None
            self._region_inputs = {}
        self.block = block
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields, self.port_checks, self.level_fields, self.failure_fields = {}, [], {}, {}
        if block is None:
            self.caption.setText("Select a block to configure its tools and ports.")
            self.refresh_result()
            return
        self.caption.setText(
            f"{block.title} / {block.block_type or block.category} / {block.subtype}"
            f"\n{block.type_id}"
        )
        self.name_field = QLineEdit(block.name)
        self.form.addRow("Name", self.name_field)
        if block.ticket_source:
            self.ticket_mode = QComboBox()
            self.ticket_mode.addItems(TICKET_MODES)
            self.ticket_mode.setCurrentText(block.ticket_settings["mode"])
            self.ticket_value = QLineEdit(json.dumps(block.ticket_settings["value"]))
            self.ticket_value.setToolTip(
                "Fixed JSON number/string or {timestamp: ISO8601}; "
                "external mode uses the ticket input."
            )
            self.form.addRow("Ticket mode", self.ticket_mode)
            self.form.addRow("Fixed ticket", self.ticket_value)
        if isinstance(block, ToolBlock):
            self.type_field = QLineEdit(block.block_type)
            self.subtype_field = QLineEdit(block.subtype)
            self.form.addRow("Block type", self.type_field)
            self.form.addRow("Subtype", self.subtype_field)
            for title, callback in (
                ("Edit internal graph", lambda: self.window.enter_toolblock(block)),
                ("Edit public terminals", self.window.edit_interface),
            ):
                button = QPushButton(title)
                button.clicked.connect(callback)
                self.form.addRow(button)
        if isinstance(block, ParallelInspect):
            for title, callback in (
                ("Edit inspection graph", lambda: self.window.enter_toolblock(block)),
                ("Load inspection ToolBlock…", self.load_inspection),
                ("Export inspection ToolBlock…", self.window.export_toolblock),
            ):
                button = QPushButton(title)
                button.clicked.connect(callback)
                self.form.addRow(button)
        if getattr(block, "supports_training", False):
            model = block.trained_model
            external = any(
                c.target == block.id and c.input == "model" for c in self.window.graph.connections
            )
            status = (
                "External model input is connected"
                if external
                else f"Trained: {len(model['points'])} features, {model['size']} pixels"
                if model
                else "Not trained"
            )
            self.form.addRow(QLabel(status))
            for label, callback in (
                ("Train from input", self.train_pattern),
                ("Load trained model…", self.load_pattern_model),
                ("Save trained model…", self.save_pattern_model),
                ("Clear trained model", self.clear_pattern_model),
            ):
                button = QPushButton(label)
                button.clicked.connect(callback)
                button.setEnabled(
                    not external
                    and (model is not None or label in ("Train from input", "Load trained model…"))
                )
                self.form.addRow(button)
        for key, value in block.parameters.items():
            if key in block.parameter_choices:
                field = QComboBox()
                field.addItems(block.parameter_choices[key])
                if field.findText(str(value)) < 0:
                    field.addItem(str(value))
                field.setCurrentText(str(value))
            elif isinstance(value, bool):
                field = QCheckBox()
                field.setChecked(value)
            else:
                field = QLineEdit(
                    json.dumps(value) if isinstance(value, (list, dict)) else str(value)
                )
            self.fields[key] = field
            field.setToolTip(block.parameter_help.get(key, ""))
            if key == "path":
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(field)
                browse = QPushButton("…")
                browse.setFixedWidth(30)
                browse.clicked.connect(lambda checked=False, f=field: self.browse_path(f))
                row_layout.addWidget(browse)
                self.form.addRow(key.replace("_", " ").title(), row)
            else:
                self.form.addRow(key.replace("_", " ").title(), field)
        if getattr(block, "supports_region_editor", False):
            kind = getattr(block, "editor_kind", "region")
            draw_region = QPushButton(f"Draw / edit {kind}…")
            draw_region.clicked.connect(self.edit_region)
            self.form.addRow(draw_region)
        if not getattr(block, "preserve_image_sets", False):
            for port in block.inputs:
                if port.kind == "image":
                    level = QComboBox()
                    level.addItems(["largest", *map(str, range(9))])
                    level.setCurrentText(block.input_levels.get(port.name, "largest"))
                    level.setToolTip("Used when this input receives an image pyramid set.")
                    self.level_fields[port.name] = level
                    self.form.addRow(f"Pyramid level: {port.name}", level)
        self.thumb = QCheckBox("Show image on node")
        self.thumb.setChecked(block.thumbnail)
        self.form.addRow(self.thumb)
        self.node_image = QComboBox()
        self.node_image.addItem("Automatic", "")
        for port in block.outputs:
            if port.kind in ("image", "depth", "pyramid"):
                self.node_image.addItem(port.name, port.name)
        index = self.node_image.findData(block.preview_output)
        self.node_image.setCurrentIndex(max(0, index))
        self.form.addRow("Node image", self.node_image)
        for direction in ("inputs", "outputs"):
            for port in getattr(block, direction):
                check = QCheckBox(f"{port.name} ({port.kind})")
                check.setChecked(port.name in getattr(block, f"exposed_{direction}"))
                self.form.addRow(f"Expose {direction[:-1]}", check)
                self.port_checks.append((direction, port.name, check))
                if direction == "inputs":
                    policy = QComboBox()
                    policy.addItems(INPUT_FAILURE_POLICIES)
                    policy.setCurrentText(block.input_failure_policies.get(port.name, "continue"))
                    policy.setToolTip(
                        "reject_failed: suppress this invocation if the connection carries an "
                        "accumulated failure. Retain the last processed inputs/results."
                    )
                    self.failure_fields[port.name] = policy
                    self.form.addRow(f"{port.name} failure policy", policy)
        apply_button = QPushButton("Apply configuration")
        apply_button.clicked.connect(self.apply)
        self.form.addRow(apply_button)
        self.editor.setEnabled(self.window.runner is None)
        self.refresh_result()

    def browse_path(self, field):
        if self.block.category == "Input":
            path, _ = QFileDialog.getOpenFileName(self, "Select input image")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Select output path")
        if path:
            field.setText(path)

    def read_parameters(self):
        b = self.block
        parameters = {}
        for key, field in self.fields.items():
            initial = b.defaults[key]
            if isinstance(field, QComboBox):
                value = field.currentText()
            elif isinstance(initial, bool):
                value = field.isChecked()
            elif isinstance(initial, int):
                value = int(field.text())
            elif isinstance(initial, float):
                value = float(field.text())
            elif isinstance(initial, (list, dict)):
                value = json.loads(field.text())
                if not isinstance(value, type(initial)):
                    raise ValueError(f"{key} requires a JSON {type(initial).__name__}")
            else:
                value = field.text()
            parameters[key] = value
        json.dumps(parameters, allow_nan=False)
        b.validate_parameters(parameters)
        return parameters

    def edit_region(self):
        block = self.block
        if (
            block is None
            or not getattr(block, "supports_region_editor", False)
            or self.window.runner
        ):
            return
        try:
            parameters = self.read_parameters()
            connections = [
                c
                for c in self.window.graph.connections
                if c.target == block.id and c.input in ("image", "pose")
            ]
            source = (
                tuple((c.input, c.source, c.output) for c in connections),
                self.level_fields["image"].currentText(),
            )
            values = dict(block.last_inputs)
            for connection in connections:
                result = self.window.graph.blocks[connection.source].last_result
                if result is not None:
                    values[connection.input] = packet_for(result, connection.output)
            if (
                not values
                and source == self._region_source
                and all(self.window.graph.blocks[c.source].state == "idle" for c in connections)
            ):
                values = self._region_inputs
            candidate = self.window.registry.create(block.type_id, parameters=parameters)
            candidate.input_levels = {
                key: field.currentText()
                for key, field in self.level_fields.items()
                if field.currentText() != "largest"
            }
            values, _ = prepare_inputs(candidate, values)
            image = values.get("image")
            if image is None:
                raise ValueError(
                    "Connect an image and Run once to capture it before drawing a region or mask."
                )
            requires_pose = any(c.input == "pose" for c in connections) or any(
                port.name == "pose" and port.required for port in block.inputs
            )
            pose = values.get("pose")
            if requires_pose and not pose:
                raise ValueError(
                    "Run once with a valid pose before editing a relative region or mask"
                )
            self._region_image, self._region_source = image, source
            self._region_inputs = values
            dialog = RegionEditor(
                image, parameters, self, pose=pose, kind=getattr(block, "editor_kind", "region")
            )
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            updated = dialog.parameters if accepted else None
            dialog.deleteLater()
            if accepted:
                for key, value in updated.items():
                    field = self.fields[key]
                    if isinstance(field, QComboBox):
                        field.setCurrentText(value)
                    elif isinstance(field, QCheckBox):
                        field.setChecked(value)
                    else:
                        field.setText(json.dumps(value) if isinstance(value, list) else str(value))
                self.apply()
                if self.block is block:
                    self._region_image, self._region_source = image, source
                    self._region_inputs = values
        except (ValueError, TypeError, KeyError) as exc:
            self.window.warn(str(exc))

    def train_pattern(self):
        block = self.block
        if block is None or not getattr(block, "supports_training", False) or self.window.runner:
            return
        try:
            candidate = self.window.registry.create(
                block.type_id, parameters=self.read_parameters()
            )
            candidate.input_levels = {
                key: field.currentText()
                for key, field in self.level_fields.items()
                if field.currentText() != "largest"
            }
            values = dict(block.last_inputs)
            for connection in self.window.graph.connections:
                if connection.target == block.id:
                    source = self.window.graph.blocks[connection.source]
                    if source.last_result is not None:
                        values[connection.input] = packet_for(source.last_result, connection.output)
            if "image" not in values:
                raise ValueError("Connect an image and Run once to capture a training input")
            values, _ = prepare_inputs(candidate, values)
            model = candidate.train(values)
            if self.apply():
                block.trained_model = model
                self.select(block)
                self.window.statusBar().showMessage(
                    f"Trained {len(model['points'])} features; save the project to retain the model"
                )
        except (ValueError, TypeError, KeyError, cv2.error) as exc:
            self.window.warn(str(exc))

    def load_inspection(self):
        block = self.block
        if self.window.runner or not isinstance(block, ParallelInspect):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load inspection ToolBlock", filter="ToolBlock (*.toolblock.json)"
        )
        if path:
            try:
                tool = ToolBlock.from_definition(
                    json.loads(Path(path).read_text(encoding="utf-8")), self.window.registry
                )
                candidate = ParallelInspect(
                    inspection=tool,
                    registry=self.window.registry,
                    parameters=self.read_parameters(),
                )
                candidate.validate_inspection()
                if self.apply():
                    block.inspection = tool
                    block.graph.bind_project_dir(self.window.root_graph.project_dir)
                    self.select(block)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.window.warn(str(exc))

    def load_pattern_model(self):
        block = self.block
        if block is None or self.window.runner:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load trained pattern", filter="Pattern JSON (*.json)"
        )
        if path:
            try:
                model = json.loads(Path(path).read_text(encoding="utf-8"))
                block.check_model(model)
                if self.apply():
                    block.trained_model = model
                    self.select(block)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.window.warn(str(exc))

    def save_pattern_model(self):
        block = self.block
        if block is None or self.window.runner or block.trained_model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save trained pattern", filter="Pattern JSON (*.json)"
        )
        if path:
            try:
                with Path(path).open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(block.trained_model, indent=2, allow_nan=False) + "\n")
            except (ValueError, OSError) as exc:
                self.window.warn(str(exc))

    def clear_pattern_model(self):
        block = self.block
        if (
            block is not None
            and not self.window.runner
            and getattr(block, "supports_training", False)
        ):
            block.trained_model = None
            self.window.graph.invalidate()
            self.window.scene.rebuild(block.id)
            self.window.mark_dirty()

    def apply(self):
        b = self.block
        if b is None or self.window.runner:
            return False
        try:
            parameters = self.read_parameters()
            exposed = {
                d: [
                    n
                    for direction, n, check in self.port_checks
                    if direction == d and check.isChecked()
                ]
                for d in ("inputs", "outputs")
            }
            for c in self.window.graph.connections:
                if (c.source == b.id and c.output not in exposed["outputs"]) or (
                    c.target == b.id and c.input not in exposed["inputs"]
                ):
                    raise ValueError("Delete the wire before hiding a connected port")
            name = self.name_field.text().strip()
            ticket_settings = None
            if b.ticket_source:
                ticket_settings = {
                    "mode": self.ticket_mode.currentText(),
                    "value": json.loads(self.ticket_value.text()),
                }
                json.dumps(ticket_settings, allow_nan=False)
            if not name:
                raise ValueError("Name cannot be empty")
            if isinstance(b, ToolBlock):
                kind, subtype = self.type_field.text().strip(), self.subtype_field.text().strip()
                if not kind or not subtype:
                    raise ValueError("ToolBlock type and subtype cannot be empty")
                b.block_type, b.subtype = kind, subtype
            b.name, b.parameters = name, parameters
            if ticket_settings is not None:
                b.ticket_settings = ticket_settings
            b.input_levels = {
                name: field.currentText()
                for name, field in self.level_fields.items()
                if field.currentText() != "largest"
            }
            b.input_failure_policies = {
                name: field.currentText()
                for name, field in self.failure_fields.items()
                if field.currentText() != "continue"
            }
            b.exposed_inputs, b.exposed_outputs = exposed["inputs"], exposed["outputs"]
            b.thumbnail = self.thumb.isChecked()
            b.preview_output = self.node_image.currentData()
            self.window.graph.invalidate()
            self.window.scene.rebuild(b.id)
            self.window.mark_dirty()
            self.window.statusBar().showMessage("Configuration applied; run to update results")
            return True
        except (ValueError, TypeError) as exc:
            self.window.warn(str(exc))
            return False

    def refresh_result(self):
        previous = self.preview_choice.currentText()
        self.preview_choice.blockSignals(True)
        self.preview_choice.clear()
        b = self.block
        if b:
            for name in b.last_inputs:
                if hasattr(b.last_inputs[name], "shape"):
                    self.preview_choice.addItem(f"Input / {name}")
            for prefix, images in (("Output", b.last_output_images), ("Graphic", b.last_graphics)):
                for name in images:
                    self.preview_choice.addItem(f"{prefix} / {name}")
        index = self.preview_choice.findText(previous)
        self.preview_choice.setCurrentIndex(index if index >= 0 else 0)
        self.preview_choice.blockSignals(False)
        self.show_preview(self.preview_choice.currentText())
        if not b:
            self.details.clear()
        else:
            result = b.last_result
            outputs = (
                {
                    key: describe_value(value)
                    for key, value in result.outputs.items()
                    if key not in b.last_output_images
                }
                if result
                else {}
            )
            score = f"{result.score:.1%}" if result and result.score is not None else "—"
            self.details.setPlainText(
                f"{b.state.upper()}   ·   {b.elapsed_ms:.2f} ms   ·   score {score}\n"
                + (b.last_rejection or b.error or (result.message if result else "Not executed"))
                + "\n\n"
                + json.dumps(
                    {"outputs": outputs, "payloads": result.payloads} if result else outputs,
                    indent=2,
                    default=str,
                )
            )

    def show_preview(self, choice):
        self.image.clear()
        if self.block and " / " in choice:
            prefix, name = choice.split(" / ", 1)
            images = {
                "Input": self.block.last_inputs,
                "Output": self.block.last_output_images,
                "Graphic": self.block.last_graphics,
            }[prefix]
            if name in images:
                self.image.setPixmap(
                    pixmap(images[name]).scaled(
                        max(100, self.image.width() - 12),
                        max(100, self.image.height() - 12),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return
        self.image.setText("Run to inspect images")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_preview(self.preview_choice.currentText())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.registry = builtin_registry()
        self.graph = demo_graph(self.registry)
        self.graph_stack = []
        self.toolblock_templates = calibration_templates(self.registry)
        self.path = None
        self.dirty = False
        self.runner = None
        self.closing = False
        self.setWindowTitle("OpenCV Dataflow · Untitled")
        self.resize(1500, 920)
        self.scene = CanvasScene(self.graph)
        self.view = CanvasView(self.scene)
        self.inspector = Inspector(self)
        self.library = QTreeWidget()
        self.library.setHeaderHidden(True)
        self.library.setIndentation(14)
        self.library.itemDoubleClicked.connect(self.add_block)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search blocks…")
        self.search.textChanged.connect(self.populate_library)
        self.populate_library()
        library_panel = QWidget()
        library_layout = QVBoxLayout(library_panel)
        label = QLabel("TOOL LIBRARY")
        label.setObjectName("sectionTitle")
        library_layout.addWidget(label)
        library_layout.addWidget(self.search)
        library_layout.addWidget(self.library)
        hint = QLabel(
            "Double-click to add\n\nDrag ports to connect\nDelete removes selection\n"
            "Wheel zooms · middle-drag pans"
        )
        hint.setWordWrap(True)
        library_layout.addWidget(hint)
        split = QSplitter()
        split.addWidget(library_panel)
        split.addWidget(self.view)
        split.addWidget(self.inspector)
        split.setSizes([210, 940, 350])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.scene.selected_block.connect(self.inspector.select)
        self.scene.edited.connect(self.mark_dirty)
        self.scene.layout_changed.connect(self.mark_layout_dirty)
        self.scene.warning.connect(self.warn)
        self.scene.open_block.connect(self.enter_toolblock)
        self.make_actions()
        self.statusBar().showMessage("Demo ready · Run once to inspect two shapes")
        QTimer.singleShot(0, self.view.fit_graph)

    def make_actions(self):
        menu = self.menuBar().addMenu("&File")
        toolbar = QToolBar("Project and execution")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.edit_actions = []
        for title, shortcut, callback in (
            ("New", QKeySequence.StandardKey.New, self.new_project),
            ("Open…", QKeySequence.StandardKey.Open, self.open_project),
            ("Save", QKeySequence.StandardKey.Save, self.save_project),
            ("Save as…", QKeySequence.StandardKey.SaveAs, lambda: self.save_project(True)),
            ("Load Python plugin…", None, self.open_plugin),
            ("Load demo", None, self.load_demo),
            ("Load starter inspection", None, self.load_starter),
            ("Load template matching example", None, self.load_template_example),
            ("Load regions example", None, self.load_regions_example),
            ("Load combined regions example", None, self.load_combined_regions_example),
            ("Load transformed regions example", None, self.load_transformed_regions_example),
            ("Load channels and pyramids example", None, self.load_channel_pyramid_example),
            ("Load object collections example", None, self.load_object_collections_example),
            ("Load trained patterns and edges example", None, self.load_trained_patterns_example),
            ("Load pose-relative inspection example", None, self.load_pose_inspection_example),
            ("Load ticketed control flow example", None, self.load_control_flow_example),
            ("Load parallel part inspection example", None, self.load_parallel_parts_example),
            ("Load queued input example", None, self.load_queued_input_example),
            ("Load ticket results example", None, self.load_ticket_results_example),
            ("Load input coordination example", None, self.load_input_coordination_example),
            ("Reset flow state", None, self.reset_flow_state),
        ):
            action = QAction(title, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(lambda checked=False, f=callback: f())
            menu.addAction(action)
            self.edit_actions.append(action)
            if title in ("New", "Open…", "Save"):
                toolbar.addAction(action)
        toolbar.addSeparator()
        self.run_action = toolbar.addAction("Run once", lambda: self.start_run(False))
        self.run_action.setShortcut("F5")
        self.continuous_action = toolbar.addAction("Continuous", lambda: self.start_run(True))
        self.stop_action = toolbar.addAction("Stop", self.stop_run)
        self.stop_action.setEnabled(False)
        toolbar.addSeparator()
        toolbar.addAction("Fit graph", self.view.fit_graph)
        results_menu = self.menuBar().addMenu("&Results")
        results_menu.addAction("Ticket results…", self.show_ticket_results)
        toolbar.addAction("Delete selected", self.scene.delete_selected)
        tools = self.menuBar().addMenu("&ToolBlocks")
        for title, callback in (
            ("Create from selected nodes…", self.create_toolblock),
            ("Edit selected ToolBlock", self.edit_selected_toolblock),
            ("Public terminals…", self.edit_interface),
            ("Export ToolBlock…", self.export_toolblock),
            ("Import ToolBlock…", self.import_toolblock),
        ):
            action = tools.addAction(title, callback)
            self.edit_actions.append(action)
        toolbar.addSeparator()
        self.back_action = toolbar.addAction("Up one level", self.leave_toolblock)
        self.edit_actions.append(self.back_action)
        self.breadcrumb = QLabel("Job")
        toolbar.addWidget(self.breadcrumb)
        self.update_navigation()

    def populate_library(self):
        self.library.clear()
        query = self.search.text().lower()
        groups = {}

        def add_entry(kind, subtype, title, type_id):
            if query not in f"{kind} {subtype} {title}".lower():
                return
            if kind not in groups:
                groups[kind] = QTreeWidgetItem(self.library, [kind])
            key = (kind, subtype)
            if key not in groups:
                groups[key] = QTreeWidgetItem(groups[kind], [subtype])
            item = QTreeWidgetItem(groups[key], [title])
            item.setToolTip(0, f"{kind} / {subtype} / {title}")
            item.setData(0, Qt.ItemDataRole.UserRole, type_id)

        for block_type in self.registry.types.values():
            add_entry(
                block_type.block_type or block_type.category,
                block_type.subtype,
                block_type.title,
                block_type.type_id,
            )
        for index, definition in enumerate(self.toolblock_templates):
            add_entry(
                definition.get("block_type", "ToolBlocks"),
                definition.get("subtype", "Reusable"),
                definition["name"],
                f"template:{index}",
            )
        self.library.expandAll()

    def add_block(self, item, column=0):
        if self.runner:
            return
        type_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not type_id:
            return
        if type_id.startswith("template:"):
            definition = self.toolblock_templates[int(type_id.split(":")[1])]
            block = ToolBlock.from_definition(definition, self.registry)
        else:
            block = self.registry.create(type_id)
        self.insert_block(block)

    def insert_block(self, block):
        center = self.view.mapToScene(self.view.viewport().rect().center())
        block.position = [center.x() - 110, center.y() - 80]
        self.graph.add(block)
        self.graph.invalidate()
        self.scene.rebuild(block.id)
        self.mark_dirty()

    @property
    def root_graph(self):
        return self.graph_stack[0][0] if self.graph_stack else self.graph

    def selected_toolblock(self):
        selected = self.inspector.block
        if isinstance(selected, ParallelInspect):
            selected.inspection._parent_node_id = selected.id
            return selected.inspection, self.graph
        if isinstance(selected, ToolBlock):
            return selected, self.graph
        if self.graph_stack:
            parent, owner = self.graph_stack[-1]
            return owner, parent
        raise ValueError("Select a ToolBlock first")

    def create_toolblock(self):
        if self.runner:
            return
        selected = [item.block.id for item in self.scene.selectedItems() if isinstance(item, Node)]
        if not selected:
            self.warn("Select the nodes to include in a ToolBlock")
            return
        name, accepted = QInputDialog.getText(self, "Create ToolBlock", "Name:", text="ToolBlock")
        if not accepted:
            return
        try:
            owner = self.graph_stack[-1][1] if self.graph_stack else None
            block = encapsulate(self.graph, selected, name, self.registry, owner)
            self.toolblock_templates.append(block.definition())
            self.populate_library()
            self.scene.rebuild(block.id)
            self.mark_dirty()
        except ValueError as exc:
            self.warn(str(exc))

    def edit_selected_toolblock(self):
        self.enter_toolblock(self.inspector.block)

    def enter_toolblock(self, block):
        if isinstance(block, ParallelInspect):
            block.inspection._parent_node_id = block.id
            block = block.inspection
        if self.runner or not isinstance(block, ToolBlock):
            return
        self.graph_stack.append((self.graph, block))
        self.show_graph(block.graph)

    def show_graph(self, graph, selected=None):
        self.graph = graph
        self.scene.graph = graph
        self.inspector.select(None)
        self.scene.rebuild(selected)
        self.update_navigation()
        self.view.fit_graph()

    def leave_toolblock(self):
        if self.runner or not self.graph_stack:
            return
        parent, owner = self.graph_stack.pop()
        self.show_graph(parent, getattr(owner, "_parent_node_id", owner.id))

    def update_navigation(self):
        self.back_action.setEnabled(bool(self.graph_stack) and self.runner is None)
        self.breadcrumb.setText("Job" + "".join(f" / {b.name}" for _, b in self.graph_stack))

    def edit_interface(self):
        if self.runner:
            return
        try:
            block, parent = self.selected_toolblock()
            owner = next((b for _, b in self.graph_stack if b.graph is parent), None)
            dialog = InterfaceDialog(block, parent, self, owner=owner)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected = self.inspector.block
                self.scene.rebuild(selected.id if selected else None)
                self.mark_dirty()
        except ValueError as exc:
            self.warn(str(exc))

    def export_toolblock(self):
        if self.runner:
            return
        try:
            block, _ = self.selected_toolblock()
            path, _ = QFileDialog.getSaveFileName(
                self, "Export ToolBlock", filter="ToolBlock (*.toolblock.json)"
            )
            if path:
                block.export(path)
                self.toolblock_templates.append(block.definition())
                self.populate_library()
        except Exception as exc:
            self.warn(str(exc))

    def import_toolblock(self):
        if self.runner:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import ToolBlock", filter="ToolBlock (*.json)")
        if path:
            try:
                block = ToolBlock.read(path, self.registry)
                self.toolblock_templates.append(block.definition())
                self.populate_library()
                self.insert_block(block)
            except Exception as exc:
                self.warn(str(exc))

    def mark_dirty(self):
        if self.graph_stack:
            self.root_graph.invalidate()
        self.dirty = True
        self.update_title()

    def mark_layout_dirty(self):
        self.dirty = True
        self.update_title()

    def update_title(self):
        self.setWindowTitle(
            f"OpenCV Dataflow · {self.path.name if self.path else 'Untitled'}"
            + (" *" if self.dirty else "")
        )

    def warn(self, message):
        QMessageBox.warning(self, "OpenCV Dataflow", message)

    def confirm_discard(self):
        if not self.dirty:
            return True
        response = QMessageBox.question(
            self,
            "Unsaved project",
            "Save your project before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if response == QMessageBox.StandardButton.Save:
            return self.save_project()
        return response == QMessageBox.StandardButton.Discard

    def replace_graph(self, graph, path=None):
        self.root_graph.close()
        self.graph_stack = []
        self.show_graph(graph)
        self.path, self.dirty = path, False
        self.update_title()
        self.view.fit_graph()
        self.statusBar().showMessage(f"Job ready · {len(graph.blocks)} blocks")

    def new_project(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(Graph())

    def load_demo(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(demo_graph(self.registry))

    def load_starter(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(starter_graph(self.registry))

    def load_template_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(template_graph(self.registry))

    def load_regions_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(regions_graph(self.registry))

    def load_combined_regions_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(combined_regions_graph(self.registry))

    def load_transformed_regions_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(transformed_regions_graph(self.registry))

    def load_channel_pyramid_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(channel_pyramid_graph(self.registry))

    def load_object_collections_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(object_collections_graph(self.registry))

    def load_trained_patterns_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(trained_patterns_graph(self.registry))

    def load_pose_inspection_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(pose_inspection_graph(self.registry))

    def load_control_flow_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(control_flow_graph(self.registry))

    def load_parallel_parts_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(parallel_parts_graph(self.registry))

    def load_queued_input_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(queued_input_graph(self.registry))

    def load_ticket_results_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(ticket_results_graph(self.registry))

    def load_input_coordination_example(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(input_coordination_graph(self.registry))

    def open_project(self):
        if self.runner or not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", filter="Dataflow (*.json)")
        if path:
            try:
                graph = Graph.load(path, self.registry)
                self.replace_graph(graph, Path(path))
            except Exception as exc:
                self.warn(str(exc))

    def save_project(self, save_as=False):
        if self.runner:
            return False
        path = self.path
        if save_as or path is None:
            selected, _ = QFileDialog.getSaveFileName(
                self, "Save project", filter="Dataflow (*.json)"
            )
            if not selected:
                return False
            path = Path(selected)
        try:
            self.root_graph.save(path)
            self.path, self.dirty = path, False
            self.update_title()
            return True
        except Exception as exc:
            self.warn(str(exc))
            return False

    def open_plugin(self):
        if self.runner:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load trusted Python plugin", filter="Python (*.py)"
        )
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "Execute Python plugin",
            f"Plugins execute Python with your permissions.\nLoad this trusted file?\n{path}",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            load_plugin(path, self.registry)
            self.populate_library()
            self.statusBar().showMessage("Plugin loaded for this session")
        except Exception as exc:
            self.warn(str(exc))

    def start_run(self, continuous):
        if self.runner:
            return
        # Worker owns a separate runtime graph. Qt paints stable display snapshots.
        try:
            execution_graph = Graph.from_dict(
                self.root_graph.to_dict(), self.registry, project_dir=self.root_graph.project_dir
            )
            execution_graph.copy_runtime_from(self.root_graph)
            for key, block in self.root_graph.blocks.items():
                execution_graph.blocks[key].copy_results_from(block)
        except Exception as exc:
            self.warn(str(exc))
            return
        self.runner = Runner(execution_graph, continuous, self.registry)
        self.runner.frame.connect(self.on_frame)
        self.runner.failed.connect(self.warn)
        self.runner.finished.connect(self.on_finished)
        self.scene.set_locked(True)
        self.library.setEnabled(False)
        self.inspector.editor.setEnabled(False)
        for action in self.edit_actions + [self.run_action, self.continuous_action]:
            action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.statusBar().showMessage("Running…")
        self.runner.start()

    def reset_flow_state(self):
        if not self.runner:
            self.root_graph.invalidate()
            self.scene.update()
            self.inspector.refresh_result()
            self.statusBar().showMessage("Flow buffers cleared; a new ticket session will start")

    def on_frame(self):
        try:
            self.root_graph.copy_runtime_from(self.runner.graph)
            for key, executed in self.runner.graph.blocks.items():
                self.root_graph.blocks[key].copy_results_from(executed)
            self.scene.update()
            self.inspector.refresh_result()
            states = [b.state for b in self.root_graph.blocks.values()]
            self.statusBar().showMessage(
                f"{len(states)} blocks · {self.runner.duration:.1f} ms · "
                f"{states.count('pass')} pass · {states.count('fail')} fail · "
                f"{states.count('error')} errors · {states.count('skipped')} skipped"
                f" · {states.count('waiting')} waiting · {states.count('inactive')} inactive"
            )
        finally:
            self.runner.ack.set()

    def stop_run(self):
        if self.runner:
            self.runner.stop()
            self.statusBar().showMessage("Stopping after the current block returns…")

    def on_finished(self):
        worker = self.runner
        # finished() can precede native thread-local cleanup; join before release.
        worker.wait()
        self.root_graph.ticket_results.restore(worker.graph.ticket_results.snapshot())
        for key, snapshot in worker.final_queue_states.items():
            block = self.root_graph.blocks.get(key)
            if isinstance(block, ImageQueue):
                block.restore_runtime(snapshot)
                block.update_metrics()
        self.scene.update()
        self.inspector.refresh_result()
        self.runner = None
        worker.deleteLater()
        self.scene.set_locked(False)
        self.library.setEnabled(True)
        self.inspector.editor.setEnabled(True)
        for action in self.edit_actions + [self.run_action, self.continuous_action]:
            action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.update_navigation()
        if self.closing:
            self.close()

    def show_ticket_results(self):
        if not hasattr(self, "ticket_results_window"):
            self.ticket_results_window = TicketResultsWindow(
                lambda: (
                    self.runner.graph.ticket_results
                    if self.runner
                    else self.root_graph.ticket_results
                ),
                self,
            )
        self.ticket_results_window.show()
        self.ticket_results_window.raise_()
        self.ticket_results_window.activateWindow()

    def closeEvent(self, event):
        if self.runner:
            self.closing = True
            self.stop_run()
            event.ignore()
        elif self.confirm_discard():
            self.root_graph.close()
            event.accept()
        else:
            self.closing = False
            event.ignore()


STYLE = """
QWidget { background: #1a2535; color: #d9e3f0; font-family: 'Segoe UI'; font-size: 12px; }
QMainWindow, QMenuBar, QToolBar { background: #202e41; }
QToolBar { spacing: 8px; padding: 8px; border-bottom: 1px solid #34465e; }
QToolButton, QPushButton { background: #30455d; border: 1px solid #46617f;
                          border-radius: 4px; padding: 7px 10px; }
QToolButton:hover, QPushButton:hover { background: #3a617c; }
QWidget:disabled { color: #6f7e91; }
QLineEdit, QComboBox, QTextEdit, QTreeWidget { background: #142031; border: 1px solid #34475f;
                                          border-radius: 4px; padding: 5px; }
QTreeWidget::item { padding: 5px 2px; }
QTreeWidget::item:selected { background: #31566c; }
QGroupBox { border: 1px solid #34475f; margin-top: 14px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; }
QLabel#sectionTitle { color: #70d3e3; font-weight: bold; padding: 10px 0; }
QSplitter::handle { background: #304056; width: 3px; }
QStatusBar { background: #111c2a; color: #a5bdd4; }
"""


def main():
    parser = argparse.ArgumentParser(description="Develop and run OpenCV node graphs")
    parser.add_argument("project", nargs="?", type=Path, help="Saved JSON project to open")
    parser.add_argument(
        "--starter", action="store_true", help="Open the synthetic inspection starter"
    )
    args = parser.parse_args()
    if args.project and args.starter:
        parser.error("Choose a project path or --starter")
    app = QApplication([sys.argv[0]])
    app.setApplicationName("OpenCV Dataflow")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    if args.starter:
        window.replace_graph(starter_graph(window.registry))
    elif args.project:
        try:
            window.replace_graph(Graph.load(args.project, window.registry), args.project)
        except Exception as exc:
            print(f"Unable to open project: {exc}", file=sys.stderr)
            sys.exit(2)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
