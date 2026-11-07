"""Desktop application entry point and inspector."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
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

from .activity_window import ActivityWindow
from .block_xml import block_xml_suffix, load_block_xml, save_block_xml
from .blocks import builtin_registry, demo_graph
from .calibration import calibration_templates
from .canvas import CanvasScene, CanvasView, Node, block_preview_images, pixmap
from .cloud_viewer import CloudViewer
from .core import DISPLAY_RESULT_FIELDS, Graph
from .diagnostic_viewer import DiagnosticViewer
from .diagnostics import exception_message
from .documentation_editor import DocumentationEditor
from .examples import starter_graph
from .filmstrip_viewer import FilmstripViewer
from .graphics_layers import RESULT_GRAPHICS, GraphicsBundle
from .image_queue import ImageQueue
from .kernel_editor import KERNEL_PARAMETERS, SHAPE_LABELS, KernelEditor
from .outcomes import INPUT_FAILURE_POLICIES
from .parallel_inspection import ParallelInspect
from .parameter_validation import ValidationError
from .payloads import ValuePacket, describe_value, packet_for, prepare_inputs
from .plugin_dialog import PluginDialogContext
from .plugins import load_plugin
from .program_documentation import export_program_documentation
from .project_compare import ProjectCompare
from .project_xml import XML_FILTER, export_project_xml
from .queued_execution import QueuedExecution, background_queue
from .recipe_editor import RecipeManager
from .recipes import validate_bindings
from .region_editor import RegionEditor
from .result_image import ResultImageWindow
from .run_profiles import PROFILES
from .scheduling import MAX_BRANCH_WORKERS
from .seed_editor import SeedEditor
from .sequence import share_native_sequence_runtime
from .ticket_window import TicketResultsWindow
from .tickets import TICKET_MODES
from .toolblock_editor import InterfaceDialog
from .toolblocks import ToolBlock, encapsulate
from .visual_style import BLOCK_COLORS, NODE_SIZES, PORT_COLORS, PORT_LABELS


def capture_display_results(blocks):
    """Hold one completed frame's result references while the worker keeps running."""
    for block in blocks:
        state = {name: getattr(block, name) for name in DISPLAY_RESULT_FIELDS}
        # Graphics policies may reassign fields on the retained Result object.
        state["last_result"] = copy.copy(state["last_result"])
        if isinstance(block, ParallelInspect):
            state["last_preview_index"] = block.last_preview_index
        yield block, state
        if isinstance(block, ParallelInspect):
            yield from capture_display_results((block.inspection,))
        elif hasattr(block, "graph"):
            yield from capture_display_results(block.graph.blocks.values())


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
        self.frame_has_results = True

    def stop(self):
        self.stop_requested.set()

    def run(self):
        queued = None
        pending = None
        interrupted = False
        published_at = None

        def cancelled():
            nonlocal interrupted
            if self.stop_requested.is_set():
                interrupted = True
            return interrupted

        def publish(has_results=True):
            self.frame_has_results = has_results
            self.ack.clear()
            self.frame.emit()
            self.ack.wait()

        try:
            if self.continuous and background_queue(self.graph) is not None:
                queued = QueuedExecution(self.graph, self.registry, self.stop_requested.is_set)
                queued.start()
            while not self.stop_requested.is_set():
                started = time.perf_counter()
                if queued is not None:
                    if not queued.run_one(cancelled):
                        break
                else:
                    self.graph.run(cancelled)
                if interrupted:
                    break
                self.duration = (time.perf_counter() - started) * 1000
                now = time.perf_counter()
                if (
                    not self.continuous
                    or published_at is None
                    or (now - published_at) * 1000 >= self.graph.run_profile.ui_interval_ms
                    or self.stop_requested.is_set()
                ):
                    # UI reads a stable snapshot; processing continues between publications.
                    publish()
                    pending = None
                    published_at = time.perf_counter()
                else:
                    pending = list(capture_display_results(self.graph.blocks.values()))
                if not self.continuous or (queued is None and self.graph.sources_exhausted()):
                    break
                self.stop_requested.wait(self.graph.run_profile.iteration_pause_ms / 1000)
        except (Exception, SystemExit) as exc:
            self.failed.emit(exception_message(exc))
        finally:
            if pending is not None:
                for block, state in pending:
                    for name, value in state.items():
                        setattr(block, name, value)
                publish()
            elif interrupted:
                publish(has_results=False)
            if queued is not None:
                queued.close()
            self.final_queue_states = {
                key: b.snapshot_runtime()
                for key, b in self.graph.blocks.items()
                if isinstance(b, ImageQueue)
            }
            self.graph.close()


class LivePreview(QThread):
    def __init__(self, block, inputs, revision, parent):
        super().__init__(parent)
        self.block, self.inputs, self.revision = block, inputs, revision
        self.error = ""

    def run(self):
        try:
            try:
                self.block.execute(self.inputs, self.isInterruptionRequested)
            finally:
                self.block._close_safely()
        except (Exception, SystemExit) as exc:
            self.error = exception_message(exc)


class Inspector(QWidget):
    live_delay_ms = 300

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.block = None
        self._region_image = None
        self._region_source = None
        self._region_inputs = {}
        self.fields, self.port_checks, self.level_fields, self.failure_fields = {}, [], {}, {}
        self.kernel_parameters = {}
        self._live_revision = 0
        self._live_due = False
        self._live_worker = None
        self._live_result = None
        self._preview_selection = None
        self.show_advanced = None
        self.image_popouts = {}
        self.draft_error = ""
        self.live_timer = QTimer(self)
        self.live_timer.setSingleShot(True)
        self.live_timer.timeout.connect(self.start_preview)
        layout = QVBoxLayout(self)
        title = QLabel("BLOCK INSPECTOR")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.caption = QLabel("Select a block to configure its tools and ports.")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.timing_label = QLabel()
        self.timing_label.setWordWrap(True)
        layout.addWidget(self.timing_label)
        self.live_results = QCheckBox("Live results")
        self.live_results.setEnabled(False)
        self.live_results.setToolTip(
            "Preview edited parameters on the last captured image. Apply configuration to save."
        )
        self.live_results.toggled.connect(self.toggle_live_results)
        preview_controls = QHBoxLayout()
        preview_controls.addWidget(self.live_results)
        self.run_block = QPushButton("Run block")
        self.run_block.setEnabled(False)
        self.run_block.setToolTip("Preview this block once using its last captured inputs.")
        self.run_block.clicked.connect(self.run_block_preview)
        preview_controls.addWidget(self.run_block)
        layout.addLayout(preview_controls)
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextFormat(Qt.TextFormat.PlainText)
        self.validation_label.setStyleSheet("color: #ffcb78;")
        self.validation_label.hide()
        layout.addWidget(self.validation_label)
        self.timing_timer = QTimer(self)
        self.timing_timer.setInterval(200)
        self.timing_timer.timeout.connect(self.refresh_timing)
        self.timing_timer.start()
        self.editor = QGroupBox("Configuration")
        self.form = QFormLayout(self.editor)
        self._plugin_dialog = None
        self._plugin_context = None
        self._plugin_output_timer = QTimer(self)
        self._plugin_output_timer.setSingleShot(True)
        self._plugin_output_timer.timeout.connect(self._notify_plugin_outputs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, 3)
        self.preview_choice = QComboBox()
        self.preview_choice.currentTextChanged.connect(self.show_preview)
        self.preview_choice.textActivated.connect(self.remember_preview_selection)
        image_controls = QHBoxLayout()
        image_controls.addWidget(self.preview_choice, 1)
        self.pop_out_image = QPushButton("Pop out…")
        self.pop_out_image.setEnabled(False)
        self.pop_out_image.setToolTip("Open a live window for the selected output or graphic.")
        self.pop_out_image.clicked.connect(self.open_image_popout)
        image_controls.addWidget(self.pop_out_image)
        layout.addLayout(image_controls)
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
        self.close_plugin_dialog()
        self.show_draft_error()
        advanced_open = (
            block is self.block
            and self.show_advanced is not None
            and self.show_advanced.isChecked()
        )
        self.show_advanced = None
        if block is not self.block:
            self._preview_selection = None
            self.stop_previews()
            self._region_image = None
            self._region_source = None
            self._region_inputs = {}
        self.block = block
        if hasattr(self.window, "diagnostics_view"):
            self.window.diagnostics_view.set_block(block)
        self.reference_capture = None
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields, self.port_checks, self.level_fields, self.failure_fields = {}, [], {}, {}
        self.kernel_parameters = {}
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
        self.description_field = QPlainTextEdit(block.description)
        self.description_field.setPlaceholderText("Shown when hovering over this block")
        self.description_field.setMaximumHeight(76)
        self.form.addRow("Description", self.description_field)
        documentation = QPushButton("Edit documentation…")
        documentation.clicked.connect(self.edit_documentation)
        self.form.addRow(documentation)
        xml_actions = QHBoxLayout()
        for title, callback in (
            ("Save block XML…", self.save_selected_block_xml),
            ("Load block XML…", self.load_selected_block_xml),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            xml_actions.addWidget(button)
        self.form.addRow(xml_actions)
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
        if getattr(block, "sequence_memory", False):
            button = QPushButton("Reset sequence memory")
            button.clicked.connect(self.reset_sequence)
            self.form.addRow(button)
        if getattr(block, "supports_reference_training", False):
            asset = block.trained_image
            self.form.addRow(
                QLabel(f"Trained reference: {asset['shape']}" if asset else "No stored reference")
            )
            for label, callback in (
                ("Capture reference from last input", self.capture_reference),
                ("Load trained reference…", self.load_reference),
                ("Export trained reference…", self.export_reference),
                ("Clear trained reference", self.clear_reference),
            ):
                button = QPushButton(label)
                button.clicked.connect(callback)
                if label.startswith("Capture"):
                    self.reference_capture = button
                button.setEnabled(
                    bool(asset)
                    if label.startswith(("Export", "Clear"))
                    else "image" in block.last_inputs
                    if label.startswith("Capture")
                    else True
                )
                self.form.addRow(button)
        if getattr(block, "supports_training", False):
            model = block.trained_model
            external = any(
                c.target == block.id and c.input == "model" for c in self.window.graph.connections
            )
            status = (
                "External model input is connected"
                if external
                else f"Recipe model: {block.recipe_model} (selected per image)"
                if block.recipe_model
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
        if getattr(block, "supports_kernel_editor", False):
            self.kernel_parameters = {key: block.parameters[key] for key in KERNEL_PARAMETERS}
            self.kernel_button = QPushButton()
            self.update_kernel_label()
            self.kernel_button.clicked.connect(self.edit_kernel)
            self.form.addRow(self.kernel_button)
        for key, value in block.parameters.items():
            if key in self.kernel_parameters:
                continue
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
            if isinstance(field, QComboBox):
                field.currentTextChanged.connect(self.schedule_live_preview)
            elif isinstance(field, QCheckBox):
                field.toggled.connect(self.schedule_live_preview)
            else:
                field.textChanged.connect(self.schedule_live_preview)
            field.setToolTip(block.parameter_help.get(key, ""))
            if key == "excluded_graphics" and getattr(block, "supports_graphics_selector", False):
                button = QPushButton("Choose graphics…")
                field.setParent(button)
                field.hide()
                button.clicked.connect(self.choose_result_graphics)
                self.form.addRow("Graphic layers", button)
            elif key in ("path", "model_path", "weights_path", "vocabulary_path"):
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
        if getattr(block, "file_source", False) and "source_mode" in block.defaults:
            for title, callback in (
                ("Select images…", self.browse_images),
                ("Select directory…", self.browse_directory),
            ):
                button = QPushButton(title)
                button.clicked.connect(callback)
                self.form.addRow(button)
        if getattr(block, "supports_region_editor", False):
            kind = getattr(block, "editor_kind", "region")
            draw_region = QPushButton(f"Draw / edit {kind}…")
            draw_region.clicked.connect(self.edit_region)
            self.form.addRow(draw_region)
        if getattr(block, "supports_seed_editor", False):
            edit_seeds = QPushButton("Edit watershed seeds…")
            edit_seeds.clicked.connect(self.edit_seeds)
            self.form.addRow(edit_seeds)
        if block.plugin_dialog_label:
            button = QPushButton(block.plugin_dialog_label)
            button.clicked.connect(self.open_plugin_dialog)
            self.form.addRow(button)
        self.show_advanced = QCheckBox("Show advanced")
        self.show_advanced.setToolTip("Show recipe, port, and node display settings.")
        advanced_panel = QWidget()
        advanced_form = QFormLayout(advanced_panel)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        self.form.addRow(self.show_advanced)
        self.form.addRow(advanced_panel)
        self.show_advanced.setChecked(advanced_open)
        advanced_panel.setVisible(advanced_open)
        self.show_advanced.toggled.connect(advanced_panel.setVisible)

        self.recipe_group = QGroupBox("Recipe overrides")
        self.recipe_group.setCheckable(True)
        self.recipe_group.setChecked(bool(block.recipe_bindings or block.recipe_model))
        recipe_form = QFormLayout(self.recipe_group)
        self.recipe_fields = {}
        for name in block.defaults:
            field = QLineEdit(block.recipe_bindings.get(name, ""))
            field.setPlaceholderText(f"settings.{name}")
            self.recipe_fields[name] = field
            recipe_form.addRow(name, field)
        self.recipe_model_field = None
        if getattr(block, "supports_training", False):
            self.recipe_model_field = QLineEdit(block.recipe_model)
            self.recipe_model_field.setPlaceholderText("models.part")
            recipe_form.addRow("Trained model", self.recipe_model_field)
        advanced_form.addRow(self.recipe_group)
        if not getattr(block, "preserve_image_sets", False):
            for port in block.inputs:
                if port.kind == "image":
                    level = QComboBox()
                    level.addItems(["largest", *map(str, range(9))])
                    level.setCurrentText(block.input_levels.get(port.name, "largest"))
                    level.setToolTip("Used when this input receives an image pyramid set.")
                    self.level_fields[port.name] = level
                    advanced_form.addRow(f"Pyramid level: {port.name}", level)
        self.node_size = QComboBox()
        for size in NODE_SIZES:
            self.node_size.addItem(size.capitalize(), size)
        self.node_size.setCurrentIndex(self.node_size.findData(block.node_size))
        self.node_size.setToolTip("Change the layout immediately. Small hides the node image.")
        self.node_size.currentIndexChanged.connect(self.resize_selected_node)
        advanced_form.addRow("Block size", self.node_size)
        self.thumb = QCheckBox("Show image on node")
        self.thumb.setChecked(block.thumbnail)
        advanced_form.addRow(self.thumb)
        self.node_image = QComboBox()
        self.node_image.addItem("Automatic", "")
        for port in block.outputs:
            if port.kind in ("image", "depth", "pyramid"):
                self.node_image.addItem(port.name, port.name)
        index = self.node_image.findData(block.preview_output)
        self.node_image.setCurrentIndex(max(0, index))
        advanced_form.addRow("Node image", self.node_image)
        self.graphics_choice = QComboBox()
        self.graphics_choice.addItem("Use project default", "inherit")
        self.graphics_choice.addItem("On", "on")
        self.graphics_choice.addItem("Off", "off")
        self.graphics_choice.setCurrentIndex(self.graphics_choice.findData(block.graphics_mode))
        self.graphics_choice.setToolTip(
            "Control optional result graphics. Declared outputs are preserved, along with "
            "exposed graphics enabled by the current run profile."
        )
        self.graphics_choice.currentIndexChanged.connect(self.change_graphics_mode)
        advanced_form.addRow("Result graphics", self.graphics_choice)
        for direction in ("inputs", "outputs"):
            ports = block.inputs if direction == "inputs" else block.output_ports
            for port in ports:
                label = "Result graphics" if port.name == RESULT_GRAPHICS else port.name
                check = QCheckBox(f"{label} ({port.kind})")
                check.setChecked(port.name in getattr(block, f"exposed_{direction}"))
                advanced_form.addRow(f"Expose {direction[:-1]}", check)
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
                    advanced_form.addRow(f"{port.name} failure policy", policy)
        self.apply_button = QPushButton("Apply configuration")
        self.apply_button.clicked.connect(self.apply)
        self.form.addRow(self.apply_button)
        self.editor.setEnabled(self.window.runner is None)
        self.validate_draft()
        self.refresh_result()

    def resize_selected_node(self):
        if self.block is None or self.window.scene.locked:
            return
        node = self.window.scene.nodes.get(self.block.id)
        size = self.node_size.currentData()
        if node is not None and node.block is self.block and size is not None:
            node.set_size(size)

    def edit_documentation(self):
        block = self.block
        if block is None or self.window.runner:
            return
        dialog = DocumentationEditor(block, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.documentation()
            if updated != block.documentation:
                block.documentation = updated
                self.window.mark_dirty()
        dialog.deleteLater()

    def save_selected_block_xml(self):
        block = self.block
        if block is None or self.window.runner:
            return
        suffix = block_xml_suffix(block.type_id)
        dialog = QFileDialog(self, "Save block XML", "", f"Block XML (*{suffix})")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix(suffix[1:])
        dialog.selectFile(f"block{suffix}")
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        selected = dialog.selectedFiles() if accepted else []
        dialog.deleteLater()
        if selected:
            try:
                save_block_xml(block, selected[0])
                self.window.statusBar().showMessage("Applied block configuration saved to XML")
            except Exception as exc:
                self.window.warn(str(exc))

    def load_selected_block_xml(self):
        block = self.block
        if block is None or self.window.runner:
            return
        suffix = block_xml_suffix(block.type_id)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load block XML", filter=f"Matching block XML (*{suffix});;Block XML (*.xml)"
        )
        if path:
            try:
                window = self.window
                owner = window.graph_stack[-1][1] if window.graph_stack else None
                loaded = load_block_xml(
                    window.graph,
                    block.id,
                    path,
                    window.registry,
                    owner=owner,
                    confirm_disconnect=self.confirm_block_xml_disconnect,
                )
                if loaded is not None:
                    window.scene.rebuild(block.id)
                    window.mark_dirty()
                    window.statusBar().showMessage("Block configuration loaded from XML")
            except Exception as exc:
                self.window.warn(str(exc))

    def confirm_block_xml_disconnect(self, wires):
        blocks = self.window.graph.blocks
        details = "\n".join(
            f"{blocks[wire.source].name}.{wire.output} → {blocks[wire.target].name}.{wire.input}"
            for wire in wires
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Disconnect wires and load block?")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(f"Loading this block will remove {len(wires)} existing connection(s).")
        dialog.setDetailedText(details)
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        confirmed = dialog.exec() == QMessageBox.StandardButton.Yes
        dialog.deleteLater()
        return confirmed

    def can_preview(self):
        b = self.block
        if b is None:
            return False
        eligible = b.preview_safe
        if eligible is None:
            eligible = type(b).__module__.startswith("opencv_dataflow.")
        return bool(
            eligible
            and self.window.runner is None
            and b.category != "Output"
            and b.block_type != "Output"
            and not b.ticket_source
            and not b.type_id.startswith(("input.", "output.", "flow.", "recipe.", "core."))
            and not getattr(b, "sequence_memory", False)
            and not getattr(b, "accept_events", False)
            and not getattr(b, "owns_workers", False)
            and not getattr(b, "queue_boundary", False)
            and any(port.name in b.last_inputs for port in b.inputs)
        )

    def toggle_live_results(self, enabled):
        if not enabled:
            self.cancel_live_preview()
            self.refresh_result()

    def cancel_live_preview(self):
        self.live_timer.stop()
        self._live_revision += 1
        self._live_due = False
        self._live_result = None
        if self._live_worker is not None:
            self._live_worker.requestInterruption()

    def stop_previews(self, wait=False):
        if self.live_results.isChecked():
            self.live_results.setChecked(False)
        else:
            self.toggle_live_results(False)
        if wait and self._live_worker is not None:
            worker, self._live_worker = self._live_worker, None
            worker.wait()
            worker.deleteLater()

    def hideEvent(self, event):
        self.close_plugin_dialog()
        self.stop_previews()
        super().hideEvent(event)

    def schedule_live_preview(self, *_):
        if self.validate_draft() is None:
            self.cancel_live_preview()
            self.refresh_result()
            return
        if self.live_results.isChecked() and self.can_preview():
            self._live_revision += 1
            self._live_due = False
            self.live_timer.start(self.live_delay_ms)

    def run_block_preview(self):
        if self._live_worker is not None or not self.can_preview():
            return
        self.live_timer.stop()
        self._live_due = False
        self._live_revision += 1
        self.start_preview()

    def start_preview(self):
        if not self.can_preview():
            return
        if self._live_worker is not None:
            self._live_due = True
            return
        b = self.block
        try:
            candidate = self.window.registry.create(b.type_id, parameters=self.read_parameters())
            candidate.restore_extra(b.serialize_extra())
            candidate.set_graphics_mode("on" if b.capture_graphics else "off")
            candidate.project_dir = b.project_dir
            candidate.recipe_bindings = dict(b.recipe_bindings)
            candidate.recipe_model = b.recipe_model
            candidate._recipes = b._recipes
            candidate._allow_recipe_activation = False
            inputs = {
                name: ValuePacket(value, b.input_payloads.get(name, {}))
                for name, value in b.last_inputs.items()
            }
        except (Exception, SystemExit) as exc:
            self.window.statusBar().showMessage(f"Block preview · {exception_message(exc)}")
            return
        worker = LivePreview(candidate, inputs, self._live_revision, self)
        self._live_worker = worker
        self.run_block.setEnabled(False)
        worker.finished.connect(self.finish_live_preview)
        worker.start()

    def finish_live_preview(self):
        worker, self._live_worker = self._live_worker, None
        if worker is None:
            return
        if worker.revision == self._live_revision:
            if worker.error:
                self.window.statusBar().showMessage(f"Block preview · {worker.error}")
            else:
                self._live_result = worker.block
                self.refresh_result()
        worker.deleteLater()
        self.run_block.setEnabled(self.can_preview() and not self.draft_error)
        if self._live_due:
            self._live_due = False
            self.start_preview()

    def browse_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images", "", "Images (*.*)")
        if paths:
            self.fields["paths"].setText(json.dumps(paths))
            self.fields["source_mode"].setCurrentText("files")
            self.fields["path_base"].setCurrentText("working_directory")

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select image directory")
        if directory:
            self.fields["directory"].setText(directory)
            self.fields["source_mode"].setCurrentText("directory")
            self.fields["path_base"].setCurrentText("working_directory")

    def browse_path(self, field):
        if self.block.category in ("Input", "ML"):
            title = (
                "Select model or index file"
                if self.block.category == "ML"
                else "Select input file"
            )
            path, _ = QFileDialog.getOpenFileName(self, title)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Select output path")
        if path:
            field.setText(path)

    def choose_result_graphics(self):
        field = self.fields["excluded_graphics"]
        excluded = {key: set(value) for key, value in json.loads(field.text()).items()}
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose result graphics")
        dialog.resize(640, 400)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Run in Debug to discover named layers from connected blocks."))
        layers = QTreeWidget()
        layers.setHeaderLabels(["Graphic", "Source", "Input"])
        layout.addWidget(layers)
        choices = []
        for wire in self.window.graph.connections:
            if wire.target != self.block.id:
                continue
            source = self.window.graph.blocks[wire.source]
            result = source.last_result
            bundle = result.outputs.get(wire.output) if result is not None else None
            if not isinstance(bundle, GraphicsBundle):
                continue
            for layer in bundle.layers:
                item = QTreeWidgetItem([layer.name, source.name, wire.input])
                layers.addTopLevelItem(item)
                if "spatial" not in layer.payload:
                    item.setDisabled(True)
                    item.setToolTip(0, "This graphic is a preview without an aligned overlay.")
                    continue
                checked = layer.name not in excluded.get(wire.input, set())
                item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                choices.append((wire.input, layer.name, item))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for slot, name, item in choices:
                names = excluded.setdefault(slot, set())
                if item.checkState(0) == Qt.CheckState.Checked:
                    names.discard(name)
                else:
                    names.add(name)
            choices = {key: sorted(value) for key, value in excluded.items() if value}
            field.setText(json.dumps(choices))
        dialog.deleteLater()

    def read_parameters(self, *, validate=True):
        b = self.block
        parameters = dict(self.kernel_parameters)
        for key, field in self.fields.items():
            initial = b.defaults[key]
            try:
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
                        raise ValueError(f"requires a JSON {type(initial).__name__}")
                else:
                    value = field.text()
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError) as exc:
                label = key.replace("_", " ").capitalize()
                raise ValidationError(f"{label}: {exc}", key) from exc
            parameters[key] = value
        json.dumps(parameters, allow_nan=False)
        if validate:
            b.validate_parameters(parameters)
        return parameters

    def validate_draft(self):
        if self.block is None:
            return None
        try:
            parameters = self.read_parameters()
        except (Exception, SystemExit) as exc:
            message = str(exc) if isinstance(exc, ValidationError) else exception_message(exc)
            self.show_draft_error(message, exc.fields if isinstance(exc, ValidationError) else ())
            return None
        self.show_draft_error()
        return parameters

    def show_draft_error(self, message="", keys=()):
        self.draft_error = message
        self.validation_label.setText(f"Invalid configuration · {message}" if message else "")
        self.validation_label.setVisible(bool(message))
        if self.block is None:
            return
        for key, field in self.fields.items():
            invalid = key in keys
            field.setStyleSheet("border: 1px solid #ffac70;" if invalid else "")
            help_text = self.block.parameter_help.get(key, "")
            limits = self.block.parameter_ranges.get(key)
            detail = f"Value must be {limits.describe()}." if limits else ""
            if invalid:
                detail = message
            field.setToolTip("\n".join(part for part in (help_text, detail) if part))
        if self.kernel_parameters:
            invalid = any(key in self.kernel_parameters for key in keys)
            self.kernel_button.setStyleSheet("border: 1px solid #ffac70;" if invalid else "")
            self.kernel_button.setToolTip(message if invalid else "")
        if hasattr(self, "apply_button"):
            self.apply_button.setEnabled(not message)
        self.run_block.setEnabled(not message and self.can_preview() and self._live_worker is None)
        node = self.window.scene.nodes.get(self.block.id)
        if node is not None and node.block is self.block:
            node.set_draft_issue(message)

    def update_kernel_label(self):
        shape = self.kernel_parameters["shape"]
        self.kernel_button.setText(f"Edit kernel… ({SHAPE_LABELS.get(shape, shape)})")

    def open_plugin_dialog(self):
        block = self.block
        if block is None or not block.plugin_dialog_label or self.window.runner:
            return
        if self._plugin_dialog is not None:
            self._plugin_dialog.raise_()
            self._plugin_dialog.activateWindow()
            return
        try:
            original = self.read_parameters(validate=False)
        except (Exception, SystemExit) as exc:
            message = str(exc) if isinstance(exc, ValidationError) else exception_message(exc)
            self.show_draft_error(message, exc.fields if isinstance(exc, ValidationError) else ())
            return
        fields = self.fields
        draft = copy.deepcopy(original)
        context = None
        try:
            context = PluginDialogContext(
                self.window.graph,
                block,
                input_levels={key: field.currentText() for key, field in self.level_fields.items()},
            )
            widget = block.create_plugin_dialog(self, draft, context)
            if not isinstance(widget, QDialog):
                raise TypeError("create_plugin_dialog must return a PySide6 QDialog")
            if self.fields is not fields or self.window.runner:
                context._close()
                widget.deleteLater()
                return
            self._plugin_dialog = widget
            self._plugin_context = context
            if widget.parentWidget() is not self:
                widget.setParent(self, widget.windowFlags())
            widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            def finish(status):
                self._release_plugin_context(context)
                self.finish_plugin_dialog(widget, block, original, draft, status)

            def destroyed():
                self._release_plugin_context(context)
                if self._plugin_dialog is widget:
                    self._plugin_dialog = None

            widget.finished.connect(finish)
            widget.destroyed.connect(destroyed)
            if block.plugin_dialog_modeless:
                widget.setWindowModality(Qt.WindowModality.NonModal)
                widget.show()
            else:
                widget.open()
        except (Exception, SystemExit) as exc:
            self.close_plugin_dialog()
            if context is not None:
                context._close()
            block.logger.exception("Plugin dialog could not open")
            self.window.warn(f"{block.title} dialog · {exception_message(exc)}")

    def _release_plugin_context(self, context):
        context._close()
        if self._plugin_context is context:
            self._plugin_context = None
            self._plugin_output_timer.stop()

    def close_plugin_dialog(self):
        if self._plugin_context is not None:
            self._release_plugin_context(self._plugin_context)
        dialog, self._plugin_dialog = self._plugin_dialog, None
        if dialog is not None:
            try:
                QDialog.reject(dialog)
            finally:
                dialog.deleteLater()

    def plugin_outputs_published(self):
        context = self._plugin_context
        if context is not None and context._outputs_published():
            if not self._plugin_output_timer.isActive():
                self._plugin_output_timer.start(0)

    def _notify_plugin_outputs(self):
        context = self._plugin_context
        if context is None:
            return
        newer_outputs = context._notify_outputs(self.block.logger.exception)
        if newer_outputs and self._plugin_context is context:
            self._plugin_output_timer.start(0)

    def finish_plugin_dialog(self, dialog, block, original, draft, status):
        if self._plugin_dialog is not dialog:
            return
        self._plugin_dialog = None
        dialog.deleteLater()
        if status != QDialog.DialogCode.Accepted or self.block is not block:
            return
        try:
            parameters = json.loads(json.dumps(draft, allow_nan=False))
            if parameters.keys() != original.keys():
                raise ValueError("The dialog must keep the block's declared parameter names")
            for name, value in parameters.items():
                expected = type(original[name])
                if expected is float and type(value) is int:
                    parameters[name] = float(value)
                elif type(value) is not expected:
                    raise ValueError(f"{name} must have type {expected.__name__}")
            updates = {name: value for name, value in parameters.items() if value != original[name]}
            if not updates:
                return
            current = self.read_parameters(validate=False)
            conflicts = [
                name
                for name, value in updates.items()
                if current[name] not in (original[name], value)
            ]
            if conflicts:
                raise ValueError(f"Inspector and dialog both changed {', '.join(conflicts)}")
            parameters = {**current, **updates}
            block.validate_parameters(parameters)
            if parameters == current:
                return
            for name, field in self.fields.items():
                value = parameters[name]
                if value == current[name]:
                    continue
                with QSignalBlocker(field):
                    if isinstance(field, QComboBox):
                        field.setCurrentText(value)
                    elif isinstance(field, QCheckBox):
                        field.setChecked(value)
                    else:
                        field.setText(
                            json.dumps(value) if isinstance(value, (list, dict)) else str(value)
                        )
            if self.kernel_parameters:
                self.kernel_parameters = {key: parameters[key] for key in self.kernel_parameters}
                self.update_kernel_label()
            self.schedule_live_preview()
        except (Exception, SystemExit) as exc:
            block.logger.exception("Plugin dialog edits could not be staged")
            self.window.warn(f"{block.title} dialog · {exception_message(exc)}")

    def edit_kernel(self):
        try:
            dialog = KernelEditor(self.block, self.kernel_parameters, self)
        except (ValueError, TypeError, KeyError) as exc:
            self.window.warn(str(exc))
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.parameters
            if updated != self.kernel_parameters:
                self.kernel_parameters = updated
                self.update_kernel_label()
                self.schedule_live_preview()
        dialog.deleteLater()

    def edit_seeds(self):
        block = self.block
        if block is None or self.window.runner:
            return
        try:
            parameters = self.read_parameters()
            image = block.last_inputs.get("image")
            connection = next(
                (
                    wire
                    for wire in self.window.graph.connections
                    if wire.target == block.id and wire.input == "image"
                ),
                None,
            )
            if connection is not None:
                result = self.window.graph.blocks[connection.source].last_result
                if result is not None and connection.output in result.outputs:
                    candidate = self.window.registry.create(block.type_id, parameters=parameters)
                    candidate.input_levels = {"image": self.level_fields["image"].currentText()}
                    values, _ = prepare_inputs(
                        candidate, {"image": packet_for(result, connection.output)}
                    )
                    if isinstance(values["image"], np.ndarray):
                        image = values["image"]
            if image is None:
                raise ValueError(
                    "Connect an image and Run once in Debug to capture it before editing seeds."
                )
            dialog = SeedEditor(image, parameters, self)
        except (ValueError, TypeError) as exc:
            self.window.warn(str(exc))
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for key, points in dialog.points.items():
                self.fields[key].setText(json.dumps(points))
        dialog.deleteLater()

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

    def reset_sequence(self):
        if self.block is not None and not self.window.runner:
            self.block.reset_runtime()
            self.block.clear_result()
            self.window.scene.update()
            self.refresh_result()

    def reference_changed(self):
        self.window.graph.invalidate()
        self.window.mark_dirty()
        self.window.scene.rebuild(self.block.id)
        self.select(self.block)

    def capture_reference(self):
        if self.block is not None and not self.window.runner:
            try:
                self.block.train_reference(self.block.last_inputs["image"])
                self.reference_changed()
            except (ValueError, TypeError, KeyError) as exc:
                self.window.warn(str(exc))

    def load_reference(self):
        if self.block is None or self.window.runner:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference", filter="Reference JSON (*.json)"
        )
        if path:
            try:
                self.block.restore_extra(
                    {"trained_image": json.loads(Path(path).read_text("utf-8"))}
                )
                self.reference_changed()
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.window.warn(str(exc))

    def export_reference(self):
        if self.block is None or self.window.runner or self.block.trained_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reference", filter="Reference JSON (*.json)"
        )
        if path:
            try:
                with Path(path).open("x", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(self.block.trained_image, indent=2, allow_nan=False) + "\n"
                    )
            except (ValueError, OSError) as exc:
                self.window.warn(str(exc))

    def clear_reference(self):
        if self.block is not None and not self.window.runner:
            self.block.trained_image = None
            self.reference_changed()

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
        parameters = self.validate_draft()
        if parameters is None:
            return False
        try:
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
            if self.window.graph_stack and RESULT_GRAPHICS not in exposed["outputs"]:
                owner = self.window.graph_stack[-1][1]
                if any(
                    binding.block == b.id and binding.port == RESULT_GRAPHICS
                    for binding in owner.output_bindings.values()
                ):
                    raise ValueError("Remove the public terminal before hiding Result graphics")
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
            recipe_bindings = {
                key: field.text().strip()
                for key, field in self.recipe_fields.items()
                if self.recipe_group.isChecked() and field.text().strip()
            }
            recipe_model = (
                self.recipe_model_field.text().strip()
                if self.recipe_group.isChecked() and self.recipe_model_field
                else ""
            )
            validate_bindings(b, recipe_bindings, recipe_model)
            if recipe_model and any(
                c.target == b.id and c.input == "model" for c in self.window.graph.connections
            ):
                raise ValueError("Disconnect the model input or clear the recipe model binding")
            if isinstance(b, ToolBlock):
                kind, subtype = self.type_field.text().strip(), self.subtype_field.text().strip()
                if not kind or not subtype:
                    raise ValueError("ToolBlock type and subtype cannot be empty")
                b.block_type, b.subtype = kind, subtype
            b.name, b.parameters = name, parameters
            b.description = self.description_field.toPlainText()
            b.recipe_bindings, b.recipe_model = recipe_bindings, recipe_model
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

    def change_graphics_mode(self, index):
        if self.block is not None and not self.window.runner:
            self.stop_previews()
            self.block.set_graphics_mode(self.graphics_choice.itemData(index))
            self.window.scene.update()
            self.window.mark_dirty()
            self.refresh_result()

    def refresh_timing(self):
        block = self.block
        runner = self.window.runner
        if block is not None and runner is not None:
            graph = runner.graph
            for _, owner in self.window.graph_stack:
                graph = graph.blocks[owner.id].graph
            block = graph.blocks.get(block.id, block)
        if block is None:
            self.timing_label.clear()
            return
        timing = block.timing.snapshot()
        fields = [
            f"{title}: {timing[key]:.2f} ms" if timing[key] is not None else f"{title}: —"
            for title, key in (
                ("Current", "current_ms"),
                ("Last", "last_ms"),
                ("Average", "average_ms"),
            )
        ]
        self.timing_label.setText(" · ".join(fields) + f" · {timing['count']} activations")

    def refresh_result(self):
        self.refresh_timing()
        available = self.can_preview()
        automatic = available and bool(self.fields or self.kernel_parameters)
        self.live_results.setEnabled(automatic)
        self.run_block.setEnabled(available and not self.draft_error and self._live_worker is None)
        if not automatic and self.live_results.isChecked():
            self.stop_previews()
            return
        self.refresh_image_popouts()
        self.preview_choice.blockSignals(True)
        self.preview_choice.clear()
        b = self._live_result or self.block
        if b:
            if self.reference_capture is not None:
                self.reference_capture.setEnabled("image" in b.last_inputs)
            self.preview_choice.addItems(
                [f"{kind} / {name}" for kind, name, _ in block_preview_images(b)]
            )
            self.preview_choice.addItems(
                [
                    f"Input / {name}"
                    for name, image in b.last_inputs.items()
                    if hasattr(image, "shape")
                ]
            )
        index = self.preview_choice.findText(self._preview_selection or "")
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
            timing = b.timing.snapshot()
            average = f"{timing['average_ms']:.2f}" if timing["average_ms"] is not None else "—"
            self.details.setPlainText(
                ("BLOCK PREVIEW · " if self._live_result is not None else "")
                + f"{b.state.upper()}   ·   {b.elapsed_ms:.2f} ms   ·   score {score}\n"
                + f"Average {average} ms · {timing['count']} activations\n"
                + (b.last_rejection or b.error or (result.message if result else "Not executed"))
                + "\n\n"
                + json.dumps(
                    {
                        "outputs": outputs,
                        "payloads": result.payloads,
                        "recipe": b.last_recipe_ref,
                        "recipe_parameters": b.last_recipe_parameters,
                        "timing": b.timing.snapshot(),
                    }
                    if result
                    else {"timing": timing},
                    indent=2,
                    default=str,
                )
            )

    def remember_preview_selection(self, choice):
        self._preview_selection = choice

    def show_preview(self, choice):
        self.pop_out_image.setEnabled(choice.startswith(("Output / ", "Result / ", "Graphic / ")))
        self.image.clear()
        block = self._live_result or self.block
        if block and " / " in choice:
            prefix, name = choice.split(" / ", 1)
            images = {
                "Input": block.last_inputs,
                "Output": block.last_output_images,
                "Result": block.last_output_images,
                "Graphic": block.last_graphics,
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

    def open_image_popout(self):
        choice = self.preview_choice.currentText()
        if self.block is None or not choice.startswith(("Output / ", "Result / ", "Graphic / ")):
            return
        kind, name = choice.split(" / ", 1)
        path = tuple(
            getattr(owner, "_parent_node_id", owner.id) for _, owner in self.window.graph_stack
        )
        key = (path, self.block.id, kind, name)
        viewer = self.image_popouts.get(key)
        if viewer is None:
            viewer = ResultImageWindow(kind, name, self.window)
            self.image_popouts[key] = viewer
            viewer.finished.connect(lambda: self.image_popouts.pop(key, None))
        viewer.refresh(self._live_result or self.block)
        viewer.show()
        viewer.raise_()

    def refresh_image_popouts(self):
        for (path, block_id, _, _), viewer in list(self.image_popouts.items()):
            graph = self.window.root_graph
            for node_id in path:
                graph = getattr(graph.blocks.get(node_id), "graph", None)
                if graph is None:
                    break
            block = graph.blocks.get(block_id) if graph is not None else None
            if block is None:
                viewer.close()
            else:
                if block is self.block and self._live_result is not None:
                    block = self._live_result
                viewer.refresh(block)

    def close_image_popouts(self):
        for viewer in list(self.image_popouts.values()):
            viewer.close()

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
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 2, 2, 0)
        sidebar_layout.setSpacing(2)
        sidebar_layout.addWidget(self.view.overview, alignment=Qt.AlignmentFlag.AlignRight)
        sidebar_layout.addWidget(self.inspector, stretch=1)
        split = QSplitter()
        split.addWidget(library_panel)
        split.addWidget(self.view)
        split.addWidget(sidebar)
        split.setSizes([210, 940, 350])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.diagnostics_dock = QDockWidget("Block diagnostics", self)
        self.diagnostics_dock.setObjectName("blockDiagnosticsDock")
        self.diagnostics_view = DiagnosticViewer(lambda: self.root_graph.activity, self)
        self.diagnostics_dock.setWidget(self.diagnostics_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.diagnostics_dock)
        self.diagnostics_dock.hide()
        self.filmstrip_dock = QDockWidget("Image results filmstrip", self)
        self.filmstrip_dock.setObjectName("imageResultsFilmstrip")
        self.filmstrip_view = FilmstripViewer(lambda: self.root_graph.filmstrip, self)
        self.filmstrip_dock.setWidget(self.filmstrip_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.filmstrip_dock)
        self.filmstrip_dock.hide()
        self.scene.selected_block.connect(self.inspector.select)
        self.scene.edited.connect(self.mark_dirty)
        self.scene.layout_changed.connect(self.mark_layout_dirty)
        self.scene.warning.connect(self.warn)
        self.scene.open_block.connect(self.enter_toolblock)
        self.make_actions()
        self.production_label = QLabel()
        self.statusBar().addPermanentWidget(self.production_label)
        self.production_timer = QTimer(self)
        self.production_timer.setInterval(500)
        self.production_timer.timeout.connect(self.refresh_production)
        self.production_timer.start()
        self.refresh_production()
        self.statusBar().showMessage("Demo ready · Run once to inspect two shapes")
        QTimer.singleShot(0, self.view.restore_view)

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
            ("Export program documentation…", None, self.export_program_guide),
            ("Export project XML…", None, self.export_project_snapshot),
            ("Compare project XML…", None, self.compare_project_snapshots),
            ("Load Python plugin…", None, self.open_plugin),
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
        toolbar.addWidget(QLabel("Run profile"))
        self.profile_choice = QComboBox()
        self.profile_choice.addItems(PROFILES)
        self.profile_choice.setToolTip(
            "Debug: full diagnostics. Balanced: sampled previews. Lean: warnings/errors, "
            "summary tickets, no image previews. "
            "Processing outputs and input isolation are preserved."
        )
        self.profile_choice.currentTextChanged.connect(self.change_run_profile)
        toolbar.addWidget(self.profile_choice)
        toolbar.addWidget(QLabel("Graphics default"))
        self.project_graphics_choice = QComboBox()
        self.project_graphics_choice.addItem("Run profile", "inherit")
        self.project_graphics_choice.addItem("On", "on")
        self.project_graphics_choice.addItem("Off", "off")
        self.project_graphics_choice.setToolTip(
            "Set the default for optional result graphics across this project."
        )
        self.project_graphics_choice.currentIndexChanged.connect(self.change_project_graphics)
        toolbar.addWidget(self.project_graphics_choice)
        toolbar.addWidget(QLabel("Branch workers"))
        self.branch_workers_choice = QComboBox()
        self.branch_workers_choice.addItem("Auto", "auto")
        self.branch_workers_choice.addItem("Serial", 1)
        for count in range(2, MAX_BRANCH_WORKERS + 1):
            self.branch_workers_choice.addItem(str(count), count)
        self.branch_workers_choice.setToolTip("Parallel worker limit for the current graph.")
        self.branch_workers_choice.currentIndexChanged.connect(self.change_branch_workers)
        toolbar.addWidget(self.branch_workers_choice)
        toolbar.addSeparator()
        toolbar.addAction("Fit graph", lambda: self.view.fit_graph())
        results_menu = self.menuBar().addMenu("&Results")
        results_menu.addAction("Ticket results…", self.show_ticket_results)
        results_menu.addAction("3D point-cloud viewer…", self.show_cloud_viewer)
        results_menu.addAction(self.filmstrip_dock.toggleViewAction())
        self.diagnostics_action = self.diagnostics_dock.toggleViewAction()
        self.diagnostics_action.setText("Block diagnostics")
        self.diagnostics_action.setShortcut("Ctrl+Shift+D")
        results_menu.addAction(self.diagnostics_action)
        results_menu.addAction("Ticket activity timeline…", lambda: self.show_activity("ticket"))
        results_menu.addAction("Block concurrency…", lambda: self.show_activity("concurrency"))
        self.edit_actions.append(
            results_menu.addAction("Reset running counts", self.reset_production)
        )
        self.validation_action = results_menu.addAction(
            "Validation image sets…", self.show_validation
        )
        self.edit_actions.append(self.validation_action)
        recipes_menu = self.menuBar().addMenu("&Recipes")
        self.recipe_action = recipes_menu.addAction("Manage recipes…", self.manage_recipes)
        self.edit_actions.append(self.recipe_action)
        toolbar.addAction("Delete selected", self.scene.delete_selected)
        workspace = self.menuBar().addMenu("&Workspace")
        workspace.addAction("Color legend…", self.show_color_legend)
        self.edit_actions.append(workspace.addAction("Auto arrange blocks", self.arrange_workspace))
        for title, shortcut, callback in (
            ("Group selected nodes…", "Ctrl+G", self.group_nodes),
            ("Ungroup selection", "Ctrl+Shift+G", self.scene.ungroup_selected),
        ):
            action = workspace.addAction(title, callback)
            action.setShortcut(shortcut)
            self.edit_actions.append(action)
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

    def show_color_legend(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Canvas color legend")
        dialog.resize(510, 650)
        layout = QVBoxLayout(dialog)
        explanation = QLabel(
            "Sockets and wires: data type. Node borders: tool category.\n"
            "Node headers: run status and score intensity.\n"
            "! Invalid configuration or input. × Execution exception.\n"
            "A white outline highlights selection without replacing type colors."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        for heading, colors, labels in (
            ("Connections", PORT_COLORS, PORT_LABELS),
            ("Block borders", BLOCK_COLORS, {}),
        ):
            form.addRow(QLabel(f"<b>{heading}</b>"))
            for kind, color in colors.items():
                swatch = QLabel()
                swatch.setFixedSize(28, 16)
                swatch.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
                form.addRow(swatch, QLabel(labels.get(kind, kind)))
        scroll.setWidget(content)
        layout.addWidget(scroll)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def arrange_workspace(self):
        if self.scene.auto_arrange():
            self.view.fit_graph()
            self.statusBar().showMessage("Blocks arranged from inputs to outputs")

    def group_nodes(self):
        if self.runner:
            return
        name, accepted = QInputDialog.getText(self, "Group nodes", "Group name:")
        if accepted:
            try:
                self.scene.group_selected(name)
            except ValueError as exc:
                self.warn(str(exc))

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
        try:
            if type_id.startswith("template:"):
                definition = self.toolblock_templates[int(type_id.split(":")[1])]
                block = ToolBlock.from_definition(definition, self.registry)
            else:
                block = self.registry.create(type_id)
        except (Exception, SystemExit) as exc:
            self.warn(exception_message(exc))
            return
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
        self.branch_workers_choice.blockSignals(True)
        self.branch_workers_choice.setCurrentIndex(
            self.branch_workers_choice.findData(graph.branch_workers)
        )
        self.branch_workers_choice.blockSignals(False)
        self.scene.graph = graph
        self.inspector.select(None)
        self.scene.rebuild(selected)
        self.update_navigation()
        self.view.restore_view()

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
        self.inspector.refresh_image_popouts()
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
        self.inspector.close_image_popouts()
        self.root_graph.close()
        self.graph_stack = []
        self.show_graph(graph)
        self.populate_library()
        self.path, self.dirty = path, False
        self.profile_choice.blockSignals(True)
        self.profile_choice.setCurrentText(graph.run_profile.name)
        self.profile_choice.blockSignals(False)
        self.project_graphics_choice.blockSignals(True)
        self.project_graphics_choice.setCurrentIndex(
            self.project_graphics_choice.findData(graph.graphics_mode)
        )
        self.project_graphics_choice.blockSignals(False)
        self.filmstrip_view.capture.setChecked(graph.run_profile.name != "lean")
        self.update_title()
        self.view.restore_view()
        self.statusBar().showMessage(f"Job ready · {len(graph.blocks)} blocks")

    def new_project(self):
        if not self.runner and self.confirm_discard():
            self.replace_graph(Graph())

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
            previous_directory = self.root_graph.project_dir
            self.root_graph.save(path)
            self.path, self.dirty = path, False
            if self.root_graph.project_dir != previous_directory:
                self.inspector.close_plugin_dialog()
            self.update_title()
            return True
        except Exception as exc:
            self.warn(str(exc))
            return False

    def export_program_guide(self):
        if self.runner:
            return
        title = self.path.stem if self.path else "Untitled program"
        proposed = (
            self.path.with_name(f"{title}-documentation.html")
            if self.path
            else Path("program-documentation.html")
        )
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export program documentation", str(proposed), "HTML (*.html *.htm)"
        )
        if not selected:
            return
        path = Path(selected)
        if not path.suffix:
            path = path.with_suffix(".html")
        try:
            if self.path and path.resolve() == self.path.resolve():
                raise ValueError("Choose a separate file for program documentation")
            export_program_documentation(self.root_graph, path, title)
            self.statusBar().showMessage(f"Program documentation exported · {path.name}")
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            self.warn(str(exc))

    def export_project_snapshot(self):
        if self.runner:
            return
        directory = str(self.path.parent) if self.path else ""
        filename = self.path.with_suffix(".xml").name if self.path else "project.xml"
        dialog = QFileDialog(self, "Export project XML", directory, XML_FILTER)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("xml")
        dialog.selectFile(filename)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        selected = dialog.selectedFiles() if accepted else []
        dialog.deleteLater()
        if not selected:
            return
        path = Path(selected[0])
        try:
            if path.suffix.lower() != ".xml" or (
                self.path and path.resolve() == self.path.resolve()
            ):
                raise ValueError("Choose a separate .xml file for the snapshot")
            export_project_xml(self.root_graph.to_dict(), path)
            self.statusBar().showMessage(f"Project XML exported · {path.name}")
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            self.warn(str(exc))

    def compare_project_snapshots(self):
        dialog = ProjectCompare(self)
        dialog.exec()
        dialog.deleteLater()

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
        block = self.inspector.block
        if block is None or not block.plugin_dialog_modeless:
            self.inspector.close_plugin_dialog()
        self.inspector.stop_previews()
        # Worker owns a separate runtime graph. Qt paints stable display snapshots.
        try:
            execution_graph = Graph.from_dict(
                self.root_graph.to_dict(), self.registry, project_dir=self.root_graph.project_dir
            )
            execution_graph.copy_runtime_from(self.root_graph)
            share_native_sequence_runtime(execution_graph, self.root_graph)
            execution_graph.bind_activity(self.root_graph.activity)
            execution_graph.filmstrip = self.root_graph.filmstrip
            execution_graph.filmstrip_enabled = self.filmstrip_view.capture.isChecked()
            if any(getattr(b, "file_source", False) for b in self.root_graph.blocks.values()):
                self.filmstrip_dock.show()
            for key, block in self.root_graph.blocks.items():
                execution_graph.blocks[key].copy_results_from(block)
        except (Exception, SystemExit) as exc:
            self.warn(exception_message(exc))
            return
        self.runner = Runner(execution_graph, continuous, self.registry)
        self.runner.frame.connect(self.on_frame)
        self.runner.failed.connect(self.warn)
        self.runner.finished.connect(self.on_finished)
        self.scene.set_locked(True)
        self.library.setEnabled(False)
        self.inspector.editor.setEnabled(False)
        self.inspector.refresh_result()
        for action in self.edit_actions + [self.run_action, self.continuous_action]:
            action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.profile_choice.setEnabled(False)
        self.project_graphics_choice.setEnabled(False)
        self.branch_workers_choice.setEnabled(False)
        self.statusBar().showMessage("Running…")
        self.runner.start()

    def reset_flow_state(self):
        if not self.runner:
            self.root_graph.invalidate()
            self.scene.update()
            self.inspector.refresh_result()
            self.statusBar().showMessage(
                "Flow and sequence memory cleared; a new ticket session will start"
            )

    def on_frame(self):
        error = None
        try:
            self.root_graph.copy_runtime_from(self.runner.graph)
            share_native_sequence_runtime(self.root_graph, self.runner.graph)
            if not self.runner.frame_has_results:
                return
            for key, executed in self.runner.graph.blocks.items():
                self.root_graph.blocks[key].copy_results_from(executed)
            self.scene.update()
            self.filmstrip_view.refresh()
            self.inspector.refresh_result()
            states = [b.state for b in self.root_graph.blocks.values()]
            self.statusBar().showMessage(
                f"{len(states)} blocks · {self.runner.duration:.1f} ms · "
                f"{states.count('pass')} pass · {states.count('fail')} fail · "
                f"{states.count('error')} errors · {states.count('skipped')} skipped"
                f" · {states.count('waiting')} waiting · {states.count('inactive')} inactive"
            )
        except (Exception, SystemExit) as exc:
            self.runner.stop()
            error = exception_message(exc)
        finally:
            self.runner.ack.set()
        if error is not None:
            self.warn(error)
        else:
            self.inspector.plugin_outputs_published()

    def stop_run(self):
        if self.runner:
            self.runner.stop()
            self.statusBar().showMessage("Stopping after the current block returns…")

    def on_finished(self):
        worker = self.runner
        # finished() can precede native thread-local cleanup; join before release.
        worker.wait()
        self.root_graph.ticket_results.restore(worker.graph.ticket_results.snapshot())
        self.root_graph.recipes.copy_runtime_from(worker.graph.recipes)
        self.root_graph.copy_timing_from(worker.graph)
        error = None
        try:
            self.root_graph.copy_source_runtime_from(worker.graph)
        except (Exception, SystemExit) as exc:
            error = exception_message(exc)
        for key, snapshot in worker.final_queue_states.items():
            block = self.root_graph.blocks.get(key)
            if isinstance(block, ImageQueue):
                block.restore_runtime(snapshot)
                block.update_metrics()
        self.scene.update()
        self.runner = None
        self.inspector.refresh_result()
        worker.deleteLater()
        self.scene.set_locked(False)
        self.library.setEnabled(True)
        self.inspector.editor.setEnabled(True)
        for action in self.edit_actions + [self.run_action, self.continuous_action]:
            action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.profile_choice.setEnabled(True)
        self.project_graphics_choice.setEnabled(True)
        self.branch_workers_choice.setEnabled(True)
        self.update_navigation()
        if error is not None:
            self.warn(error)
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

    def show_cloud_viewer(self):
        if not hasattr(self, "cloud_viewer"):
            self.cloud_viewer = CloudViewer(lambda: self.graph, self)
        self.cloud_viewer.refresh()
        self.cloud_viewer.show()
        self.cloud_viewer.raise_()

    def show_activity(self, mode):
        name = f"activity_{mode}_window"
        if not hasattr(self, name):
            setattr(self, name, ActivityWindow(lambda: self.root_graph.activity, mode, self))
        window = getattr(self, name)
        window.show()
        window.raise_()
        window.activateWindow()

    def refresh_production(self):
        consumer = (
            self.runner.graph.ticket_results if self.runner else self.root_graph.ticket_results
        )
        stats = consumer.statistics()
        rate = f"{stats['pass_rate']:.1f}%" if stats["pass_rate"] is not None else "—"
        average = f"{stats['average_ms']:.2f} ms" if stats["average_ms"] is not None else "—"
        self.production_label.setText(
            f"Total {stats['total']} · Pass {stats['pass']} · Fail {stats['fail']} · "
            f"Pass rate {rate} · Avg end-to-end {average}"
        )

    def change_run_profile(self, name):
        if not self.runner:
            self.root_graph.set_run_profile(name)
            self.filmstrip_view.capture.setChecked(name != "lean")
            self.mark_dirty()
            self.scene.update()
            self.inspector.refresh_result()
            self.statusBar().showMessage(
                f"Run profile: {name}. Run once to collect its current diagnostics."
            )

    def change_project_graphics(self, index):
        if not self.runner:
            self.inspector.stop_previews()
            self.root_graph.set_graphics_mode(self.project_graphics_choice.itemData(index))
            self.scene.update()
            self.mark_dirty()
            self.inspector.refresh_result()

    def change_branch_workers(self, index):
        if not self.runner:
            self.graph.set_branch_workers(self.branch_workers_choice.itemData(index))
            self.mark_dirty()

    def reset_production(self):
        if not self.runner:
            self.root_graph.ticket_results.reset_statistics()
            self.refresh_production()

    def show_validation(self):
        from .validation_editor import ValidationManager

        if not self.runner:
            ValidationManager(self).exec()

    def manage_recipes(self):
        if not self.runner:
            RecipeManager(self).exec()

    def closeEvent(self, event):
        if self.runner:
            self.closing = True
            self.stop_run()
            event.ignore()
        elif self.confirm_discard():
            self.inspector.close_plugin_dialog()
            self.inspector.close_image_popouts()
            self.inspector.stop_previews(wait=True)
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
