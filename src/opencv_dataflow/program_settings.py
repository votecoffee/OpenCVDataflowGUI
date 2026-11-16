"""Program settings, lifecycle scripts and owned services, separate from block editing."""

import copy
import json

from PySide6.QtCore import Qt
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
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .program_config import DEFAULT_SERVICE, HOOKS, validate_program, validate_scripts


class ProgramSettingsDialog(QDialog):
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self.settings = copy.deepcopy(graph.program)
        self.setWindowTitle("Program settings and scripts")
        self.resize(960, 760)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        general = QWidget()
        form = QFormLayout(general)
        self.name = QLineEdit(self.settings["name"])
        self.description = QPlainTextEdit(self.settings["description"])
        self.description.setMaximumHeight(100)
        self.enabled = QCheckBox("Enable lifecycle scripts and managed services on first run")
        self.enabled.setChecked(self.settings["enabled"])
        self.enabled.setToolTip("Only enable programs whose Python code and commands you trust.")
        self.variables = QPlainTextEdit(json.dumps(self.settings["variables"], indent=2))
        self.variables.setToolTip(
            "Initial JSON state shared by lifecycle scripts; runtime changes are not saved."
        )
        self.assets = QPlainTextEdit("\n".join(self.settings["assets"]))
        self.assets.setMaximumHeight(100)
        self.assets.setToolTip(
            "One program-relative file per line. Save As copies declared assets."
        )
        for title, widget in (
            ("Name", self.name),
            ("Description", self.description),
            ("Extensions", self.enabled),
            ("Initial variables (JSON)", self.variables),
            ("Program assets", self.assets),
        ):
            form.addRow(title, widget)
        self.tabs.addTab(general, "Program")
        scripts = QWidget()
        script_layout = QVBoxLayout(scripts)
        hint = QLabel(
            "Startup runs once before services start. Before run and After run wrap each cycle. "
            "Shutdown runs after block resources close and before services stop.\n"
            "Scripts have context (event, project_dir, run_index, status, services) and a mutable "
            "JSON state dictionary. print() goes to program diagnostics."
        )
        hint.setWordWrap(True)
        script_layout.addWidget(hint)
        self.script_tabs = QTabWidget()
        self.scripts = {}
        for event in HOOKS:
            editor = QPlainTextEdit(self.settings["scripts"][event])
            editor.setPlaceholderText(
                "# Trusted Python\n# state['count'] = state.get('count', 0) + 1"
            )
            self.scripts[event] = editor
            self.script_tabs.addTab(editor, event.replace("_", " ").title())
        script_layout.addWidget(self.script_tabs)
        self.script_timeout = QDoubleSpinBox()
        self.script_timeout.setRange(0.05, 120)
        self.script_timeout.setSuffix(" seconds")
        self.script_timeout.setValue(self.settings["script_timeout_seconds"])
        script_layout.addWidget(QLabel("Maximum time per script"))
        script_layout.addWidget(self.script_timeout)
        self.tabs.addTab(scripts, "Scripts")
        self._make_services()
        runtime = QWidget()
        runtime_layout = QVBoxLayout(runtime)
        self.runtime_text = QPlainTextEdit()
        self.runtime_text.setReadOnly(True)
        runtime_layout.addWidget(self.runtime_text)
        refresh = QPushButton("Refresh status")
        refresh.clicked.connect(self.refresh_status)
        runtime_layout.addWidget(refresh)
        stop = QPushButton("Stop program services")
        stop.setToolTip(
            "Closes block connections and services. The next run starts a fresh program session."
        )
        stop.clicked.connect(self.stop_program)
        runtime_layout.addWidget(stop)
        self.tabs.addTab(runtime, "Runtime")
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ff7b79")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.apply_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh_status()

    def _make_services(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        hint = QLabel(
            "Services are commands owned by this program. Use {python} for the selected runtime "
            "and {project_dir} for files beside the saved program. Enter one command argument "
            "per line, without shell quoting. Readiness waits for the given text in output."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.service_list = QListWidget()
        left_layout.addWidget(self.service_list)
        controls = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self.add_service)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_service)
        controls.addWidget(add)
        controls.addWidget(remove)
        left_layout.addLayout(controls)
        split.addWidget(left)
        right = QWidget()
        form = QFormLayout(right)
        self.service_fields = {}
        for name in DEFAULT_SERVICE:
            if name == "enabled":
                widget = QCheckBox()
            elif name == "runtime":
                widget = QComboBox()
                widget.addItems(("application", "ros2"))
            elif name.endswith("timeout_seconds"):
                widget = QDoubleSpinBox()
                widget.setRange(0.05, 15 if name.startswith("shutdown") else 120)
            elif name in ("argv", "environment", "stop_input"):
                widget = QPlainTextEdit()
                widget.setMaximumHeight(100)
            else:
                widget = QLineEdit()
            self.service_fields[name] = widget
            form.addRow(name.replace("_", " ").title(), widget)
        self.service_fields["environment"].setPlaceholderText("NAME=value, one per line")
        self.service_fields["stop_input"].setToolTip(
            "Text sent to standard input before closing it; empty sends EOF."
        )
        split.addWidget(right)
        split.setSizes([220, 650])
        self.tabs.addTab(tab, "Services")
        self._service_index = -1
        self.service_list.currentRowChanged.connect(self.select_service)
        for service in self.settings["services"]:
            self.service_list.addItem(service["id"])
        if self.settings["services"]:
            self.service_list.setCurrentRow(0)

    def store_service(self):
        if self._service_index < 0:
            return
        service = {}
        for name, widget in self.service_fields.items():
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                value = widget.toPlainText()
                if name == "argv":
                    original = self.settings["services"][self._service_index][name]
                    value = original if value == "\n".join(original) else value.splitlines()
                elif name == "environment":
                    original = self.settings["services"][self._service_index][name]
                    if value == "\n".join(f"{key}={item}" for key, item in original.items()):
                        service[name] = copy.deepcopy(original)
                        continue
                    entries = [line.partition("=") for line in value.splitlines() if line.strip()]
                    if any(not middle for _, middle, _ in entries):
                        raise ValueError("Environment entries must use NAME=value")
                    value = {key: item for key, _, item in entries}
            else:
                value = widget.text()
            service[name] = value
        self.settings["services"][self._service_index] = service
        self.service_list.item(self._service_index).setText(service["id"])

    def select_service(self, index):
        try:
            self.store_service()
        except ValueError as exc:
            self.error.setText(str(exc))
            self.service_list.blockSignals(True)
            self.service_list.setCurrentRow(self._service_index)
            self.service_list.blockSignals(False)
            return
        self._service_index = index
        if index < 0:
            return
        for name, widget in self.service_fields.items():
            value = self.settings["services"][index][name]
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                if name == "argv":
                    value = "\n".join(value)
                elif name == "environment":
                    value = "\n".join(f"{key}={item}" for key, item in value.items())
                widget.setPlainText(value)
            else:
                widget.setText(value)

    def add_service(self):
        try:
            self.store_service()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        used = {service["id"] for service in self.settings["services"]}
        number = next(n for n in range(1, 100) if f"service{n}" not in used)
        service = {
            **copy.deepcopy(DEFAULT_SERVICE),
            "id": f"service{number}",
            "argv": ["{python}", "-u", "{project_dir}/service.py"],
        }
        self.settings["services"].append(service)
        self.service_list.addItem(service["id"])
        self.service_list.setCurrentRow(len(self.settings["services"]) - 1)

    def remove_service(self):
        index = self.service_list.currentRow()
        if index >= 0:
            self._service_index = -1
            self.settings["services"].pop(index)
            self.service_list.takeItem(index)
            self.select_service(self.service_list.currentRow())

    def refresh_status(self):
        self.runtime_text.setPlainText(json.dumps(self.graph.program_status(), indent=2))

    def stop_program(self):
        self.graph.stop_program()
        self.refresh_status()

    def apply_settings(self):
        try:
            self.store_service()
            self.settings.update(
                name=self.name.text(),
                description=self.description.toPlainText(),
                enabled=self.enabled.isChecked(),
                variables=json.loads(self.variables.toPlainText()),
                assets=self.assets.toPlainText().splitlines(),
                scripts={event: editor.toPlainText() for event, editor in self.scripts.items()},
                script_timeout_seconds=self.script_timeout.value(),
            )
            settings = validate_program(self.settings)
            validate_scripts(settings)
            self.graph.set_program(settings)
        except (ValueError, SyntaxError) as exc:
            self.error.setText(str(exc))
            return
        self.accept()
