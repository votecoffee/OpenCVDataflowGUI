"""Edit saved recipe revisions and import/copy trained pattern assets."""

import copy
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .recipes import check_model, lookup, reference, validate_recipe


class RecipeManager(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.catalog = window.root_graph.recipes
        self.models = {}
        self.setWindowTitle("Manage recipes")
        self.resize(850, 680)
        layout = QVBoxLayout(self)
        self.status = QLabel()
        layout.addWidget(self.status)
        form = QFormLayout()
        self.revisions = QComboBox()
        self.revisions.currentIndexChanged.connect(self.load_selected)
        form.addRow("Saved revision", self.revisions)
        self.name = QLineEdit()
        form.addRow("Recipe ID", self.name)
        layout.addLayout(form)
        self.settings = QTextEdit()
        self.settings.setPlaceholderText(
            'Settings JSON, for example {"threshold": 120, "min_score": 0.8}'
        )
        layout.addWidget(self.settings)
        self.assets = QTreeWidget()
        self.assets.setHeaderLabels(["Model name", "Type", "Features", "Model checksum"])
        layout.addWidget(self.assets)
        self.default = QCheckBox("Use as project default when activating")
        self.default.setChecked(True)
        layout.addWidget(self.default)
        for actions in (
            (
                ("New", self.new),
                ("Save new revision", self.save_revision),
                ("Activate saved revision", self.activate),
            ),
            (
                ("Import recipe…", self.import_recipe),
                ("Export saved revision…", self.export_recipe),
                ("Import trained model…", self.import_model),
                ("Copy selected tool model…", self.copy_model),
                ("Remove selected model", self.remove_model),
            ),
        ):
            row = QHBoxLayout()
            for title, callback in actions:
                button = QPushButton(title)
                button.clicked.connect(callback)
                row.addWidget(button)
            layout.addLayout(row)
        note = QLabel("Revisions are immutable. Saving/activating preserves older queued work.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.refresh()

    def refresh(self, selected=None):
        self.revisions.blockSignals(True)
        self.revisions.clear()
        for recipe in self.catalog.definition()["recipes"]:
            self.revisions.addItem(
                f"{recipe['id']} · revision {recipe['revision']}", reference(recipe)
            )
        if selected:
            for i in range(self.revisions.count()):
                if self.revisions.itemData(i) == selected:
                    self.revisions.setCurrentIndex(i)
                    break
        self.revisions.blockSignals(False)
        active = self.catalog.active_ref()
        self.status.setText(
            f"Active: {active['id']} r{active['revision']}" if active else "No active recipe"
        )
        if self.revisions.count():
            self.load_selected()
        else:
            self.new()

    def show_models(self):
        self.assets.clear()
        for name, model in self.models.items():
            item = QTreeWidgetItem(
                self.assets, [name, model["model"], str(len(model["points"])), model["id"]]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, name)

    def load_selected(self):
        selected = self.revisions.currentData()
        if selected is None:
            return
        recipe = self.catalog.resolve(selected)
        self.name.setText(recipe["id"])
        self.settings.setPlainText(json.dumps(recipe["settings"], indent=2))
        self.models = recipe["models"]
        self.show_models()

    def new(self):
        self.name.clear()
        self.settings.setPlainText("{}")
        self.models = {}
        self.show_models()

    def save_revision(self):
        try:
            recipe = self.catalog.put(
                self.name.text(), json.loads(self.settings.toPlainText()), self.models
            )
            ref = reference(recipe)
            if self.catalog.active_ref() is None:
                self.catalog.activate(ref, project_default=True)
            self.window.mark_dirty()
            self.refresh(ref)
            return recipe
        except (ValueError, TypeError) as exc:
            self.window.warn(str(exc))
            return None

    def activate(self):
        selected = self.revisions.currentData()
        if selected is not None:
            self.catalog.activate(selected, project_default=self.default.isChecked())
            self.window.mark_dirty()
            self.refresh(selected)

    def import_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import recipe content", filter="JSON (*.json)")
        if path:
            try:
                recipe = validate_recipe(json.loads(Path(path).read_text(encoding="utf-8")))
                self.name.setText(recipe["id"])
                self.settings.setPlainText(json.dumps(recipe["settings"], indent=2))
                self.models = copy.deepcopy(recipe["models"])
                self.show_models()
                self.status.setText("Imported content; Save new revision to install it")
            except (ValueError, TypeError, OSError) as exc:
                self.window.warn(str(exc))

    def export_recipe(self):
        selected = self.revisions.currentData()
        if selected is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export saved recipe", filter="JSON (*.json)")
        if path:
            try:
                with Path(path).open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(self.catalog.resolve(selected), indent=2) + "\n")
            except (ValueError, OSError) as exc:
                self.window.warn(str(exc))

    def add_model(self, model):
        check_model(model)
        name, accepted = QInputDialog.getText(self, "Recipe model", "Model name", text="part")
        if accepted:
            self.models[name] = copy.deepcopy(model)
            self.show_models()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import trained model", filter="JSON (*.json)")
        if path:
            try:
                self.add_model(json.loads(Path(path).read_text(encoding="utf-8")))
            except (ValueError, TypeError, OSError) as exc:
                self.window.warn(str(exc))

    def copy_model(self):
        block = self.window.inspector.block
        model = getattr(block, "trained_model", None)
        if model is None and getattr(block, "last_result", None):
            model = block.last_result.outputs.get("model")
        if model is None and getattr(block, "recipe_model", ""):
            try:
                model = lookup(self.catalog.resolve(block.last_recipe_ref), block.recipe_model)
            except ValueError as exc:
                self.window.warn(str(exc))
                return
        if model is None:
            self.window.warn("Select a trained pattern tool in the inspector first")
            return
        try:
            self.add_model(model)
        except (ValueError, TypeError) as exc:
            self.window.warn(str(exc))

    def remove_model(self):
        selected = self.assets.currentItem()
        if selected:
            self.models.pop(selected.data(0, Qt.ItemDataRole.UserRole), None)
            self.show_models()
