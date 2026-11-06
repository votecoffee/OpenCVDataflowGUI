"""Optional Qt image picker for plugin-owned dialogs."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QWidget

from .payloads import ValuePacket


class PluginImagePicker(QWidget):
    """Choose a cached image and emit an independent packet on capture."""

    captured = Signal(object)

    def __init__(self, context, parent=None, *, input_port=None):
        super().__init__(parent)
        self._context = context
        self._input_port = input_port
        self._choices = {}
        self._source_key = None
        self._level = 0

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.source_combo = QComboBox(self)
        self.source_combo.setPlaceholderText("Choose an image source")
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        form.addRow("Image source", self.source_combo)
        self.level_combo = QComboBox(self)
        self.level_combo.setPlaceholderText("Choose an available level")
        self.level_combo.currentIndexChanged.connect(self._level_changed)
        form.addRow("Pyramid level", self.level_combo)
        self.capture_button = QPushButton("Capture image", self)
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self._capture)
        form.addRow(self.capture_button)
        self.status_label = QLabel("Run the graph to publish images.", self)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        self.refresh()
        unsubscribe = context.subscribe_outputs(lambda revision: self.refresh())
        self.destroyed.connect(unsubscribe)

    def refresh(self):
        """Refresh source choices from the editor cache without copying image pixels."""
        choices = {}
        if self._input_port is not None:
            choices["connected", "", self._input_port] = (f"Wired input / {self._input_port}", 1)
            choices["processed", "", self._input_port] = (
                f"Last processed input / {self._input_port}",
                1,
            )
        for output in self._context.refresh_images():
            key = ("output", output.block_id, output.port)
            choices[key] = (f"{output.block_name} / {output.port}", output.level_count)
        if choices == self._choices:
            return

        self._choices = choices
        if self._source_key is None:
            self._source_key = next(iter(choices), None)
        selected = -1
        with QSignalBlocker(self.source_combo):
            self.source_combo.clear()
            for row, (key, (label, _)) in enumerate(choices.items()):
                self.source_combo.addItem(label, key)
                if key[0] == "output":
                    self.source_combo.setItemData(row, key[1], Qt.ItemDataRole.ToolTipRole)
                if key == self._source_key:
                    selected = row
            self.source_combo.setCurrentIndex(selected)
        self._update_levels()

    def _source_changed(self, row):
        self._source_key = self.source_combo.currentData() if row >= 0 else None
        self._level = 0
        self._update_levels()

    def _level_changed(self, row):
        if row >= 0:
            self._level = row
        self._update_capture()

    def _update_levels(self):
        count = self._choices.get(self._source_key, ("", 0))[1]
        with QSignalBlocker(self.level_combo):
            self.level_combo.clear()
            if count:
                mode = self._source_key[0]
                if mode == "connected":
                    self.level_combo.addItem("Configured input level")
                elif mode == "processed":
                    self.level_combo.addItem("As processed")
                else:
                    self.level_combo.addItem("0 (largest)" if count > 1 else "Image")
                    self.level_combo.addItems([str(level) for level in range(1, count)])
                self.level_combo.setCurrentIndex(self._level if self._level < count else -1)
        self._update_capture()

    def _update_capture(self):
        has_level = self.level_combo.currentIndex() >= 0
        level_count = self.level_combo.count()
        self.level_combo.setEnabled(level_count > 1 or (level_count == 1 and not has_level))
        available = self.source_combo.currentIndex() >= 0 and has_level
        self.capture_button.setEnabled(available)
        if available:
            self.status_label.clear()
        elif not self._choices:
            self.status_label.setText("Run the graph to publish images.")
        else:
            self.status_label.setText("Choose an available image source and level.")

    def snapshot(self) -> ValuePacket:
        """Copy the latest cached image for the selected source and level."""
        self.refresh()
        if self.source_combo.currentIndex() < 0 or self.level_combo.currentIndex() < 0:
            raise ValueError("Choose an available image source and level")
        mode, block_id, port = self._source_key
        if mode == "output":
            return self._context.snapshot_image(block_id, port, level=self._level)
        return self._context.snapshot_input_image(port, source=mode)

    def _capture(self):
        try:
            packet = self.snapshot()
        except (TypeError, ValueError, RuntimeError) as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.clear()
        self.captured.emit(packet)
