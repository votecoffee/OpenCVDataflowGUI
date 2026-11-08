"""Qt orbit viewer for retained cloud results without a VTK/OpenGL dependency."""

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

from .canvas import pixmap
from .pointclouds import cloud_outputs, render_cloud
from .result_image import _ticket_label
from .visual_style import BLOCK_STATE_COLORS


class CloudViewer(QDialog):
    def __init__(self, graph_provider, parent=None, *, output_name=None):
        super().__init__(parent)
        self.graph_provider = graph_provider
        self.output_name = output_name
        self.cloud = None
        self._rendered = None
        self.setWindowTitle("3D point-cloud viewer")
        self.resize(800, 680)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        if graph_provider is not None:
            self.choice = QComboBox()
            self.choice.currentIndexChanged.connect(self.refresh)
            form.addRow("Cloud output", self.choice)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.yaw = self.slider(-180, 180, 25)
        self.pitch = self.slider(-90, 90, -20)
        self.size = self.slider(1, 6, 2)
        for name, slider in (("Yaw", self.yaw), ("Pitch", self.pitch), ("Point size", self.size)):
            form.addRow(name, slider)
        self.flyaround = QCheckBox("Flyaround")
        self.flyaround.setEnabled(False)
        self.flyaround.toggled.connect(self.toggle_flyaround)
        form.addRow(self.flyaround)
        layout.addLayout(form)
        self.info = QLabel()
        layout.addWidget(self.info)
        self.image = QLabel()
        self.image.setMinimumSize(400, 300)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.installEventFilter(self)
        layout.addWidget(self.image, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.orbit_timer = QTimer(self)
        self.orbit_timer.setInterval(80)
        self.orbit_timer.timeout.connect(self.advance_orbit)

    def slider(self, low, high, value):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(self.render)
        return slider

    def showEvent(self, event):
        super().showEvent(event)
        if self.graph_provider is not None:
            self.refresh()
            self.timer.start()
        self.sync_flyaround()

    def hideEvent(self, event):
        self.timer.stop()
        self.orbit_timer.stop()
        super().hideEvent(event)

    def refresh(self, block=None):
        if self.output_name is not None:
            name = self.output_name
            cloud = cloud_outputs(block).get(name)
            payload = block.last_result.payloads.get(name, {}) if block.last_result else {}
            self.setWindowTitle(
                f"{block.name} · Cloud / {name} · {_ticket_label(block, cloud, payload)}"
                f" · {block.state.upper()}"
            )
            border = f"border: 4px solid {BLOCK_STATE_COLORS[block.state]};"
            if self.image.styleSheet() != border:
                self.image.setStyleSheet(border)
            self.set_cloud(cloud)
            return

        graph = self.graph_provider()
        available = [
            ((block.id, name), f"{block.name} / {name}", value)
            for block in graph.blocks.values()
            for name, value in cloud_outputs(block).items()
        ]
        key = self.choice.currentData()
        items = [
            (self.choice.itemData(i), self.choice.itemText(i)) for i in range(self.choice.count())
        ]
        if items != [(k, name) for k, name, _ in available]:
            self.choice.blockSignals(True)
            self.choice.clear()
            for k, name, _ in available:
                self.choice.addItem(name, k)
            index = self.choice.findData(key)
            self.choice.setCurrentIndex(index if index >= 0 else 0)
            self.choice.blockSignals(False)
        key = self.choice.currentData()
        self.set_cloud(next((value for k, _, value in available if k == key), None))

    def set_cloud(self, cloud):
        if cloud is not self.cloud or self._rendered is None:
            self.cloud = cloud
            self.flyaround.setEnabled(cloud is not None and len(cloud.points) > 0)
            self.sync_flyaround()
            self.render()

    def sync_flyaround(self):
        active = self.isVisible() and self.flyaround.isEnabled() and self.flyaround.isChecked()
        if active and not self.orbit_timer.isActive():
            self.orbit_timer.start()
        elif not active:
            self.orbit_timer.stop()

    def toggle_flyaround(self):
        self.sync_flyaround()
        self.render()

    def advance_orbit(self):
        self.yaw.setValue((self.yaw.value() + 4 + 180) % 360 - 180)

    def render(self, *args):
        if self.cloud is None:
            self._rendered = None
            self.image.clear()
            self.info.setText("No retained point cloud. Run in Debug or Balanced.")
            return
        cloud = self.cloud
        self.info.setText(f"{len(cloud.points):,} points · {cloud.units} · frame {cloud.frame}")
        image = render_cloud(
            cloud, 720, 540, self.yaw.value(), self.pitch.value(),
            limit=6000 if self.flyaround.isChecked() else 20000,
            point_size=self.size.value(), stable_scale=True,
        )
        self._rendered = pixmap(image)
        self.fit_image()

    def fit_image(self):
        if self._rendered is not None:
            self.image.setPixmap(
                self._rendered.scaled(
                    self.image.contentsRect().size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def eventFilter(self, watched, event):
        if watched is self.image and event.type() == QEvent.Type.Resize:
            self.fit_image()
        return super().eventFilter(watched, event)
