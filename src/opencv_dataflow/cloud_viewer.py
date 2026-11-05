"""Qt orbit viewer for retained cloud results without a VTK/OpenGL dependency."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QLabel, QSlider, QVBoxLayout

from .canvas import pixmap
from .pointclouds import PointCloud, render_cloud


class CloudViewer(QDialog):
    def __init__(self, graph_provider, parent=None):
        super().__init__(parent)
        self.graph_provider = graph_provider
        self.cloud = None
        self.setWindowTitle("3D point-cloud viewer")
        self.resize(800, 680)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.choice = QComboBox()
        self.choice.currentIndexChanged.connect(self.refresh)
        form.addRow("Cloud output", self.choice)
        self.yaw = self.slider(-180, 180, 25)
        self.pitch = self.slider(-90, 90, -20)
        self.size = self.slider(1, 6, 2)
        for name, slider in (("Yaw", self.yaw), ("Pitch", self.pitch), ("Point size", self.size)):
            form.addRow(name, slider)
        layout.addLayout(form)
        self.info = QLabel()
        layout.addWidget(self.info)
        self.image = QLabel()
        self.image.setMinimumSize(400, 300)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)
        self.refresh()

    def slider(self, low, high, value):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(self.render)
        return slider

    def refresh(self, *args):
        graph = self.graph_provider()
        available = []
        for block in graph.blocks.values():
            if block.last_result is None:
                continue
            for name, value in block.last_result.outputs.items():
                if isinstance(value, PointCloud):
                    available.append(((block.id, name), f"{block.name} / {name}", value))
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
        cloud = next((value for k, _, value in available if k == key), None)
        if cloud is not self.cloud:
            self.cloud = cloud
            self.render()
        elif cloud is None:
            self.info.setText("Run in Debug or Balanced to inspect retained point clouds.")

    def render(self, *args):
        if self.cloud is None:
            self.image.clear()
            self.info.setText("No retained point cloud. Run in Debug or Balanced.")
            return
        cloud = self.cloud
        self.info.setText(f"{len(cloud.points):,} points · {cloud.units} · frame {cloud.frame}")
        image = render_cloud(
            cloud, 720, 540, self.yaw.value(), self.pitch.value(), point_size=self.size.value()
        )
        self.image.setPixmap(
            pixmap(image).scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
