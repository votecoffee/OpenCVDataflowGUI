"""A modeless window pinned to one output or result graphic."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QLabel, QSizePolicy, QVBoxLayout

from .canvas import pixmap
from .payloads import metadata
from .tickets import ticket_key
from .visual_style import BLOCK_STATE_COLORS


def _ticket_label(block, image, payload):
    ticket = payload.get("ticket")
    if ticket is None and image is not None:
        ticket = metadata(image).get("ticket")
    if ticket is not None:
        return f"Ticket {ticket['value']}"
    result = block.last_result
    for payloads in (result.payloads if result is not None else {}, block.input_payloads):
        tickets = {
            ticket_key(item["ticket"]): item["ticket"]["value"]
            for item in payloads.values()
            if "ticket" in item
        }
        if len(tickets) == 1:
            return f"Ticket {next(iter(tickets.values()))}"
        if tickets:
            return "Multiple tickets"
    return "No ticket"


class ResultImageWindow(QDialog):
    def __init__(self, kind, name, parent=None):
        super().__init__(parent)
        self.kind, self.name = kind, name
        self._image = self._pixmap = self._state = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(700, 520)
        self.setMinimumSize(250, 180)
        layout = QVBoxLayout(self)
        self.frame = QFrame()
        self.frame.setObjectName("resultImageFrame")
        layout.addWidget(self.frame)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(6, 6, 6, 6)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        frame_layout.addWidget(self.image)

    def refresh(self, block):
        result = block.last_result
        payload = {}
        if self.kind == "Output":
            image = block.last_output_images.get(self.name)
            if result is not None:
                payload = result.payloads.get(self.name, {})
        else:
            image = block.last_graphics.get(self.name)
        state = block.state
        passed = payload.get("outcome", {}).get("passed")
        if state in ("pass", "fail") and isinstance(passed, bool):
            state = "pass" if passed else "fail"
        self.setWindowTitle(
            f"{block.name} · {self.kind} / {self.name} · "
            f"{_ticket_label(block, image, payload)} · {state.upper()}"
        )
        if state != self._state:
            self._state = state
            color = BLOCK_STATE_COLORS.get(state, BLOCK_STATE_COLORS["idle"])
            self.frame.setStyleSheet(f"QFrame#resultImageFrame {{ border: 4px solid {color}; }}")
        if image is not self._image or self._pixmap is None:
            self._image = image
            self._pixmap = pixmap(image) if image is not None else None
            self.fit_image()

    def fit_image(self):
        if self._pixmap is None:
            self.image.setText("No image for this result")
            return
        self.image.setPixmap(
            self._pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()
