"""Bundled example of an image block with validation and a custom dialog."""

import cv2

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import NumericRange, ValidationError


class Invert(Block):
    type_id = "example.invert"
    title = "Invert"
    category = "Custom"
    preview_safe = True
    plugin_dialog_label = "Edit inversion band…"
    plugin_dialog_modeless = True
    inputs = (Port("image"),)
    outputs = (Port("image"), Port("original"))
    defaults = {"start_percent": 0, "end_percent": 100}
    parameter_ranges = {
        "start_percent": NumericRange(0, 100, whole=True),
        "end_percent": NumericRange(0, 100, whole=True),
    }
    parameter_help = {
        "start_percent": "Left edge of the inversion band, measured from the left of the image.",
        "end_percent": "Right edge of the inversion band, measured from the left of the image.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        if values["start_percent"] >= values["end_percent"]:
            raise ValidationError(
                "Start percent must be less than end percent", "start_percent", "end_percent"
            )

    def create_plugin_dialog(self, parent, parameters, context):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QSpinBox,
        )

        from opencv_dataflow.plugin_image_picker import PluginImagePicker

        dialog = QDialog(parent)
        dialog.setWindowTitle("Inversion band")
        form = QFormLayout(dialog)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)

        def update(key, value):
            parameters[key] = value
            ok.setEnabled(parameters["start_percent"] < parameters["end_percent"])

        for key, label in (("start_percent", "Start (%)"), ("end_percent", "End (%)")):
            field = QSpinBox(dialog)
            field.setObjectName(key)
            field.setRange(0, 100)
            field.setValue(parameters[key])
            update(key, field.value())
            field.valueChanged.connect(lambda value, name=key: update(name, value))
            form.addRow(label, field)
        captures = []
        output_notice = QLabel("Run the graph to publish outputs.")
        output_notice.setObjectName("output_notice")
        context.subscribe_outputs(
            lambda revision: output_notice.setText(f"Graph output {revision} ready to capture.")
        )
        form.addRow(output_notice)
        summary = QLabel("Run once, then capture the source image.")
        summary.setObjectName("capture_summary")
        summary.setTextFormat(Qt.TextFormat.PlainText)

        def capture(packet):
            captures.append(packet)
            del captures[:-2]
            rows = []
            labels = ("Previous", "Latest")[-len(captures) :]
            for label, packet in zip(labels, captures, strict=True):
                height, width = packet.value.shape[:2]
                ticket = packet.payload.get("ticket", {}).get("value", "unticketed")
                rows.append(f"{label} · {width} × {height} · ticket {ticket}")
            summary.setText("\n".join(rows))

        picker = PluginImagePicker(context, dialog, input_port="image")
        picker.captured.connect(capture)
        form.addRow(picker)
        form.addRow(summary)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        return dialog

    def process(self, inputs):
        image = inputs["image"]
        width = image.shape[1]
        start, end = (int(self.parameters[key]) for key in ("start_percent", "end_percent"))
        left, right = width * start // 100, (width * end + 99) // 100
        band = image[:, left:right]
        output = image.copy()
        output[:, left:right] = cv2.bitwise_not(band).reshape(band.shape)
        return Result(
            {"image": output, "original": image},
            message=f"Inverted columns {left}–{right - 1} ({start}–{end}% of the width)",
        )


def register(registry):
    if registry.types.get(Invert.type_id) is not Invert:
        registry.register(Invert)
