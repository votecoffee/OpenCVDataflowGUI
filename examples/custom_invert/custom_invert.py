"""Load using File > Load Python plugin, then add Invert from the library."""

import cv2

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.parameter_validation import NumericRange, ValidationError


class Invert(Block):
    type_id = "example.invert"
    title = "Invert"
    category = "Custom"
    preview_safe = True
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
    registry.register(Invert)
