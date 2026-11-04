"""Load using File > Load Python plugin, then add Invert from the library."""

import cv2

from opencv_dataflow.core import Block, Port, Result


class Invert(Block):
    type_id = "example.invert"
    title = "Invert"
    category = "Custom"
    inputs = (Port("image"),)
    outputs = (Port("image"), Port("original"))

    def process(self, inputs):
        image = inputs["image"]
        return Result(
            {"image": cv2.bitwise_not(image), "original": image}, message="Custom Python block"
        )


def register(registry):
    registry.register(Invert)
