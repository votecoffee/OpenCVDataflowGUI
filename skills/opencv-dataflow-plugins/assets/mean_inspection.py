"""MIT example plugin: measure selected 8-bit image intensity without altering pixels."""

import numpy as np

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.graphics_layers import GraphicDrawing, graphic_object
from opencv_dataflow.parameter_validation import NumericRange, ValidationError
from opencv_dataflow.processing import gray_image
from opencv_dataflow.selection import SELECTION_INPUTS, selection_mask


class MeanInspection(Block):
    type_id = "example.mean_inspection"
    title, category, subtype = "Inspect mean intensity", "Inspection", "Intensity"
    inputs = SELECTION_INPUTS
    outputs = (Port("image"), Port("mean", "number"), Port("passed", "boolean"))
    defaults = {"minimum": 20.0, "maximum": 100.0, "comparison": "inside"}
    parameter_choices = {"comparison": ("inside", "outside")}
    parameter_ranges = {"minimum": NumericRange(0, 255), "maximum": NumericRange(0, 255)}
    parameter_help = {
        "minimum": "Lower mean grayscale intensity in 8-bit sample units, inclusive.",
        "maximum": "Upper mean grayscale intensity in 8-bit sample units, inclusive.",
        "comparison": "Inside accepts the closed interval; outside accepts its complement.",
    }
    preview_safe = True  # Stateless, private snapshots, no external side effects.
    parallel_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum"] > p["maximum"]:
            raise ValidationError("Minimum must be <= maximum", "minimum", "maximum")

    def process(self, inputs):
        image = inputs["image"]
        if image.dtype != np.uint8:
            raise ValidationError("Convert the inspection image explicitly to uint8")
        gray = gray_image(image)
        mask = selection_mask(inputs, image)
        pixels = gray if mask is None else gray[mask != 0]
        if not pixels.size:
            raise ValidationError("The inspection selection is empty")
        mean = float(np.mean(pixels))
        p = self.parameters
        inside = p["minimum"] <= mean <= p["maximum"]
        passed = inside if p["comparison"] == "inside" else not inside
        message = f"Mean {mean:.3f} DN: {'PASS' if passed else 'FAIL'}"
        drawings = {}
        if self.generate_graphics:
            drawings["measurement"] = GraphicDrawing(
                (
                    graphic_object(
                        "label",
                        [(8, 20)],
                        (60, 230, 120) if passed else (60, 90, 240),
                        text=message,
                    ),
                )
            )
        self.logger.debug("Measured intensity %.3f DN", mean)
        return Result(
            {"image": image, "mean": mean, "passed": passed},
            passed=passed,
            message=message,
            drawings=drawings,
            event_data={"mean": mean, "minimum": p["minimum"], "maximum": p["maximum"]},
            fail_reasons=[] if passed else [{"code": "mean_intensity", "message": message}],
        )


def register(registry):
    registry.register(MeanInspection)
