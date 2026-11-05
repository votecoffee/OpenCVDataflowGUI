"""Per-channel numeric histograms with a result graphic."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .selection import SELECTION_INPUTS, selection_mask

HISTOGRAM_CHANNELS = {
    "gray": ("Gray",),
    "bgr": ("B", "G", "R"),
    "rgb": ("R", "G", "B"),
    "bgra": ("B", "G", "R", "A"),
    "rgba": ("R", "G", "B", "A"),
    "hsv": ("H", "S", "V"),
    "lab": ("L", "a", "b"),
}


class Histogram(Block):
    type_id, title, category = "inspect.histogram", "Histogram", "Image processing"
    block_type, subtype = "Image processing", "Statistics"
    inputs = SELECTION_INPUTS
    outputs = (Port("histogram", "data"), Port("plot"))
    defaults = {
        "bins": 256,
        "range_mode": "automatic",
        "range_min": 0.0,
        "range_max": 1.0,
        "include_alpha": False,
    }
    parameter_choices = {"range_mode": ("automatic", "manual")}
    parameter_ranges = {"bins": NumericRange(2, 4096, whole=True)}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["range_mode"] == "manual":
            low = NumericRange().check(p["range_min"], "Range minimum", "range_min")
            high = NumericRange().check(p["range_max"], "Range maximum", "range_max")
            if high <= low:
                raise ValidationError(
                    "Histogram range must have max > min", "range_min", "range_max"
                )

    def process(self, inputs):
        image = inputs["image"]
        encoding = image_format(image)
        bins = int(self.parameters["bins"])
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValidationError("Histogram requires grayscale, three or four channels")
        labels = HISTOGRAM_CHANNELS[encoding]
        planes = [image] if image.ndim == 2 else [image[..., i] for i in range(image.shape[2])]
        if len(planes) == 4 and not self.parameters["include_alpha"]:
            planes, labels = planes[:3], labels[:3]
        graphic = np.full((300, 600, 3), 25, np.uint8)
        colors = {"B": (240, 150, 70), "G": (90, 220, 90), "R": (80, 80, 245)}
        channels = []
        selection = selection_mask(inputs, image)
        for index, (label, plane) in enumerate(zip(labels, planes)):
            valid = np.isfinite(plane)
            if selection is not None:
                valid &= selection != 0
            finite = np.asarray(plane)[valid]
            mode = self.parameters["range_mode"]
            if mode == "manual":
                low, high = float(self.parameters["range_min"]), float(self.parameters["range_max"])
            elif mode == "automatic":
                if np.issubdtype(image.dtype, np.unsignedinteger):
                    low, high = 0.0, float(np.iinfo(image.dtype).max) + 1
                    if encoding == "hsv" and label == "H":
                        high = 180.0
                elif finite.size:
                    low, high = float(finite.min()), float(finite.max())
                    if low == high:
                        high = low + max(1.0, abs(low) * 0.01)
                else:
                    low, high = 0.0, 1.0
            else:
                raise ValueError("Range mode must be automatic or manual")
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError("Histogram range must be finite with max > min")
            counts, edges = np.histogram(finite, bins=bins, range=(low, high))
            channels.append(
                {
                    "channel": label,
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "finite_samples": int(finite.size),
                    "outside_range": int(finite.size - counts.sum()),
                }
            )
            x = np.linspace(36, 580, bins)
            y = 260 - counts.astype(float) / max(int(counts.max()), 1) * 205
            points = np.column_stack((x, y)).astype(np.int32)
            color = colors.get(label, [(200, 200, 200), (220, 180, 90), (180, 90, 220)][index % 3])
            cv2.polylines(graphic, [points], False, color, 1)
            cv2.putText(
                graphic, label, (40 + index * 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        cv2.line(graphic, (36, 260), (580, 260), (150, 150, 150), 1)
        return Result(
            {
                "histogram": {"color_format": encoding, "bins": bins, "channels": channels},
                "plot": graphic,
            },
            graphics={"histogram": graphic},
            payloads={"plot": {}},
            message=f"{len(channels)} channels · {bins} bins",
        )
