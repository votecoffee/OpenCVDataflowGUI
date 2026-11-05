"""Decode byte-packed camera YUV formats to ordinary image pixels."""

from pathlib import Path

import cv2
import numpy as np

from .basic_processing import ALGORITHM_HINTS
from .core import Block, Port, Result
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame
from .processing import ImageTool

YUV_420 = ("NV12", "NV21", "I420", "YV12")
YUV_422 = ("YUY2", "UYVY", "YVYU")
YUV_DEFAULTS = {"layout": "NV12", "color_format": "bgr", "algorithm_hint": "default"}
YUV_CHOICES = {
    "layout": (*YUV_420, *YUV_422),
    "color_format": ("gray", "bgr", "rgb", "bgra", "rgba"),
    "algorithm_hint": tuple(ALGORITHM_HINTS),
}
YUV_HELP = {
    "layout": "Byte order and chroma subsampling of the input buffer.",
    "color_format": "Gray extracts the stored luma samples.",
}
MAX_FRAME_PIXELS = 16_777_216


def packed_shape(width, height, layout):
    if width % 2 or (layout in YUV_420 and height % 2):
        raise ValidationError("YUV width must be even, and 4:2:0 also needs even height")
    if width * height > MAX_FRAME_PIXELS:
        raise ValidationError("Decoded YUV frame exceeds the 16,777,216-pixel limit")
    if layout in YUV_420:
        return height * 3 // 2, width
    return height, width * 2


def decode_packed_yuv(buffer, parameters):
    layout = parameters["layout"]
    if buffer.ndim != 2 or buffer.dtype != np.uint8 or not buffer.size:
        raise ValidationError("Packed YUV input must be a nonempty 2D uint8 byte matrix")
    rows, columns = buffer.shape
    if layout in YUV_420:
        if rows % 3:
            raise ValidationError(
                "Packed 4:2:0 needs a byte matrix with 3 Ã— image height / 2 rows"
            )
        width, height = columns, rows * 2 // 3
    else:
        if columns % 4:
            raise ValidationError("Packed 4:2:2 needs four bytes per pair of pixels")
        width, height = columns // 2, rows
        buffer = buffer.reshape(height, width, 2)
    packed_shape(width, height, layout)
    encoding = parameters["color_format"]
    code = getattr(cv2, f"COLOR_YUV2{encoding.upper()}_{layout}")
    decoded = cv2.cvtColor(buffer, code, hint=ALGORITHM_HINTS[parameters["algorithm_hint"]])
    return ImageArray(decoded, encoding)


class DecodePackedYuv(ImageTool):
    type_id, title, subtype = "cv.decode_yuv", "Decode packed YUV", "Color and depth"
    defaults = YUV_DEFAULTS
    parameter_choices = YUV_CHOICES
    parameter_help = YUV_HELP
    input_help = {"image": "Packed 2D uint8 byte matrix with no row padding."}

    def process(self, inputs):
        image = decode_packed_yuv(np.asarray(inputs["image"]), self.parameters)
        size = [image.shape[1], image.shape[0]]
        payload = {
            key: value
            for key, value in self.input_payloads["image"].items()
            if key not in ("spatial", "pyramid")
        }
        payload["spatial"] = frame(size, f"{self.id}:decoded")
        return Result({"image": image}, payloads={"image": payload})


class RawYuvFile(Block):
    type_id, title, category = "input.raw_yuv", "Raw YUV frame", "Input"
    block_type, subtype = "Input", "Files"
    file_source = True
    finite_source = False
    outputs = (Port("image"), Port("info", "data"))
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "width": 640,
        "height": 480,
        "filename_ticket": True,
        **YUV_DEFAULTS,
    }
    parameter_choices = {**YUV_CHOICES, "path_base": ("working_directory", "project")}
    parameter_ranges = {axis: NumericRange(1, 16384, whole=True) for axis in ("width", "height")}
    parameter_help = {
        **YUV_HELP,
        "width": "Decoded image width in pixels, excluding row padding.",
        "height": "Decoded image height in pixels.",
        "filename_ticket": "Use the filename for automatic tickets.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        packed_shape(int(p["width"]), int(p["height"]), p["layout"])

    def issue_source_ticket(self):
        ticket = super().issue_source_ticket()
        if self.parameters["filename_ticket"] and self.ticket_settings["mode"] == "auto_number":
            value = self.parameters["path"]
            name = Path(value).name if isinstance(value, str) else ""
            if name:
                ticket.update(
                    kind="string",
                    value=name,
                    session=f"{ticket['session']}:yuv:{ticket['sequence']}",
                )
        return ticket

    def process(self, inputs):
        p = self.parameters
        value = p["path"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Select a raw YUV file", "path")
        path = Path(value)
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save or open the project before using a relative YUV path")
            path = self.project_dir / path
        width, height = int(p["width"]), int(p["height"])
        rows, columns = packed_shape(width, height, p["layout"])
        expected = rows * columns
        with path.open("rb") as stream:
            data = stream.read(expected + 1)
        if len(data) != expected:
            raise ValidationError(f"Expected exactly {expected:,} bytes for one packed YUV frame")
        image = decode_packed_yuv(np.frombuffer(data, np.uint8).reshape(rows, columns), p)
        return Result(
            {
                "image": image,
                "info": {
                    "filename": path.name,
                    "layout": p["layout"],
                    "width": width,
                    "height": height,
                    "byte_count": expected,
                },
            },
            message=f"{path.name} Â· {p['layout']} Â· {width} Ã— {height}",
        )
