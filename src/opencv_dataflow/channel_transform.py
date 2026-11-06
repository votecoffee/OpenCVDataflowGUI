"""Apply linear transforms and copy channels in stored sample order."""

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .payloads import metadata, require_same_frame
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, RegionalOperation, selection_mask
from .tickets import ticket_key

_CHANNEL_COUNTS = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4, "hsv": 3, "lab": 3}


def _route_indices(value, count):
    if not isinstance(value, (list, tuple)):
        raise ValidationError("Channel map must be a list of source indices", "channel_map")
    if not value:
        return list(range(count))
    if len(value) != count or any(
        type(index) is not int or not -1 <= index <= 7 for index in value
    ):
        raise ValidationError(
            f"Use {count} whole channel indices from -1 to 7 for this output format", "channel_map"
        )
    return list(value)


class RouteChannels(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.route_channels", "Route image channels", "Channels"
    region_source, region_background = "image_a", "zero"
    process_inputs_readonly = True
    inputs = (
        Port("image_a"),
        Port("image_b", required=False, required_when_connected=True),
        *SELECTION_INPUTS[1:],
        Port("channel_map", "data", required=False, required_when_connected=True),
    )
    defaults = {"color_format": "bgr", "channel_map": [], "alignment": "same_frame"}
    parameter_choices = {
        "color_format": tuple(_CHANNEL_COUNTS),
        "alignment": ("same_frame", "pixel_grid"),
    }
    parameter_help = {
        "color_format": "Encoding of the assembled channels. Samples are copied verbatim.",
        "channel_map": (
            "One index per output channel, numbering A then B from zero. "
            "Use -1 for zero fill or an empty list to copy the first channels in order."
        ),
        "alignment": "Pixel grid pairs equal-size images by index, even with different frame IDs.",
    }
    input_help = {
        "image_a": "First group of channels and the output coordinate frame.",
        "image_b": "Optional second group. Size and pixel depth must match Image A.",
        "channel_map": "Optional list of channel indices that overrides the configuration.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _route_indices(p["channel_map"], _CHANNEL_COUNTS[p["color_format"]])

    def process(self, inputs):
        a = inputs["image_a"]
        first = image_value(a)
        p = self.parameters
        encoding = p["color_format"]
        if encoding in ("hsv", "lab") and first.dtype not in (np.uint8, np.float32):
            raise ValidationError("HSV/Lab output needs uint8 or float32 samples", "color_format")
        count = _CHANNEL_COUNTS[encoding]
        indices = _route_indices(inputs.get("channel_map", p["channel_map"]), count)
        images = [first]
        payload = metadata(a)
        payload.pop("channel", None)
        if "image_b" in inputs:
            b = inputs["image_b"]
            second = image_value(b)
            if second.shape[:2] != first.shape[:2] or second.dtype != first.dtype:
                raise ValidationError("Channel sources must have the same size and pixel depth")
            if p["alignment"] == "same_frame":
                require_same_frame((a, b))
            other = metadata(b)
            if "ticket" in other:
                if "ticket" in payload and ticket_key(payload["ticket"]) != ticket_key(
                    other["ticket"]
                ):
                    raise ValidationError("Route channels from the same ticket")
                if "ticket" not in payload:
                    payload.update(
                        {key: other[key] for key in ("ticket", "processing_time") if key in other}
                    )
            images.append(second)
        available = sum(image.shape[2] if image.ndim == 3 else 1 for image in images)
        if max(indices) >= available:
            raise ValidationError(f"Only {available} source channels are available", "channel_map")
        shape = first.shape[:2] if count == 1 else (*first.shape[:2], count)
        output = np.empty(shape, dtype=first.dtype)
        pairs = []
        for target, source in enumerate(indices):
            pairs.extend((source, target))
        cv2.mixChannels(images, [output], pairs)
        return Result(
            {"image": ImageArray(output, encoding)},
            payloads={"image": payload},
            message=f"Routed {count} channels into {encoding.upper()}",
        )


def _coefficients(value):
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "Enter a rectangular matrix of numeric coefficients", "matrix"
        ) from exc
    if matrix.shape not in ((1, 1), (1, 2), (1, 3), (1, 4), (3, 3), (3, 4)):
        raise ValidationError("Use one or three rows with weights and an optional offset", "matrix")
    if not np.isfinite(matrix).all():
        raise ValidationError("Matrix coefficients must be finite", "matrix")
    return matrix


class ChannelTransform(ImageTool):
    type_id, title, subtype = "cv.channel_transform", "Linear channel transform", "Color and depth"
    inputs = (*SELECTION_INPUTS, Port("matrix", "data", required=False))
    defaults = {"matrix": []}
    parameter_help = {
        "matrix": (
            "One row for gray or three for color, using stored channel order without alpha. "
            "Append an optional offset. Empty uses identity. A wired matrix takes precedence."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _coefficients(p["matrix"])

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or not image.size:
            raise ValidationError("Provide a nonempty grayscale, RGB, BGR, RGBA, or BGRA image")
        mask = selection_mask(inputs, image)
        matrix = _coefficients(inputs.get("matrix", self.parameters["matrix"]))
        if matrix is None:
            return Result({"image": source})
        channels = 1 if image.ndim == 2 else 3
        if matrix.shape[1] not in (channels, channels + 1):
            raise ValidationError(
                f"Each row needs {channels} input weight(s) and an optional offset", "matrix"
            )

        has_alpha = encoding in ("rgba", "bgra")
        signal = image[..., :3] if has_alpha else image
        grayscale = matrix.shape[0] == 1
        if mask is None:
            output = self._transform(signal, matrix)
        else:
            selected = mask != 0
            output = (
                np.zeros(image.shape[:2], dtype=image.dtype)
                if grayscale and channels > 1
                else signal.copy()
            )
            if selected.any():
                samples = signal[selected].reshape(-1, 1, channels)
                transformed = self._transform(samples, matrix)
                output[selected] = transformed.ravel() if grayscale else transformed[:, 0, :]
        if grayscale:
            encoding = "gray"
        elif has_alpha:
            output = np.dstack((output, image[..., 3]))
        return Result({"image": ImageArray(output, encoding)})

    @staticmethod
    def _transform(signal, matrix):
        output = cv2.transform(signal.astype(np.float64, copy=False), matrix)
        if not np.isfinite(output).all():
            raise ValidationError("Channel transform produced nonfinite values", "matrix")
        if signal.dtype.kind in "iu":
            limits = np.iinfo(signal.dtype)
            output = np.rint(output).clip(limits.min, limits.max)
        elif np.max(np.abs(output)) > np.finfo(signal.dtype).max:
            raise ValidationError("Channel transform exceeded the source pixel depth", "matrix")
        return output.astype(signal.dtype, copy=False)
