"""Bounded MIPI RAW10/RAW12 unpacking with explicit byte and pixel geometry."""

import copy
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame, validate_payload
from .result_graphics import ACCEPTED

PACKINGS = {"mipi_raw10": (10, 4, 5), "mipi_raw12": (12, 2, 3)}
BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
MAX_FRAME_BYTES = 128 * 1024 * 1024
RAW_DEFAULTS = {
    "packing": "mipi_raw10",
    "width": 640,
    "height": 480,
    "row_stride": 0,
    "byte_offset": 0,
    "size_policy": "exact",
    "alignment": "right",
    "bayer_pattern": "RGGB",
    "column_offset": "even",
    "row_offset": "even",
}
RAW_CHOICES = {
    "packing": tuple(PACKINGS),
    "size_policy": ("exact", "at_least"),
    "alignment": ("right", "left"),
    "bayer_pattern": ("monochrome", *BAYER_PATTERNS),
    "column_offset": ("even", "odd"),
    "row_offset": ("even", "odd"),
}
RAW_RANGES = {
    "width": NumericRange(1, 16384, whole=True),
    "height": NumericRange(1, 16384, whole=True),
    "row_stride": NumericRange(0, 1048576, whole=True),
    "byte_offset": NumericRange(0, MAX_FRAME_BYTES, whole=True),
}
RAW_HELP = {
    "packing": (
        "MIPI RAW10: four high bytes plus packed low 2-bit fields; RAW12: two high "
        "bytes plus low nibbles. Other vendor/PFNC layouts are not inferred."
    ),
    "width": "Decoded pixels per row; RAW10 needs a multiple of four, RAW12 a multiple of two.",
    "height": "Decoded image rows.",
    "row_stride": (
        "Bytes per stored row, including padding on the final row. Zero uses tightly packed rows."
    ),
    "byte_offset": "Bytes to skip before the first row, such as a known file header.",
    "size_policy": (
        "Exact rejects extra bytes. At least selects this frame from a buffer/file with"
        " trailing data."
    ),
    "alignment": (
        "Right retains ADC counts (0..1023 or 0..4095). Left shifts into uint16; it "
        "does not stretch the maximum to 65535."
    ),
    "bayer_pattern": (
        "Sensor tile at its original origin. Monochrome disables metadata-driven demosaicing."
    ),
    "column_offset": "Parity of the captured first column relative to the sensor tile origin.",
    "row_offset": "Parity of the captured first row relative to the sensor tile origin.",
}


def packed_raw_layout(p):
    for key, rule in RAW_RANGES.items():
        rule.check(p[key], key.replace("_", " ").capitalize(), key)
    width, height = int(p["width"]), int(p["height"])
    bits, pixels, group_bytes = PACKINGS[p["packing"]]
    if width % pixels:
        raise ValidationError(f"{p['packing']} width must be a multiple of {pixels}", "width")
    if width * height > 16_777_216:
        raise ValidationError("Decoded RAW frame exceeds the 16,777,216-pixel limit")
    row_bytes = width // pixels * group_bytes
    stride = int(p["row_stride"]) or row_bytes
    if stride < row_bytes:
        raise ValidationError("Row stride is smaller than the packed pixel row", "row_stride")
    offset = int(p["byte_offset"])
    end = offset + stride * height
    if end > MAX_FRAME_BYTES:
        raise ValidationError("RAW header and frame exceed the 128 MiB buffer limit")
    return {
        "width": width,
        "height": height,
        "bit_depth": bits,
        "row_bytes": row_bytes,
        "row_stride": stride,
        "byte_offset": offset,
        "frame_bytes": stride * height,
        "end_offset": end,
    }


def unpack_raw_sensor(buffer, p):
    layout = packed_raw_layout(p)
    if isinstance(buffer, np.ndarray):
        if buffer.dtype != np.uint8 or buffer.ndim not in (1, 2) or not buffer.size:
            raise ValidationError("RAW buffer must be a nonempty uint8 vector or byte matrix")
        data = np.ascontiguousarray(buffer).reshape(-1)
    elif isinstance(buffer, (bytes, bytearray, memoryview)):
        try:
            view = memoryview(buffer)
            if not view.contiguous or view.itemsize != 1:
                raise ValueError("Byte view must be contiguous with one-byte elements")
            data = np.frombuffer(view, np.uint8)
        except (ValueError, TypeError, BufferError) as exc:
            raise ValidationError("RAW input needs a contiguous byte buffer") from exc
    else:
        raise ValidationError("RAW input requires bytes or a uint8 byte array")
    end = layout["end_offset"]
    if data.size < end or (p["size_policy"] == "exact" and data.size != end):
        kind = "exactly" if p["size_policy"] == "exact" else "at least"
        raise ValidationError(f"RAW frame requires {kind} {end:,} bytes including offset/padding")
    rows = data[layout["byte_offset"] : end].reshape(layout["height"], layout["row_stride"])
    groups = rows[:, : layout["row_bytes"]].reshape(layout["height"], -1, PACKINGS[p["packing"]][2])
    if layout["bit_depth"] == 10:
        values = groups[..., :4].astype(np.uint16) << 2
        low = groups[..., 4].astype(np.uint16)
        for index in range(4):
            values[..., index] |= (low >> (2 * index)) & 3
    else:
        values = groups[..., :2].astype(np.uint16) << 4
        low = groups[..., 2].astype(np.uint16)
        values[..., 0] |= low & 15
        values[..., 1] |= low >> 4
    image = values.reshape(layout["height"], layout["width"])
    if p["alignment"] == "left":
        image <<= 16 - layout["bit_depth"]
    return ImageArray(image, "gray"), layout


def bayer_pattern_from_sensor(payload):
    """Follow Bayer phase through integer crops, flips and right-angle rotations."""
    raw = payload.get("raw_sensor")
    if (
        not isinstance(raw, dict)
        or raw.get("model") != "raw_sensor_v1"
        or raw.get("bayer_pattern") not in BAYER_PATTERNS
        or raw.get("mosaic") is not True
        or raw.get("column_offset") not in ("even", "odd")
        or raw.get("row_offset") not in ("even", "odd")
    ):
        raise ValidationError(
            "Input needs valid, unconsumed Bayer sensor metadata", "use_sensor_metadata"
        )
    source, current = raw.get("pixel_spatial"), payload.get("spatial")
    try:
        validate_payload({"spatial": source})
        validate_payload({"spatial": current})
    except (ValueError, TypeError, KeyError) as exc:
        raise ValidationError(
            "Sensor Bayer metadata needs valid source/current pixel frames"
        ) from exc
    if any(source[key] != current[key] for key in ("root_id", "root_size")):
        raise ValidationError("Sensor Bayer metadata uses a different coordinate frame")
    transform = np.linalg.solve(np.asarray(source["to_root"]), current["to_root"])
    integer = np.rint(transform)
    linear = integer[:2, :2]
    if (
        not np.allclose(transform, integer, atol=1e-9, rtol=0)
        or not np.array_equal(integer[2], [0, 0, 1])
        or not np.array_equal(np.abs(linear).sum(axis=0), [1, 1])
        or not np.array_equal(np.abs(linear).sum(axis=1), [1, 1])
    ):
        raise ValidationError(
            "Bayer metadata requires an unresampled pixel lattice; "
            "resized/warped mosaics cannot be inferred"
        )
    ox, oy = int(raw["column_offset"] == "odd"), int(raw["row_offset"] == "odd")
    pattern = raw["bayer_pattern"]
    result = []
    for y in range(2):
        for x in range(2):
            px, py = (integer @ [x, y, 1])[:2].astype(np.int64)
            result.append(pattern[2 * ((int(py) + oy) % 2) + (int(px) + ox) % 2])
    return "".join(result)


class _RawSensorBlock(Block):
    outputs = (Port("image"), Port("info", "data"))
    defaults, parameter_choices, parameter_ranges = RAW_DEFAULTS, RAW_CHOICES, RAW_RANGES
    parameter_help = RAW_HELP
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        packed_raw_layout(self.parameters if parameters is None else parameters)

    def decode(self, buffer, inputs):
        image, layout = unpack_raw_sensor(buffer, self.parameters)
        payload = copy.deepcopy(self.input_payloads.get("buffer", {}))
        payload.pop("spatial", None)
        payload.pop("pyramid", None)
        spatial = inputs.get(
            "pixel_frame", frame([layout["width"], layout["height"]], f"{self.id}:decoded")
        )
        try:
            validate_payload({"spatial": spatial})
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                "Pixel frame must be a valid spatial frame", "pixel_frame"
            ) from exc
        if spatial["image_size"] != [layout["width"], layout["height"]]:
            raise ValidationError(
                "Pixel frame dimensions must match the decoded image", "pixel_frame"
            )
        payload["spatial"] = copy.deepcopy(spatial)
        p = self.parameters
        maximum = (1 << layout["bit_depth"]) - 1
        if p["alignment"] == "left":
            maximum <<= 16 - layout["bit_depth"]
        raw = {
            "model": "raw_sensor_v1",
            "packing": p["packing"],
            "bit_depth": layout["bit_depth"],
            "alignment": p["alignment"],
            "max_code": maximum,
            "bayer_pattern": p["bayer_pattern"],
            "column_offset": p["column_offset"],
            "row_offset": p["row_offset"],
            "mosaic": p["bayer_pattern"] != "monochrome",
            "pixel_spatial": copy.deepcopy(spatial),
        }
        payload["raw_sensor"] = raw
        info = {**layout, **{key: value for key, value in raw.items() if key != "pixel_spatial"}}
        result = Result(
            {"image": image, "info": info},
            payloads={"image": payload, "info": payload},
            message=f"{p['packing']} {layout['width']}x{layout['height']}, uint16 {p['alignment']}",
        )
        if self.capture_graphics:
            result.graphics["sensor counts preview"] = cv2.convertScaleAbs(
                image, alpha=255 / maximum
            )
        if self.generate_graphics:
            label = graphic_object("label", ((8, 22),), ACCEPTED, text=result.message)
            result.drawings["RAW sensor decode"] = GraphicDrawing((label,), from_output=True)
        return result


class DecodePackedRawSensor(_RawSensorBlock):
    process_inputs_readonly = True
    type_id, title = "cv.decode_raw_sensor", "Unpack RAW10 / RAW12 sensor frame"
    category, block_type, subtype = "Image processing", "Image processing", "Color and depth"
    inputs = (
        Port("buffer", "any", visual_type="data"),
        Port("pixel_frame", "data", required=False, required_when_connected=True),
    )
    input_help = {
        "buffer": (
            "Bytes, bytearray, or uint8 vector/matrix. Convert SDK memoryviews to bytes"
            " before connecting; geometry is byte storage."
        ),
        "pixel_frame": (
            "Optional decoded-pixel spatial frame; byte-grid coordinates are not reused."
        ),
    }

    def process(self, inputs):
        return self.decode(inputs["buffer"], inputs)


class RawSensorFile(_RawSensorBlock):
    process_inputs_readonly = True
    type_id, title, category = "input.raw_sensor", "Packed RAW10 / RAW12 file", "Input"
    block_type, subtype = "Input", "Files"
    inputs = DecodePackedRawSensor.inputs[1:]
    input_help = {"pixel_frame": DecodePackedRawSensor.input_help["pixel_frame"]}
    file_source, finite_source = True, False
    defaults = {
        "path": "",
        "path_base": "working_directory",
        "filename_ticket": True,
        **RAW_DEFAULTS,
    }
    parameter_choices = {"path_base": ("working_directory", "project"), **RAW_CHOICES}
    parameter_help = {
        "path": "One headerless/offset packed sensor frame, with explicit byte layout.",
        "filename_ticket": "Use the filename as the automatic source ticket.",
        **RAW_HELP,
    }

    def issue_source_ticket(self):
        ticket = super().issue_source_ticket()
        if self.parameters["filename_ticket"] and self.ticket_settings["mode"] == "auto_number":
            value = self.parameters["path"]
            if isinstance(value, str) and value.strip():
                ticket.update(
                    kind="string",
                    value=Path(value).name,
                    session=f"{ticket['session']}:raw:{ticket['sequence']}",
                )
        return ticket

    def process(self, inputs):
        p, layout = self.parameters, packed_raw_layout(self.parameters)
        if not isinstance(p["path"], str) or not p["path"].strip():
            raise ValidationError("Select a packed RAW sensor file", "path")
        path = Path(p["path"])
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save or open the project before using a relative RAW path")
            path = self.project_dir / path
        with path.open("rb") as stream:
            buffer = stream.read(layout["end_offset"] + 1)
        result = self.decode(buffer, inputs)
        result.outputs["info"]["filename"] = path.name
        return result


RAW_SENSOR_BLOCKS = (DecodePackedRawSensor, RawSensorFile)
