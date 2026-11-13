"""Native QR symbols with explicit text encodings, quiet zones and separate output grids."""

import copy
import hashlib
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, ValuePacket, frame

MODES = {
    "auto": cv2.QRCodeEncoder_MODE_AUTO,
    "numeric": cv2.QRCodeEncoder_MODE_NUMERIC,
    "alphanumeric": cv2.QRCodeEncoder_MODE_ALPHANUMERIC,
    "byte": cv2.QRCodeEncoder_MODE_BYTE,
    "eci_utf8": cv2.QRCodeEncoder_MODE_ECI,
    "kanji": cv2.QRCodeEncoder_MODE_KANJI,
    "structured_append": cv2.QRCodeEncoder_MODE_STRUCTURED_APPEND,
}
CORRECTION = {
    "L": cv2.QRCodeEncoder_CORRECT_LEVEL_L,
    "M": cv2.QRCodeEncoder_CORRECT_LEVEL_M,
    "Q": cv2.QRCodeEncoder_CORRECT_LEVEL_Q,
    "H": cv2.QRCodeEncoder_CORRECT_LEVEL_H,
}
ALPHANUMERIC = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")


def native_text(text, mode):
    if not isinstance(text, str) or not text or "\0" in text:
        raise ValidationError("Use nonempty text without embedded NUL", "text")
    try:
        raw = text.encode("shift_jis" if mode == "kanji" else "utf8")
    except UnicodeError as exc:
        raise ValidationError(
            "Text cannot be represented in the selected encoding", "text", "mode"
        ) from exc
    if len(raw) > 8192:
        raise ValidationError(
            "QR input is limited to 8192 encoded bytes; native capacity may be lower", "text"
        )
    if mode == "numeric" and any(c not in "0123456789" for c in text):
        raise ValidationError("Numeric mode accepts ASCII digits only", "text", "mode")
    if mode == "alphanumeric" and not set(text) <= ALPHANUMERIC:
        raise ValidationError(
            "Alphanumeric mode accepts uppercase QR alphabet characters", "text", "mode"
        )
    if mode == "kanji":
        if len(raw) % 2 or any(
            not (
                0x8140 <= int.from_bytes(raw[i : i + 2], "big") <= 0x9FFC
                or 0xE040 <= int.from_bytes(raw[i : i + 2], "big") <= 0xEBBF
            )
            for i in range(0, len(raw), 2)
        ):
            raise ValidationError(
                "Kanji mode requires double-byte QR-range Shift-JIS characters", "text", "mode"
            )
    if mode == "structured_append" and not text.isascii():
        raise ValidationError(
            "Structured-append generation currently requires ASCII text", "text", "mode"
        )
    return (raw if mode == "kanji" else text), raw


def module_grid(native):
    if not isinstance(native, np.ndarray) or native.ndim != 2 or native.dtype != np.uint8:
        raise ValidationError("Native encoder returned an invalid symbol image")
    if native.shape[0] != native.shape[1] or not np.isin(native, [0, 255]).all():
        raise ValidationError("Native QR symbol must be a square binary image")
    y, x = np.where(native == 0)
    if not len(y):
        raise ValidationError("Native encoder returned no dark modules")
    borders = (
        int(y.min()),
        int(x.min()),
        native.shape[0] - 1 - int(y.max()),
        native.shape[1] - 1 - int(x.max()),
    )
    if len(set(borders)) != 1:
        raise ValidationError("Native QR quiet zone is not symmetric")
    quiet = borders[0]
    core = (
        native[quiet : native.shape[0] - quiet, quiet : native.shape[1] - quiet]
        if quiet
        else native
    )
    side = core.shape[0]
    if not 21 <= side <= 177 or (side - 17) % 4:
        raise ValidationError("Native QR module dimensions do not encode a valid version")
    return core.copy(), (side - 17) // 4, quiet


class EncodeQRCode(Block):
    type_id, title = "cv.encode_qr", "Generate QR labels"
    category = block_type = "Image processing"
    subtype = "Code generation"
    payload_reference = "text"
    inputs = (Port("text", "text", required=False, required_when_connected=True),)
    outputs = (
        Port("image"),
        Port("symbols", "any"),
        Port("modules", required=False),
        Port("count", "integer"),
        Port("text", "text"),
        Port("info", "data"),
    )
    defaults = {
        "text": "PART-0042/LOT-A",
        "version": 0,
        "correction_level": "M",
        "mode": "auto",
        "structure_number": 2,
        "pixels_per_module": 8,
        "quiet_zone": 4,
        "sheet_columns": 0,
        "sheet_gap": 4,
        "maximum_output_pixels": 16_777_216,
    }
    parameter_choices = {"correction_level": tuple(CORRECTION), "mode": tuple(MODES)}
    parameter_ranges = {
        "version": NumericRange(0, 40, whole=True),
        "structure_number": NumericRange(1, 16, whole=True),
        "pixels_per_module": NumericRange(1, 32, whole=True),
        "quiet_zone": NumericRange(0, 64, whole=True),
        "sheet_columns": NumericRange(0, 16, whole=True),
        "sheet_gap": NumericRange(0, 64, whole=True),
        "maximum_output_pixels": NumericRange(1, 32_000_000, whole=True),
    }
    parameter_help = {
        "text": "Default payload; a wired Text input overrides it. No truncation or NUL padding.",
        "version": "Native version 1..40, or zero for automatic sizing; capacity errors fail.",
        "correction_level": "Native L/M/Q/H error-correction level, not a print-quality grade.",
        "mode": "Native mode; ECI uses UTF-8. Kanji converts validated double-byte Shift-JIS.",
        "structure_number": "Native structured_append part count; retained when inactive.",
        "pixels_per_module": "Integer nearest-neighbor scaling; no interpolation of module edges.",
        "quiet_zone": "Total blank modules per edge, replacing the native border; default four.",
        "sheet_columns": "Contact-sheet columns; zero chooses a compact grid.",
        "sheet_gap": "Additional white modules between labels on a contact sheet.",
        "maximum_output_pixels": "Symbol/sheet pixel budget before scaling; copies add memory.",
    }
    input_help = {
        "text": "Optional text with ticket/companion payload; generated labels get new grids."
    }
    process_inputs_readonly = parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "count", "info"]

    def process(self, inputs):
        p = self.parameters
        text = inputs.get("text", p["text"])
        encoded, raw = native_text(text, p["mode"])
        structured = p["mode"] == "structured_append"
        parameters = cv2.QRCodeEncoder_Params()
        parameters.version = int(p["version"])
        parameters.correction_level = CORRECTION[p["correction_level"]]
        parameters.mode = MODES[p["mode"]]
        parameters.structure_number = int(p["structure_number"]) if structured else 1
        if structured and parameters.structure_number > len(raw):
            raise ValidationError(
                "Structured-append count must not exceed the ASCII byte count",
                "structure_number",
                "text",
            )
        try:
            encoder = cv2.QRCodeEncoder_create(parameters)
            native = (
                encoder.encodeStructuredAppend(encoded) if structured else [encoder.encode(encoded)]
            )
        except cv2.error as exc:
            raise ValidationError(
                "Native QR encoding failed; check version, mode and payload capacity",
                "version",
                "mode",
                "text",
            ) from exc
        if len(native) != parameters.structure_number:
            raise ValidationError("Native structured-append count differs from the requested count")
        grids = [module_grid(symbol) for symbol in native]
        if p["version"] and any(version != p["version"] for _, version, _ in grids):
            raise ValidationError("Native encoder did not honor the requested QR version")
        scale, quiet = int(p["pixels_per_module"]), int(p["quiet_zone"])
        sizes = [(grid.shape[0] + 2 * quiet) * scale for grid, _, _ in grids]
        count = len(grids)
        columns = min(count, int(p["sheet_columns"]) or math.ceil(math.sqrt(count)))
        rows = math.ceil(count / columns)
        cell, gap = max(sizes), int(p["sheet_gap"]) * scale
        width, height = columns * cell + (columns - 1) * gap, rows * cell + (rows - 1) * gap
        work = sum(size * size for size in sizes) + (width * height if count > 1 else 0)
        if work > p["maximum_output_pixels"] or max(width, height) > 16384:
            raise ValidationError(
                "QR label/sheet exceeds output pixel or axis limits",
                "maximum_output_pixels",
                "pixels_per_module",
            )
        context = copy.deepcopy(self.input_payloads.get("text", {}))
        for name in ("spatial", "pyramid", "raw_sensor", "alpha_representation"):
            context.pop(name, None)
        group_id = hashlib.sha256(raw).hexdigest()
        symbols, details, objects = [], [], []
        sheet = np.full((height, width), 255, np.uint8) if count > 1 else None
        for index, ((grid, version, native_quiet), size) in enumerate(
            zip(grids, sizes, strict=True)
        ):
            padded = cv2.copyMakeBorder(
                grid, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
            )
            pixels = cv2.resize(padded, (size, size), interpolation=cv2.INTER_NEAREST)
            x, y = (index % columns) * (cell + gap), (index // columns) * (cell + gap)
            if sheet is not None:
                sheet[y : y + size, x : x + size] = pixels
            part = {
                "group_sha256": group_id,
                "position": index,
                "count": count,
                "requires_reassembly": structured,
                "version": version,
                "quiet_zone": quiet,
                "pixels_per_module": scale,
                "module_size": int(grid.shape[0]),
            }
            payload = {
                **copy.deepcopy(context),
                "qr_label": part,
                "spatial": frame([size, size], f"{self.id}:qr:{index}"),
            }
            image = ImageArray(pixels, "gray", payload)
            symbols.append(ValuePacket(image, payload))
            details.append(
                {**part, "bounds": [x, y, size, size], "native_quiet_zone": native_quiet}
            )
            if self.generate_graphics:
                objects.append(graphic_box(x, y, size, size, (60, 200, 100)))
        if sheet is None:
            image, image_payload = symbols[0].value, symbols[0].payload
        else:
            image_payload = {
                **copy.deepcopy(context),
                "spatial": frame([width, height], f"{self.id}:qr_sheet"),
            }
            image = ImageArray(sheet, "gray", image_payload)
        modules, module_payload = NO_VALUE, {}
        if count == 1 and "modules" in self.exposed_outputs:
            grid = grids[0][0]
            module_payload = {
                **copy.deepcopy(context),
                "spatial": frame(list(grid.shape[::-1]), f"{self.id}:qr_modules"),
            }
            modules = ImageArray(grid, "gray", module_payload)
        outputs = {
            "image": image,
            "symbols": FlowGroup(tuple(symbols), tuple(f"symbol-{i}" for i in range(count))),
            "modules": modules,
            "count": count,
            "text": text,
            "info": {
                "model": "qr_labels_v1",
                "mode": p["mode"],
                "correction_level": p["correction_level"],
                "input_bytes": len(raw),
                "encoding": "shift_jis" if p["mode"] == "kanji" else "utf8",
                "requires_reassembly": structured,
                "symbols": details,
                "sheet_size": [width, height],
                "payload_sha256": group_id,
            },
        }
        payloads = {name: copy.deepcopy(context) for name in outputs}
        payloads["image"] = image_payload
        if modules is not NO_VALUE:
            payloads["modules"] = module_payload
        return Result(
            outputs,
            payloads=payloads,
            drawings={
                "label bounds": GraphicDrawing(tuple(objects), source="image", from_output=True)
            }
            if objects
            else {},
            message=f"Generated {count} QR label(s) · {p['mode']} / {p['correction_level']}",
        )
