"""Strict, bounded JSON persistence for reusable mosaic calibration geometry."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .core import Block, Port, Result
from .mosaic_layout import validate_layout
from .parameter_validation import ValidationError

MAX_BYTES = 1024 * 1024


def unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValidationError("Duplicate layout JSON field: " + name)
        result[name] = value
    return result


def layout_bytes(layout):
    validate_layout(layout)
    try:
        raw = (json.dumps(layout, allow_nan=False, ensure_ascii=False, indent=2) + "\n").encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError("Mosaic layout must be finite JSON data") from exc
    if len(raw) > MAX_BYTES:
        raise ValidationError("Layout exceeds the 1 MiB file limit")
    return raw


def layout_path(block):
    value = block.parameters["path"]
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\0" in value
        or Path(value).suffix.lower() != ".json"
    ):
        raise ValidationError("Choose a mosaic layout .json file", "path")
    path = Path(value)
    if not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError("Save/open the project or choose an absolute path", "path")
        path = block.project_dir / path
    return path.resolve()


class LoadMosaicLayout(Block):
    type_id, title = "cv.load_mosaic_layout", "Load mosaic layout"
    category, block_type, subtype = "Image processing", "Image processing", "Calibrated layouts"
    outputs = (Port("layout", "data"), Port("info", "data"))
    defaults = {"path": "", "expected_sha256": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")
    parameter_help = {
        "path": "Strict JSON layout, up to 1 MiB. Relative paths use the saved project folder.",
        "expected_sha256": "Optional file-byte SHA256; Save As copies the sealed layout asset.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        digest = p["expected_sha256"]
        if not isinstance(digest, str) or (
            digest and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest))
        ):
            raise ValidationError(
                "Expected SHA256 must be empty or 64 hex digits", "expected_sha256"
            )

    def process(self, inputs):
        path = layout_path(self)
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("Layout exceeds the 1 MiB file limit", "path")
        digest = hashlib.sha256(raw).hexdigest()
        if (
            self.parameters["expected_sha256"]
            and digest != self.parameters["expected_sha256"].lower()
        ):
            raise ValidationError(
                "Layout file digest differs from expected bytes", "expected_sha256"
            )
        try:
            layout = json.loads(
                raw.decode("utf8"),
                object_pairs_hook=unique_fields,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValidationError("Invalid strict UTF-8 layout JSON: " + str(exc), "path") from exc
        validate_layout(layout)
        return Result(
            {
                "layout": layout,
                "info": {
                    "sha256": digest,
                    "bytes": len(raw),
                    "model": layout["model"],
                    "layout_id": layout.get("layout_id"),
                    "views": len(layout["views"]),
                },
            },
            message="Loaded mosaic calibration: " + path.name,
        )


class SaveMosaicLayout(Block):
    type_id, title = "output.mosaic_layout", "Save mosaic layout"
    category, block_type, subtype = "Output", "Output", "Calibrated layouts"
    preview_safe = False
    inputs = (Port("layout", "data"),)
    outputs = (Port("info", "data"),)
    defaults = {"path": "mosaic_layout.json", "write_mode": "new"}
    parameter_choices = {"write_mode": ("new", "replace")}
    parameter_help = {
        "path": "Destination .json; relative paths use the saved project folder.",
        "write_mode": "New preserves existing files. Replace atomically publishes this layout.",
    }

    def process(self, inputs):
        raw = layout_bytes(inputs["layout"])
        path = layout_path(self)
        if self.parameters["write_mode"] == "new":
            with path.open("xb") as stream:
                stream.write(raw)
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=".mosaic-", suffix=".tmp", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return Result(
            {
                "info": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "layout_id": inputs["layout"].get("layout_id"),
                }
            },
            message="Saved mosaic calibration: " + path.name,
        )
