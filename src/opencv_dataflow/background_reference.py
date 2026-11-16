"""Sealed clean-background images for bounded native model reconstruction."""

import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import numpy as np

from .core import Block, Port, Result
from .image_assets import decode_image, encode_image, seal
from .images import ImageArray, image_format
from .parameter_validation import ValidationError
from .payloads import check_spatial
from .processing import image_value
from .selection import selection_mask

MODEL = "background_reference_v1"
MAX_BYTES = 12 * 1024**2


def identifiers(camera, calibration, *, fields=("camera_id", "calibration_id")):
    if (
        any(not isinstance(v, str) or len(v) > 256 or v != v.strip() for v in (camera, calibration))
        or not camera
    ):
        raise ValidationError(
            "Camera ID is required; camera/calibration IDs use at most 256 trimmed characters",
            *fields,
        )


def grid_context(context, recipe_binding):
    result = copy.deepcopy(
        {
            k: v
            for k, v in context.items()
            if k in ("spatial", "pyramid") or k == "recipe_ref" and recipe_binding == "strict"
        }
    )
    if "spatial" in result:
        result["spatial"].pop("root_id", None)  # Acquisition roots change after restart.
    return result


def validate_reference(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "camera_id", "calibration_id", "recipe_binding", "image", "id"}
        or value.get("model") != MODEL
        or value.get("recipe_binding") not in ("strict", "ignore")
    ):
        raise ValidationError("Choose a supported background_reference_v1 record")
    identifiers(value["camera_id"], value["calibration_id"])
    image = decode_image(value["image"])
    if value["id"] != seal(value):
        raise ValidationError("Background reference checksum mismatch")
    image_value(image, (np.uint8, np.float32))
    if image_format(image) not in ("gray", "rgb", "bgr"):
        raise ValidationError("Background references need grayscale, RGB or BGR pixels")
    if image.dtype == np.float32 and ((image < 0) | (image > 255)).any():
        raise ValidationError("Float background references must use [0,255] intensity units")
    try:
        check_spatial(image.metadata, list(image.shape[1::-1]))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return image


def make_reference(image, camera_id, calibration_id="", recipe_binding="strict"):
    identifiers(camera_id, calibration_id)
    record = {
        "model": MODEL,
        "camera_id": camera_id,
        "calibration_id": calibration_id,
        "recipe_binding": recipe_binding,
        "image": encode_image(image),
    }
    record["id"] = seal(record)
    validate_reference(record)
    return record


def match_reference(record, reference, image, payload, camera_id, calibration_id):
    identifiers(
        camera_id, calibration_id, fields=("reference_camera_id", "reference_calibration_id")
    )
    if camera_id != record["camera_id"] or calibration_id != record["calibration_id"]:
        raise ValidationError(
            "Background reference camera/calibration identity differs from this setup",
            "reference_camera_id",
            "reference_calibration_id",
        )
    if (
        reference.shape != image.shape
        or reference.dtype != image.dtype
        or image_format(reference) != image_format(image)
    ):
        raise ValidationError("Background reference size/depth/encoding differs from current input")
    if grid_context(reference.metadata, record["recipe_binding"]) != grid_context(
        payload, record["recipe_binding"]
    ):
        raise ValidationError("Background reference grid/pyramid/recipe differs from current input")


def reference_bytes(record):
    validate_reference(record)
    raw = (json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValidationError("Background reference JSON exceeds 12 MiB")
    return raw


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate reference JSON field: " + key)
        result[key] = value
    return result


def load_reference(path, expected_sha256=""):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValidationError("Background reference must be a regular file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > MAX_BYTES or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValidationError("Oversized or changing background reference file")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValidationError(
            "Background reference file differs from expected SHA256", "reference_sha256"
        )
    try:
        record = json.loads(
            raw.decode("utf8"),
            object_pairs_hook=_unique,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
        )
        image = validate_reference(record)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("Invalid background reference: " + str(exc)) from exc
    return {k: v for k, v in record.items() if k != "image"}, image, digest


def reference_path(block, field):
    value = block.parameters[field]
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\0" in value
        or Path(value).suffix.lower() != ".json"
    ):
        raise ValidationError("Choose a background reference .json path", field)
    path = Path(value)
    if not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError("Save/open the project or use an absolute reference path", field)
        path = block.project_dir / path
    return path.resolve()


class CaptureBackgroundReference(Block):
    type_id, title = "sequence.capture_background_reference", "Capture clean background reference"
    category = block_type = "Sequence"
    subtype = "Foreground segmentation"
    inputs = (Port("image"), Port("known_foreground", required=False, required_when_connected=True))
    outputs = (Port("reference", "data"), Port("image"), Port("info", "data"))
    defaults = {"camera_id": "", "calibration_id": "", "recipe_binding": "strict"}
    parameter_choices = {"recipe_binding": ("strict", "ignore")}
    parameter_help = {
        "camera_id": "Required stable camera/setup identity; copied into the sealed reference.",
        "calibration_id": (
            "Optional calibration revision; restored consumers must match it exactly."
        ),
        "recipe_binding": (
            "Strict checks captured recipe context; Ignore permits recipe-independent backgrounds."
        ),
    }
    input_help = {
        "image": "Clean gray/RGB/BGR uint8 or float32 [0,255] scene, at most 8 MiB.",
        "known_foreground": "Any marked pixel rejects capture; wait for a clean scene first.",
    }
    parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.documentation = (
            "<p>Capture an entire clean background image as a versioned, checksummed record. "
            "Set stable camera and calibration identities and choose recipe binding. Any marked "
            "known foreground rejects capture. Unmarked foreground cannot be detected reliably "
            "by this recorder. Save the record with Save background reference, then select that "
            "JSON in a MOG2 or KNN block's Reference path.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        identifiers(p["camera_id"], p["calibration_id"])

    def process(self, inputs):
        source = inputs["image"]
        known = selection_mask({"image": source, "mask": inputs.get("known_foreground")}, source)
        if known is not None and known.any():
            raise ValidationError(
                "Known foreground is present; background reference capture rejected"
            )
        p = self.parameters
        record = make_reference(
            ImageArray(source, image_format(source), self.input_payloads["image"]),
            p["camera_id"],
            p["calibration_id"],
            p["recipe_binding"],
        )
        return Result(
            {
                "reference": record,
                "image": source,
                "info": {
                    "reference_id": record["id"],
                    "camera_id": p["camera_id"],
                    "calibration_id": p["calibration_id"],
                },
            },
            message="Captured clean background reference",
        )


class SaveBackgroundReference(Block):
    type_id, title = "output.background_reference", "Save background reference"
    category = block_type = "Output"
    subtype = "Foreground segmentation"
    preview_safe = False
    inputs = (Port("reference", "data"),)
    outputs = (Port("info", "data"),)
    defaults = {"path": "background_reference.json", "write_mode": "new"}
    parameter_choices = {"write_mode": ("new", "replace")}
    parameter_help = {
        "path": "Versioned UTF-8 JSON relative to the project; directory must exist.",
        "write_mode": "New preserves existing files; Replace publishes atomically.",
    }

    def process(self, inputs):
        raw = reference_bytes(inputs["reference"])
        path = reference_path(self, "path")
        if self.parameters["write_mode"] == "new":
            with path.open("xb") as stream:
                stream.write(raw)
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=".background-", suffix=".tmp", delete=False
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
                    "reference_id": inputs["reference"]["id"],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            },
            message="Saved background reference: " + path.name,
        )
