"""Bounded native .flo transport with optional content-bound image-pair metadata."""

import copy
import hashlib
import json
import os
import re
import stat
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, check_spatial, validate_payload
from .sequence_tools import DenseFlow

MAX_PIXELS = 2_100_000
MAX_BYTES = 12 + MAX_PIXELS * 8
MAX_METADATA = 65536
DIRECTIONS = ("unspecified", "source_to_target", "target_to_source")
MODEL = "opencv_dataflow_flow_file_v1"
ANCHORS = tuple(
    Port(name, required=False, required_when_connected=True) for name in ("source", "target")
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def header(data):
    if len(data) < 12:
        raise ValidationError("Truncated .flo header")
    magic, width, height = struct.unpack("<4sii", data[:12])
    if magic != b"PIEH" or not 1 <= width < 32767 or not 1 <= height < 32767:
        raise ValidationError("Invalid .flo signature or dimensions")
    if width * height > MAX_PIXELS:
        raise ValidationError("Flow files are limited to 2.1M vectors")
    return width, height


def field_hash(vectors):
    h, w = vectors.shape[:2]
    return digest(struct.pack("<4sii", b"PIEH", w, h) + np.asarray(vectors, "<f4").tobytes())


def observations(vectors):
    nonfinite = ~np.isfinite(vectors).all(axis=2)
    unknown = np.isfinite(vectors).all(axis=2) & (np.abs(vectors) > 1e9).any(axis=2)
    return {
        "nonfinite_vectors": int(nonfinite.sum()),
        "unknown_vectors": int(unknown.sum()),
        "invalid_vectors": int((nonfinite | unknown).sum()),
    }


def endpoint(image, payload, label=""):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim not in (2, 3)
        or min(image.shape[:2]) < 1
        or image.shape[0] * image.shape[1] > MAX_PIXELS
        or image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
        or image.ndim == 3
        and image.shape[2] not in (1, 3, 4)
    ):
        raise ValidationError("Image anchors need bounded gray/color u8/u16/i16/f32/f64 pixels")
    context = {key: copy.deepcopy(payload[key]) for key in ("spatial", "pyramid") if key in payload}
    check_spatial(context, [image.shape[1], image.shape[0]])
    validate_payload(context)
    shape, dtype, encoding = list(image.shape), str(image.dtype), image_format(image)
    identity = json.dumps([shape, dtype, encoding], separators=(",", ":")).encode()
    return {
        "label": label,
        "image": {
            "shape": shape,
            "dtype": dtype,
            "encoding": encoding,
            "sha256": digest(identity + np.ascontiguousarray(image).tobytes()),
        },
        **context,
    }


def same_endpoint(saved, image, payload, name):
    actual = endpoint(image, payload)
    if saved["image"] != actual["image"]:
        raise ValidationError(
            f"Flow {name} image content/type/encoding does not match its manifest"
        )
    for key in ("spatial", "pyramid"):
        old, new = saved.get(key), actual.get(key)
        if key == "spatial" and old is not None and new is not None:
            old, new = dict(old), dict(new)
            old.pop("root_id", None)
            new.pop("root_id", None)
        if old != new:
            raise ValidationError(
                f"Flow {name} coordinate/pyramid geometry does not match its manifest"
            )


def validate_manifest(value, shape, sha256):
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "sha256", "shape", "direction", "source", "target"}
        or value["model"] != MODEL
        or value["shape"] != list(shape)
        or value["sha256"] != sha256
        or value["direction"] not in DIRECTIONS
    ):
        raise ValidationError(
            "Flow manifest schema, dimensions, direction or SHA-256 does not match"
        )
    if (value["source"] is None) != (value["target"] is None):
        raise ValidationError("Flow manifest needs both image endpoints or neither")
    for name in ("source", "target"):
        item = value[name]
        if item is None:
            continue
        if (
            not isinstance(item, dict)
            or item.keys() - {"label", "image", "spatial", "pyramid"}
            or not isinstance(item.get("label"), str)
            or len(item["label"]) > 256
        ):
            raise ValidationError("Invalid flow image endpoint")
        image = item.get("image")
        if (
            not isinstance(image, dict)
            or set(image) != {"shape", "dtype", "encoding", "sha256"}
            or not isinstance(image["shape"], list)
            or image["shape"][:2] != list(shape[:2])
            or len(image["shape"]) not in (2, 3)
            or image["dtype"] not in ("uint8", "uint16", "int16", "float32", "float64")
            or not isinstance(image["encoding"], str)
            or not isinstance(image["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", image["sha256"])
        ):
            raise ValidationError("Invalid flow image fingerprint")
        context = {key: item[key] for key in ("spatial", "pyramid") if key in item}
        check_spatial(context, [shape[1], shape[0]])
        validate_payload(context)
    return value


def check_use(value, payload, source, target, source_payload, target_payload, direction):
    """Validate file-bound metadata at a motion consumer; ordinary fields are unaffected."""
    association = payload.get("flow_association")
    if association is None:
        return {"present": False}
    vectors = value.vectors if isinstance(value, DenseFlow) else value
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.ndim != 3
        or vectors.shape[2] != 2
        or vectors.dtype not in (np.float32, np.float64)
    ):
        raise ValidationError("File-associated flow must remain a floating H×W×2 field")
    validate_manifest(association, vectors.shape, field_hash(vectors))
    if association["direction"] not in ("unspecified", direction):
        raise ValidationError(
            "Configured flow direction contradicts its file manifest", "direction"
        )
    checked = []
    for name, image, context in (
        ("source", source, source_payload),
        ("target", target, target_payload),
    ):
        if association[name] is not None and image is not None:
            same_endpoint(association[name], image, context, name)
            checked.append(name)
    return {"present": True, "direction": association["direction"], "checked_images": checked}


def read_bytes(path):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValidationError("Flow path must be a regular file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        prefix = stream.read(12)
        width, height = header(prefix)
        expected = width * height * 8
        if before.st_size != 12 + expected:
            raise ValidationError("Flow bytes do not match the declared dimensions")
        data = prefix + stream.read(expected + 1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValidationError("Flow file changed during the read")
    if len(data) != 12 + expected:
        raise ValidationError("Flow bytes do not match the declared dimensions")
    return data, width, height


def decode(data, width, height):
    with TemporaryDirectory(prefix="dataflow-flo-") as temporary:
        path = Path(temporary) / "snapshot.flo"
        path.write_bytes(data)
        vectors = cv2.readOpticalFlow(str(path))
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.dtype != np.float32
        or vectors.shape != (height, width, 2)
        or np.asarray(vectors, "<f4").tobytes() != data[12:]
    ):
        raise ValidationError("Native flow decoder did not preserve the validated float32 samples")
    return vectors


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("Duplicate flow manifest field")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValidationError("Flow manifest must contain finite JSON")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def path_for(block, value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\0" in value
        or Path(value).suffix.lower() != ".flo"
    ):
        raise ValidationError("Choose a .flo displacement file", "path")
    path = Path(value)
    if block.parameters["path_base"] == "project" and not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError(
                "Save or open the project before using a relative flow path", "path"
            )
        path = block.project_dir / path
    return path.resolve()


def publication(path, data, manifest):
    """Publish exclusively, metadata first; roll back only unchanged files we created."""
    companion = Path(str(path) + ".json")
    if any(p.exists() or p.is_symlink() for p in (path, companion)):
        raise FileExistsError("Flow destination or companion already exists")
    members = [(path, data)]
    if manifest is not None:
        encoded = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        if len(encoded) > MAX_METADATA:
            raise ValidationError("Flow manifest exceeds 64 KiB")
        members.insert(0, (companion, encoded))
    published = []
    try:
        for target, content in members:
            temporary = None
            try:
                with NamedTemporaryFile(dir=path.parent, prefix=".flow-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                os.link(temporary, target)
                published.append((target, temporary.stat(), digest(content)))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except BaseException:
        for target, original, sha256 in reversed(published):
            try:
                current = target.stat()
                if (current.st_dev, current.st_ino, current.st_size) == (
                    original.st_dev, original.st_ino, original.st_size
                ):
                    with target.open("rb") as stream:
                        retained = stream.read(original.st_size + 1)
                    if len(retained) == original.st_size and digest(retained) == sha256:
                        target.unlink()
            except OSError:
                pass
        raise


def preview(vectors, maximum):
    good = np.isfinite(vectors).all(axis=2) & (np.abs(vectors) <= 1e9).all(axis=2)
    safe = np.where(good[..., None], vectors, 0)
    magnitude, angle = cv2.cartToPolar(safe[..., 0], safe[..., 1], angleInDegrees=True)
    hsv = np.zeros((*vectors.shape[:2], 3), np.uint8)
    hsv[..., 0] = np.rint(angle / 2).astype(np.uint8) % 180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * 255 / maximum, 0, 255).astype(np.uint8)
    hsv[~good] = 0
    return ImageArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "bgr")


class StoredFlow(DenseFlow):
    def describe(self):
        return {
            "model": "dense_flow_v1",
            "shape": list(self.vectors.shape),
            "dtype": str(self.vectors.dtype),
            **observations(self.vectors),
            "coordinates": self.metadata.get("flow_association", {}).get(
                "direction", "File-defined displacement; consumer declares direction and grids"
            ),
        }


class FlowFileBlock(Block, ABC):
    preview_safe = False
    process_inputs_readonly = True

    @abstractmethod
    def process(self, inputs):
        """Load or save a field, then use the common result/preview contract."""
        raise NotImplementedError

    def finish(self, vectors, info, context, path=None):
        visible = NO_VALUE
        if self.generate_graphics or "visualization" in self.exposed_outputs:
            visible = preview(vectors, self.parameters["visual_max_motion"])
        outputs = {"flow": StoredFlow(vectors, context), "info": info, "visualization": visible}
        if path is not None:
            outputs["path"] = str(path)
        return Result(
            outputs,
            graphics={"Flow field": visible} if visible is not NO_VALUE else {},
            payloads={name: copy.deepcopy(context) for name in outputs},
        )


class LoadFlowFile(FlowFileBlock):
    type_id, title = "input.flow_file", "Load dense flow file"
    category = block_type = "Input"
    subtype = "Motion fields"
    inputs = (Port("path", "text", required=False, required_when_connected=True), *ANCHORS)
    outputs = (Port("flow", "any"), Port("info", "data"), Port("visualization", required=False))
    defaults = {
        "path": "",
        "path_base": "project",
        "metadata_mode": "auto",
        "expected_direction": "any",
        "invalid_vectors": "preserve",
        "visual_max_motion": 10.0,
    }
    parameter_choices = {
        "path_base": ("project", "working_directory"),
        "metadata_mode": ("auto", "ignore", "required", "verify_images"),
        "expected_direction": ("any", *DIRECTIONS),
        "invalid_vectors": ("preserve", "reject"),
    }
    parameter_ranges = {"visual_max_motion": NumericRange(0.001, 1e9)}
    parameter_help = {
        "path": "Middlebury .flo: float32 local-pixel dx/dy. A wired Path overrides this filename.",
        "metadata_mode": (
            "Optional .flo.json binds direction and image content; Verify needs both anchors."
        ),
        "expected_direction": (
            "Reject a different declaration; untagged files cannot establish direction."
        ),
        "invalid_vectors": (
            "Preserve/reject nonfinite and Middlebury unknown (>1e9 per-axis) vectors."
        ),
        "visual_max_motion": "Local-pixel magnitude at full brightness; invalid vectors are black.",
    }
    parallel_safe = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [name for name in self.exposed_outputs if name != "visualization"]

    def process(self, inputs):
        p = self.parameters
        path = path_for(self, inputs.get("path", p["path"]))
        data, width, height = read_bytes(path)
        context, association = {}, None
        mode = p["metadata_mode"]
        companion = Path(str(path) + ".json")
        if mode != "ignore" and companion.exists():
            with companion.open("rb") as stream:
                encoded = stream.read(MAX_METADATA + 1)
            if len(encoded) > MAX_METADATA:
                raise ValidationError("Flow manifest exceeds 64 KiB")
            association = validate_manifest(strict_json(encoded), (height, width, 2), digest(data))
        elif mode in ("required", "verify_images"):
            raise ValidationError("A matching .flo.json manifest is required", "metadata_mode")
        if p["expected_direction"] != "any" and (
            association is None or association["direction"] != p["expected_direction"]
        ):
            raise ValidationError(
                "The file does not establish the expected direction", "expected_direction"
            )
        verified = mode != "ignore" and (
            mode == "verify_images" or "source" in inputs or "target" in inputs
        )
        if verified:
            if (
                association is None
                or association["source"] is None
                or not {"source", "target"} <= inputs.keys()
            ):
                raise ValidationError(
                    "Image verification needs a paired manifest and both image anchors"
                )
            for name in ("source", "target"):
                same_endpoint(association[name], inputs[name], self.input_payloads[name], name)
            if association["direction"] != "unspecified":
                name = "source" if association["direction"] == "source_to_target" else "target"
                context = {
                    k: copy.deepcopy(v)
                    for k, v in self.input_payloads[name].items()
                    if k in ("spatial", "pyramid")
                }
        if association is not None:
            context["flow_association"] = copy.deepcopy(association)
        vectors = decode(data, width, height)
        counts = observations(vectors)
        if p["invalid_vectors"] == "reject" and counts["invalid_vectors"]:
            raise ValidationError(
                "Flow contains nonfinite or unknown displacement vectors", "invalid_vectors"
            )
        info = {
            "filename": path.name,
            "sha256": digest(data),
            "size": [width, height],
            **counts,
            "association": association or "unspecified; pair with intended images/direction",
            "images_verified": verified,
        }
        context["flow_file"] = copy.deepcopy(info)
        return self.finish(vectors, info, context)


class SaveFlowFile(FlowFileBlock):
    type_id, title = "output.flow_file", "Save dense flow file"
    category = block_type = "Output"
    subtype = "Motion fields"
    payload_reference = "flow"
    inputs = (Port("flow", "any"), *ANCHORS)
    outputs = (
        Port("path", "text"),
        Port("flow", "any"),
        Port("info", "data"),
        Port("visualization", required=False),
    )
    defaults = {
        "path": "flow.flo",
        "path_base": "project",
        "direction": "source_to_target",
        "metadata_mode": "if_connected",
        "source_label": "",
        "target_label": "",
        "precision": "float32_only",
        "invalid_vectors": "reject",
        "unique_name": False,
        "create_directories": False,
        "visual_max_motion": 10.0,
    }
    parameter_choices = {
        "path_base": ("project", "working_directory"),
        "direction": DIRECTIONS,
        "metadata_mode": ("none", "if_connected", "require_images"),
        "precision": ("float32_only", "convert_float64"),
        "invalid_vectors": ("reject", "preserve"),
    }
    parameter_ranges = {"visual_max_motion": NumericRange(0.001, 1e9)}
    parameter_help = {
        "direction": "Declared image-pair direction, not inferred from vector signs.",
        "metadata_mode": (
            "Optional SHA-bound .flo.json; both image anchors bind exact pixels and grids."
        ),
        "precision": "Native .flo is float32; conversion explicitly permits float64 rounding.",
        "unique_name": "Append a fresh UUID to avoid overwriting files on repeated runs.",
        "create_directories": "Create output directories only after input/config validation.",
        "invalid_vectors": (
            "Reject by default; Preserve writes nonfinite/unknown samples unchanged."
        ),
        "visual_max_motion": "Local-pixel magnitude at full brightness; invalid vectors are black.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs.remove("visualization")

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("unique_name", "create_directories"):
            if type(p[name]) is not bool:
                raise ValidationError(f"{name} must be boolean", name)
        for name in ("source_label", "target_label"):
            if not isinstance(p[name], str) or len(p[name]) > 256:
                raise ValidationError("Frame labels must be text of at most 256 characters", name)

    def process(self, inputs):
        p = self.parameters
        path = path_for(self, p["path"])
        value = inputs["flow"]
        raw = value.vectors if isinstance(value, DenseFlow) else value
        if (
            not isinstance(raw, np.ndarray)
            or raw.ndim != 3
            or raw.shape[2] != 2
            or min(raw.shape[:2]) < 1
            or max(raw.shape[:2]) >= 32767
            or raw.shape[0] * raw.shape[1] > MAX_PIXELS
            or raw.dtype not in (np.float32, np.float64)
        ):
            raise ValidationError("Flow must be float32/64 H×W×2, at most 2.1M vectors")
        if raw.dtype == np.float64 and p["precision"] != "convert_float64":
            raise ValidationError(
                "Choose Convert float64 to permit narrowing to float32", "precision"
            )
        if np.any(np.isfinite(raw) & (np.abs(raw) > np.finfo(np.float32).max)):
            raise ValidationError("Finite vectors overflow float32")
        vectors = np.ascontiguousarray(raw, dtype=np.float32)
        counts = observations(vectors)
        if p["invalid_vectors"] == "reject" and counts["invalid_vectors"]:
            raise ValidationError("Flow contains nonfinite or unknown vectors", "invalid_vectors")
        context = copy.deepcopy(self.input_payloads["flow"])
        check_use(
            value,
            context,
            inputs.get("source"),
            inputs.get("target"),
            self.input_payloads.get("source", {}),
            self.input_payloads.get("target", {}),
            p["direction"],
        )
        association = {
            "model": MODEL,
            "sha256": field_hash(vectors),
            "shape": list(vectors.shape),
            "direction": p["direction"],
            "source": None,
            "target": None,
        }
        anchors = {"source", "target"} & inputs.keys()
        if p["metadata_mode"] != "none" and (anchors or p["metadata_mode"] == "require_images"):
            if anchors != {"source", "target"}:
                raise ValidationError("Provide both Source and Target image anchors")
            for name in ("source", "target"):
                association[name] = endpoint(
                    inputs[name], self.input_payloads[name], p[name + "_label"]
                )
            validate_manifest(association, vectors.shape, association["sha256"])
            if p["direction"] != "unspecified":
                index = "source" if p["direction"] == "source_to_target" else "target"
                for key in ("spatial", "pyramid"):
                    context.pop(key, None)
                    if key in self.input_payloads[index]:
                        context[key] = copy.deepcopy(self.input_payloads[index][key])
        manifest = association if p["metadata_mode"] != "none" else None
        if (
            manifest is not None
            and len(json.dumps(manifest, allow_nan=False).encode()) > MAX_METADATA
        ):
            raise ValidationError("Flow manifest exceeds 64 KiB")
        if p["unique_name"]:
            path = path.with_name(f"{path.stem}-{uuid4().hex}{path.suffix}")
        with TemporaryDirectory(prefix="dataflow-flo-") as temporary:
            native = Path(temporary) / "field.flo"
            if not cv2.writeOpticalFlow(str(native), vectors):
                raise OSError("Native flow writer returned failure")
            data, width, height = read_bytes(native)
        if digest(data) != association["sha256"] or (height, width, 2) != vectors.shape:
            raise ValueError("Native .flo writer changed the samples")
        if p["create_directories"]:
            path.parent.mkdir(parents=True, exist_ok=True)
        publication(path, data, manifest)
        context.pop("flow_association", None)
        if manifest is not None:
            context["flow_association"] = copy.deepcopy(manifest)
        finite = np.isfinite(raw)
        rounding = float(
            np.max(np.abs(raw[finite].astype(np.float64) - vectors[finite]), initial=0)
        )
        info = {
            "path": str(path),
            "sha256": digest(data),
            "size": [width, height],
            **counts,
            "manifest": str(path) + ".json" if manifest is not None else None,
            "association": manifest,
            "maximum_rounding_error": rounding,
        }
        context["flow_file"] = copy.deepcopy(info)
        return self.finish(vectors, info, context, path)
