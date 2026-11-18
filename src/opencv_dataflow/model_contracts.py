"""Versioned local ONNX bundles and bounded, explicit execution evidence."""

import copy
import hashlib
import json
import platform
import re
from functools import lru_cache
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from .ml_common import local_path
from .parameter_validation import ValidationError

MAX_MODEL_BYTES = 256 * 1024 * 1024
OBSERVATION_FIELDS = ("model_path", "contract_path", "receipt_mode")
CONTRACT_DEFAULTS = {"contract_path": "", "receipt_mode": "summary", "fallback_policy": "error"}
CONTRACT_CHOICES = {
    "receipt_mode": ("summary", "detailed", "off"),
    "fallback_policy": ("error", "cpu_if_unavailable"),
}
CONTRACT_HELP = {
    "contract_path": "Project-relative contract.json; uses its bundled, verified ONNX model.",
    "receipt_mode": "Summary: identity/settings/shapes. Detailed: tensor hashes. Off: no receipt.",
    "fallback_policy": "Error by default; CPU fallback only for an unavailable requested target.",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def revision(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def configuration(block):
    return copy.deepcopy({k: v for k, v in block.parameters.items() if k not in OBSERVATION_FIELDS})


def relative_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or Path(value).is_absolute()
        or PureWindowsPath(value).anchor
        or any(p in ("", ".", "..") for p in value.split("/"))
        or ":" in value
    ):
        raise ValidationError(
            "Contract paths must stay within the project, using /", "contract_path"
        )
    return value


def read_contract(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("Duplicate field in ONNX contract", "contract_path")
            result[key] = value
        return result

    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValidationError("ONNX contract exceeds 128 KiB", "contract_path")
    data = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    fields = {
        "model",
        "schema_version",
        "block_type",
        "configuration",
        "preprocessing",
        "artifact",
        "revision",
    }
    if (
        not isinstance(data, dict)
        or data.keys() != fields
        or data["model"] != "onnx_inference_contract_v1"
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["block_type"], str)
        or not isinstance(data["configuration"], dict)
        or not isinstance(data["preprocessing"], dict)
        or data["revision"] != revision({k: v for k, v in data.items() if k != "revision"})
    ):
        raise ValidationError("Invalid ONNX contract schema or revision", "contract_path")
    artifact = data["artifact"]
    if (
        not isinstance(artifact, dict)
        or artifact.keys() != {"file", "sha256", "bytes"}
        or artifact["file"] != "model.onnx"
        or type(artifact["bytes"]) is not int
        or not 1 <= artifact["bytes"] <= MAX_MODEL_BYTES
        or not isinstance(artifact["sha256"], str)
        or not re.fullmatch("[0-9a-f]{64}", artifact["sha256"])
    ):
        raise ValidationError("Invalid bundled model identity", "contract_path")
    return data


def resolve_contract(block, *, check_configuration=True):
    value = block.parameters.get("contract_path", "")
    if not value:
        return local_path(block, block.parameters["model_path"]), None
    relative_path(value)
    if block.project_dir is None:
        raise ValidationError("A bound ONNX contract needs a project directory", "contract_path")
    root = Path(block.project_dir).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not (path.parent / "model.onnx").resolve().is_relative_to(
        root
    ):
        raise ValidationError("ONNX bundle resolves outside its project", "contract_path")
    contract = read_contract(path)
    if contract["block_type"] != block.type_id or (
        check_configuration
        and canonical(contract["configuration"]) != canonical(configuration(block))
    ):
        raise ValidationError(
            "Block settings differ from the pinned ONNX contract", "contract_path"
        )
    return path.parent / "model.onnx", contract


def model_bytes(path, *, bounded=False):
    with path.open("rb") as stream:
        data = stream.read(MAX_MODEL_BYTES + 1) if bounded else stream.read()
    if not data or bounded and len(data) > MAX_MODEL_BYTES:
        raise ValidationError("Bundled ONNX model must contain 1..256 MiB", "contract_path")
    return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def export_contract(block, directory, *, preprocessing=None):
    """Publish a new self-contained folder without running processing or overwriting files."""
    if block.type_id not in (
        "ml.dnn",
        "ml.dnn_multi",
        "ml.dnn_batch",
        "ml.efficientsam",
        "ml.recognize_text_batch",
        "ml.recognize_constrained_text",
        "ml.restore_image",
        "ml.monocular_depth",
        "ml.stereo_disparity",
    ):
        raise ValidationError("Choose an ONNX inference block")
    block.validate_parameters()
    source, bound = resolve_contract(block, check_configuration=False)
    raw, identity = model_bytes(source, bounded=True)
    if bound is not None and identity != {k: bound["artifact"][k] for k in ("sha256", "bytes")}:
        raise ValidationError("Bundled ONNX model checksum differs", "contract_path")
    document = {
        "model": "onnx_inference_contract_v1",
        "schema_version": 1,
        "block_type": block.type_id,
        "configuration": configuration(block),
        "preprocessing": copy.deepcopy(
            preprocessing if preprocessing is not None else bound["preprocessing"] if bound else {}
        ),
        "artifact": {"file": "model.onnx", **identity},
    }
    if not isinstance(document["preprocessing"], dict):
        raise ValidationError("Expected preprocessing must be a mapping")
    canonical(document)
    document["revision"] = revision(document)
    # A fresh directory makes both files reviewable and preserves existing bundles.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "model.onnx").open("xb") as stream:
        stream.write(raw)
    with (directory / "contract.json").open("xb") as stream:
        stream.write((json.dumps(document, indent=2, allow_nan=False) + "\n").encode())
    return document


def image_preprocessing(parameters):
    keys = (
        "width",
        "height",
        "color_format",
        "resize_mode",
        "padding_value",
        "scale",
        "mean",
        "std",
    )
    settings = {k: copy.deepcopy(parameters[k]) for k in keys}
    settings["layout"] = parameters.get("layout", "nchw")
    result = {"model": "dnn_image_preprocessing_v1", "schema_version": 1, "settings": settings}
    result["revision"] = revision(result)
    return result


@lru_cache(maxsize=1)
def environment():
    return {
        "opencv": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def tensor_description(name, value, detailed):
    result = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
    if detailed:
        result["dtype_code"] = value.dtype.str
        result["sha256"] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        result["hash_order"] = "C-contiguous native dtype bytes"
    return result
