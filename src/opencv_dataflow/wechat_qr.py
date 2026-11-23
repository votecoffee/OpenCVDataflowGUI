"""Decode QR codes through the OpenCV 5 WeChat contrib binding."""

import hashlib
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .code_readers import CodeReader, barcode_image, code_record
from .core import Port
from .ml_common import local_path
from .native_runtime import cpu_execution
from .parameter_validation import NumericRange, ValidationError

_MODEL_ROLES = ("detector", "super_resolution")
_MAX_MODEL_BYTES = 16 * 1024**2


class ReadWeChatQR(CodeReader):
    type_id, title = "inspect.read_wechat_qr", "Read WeChat QR codes"
    description = "Decode QR codes with optional ONNX detection and super-resolution models."
    preview_safe = parallel_safe = process_inputs_readonly = True
    outputs = (*CodeReader.outputs, Port("info", "data", required=False))
    project_path_parameters = tuple(f"{role}_model_path" for role in _MODEL_ROLES)
    defaults = {
        **CodeReader.defaults,
        "detector_model_path": "",
        "detector_sha256": "",
        "super_resolution_model_path": "",
        "super_resolution_sha256": "",
        "scale_factor": 0.0,
        "maximum_pixels": 4_194_304,
    }
    parameter_ranges = {
        **CodeReader.parameter_ranges,
        "scale_factor": NumericRange(0.01, 1, zero_allowed=True),
        "maximum_pixels": NumericRange(441, 16_777_216, whole=True),
    }
    parameter_help = {
        "detector_model_path": "Local or project-relative ONNX detector. Empty searches the image.",
        "detector_sha256": "Required SHA-256 pin when the detector model is selected.",
        "super_resolution_model_path": "Local or project-relative ONNX upscaler. Empty uses cubic.",
        "super_resolution_sha256": "Required SHA-256 pin when the upscaler is selected.",
        "scale_factor": "CNN detector input scale. Zero selects OpenCV's automatic scale.",
        "maximum_pixels": "Reject larger images before decoding.",
    }
    output_help = {
        "codes": "Decoded text, local image corners and center for each native result.",
        "info": "Native decoder, ONNX checksums and detection scale used for this invocation.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for role in _MODEL_ROLES:
            path_key, hash_key = f"{role}_model_path", f"{role}_sha256"
            path, digest = p[path_key], p[hash_key]
            if not isinstance(path, str) or len(path) > 4096:
                raise ValidationError("Choose a local ONNX path", path_key)
            if not isinstance(digest, str) or (
                path and not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            ):
                raise ValidationError("Supply the model's 64-character SHA-256", hash_key)
            if digest and not path:
                raise ValidationError("Choose a model file for this SHA-256", path_key, hash_key)
            if path and Path(path).suffix.lower() != ".onnx":
                raise ValidationError("OpenCV 5 WeChat QR requires an ONNX model", path_key)
        if float(p["scale_factor"]) and not p["detector_model_path"]:
            raise ValidationError("A custom scale requires a CNN detector", "scale_factor")

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 8192 or height * width > int(p["maximum_pixels"]):
            raise ValidationError("Image exceeds the WeChat QR size limit", "maximum_pixels")
        scale = float(p["scale_factor"])
        if p["detector_model_path"] and min(height, width) > 20:
            native_scale = scale or min(1, math.sqrt(160_000 / (height * width)))
            if min(height, width) * native_scale < 32:
                raise ValidationError(
                    "The CNN detector input needs 32 pixels per axis", "scale_factor"
                )
        image, _ = barcode_image(inputs)
        models, codes = {}, []
        if image is not None and min(height, width) > 20:
            native = getattr(getattr(cv2, "wechat_qrcode", None), "WeChatQRCode", None)
            if native is None:
                raise ValidationError("WeChat QR requires the OpenCV contrib wechat_qrcode module")
            with cpu_execution():
                if any(p[field] for field in self.project_path_parameters):
                    # Load the hashed bytes, even if an editor replaces the original model file.
                    with TemporaryDirectory(prefix="opencv-dataflow-qr-") as work:
                        paths, models = self._snapshot_models(Path(work))
                        detector = native(*paths)
                else:
                    detector = native()
                detector.setScaleFactor(scale if scale else -1.0)
                texts, points = detector.detectAndDecode(image)
            for text, corners in zip(texts, () if points is None else points, strict=True):
                vertices = np.asarray(corners, dtype=float)
                if vertices.size != 8 or not np.isfinite(vertices).all():
                    raise ValueError("WeChat QR returned invalid quadrilateral geometry")
                if text:
                    code = code_record(text, "QR_CODE", vertices.reshape(4, 2))
                    code.update(backend="wechat", corner_model="native_detection_quadrilateral")
                    codes.append(code)
        info = {
            "backend": "wechat",
            "models_sha256": models,
            "scale_factor": scale if scale else None,
            "decoded_count": len(codes),
        }
        result = self.code_result(source, codes, info=info)
        result.event_data["qr_backend"] = "wechat"
        return result

    def _snapshot_models(self, directory):
        paths, digests = [], {}
        for role in _MODEL_ROLES:
            path_key, hash_key = f"{role}_model_path", f"{role}_sha256"
            configured = self.parameters[path_key]
            if not configured:
                paths.append("")
                continue
            source = local_path(self, configured)
            if not source.is_file():
                raise ValidationError(
                    f"Choose an existing {role.replace('_', ' ')} model", path_key
                )
            with source.open("rb") as stream:
                data = stream.read(_MAX_MODEL_BYTES + 1)
            if not 0 < len(data) <= _MAX_MODEL_BYTES:
                raise ValidationError("The ONNX model must contain 1 byte to 16 MiB", path_key)
            digest = hashlib.sha256(data).hexdigest()
            if digest != self.parameters[hash_key].lower():
                raise ValidationError("The ONNX model SHA-256 differs", hash_key, path_key)
            destination = directory / f"{role}.onnx"
            destination.write_bytes(data)
            paths.append(str(destination))
            digests[role] = digest
        return paths, digests
