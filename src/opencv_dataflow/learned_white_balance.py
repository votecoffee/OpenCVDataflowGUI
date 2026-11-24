"""OpenCV learned white balance with checked model and palette support."""

import hashlib
import re
import tempfile

import cv2
import numpy as np

from .core import Port, Result
from .images import ImageArray, image_format, opencv_image
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, metadata
from .processing import ImageTool, image_value
from .selection import selection_mask

_PALETTE_SIZE = 300
_MODEL_LIMIT = 8 * 1024 * 1024
_MODEL_REVISION = "755e50675d97db9b7d449d8bd6b09888646f6c6e"
_MODEL_HEADER_SHA256 = "c18df1a6529202020acc28f7cde5d63d582d7f172aa1e3433190d2763b4186b0"
_FEATURES = ("average", "brightest", "dominant", "palette_mode")
_ENCODINGS = {"bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}


def _model_matrix(storage, name, count, cv_type, dtype):
    node = storage.getNode(name)
    if node.empty() or not node.isMap():
        raise ValidationError(f"White balance model lacks matrix {name}", "model_path")
    rows = float(node.getNode("rows").real())
    cols = float(node.getNode("cols").real())
    data = node.getNode("data")
    if (
        not rows.is_integer() or not cols.is_integer() or rows <= 0 or cols <= 0
        or rows * cols != count
        or node.getNode("dt").string() != cv_type or not data.isSeq() or data.size() != count
    ):
        raise ValidationError(f"Invalid white balance model matrix {name}", "model_path")
    values = node.mat()
    if (
        values is None or values.size != count or values.dtype != dtype
        or not np.isfinite(values).all()
    ):
        raise ValidationError(f"Invalid white balance model samples in {name}", "model_path")
    return values


def _check_model(raw):
    try:
        model = cv2.FileStorage(raw.decode("utf-8-sig"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            trees = model.getNode("num_trees").real()
            nodes = model.getNode("num_tree_nodes").real()
            if (
                not model.isOpened() or not 1 <= trees <= 256 or not float(trees).is_integer()
                or nodes not in (2, 4, 8, 16, 32, 64, 128, 256)
            ):
                raise ValidationError("Invalid white balance model tree dimensions", "model_path")
            forest_size = 2 * len(_FEATURES) * int(trees)
            splits = forest_size * (int(nodes) - 1)
            selectors = _model_matrix(model, "feature_idx", splits, "u", np.uint8)
            _model_matrix(model, "thresh_vals", splits, "f", np.float32)
            leaves = _model_matrix(model, "leaf_vals", forest_size * int(nodes), "f", np.float32)
            if selectors.max() > 1 or leaves.min() <= 0 or leaves.max() >= 1:
                raise ValidationError(
                    "Invalid white balance model indices or chromaticities", "model_path"
                )
        finally:
            model.release()
    except (cv2.error, SystemError, UnicodeError) as exc:
        raise ValidationError(
            "Read a valid OpenCV learned white balance YAML model", "model_path"
        ) from exc


def _palette_support(bgr, bins, cutoff, image_max):
    # Match xphoto's exclusive high histogram bound and strict saturation test.
    brightest = bgr.max(axis=2)
    eligible = np.uint8(brightest < cutoff) * 255
    upper = max(bins, image_max)
    histogram = cv2.calcHist([bgr], [0, 1, 2], eligible, [bins] * 3, [0, upper] * 3)
    return cv2.countNonZero(eligible), int(np.count_nonzero(histogram))


class LearnedWhiteBalance(ImageTool):
    type_id, title = "cv.learned_white_balance", "Learned white balance"
    category, block_type, subtype = "Calibration", "Calibration", "Color balance"
    description = "Use OpenCV's learned illuminant estimator and apply its channel gains."
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("report", "data"),
        Port("features", "data", required=False),
    )
    defaults = {
        "saturation_threshold": 0.98, "range_max": 0, "histogram_bins": 64,
        "model_path": "", "model_sha256": "",
    }
    parameter_ranges = {
        "saturation_threshold": NumericRange(0, 1),
        "range_max": NumericRange(0, 65535, whole=True),
        "histogram_bins": NumericRange(7, 256, whole=True),
    }
    parameter_help = {
        "saturation_threshold": "Reject pixels at or above this fraction of the sensor range.",
        "range_max": "Largest sensor code. Zero uses 255 or 65535 based on image depth.",
        "histogram_bins": "Number of bins per axis of the three-dimensional color histogram.",
        "model_path": "Optional local OpenCV trained YAML model. Empty uses the embedded forest.",
        "model_sha256": "Optional SHA-256 pin for the external model. The actual digest is reported.",
    }
    input_help = {
        "image": "8- or 16-bit BGR/RGB. Straight alpha is preserved.",
        "mask": "Restrict corrected pixels. The illuminant is estimated from the whole image.",
        "region": "Restrict corrected pixels, intersected with the mask when connected.",
    }
    output_help = {
        "report": "Usable sample count, populated histogram bins, settings, and model provenance.",
        "features": "Optional native average, brightest, dominant and palette-mode [r,g] features.",
    }
    project_path_parameters = ("model_path",)
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [name for name in self.exposed_outputs if name != "features"]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["model_path"], str):
            raise ValidationError("Choose a local learned white balance model path", "model_path")
        digest = p["model_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{64})?", digest):
            raise ValidationError("Use empty text or a 64-hex SHA-256", "model_sha256")
        if digest and not p["model_path"].strip():
            raise ValidationError("Choose the model file to match the digest", "model_path")

    def _load_model(self):
        create = getattr(getattr(cv2, "xphoto", None), "createLearningBasedWB", None)
        if create is None:
            raise ValidationError("Learned white balance requires OpenCV contrib xphoto")
        p = self.parameters
        provenance = {"source": "embedded_opencv", "opencv_version": cv2.__version__}
        if not p["model_path"].strip():
            if cv2.__version__ == "5.0.0":
                provenance.update({
                    "reference_opencv_contrib_commit": _MODEL_REVISION,
                    "reference_header_sha256": _MODEL_HEADER_SHA256,
                })
            return create(), provenance

        path = local_path(self, p["model_path"])
        if path.suffix.lower() not in (".yml", ".yaml") or not path.is_file():
            raise ValidationError("Choose an existing OpenCV learned white balance YAML model", "model_path")
        with path.open("rb") as stream:
            raw = stream.read(_MODEL_LIMIT + 1)
        if not raw or len(raw) > _MODEL_LIMIT:
            raise ValidationError("Model must contain up to 8 MiB of YAML", "model_path")
        digest = hashlib.sha256(raw).hexdigest()
        if p["model_sha256"] and p["model_sha256"].lower() != digest:
            raise ValidationError("White balance model SHA-256 differs", "model_sha256", "model_path")
        _check_model(raw)
        # The native factory accepts a filename and may fall back when it cannot open it.
        with tempfile.NamedTemporaryFile(suffix=".yml", delete_on_close=False) as frozen:
            frozen.write(raw)
            frozen.close()
            balance = create(frozen.name)
        provenance.update({"source": "external_yaml", "sha256": digest})
        return balance, provenance

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16))
        encoding = image_format(source)
        if not image.size or image.ndim != 3 or image.shape[2] != _ENCODINGS.get(encoding):
            raise ValidationError("Use a nonempty uint8 or uint16 BGR/RGB image")
        if image.shape[2] == 4 and metadata(source).get("alpha_representation", "straight") != "straight":
            raise ValidationError("Unassociate premultiplied alpha before white balance")
        if image.shape[0] * image.shape[1] > 16_000_000:
            raise ValidationError("Learned white balance is limited to 16 million pixels")
        selected = selection_mask(inputs, source)
        bgr = np.ascontiguousarray(opencv_image(source)[..., :3])
        p = self.parameters
        depth_max = np.iinfo(image.dtype).max
        range_max = int(p["range_max"]) or depth_max
        image_max = int(bgr.max())
        if not image_max <= range_max <= depth_max:
            raise ValidationError("Sensor range must cover the color samples and fit the image depth", "range_max")
        threshold = float(p["saturation_threshold"])
        cutoff = int(np.float32(threshold) * np.float32(range_max))
        bins = int(p["histogram_bins"])
        eligible, populated = _palette_support(bgr, bins, cutoff, image_max)
        report = {
            "range_max": range_max, "saturation_threshold": threshold, "histogram_bins": bins,
            "saturation_cutoff": cutoff, "eligible_pixels": eligible, "populated_bins": populated,
        }
        if populated < _PALETTE_SIZE:
            message = f"Learned white balance needs 300 populated color bins, found {populated}"
            return Result(
                {"image": NO_VALUE, "report": report, "features": NO_VALUE},
                passed=False, message=message,
                fail_reasons=[{"code": "learned_white_balance_palette_too_small", "message": message}],
            )
        balancer, report["model"] = self._load_model()
        balancer.setRangeMaxVal(range_max)
        balancer.setSaturationThreshold(threshold)
        balancer.setHistBinNum(bins)
        features = NO_VALUE
        if "features" in self.exposed_outputs:
            values = balancer.extractSimpleFeatures(bgr)
            if values is None or values.size != 8 or not np.isfinite(values).all():
                raise ValidationError("Native white balance returned invalid chromaticity features")
            features = dict(zip(_FEATURES, values.reshape(4, 2).tolist(), strict=True))
        result = balancer.balanceWhite(bgr)
        if result is None or result.shape != bgr.shape or result.dtype != bgr.dtype or np.any(result > bgr):
            raise ValidationError("Native white balance returned invalid channel gains")
        if encoding in ("rgb", "rgba"):
            result = result[..., ::-1]
        if selected is not None or image.shape[2] == 4:
            output = image.copy()
            color = output[..., :3]
            if selected is None:
                color[:] = result
            else:
                write_pixels = selected != 0
                color[write_pixels] = result[write_pixels]
            result = output
        return Result({
            "image": ImageArray(result, encoding), "report": report, "features": features,
        })
