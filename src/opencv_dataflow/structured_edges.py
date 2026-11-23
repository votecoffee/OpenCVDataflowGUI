"""Native structured-forest responses in the source image's coordinate frame."""

import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Port, Result
from .images import format_image, image_format, opencv_image
from .ml_common import file_key, local_path
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, selection_mask

MODEL_SIZE = 6_390_625
MODEL_SHA256 = "fe3586bca5ec124b73fa769c2b54e195e4526d6ceeddf161fad96bf68a65bf17"


class StructuredEdges(ImageTool):
    type_id, title, subtype = "cv.structured_edges", "Structured forest edges", "Edges"
    description = "Measure and thin edge responses using the OpenCV structured-forest model."
    inputs = SELECTION_INPUTS
    outputs = (
        Port("edges", visual_type="mask"),
        Port("response"),
        Port("orientation"),
        Port("thinned"),
        Port("model_info", "data"),
    )
    project_path_parameters = ("model_path",)
    payload_reference = "image"
    preview_safe = parallel_safe = process_inputs_readonly = True
    defaults = {
        "model_path": "",
        "suppression_radius": 2,
        "boundary_radius": 0,
        "suppression_multiplier": 1.0,
        "parallel_nms": True,
        "edge_threshold": 0.15,
    }
    parameter_ranges = {
        "suppression_radius": NumericRange(0, 32, whole=True),
        "boundary_radius": NumericRange(0, 512, whole=True),
        "suppression_multiplier": NumericRange(0.01, 100),
        "edge_threshold": NumericRange(0),
    }
    parameter_help = {
        "model_path": "Local model.yml.gz from OpenCV's 5.0.0 reference data. Its exact "
        "release bytes are verified before native loading.",
        "suppression_radius": "Native non-maximum suppression radius in pixels. Zero skips "
        "the directional comparisons.",
        "boundary_radius": "Native fading distance at the image border. Zero disables fading.",
        "suppression_multiplier": "Multiply the center response before the directional "
        "comparison. Higher values preserve more responses.",
        "parallel_nms": "Use OpenCV's parallel path for non-maximum suppression.",
        "edge_threshold": "Keep positive thinned responses at or above this value in the mask.",
    }
    input_help = {
        "image": "Gray or color in 8-bit, 16-bit, or normalized float32. Color is converted "
        "to float32 RGB and alpha is ignored by inference.",
        "mask": "Zero returned maps outside this mask after full-image inference.",
        "region": "Zero returned maps outside this region and any wired mask.",
    }
    output_help = {
        "edges": "Thresholded uint8 mask of the thinned responses.",
        "response": "Original float32 forest response. Native values can exceed 1.",
        "orientation": "Native undirected edge-normal angles in radians modulo pi.",
        "thinned": "Native non-maximum-suppressed float32 response.",
        "model_info": "Reference model revision and SHA-256.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._detector = None
        self._model_key = None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["model_path"], str):
            raise ValidationError("Use a local structured-edge model path", "model_path")
        if type(p["parallel_nms"]) is not bool:
            raise ValidationError("Parallel NMS must be on or off", "parallel_nms")

    def _load_detector(self):
        create = getattr(getattr(cv2, "ximgproc", None), "createStructuredEdgeDetection", None)
        if create is None:
            raise ValidationError("Structured forest edges requires OpenCV contrib ximgproc")
        configured = self.parameters["model_path"]
        if not configured.strip():
            raise ValidationError("Choose the OpenCV 5.0.0 model.yml.gz file", "model_path")
        path = local_path(self, configured)
        if not path.is_file():
            raise ValidationError(
                "Structured-edge model is missing. Prepare the example model or choose its path.",
                "model_path",
            )
        key = file_key(path)
        if key != self._model_key:
            self.reset_runtime()
            with path.open("rb") as stream:
                snapshot = stream.read(MODEL_SIZE + 1)
            if len(snapshot) != MODEL_SIZE or hashlib.sha256(snapshot).hexdigest() != MODEL_SHA256:
                raise ValidationError(
                    "Model differs from the qualified OpenCV 5.0.0 reference forest", "model_path"
                )
            # OpenCV takes a filename. Pass the verified bytes even if the original changes.
            with tempfile.TemporaryDirectory(prefix="dataflow-structured-edges-") as directory:
                frozen = Path(directory) / "model.yml.gz"
                frozen.write_bytes(snapshot)
                self._detector = create(str(frozen))
            self._model_key = key
        return self._detector

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 16 or max(height, width) > 4096 or height * width > 2_097_152:
            raise ValidationError(
                "Use at least 16 pixels per axis and at most 4096 per axis or 2,097,152 pixels"
            )
        image_value(source, (np.uint8, np.uint16, np.float32))
        encoding = image_format(source)
        if source.dtype == np.float32 and encoding not in ("hsv", "lab"):
            color = np.asarray(source[..., :3] if source.ndim == 3 else source)
            if color.min() < 0 or color.max() > 1:
                raise ValidationError("Float color and intensity values must lie in [0,1]")
        selection = selection_mask(inputs, source)
        rgb = np.asarray(format_image(opencv_image(source), "rgb", "float32"))
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValidationError("Converted RGB values must lie in [0,1]")

        p = self.parameters
        detector = self._load_detector()
        response = detector.detectEdges(rgb)
        orientation = detector.computeOrientation(response)
        if not np.isfinite(response).all() or not np.isfinite(orientation).all():
            raise ValidationError("Structured forest returned a nonfinite map")
        thinned = detector.edgesNms(
            response,
            orientation,
            r=int(p["suppression_radius"]),
            s=int(p["boundary_radius"]),
            m=float(p["suppression_multiplier"]),
            isParallel=p["parallel_nms"],
        )
        if selection is not None:
            outside = selection == 0
            response[outside] = orientation[outside] = thinned[outside] = 0
        edges = (thinned > 0) & (thinned >= float(p["edge_threshold"]))
        return Result(
            {
                "edges": edges.astype(np.uint8) * 255,
                "response": response,
                "orientation": orientation,
                "thinned": thinned,
                "model_info": {"revision": "opencv_extra 5.0.0", "sha256": MODEL_SHA256},
            },
            message=f"{np.count_nonzero(edges)} pixels above the structured-edge threshold",
        )
