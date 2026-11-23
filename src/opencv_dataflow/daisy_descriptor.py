"""Sample bounded native DAISY descriptors at a grid or wired keypoints."""

import math
from threading import Lock

import cv2
import numpy as np

from .compact_descriptors import KeypointDescriptor
from .core import Port
from .parameter_validation import NumericRange, ValidationError
from .registration import matrix_value
from .selection import contains

_NATIVE_LOCK = Lock()
_MAX_SAMPLES = 4096
_MAX_DESCRIPTOR_VALUES = 1_048_576
_MAX_HISTOGRAM_VALUES = 128 * 1024**2


def _sampling_matrix(parameters):
    value = parameters["sampling_homography"]
    if value == []:
        return None
    if parameters["interpolation"]:
        raise ValidationError(
            "A DAISY sampling homography requires nearest sampling on this OpenCV build",
            "sampling_homography",
            "interpolation",
        )
    try:
        matrix = matrix_value(value, (3, 3))
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValidationError(str(exc), "sampling_homography") from exc
    return matrix / np.abs(matrix).max()


def _check_projection(matrix, shape, radius):
    height, width = shape
    # Native nearest sampling also transforms an offset histogram support position.
    margin = 2 * radius
    corners = np.array(
        [
            [-margin, -margin, 1],
            [width + margin, -margin, 1],
            [-margin, height + margin, 1],
            [width + margin, height + margin, 1],
        ]
    )
    projected = corners @ matrix.T
    denominator = projected[:, 2]
    if not (np.min(denominator) > 1e-9 or np.max(denominator) < -1e-9) or np.any(
        np.abs(projected[:, :2]) > 1_000_000 * np.abs(denominator[:, None])
    ):
        raise ValidationError(
            "DAISY homography must have a finite bounded projection over the image and its support",
            "sampling_homography",
        )


class DaisyDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.daisy_descriptor", "DAISY descriptors", "DAISY"
    description = "Measure local gradient histograms on a regular grid or at wired keypoints."
    maximum_pixels = 16_777_216
    inputs = (
        Port("image"),
        Port("keypoints", "data", required=False, required_when_connected=True),
        *KeypointDescriptor.inputs[2:],
    )
    defaults = {
        "radius": 15.0,
        "radial_divisions": 3,
        "angular_divisions": 8,
        "histogram_bins": 8,
        "normalization": "none",
        "interpolation": True,
        "use_orientation": False,
        "sampling_homography": [],
        "grid_step": 24,
        "grid_origin": [0, 0],
        "grid_angle": 0.0,
    }
    parameter_choices = {"normalization": ("none", "partial", "full", "sift")}
    parameter_ranges = {
        "radius": NumericRange(1, 128),
        "radial_divisions": NumericRange(1, 8, whole=True),
        "angular_divisions": NumericRange(1, 32, whole=True),
        "histogram_bins": NumericRange(2, 32, whole=True),
        "grid_step": NumericRange(1, 32768, whole=True),
        "grid_angle": NumericRange(0, 360),
    }
    parameter_help = {
        "radius": "Outer sampling radius in pixels.",
        "radial_divisions": "Number of concentric rings around the center.",
        "angular_divisions": "Histogram locations per ring.",
        "histogram_bins": "Gradient directions per local histogram.",
        "normalization": "Choose raw, per-histogram L2, whole-row L2, or SIFT-style clipping.",
        "interpolation": "Interpolate samples and gradient directions. Turn off for a homography.",
        "use_orientation": "Use wired keypoint angles or the fixed Grid angle.",
        "sampling_homography": "Optional 3 × 3 sampling transform. Keypoint coordinates stay in "
        "the source grid. Available with nearest sampling.",
        "grid_step": "Spacing in pixels when Keypoints is disconnected.",
        "grid_origin": "First [x, y] point of the lattice before border and selection filtering.",
        "grid_angle": "Fixed angle in degrees for grid points when Use orientation is enabled.",
    }
    input_help = {
        **KeypointDescriptor.input_help,
        "keypoints": "Optional feature record, keypoint records, or [x, y] rows. "
        "Disconnect to sample a grid.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        origin = p["grid_origin"]
        if (
            not isinstance(origin, list)
            or len(origin) != 2
            or any(type(value) is not int or not 0 <= value <= 32768 for value in origin)
        ):
            raise ValidationError("Grid origin must be two whole pixels in 0..32768", "grid_origin")
        width = (int(p["radial_divisions"]) * int(p["angular_divisions"]) + 1) * int(
            p["histogram_bins"]
        )
        if width > 512:
            raise ValidationError(
                "Choose radial, angular and histogram divisions for at most 512 values per sample",
                "radial_divisions",
                "angular_divisions",
                "histogram_bins",
            )
        _sampling_matrix(p)

    def sample_keypoints(self, inputs, size, selection):
        if "keypoints" in inputs:
            return super().sample_keypoints(inputs, size, selection)
        p = self.parameters
        radius = float(p["radius"])
        margin = math.ceil(radius) + 2
        step = int(p["grid_step"])
        x, y = (
            origin + max(0, math.ceil((margin - origin) / step)) * step
            for origin in p["grid_origin"]
        )
        xs, ys = range(x, size[0] - margin, step), range(y, size[1] - margin, step)
        if len(xs) * len(ys) > _MAX_SAMPLES:
            raise ValidationError(
                "DAISY grid exceeds 4,096 points. Increase Grid step.", "grid_step"
            )
        angle = float(p["grid_angle"]) % 360 if p["use_orientation"] else 0.0
        return [
            cv2.KeyPoint(x, y, 2 * radius, angle, 0, 0, i * len(xs) + j)
            for i, y in enumerate(ys)
            for j, x in enumerate(xs)
            if contains(selection, (x, y))
        ]

    def extractor(self):
        p = self.parameters
        matrix = _sampling_matrix(p)
        settings = {
            "radius": float(np.float32(p["radius"])),
            "radial_divisions": int(p["radial_divisions"]),
            "angular_divisions": int(p["angular_divisions"]),
            "histogram_bins": int(p["histogram_bins"]),
            "normalization": p["normalization"],
            "interpolation": p["interpolation"],
            "use_orientation": p["use_orientation"],
            "sampling_homography": [] if matrix is None else matrix.tolist(),
        }
        factory = self.factory("DAISY_create")
        norm = getattr(cv2.xfeatures2d, "DAISY_NRM_" + settings["normalization"].upper())
        return factory(
            settings["radius"],
            settings["radial_divisions"],
            settings["angular_divisions"],
            settings["histogram_bins"],
            norm,
            matrix,
            settings["interpolation"],
            settings["use_orientation"],
        ), settings

    def compute_descriptors(self, gray, keypoints):
        p = self.parameters
        radial = int(p["radial_divisions"])
        histogram = int(p["histogram_bins"])
        width = (radial * int(p["angular_divisions"]) + 1) * histogram
        if len(keypoints) > _MAX_SAMPLES or len(keypoints) * width > _MAX_DESCRIPTOR_VALUES:
            raise ValidationError("Reduce the number of DAISY samples or histogram divisions")
        if keypoints and gray.size * (radial + 3) * histogram > _MAX_HISTOGRAM_VALUES:
            raise ValidationError(
                "DAISY histogram workspace exceeds the limit. Crop or reduce divisions."
            )
        matrix = _sampling_matrix(p)
        if matrix is not None:
            _check_projection(matrix, gray.shape, float(p["radius"]))
        # OpenCV shares selected histogram layers between DAISY instances.
        with _NATIVE_LOCK:
            keys, descriptors, width, settings = super().compute_descriptors(gray, keypoints)
        if descriptors is not None and not np.isfinite(descriptors).all():
            raise ValidationError("DAISY produced nonfinite descriptor values")
        return keys, descriptors, width, settings
