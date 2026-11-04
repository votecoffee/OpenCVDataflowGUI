"""Further wrappers with explicit algorithm controls and typed enum metadata."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameters import BORDERS, OUTPUT_DEPTHS, anchor_value
from .processing import ImageTool, encoded_result, finite, gray_image, image_value, integer
from .registration import BORDERS as WARP_BORDERS
from .registration import INTERPOLATION
from .selection import RegionalOperation


def kernel_vector(value, name):
    kernel = np.asarray(value, np.float64)
    if kernel.ndim != 1 or not 1 <= kernel.size <= 63 or not np.isfinite(kernel).all():
        raise ValueError(f"{name} must be a finite vector of 1..63 coefficients")
    return kernel


def criteria(parameters):
    mode = {
        "count": cv2.TERM_CRITERIA_COUNT,
        "epsilon": cv2.TERM_CRITERIA_EPS,
        "both": cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
    }[parameters["termination"]]
    return (
        mode,
        integer(parameters["max_iterations"], "Maximum iterations", 1, 10000),
        finite(parameters["epsilon"], "Epsilon", 0),
    )


class BoxFilter(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.box_filter", "Box filter (depth/normalization)", "Filtering"
    defaults = {
        "width": 3,
        "height": 3,
        "normalize": True,
        "anchor": [-1, -1],
        "output_depth": "same",
        "border": "reflect",
    }
    parameter_choices = {"output_depth": tuple(OUTPUT_DEPTHS), "border": tuple(BORDERS)}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        width, height = (integer(self.parameters[k], k, 1, 255) for k in ("width", "height"))
        output = cv2.boxFilter(
            image,
            OUTPUT_DEPTHS[self.parameters["output_depth"]],
            (width, height),
            anchor=anchor_value(self.parameters["anchor"], width, height),
            normalize=self.parameters["normalize"],
            borderType=BORDERS[self.parameters["border"]],
        )
        return encoded_result(output, source)


class SeparableFilter(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.separable_filter", "Separable 2D filter", "Filtering"
    defaults = {
        "kernel_x": [0.25, 0.5, 0.25],
        "kernel_y": [0.25, 0.5, 0.25],
        "anchor": [-1, -1],
        "delta": 0.0,
        "output_depth": "same",
        "border": "reflect",
    }
    parameter_choices = {"output_depth": tuple(OUTPUT_DEPTHS), "border": tuple(BORDERS)}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        x, y = (kernel_vector(self.parameters[k], k) for k in ("kernel_x", "kernel_y"))
        output = cv2.sepFilter2D(
            image,
            OUTPUT_DEPTHS[self.parameters["output_depth"]],
            x,
            y,
            anchor=anchor_value(self.parameters["anchor"], len(x), len(y)),
            delta=finite(self.parameters["delta"], "Delta"),
            borderType=BORDERS[self.parameters["border"]],
        )
        if not np.isfinite(output).all():
            raise ValueError("Separable filter overflowed its output depth")
        return encoded_result(output, source)


class CornerResponse(RegionalOperation, ImageTool):
    region_outputs = ("response", "preview")
    region_background = "zero"
    type_id, title, subtype = (
        "inspect.corner_response",
        "Harris / minimum eigenvalue response",
        "Features",
    )
    category, block_type = "Inspection", "Inspection"
    outputs = (Port("response"), Port("preview"))
    defaults = {
        "method": "harris",
        "block_size": 3,
        "aperture": "3",
        "harris_k": 0.04,
        "border": "reflect",
    }
    parameter_choices = {
        "method": ("harris", "min_eigen"),
        "aperture": ("3", "5", "7"),
        "border": tuple(BORDERS),
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        p = self.parameters
        block = integer(p["block_size"], "Block size", 2, 255)
        kernel = int(p["aperture"])
        border = BORDERS[p["border"]]
        response = (
            cv2.cornerHarris(
                image, block, kernel, finite(p["harris_k"], "Harris K", 0), borderType=border
            )
            if p["method"] == "harris"
            else cv2.cornerMinEigenVal(image, block, ksize=kernel, borderType=border)
        )
        preview = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Result(
            {"response": response, "preview": preview}, graphics={"corner response": preview}
        )


class RefineCorners(Block):
    type_id, title, category = "inspect.refine_corners", "Refine corners to subpixel", "Inspection"
    block_type, subtype = "Inspection", "Features"
    inputs = (Port("image"), Port("points", "data"))
    outputs = (Port("points", "data"),)
    defaults = {
        "window_x": 5,
        "window_y": 5,
        "zero_zone": [-1, -1],
        "termination": "both",
        "max_iterations": 30,
        "epsilon": 0.01,
    }
    parameter_choices = {"termination": ("count", "epsilon", "both")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        points = np.asarray(inputs["points"], np.float32)
        if not points.size:
            return Result({"points": []})
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points[:, 0] >= image.shape[1])
            or np.any(points[:, 1] >= image.shape[0])
        ):
            raise ValueError("Corner points must be finite and inside the image")
        window = tuple(integer(self.parameters[k], k, 1, 127) for k in ("window_x", "window_y"))
        zero = anchor_value(self.parameters["zero_zone"], *window)
        if (zero[0] == -1) != (zero[1] == -1):
            raise ValueError("Zero zone must be [-1,-1] or two nonnegative dimensions")
        refined = cv2.cornerSubPix(
            image, points.reshape(-1, 1, 2).copy(), window, zero, criteria(self.parameters)
        )
        return Result({"points": refined.reshape(-1, 2).tolist()})


class PhaseCorrelation(Block):
    type_id, title, category = "inspect.phase_correlation", "Phase correlation shift", "Inspection"
    block_type, subtype = "Inspection", "Registration"
    inputs = (Port("image_a"), Port("image_b"), Port("window", required=False))
    outputs = (Port("shift", "data"), Port("response", "number"))
    defaults = {"precision": "float32", "hanning_window": False, "min_response": 0.0}
    parameter_choices = {"precision": ("float32", "float64")}

    def process(self, inputs):
        a, b = (
            gray_image(inputs[k], (np.uint8, np.uint16, np.float32, np.float64))
            for k in ("image_a", "image_b")
        )
        if a.shape != b.shape:
            raise ValueError("Phase correlation images must have identical dimensions")
        dtype = np.float32 if self.parameters["precision"] == "float32" else np.float64
        a, b = a.astype(dtype), b.astype(dtype)
        window = inputs.get("window")
        if window is not None:
            if window.shape != a.shape or window.dtype != dtype or not np.isfinite(window).all():
                raise ValueError(
                    "Window must match the image shape and selected floating precision"
                )
            window = np.asarray(window)
        elif self.parameters["hanning_window"]:
            if min(a.shape) < 2:
                raise ValueError("Hanning window requires at least two rows and columns")
            window = cv2.createHanningWindow(
                (a.shape[1], a.shape[0]), cv2.CV_32F if dtype == np.float32 else cv2.CV_64F
            )
        shift, response = cv2.phaseCorrelate(a, b, window)
        if not np.isfinite([*shift, response]).all():
            raise ValueError("Phase correlation produced a nonfinite result")
        minimum = finite(self.parameters["min_response"], "Minimum response", 0)
        if minimum > 1:
            raise ValueError("Minimum response must be at most one")
        return Result(
            {
                "shift": {
                    "x": shift[0],
                    "y": shift[1],
                    "direction": "image_a to image_b",
                    "units": "pixels",
                },
                "response": response,
            },
            passed=response >= minimum,
            score=float(np.clip(response, 0, 1)),
        )


class EstimateAffine(Block):
    type_id, title, category = "inspect.affine", "Estimate affine transform", "Inspection"
    block_type, subtype = "Inspection", "Registration"
    inputs = (Port("matches", "data"),)
    outputs = (Port("transform", "data"), Port("inliers", "number"))
    defaults = {
        "model": "full",
        "method": "ransac",
        "reprojection_pixels": 3.0,
        "max_iterations": 2000,
        "confidence": 0.99,
        "refine_iterations": 10,
        "min_inliers": 3,
        "min_inlier_ratio": 0.5,
    }
    parameter_choices = {"model": ("full", "partial"), "method": ("ransac", "lmeds")}

    def process(self, inputs):
        p = self.parameters
        data = inputs["matches"]
        a, b = (np.asarray(data[k], np.float64) for k in ("query_points", "train_points"))
        count = 3 if p["model"] == "full" else 2
        if (
            a.ndim != 2
            or a.shape[1] != 2
            or len(a) < count
            or b.shape != a.shape
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError(
                f"Affine {p['model']} model needs at least {count} finite paired points"
            )
        confidence = finite(p["confidence"], "Confidence", 0)
        threshold = finite(p["reprojection_pixels"], "Reprojection threshold", 0)
        ratio = finite(p["min_inlier_ratio"], "Minimum inlier ratio", 0)
        if not 0 < confidence < 1 or threshold <= 0 or ratio > 1:
            raise ValueError("Invalid confidence, reprojection threshold or inlier ratio")
        function = cv2.estimateAffine2D if p["model"] == "full" else cv2.estimateAffinePartial2D
        matrix, mask = function(
            a,
            b,
            method=cv2.RANSAC if p["method"] == "ransac" else cv2.LMEDS,
            ransacReprojThreshold=threshold,
            maxIters=integer(p["max_iterations"], "Iterations", 1, 100000),
            confidence=confidence,
            refineIters=integer(p["refine_iterations"], "Refine iterations", 0, 10000),
        )
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            raise ValueError("Cannot estimate affine transform")
        homogeneous = np.vstack((matrix, [0, 0, 1]))
        if np.linalg.matrix_rank(homogeneous) != 3:
            raise ValueError("Affine estimate is degenerate")
        valid = mask.reshape(-1).astype(bool)
        n = int(valid.sum())
        score = n / len(a)
        transform = {
            "model": "image_affine_v1",
            "source_to_target": homogeneous.tolist(),
            "inlier_mask": valid.tolist(),
            "inlier_ratio": score,
        }
        return Result(
            {"transform": transform, "inliers": n},
            score=score,
            passed=n >= integer(p["min_inliers"], "Minimum inliers", 2, 100000) and score >= ratio,
        )


class Remap(Block):
    type_id, title, category = "cv.remap", "Remap image coordinates", "Image processing"
    block_type, subtype = "Image processing", "Geometry"
    inputs = (Port("image"), Port("map_x"), Port("map_y"))
    outputs = (Port("image"),)
    defaults = {"interpolation": "linear", "border": "constant", "border_value": 0.0}
    parameter_choices = {"interpolation": tuple(INTERPOLATION), "border": tuple(WARP_BORDERS)}

    def process(self, inputs):
        image = image_value(inputs["image"])
        x, y = (image_value(inputs[k], (np.float32,)) for k in ("map_x", "map_y"))
        if x.ndim != 2 or y.shape != x.shape or x.size > 64_000_000:
            raise ValueError("Remap needs matching 2D float32 coordinate maps")
        value = finite(self.parameters["border_value"], "Border value")
        fill = tuple([value] * image.shape[2]) if image.ndim == 3 else value
        result = cv2.remap(
            image,
            x,
            y,
            INTERPOLATION[self.parameters["interpolation"]],
            borderMode=WARP_BORDERS[self.parameters["border"]],
            borderValue=fill,
        )
        return encoded_result(result, inputs["image"])


class CartesianToPolar(Block):
    type_id, title, category = "cv.cart_to_polar", "Cartesian to polar fields", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (Port("x"), Port("y"))
    outputs = (Port("magnitude"), Port("angle"))
    defaults = {"angle_units": "radians"}
    parameter_choices = {"angle_units": ("radians", "degrees")}

    def process(self, inputs):
        x, y = (image_value(inputs[k], (np.float32, np.float64)) for k in ("x", "y"))
        if x.ndim != 2 or x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("Cartesian fields must be matching monochrome floating arrays")
        magnitude, angle = cv2.cartToPolar(
            x, y, angleInDegrees=self.parameters["angle_units"] == "degrees"
        )
        if not np.isfinite(magnitude).all() or not np.isfinite(angle).all():
            raise ValueError("Polar conversion overflowed")
        return Result({"magnitude": magnitude, "angle": angle})


ANALYSIS_BLOCKS = (
    BoxFilter,
    SeparableFilter,
    CornerResponse,
    RefineCorners,
    PhaseCorrelation,
    EstimateAffine,
    Remap,
    CartesianToPolar,
)
