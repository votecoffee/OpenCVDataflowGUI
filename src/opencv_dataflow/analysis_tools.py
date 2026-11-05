"""Further wrappers with explicit algorithm controls and typed enum metadata."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, OUTPUT_DEPTHS, anchor_value
from .processing import ImageTool, encoded_result, finite, gray_image, image_value, integer
from .registration import BORDERS as WARP_BORDERS
from .registration import INTERPOLATION
from .selection import SELECTION_INPUTS, RegionalOperation, selection_mask


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


class IntensityTransform(ImageTool):
    type_id, title, subtype = "cv.intensity_transform", "Intensity transform", "Image math"
    inputs = SELECTION_INPUTS
    defaults = {
        "operation": "power",
        "exponent": 1.0,
        "minimum": 0.0,
        "maximum": 255.0,
        "output_depth": "float64",
        "preserve_alpha": True,
    }
    parameter_choices = {
        "operation": ("log", "exp", "sqrt", "power", "clip"),
        "output_depth": ("float64", "float32", "same"),
    }
    parameter_ranges = {
        "exponent": NumericRange(-100, 100),
        "minimum": NumericRange(),
        "maximum": NumericRange(),
    }
    parameter_help = {
        "operation": "Operate in raw sample units. Log is natural and requires positive input.",
        "exponent": "Power only. Negative bases require a whole exponent.",
        "minimum": "Clip only. Values below this lower bound are raised to it.",
        "maximum": "Clip only. Values above this upper bound are lowered to it.",
        "output_depth": "Same keeps the source depth, rounding and saturating integer results.",
        "preserve_alpha": "Copy the alpha channel through unchanged in the chosen output depth.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["operation"] == "clip" and float(p["minimum"]) > float(p["maximum"]):
            raise ValidationError("Minimum must not exceed maximum", "minimum", "maximum")

    def transform_samples(self, values):
        p = self.parameters
        operation = p["operation"]
        if operation == "log":
            if np.any(values <= 0):
                raise ValidationError("Log requires positive samples in the selection")
            return cv2.log(values)
        if operation == "exp":
            return cv2.exp(values)
        if operation == "sqrt":
            if np.any(values < 0):
                raise ValidationError("Square root requires nonnegative samples in the selection")
            return cv2.sqrt(values)
        if operation == "power":
            exponent = float(p["exponent"])
            if not exponent.is_integer() and np.any(values < 0):
                raise ValidationError("Negative samples require a whole exponent", "exponent")
            if exponent < 0 and np.any(values == 0):
                raise ValidationError("A negative exponent requires nonzero samples", "exponent")
            return cv2.pow(values, exponent)
        return np.clip(values, float(p["minimum"]), float(p["maximum"]))

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if encoding in ("hsv", "lab"):
            raise ValidationError("Intensity transform needs monochrome or RGB/BGR samples")

        p = self.parameters
        output = image.astype(np.float64)
        has_alpha = output.ndim == 3 and output.shape[2] == 4
        signal = output[..., :3] if has_alpha and p["preserve_alpha"] else output
        mask = selection_mask(inputs, image)
        selection = None if mask is None else mask != 0
        samples = signal if selection is None else signal[selection]
        if samples.size:
            mapped = self.transform_samples(samples.reshape(-1, 1)).reshape(samples.shape)
            if not np.isfinite(mapped).all():
                raise ValidationError("Intensity transform overflowed. Reduce the input range")
            if selection is None:
                signal[...] = mapped
            else:
                signal[selection] = mapped

        depth = image.dtype if p["output_depth"] == "same" else np.dtype(p["output_depth"])
        if depth.kind in "iu":
            limits = np.iinfo(depth)
            output = np.clip(np.rint(output), limits.min, limits.max)
        elif depth == np.float32 and np.any(np.abs(output) > np.finfo(depth).max):
            raise ValidationError(
                "Result exceeds the float32 range. Choose float64", "output_depth"
            )
        output = output.astype(depth, copy=False)
        graphics = {}
        if self.generate_graphics and depth.kind == "f":
            display = output[..., :3] if has_alpha else output
            display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            graphics["intensity"] = ImageArray(
                display, encoding.removesuffix("a") if has_alpha else encoding
            )
        return encoded_result(output, source, graphics=graphics)


def rectangular_sums(integral, y0, y1, x0, x1):
    return (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )


class LocalWindowStatistics(ImageTool):
    type_id, title, subtype = "cv.local_statistics", "Local window statistics", "Intensity"
    inputs = SELECTION_INPUTS
    outputs = (Port("mean"), Port("standard_deviation"), Port("sample_count"))
    defaults = {
        "width": 15,
        "height": 15,
        "anchor": [-1, -1],
        "border": "crop",
        "selection_mode": "samples_and_centers",
    }
    parameter_choices = {
        "border": ("crop", "valid"),
        "selection_mode": ("samples_and_centers", "centers"),
    }
    parameter_ranges = {key: NumericRange(1, 4095, whole=True) for key in ("width", "height")}
    parameter_help = {
        "anchor": "Pixel within the window placed at the output coordinate. [-1, -1] centers it.",
        "border": "Crop uses in-image samples. Valid suppresses windows crossing the image border.",
        "selection_mode": "Limit both contributing samples and output centers, or only centers.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        try:
            anchor_value(p["anchor"], int(p["width"]), int(p["height"]))
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc), "anchor") from exc

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        encoding = image_format(source)
        if encoding in ("hsv", "lab"):
            raise ValidationError("Local statistics need monochrome, RGB, or BGR intensity samples")
        p = self.parameters
        width, height = int(p["width"]), int(p["height"])
        anchor_x, anchor_y = anchor_value(p["anchor"], width, height)
        image_height, image_width = image.shape[:2]
        if anchor_x == -1:
            anchor_x = width // 2
        if anchor_y == -1:
            anchor_y = height // 2
        x = np.arange(image_width) - anchor_x
        y = np.arange(image_height) - anchor_y
        x0, x1 = x.clip(0, image_width), (x + width).clip(0, image_width)
        y0, y1 = y.clip(0, image_height), (y + height).clip(0, image_height)
        bounds = (y0, y1, x0, x1)

        selection = selection_mask(inputs, image)
        centers = None if selection is None else selection != 0
        sample_mask = centers if p["selection_mode"] == "samples_and_centers" else None
        if sample_mask is None:
            count = (y1 - y0)[:, None] * (x1 - x0)
        else:
            mask_sum = cv2.integral(sample_mask.astype(np.uint8), sdepth=cv2.CV_64F)
            count = rectangular_sums(mask_sum, *bounds)
        if p["border"] == "valid":
            within_x = (x >= 0) & (x + width <= image_width)
            within_y = (y >= 0) & (y + height <= image_height)
            count[~(within_y[:, None] & within_x)] = 0
        if centers is not None:
            count[~centers] = 0
        valid = count > 0
        divisor = np.maximum(count, 1)
        planes = image[..., None] if image.ndim == 2 else image
        mean = np.zeros(planes.shape, dtype=np.float64)
        deviation = np.zeros_like(mean)
        if valid.any():
            index = 0 if sample_mask is None else int(np.argmax(sample_mask))
            for channel in range(planes.shape[2]):
                plane = planes[..., channel]
                offset = float(plane.flat[index])
                with np.errstate(over="ignore", invalid="ignore"):
                    shifted = np.subtract(plane, offset, dtype=np.float64)
                if sample_mask is not None:
                    shifted[~sample_mask] = 0
                sums, squares = cv2.integral2(shifted, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
                if not np.isfinite(sums).all() or not np.isfinite(squares).all():
                    raise ValidationError("Local statistics overflowed. Reduce the sample range")
                centered_mean = rectangular_sums(sums, *bounds) / divisor
                centered_mean[~valid] = 0
                variance = rectangular_sums(squares, *bounds) / divisor - centered_mean**2
                mean[..., channel] = np.where(valid, centered_mean + offset, 0)
                deviation[..., channel] = np.where(valid, np.sqrt(np.maximum(variance, 0)), 0)
        if not np.isfinite(mean).all() or not np.isfinite(deviation).all():
            raise ValidationError("Local statistics produced nonfinite samples")
        graphics = {}
        if self.generate_graphics:
            contrast = np.max(deviation, axis=2)
            graphics["local contrast"] = cv2.normalize(
                contrast, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        if image.ndim == 2:
            mean, deviation = mean[..., 0], deviation[..., 0]
        return Result(
            {
                "mean": ImageArray(mean, encoding),
                "standard_deviation": ImageArray(deviation, encoding),
                "sample_count": ImageArray(count.astype(np.float32), "gray"),
            },
            graphics=graphics,
        )


ANALYSIS_BLOCKS = (
    BoxFilter,
    LocalWindowStatistics,
    IntensityTransform,
    SeparableFilter,
    CornerResponse,
    RefineCorners,
    PhaseCorrelation,
    EstimateAffine,
    Remap,
    CartesianToPolar,
)
