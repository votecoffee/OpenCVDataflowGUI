"""Small explicit OpenCV wrappers for building practical inspection graphs.

These are eager OpenCV operations. Native G-API blocks remain in blocks.py;
tools can opt into deterministic conversion to their native working depth.
"""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import (
    DepthArray,
    ImageArray,
    convert_depth,
    format_image,
    image_format,
    opencv_image,
    preview_rgb,
)
from .objects import blob_item, collection
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, OUTPUT_DEPTHS, anchor_value, optional_mask, scalar_value
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask


def integer(value, name, low=1, high=4095, odd=False):
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
        raise ValidationError(f"{name} must be a whole number")
    value = int(value)
    if not low <= value <= high or (odd and value % 2 == 0):
        raise ValidationError(f"{name} must be {'odd and ' if odd else ''}between {low} and {high}")
    return value


def finite(value, name, low=None):
    value = float(value)
    if not np.isfinite(value) or (low is not None and value < low):
        raise ValidationError(
            f"{name} must be finite" + (f" and at least {low}" if low is not None else "")
        )
    return value


def image_value(
    image,
    depths=(np.uint8, np.uint16, np.int16, np.float32, np.float64),
    canonical=False,
    *,
    allow_labels=False,
):
    if isinstance(image, DepthArray):
        raise ValidationError("Use depth tools for physical depth planes")
    if allow_labels and image.dtype == np.int32 and image.ndim == 2:
        return np.asarray(image)
    if (
        image.dtype not in depths
        or image.ndim not in (2, 3)
        or (image.ndim == 3 and image.shape[2] not in (3, 4))
    ):
        raise ValidationError("Unsupported image depth or channel layout for this operation")
    if not np.isfinite(image).all():
        raise ValidationError("Image samples must be finite")
    return opencv_image(image) if canonical else np.asarray(image)


def gray_image(image, depths=(np.uint8, np.uint16, np.float32), *, auto_depth=False):
    supported = (np.uint8, np.uint16, np.float32, np.float64) if auto_depth else depths
    native = image_value(image, supported)
    if native.ndim == 3:
        if auto_depth and native.dtype == np.float64:
            image = ImageArray(convert_depth(native, "float32"), image_format(image))
        color = opencv_image(image)
        code = cv2.COLOR_BGRA2GRAY if color.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        native = cv2.cvtColor(color, code)
    if native.dtype not in depths:
        native = convert_depth(native, np.dtype(depths[-1]).name)
    return native


def encoded_result(image, original, **kwargs):
    return Result({"image": ImageArray(image, image_format(original))}, **kwargs)


def geometry_result(output, original, matrix):
    if isinstance(original, DepthArray):
        image = DepthArray(output, original.units, original.display_min, original.display_max)
    else:
        image = ImageArray(output, image_format(original))
    return Result(
        {
            "image": image,
            "geometry": {
                "output_to_input": np.asarray(matrix, float).tolist(),
                "input_size": [original.shape[1], original.shape[0]],
                "output_size": [output.shape[1], output.shape[0]],
                "coordinate_convention": "continuous pixel centers; not resampling indices",
            },
        },
        message="Image geometry changed; measurement coordinates must follow its transform",
    )


class ImageTool(Block):
    category, block_type = "Image processing", "Image processing"
    inputs, outputs = (Port("image"),), (Port("image"),)


class ConvertImage(ImageTool):
    type_id, title, subtype = "cv.convert", "Convert image", "Color and depth"
    defaults = {"color_format": "gray", "pixel_depth": "unchanged"}
    parameter_choices = {
        "color_format": ("unchanged", "bgr", "rgb", "bgra", "rgba", "gray", "hsv", "lab"),
        "pixel_depth": ("unchanged", "uint8", "uint16", "float32"),
    }

    def process(self, inputs):
        source = inputs["image"]
        p = self.parameters
        if p["color_format"] == "unchanged":
            from .images import convert_depth

            image_value(source)
            output = ImageArray(convert_depth(source, p["pixel_depth"]), image_format(source))
            if image_format(source) in ("hsv", "lab") and output.dtype != source.dtype:
                raise ValueError("Select an explicit color format when changing HSV/Lab depth")
        else:
            source = image_value(source, canonical=True)
            output = format_image(source, p["color_format"], p["pixel_depth"])
        return Result({"image": output})


class Crop(ImageTool):
    type_id, title, subtype = "cv.crop", "Crop ROI", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {"x": 0, "y": 0, "width": 0, "height": 0}
    parameter_ranges = {key: NumericRange(0, whole=True) for key in defaults}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        x = integer(self.parameters["x"], "X", 0, image.shape[1] - 1)
        y = integer(self.parameters["y"], "Y", 0, image.shape[0] - 1)
        width = integer(self.parameters["width"], "Width", 0, image.shape[1]) or image.shape[1] - x
        height = (
            integer(self.parameters["height"], "Height", 0, image.shape[0]) or image.shape[0] - y
        )
        if x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValidationError("Crop ROI extends outside the image")
        return geometry_result(
            image[y : y + height, x : x + width].copy(), source, [[1, 0, x], [0, 1, y], [0, 0, 1]]
        )


class Resize(ImageTool):
    type_id, title, subtype = "cv.resize", "Resize", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {
        "width": 640,
        "height": 480,
        "interpolation": "linear",
        "size_mode": "absolute",
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    parameter_ranges = {key: NumericRange(1, 16384, whole=True) for key in ("width", "height")}
    parameter_choices = {
        "interpolation": (
            "nearest",
            "linear",
            "area",
            "cubic",
            "lanczos4",
            "nearest_exact",
            "linear_exact",
        ),
        "size_mode": ("absolute", "scale"),
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        width, height = (int(self.parameters[key]) for key in ("width", "height"))
        if self.parameters["size_mode"] == "scale":
            fx, fy = (finite(self.parameters[k], k, 0) for k in ("scale_x", "scale_y"))
            width, height = round(image.shape[1] * fx), round(image.shape[0] * fy)
            if width < 1 or height < 1:
                raise ValidationError("Resize scales must produce nonempty dimensions")
        if width * height > 64_000_000:
            raise ValidationError("Resize output is limited to 64 million pixels")
        method = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
            "nearest_exact": cv2.INTER_NEAREST_EXACT,
            "linear_exact": cv2.INTER_LINEAR_EXACT,
        }[self.parameters["interpolation"]]
        output = cv2.resize(image, (width, height), interpolation=method)
        sx, sy = image.shape[1] / width, image.shape[0] / height
        return geometry_result(
            output, source, [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]
        )


class Flip(ImageTool):
    type_id, title, subtype = "cv.flip", "Flip", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {"direction": "horizontal"}
    parameter_choices = {"direction": ("horizontal", "vertical", "both")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        direction = self.parameters["direction"]
        code = {"horizontal": 1, "vertical": 0, "both": -1}[direction]
        xflip, yflip = direction != "vertical", direction != "horizontal"
        return geometry_result(
            cv2.flip(image, code),
            source,
            [
                [-1 if xflip else 1, 0, image.shape[1] - 1 if xflip else 0],
                [0, -1 if yflip else 1, image.shape[0] - 1 if yflip else 0],
                [0, 0, 1],
            ],
        )


class Rotate90(ImageTool):
    type_id, title, subtype = "cv.rotate90", "Rotate 90 / 180", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {"rotation": "clockwise"}
    parameter_choices = {"rotation": ("clockwise", "counterclockwise", "180")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        code, matrix = {
            "clockwise": (
                cv2.ROTATE_90_CLOCKWISE,
                [[0, 1, 0], [-1, 0, image.shape[0] - 1], [0, 0, 1]],
            ),
            "counterclockwise": (
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                [[0, -1, image.shape[1] - 1], [1, 0, 0], [0, 0, 1]],
            ),
            "180": (
                cv2.ROTATE_180,
                [[-1, 0, image.shape[1] - 1], [0, -1, image.shape[0] - 1], [0, 0, 1]],
            ),
        }[self.parameters["rotation"]]
        return geometry_result(cv2.rotate(image, code), source, matrix)


class TransposeImage(ImageTool):
    type_id, title, subtype = "cv.transpose_image", "Transpose image", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {"diagonal": "main"}
    parameter_choices = {"diagonal": ("main", "anti")}
    parameter_help = {
        "diagonal": "Main swaps X and Y. Anti swaps axes and reverses both output directions.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        if isinstance(source, DepthArray) and source.ndim == 2:
            image = np.asarray(source)
        else:
            image = image_value(source, allow_labels=True)
        output = cv2.transpose(image)
        if self.parameters["diagonal"] == "anti":
            height, width = image.shape[:2]
            matrix = [[0, -1, width - 1], [-1, 0, height - 1], [0, 0, 1]]
            output = cv2.flip(output, -1)
        else:
            matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        return geometry_result(output, source, matrix)


class BoxBlur(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.box_blur", "Box blur", "Filtering"
    defaults = {"kernel": 5, "kernel_y": 0, "anchor": [-1, -1], "border": "reflect"}
    parameter_choices = {"border": tuple(BORDERS)}
    parameter_ranges = {
        "kernel": NumericRange(1, 255, whole=True, odd=True),
        "kernel_y": NumericRange(0, 255, whole=True),
    }

    def process(self, inputs):
        image = image_value(inputs["image"])
        kernel = int(self.parameters["kernel"])
        ky = int(self.parameters["kernel_y"]) or kernel
        anchor = anchor_value(self.parameters["anchor"], kernel, ky)
        return encoded_result(
            cv2.blur(
                image, (kernel, ky), anchor=anchor, borderType=BORDERS[self.parameters["border"]]
            ),
            inputs["image"],
        )


class MedianBlur(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.median_blur", "Median blur", "Filtering"
    defaults = {"kernel": 3}
    parameter_ranges = {"kernel": NumericRange(3, 255, whole=True, odd=True)}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32))
        kernel = int(self.parameters["kernel"])
        if image.dtype != np.uint8 and kernel > 5:
            raise ValidationError("Median kernels above 5 require uint8")
        return encoded_result(cv2.medianBlur(image, kernel), inputs["image"])


class BilateralFilter(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.bilateral", "Bilateral filter", "Filtering"
    defaults = {"diameter": 5, "sigma_color": 50.0, "sigma_space": 5.0, "border": "reflect"}
    parameter_choices = {"border": tuple(BORDERS)}
    parameter_ranges = {
        "diameter": NumericRange(0, 255, whole=True),
        "sigma_color": NumericRange(0),
        "sigma_space": NumericRange(0),
    }

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.float32))
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValidationError("Bilateral filter accepts one or three channels")
        diameter = int(self.parameters["diameter"])
        color, space = (float(self.parameters[key]) for key in ("sigma_color", "sigma_space"))
        return encoded_result(
            cv2.bilateralFilter(
                image, diameter, color, space, borderType=BORDERS[self.parameters["border"]]
            ),
            inputs["image"],
        )


class AutoThreshold(ImageTool):
    inputs = SELECTION_INPUTS
    process_inputs_readonly = True
    input_help = {
        "mask": "Learn and write the threshold only on selected pixels. Elsewhere stays zero.",
        "region": "Learn and write the threshold only inside this region. Elsewhere stays zero.",
    }
    type_id, title, subtype = "cv.auto_threshold", "Automatic threshold", "Segmentation"
    outputs = (Port("image"), Port("threshold", "number"))
    defaults = {"method": "otsu", "invert": False, "max_value": -1.0}
    parameter_choices = {"method": ("otsu", "triangle")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.uint16))
        method = self.parameters["method"]
        if method == "triangle" and image.dtype != np.uint8:
            raise ValueError("Triangle threshold requires uint8")
        mode = {"otsu": cv2.THRESH_OTSU, "triangle": cv2.THRESH_TRIANGLE}[method]
        mode |= cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        maximum = finite(self.parameters["max_value"], "Maximum value")
        if maximum < 0:
            maximum = int(np.iinfo(image.dtype).max)
        selection = selection_mask(inputs, image)
        if selection is None:
            level, mask = cv2.threshold(image, 0, maximum, mode)
        else:
            if not selection.any():
                raise ValueError("Automatic threshold requires a nonempty selection")
            level, mask = cv2.thresholdWithMask(
                image, np.zeros_like(image), selection, 0, maximum, mode
            )
        return Result({"image": mask, "threshold": float(level)})


class AdaptiveThreshold(RegionalOperation, ImageTool):
    region_background = "zero"
    type_id, title, subtype = "cv.adaptive_threshold", "Adaptive threshold", "Segmentation"
    defaults = {
        "method": "gaussian",
        "block_size": 11,
        "constant": 2.0,
        "invert": False,
        "max_value": 255.0,
    }
    parameter_choices = {"method": ("mean", "gaussian")}
    parameter_ranges = {
        "block_size": NumericRange(3, 255, whole=True, odd=True),
        "max_value": NumericRange(0),
        "constant": NumericRange(),
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        size = int(self.parameters["block_size"])
        method = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C, "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}[
            self.parameters["method"]
        ]
        mode = cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        output = cv2.adaptiveThreshold(
            image,
            float(self.parameters["max_value"]),
            method,
            mode,
            size,
            float(self.parameters["constant"]),
        )
        return Result({"image": output})


class InRange(RegionalOperation, ImageTool):
    region_background = "zero"
    type_id, title, subtype = "cv.in_range", "In range mask", "Segmentation"
    inputs = (
        *SELECTION_INPUTS,
        Port("lower", "data", required=False, required_when_connected=True),
        Port("upper", "data", required=False, required_when_connected=True),
    )
    defaults = {"lower": [0], "upper": [255]}
    input_help = {
        "image": "Grayscale or color image, or a signed int32 label plane.",
        "lower": "Inclusive lower bound. Wire one number or a list of values in channel order.",
        "upper": "Inclusive upper bound. Wire one number or a list of values in channel order.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = [port.name for port in SELECTION_INPUTS]

    def process(self, inputs):
        # Deliberately retain the actual encoding: HSV masks must test HSV samples.
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32), allow_labels=True)
        count = 1 if image.ndim == 2 else image.shape[2]
        bounds = []
        for name in ("lower", "upper"):
            try:
                values = np.asarray(inputs.get(name, self.parameters[name]), float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(f"{name} needs finite numeric bounds") from exc
            if values.ndim == 0:
                values = values.reshape(1)
            if values.ndim != 1 or values.size not in (1, count) or not np.isfinite(values).all():
                raise ValidationError(f"{name} needs one finite value or one value per channel")
            bounds.append(np.repeat(values, count) if values.size == 1 else values)
        if np.any(bounds[0] > bounds[1]):
            raise ValidationError("Lower bounds must not exceed upper bounds")
        wired_bounds = "lower" in inputs or "upper" in inputs
        if image.dtype == np.int32 or (image.dtype.kind == "u" and wired_bounds):
            limits = np.iinfo(image.dtype)
            lower = np.maximum(np.ceil(bounds[0]), limits.min)
            upper = np.minimum(np.floor(bounds[1]), limits.max)
            if np.any(lower > upper):
                return Result({"image": np.zeros(image.shape[:2], dtype=np.uint8)})
            bounds = (lower, upper)
        return Result({"image": cv2.inRange(image, tuple(bounds[0]), tuple(bounds[1]))})


class Morphology(RegionalOperation, ImageTool):
    region_background = "zero"
    supports_kernel_editor = True
    parameter_ranges = {"iterations": NumericRange(1, 100, whole=True)}
    type_id, title, subtype = "cv.morphology", "Morphology", "Morphology"
    defaults = {
        "operation": "open",
        "shape": "ellipse",
        "kernel": 3,
        "iterations": 1,
        "kernel_y": 0,
        "custom_kernel": [],
        "anchor": [-1, -1],
        "shape_anchor": [-1, -1],
        "border": "constant",
        "border_value": [],
    }
    parameter_choices = {
        "operation": ("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"),
        "shape": ("rect", "ellipse", "cross", "diamond", "custom"),
        "border": tuple(BORDERS),
    }
    parameter_help = {
        "border_value": "Empty keeps the neutral native border; otherwise one to four values.",
        "shape_anchor": "Kernel construction anchor (cross shape); operation anchor is separate.",
        "custom_kernel": "Rectangular grid of 0s and 1s used when Shape is custom.",
    }

    @staticmethod
    def kernel_matrix(parameters):
        if parameters["shape"] == "custom":
            kernel = np.asarray(parameters["custom_kernel"])
            if (
                kernel.ndim != 2
                or not all(1 <= size <= 255 for size in kernel.shape)
                or not np.isin(kernel, (0, 1)).all()
                or not np.any(kernel)
            ):
                raise ValueError(
                    "Custom kernel needs a rectangular grid of 0s and 1s, "
                    "up to 255 × 255, with at least one active cell"
                )
            return kernel.astype(np.uint8)
        width = integer(parameters["kernel"], "Kernel width", high=255)
        height = integer(parameters["kernel_y"], "Kernel height", 0, 255) or width
        shape = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS,
            "diamond": cv2.MORPH_DIAMOND,
        }[parameters["shape"]]
        return cv2.getStructuringElement(
            shape, (width, height), anchor_value(parameters["shape_anchor"], width, height)
        )

    def process(self, inputs):
        image = image_value(inputs["image"])
        kernel = self.kernel_matrix(self.parameters)
        height, width = kernel.shape
        iterations = int(self.parameters["iterations"])
        op = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }[self.parameters["operation"]]
        options = {
            "iterations": iterations,
            "anchor": anchor_value(self.parameters["anchor"], width, height),
            "borderType": BORDERS[self.parameters["border"]],
        }
        if self.parameters["border_value"]:
            options["borderValue"] = scalar_value(self.parameters["border_value"])
        output = cv2.morphologyEx(image, op, kernel, **options)
        return encoded_result(output, inputs["image"])


def matching_images(inputs):
    a = image_value(inputs["image_a"], canonical=True)
    if "image_b" not in inputs:
        raise ValueError("This operation requires image_b")
    b = image_value(inputs["image_b"], canonical=True)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Both images must have identical shape and pixel depth")
    return a, b


class Bitwise(Block):
    type_id, title, category = "cv.bitwise", "Bitwise image operation", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (
        Port("image_a"),
        Port("image_b", required=False),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("image"),)
    defaults = {"operation": "and"}
    parameter_choices = {"operation": ("and", "or", "xor", "not")}

    def process(self, inputs):
        a = image_value(inputs["image_a"], (np.uint8, np.uint16), canonical=True)
        op = self.parameters["operation"]
        mask = optional_mask(inputs, a)
        if op == "not":
            output = cv2.bitwise_not(a, mask=mask)
        else:
            a, b = matching_images(inputs)
            output = {"and": cv2.bitwise_and, "or": cv2.bitwise_or, "xor": cv2.bitwise_xor}[op](
                a, b, mask=mask
            )
        return Result({"image": output})


class Blend(RegionalOperation, Block):
    region_source = "image_a"
    type_id, title, category = "cv.blend", "Weighted blend", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (Port("image"),)
    defaults = {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "output_depth": "same"}
    parameter_choices = {"output_depth": tuple(OUTPUT_DEPTHS)}

    def process(self, inputs):
        a, b = matching_images(inputs)
        alpha, beta, gamma = (finite(self.parameters[k], k) for k in ("alpha", "beta", "gamma"))
        output = cv2.addWeighted(
            a, alpha, b, beta, gamma, dtype=OUTPUT_DEPTHS[self.parameters["output_depth"]]
        )
        if not np.isfinite(output).all():
            raise ValueError("Blend coefficients overflowed the output image depth")
        return Result({"image": output})


class Gradient(RegionalOperation, ImageTool):
    region_outputs = ("gradient", "preview")
    region_background = "zero"
    type_id, title, subtype = "cv.gradient", "Sobel / Scharr / Laplacian", "Edges"
    outputs = (Port("gradient"), Port("preview"))
    defaults = {
        "operator": "sobel",
        "dx": 1,
        "dy": 0,
        "kernel": 3,
        "scale": 1.0,
        "delta": 0.0,
        "output_depth": "float32",
        "border": "reflect",
    }
    parameter_choices = {
        "operator": ("sobel", "scharr", "laplacian"),
        "output_depth": ("int16", "float32", "float64"),
        "border": tuple(BORDERS),
    }

    def process(self, inputs):
        image = gray_image(inputs["image"])
        op = self.parameters["operator"]
        size = integer(self.parameters["kernel"], "Kernel", high=7, odd=True)
        scale = finite(self.parameters["scale"], "Scale")
        depth = OUTPUT_DEPTHS[self.parameters["output_depth"]]
        options = {
            "scale": scale,
            "delta": finite(self.parameters["delta"], "Delta"),
            "borderType": BORDERS[self.parameters["border"]],
        }
        if op == "laplacian":
            gradient = cv2.Laplacian(image, depth, ksize=size, **options)
        else:
            dx = integer(self.parameters["dx"], "DX", 0, 2)
            dy = integer(self.parameters["dy"], "DY", 0, 2)
            if dx + dy == 0 or (op == "scharr" and dx + dy != 1):
                raise ValueError("Derivative order must be positive; Scharr requires DX+DY=1")
            if op == "scharr":
                gradient = cv2.Scharr(image, depth, dx, dy, **options)
            elif op == "sobel":
                gradient = cv2.Sobel(image, depth, dx, dy, ksize=size, **options)
            else:
                raise ValueError("Unknown gradient operator")
        if not np.isfinite(gradient).all():
            raise ValueError("Gradient overflowed float32 output")
        absolute = np.abs(gradient.astype(np.float64))
        preview = np.rint(absolute / max(float(absolute.max()), 1e-20) * 255).astype(np.uint8)
        return Result(
            {"gradient": gradient, "preview": preview}, graphics={"absolute gradient": preview}
        )


class ConnectedComponents(ImageTool):
    parameter_help = {
        "region_mode": "Center keeps whole objects; clip measures only their selected pixels."
    }
    inputs = SELECTION_INPUTS
    type_id, title, subtype = "inspect.components", "Connected components", "Shapes"
    category, block_type = "Inspection", "Inspection"
    outputs = (
        Port("overlay"),
        Port("mask"),
        Port("count", "number"),
        Port("centroids", "data"),
        Port("data", "data"),
        Port("collection", "data"),
    )
    defaults = {
        "connectivity": 8,
        "min_area": 100,
        "max_area": 0,
        "expected_count": 2,
        "algorithm": "default",
        "region_mode": "center",
    }
    parameter_choices = {
        "connectivity": ("4", "8"),
        "algorithm": ("default", "sauf", "bbdt", "spaghetti"),
        "region_mode": ("center", "clip"),
    }
    parameter_ranges = {
        "min_area": NumericRange(1, 2**31 - 1, whole=True),
        "max_area": NumericRange(0, 2**31 - 1, whole=True),
        "expected_count": NumericRange(0, 2**31 - 1, whole=True),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["max_area"] and p["max_area"] < p["min_area"]:
            raise ValidationError(
                "Maximum area must be zero (unlimited) or at least minimum area",
                "min_area",
                "max_area",
            )

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        connectivity = int(self.parameters["connectivity"])
        minimum, maximum, expected = (
            int(self.parameters[key]) for key in ("min_area", "max_area", "expected_count")
        )
        algorithm = {
            "default": cv2.CCL_DEFAULT,
            "sauf": cv2.CCL_SAUF,
            "bbdt": cv2.CCL_BBDT,
            "spaghetti": cv2.CCL_SPAGHETTI,
        }[self.parameters["algorithm"]]
        selection = selection_mask(inputs, image)
        search = image != 0
        if selection is not None and self.parameters["region_mode"] == "clip":
            search &= selection != 0
        total, labels, stats, centers = cv2.connectedComponentsWithStatsWithAlgorithm(
            search.astype(np.uint8), connectivity, cv2.CV_32S, algorithm
        )
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(image)
        components = []
        items = []
        for label in range(1, total):
            x, y, width, height, area = map(int, stats[label])
            if area < minimum or (maximum and area > maximum):
                continue
            center = centers[label].tolist()
            if self.parameters["region_mode"] == "center" and not contains(selection, center):
                continue
            components.append({"area": area, "bounds": [x, y, width, height], "centroid": center})
            items.append(
                blob_item(
                    (labels[y : y + height, x : x + width] == label).astype(np.uint8), x, y, label
                )
            )
            mask[labels == label] = 255
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (70, 220, 140), 1)
            cv2.drawMarker(
                overlay, tuple(np.rint(center).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 12, 1
            )
        count = len(components)
        return Result(
            {
                "overlay": overlay,
                "mask": mask,
                "count": count,
                "centroids": [c["centroid"] for c in components],
                "data": {"count": count, "components": components},
                "collection": collection("blobs", items, image, self.input_payloads["image"]),
            },
            graphics={"components": overlay},
            drawings={
                "components": GraphicDrawing(
                    tuple(graphic_box(*item["bounds"], (70, 220, 140)) for item in components)
                )
            } if self.generate_graphics else {},
            passed=count == expected,
            score=max(0.0, 1 - abs(count - expected) / max(expected, 1)),
            message=f"{count} components / expected {expected}; coordinates in input image",
        )


class MatchTemplate(Block):
    type_id, title, category = "inspect.match_template", "Match template", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs = (
        Port("image"),
        Port("template"),
        Port("mask", required=False),
        Port("search_mask", required=False),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("overlay"),
        Port("score", "number"),
        Port("location", "data"),
        Port("center", "data"),
        Port("raw_score", "number"),
        Port("response"),
    )
    defaults = {"method": "ccoeff_normed", "min_score": 0.8, "subpixel": False}
    parameter_choices = {
        "method": ("ccoeff_normed", "ccorr_normed", "sqdiff_normed", "ccoeff", "ccorr", "sqdiff")
    }

    parameter_help = {
        "subpixel": "Fit a local quadratic to refine centers when the response has a clear peak.",
    }

    def refined_top_left(self, response, allowed, method, x, y):
        if not self.parameters["subpixel"]:
            return None
        if not (0 < x < response.shape[1] - 1 and 0 < y < response.shape[0] - 1):
            return None
        window = np.s_[y - 1:y + 2, x - 1:x + 2]
        if allowed is not None and not allowed[window].all():
            return None
        peak = response[window].astype(np.float64)
        if method.startswith("sqdiff"):
            peak = -peak
        center = peak[1, 1]
        neighbors = np.delete(peak.ravel(), 4)
        if center <= neighbors.max():
            return None
        left, right, above, below = peak[1, 0], peak[1, 2], peak[0, 1], peak[2, 1]
        xx, yy = left + right - 2 * center, above + below - 2 * center
        xy = (peak[2, 2] - peak[2, 0] - peak[0, 2] + peak[0, 0]) / 4
        if xx >= 0 or yy >= 0 or xx * yy <= xy * xy:
            return None
        dx, dy = np.linalg.solve([[xx, xy], [xy, yy]], [(left - right) / 2, (above - below) / 2])
        if abs(dx) > 1 or abs(dy) > 1:
            return None
        return [x + float(dx), y + float(dy)]

    def response_maps(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        template = gray_image(inputs["template"], (np.uint8, np.float32))
        if image.dtype != template.dtype:
            raise ValueError("Image and template must have the same pixel depth")
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            raise ValueError("Template must fit inside the search image")
        method = self.parameters["method"]
        if method in ("ccoeff_normed", "ccoeff") and float(template.std()) < 1e-12:
            raise ValueError("Normalized coefficient matching requires a nonconstant template")
        if float(np.square(template.astype(float)).sum()) <= 1e-20:
            raise ValueError("Normalized matching requires a nonzero template")
        threshold = finite(self.parameters["min_score"], "Minimum score", 0)
        if threshold > 1:
            raise ValueError("Minimum score must be between 0 and 1")
        code = {
            "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
            "ccorr_normed": cv2.TM_CCORR_NORMED,
            "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
            "ccoeff": cv2.TM_CCOEFF,
            "ccorr": cv2.TM_CCORR,
            "sqdiff": cv2.TM_SQDIFF,
        }[method]
        mask = inputs.get("mask")
        if mask is not None:
            if (
                mask.dtype not in (np.uint8, np.float32)
                or mask.ndim != 2
                or mask.shape != template.shape
            ):
                raise ValueError(
                    "Template mask must be uint8/float32 monochrome and template-sized"
                )
            if method not in ("sqdiff", "ccorr_normed"):
                raise ValueError("Portable mask support is available for SQDIFF or CCORR_NORMED")
        response = cv2.matchTemplate(image, template, code, mask=mask)
        if not np.isfinite(response).all():
            raise ValueError("Template matching produced a nonfinite response")
        selection = selection_mask(
            {
                "image": inputs["image"],
                "mask": inputs.get("search_mask"),
                "region": inputs.get("region"),
            },
            image,
        )
        candidate_mask = None
        if selection is not None:
            # Match positions are top-left coordinates; selection tests the template center.
            cy, cx = np.rint([(template.shape[0] - 1) / 2, (template.shape[1] - 1) / 2]).astype(int)
            candidate_mask = selection[cy : cy + response.shape[0], cx : cx + response.shape[1]]
            if not candidate_mask.any():
                raise ValueError("Search region contains no legal template centers")
        quality_code = {
            "sqdiff": cv2.TM_SQDIFF_NORMED,
            "ccorr": cv2.TM_CCORR_NORMED,
            "ccoeff": cv2.TM_CCOEFF_NORMED,
        }.get(method, code)
        quality = (
            response
            if quality_code == code
            else cv2.matchTemplate(image, template, quality_code, mask=mask)
        )
        return template, response, quality, candidate_mask, method, threshold

    def process(self, inputs):
        template, response, quality, candidate_mask, method, threshold = self.response_maps(inputs)
        _, _, low_pos, high_pos = cv2.minMaxLoc(response, mask=candidate_mask)
        location = low_pos if method in ("sqdiff_normed", "sqdiff") else high_pos
        x, y = location
        score = float(
            np.clip(
                1 - quality[y, x] if method in ("sqdiff_normed", "sqdiff") else quality[y, x], 0, 1
            )
        )
        if not np.isfinite(score):
            raise ValueError("Template quality is nonfinite")
        height, width = template.shape
        overlay = cv2.cvtColor(preview_rgb(inputs["image"]), cv2.COLOR_RGB2BGR)
        cv2.rectangle(
            overlay,
            location,
            (x + width - 1, y + height - 1),
            (70, 220, 140) if score >= threshold else (60, 90, 240),
            2,
        )
        top_left = self.refined_top_left(response, candidate_mask, method, x, y)
        details = {"top_left": [x, y], "bounds": [x, y, width, height], "method": method}
        if top_left is not None:
            details["subpixel_top_left"] = top_left
        cx, cy = top_left if top_left is not None else (x, y)
        heat = 1 - quality if method in ("sqdiff_normed", "sqdiff") else quality
        heat = cv2.applyColorMap(
            np.rint(np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        return Result(
            {
                "overlay": overlay,
                "score": score,
                "location": details,
                "center": [[cx + (width - 1) / 2, cy + (height - 1) / 2]],
                "raw_score": float(response[y, x]),
                "response": response,
            },
            graphics={"match": overlay, "response": heat},
            drawings={
                "match": GraphicDrawing((
                    graphic_box(
                        x, y, width, height,
                        (70, 220, 140) if score >= threshold else (60, 90, 240),
                    ),
                ))
            } if self.generate_graphics else {},
            passed=score >= threshold,
            score=score,
            message=f"Best match {score:.1%} at ({x}, {y})",
        )


PROCESSING_BLOCKS = (
    ConvertImage,
    Crop,
    Resize,
    Flip,
    Rotate90,
    TransposeImage,
    BoxBlur,
    MedianBlur,
    BilateralFilter,
    AutoThreshold,
    AdaptiveThreshold,
    InRange,
    Morphology,
    Bitwise,
    Blend,
    Gradient,
    ConnectedComponents,
    MatchTemplate,
)
