"""Small explicit OpenCV wrappers for building practical inspection graphs.

These are eager OpenCV operations. Native G-API blocks remain in blocks.py;
unsupported image types fail explicitly rather than silently changing depth.
"""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import DepthArray, ImageArray, format_image, image_format, opencv_image, preview_rgb
from .parameters import BORDERS, OUTPUT_DEPTHS, anchor_value, optional_mask, scalar_value
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask


def integer(value, name, low=1, high=4095, odd=False):
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be a whole number")
    value = int(value)
    if not low <= value <= high or (odd and value % 2 == 0):
        raise ValueError(f"{name} must be {'odd and ' if odd else ''}between {low} and {high}")
    return value


def finite(value, name, low=None):
    value = float(value)
    if not np.isfinite(value) or (low is not None and value < low):
        raise ValueError(
            f"{name} must be finite" + (f" and at least {low}" if low is not None else "")
        )
    return value


def image_value(
    image, depths=(np.uint8, np.uint16, np.int16, np.float32, np.float64), canonical=False
):
    if isinstance(image, DepthArray):
        raise ValueError("Use depth tools for physical depth planes")
    if (
        image.dtype not in depths
        or image.ndim not in (2, 3)
        or (image.ndim == 3 and image.shape[2] not in (3, 4))
    ):
        raise ValueError("Unsupported image depth or channel layout for this operation")
    if not np.isfinite(image).all():
        raise ValueError("Image samples must be finite")
    return opencv_image(image) if canonical else np.asarray(image)


def gray_image(image, depths=(np.uint8, np.uint16, np.float32)):
    image = image_value(image, depths, canonical=True)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def encoded_result(image, original, **kwargs):
    return Result({"image": ImageArray(image, image_format(original))}, **kwargs)


def geometry_result(output, original, matrix):
    return Result(
        {
            "image": ImageArray(output, image_format(original)),
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
            raise ValueError("Crop ROI extends outside the image")
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
        width = integer(self.parameters["width"], "Width", high=16384)
        height = integer(self.parameters["height"], "Height", high=16384)
        if self.parameters["size_mode"] == "scale":
            fx, fy = (finite(self.parameters[k], k, 0) for k in ("scale_x", "scale_y"))
            width, height = round(image.shape[1] * fx), round(image.shape[0] * fy)
            if width < 1 or height < 1:
                raise ValueError("Resize scales must produce nonempty dimensions")
        if width * height > 64_000_000:
            raise ValueError("Resize output is limited to 64 million pixels")
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


class BoxBlur(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.box_blur", "Box blur", "Filtering"
    defaults = {"kernel": 5, "kernel_y": 0, "anchor": [-1, -1], "border": "reflect"}
    parameter_choices = {"border": tuple(BORDERS)}

    def process(self, inputs):
        image = image_value(inputs["image"])
        kernel = integer(self.parameters["kernel"], "Kernel", high=255, odd=True)
        ky = integer(self.parameters["kernel_y"], "Kernel Y", 0, 255) or kernel
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

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32))
        kernel = integer(self.parameters["kernel"], "Kernel", low=3, high=255, odd=True)
        if image.dtype != np.uint8 and kernel > 5:
            raise ValueError("Median kernels above 5 require uint8")
        return encoded_result(cv2.medianBlur(image, kernel), inputs["image"])


class BilateralFilter(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.bilateral", "Bilateral filter", "Filtering"
    defaults = {"diameter": 5, "sigma_color": 50.0, "sigma_space": 5.0, "border": "reflect"}
    parameter_choices = {"border": tuple(BORDERS)}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.float32))
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Bilateral filter accepts one or three channels")
        diameter = integer(self.parameters["diameter"], "Diameter", low=0, high=255)
        color = finite(self.parameters["sigma_color"], "Sigma color", 0)
        space = finite(self.parameters["sigma_space"], "Sigma space", 0)
        return encoded_result(
            cv2.bilateralFilter(
                image, diameter, color, space, borderType=BORDERS[self.parameters["border"]]
            ),
            inputs["image"],
        )


class AutoThreshold(RegionalOperation, ImageTool):
    region_background = "zero"
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
            samples = image[selection != 0]
            if not samples.size:
                raise ValueError("Automatic threshold requires a nonempty selection")
            level, _ = cv2.threshold(samples.reshape(-1, 1), 0, maximum, mode)
            basic = cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
            _, mask = cv2.threshold(image, level, maximum, basic)
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

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        size = integer(self.parameters["block_size"], "Block size", low=3, high=255, odd=True)
        method = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C, "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}[
            self.parameters["method"]
        ]
        mode = cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        output = cv2.adaptiveThreshold(
            image,
            finite(self.parameters["max_value"], "Maximum value", 0),
            method,
            mode,
            size,
            finite(self.parameters["constant"], "Constant"),
        )
        return Result({"image": output})


class InRange(RegionalOperation, ImageTool):
    region_background = "zero"
    type_id, title, subtype = "cv.in_range", "In range mask", "Segmentation"
    defaults = {"lower": [0], "upper": [255]}

    def process(self, inputs):
        # Deliberately retain the actual encoding: HSV masks must test HSV samples.
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32))
        count = 1 if image.ndim == 2 else image.shape[2]
        bounds = []
        for name in ("lower", "upper"):
            values = np.asarray(self.parameters[name], float)
            if values.ndim != 1 or values.size not in (1, count) or not np.isfinite(values).all():
                raise ValueError(f"{name} needs one finite value or one value per channel")
            bounds.append(np.repeat(values, count) if values.size == 1 else values)
        if np.any(bounds[0] > bounds[1]):
            raise ValueError("Lower bounds must not exceed upper bounds")
        return Result({"image": cv2.inRange(image, tuple(bounds[0]), tuple(bounds[1]))})


class Morphology(RegionalOperation, ImageTool):
    region_background = "zero"
    type_id, title, subtype = "cv.morphology", "Morphology", "Morphology"
    defaults = {
        "operation": "open",
        "shape": "ellipse",
        "kernel": 3,
        "iterations": 1,
        "kernel_y": 0,
        "anchor": [-1, -1],
        "shape_anchor": [-1, -1],
        "border": "constant",
        "border_value": [],
    }
    parameter_choices = {
        "operation": ("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"),
        "shape": ("rect", "ellipse", "cross"),
        "border": tuple(BORDERS),
    }
    parameter_help = {
        "border_value": "Empty keeps the neutral native border; otherwise one to four values.",
        "shape_anchor": "Kernel construction anchor (cross shape); operation anchor is separate.",
    }

    def process(self, inputs):
        image = image_value(inputs["image"])
        size = integer(self.parameters["kernel"], "Kernel", high=255, odd=True)
        iterations = integer(self.parameters["iterations"], "Iterations", high=100)
        shape = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}[
            self.parameters["shape"]
        ]
        op = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }[self.parameters["operation"]]
        ky = integer(self.parameters["kernel_y"], "Kernel Y", 0, 255) or size
        anchor = anchor_value(self.parameters["anchor"], size, ky)
        shape_anchor = anchor_value(self.parameters["shape_anchor"], size, ky)
        options = {
            "iterations": iterations,
            "anchor": anchor,
            "borderType": BORDERS[self.parameters["border"]],
        }
        if self.parameters["border_value"]:
            options["borderValue"] = scalar_value(self.parameters["border_value"])
        output = cv2.morphologyEx(
            image, op, cv2.getStructuringElement(shape, (size, ky), shape_anchor), **options
        )
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

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        connectivity = integer(float(self.parameters["connectivity"]), "Connectivity", 4, 8)
        if connectivity not in (4, 8):
            raise ValueError("Connectivity must be 4 or 8")
        minimum = integer(self.parameters["min_area"], "Minimum area", 1, 2**31 - 1)
        maximum = integer(self.parameters["max_area"], "Maximum area", 0, 2**31 - 1)
        expected = integer(self.parameters["expected_count"], "Expected count", 0, 2**31 - 1)
        if maximum and maximum < minimum:
            raise ValueError("Maximum area must be zero (unlimited) or at least minimum area")
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
        for label in range(1, total):
            x, y, width, height, area = map(int, stats[label])
            if area < minimum or (maximum and area > maximum):
                continue
            center = centers[label].tolist()
            if self.parameters["region_mode"] == "center" and not contains(selection, center):
                continue
            components.append({"area": area, "bounds": [x, y, width, height], "centroid": center})
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
            },
            graphics={"components": overlay},
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
    defaults = {"method": "ccoeff_normed", "min_score": 0.8}
    parameter_choices = {
        "method": ("ccoeff_normed", "ccorr_normed", "sqdiff_normed", "ccoeff", "ccorr", "sqdiff")
    }

    def process(self, inputs):
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
            {"mask": inputs.get("search_mask"), "region": inputs.get("region")}, image
        )
        candidate_mask = None
        if selection is not None:
            # Match positions are top-left coordinates; selection tests the template center.
            cy, cx = np.rint([(template.shape[0] - 1) / 2, (template.shape[1] - 1) / 2]).astype(int)
            candidate_mask = selection[cy : cy + response.shape[0], cx : cx + response.shape[1]]
            if not candidate_mask.any():
                raise ValueError("Search region contains no legal template centers")
        low, high, low_pos, high_pos = cv2.minMaxLoc(response, mask=candidate_mask)
        location = low_pos if method in ("sqdiff_normed", "sqdiff") else high_pos
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
        heat = 1 - quality if method in ("sqdiff_normed", "sqdiff") else quality
        heat = cv2.applyColorMap(
            np.rint(np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        return Result(
            {
                "overlay": overlay,
                "score": score,
                "location": {"top_left": [x, y], "bounds": [x, y, width, height], "method": method},
                "center": [[x + (width - 1) / 2, y + (height - 1) / 2]],
                "raw_score": float(response[y, x]),
                "response": response,
            },
            graphics={"match": overlay, "response": heat},
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
