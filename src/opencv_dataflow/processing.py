"""Small explicit OpenCV wrappers for building practical inspection graphs.

These are eager OpenCV operations. Native G-API blocks remain in blocks.py;
unsupported image types fail explicitly rather than silently changing depth.
"""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import DepthArray, ImageArray, format_image, image_format, opencv_image, preview_rgb


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


def image_value(image, depths=(np.uint8, np.uint16, np.float32, np.float64), canonical=False):
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
    defaults = {"width": 640, "height": 480, "interpolation": "linear"}
    parameter_choices = {"interpolation": ("nearest", "linear", "area", "cubic")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        width = integer(self.parameters["width"], "Width", high=16384)
        height = integer(self.parameters["height"], "Height", high=16384)
        if width * height > 64_000_000:
            raise ValueError("Resize output is limited to 64 million pixels")
        method = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
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


class BoxBlur(ImageTool):
    type_id, title, subtype = "cv.box_blur", "Box blur", "Filtering"
    defaults = {"kernel": 5}

    def process(self, inputs):
        image = image_value(inputs["image"])
        kernel = integer(self.parameters["kernel"], "Kernel", high=255, odd=True)
        return encoded_result(cv2.blur(image, (kernel, kernel)), inputs["image"])


class MedianBlur(ImageTool):
    type_id, title, subtype = "cv.median_blur", "Median blur", "Filtering"
    defaults = {"kernel": 3}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.uint16, np.float32))
        kernel = integer(self.parameters["kernel"], "Kernel", low=3, high=255, odd=True)
        if image.dtype != np.uint8 and kernel > 5:
            raise ValueError("Median kernels above 5 require uint8")
        return encoded_result(cv2.medianBlur(image, kernel), inputs["image"])


class BilateralFilter(ImageTool):
    type_id, title, subtype = "cv.bilateral", "Bilateral filter", "Filtering"
    defaults = {"diameter": 5, "sigma_color": 50.0, "sigma_space": 5.0}

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8, np.float32))
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Bilateral filter accepts one or three channels")
        diameter = integer(self.parameters["diameter"], "Diameter", high=255)
        color = finite(self.parameters["sigma_color"], "Sigma color", 0)
        space = finite(self.parameters["sigma_space"], "Sigma space", 0)
        return encoded_result(cv2.bilateralFilter(image, diameter, color, space), inputs["image"])


class AutoThreshold(ImageTool):
    type_id, title, subtype = "cv.auto_threshold", "Automatic threshold", "Segmentation"
    outputs = (Port("image"), Port("threshold", "number"))
    defaults = {"method": "otsu", "invert": False}
    parameter_choices = {"method": ("otsu", "triangle")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.uint16))
        method = self.parameters["method"]
        if method == "triangle" and image.dtype != np.uint8:
            raise ValueError("Triangle threshold requires uint8")
        mode = {"otsu": cv2.THRESH_OTSU, "triangle": cv2.THRESH_TRIANGLE}[method]
        mode |= cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        level, mask = cv2.threshold(image, 0, int(np.iinfo(image.dtype).max), mode)
        return Result({"image": mask, "threshold": float(level)})


class AdaptiveThreshold(ImageTool):
    type_id, title, subtype = "cv.adaptive_threshold", "Adaptive threshold", "Segmentation"
    defaults = {"method": "gaussian", "block_size": 11, "constant": 2.0, "invert": False}
    parameter_choices = {"method": ("mean", "gaussian")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        size = integer(self.parameters["block_size"], "Block size", low=3, high=255, odd=True)
        method = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C, "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}[
            self.parameters["method"]
        ]
        mode = cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        output = cv2.adaptiveThreshold(
            image, 255, method, mode, size, finite(self.parameters["constant"], "Constant")
        )
        return Result({"image": output})


class InRange(ImageTool):
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


class Morphology(ImageTool):
    type_id, title, subtype = "cv.morphology", "Morphology", "Morphology"
    defaults = {"operation": "open", "shape": "ellipse", "kernel": 3, "iterations": 1}
    parameter_choices = {
        "operation": ("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"),
        "shape": ("rect", "ellipse", "cross"),
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
        output = cv2.morphologyEx(
            image, op, cv2.getStructuringElement(shape, (size, size)), iterations=iterations
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
    inputs = (Port("image_a"), Port("image_b", required=False))
    outputs = (Port("image"),)
    defaults = {"operation": "and"}
    parameter_choices = {"operation": ("and", "or", "xor", "not")}

    def process(self, inputs):
        a = image_value(inputs["image_a"], (np.uint8, np.uint16), canonical=True)
        op = self.parameters["operation"]
        if op == "not":
            output = cv2.bitwise_not(a)
        else:
            a, b = matching_images(inputs)
            output = {"and": cv2.bitwise_and, "or": cv2.bitwise_or, "xor": cv2.bitwise_xor}[op](
                a, b
            )
        return Result({"image": output})


class Blend(Block):
    type_id, title, category = "cv.blend", "Weighted blend", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (Port("image_a"), Port("image_b"))
    outputs = (Port("image"),)
    defaults = {"alpha": 0.5, "beta": 0.5, "gamma": 0.0}

    def process(self, inputs):
        a, b = matching_images(inputs)
        alpha, beta, gamma = (finite(self.parameters[k], k) for k in ("alpha", "beta", "gamma"))
        output = cv2.addWeighted(a, alpha, b, beta, gamma)
        if not np.isfinite(output).all():
            raise ValueError("Blend coefficients overflowed the output image depth")
        return Result({"image": output})


class Gradient(ImageTool):
    type_id, title, subtype = "cv.gradient", "Sobel / Scharr / Laplacian", "Edges"
    outputs = (Port("gradient"), Port("preview"))
    defaults = {"operator": "sobel", "dx": 1, "dy": 0, "kernel": 3, "scale": 1.0}
    parameter_choices = {"operator": ("sobel", "scharr", "laplacian")}

    def process(self, inputs):
        image = gray_image(inputs["image"])
        op = self.parameters["operator"]
        size = integer(self.parameters["kernel"], "Kernel", high=7, odd=True)
        scale = finite(self.parameters["scale"], "Scale")
        if op == "laplacian":
            gradient = cv2.Laplacian(image, cv2.CV_32F, ksize=size, scale=scale)
        else:
            dx = integer(self.parameters["dx"], "DX", 0, 2)
            dy = integer(self.parameters["dy"], "DY", 0, 2)
            if dx + dy == 0 or (op == "scharr" and dx + dy != 1):
                raise ValueError("Derivative order must be positive; Scharr requires DX+DY=1")
            if op == "scharr":
                gradient = cv2.Scharr(image, cv2.CV_32F, dx, dy, scale=scale)
            elif op == "sobel":
                gradient = cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=size, scale=scale)
            else:
                raise ValueError("Unknown gradient operator")
        if not np.isfinite(gradient).all():
            raise ValueError("Gradient overflowed float32 output")
        absolute = np.abs(gradient)
        preview = np.rint(absolute / max(float(absolute.max()), 1e-20) * 255).astype(np.uint8)
        return Result(
            {"gradient": gradient, "preview": preview}, graphics={"absolute gradient": preview}
        )


class ConnectedComponents(ImageTool):
    type_id, title, subtype = "inspect.components", "Connected components", "Shapes"
    category, block_type = "Inspection", "Inspection"
    outputs = (
        Port("overlay"),
        Port("mask"),
        Port("count", "number"),
        Port("centroids", "data"),
        Port("data", "data"),
    )
    defaults = {"connectivity": 8, "min_area": 100, "max_area": 0, "expected_count": 2}
    parameter_choices = {"connectivity": ("4", "8")}

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
        total, labels, stats, centers = cv2.connectedComponentsWithStats(
            (image != 0).astype(np.uint8), connectivity=connectivity
        )
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(image)
        components = []
        for label in range(1, total):
            x, y, width, height, area = map(int, stats[label])
            if area < minimum or (maximum and area > maximum):
                continue
            center = centers[label].tolist()
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
    inputs = (Port("image"), Port("template"))
    outputs = (
        Port("overlay"),
        Port("score", "number"),
        Port("location", "data"),
        Port("center", "data"),
    )
    defaults = {"method": "ccoeff_normed", "min_score": 0.8}
    parameter_choices = {"method": ("ccoeff_normed", "ccorr_normed", "sqdiff_normed")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        template = gray_image(inputs["template"], (np.uint8, np.float32))
        if image.dtype != template.dtype:
            raise ValueError("Image and template must have the same pixel depth")
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            raise ValueError("Template must fit inside the search image")
        method = self.parameters["method"]
        if method == "ccoeff_normed" and float(template.std()) < 1e-12:
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
        }[method]
        response = cv2.matchTemplate(image, template, code)
        if not np.isfinite(response).all():
            raise ValueError("Template matching produced a nonfinite response")
        low, high, low_pos, high_pos = cv2.minMaxLoc(response)
        location = low_pos if method == "sqdiff_normed" else high_pos
        score = float(np.clip(1 - low if method == "sqdiff_normed" else high, 0, 1))
        x, y = location
        height, width = template.shape
        overlay = cv2.cvtColor(preview_rgb(inputs["image"]), cv2.COLOR_RGB2BGR)
        cv2.rectangle(
            overlay,
            location,
            (x + width - 1, y + height - 1),
            (70, 220, 140) if score >= threshold else (60, 90, 240),
            2,
        )
        heat = 1 - response if method == "sqdiff_normed" else response
        heat = cv2.applyColorMap(
            np.rint(np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        return Result(
            {
                "overlay": overlay,
                "score": score,
                "location": {"top_left": [x, y], "bounds": [x, y, width, height], "method": method},
                "center": [[x + (width - 1) / 2, y + (height - 1) / 2]],
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
