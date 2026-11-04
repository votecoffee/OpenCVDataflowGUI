"""Additional contrast, channel, masking and geometric-feature OpenCV blocks."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format, preview_rgb
from .processing import (
    ImageTool,
    encoded_result,
    finite,
    gray_image,
    image_value,
    integer,
    matching_images,
)


def overlay_image(image):
    return cv2.cvtColor(preview_rgb(image), cv2.COLOR_RGB2BGR)


class EqualizeHistogram(ImageTool):
    type_id, title, subtype = "cv.equalize", "Equalize histogram", "Contrast"

    def process(self, inputs):
        return Result({"image": cv2.equalizeHist(gray_image(inputs["image"], (np.uint8,)))})


class CLAHE(ImageTool):
    type_id, title, subtype = "cv.clahe", "CLAHE local contrast", "Contrast"
    defaults = {"clip_limit": 2.0, "tiles_x": 8, "tiles_y": 8}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.uint16))
        clip = finite(self.parameters["clip_limit"], "Clip limit", 0)
        tiles = tuple(integer(self.parameters[k], k, 1, 128) for k in ("tiles_x", "tiles_y"))
        return Result({"image": cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(image)})


class Normalize(ImageTool):
    type_id, title, subtype = "cv.normalize", "Normalize intensity", "Contrast"
    defaults = {"minimum": 0.0, "maximum": 255.0, "pixel_depth": "uint8"}
    parameter_choices = {"pixel_depth": ("uint8", "uint16", "float32")}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        low, high = (finite(self.parameters[k], k) for k in ("minimum", "maximum"))
        if high <= low:
            raise ValueError("Maximum must be greater than minimum")
        depth = {"uint8": cv2.CV_8U, "uint16": cv2.CV_16U, "float32": cv2.CV_32F}[
            self.parameters["pixel_depth"]
        ]
        if image_format(source) in ("hsv", "lab"):
            raise ValueError("Convert HSV/Lab to grayscale or BGR before intensity normalization")
        output = cv2.normalize(image, None, low, high, cv2.NORM_MINMAX, dtype=depth)
        return encoded_result(output, source)


class GammaCorrection(ImageTool):
    type_id, title, subtype = "cv.gamma", "Gamma correction", "Contrast"
    defaults = {"gamma": 1.0}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8,))
        gamma = finite(self.parameters["gamma"], "Gamma")
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        if image_format(source) in ("hsv", "lab"):
            raise ValueError("Gamma correction requires grayscale, BGR or RGB color")
        table = np.rint(255 * (np.arange(256, dtype=float) / 255) ** (1 / gamma)).astype(np.uint8)
        output = cv2.LUT(image, table)
        if image.ndim == 3 and image.shape[2] == 4:
            output[..., 3] = image[..., 3]
        return encoded_result(output, source)


class SplitChannels(ImageTool):
    type_id, title, subtype = "cv.split_channels", "Split three channels", "Channels"
    outputs = (Port("channel_0"), Port("channel_1"), Port("channel_2"))

    def process(self, inputs):
        image = image_value(inputs["image"])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Split requires exactly three channels")
        planes = cv2.split(image)
        return Result(
            dict(zip((p.name for p in self.outputs), planes)),
            message=f"Channels in {image_format(inputs['image'])} order",
        )


class MergeChannels(Block):
    type_id, title, category = "cv.merge_channels", "Merge three channels", "Image processing"
    block_type, subtype = "Image processing", "Channels"
    inputs = (Port("channel_0"), Port("channel_1"), Port("channel_2"))
    outputs = (Port("image"),)
    defaults = {"color_format": "bgr"}
    parameter_choices = {"color_format": ("bgr", "rgb", "hsv", "lab")}

    def process(self, inputs):
        planes = [
            image_value(inputs[p.name], (np.uint8, np.uint16, np.float32)) for p in self.inputs
        ]
        if any(
            p.ndim != 2 or p.shape != planes[0].shape or p.dtype != planes[0].dtype for p in planes
        ):
            raise ValueError("Merge requires three monochrome planes with identical size and depth")
        encoding = self.parameters["color_format"]
        if encoding not in self.parameter_choices["color_format"]:
            raise ValueError("Unknown merged channel encoding")
        if encoding in ("hsv", "lab") and planes[0].dtype == np.uint16:
            raise ValueError("HSV/Lab channel encodings require uint8 or float32")
        return Result({"image": ImageArray(cv2.merge(planes), encoding)})


class ApplyMask(Block):
    type_id, title, category = "cv.apply_mask", "Apply mask", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (Port("image"), Port("mask"))
    outputs = (Port("image"),)

    def process(self, inputs):
        image = image_value(inputs["image"])
        mask = image_value(inputs["mask"], (np.uint8,))
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError("Mask must be uint8 monochrome with matching image dimensions")
        return encoded_result(cv2.copyTo(image, mask), inputs["image"])


class AbsoluteDifference(Block):
    type_id, title, category = "cv.absdiff", "Absolute difference", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (Port("image_a"), Port("image_b"))
    outputs = (Port("image"),)

    def process(self, inputs):
        a, b = matching_images(inputs)
        output = cv2.absdiff(a, b)
        if not np.isfinite(output).all():
            raise ValueError("Difference overflowed the output pixel depth")
        return Result({"image": output})


class GoodFeatures(ImageTool):
    type_id, title, subtype = "inspect.corners", "Good features / corners", "Features"
    category, block_type = "Inspection", "Inspection"
    outputs = (Port("points", "data"), Port("count", "number"), Port("overlay"))
    defaults = {
        "max_corners": 100,
        "quality": 0.01,
        "min_distance": 10.0,
        "block_size": 3,
        "use_harris": False,
        "harris_k": 0.04,
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        count = integer(self.parameters["max_corners"], "Maximum corners", 1, 10000)
        quality = finite(self.parameters["quality"], "Quality", 0)
        if not 0 < quality <= 1:
            raise ValueError("Quality must be greater than zero and at most one")
        distance = finite(self.parameters["min_distance"], "Minimum distance", 0)
        size = integer(self.parameters["block_size"], "Block size", 3, 255, odd=True)
        k = finite(self.parameters["harris_k"], "Harris K", 0)
        found = cv2.goodFeaturesToTrack(
            image,
            count,
            quality,
            distance,
            blockSize=size,
            useHarrisDetector=self.parameters["use_harris"],
            k=k,
        )
        points = found.reshape(-1, 2).tolist() if found is not None else []
        overlay = overlay_image(inputs["image"])
        for point in points:
            cv2.drawMarker(
                overlay, tuple(np.rint(point).astype(int)), (0, 220, 120), cv2.MARKER_CROSS, 8, 1
            )
        return Result(
            {"points": points, "count": len(points), "overlay": overlay},
            graphics={"corners": overlay},
            message=f"{len(points)} corners in input pixels",
        )


class HoughLines(ImageTool):
    type_id, title, subtype = "inspect.hough_lines", "Hough line segments", "Features"
    category, block_type = "Inspection", "Inspection"
    outputs = (Port("segments", "data"), Port("count", "number"), Port("overlay"))
    defaults = {"votes": 50, "min_length": 30.0, "max_gap": 10.0, "rho": 1.0, "theta_degrees": 1.0}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        votes = integer(self.parameters["votes"], "Votes", 1, 100000)
        rho = finite(self.parameters["rho"], "Rho", 0)
        theta = finite(self.parameters["theta_degrees"], "Theta", 0)
        if rho <= 0 or not 0 < theta <= 180:
            raise ValueError("Rho must be positive and theta in (0,180] degrees")
        found = cv2.HoughLinesP(
            image,
            rho,
            np.deg2rad(theta),
            votes,
            minLineLength=finite(self.parameters["min_length"], "Minimum length", 0),
            maxLineGap=finite(self.parameters["max_gap"], "Maximum gap", 0),
        )
        segments = found.reshape(-1, 4).tolist() if found is not None else []
        overlay = overlay_image(inputs["image"])
        for x1, y1, x2, y2 in segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 120), 1)
        return Result(
            {"segments": segments, "count": len(segments), "overlay": overlay},
            graphics={"lines": overlay},
            message=f"{len(segments)} line segments",
        )


class HoughCircles(ImageTool):
    type_id, title, subtype = "inspect.hough_circles", "Hough circles", "Features"
    category, block_type = "Inspection", "Inspection"
    outputs = (
        Port("circles", "data"),
        Port("centers", "data"),
        Port("count", "number"),
        Port("overlay"),
    )
    defaults = {
        "dp": 1.2,
        "min_distance": 20.0,
        "edge_threshold": 100.0,
        "center_threshold": 30.0,
        "min_radius": 0,
        "max_radius": 0,
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        dp = finite(p["dp"], "DP", 1)
        distance = finite(p["min_distance"], "Minimum distance", 0)
        edge = finite(p["edge_threshold"], "Edge threshold", 0)
        center = finite(p["center_threshold"], "Center threshold", 0)
        minimum = integer(p["min_radius"], "Minimum radius", 0, 100000)
        maximum = integer(p["max_radius"], "Maximum radius", 0, 100000)
        if min(distance, edge, center) <= 0 or (maximum and maximum < minimum):
            raise ValueError("Distances/thresholds must be positive and radius range ordered")
        found = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp,
            distance,
            param1=edge,
            param2=center,
            minRadius=minimum,
            maxRadius=maximum,
        )
        circles = found.reshape(-1, 3).tolist() if found is not None else []
        overlay = overlay_image(inputs["image"])
        for x, y, radius in circles:
            cv2.circle(overlay, (round(x), round(y)), round(radius), (0, 220, 120), 1)
        return Result(
            {
                "circles": circles,
                "centers": [c[:2] for c in circles],
                "count": len(circles),
                "overlay": overlay,
            },
            graphics={"circles": overlay},
            message=f"{len(circles)} circles; coordinates and radii in pixels",
        )


class DistanceTransform(ImageTool):
    type_id, title, subtype = "cv.distance_transform", "Distance transform", "Morphology"
    outputs = (Port("distance"), Port("preview"))
    defaults = {"mask_size": "5"}
    parameter_choices = {"mask_size": ("3", "5", "precise")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        if np.all(image != 0):
            raise ValueError("Distance transform needs background zero pixels")
        mask = {"3": 3, "5": 5, "precise": cv2.DIST_MASK_PRECISE}[self.parameters["mask_size"]]
        distance = cv2.distanceTransform(image, cv2.DIST_L2, mask)
        preview = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Result(
            {"distance": distance, "preview": preview},
            graphics={"distance": preview},
            message="Euclidean distance to background in pixels; preview separately scaled",
        )


FEATURE_BLOCKS = (
    EqualizeHistogram,
    CLAHE,
    Normalize,
    GammaCorrection,
    SplitChannels,
    MergeChannels,
    ApplyMask,
    AbsoluteDifference,
    GoodFeatures,
    HoughLines,
    HoughCircles,
    DistanceTransform,
)
