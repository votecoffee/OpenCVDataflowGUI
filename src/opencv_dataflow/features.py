"""Additional contrast, channel, masking and geometric-feature OpenCV blocks."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format, preview_rgb
from .objects import collection, line_item
from .parameters import optional_mask
from .payloads import require_same_frame
from .processing import (
    ImageTool,
    encoded_result,
    finite,
    gray_image,
    image_value,
    integer,
    matching_images,
)
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask


def overlay_image(image):
    return cv2.cvtColor(preview_rgb(image), cv2.COLOR_RGB2BGR)


class EqualizeHistogram(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.equalize", "Equalize histogram", "Contrast"

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        selection = selection_mask(inputs, image)
        if selection is None:
            output = cv2.equalizeHist(image)
        else:
            output = image.copy()
            selected = selection != 0
            if selected.any():
                output[selected] = cv2.equalizeHist(image[selected].reshape(-1, 1)).reshape(-1)
        return Result({"image": output})


class CLAHE(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.clahe", "CLAHE local contrast", "Contrast"
    defaults = {"clip_limit": 2.0, "tiles_x": 8, "tiles_y": 8}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.uint16))
        clip = finite(self.parameters["clip_limit"], "Clip limit", 0)
        tiles = tuple(integer(self.parameters[k], k, 1, 128) for k in ("tiles_x", "tiles_y"))
        return Result({"image": cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(image)})


class Normalize(ImageTool):
    type_id, title, subtype = "cv.normalize", "Normalize intensity", "Contrast"
    inputs = SELECTION_INPUTS
    defaults = {
        "minimum": 0.0,
        "maximum": 255.0,
        "pixel_depth": "uint8",
        "norm_type": "minmax",
        "norm_value": 1.0,
    }
    parameter_choices = {
        "pixel_depth": ("uint8", "uint16", "int16", "float32", "float64"),
        "norm_type": ("minmax", "l1", "l2", "inf"),
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        low, high = (finite(self.parameters[k], k) for k in ("minimum", "maximum"))
        if high <= low:
            raise ValueError("Maximum must be greater than minimum")
        depth = {
            "uint8": cv2.CV_8U,
            "uint16": cv2.CV_16U,
            "int16": cv2.CV_16S,
            "float32": cv2.CV_32F,
            "float64": cv2.CV_64F,
        }[self.parameters["pixel_depth"]]
        if image_format(source) in ("hsv", "lab"):
            raise ValueError("Convert HSV/Lab to grayscale or BGR before intensity normalization")
        mask = optional_mask(inputs, image)
        if mask is not None and image.ndim != 2:
            raise ValueError("Masked normalization requires monochrome input")
        norm = self.parameters["norm_type"]
        alpha, beta = (
            (low, high)
            if norm == "minmax"
            else (finite(self.parameters["norm_value"], "Norm value", 0), 0)
        )
        mode = {
            "minmax": cv2.NORM_MINMAX,
            "l1": cv2.NORM_L1,
            "l2": cv2.NORM_L2,
            "inf": cv2.NORM_INF,
        }[norm]
        output = cv2.normalize(image, None, alpha, beta, mode, dtype=depth, mask=mask)
        return encoded_result(output, source)


class GammaCorrection(RegionalOperation, ImageTool):
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
            payloads={
                p.name: {
                    **self.input_payloads["image"],
                    "channel": {"index": index, "source_format": image_format(inputs["image"])},
                }
                for index, p in enumerate(self.outputs)
            },
        )


class MergeChannels(Block):
    type_id, title, category = "cv.merge_channels", "Merge three channels", "Image processing"
    block_type, subtype = "Image processing", "Channels"
    inputs = (Port("channel_0"), Port("channel_1"), Port("channel_2"))
    outputs = (Port("image"),)
    defaults = {"color_format": "bgr"}
    parameter_choices = {"color_format": ("bgr", "rgb", "hsv", "lab")}

    def process(self, inputs):
        require_same_frame([inputs[p.name] for p in self.inputs])
        planes = [image_value(inputs[p.name]) for p in self.inputs]
        if any(
            p.ndim != 2 or p.shape != planes[0].shape or p.dtype != planes[0].dtype for p in planes
        ):
            raise ValueError("Merge requires monochrome planes with identical size and depth")
        encoding = self.parameters["color_format"]
        if encoding not in self.parameter_choices["color_format"]:
            raise ValueError("Unknown merged channel encoding")
        if encoding in ("hsv", "lab") and planes[0].dtype not in (np.uint8, np.float32):
            raise ValueError("HSV/Lab channel encodings require uint8 or float32")
        payload = dict(self.input_payloads.get("channel_0", {}))
        payload.pop("channel", None)
        return Result(
            {"image": ImageArray(cv2.merge(planes), encoding)}, payloads={"image": payload}
        )


class SplitAlphaChannels(ImageTool):
    type_id, title, subtype = "cv.split_alpha_channels", "Split four channels (alpha)", "Channels"
    outputs = (Port("channel_0"), Port("channel_1"), Port("channel_2"), Port("alpha"))

    def process(self, inputs):
        image = image_value(inputs["image"])
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("Alpha split requires exactly four channels")
        return Result(
            dict(zip((p.name for p in self.outputs), cv2.split(image))),
            message=f"Channels in {image_format(inputs['image'])} order",
            payloads={
                p.name: {
                    **self.input_payloads["image"],
                    "channel": {"index": index, "source_format": image_format(inputs["image"])},
                }
                for index, p in enumerate(self.outputs)
            },
        )


class MergeAlphaChannels(MergeChannels):
    type_id, title = "cv.merge_alpha_channels", "Merge four channels (alpha)"
    inputs = (*MergeChannels.inputs, Port("alpha"))
    defaults = {"color_format": "bgra"}
    parameter_choices = {"color_format": ("bgra", "rgba")}


class ExtractChannel(ImageTool):
    type_id, title, subtype = "cv.extract_channel", "Extract image channel", "Channels"
    defaults = {"channel": "0"}
    parameter_choices = {"channel": ("0", "1", "2", "3")}

    def process(self, inputs):
        image = image_value(inputs["image"])
        count = image.shape[2] if image.ndim == 3 else 1
        index = int(self.parameters["channel"])
        if index >= count:
            raise ValueError(f"Image has no channel {index}")
        output = image.copy() if image.ndim == 2 else cv2.extractChannel(image, index)
        return Result(
            {"image": output},
            payloads={
                "image": {
                    **self.input_payloads["image"],
                    "channel": {"index": index, "source_format": image_format(inputs["image"])},
                }
            },
        )


class InsertChannel(ImageTool):
    type_id, title, subtype = "cv.insert_channel", "Replace image channel", "Channels"
    inputs = (Port("image"), Port("channel"))
    defaults = ExtractChannel.defaults
    parameter_choices = ExtractChannel.parameter_choices

    def process(self, inputs):
        require_same_frame([inputs["image"], inputs["channel"]])
        image, channel = (image_value(inputs[k]) for k in ("image", "channel"))
        count = image.shape[2] if image.ndim == 3 else 1
        index = int(self.parameters["channel"])
        if (
            index >= count
            or channel.ndim != 2
            or channel.shape != image.shape[:2]
            or channel.dtype != image.dtype
        ):
            raise ValueError(
                "Replacement requires a valid channel index and matching monochrome size/depth"
            )
        output = image.copy()
        if output.ndim == 2:
            output[:] = channel
        else:
            cv2.insertChannel(channel, output, index)
        return encoded_result(output, inputs["image"])


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


class AbsoluteDifference(RegionalOperation, Block):
    region_source = "image_a"
    region_background = "zero"
    type_id, title, category = "cv.absdiff", "Absolute difference", "Image processing"
    block_type, subtype = "Image processing", "Image math"
    inputs = (
        Port("image_a"),
        Port("image_b"),
        Port("mask", required=False),
        Port("region", "data", required=False),
    )
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
    inputs = SELECTION_INPUTS
    outputs = (Port("points", "data"), Port("count", "number"), Port("overlay"))
    defaults = {
        "max_corners": 100,
        "quality": 0.01,
        "min_distance": 10.0,
        "block_size": 3,
        "use_harris": False,
        "harris_k": 0.04,
        "gradient_size": "3",
    }
    parameter_choices = {"gradient_size": ("3", "5", "7")}

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        count = integer(self.parameters["max_corners"], "Maximum corners", 0, 10000)
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
            gradientSize=int(self.parameters["gradient_size"]),
            mask=optional_mask(inputs, image),
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
    inputs = SELECTION_INPUTS
    type_id, title, subtype = "inspect.hough_lines", "Hough line segments", "Features"
    category, block_type = "Inspection", "Inspection"
    outputs = (
        Port("segments", "data"),
        Port("count", "number"),
        Port("overlay"),
        Port("collection", "data"),
    )
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
        selection = selection_mask(inputs, image)
        segments = [
            s for s in segments if contains(selection, s[:2]) and contains(selection, s[2:])
        ]
        overlay = overlay_image(inputs["image"])
        for x1, y1, x2, y2 in segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 120), 1)
        return Result(
            {
                "segments": segments,
                "count": len(segments),
                "overlay": overlay,
                "collection": collection(
                    "lines",
                    [line_item(s, i) for i, s in enumerate(segments)],
                    image,
                    self.input_payloads["image"],
                ),
            },
            graphics={"lines": overlay},
            message=f"{len(segments)} line segments",
        )


class HoughCircles(ImageTool):
    inputs = SELECTION_INPUTS
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
        "method": "gradient",
    }
    parameter_choices = {"method": ("gradient", "gradient_alt")}
    parameter_help = {
        "center_threshold": "For gradient_alt use a perfectness threshold below 1 (e.g. 0.9)."
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        dp = finite(p["dp"], "DP", 1)
        distance = finite(p["min_distance"], "Minimum distance", 0)
        edge = finite(p["edge_threshold"], "Edge threshold", 0)
        center = finite(p["center_threshold"], "Center threshold", 0)
        minimum = integer(p["min_radius"], "Minimum radius", 0, 100000)
        maximum = integer(p["max_radius"], "Maximum radius", -1, 100000)
        if min(distance, edge, center) <= 0 or (maximum > 0 and maximum < minimum):
            raise ValueError("Distances/thresholds must be positive and radius range ordered")
        mode = {"gradient": cv2.HOUGH_GRADIENT, "gradient_alt": cv2.HOUGH_GRADIENT_ALT}[p["method"]]
        if p["method"] == "gradient_alt" and (center >= 1 or maximum < 0):
            raise ValueError("Gradient ALT needs center threshold <1 and nonnegative max radius")
        found = cv2.HoughCircles(
            image,
            mode,
            dp,
            distance,
            param1=edge,
            param2=center,
            minRadius=minimum,
            maxRadius=maximum,
        )
        circles = found.reshape(-1, 3).tolist() if found is not None else []
        selection = selection_mask(inputs, image)
        circles = [c for c in circles if contains(selection, c[:2])]
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
    defaults = {"mask_size": "5", "distance_type": "l2", "output_depth": "float32"}
    parameter_choices = {
        "mask_size": ("3", "5", "precise"),
        "distance_type": ("l1", "l2", "chessboard"),
        "output_depth": ("float32", "uint8"),
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        if np.all(image != 0):
            raise ValueError("Distance transform needs background zero pixels")
        mask = {"3": 3, "5": 5, "precise": cv2.DIST_MASK_PRECISE}[self.parameters["mask_size"]]
        mode = {"l1": cv2.DIST_L1, "l2": cv2.DIST_L2, "chessboard": cv2.DIST_C}[
            self.parameters["distance_type"]
        ]
        depth = cv2.CV_8U if self.parameters["output_depth"] == "uint8" else cv2.CV_32F
        if depth == cv2.CV_8U and mode != cv2.DIST_L1:
            raise ValueError("uint8 distance output is supported only for L1")
        distance = cv2.distanceTransform(image, mode, mask, dstType=depth)
        preview = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Result(
            {"distance": distance, "preview": preview},
            graphics={"distance": preview},
            message=f"{self.parameters['distance_type']} pixel distances; separate display scale",
        )


FEATURE_BLOCKS = (
    EqualizeHistogram,
    CLAHE,
    Normalize,
    GammaCorrection,
    SplitChannels,
    MergeChannels,
    SplitAlphaChannels,
    MergeAlphaChannels,
    ExtractChannel,
    InsertChannel,
    ApplyMask,
    AbsoluteDifference,
    GoodFeatures,
    HoughLines,
    HoughCircles,
    DistanceTransform,
)
