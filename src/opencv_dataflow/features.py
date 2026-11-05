"""Additional contrast, channel, masking and geometric-feature OpenCV blocks."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box, graphic_circle, graphic_object
from .images import ImageArray, convert_depth, format_image, image_format, preview_rgb
from .objects import collection, line_item
from .parameter_validation import NumericRange, ValidationError
from .parameters import optional_mask
from .payloads import NO_VALUE, require_same_frame
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


class LuminanceContrast(RegionalOperation, ImageTool, ABC):
    defaults = {"output_format": "source"}
    parameter_choices = {"output_format": ("source", "gray")}
    parameter_help = {
        "output_format": "Source keeps channel encoding, depth and alpha; gray returns the "
        "processed monochrome plane at the working depth."
    }
    working_depths = (np.uint8,)

    @abstractmethod
    def enhance(self, image, inputs):
        """Apply the native operation to the working luminance plane."""

    def restore_extra(self, data):
        if "parameters" in data and "output_format" not in data["parameters"]:
            self.parameters["output_format"] = "gray"

    def process(self, inputs):
        source = inputs["image"]
        if self.parameters["output_format"] == "gray":
            work = gray_image(source, self.working_depths, auto_depth=True)
            return Result({"image": self.enhance(work, inputs)})
        native = image_value(source, (np.uint8, np.uint16, np.float32), canonical=True)
        ycc = cv2.cvtColor(native[..., :3], cv2.COLOR_BGR2YCrCb) if native.ndim == 3 else None
        work = native if ycc is None else ycc[..., 0]
        if work.dtype not in self.working_depths:
            work = convert_depth(work, np.dtype(self.working_depths[-1]).name)
        enhanced = self.enhance(work, inputs)
        if enhanced.dtype != native.dtype:
            enhanced = convert_depth(enhanced, native.dtype.name)
        if ycc is None:
            return encoded_result(enhanced, source)
        ycc[..., 0] = enhanced
        bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        if bgr.dtype == np.float32:
            np.clip(bgr, 0, 1, out=bgr)
        restored = format_image(bgr, image_format(source), "unchanged")
        if source.shape[2] == 4:
            restored[..., 3] = source[..., 3]
        return Result({"image": restored})


class EqualizeHistogram(LuminanceContrast):
    type_id, title, subtype = "cv.equalize", "Equalize histogram", "Contrast"

    def enhance(self, image, inputs):
        selection = selection_mask(inputs, inputs["image"])
        if selection is None:
            output = cv2.equalizeHist(image)
        else:
            output = image.copy()
            selected = selection != 0
            if selected.any():
                output[selected] = cv2.equalizeHist(image[selected].reshape(-1, 1)).reshape(-1)
        return output


class CLAHE(LuminanceContrast):
    type_id, title, subtype = "cv.clahe", "CLAHE local contrast", "Contrast"
    defaults = {**LuminanceContrast.defaults, "clip_limit": 2.0, "tiles_x": 8, "tiles_y": 8}
    working_depths = (np.uint8, np.uint16)

    def enhance(self, image, inputs):
        clip = finite(self.parameters["clip_limit"], "Clip limit", 0)
        tiles = tuple(integer(self.parameters[k], k, 1, 128) for k in ("tiles_x", "tiles_y"))
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(image)


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
    parameter_ranges = {
        "max_corners": NumericRange(0, 10000, whole=True),
        "quality": NumericRange(0, 1),
        "min_distance": NumericRange(0),
        "block_size": NumericRange(3, 255, whole=True, odd=True),
        "harris_k": NumericRange(0),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["quality"]) <= 0:
            raise ValidationError("Quality must be greater than zero and at most one", "quality")

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8, np.float32))
        p = self.parameters
        count, size = int(p["max_corners"]), int(p["block_size"])
        quality, distance, k = (float(p[key]) for key in ("quality", "min_distance", "harris_k"))
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
            drawings={"corners": GraphicDrawing(tuple(
                graphic_object("marker", [point], (0, 220, 120)) for point in points
            ))} if self.generate_graphics and points else {},
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
    parameter_ranges = {
        "votes": NumericRange(1, 100000, whole=True),
        "min_length": NumericRange(0),
        "max_gap": NumericRange(0),
        "rho": NumericRange(0),
        "theta_degrees": NumericRange(0, 180),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("rho", "theta_degrees"):
            if float(p[name]) <= 0:
                label = name.replace("_", " ").capitalize()
                raise ValidationError(f"{label} must be positive", name)

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        votes, rho, theta = int(p["votes"]), float(p["rho"]), float(p["theta_degrees"])
        found = cv2.HoughLinesP(
            image,
            rho,
            np.deg2rad(theta),
            votes,
            minLineLength=float(p["min_length"]),
            maxLineGap=float(p["max_gap"]),
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
            drawings={"lines": GraphicDrawing(tuple(
                graphic_object("polyline", (s[:2], s[2:]), (0, 220, 120), thickness=1)
                for s in segments
            ))} if self.generate_graphics and segments else {},
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
    parameter_ranges = {
        "dp": NumericRange(1),
        "min_distance": NumericRange(0),
        "edge_threshold": NumericRange(0),
        "center_threshold": NumericRange(0),
        "min_radius": NumericRange(0, 100000, whole=True),
        "max_radius": NumericRange(-1, 100000, whole=True),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("min_distance", "edge_threshold", "center_threshold"):
            if float(p[key]) <= 0:
                raise ValidationError(f"{key.replace('_', ' ').capitalize()} must be positive", key)
        if 0 < p["max_radius"] < p["min_radius"]:
            raise ValidationError("Radius range must be ordered", "min_radius", "max_radius")
        if p["method"] == "gradient_alt":
            if float(p["center_threshold"]) >= 1:
                raise ValidationError("Gradient ALT needs center threshold <1", "center_threshold")
            if p["max_radius"] < 0:
                raise ValidationError("Gradient ALT needs nonnegative max radius", "max_radius")

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        dp, distance, edge, center = (
            float(p[key]) for key in ("dp", "min_distance", "edge_threshold", "center_threshold")
        )
        minimum, maximum = int(p["min_radius"]), int(p["max_radius"])
        mode = {"gradient": cv2.HOUGH_GRADIENT, "gradient_alt": cv2.HOUGH_GRADIENT_ALT}[p["method"]]
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
            drawings={"circles": GraphicDrawing(tuple(
                graphic_circle(x, y, radius, (0, 220, 120)) for x, y, radius in circles
            ))} if self.generate_graphics and circles else {},
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


class MSERRegions(Block):
    type_id, title = "inspect.mser", "Detect MSER regions"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("regions", "data"),
        Port("count", "number"),
        Port("mask", visual_type="region"),
        Port("overlay", required=False),
    )
    defaults = {
        "mode": "grayscale",
        "polarity": "both",
        "delta": 5,
        "min_area": 60,
        "max_area": 14400,
        "max_variation": 0.25,
        "min_diversity": 0.2,
        "max_evolution": 200,
        "area_threshold": 1.01,
        "min_margin": 0.003,
        "edge_blur_size": 5,
        "max_regions": 1000,
        "draw_overlay": True,
    }
    parameter_choices = {"mode": ("grayscale", "color"), "polarity": ("both", "bright", "dark")}
    parameter_ranges = {
        "delta": NumericRange(1, 255, whole=True),
        "min_area": NumericRange(1, 1_000_000, whole=True),
        "max_area": NumericRange(1, 1_000_000, whole=True),
        "max_variation": NumericRange(0, 10),
        "min_diversity": NumericRange(0, 1),
        "max_evolution": NumericRange(1, 10000, whole=True),
        "area_threshold": NumericRange(1, 100),
        "min_margin": NumericRange(0, 1),
        "edge_blur_size": NumericRange(1, 31, whole=True, odd=True, zero_allowed=True),
        "max_regions": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        "polarity": "Grayscale only. Detect bright, dark, or both kinds of extremal region.",
        "delta": "Intensity interval used to measure region stability.",
        "min_diversity": "Color mode minimum diversity between nested regions.",
        "max_evolution": "Color mode maximum evolution steps.",
        "area_threshold": "Color mode reinitialization area ratio.",
        "min_margin": "Color mode minimum stability margin.",
        "edge_blur_size": "Color mode edge blur aperture. Zero disables it.",
        "max_regions": "Maximum number of retained regions in native detector order.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["min_area"] > p["max_area"]:
            raise ValidationError("Minimum area cannot exceed maximum area", "min_area", "max_area")

    def process(self, inputs):
        source = inputs["image"]
        if min(source.shape[:2]) < 3:
            raise ValidationError("MSER needs an image at least 3 × 3 pixels")
        p = self.parameters
        if p["mode"] == "color":
            canonical = image_value(source, (np.uint8, np.uint16, np.float32), canonical=True)
            search = format_image(canonical, "bgr", "uint8")
        else:
            search = gray_image(source, (np.uint8,), auto_depth=True)
            if p["polarity"] == "dark":
                search = cv2.bitwise_not(search)
        detector = cv2.MSER_create(
            int(p["delta"]),
            int(p["min_area"]),
            int(p["max_area"]),
            float(p["max_variation"]),
            float(p["min_diversity"]),
            int(p["max_evolution"]),
            float(p["area_threshold"]),
            float(p["min_margin"]),
            int(p["edge_blur_size"]),
        )
        detector.setPass2Only(p["mode"] == "grayscale" and p["polarity"] != "both")
        selection = selection_mask(inputs, source)
        regions, boxes = detector.detectRegions(search)
        mask = np.zeros(source.shape[:2], np.uint8)
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        draw = self.generate_graphics
        items, objects = [], []
        truncated = False
        for index, (region, box) in enumerate(zip(regions, boxes, strict=True)):
            xs, ys = region.T
            if selection is not None and not selection[ys, xs].all():
                continue
            if len(items) >= int(p["max_regions"]):
                truncated = True
                break
            x, y, width, height = map(int, box)
            items.append(
                {
                    "id": index,
                    "bounds": [x, y, width, height],
                    "area": len(region),
                    "center": region.mean(axis=0).tolist(),
                }
            )
            mask[ys, xs] = 255
            color = (70, 220, 140)
            if overlay is not NO_VALUE:
                cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, 2)
            if draw:
                objects.append(graphic_box(x, y, width, height, color))
        count = len(items)
        return Result(
            {
                "regions": {"model": "mser_regions_v1", "items": items, "truncated": truncated},
                "count": count,
                "mask": mask,
                "overlay": overlay,
            },
            graphics={"MSER regions": overlay} if draw and overlay is not NO_VALUE else {},
            drawings={"MSER regions": GraphicDrawing(tuple(objects))} if objects else {},
            message=f"{count} MSER regions" + (" · region limit reached" if truncated else ""),
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
    MSERRegions,
    HoughLines,
    HoughCircles,
    DistanceTransform,
)
