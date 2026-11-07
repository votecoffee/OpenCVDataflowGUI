"""OpenCV graph segmentation and ranked candidate rectangles."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import image_format
from .objects import collection
from .parameter_validation import NumericRange, ValidationError
from .processing import ImageTool, image_value
from .selection import SELECTION_INPUTS, contains, selection_mask


def segmentation_image(source):
    if image_format(source) not in ("gray", "bgr", "rgb"):
        raise ValidationError("Use uint8 grayscale, BGR or RGB images")
    return image_value(source, (np.uint8,), canonical=True)


class GraphSegments(ImageTool):
    type_id, title, subtype = "cv.graph_segments", "Graph-based segmentation", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("labels"), Port("boundaries"), Port("count", "number"))
    defaults = {"sigma": 0.5, "k": 300.0, "min_size": 100}
    parameter_ranges = {
        "sigma": NumericRange(0.01, 5),
        "k": NumericRange(0.01, 10_000),
        "min_size": NumericRange(1, 2_100_000, whole=True),
    }
    parameter_help = {
        "sigma": "Gaussian smoothing before computing neighboring pixel differences.",
        "k": "Region merge threshold scale. Larger values generally yield larger regions.",
        "min_size": "Native post-merge size threshold in pixels before selection.",
    }
    input_help = {
        "image": "Grayscale, BGR or RGB uint8 image with at most 2.1 million pixels.",
        "mask": "Optional output selection. Pixels outside it receive label zero.",
        "region": "Optional output selection. Native segmentation uses the full image.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if min(height, width) < 2 or height * width > 2_100_000:
            raise ValidationError("Graph segmentation requires 2 × 2 to 2.1 million pixels")
        image = segmentation_image(source)
        selection = selection_mask(inputs, source)
        p = self.parameters
        segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(
            float(p["sigma"]), float(p["k"]), int(p["min_size"]),
        )
        labels = segmenter.processImage(image)
        if labels is None or labels.shape != (height, width) or labels.dtype != np.int32:
            raise ValidationError("OpenCV returned an invalid label plane")
        labels += 1  # Native region zero is valid. Reserve zero for unselected pixels.
        if selection is not None:
            labels[selection == 0] = 0
        boundaries = np.zeros((height, width), np.uint8)
        horizontal = labels[:, :-1] != labels[:, 1:]
        vertical = labels[:-1, :] != labels[1:, :]
        boundaries[:, :-1][horizontal] = 255
        boundaries[:, 1:][horizontal] = 255
        boundaries[:-1, :][vertical] = 255
        boundaries[1:, :][vertical] = 255
        boundaries[labels == 0] = 0
        unique = np.unique(labels)
        count = int(unique.size - int(unique[0] == 0))
        return Result(
            {"labels": labels, "boundaries": boundaries, "count": count},
            message=f"{count} graph segments in the output selection",
        )


def box_item(rank, x0, y0, x1, y1):
    width, height = x1 - x0, y1 - y0
    return {
        "id": rank,
        "geometry": {
            "bounds": [x0, y0, width, height],
            "points": [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]],
        },
        "metrics": {
            "x": x0, "y": y0, "width": width, "height": height,
            "center_x": (x0 + x1 - 1) / 2, "center_y": (y0 + y1 - 1) / 2,
            "area": width * height, "aspect_ratio": width / height,
            "major_size": max(width, height), "minor_size": min(width, height),
        },
    }


class SelectiveSearchProposals(ImageTool):
    type_id, title = "inspect.selective_search", "Selective Search proposals"
    subtype = "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("collection", "data"), Port("count", "number"), Port("info", "data"))
    defaults = {
        "mode": "fast", "base_k": 150, "inc_k": 150, "sigma": 0.8,
        "working_side": 320, "max_proposals": 200, "preview_count": 25,
    }
    parameter_choices = {"mode": ("single", "fast", "quality")}
    parameter_ranges = {
        "base_k": NumericRange(1, 10_000, whole=True),
        "inc_k": NumericRange(1, 10_000, whole=True),
        "sigma": NumericRange(0.01, 5),
        "working_side": NumericRange(64, 640, whole=True),
        "max_proposals": NumericRange(1, 5000, whole=True),
        "preview_count": NumericRange(0, 250, whole=True),
    }
    parameter_help = {
        "mode": "Native preset. Quality evaluates more color spaces and strategies.",
        "base_k": "Initial graph segmentation threshold. Single uses only this value.",
        "inc_k": "Threshold increment between segmentations in Fast or Quality mode.",
        "sigma": "Gaussian smoothing used by the graph segmentations.",
        "working_side": "Downscale the longest side to this limit before generating proposals.",
        "max_proposals": "Maximum number of retained boxes after the selection filter.",
        "preview_count": "Number of retained boxes to draw. Zero disables the overlay.",
    }
    input_help = {
        "image": "Grayscale, BGR or RGB uint8 image. Output boxes use this image's grid.",
        "mask": "Keep proposals whose centers fall within this optional selection.",
        "region": "Keep proposals whose centers fall within this optional selection.",
    }
    process_inputs_readonly = True

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        if max(height, width) > 16384 or height * width > 64_000_000:
            raise ValidationError("Input exceeds the collection image size limits")
        p = self.parameters
        scale = min(1.0, int(p["working_side"]) / max(height, width))
        work_width, work_height = round(width * scale), round(height * scale)
        if min(work_height, work_width) < 8:
            raise ValidationError("The Selective Search working image needs at least 8 × 8 pixels")
        image = segmentation_image(source)
        selection = selection_mask(inputs, source)
        if scale < 1:
            image = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        search.setBaseImage(image)
        mode, k, step, sigma = p["mode"], int(p["base_k"]), int(p["inc_k"]), float(p["sigma"])
        if mode == "single":
            search.switchToSingleStrategy(k, sigma)
        elif mode == "fast":
            search.switchToSelectiveSearchFast(k, step, sigma)
        else:
            search.switchToSelectiveSearchQuality(k, step, sigma)
        rectangles = search.process()
        items, selected = [], 0
        limit = int(p["max_proposals"])
        for rank, (x, y, w, h) in enumerate(rectangles, 1):
            x, y, w, h = map(int, (x, y, w, h))
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > work_width or y + h > work_height:
                raise ValidationError("OpenCV returned an out-of-bounds proposal")
            x0, y0 = x * width // work_width, y * height // work_height
            x1 = ((x + w) * width + work_width - 1) // work_width
            y1 = ((y + h) * height + work_height - 1) // work_height
            center = ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2)
            if contains(selection, center):
                selected += 1
                if len(items) < limit:
                    items.append(box_item(rank, x0, y0, x1, y1))
        objects = ()
        if self.generate_graphics:
            objects = tuple(
                graphic_box(*item["geometry"]["bounds"], (55, 210, 255))
                for item in items[:int(p["preview_count"])]
            )
        info = {
            "mode": mode, "working_size": [work_width, work_height],
            "native_count": len(rectangles), "selected_count": selected,
            "truncated": len(items) < selected,
        }
        return Result(
            {
                "collection": collection("boxes", items, source, self.input_payloads["image"]),
                "count": len(items), "info": info,
            },
            drawings={"proposals": GraphicDrawing(objects)} if objects else {},
            message=f"{len(items)} retained candidate boxes",
        )
