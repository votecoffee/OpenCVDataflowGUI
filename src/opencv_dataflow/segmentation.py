"""Seeded segmentation in the input image's coordinate frame."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .payloads import NO_VALUE, require_same_frame
from .processing import finite, image_value, integer
from .selection import SELECTION_INPUTS, selection_mask


def bgr8(source):
    image = image_value(source, (np.uint8,), canonical=True)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class FloodFill(Block):
    type_id, title = "inspect.flood_fill", "Seeded flood fill"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("mask"), Port("area", "number"), Port("bounds", "data"))
    defaults = {
        "seed_x": 0,
        "seed_y": 0,
        "lower_difference": 0.0,
        "upper_difference": 0.0,
        "connectivity": "4",
        "range_mode": "fixed",
    }
    parameter_choices = {"connectivity": ("4", "8"), "range_mode": ("fixed", "floating")}
    parameter_help = {
        "lower_difference": "Nonnegative tolerance below the reference value. Number or BGR list.",
        "upper_difference": "Nonnegative tolerance above the reference value. Number or BGR list.",
        "range_mode": "Compare with the seed (fixed) or an accepted adjacent pixel (floating).",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.float32), canonical=True)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        x = integer(self.parameters["seed_x"], "Seed x", 0, width - 1)
        y = integer(self.parameters["seed_y"], "Seed y", 0, height - 1)
        channels = 1 if image.ndim == 2 else 3
        tolerances = []
        for key in ("lower_difference", "upper_difference"):
            values = np.asarray(self.parameters[key], dtype=np.float64).reshape(-1)
            if values.size == 1:
                values = np.repeat(values, channels)
            if values.size != channels or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{key} needs one or {channels} finite nonnegative values")
            tolerances.append(tuple(values))
        selection = selection_mask(inputs, image)
        mask = np.zeros((height + 2, width + 2), np.uint8)
        interior = mask[1:-1, 1:-1]
        if selection is not None:
            interior[selection == 0] = 1
        flags = int(self.parameters["connectivity"]) | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        if self.parameters["range_mode"] == "fixed":
            flags |= cv2.FLOODFILL_FIXED_RANGE
        area, _, _, bounds = cv2.floodFill(
            np.ascontiguousarray(image), mask, (x, y), 0, *tolerances, flags
        )
        filled = cv2.compare(interior, 255, cv2.CMP_EQ)
        return Result({"mask": filled, "area": area, "bounds": list(bounds)})


class Watershed(Block):
    type_id, title = "inspect.watershed", "Marker-controlled watershed"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (Port("image"), Port("markers"))
    outputs = (Port("labels"), Port("boundaries"), Port("mask"), Port("count", "number"))
    defaults = {
        "marker_mode": "binary_components",
        "connectivity": "8",
        "output_label": 0,
        "background_label": 0,
    }
    parameter_choices = {"marker_mode": ("binary_components", "labels"), "connectivity": ("4", "8")}
    parameter_help = {
        "marker_mode": "Label connected uint8 seeds or use an int32 map with 0 for unknown.",
        "connectivity": "Connectivity used to label binary seeds.",
        "output_label": "Label to output as a mask. Zero includes all foreground labels.",
        "background_label": "Label excluded from the count and default mask. Zero disables this.",
    }

    def process(self, inputs):
        image = bgr8(inputs["image"])
        markers = inputs["markers"]
        require_same_frame((inputs["image"], markers))
        if markers.shape != image.shape[:2]:
            raise ValueError("Watershed markers must be monochrome and match the image size")
        if self.parameters["marker_mode"] == "binary_components":
            if markers.dtype != np.uint8:
                raise ValueError("Binary watershed markers must be uint8")
            _, labels = cv2.connectedComponentsWithAlgorithm(
                cv2.compare(markers, 0, cv2.CMP_NE),
                int(self.parameters["connectivity"]),
                cv2.CV_32S,
                cv2.CCL_SAUF,
            )
        else:
            if markers.dtype != np.int32 or (markers < 0).any():
                raise ValueError("Watershed label markers must be nonnegative int32")
            labels = np.array(markers, copy=True, order="C")
        if not np.any(labels > 0):
            raise ValueError("Watershed requires at least one positive seed")
        labels = cv2.watershed(image, labels)
        boundaries = cv2.compare(labels, -1, cv2.CMP_EQ)
        background = integer(self.parameters["background_label"], "Background label", 0, 2**31 - 1)
        unique = np.unique(labels)
        count = int(np.count_nonzero((unique > 0) & (unique != background)))
        output_label = integer(self.parameters["output_label"], "Output label", 0, 2**31 - 1)
        mask = cv2.compare(labels, output_label, cv2.CMP_EQ if output_label else cv2.CMP_GT)
        if background and not output_label:
            mask[labels == background] = 0
        return Result({"labels": labels, "boundaries": boundaries, "mask": mask, "count": count})


class GrabCut(Block):
    type_id, title = "inspect.grabcut", "GrabCut foreground"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (Port("image"), Port("markers", required=False))
    outputs = (Port("mask"), Port("labels"))
    defaults = {"mode": "rectangle", "rectangle": [10, 10, 100, 100], "iterations": 3}
    parameter_choices = {"mode": ("rectangle", "binary_mask", "labels")}
    parameter_help = {
        "rectangle": "Initial foreground rectangle [x, y, width, height] in image pixels.",
        "mode": "Use a rectangle, a probable-foreground binary mask, or GrabCut labels 0–3.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = bgr8(source)
        height, width = image.shape[:2]
        mode = self.parameters["mode"]
        iterations = integer(self.parameters["iterations"], "Iterations", 1, 100)
        rect = None
        markers = np.zeros((height, width), np.uint8)
        flag = cv2.GC_INIT_WITH_RECT
        if mode == "rectangle":
            x, y, w, h = self.parameters["rectangle"]
            x = integer(x, "Rectangle x", 0, width - 1)
            y = integer(y, "Rectangle y", 0, height - 1)
            w = integer(w, "Rectangle width", 1, width - x)
            h = integer(h, "Rectangle height", 1, height - y)
            if w * h == width * height:
                raise ValueError("GrabCut rectangle must leave some background pixels")
            rect = x, y, w, h
        else:
            supplied = inputs.get("markers")
            if supplied is None or supplied.shape != markers.shape or supplied.dtype != np.uint8:
                raise ValueError("GrabCut requires a matching uint8 monochrome marker image")
            require_same_frame((source, supplied))
            if mode == "binary_mask":
                markers[supplied != 0] = cv2.GC_PR_FGD
            else:
                if supplied.max() > cv2.GC_PR_FGD:
                    raise ValueError("GrabCut marker labels must be 0, 1, 2, or 3")
                markers = np.array(supplied, copy=True, order="C")
            foreground = (markers & 1) != 0
            if not foreground.any() or foreground.all():
                raise ValueError("GrabCut requires foreground and background samples")
            flag = cv2.GC_INIT_WITH_MASK
        labels, _, _ = cv2.grabCut(image, markers, rect, None, None, iterations, flag)
        return Result({"mask": (labels & 1) * np.uint8(255), "labels": labels})


class ThinMask(Block):
    type_id, title = "inspect.thin_mask", "Thin binary mask"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("mask"),)
    defaults = {"method": "zhang_suen"}
    parameter_choices = {"method": ("zhang_suen", "guo_hall")}

    def process(self, inputs):
        source = inputs["image"]
        if source.ndim != 2 or source.dtype != np.uint8:
            raise ValueError("Thinning requires a uint8 monochrome mask")
        mask = cv2.compare(np.asarray(source), 0, cv2.CMP_NE)
        selection = selection_mask(inputs, mask)
        if selection is not None:
            mask = cv2.bitwise_and(mask, selection)
        method = {
            "zhang_suen": cv2.ximgproc.THINNING_ZHANGSUEN,
            "guo_hall": cv2.ximgproc.THINNING_GUOHALL,
        }[self.parameters["method"]]
        # Padding gives components that touch the image edge a background border.
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        result = cv2.ximgproc.thinning(padded, thinningType=method)[1:-1, 1:-1].copy()
        return Result({"mask": result})


class FillMaskHoles(Block):
    type_id, title = "inspect.fill_mask_holes", "Fill holes in a binary mask"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (Port("image"),)
    outputs = (Port("mask"), Port("holes"), Port("count", "number"))
    defaults = {"background_connectivity": "4", "maximum_area": 0}
    parameter_choices = {"background_connectivity": ("4", "8")}
    parameter_help = {
        "maximum_area": "Largest hole to fill, in pixels. Zero fills all enclosed holes.",
        "background_connectivity": "Connectivity of background paths to the image edge.",
    }

    def process(self, inputs):
        source = inputs["image"]
        if source.ndim != 2 or source.dtype != np.uint8:
            raise ValueError("Hole fill requires a uint8 monochrome mask")
        background = cv2.compare(np.asarray(source), 0, cv2.CMP_EQ)
        maximum = integer(self.parameters["maximum_area"], "Maximum area", 0, 2**31 - 1)
        count, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
            background, int(self.parameters["background_connectivity"]), cv2.CV_32S, cv2.CCL_SAUF
        )
        enclosed = np.ones(count, dtype=bool)
        enclosed[0] = False
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            enclosed[edge] = False
        if maximum:
            enclosed &= stats[:, cv2.CC_STAT_AREA] <= maximum
        holes = enclosed[labels]
        return Result(
            {
                "mask": ((background == 0) | holes).astype(np.uint8) * 255,
                "holes": holes.astype(np.uint8) * 255,
                "count": int(enclosed.sum()),
            }
        )


class DistancePeakMarkers(Block):
    type_id, title = "inspect.distance_markers", "Watershed seeds from distance peaks"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("markers"), Port("seeds"), Port("distance"), Port("count", "number"))
    defaults = {
        "distance_type": "l2",
        "l2_mask_size": "precise",
        "minimum_distance": 2.0,
        "peak_radius": 3,
        "background_gap": 1,
        "connectivity": "8",
    }
    parameter_choices = {
        "distance_type": ("l2", "l1", "chessboard"),
        "l2_mask_size": ("precise", "3", "5"),
        "connectivity": ("4", "8"),
    }
    parameter_help = {
        "minimum_distance": "Minimum distance from a peak to the background, in pixels.",
        "peak_radius": "Compare each peak with neighbors in a square of this radius.",
        "background_gap": "Number of pixel layers to leave unlabeled outside the foreground.",
        "connectivity": "Connectivity used to group adjacent peak pixels into one seed.",
    }

    def process(self, inputs):
        source = inputs["image"]
        if source.ndim != 2 or source.dtype != np.uint8:
            raise ValueError("Distance seeds require a uint8 monochrome foreground mask")
        foreground = cv2.compare(np.asarray(source), 0, cv2.CMP_NE)
        selection = selection_mask(inputs, foreground)
        if selection is not None:
            foreground = cv2.bitwise_and(foreground, selection)
        distance_type = self.parameters["distance_type"]
        mode = {"l1": cv2.DIST_L1, "l2": cv2.DIST_L2, "chessboard": cv2.DIST_C}[distance_type]
        mask_size = (
            {"3": 3, "5": 5, "precise": cv2.DIST_MASK_PRECISE}[self.parameters["l2_mask_size"]]
            if distance_type == "l2"
            else 3
        )
        padded = cv2.copyMakeBorder(foreground, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(padded, mode, mask_size)[1:-1, 1:-1].copy()
        radius = integer(self.parameters["peak_radius"], "Peak radius", 1, 63)
        threshold = finite(self.parameters["minimum_distance"], "Minimum distance", 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1,) * 2)
        local_maximum = cv2.dilate(distance, kernel)
        peaks = (foreground != 0) & (distance >= threshold) & (distance == local_maximum)
        seeds = peaks.astype(np.uint8) * 255
        count, labels = cv2.connectedComponentsWithAlgorithm(
            seeds, int(self.parameters["connectivity"]), cv2.CV_32S, cv2.CCL_SAUF
        )
        gap = integer(self.parameters["background_gap"], "Background gap", 0, 255)
        expanded = cv2.dilate(foreground, None, iterations=gap) if gap else foreground
        markers = np.where(peaks, labels + 1, expanded == 0).astype(np.int32)
        return Result(
            {"markers": markers, "seeds": seeds, "distance": distance, "count": count - 1}
        )


class MorphologicalReconstruction(Block):
    type_id, title = "cv.morph_reconstruct", "Morphological reconstruction"
    category, block_type, subtype = "Image processing", "Image processing", "Morphology"
    inputs = (Port("image"), Port("marker"))
    outputs = (Port("image"), Port("iterations", "number"), Port("info", "data"))
    defaults = {"method": "dilation", "connectivity": "8", "maximum_iterations": 1024}
    parameter_choices = {"method": ("dilation", "erosion"), "connectivity": ("4", "8")}
    parameter_help = {
        "method": "Dilation requires marker <= image; erosion requires marker >= image.",
        "maximum_iterations": "Maximum geodesic steps. The image is emitted only on convergence.",
    }

    def process(self, inputs):
        source, marker = inputs["image"], inputs["marker"]
        require_same_frame((source, marker))
        mask = image_value(source)
        current = image_value(marker)
        if mask.ndim != 2 or current.shape != mask.shape or current.dtype != mask.dtype:
            raise ValueError("Reconstruction requires matching monochrome image and marker planes")
        dilation = self.parameters["method"] == "dilation"
        if np.any(current > mask if dilation else current < mask):
            raise ValueError("Marker values must respect the reconstruction image bounds")
        operation = cv2.dilate if dilation else cv2.erode
        constrain = cv2.min if dilation else cv2.max
        shape = cv2.MORPH_CROSS if self.parameters["connectivity"] == "4" else cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, (3, 3))
        maximum = integer(self.parameters["maximum_iterations"], "Maximum iterations", 1, 100_000)
        steps, reason = 0, "reconstruction_iteration_limit"
        while steps <= maximum:
            if self._cancelled():
                reason = "reconstruction_cancelled"
                break
            following = constrain(operation(current, kernel), mask)
            if np.array_equal(current, following):
                return Result({"image": current, "iterations": steps, "info": {"converged": True}})
            if steps == maximum:
                break
            current = following
            steps += 1
        return Result(
            {"image": NO_VALUE, "iterations": steps, "info": {"converged": False}},
            passed=False,
            message=reason.replace("_", " "),
            fail_reasons=[reason],
        )


SEGMENTATION_BLOCKS = (
    FloodFill,
    Watershed,
    GrabCut,
    ThinMask,
    FillMaskHoles,
    DistancePeakMarkers,
    MorphologicalReconstruction,
)
