"""Build OpenCV watershed markers from point selections or labeled mask cores."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import label_color_map
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .selection import SELECTION_INPUTS, selection_mask

SEED_OUTPUTS = (
    Port("markers"), Port("seeds", visual_type="region"), Port("preview"), Port("count", "number")
)


def seed_result(markers, count):
    return Result(
        {
            "markers": markers,
            "seeds": (markers > 1).astype(np.uint8) * 255,
            "preview": label_color_map(markers, 1),
            "count": count,
        },
        message=f"{count} foreground seeds · background label 1",
    )


def seed_points(values, shape, name):
    try:
        points = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} needs a list of [x,y] points") from exc
    if points.shape == (0,):
        return []
    if points.ndim != 2 or points.shape[1] != 2 or len(points) > 512:
        raise ValidationError(f"{name} needs up to 512 [x,y] points")
    height, width = shape
    points = np.rint(points)
    if (
        not np.isfinite(points).all()
        or np.any(points < 1)
        or np.any(points > [width - 2, height - 2])
    ):
        raise ValidationError(f"{name} must be at least one pixel inside the image border")
    return points.astype(int).tolist()


def paint_seed(markers, point, radius, label, selection=None):
    x, y = point
    height, width = markers.shape
    left, right = max(1, x - radius), min(width - 1, x + radius + 1)
    top, bottom = max(1, y - radius), min(height - 1, y + radius + 1)
    yy, xx = np.ogrid[top:bottom, left:right]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    if selection is not None:
        if not selection[y, x]:
            raise ValidationError("Foreground seed centers must be inside the mask or region")
        disk &= selection[top:bottom, left:right] != 0
    patch = markers[top:bottom, left:right]
    if np.any(disk & (patch != 0) & (patch != label)):
        raise ValidationError(
            "Seed areas overlap across different labels; separate them or reduce radius"
        )
    patch[disk] = label


class WatershedPointSeeds(Block):
    type_id, title = "inspect.watershed_point_seeds", "Watershed seeds from points"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    supports_seed_editor = True
    inputs = (
        *SELECTION_INPUTS,
        Port("foreground_points", "data", required=False),
        Port("background_points", "data", required=False),
    )
    outputs = SEED_OUTPUTS
    defaults = {
        "foreground_points": [],
        "background_points": [],
        "radius": 2,
        "background_mode": "border",
    }
    parameter_choices = {"background_mode": ("border", "points")}
    parameter_ranges = {"radius": NumericRange(0, 64, whole=True)}
    parameter_help = {
        "foreground_points": "One local image [x,y] point per foreground label.",
        "background_points": "All points share background label 1.",
        "background_mode": "Border adds an inset ring. Points uses only specified points.",
    }
    input_help = {
        "foreground_points": "Overrides the saved foreground points.",
        "background_points": "Overrides the saved background points.",
    }

    def process(self, inputs):
        image = inputs["image"]
        shape = image.shape[:2]
        if min(shape) < 3:
            raise ValidationError("Watershed seeds require an image at least 3 by 3 pixels")
        try:
            require_same_frame(
                [
                    image,
                    *(
                        ValuePacket(inputs[key], self.input_payloads[key])
                        for key in ("foreground_points", "background_points")
                        if key in inputs
                    ),
                ]
            )
        except ValueError as exc:
            raise ValidationError("Seed points must use the image coordinate frame") from exc
        p = self.parameters
        foreground = seed_points(
            inputs.get("foreground_points", p["foreground_points"]), shape, "Foreground seeds"
        )
        background = seed_points(
            inputs.get("background_points", p["background_points"]), shape, "Background seeds"
        )
        if not foreground:
            raise ValidationError("Add at least one foreground seed")
        if p["background_mode"] == "points" and not background:
            raise ValidationError("Points background mode requires at least one background seed")
        markers = np.zeros(shape, dtype=np.int32)
        if p["background_mode"] == "border":
            cv2.rectangle(markers, (1, 1), (shape[1] - 2, shape[0] - 2), 1, 1)
        radius = int(p["radius"])
        for point in background:
            paint_seed(markers, point, radius, 1)
        selection = selection_mask(inputs, image)
        for label, point in enumerate(foreground, 2):
            paint_seed(markers, point, radius, label, selection)
        return seed_result(markers, len(foreground))


class WatershedMaskSeeds(Block):
    type_id, title = "inspect.watershed_mask_seeds", "Watershed seeds from masks"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = (Port("foreground", visual_type="region"), Port("background", visual_type="region"))
    outputs = SEED_OUTPUTS
    defaults = {"connectivity": "8"}
    parameter_choices = {"connectivity": ("4", "8")}
    parameter_help = {"connectivity": "Groups adjacent foreground seed pixels into one label."}
    input_help = {
        "foreground": "One connected component per foreground seed.",
        "background": "Known background pixels. Keep uncertain pixels clear in both masks.",
    }

    def process(self, inputs):
        foreground, background = inputs["foreground"], inputs["background"]
        if (
            foreground.ndim != 2
            or min(foreground.shape) < 3
            or foreground.shape != background.shape
            or foreground.dtype != np.uint8
            or background.dtype != np.uint8
        ):
            raise ValidationError("Use matching uint8 monochrome seed masks at least 3 by 3 pixels")
        try:
            require_same_frame((foreground, background))
        except ValueError as exc:
            raise ValidationError("Seed masks must use the same coordinate frame") from exc
        if np.any((foreground != 0) & (background != 0)):
            raise ValidationError("Foreground and background seed masks overlap")
        cores = (foreground != 0).astype(np.uint8)
        cores[[0, -1], :] = 0
        cores[:, [0, -1]] = 0
        known = background[1:-1, 1:-1] != 0
        if not cores.any() or not known.any():
            raise ValidationError("Both seed masks need pixels inside the one-pixel image border")
        count, labels = cv2.connectedComponentsWithAlgorithm(
            cores, int(self.parameters["connectivity"]), cv2.CV_32S, cv2.CCL_SAUF
        )
        labels[labels > 0] += 1
        labels[1:-1, 1:-1][known] = 1
        return seed_result(labels, count - 1)
