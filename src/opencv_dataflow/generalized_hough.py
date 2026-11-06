"""Locate template shapes by voting on grayscale edges."""

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_box, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .poses import make_pose
from .processing import gray_image
from .result_graphics import ACCEPTED
from .selection import SELECTION_INPUTS, selection_mask, whole

_MAX_VOTES = 150_000_000
_MAX_GUIL_WORK = 400_000_000


@dataclass(frozen=True)
class _HoughHit:
    x: float
    y: float
    scale: float
    clockwise_angle: float
    votes: int
    scale_votes: int
    angle_votes: int


def _template_anchor(value):
    try:
        if not isinstance(value, (list, tuple)):
            raise TypeError
        x, y = value
        return whole(x, "Anchor x", -1), whole(y, "Anchor y", -1)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(
            "Use a whole-pixel [x, y] template anchor", "template_anchor"
        ) from exc


def _detect(detector, template, template_edges, anchor, image, edges):
    detector.setTemplate(
        edges=template_edges,
        dx=cv2.Sobel(template, cv2.CV_32F, 1, 0),
        dy=cv2.Sobel(template, cv2.CV_32F, 0, 1),
        templCenter=anchor,
    )
    return detector.detect(
        edges=edges,
        dx=cv2.Sobel(image, cv2.CV_32F, 1, 0),
        dy=cv2.Sobel(image, cv2.CV_32F, 0, 1),
    )


def _ranked_hits(positions, votes, shape, selection, distance, maximum):
    if positions is None:
        return []
    positions = positions.reshape(-1, 4)
    votes = votes.reshape(-1, 3)
    centers = positions[:, :2]
    counts = votes[:, 0]
    pixels = np.rint(centers).astype(np.intp)
    height, width = shape
    x, y = pixels.T
    allowed = np.flatnonzero((x >= 0) & (x < width) & (y >= 0) & (y < height))
    if selection is not None:
        allowed = allowed[selection[y[allowed], x[allowed]] != 0]
    order = np.lexsort(
        (positions[allowed, 2], positions[allowed, 3], x[allowed], y[allowed], -counts[allowed])
    )
    accepted = []
    min_squared = distance**2
    for index in allowed[order]:
        cx, cy, scale, angle = map(float, positions[index])
        if any((cx - hit.x) ** 2 + (cy - hit.y) ** 2 < min_squared for hit in accepted):
            continue
        accepted.append(_HoughHit(cx, cy, scale, angle, *map(int, votes[index])))
        if len(accepted) == maximum:
            break
    return accepted


class FindHoughTemplate(Block):
    type_id, title, category = "inspect.hough_template", "Find Hough template", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs = (*SELECTION_INPUTS, Port("template"))
    outputs = (
        Port("matches", "data"),
        Port("poses", "data"),
        Port("count", "number"),
        Port("overlay", required=False),
    )
    composes_result_image = True
    max_image_pixels = 8_000_000
    requires_template_fit = True
    defaults = {
        "template_anchor": [-1, -1],
        "canny_low": 40.0,
        "canny_high": 100.0,
        "levels": 360,
        "dp": 1.0,
        "minimum_votes": 25,
        "min_center_distance": 10.0,
        "minimum_matches": 1,
        "maximum_matches": 25,
        "draw_overlay": True,
    }
    parameter_ranges = {
        "canny_low": NumericRange(0),
        "canny_high": NumericRange(0),
        "levels": NumericRange(1, 720, whole=True),
        "dp": NumericRange(1, 16),
        "minimum_votes": NumericRange(2, _MAX_VOTES, whole=True),
        "min_center_distance": NumericRange(0, 10_000),
        "minimum_matches": NumericRange(0, 64, whole=True),
        "maximum_matches": NumericRange(1, 64, whole=True),
    }
    parameter_help = {
        "template_anchor": "Template pose origin. [-1, -1] uses the integer center.",
        "levels": "Number of gradient orientation bins used for voting.",
        "dp": "Accumulator cell spacing in source pixels. Larger values use a coarser grid.",
        "minimum_votes": "Minimum accepted center vote count.",
        "min_center_distance": "Minimum distance between retained anchors after selection.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        anchor = _template_anchor(p["template_anchor"])
        if -1 in anchor and anchor != (-1, -1):
            raise ValidationError("Set both anchor coordinates or use [-1, -1]", "template_anchor")
        if p["canny_low"] >= p["canny_high"]:
            raise ValidationError("Canny low must be below Canny high", "canny_low", "canny_high")
        if p["minimum_matches"] > p["maximum_matches"]:
            raise ValidationError(
                "Minimum matches must not exceed maximum matches",
                "minimum_matches",
                "maximum_matches",
            )

    def process(self, inputs):
        p = self.parameters
        source = inputs["image"]
        image = gray_image(source, (np.uint8,), auto_depth=True)
        template = gray_image(inputs["template"], (np.uint8,), auto_depth=True)
        height, width = template.shape
        if min(height, width) < 3:
            raise ValidationError("Use a template of at least 3 × 3 pixels")
        if self.requires_template_fit and (height > image.shape[0] or width > image.shape[1]):
            raise ValidationError("The template must fit the search image")
        if image.size > self.max_image_pixels or template.size > 1_000_000:
            raise ValidationError("Crop the search image or template to fit the Hough pixel limit")
        anchor = _template_anchor(p["template_anchor"])
        if anchor == (-1, -1):
            anchor = (width // 2, height // 2)
        if not (0 <= anchor[0] < width and 0 <= anchor[1] < height):
            raise ValidationError("The anchor must be inside the template", "template_anchor")
        selection = selection_mask(inputs, image)
        low, high = float(p["canny_low"]), float(p["canny_high"])
        template_edges = cv2.Canny(template, low, high)
        edge_count = cv2.countNonZero(template_edges)
        if not edge_count:
            raise ValidationError("Template has no edges at the selected Canny thresholds")
        edges = cv2.Canny(image, low, high)
        self._check_work(edge_count, cv2.countNonZero(edges), image.shape)
        detector = self._detector()
        positions, votes = _detect(detector, template, template_edges, anchor, image, edges)
        hits = _ranked_hits(
            positions,
            votes,
            image.shape,
            selection,
            float(p["min_center_distance"]),
            int(p["maximum_matches"]),
        )
        return self._result(source, hits, anchor, width, height)

    def _check_work(self, template_edges, scene_edges, shape):
        if template_edges * scene_edges > _MAX_VOTES:
            raise ValidationError(
                "Hough search is too large. Crop inputs or increase Canny thresholds."
            )

    def _detector(self):
        p = self.parameters
        detector = cv2.createGeneralizedHoughBallard()
        detector.setLevels(int(p["levels"]))
        detector.setDp(float(p["dp"]))
        detector.setVotesThreshold(int(p["minimum_votes"]) - 1)
        return detector

    def _match(self, hit, anchor, width, height, size, payload):
        matrix = np.eye(3)
        matrix[:2, 2] = (hit.x, hit.y)
        return {
            "center": [hit.x, hit.y],
            "bounds": [hit.x - anchor[0], hit.y - anchor[1], width, height],
            "votes": hit.votes,
            "pose": make_pose(matrix, size, payload),
        }

    def _result(self, source, hits, anchor, width, height):
        p = self.parameters
        size = [source.shape[1], source.shape[0]]
        payload = self.input_payloads.get("image", {})
        matches = []
        objects = []
        for hit in hits:
            match = self._match(hit, anchor, width, height, size, payload)
            matches.append(match)
            if p["draw_overlay"] or self.generate_graphics:
                bounds = match["bounds"]
                outline = (
                    graphic_object("polyline", match["corners"], ACCEPTED, closed=True)
                    if "corners" in match
                    else graphic_box(*bounds, ACCEPTED)
                )
                objects.extend(
                    (
                        outline,
                        graphic_object("marker", ((hit.x, hit.y),), ACCEPTED),
                        graphic_object(
                            "text", ((bounds[0], bounds[1] - 3),), ACCEPTED, text=str(hit.votes)
                        ),
                    )
                )
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        if overlay is not NO_VALUE:
            identity = np.eye(3)
            for item in objects:
                draw_object(overlay, item, identity)
        count = len(matches)
        passed = count >= int(p["minimum_matches"])
        return Result(
            {
                "matches": matches,
                "poses": [m["pose"] for m in matches],
                "count": count,
                "overlay": overlay,
            },
            drawings={"Hough matches": GraphicDrawing(tuple(objects))}
            if self.generate_graphics
            else {},
            passed=passed,
            message=f"Found {count} Hough template matches",
            fail_reasons=[] if passed else ["hough_match_count_below_minimum"],
        )


class FindGuilTemplate(FindHoughTemplate):
    type_id, title = "inspect.guil_template", "Find rotated and scaled Hough template"
    max_image_pixels = 1_500_000
    requires_template_fit = False
    defaults = {
        **FindHoughTemplate.defaults,
        "dp": 2.0,
        "levels": 90,
        "minimum_votes": 50,
        "min_angle": 0.0,
        "max_angle": 360.0,
        "angle_step": 10.0,
        "min_scale": 0.5,
        "max_scale": 2.0,
        "scale_step": 0.1,
        "min_angle_votes": 300,
        "min_scale_votes": 50,
        "max_buffer_size": 48,
        "pair_angle": 90.0,
        "angle_epsilon": 4.0,
    }
    parameter_ranges = {
        **FindHoughTemplate.parameter_ranges,
        "minimum_votes": NumericRange(1, _MAX_GUIL_WORK, whole=True),
        "min_angle": NumericRange(0, 360),
        "max_angle": NumericRange(0, 360),
        "angle_step": NumericRange(0.5, 359),
        "min_scale": NumericRange(0.05, 8),
        "max_scale": NumericRange(0.05, 8),
        "scale_step": NumericRange(0.01, 8),
        "min_angle_votes": NumericRange(1, _MAX_GUIL_WORK, whole=True),
        "min_scale_votes": NumericRange(1, _MAX_GUIL_WORK, whole=True),
        "max_buffer_size": NumericRange(1, 256, whole=True),
        "pair_angle": NumericRange(1, 359),
        "angle_epsilon": NumericRange(0.1, 30),
    }
    parameter_help = {
        **FindHoughTemplate.parameter_help,
        "min_angle": "Start of the clockwise image-angle search in degrees.",
        "max_angle": "Exclusive end of the clockwise angle grid, up to 360 degrees.",
        "angle_step": "Spacing between angle hypotheses in degrees.",
        "min_scale": "Smallest template magnification to search.",
        "max_scale": "Exclusive end of the magnification grid.",
        "scale_step": "Spacing between magnification hypotheses.",
        "max_buffer_size": "Maximum feature pairs kept per orientation bin.",
        "pair_angle": "Target angle difference between the two gradients of a feature pair.",
        "angle_epsilon": "Angular tolerance in degrees for building and matching pairs.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for axis in ("angle", "scale"):
            low, high = f"min_{axis}", f"max_{axis}"
            if p[low] >= p[high]:
                raise ValidationError(f"Minimum {axis} must be below maximum {axis}", low, high)
        if min(p["pair_angle"], 360 - p["pair_angle"]) <= p["angle_epsilon"]:
            raise ValidationError(
                "Pair angle must exclude equal directions", "pair_angle", "angle_epsilon"
            )

    def _check_work(self, template_edges, scene_edges, shape):
        p = self.parameters
        pair_work = template_edges**2 + scene_edges**2
        buffer = int(p["max_buffer_size"])
        comparisons = min(
            (int(p["levels"]) + 1) * buffer**2, min(template_edges, scene_edges) ** 2 * buffer
        )
        angles = ceil((p["max_angle"] - p["min_angle"]) / p["angle_step"])
        scales = ceil((p["max_scale"] - p["min_scale"]) / p["scale_step"])
        grid = ceil(shape[0] / p["dp"]) * ceil(shape[1] / p["dp"])
        position_work = angles * scales * comparisons
        work = pair_work + (1 + angles) * comparisons + position_work + angles * scales * grid
        if work > _MAX_GUIL_WORK or position_work // int(p["minimum_votes"]) > 4_000_000:
            raise ValidationError(
                "Guil search is too large. Crop inputs or narrow the search grid."
            )

    def _detector(self):
        p = self.parameters
        detector = cv2.createGeneralizedHoughGuil()
        detector.setLevels(int(p["levels"]))
        detector.setDp(float(p["dp"]))
        detector.setMinAngle(float(p["min_angle"]))
        detector.setMaxAngle(float(p["max_angle"]))
        detector.setAngleStep(float(p["angle_step"]))
        detector.setMinScale(float(p["min_scale"]))
        detector.setMaxScale(float(p["max_scale"]))
        detector.setScaleStep(float(p["scale_step"]))
        detector.setAngleThresh(int(p["min_angle_votes"]))
        detector.setScaleThresh(int(p["min_scale_votes"]))
        detector.setPosThresh(int(p["minimum_votes"]))
        detector.setMaxBufferSize(int(p["max_buffer_size"]))
        detector.setXi(float(p["pair_angle"]))
        detector.setAngleEpsilon(float(p["angle_epsilon"]))
        return detector

    def _match(self, hit, anchor, width, height, size, payload):
        matrix = np.eye(3)
        matrix[:2] = cv2.getRotationMatrix2D((0, 0), -hit.clockwise_angle, hit.scale)
        matrix[:2, 2] = (hit.x, hit.y)
        pose = make_pose(matrix, size, payload)
        corners = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)))
        mapped = (corners - anchor) @ matrix[:2, :2].T + matrix[:2, 2]
        left, top = np.min(mapped, axis=0)
        right, bottom = np.max(mapped, axis=0)
        return {
            "center": [hit.x, hit.y],
            "bounds": [float(left), float(top), float(right - left + 1), float(bottom - top + 1)],
            "corners": mapped.tolist(),
            "votes": hit.votes,
            "scale_votes": hit.scale_votes,
            "angle_votes": hit.angle_votes,
            "scale": hit.scale,
            "angle_degrees": pose["angle_degrees"],
            "pose": pose,
        }
