"""Review sparse stereo correspondences against a fundamental matrix."""

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .geometry3d_common import Tool3D, matrix, points
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, check_spatial, require_same_frame
from .result_graphics import ACCEPTED, REJECTED
from .tickets import ticket_key


def _epilines(xy, source, fundamental):
    lines = cv2.computeCorrespondEpilines(xy[:, None, :], source, fundamental).reshape(-1, 3)
    if not np.isfinite(lines).all():
        raise ValidationError("Epipolar lines must be finite")
    return lines, np.hypot(lines[:, 0], lines[:, 1]) > 1e-12


def _visible_segment(line, size):
    a, b, c = line
    width, height = size
    dx, dy = a * (width - 1), b * (height - 1)
    if not min(0, dx) + min(0, dy) <= -c <= max(0, dx) + max(0, dy):
        return None
    if abs(b) >= abs(a):
        x = width - 1
        first, last = (0, round(-c / b)), (x, round(-(c + a * x) / b))
    else:
        y = height - 1
        first, last = (round(-c / a), 0), (round(-(c + b * y) / a), y)
    visible, first, last = cv2.clipLine((0, 0, width, height), first, last)
    return (first, last) if visible else None


class CheckEpipolarCorrespondences(Tool3D):
    type_id, title, subtype = (
        "3d.check_epipolar",
        "Check epipolar correspondences",
        "Reconstruction",
    )
    inputs = (
        Port("points_left", "data"),
        Port("points_right", "data"),
        Port("matrix", "data"),
        Port("left_image", required=False, required_when_connected=True),
        Port("right_image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("lines_left", "data"),
        Port("lines_right", "data"),
        Port("inliers", "data"),
        Port("inlier_ratio", "number"),
        Port("report", "data"),
        Port("left_overlay", required=False),
        Port("right_overlay", required=False),
    )
    defaults = {"max_distance": 1.5, "minimum_inlier_ratio": 1.0, "max_drawn_pairs": 32}
    parameter_ranges = {
        "max_distance": NumericRange(0, 100_000),
        "minimum_inlier_ratio": NumericRange(0, 1),
        "max_drawn_pairs": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "max_distance": "Both point-to-line distances must be at or below this pixel tolerance.",
        "minimum_inlier_ratio": "Minimum fraction of all input pairs that must meet the tolerance.",
        "max_drawn_pairs": "Draw this many pairs in input order. Zero hides epipolar graphics.",
    }
    input_help = {
        "points_left": "First list of paired points in left-image pixels.",
        "points_right": "Corresponding points in right-image pixels, in the same pair order.",
        "matrix": "Fundamental matrix F for left-to-right mapping, where p_rightᵀ F p_left = 0.",
        "left_image": "Optional left source image for diagnostics.",
        "right_image": "Optional right source image for diagnostics.",
    }
    process_inputs_readonly = True
    composes_result_image = True
    payload_reference = "points_left"

    def _view(self, inputs, side):
        key = f"points_{side}"
        xy = points(inputs[key], 2)
        if len(xy) > 100_000 or np.any(np.abs(xy) > 1e7):
            raise ValidationError("Use up to 100,000 points within ±10 million pixels")
        context = self.input_payloads[key]
        image_key = f"{side}_image"
        image = inputs.get(image_key)
        picture_context = self.input_payloads.get(image_key, {})
        expected = self.input_payloads["matrix"].get("epipolar_frames", {}).get(side)
        require_same_frame(
            (
                ValuePacket(xy, context),
                ValuePacket(image, picture_context),
                ValuePacket(None, {"spatial": expected}),
            ),
            message=f"{side.title()} points, image and fundamental matrix use different frames",
        )
        a, b = context.get("ticket"), picture_context.get("ticket")
        if a is not None and b is not None and ticket_key(a) != ticket_key(b):
            raise ValidationError(f"{side.title()} points and overlay image must share a ticket")
        context = {**context, **picture_context}
        spatial = context.get("spatial")
        if image is not None:
            size = list(image.shape[1::-1])
        elif spatial and spatial.get("root_id") is not None:
            size = spatial["image_size"]
        else:
            size = None
            context.pop("spatial", None)
        if size is not None:
            check_spatial(context, size)
            if (xy < 0).any() or (xy >= size).any():
                raise ValidationError(f"{side.title()} points must lie inside their image grid")
        return xy, context, image, size

    def _graphics(self, xy, lines, valid, inliers, image, size):
        if size is None:
            return (), NO_VALUE
        objects = []
        for i in range(min(len(xy), int(self.parameters["max_drawn_pairs"]))):
            color = ACCEPTED if inliers[i] else REJECTED
            segment = _visible_segment(lines[i], size) if valid[i] else None
            if segment is not None:
                objects.append(graphic_object("polyline", segment, color, thickness=1))
            objects.append(graphic_object("marker", [xy[i]], color))
        overlay = NO_VALUE
        if image is not None:
            overlay = overlay_image(image)
            identity = np.eye(3)
            for obj in objects:
                draw_object(overlay, obj, identity)
        return tuple(objects), overlay

    def process(self, inputs):
        left, right = (self._view(inputs, side) for side in ("left", "right"))
        if len(left[0]) != len(right[0]):
            raise ValidationError("Left and right point counts must match")
        f = matrix(inputs["matrix"], (3, 3), "Fundamental matrix")
        magnitude = np.max(np.abs(f))
        if magnitude == 0:
            raise ValidationError("Fundamental matrix must have rank two")
        f = f / magnitude
        singular = np.linalg.svd(f, compute_uv=False)
        if singular[1] <= singular[0] * 1e-12 or singular[2] > singular[1] * 1e-5:
            raise ValidationError("Fundamental matrix must have rank two")

        views = (left, right)
        left_lines, left_valid = _epilines(right[0], 2, f)
        right_lines, right_valid = _epilines(left[0], 1, f)
        lines, defined = (left_lines, right_lines), (left_valid, right_valid)
        distances = [
            np.abs(np.sum(xy * line[:, :2], axis=1) + line[:, 2])
            for (xy, _, _, _), line in zip(views, lines, strict=True)
        ]
        valid = left_valid & right_valid
        inliers = valid & (np.maximum(*distances) <= float(self.parameters["max_distance"]))
        count = int(inliers.sum())
        ratio = count / len(inliers)
        passed = bool(valid.any()) and ratio >= float(self.parameters["minimum_inlier_ratio"])

        report = {"pairs": len(inliers), "inliers": count, "undefined": int((~valid).sum())}
        outputs = {"inliers": inliers.tolist(), "inlier_ratio": ratio, "report": report}
        payloads, drawings = {}, {}
        for side, view, line, defined_side, distance in zip(
            ("left", "right"), views, lines, defined, distances, strict=True
        ):
            xy, context, image, size = view
            line_port, overlay_port = f"lines_{side}", f"{side}_overlay"
            outputs[line_port] = [
                row.tolist() if ok else None for row, ok in zip(line, defined_side, strict=True)
            ]
            report[f"distance_{side}_pixels"] = [
                float(value) if ok else None
                for value, ok in zip(distance, defined_side, strict=True)
            ]
            payloads[line_port] = context
            outputs[overlay_port] = NO_VALUE
            if self.generate_graphics or image is not None:
                objects, outputs[overlay_port] = self._graphics(
                    xy, line, defined_side, inliers, image, size
                )
                if self.generate_graphics and objects:
                    drawings[f"{side.title()} epipolar lines"] = GraphicDrawing(
                        objects, line_port, from_output=True
                    )
                if image is not None:
                    payloads[overlay_port] = context
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            passed=passed,
            score=ratio,
            fail_reasons=[] if passed else ["epipolar_inlier_ratio_below_minimum"],
            message=f"{count}/{len(inliers)} pairs meet the epipolar tolerance",
        )
