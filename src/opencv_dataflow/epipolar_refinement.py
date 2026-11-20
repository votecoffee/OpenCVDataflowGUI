"""Score and correct sparse matches on their original pixel grids."""

import cv2
import numpy as np

from .core import Port, Result
from .epipolar import _EpipolarPairTool
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange
from .payloads import NO_VALUE
from .result_graphics import ACCEPTED, REJECTED


def _sampson(left, right, fundamental):
    """Square-root Sampson distance, with undefined directions left as NaN."""
    a, b = [np.c_[view, np.ones(len(view))] for view in (left, right)]
    line_left, line_right = b @ fundamental, a @ fundamental.T
    length_left, length_right = [
        np.linalg.norm(line[:, :2], axis=1) for line in (line_left, line_right)
    ]
    roundoff = 32 * np.finfo(float).eps
    noise_left = roundoff * np.linalg.norm(np.abs(b) @ np.abs(fundamental[:, :2]), axis=1)
    noise_right = roundoff * np.linalg.norm(np.abs(a) @ np.abs(fundamental[:2]).T, axis=1)
    defined = (length_left > noise_left) & (length_right > noise_right)
    residual = np.abs(np.einsum("ij,ij->i", b, line_right))
    return np.divide(
        residual,
        np.hypot(length_left, length_right),
        out=np.full(len(left), np.nan),
        where=defined,
    )


def _inside(xy, size):
    return (
        (xy >= 0).all(axis=1) & (xy < size).all(axis=1)
        if size is not None
        else np.ones(len(xy), dtype=bool)
    )


def _refine(fundamental, src, dst, indices, views, parameters):
    # Only the pairs that already passed input quality are eligible for correction.
    left, right = cv2.correctMatches(
        fundamental, src[indices][None, :, :], dst[indices][None, :, :]
    )
    left, right = left.reshape(-1, 2), right.reshape(-1, 2)
    finite = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    bounded = finite & (np.abs(left) <= 1e7).all(axis=1) & (np.abs(right) <= 1e7).all(axis=1)
    supported = _inside(left, views[0][3]) & _inside(right, views[1][3])
    shifts = np.full((len(indices), 2), np.nan)
    shifts[bounded, 0] = np.linalg.norm(left[bounded] - src[indices[bounded]], axis=1)
    shifts[bounded, 1] = np.linalg.norm(right[bounded] - dst[indices[bounded]], axis=1)
    residual = np.full(len(indices), np.nan)
    residual[bounded] = _sampson(left[bounded], right[bounded], fundamental)
    status = np.select(
        (
            ~finite,
            ~bounded | ~supported,
            (shifts > float(parameters["maximum_shift_px"])).any(axis=1),
            ~np.isfinite(residual),
            residual > float(parameters["corrected_tolerance_px"]),
        ),
        (
            "nonfinite_correction",
            "outside_grid",
            "shift_limit",
            "undefined_corrected_epiline",
            "closure_limit",
        ),
        default="accepted",
    )
    return (left, right), shifts, residual, status


def _correction_graphics(original, target, accepted, limit):
    objects = []
    for i in range(min(len(original), limit)):
        color = ACCEPTED if accepted[i] else REJECTED
        before = original[i]
        objects.append(graphic_object("marker", [before], color, thickness=1))
        if accepted[i] and np.linalg.norm(target[i] - before) > 1e-6:
            objects.append(graphic_object("polyline", (before, target[i]), color))
            objects.append(graphic_object("marker", [target[i]], color, thickness=2))
    return tuple(objects)


def _finite_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


class RefineEpipolarCorrespondences(_EpipolarPairTool):
    type_id, title, subtype = (
        "3d.refine_epipolar",
        "Score and refine epipolar correspondences",
        "Reconstruction",
    )
    outputs = (
        Port("points_left", "data", required=False),
        Port("points_right", "data", required=False),
        Port("indices", "data"),
        Port("inliers", "data"),
        Port("inlier_ratio", "number"),
        Port("report", "data"),
    )
    defaults = {
        "mode": "correct",
        "maximum_sampson_px": 2.0,
        "maximum_shift_px": 3.0,
        "corrected_tolerance_px": 1e-4,
        "minimum_inlier_ratio": 1.0,
        "max_drawn_pairs": 64,
    }
    parameter_choices = {"mode": ("evaluate", "correct")}
    parameter_ranges = {
        "maximum_sampson_px": NumericRange(0, 100_000),
        "maximum_shift_px": NumericRange(0, 100_000),
        "corrected_tolerance_px": NumericRange(1e-9, 100),
        "minimum_inlier_ratio": NumericRange(0, 1),
        "max_drawn_pairs": NumericRange(0, 512, whole=True),
    }
    parameter_help = {
        "mode": "Evaluate accepts original points. Correct applies OpenCV optimal correction.",
        "maximum_sampson_px": "Reject raw pairs above this square-root Sampson error in pixels.",
        "maximum_shift_px": "Maximum correction displacement in either image, in pixels.",
        "corrected_tolerance_px": "Maximum residual square-root Sampson error after correction.",
        "minimum_inlier_ratio": "Fraction of all input pairs that must remain accepted to pass.",
        "max_drawn_pairs": "Maximum input pairs per preview. Zero disables correction drawings.",
    }
    process_inputs_readonly = parallel_safe = preview_safe = True
    payload_reference = "points_left"

    def process(self, inputs):
        p = self.parameters
        left, right, f = self._read_pair(inputs)
        views = (left, right)
        a, b = left[0], right[0]
        score = _sampson(a, b, f)
        defined = np.isfinite(score)
        eligible = defined & (score <= float(p["maximum_sampson_px"]))
        candidates = np.flatnonzero(eligible)
        status = np.where(defined, "sampson_limit", "undefined_epiline").astype(object)
        status[eligible] = "accepted"
        target = (a.copy(), b.copy())
        residual = np.full(len(a), np.nan)
        shifts = np.full((len(a), 2), np.nan)
        if p["mode"] == "correct" and len(candidates):
            corrected, move, after, reason = _refine(f, a, b, candidates, views, p)
            for array, changes in zip(target, corrected, strict=True):
                array[candidates] = changes
            status[candidates], shifts[candidates], residual[candidates] = reason, move, after
        else:
            residual[eligible] = score[eligible]
            shifts[eligible] = 0
        accepted = status == "accepted"
        indices = np.flatnonzero(accepted)
        count = len(indices)
        ratio = count / len(a)
        passed = count > 0 and ratio >= float(p["minimum_inlier_ratio"])
        report = {
            "model": "epipolar_refinement_v1",
            "mode": p["mode"],
            "pair_count": len(a),
            "accepted_count": count,
            "inlier_ratio": ratio,
            "status": status.tolist(),
            "sampson_px": _finite_values(score),
            "output_sampson_px": _finite_values(residual),
            "left_shift_px": _finite_values(shifts[:, 0]),
            "right_shift_px": _finite_values(shifts[:, 1]),
            "accepted_input_rms_px": float(np.sqrt(np.mean(score[accepted] ** 2)))
            if count
            else None,
            "accepted_output_rms_px": (
                float(np.sqrt(np.mean(residual[accepted] ** 2))) if count else None
            ),
        }
        outputs = {
            "indices": indices.tolist(),
            "inliers": accepted.tolist(),
            "inlier_ratio": ratio,
            "report": report,
        }
        payloads, graphics, drawings = {}, {}, {}
        for side, view, corrected in zip(("left", "right"), views, target, strict=True):
            original, context, image, size = view
            point_port = f"points_{side}"
            outputs[point_port] = corrected[accepted].tolist() if count else NO_VALUE
            if count:
                payloads[point_port] = context
            if not self.generate_graphics or size is None or not p["max_drawn_pairs"]:
                continue
            objects = _correction_graphics(original, corrected, accepted, int(p["max_drawn_pairs"]))
            name = f"{side.title()} correspondence corrections"
            source = f"{side}_image" if image is not None else point_port
            drawings[name] = GraphicDrawing(objects, source)
            if self.capture_graphics and image is not None:
                overlay = overlay_image(image)
                identity = np.eye(3)
                for obj in objects:
                    draw_object(overlay, obj, identity)
                graphics[name] = overlay
        return Result(
            outputs,
            payloads=payloads,
            graphics=graphics,
            drawings=drawings,
            passed=passed,
            score=ratio,
            fail_reasons=[] if passed else ["epipolar_refinement_insufficient_pairs"],
            message=f"{count}/{len(a)} pairs accepted",
        )
