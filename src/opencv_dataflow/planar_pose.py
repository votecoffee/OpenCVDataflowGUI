"""Keep calibrated planar pose alternatives and gate ambiguous single-pose output."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import (
    Tool3D,
    camera_for_image,
    points,
    pose_matrix,
    pose_record,
    require_pinhole,
)
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, ValuePacket, require_same_frame

_METHODS = {"ippe": cv2.SOLVEPNP_IPPE, "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE}
_GEOMETRY_TOLERANCE = 1e-7
_AXES = np.asarray([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float)
_COLORS = ((112, 230, 32), (240, 112, 40))


def _normalize(values, label):
    with np.errstate(over="ignore", invalid="ignore"):
        center = values.mean(axis=0)
        centered = values - center
    scale = float(np.max(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValidationError(f"{label} need distinct points with a finite coordinate span")
    return centered / scale, center, scale


def _planar_geometry(xyz, xy, method):
    local, center, scale = _normalize(xyz, "Object points")
    if method == "ippe_square" and (
        len(xyz) != 4 or not np.allclose(xyz / scale, _AXES, rtol=0, atol=_GEOMETRY_TOLERANCE)
    ):
        raise ValidationError(
            "IPPE square needs four centered Z=0 corners in TL, TR, BR, BL order with object +Y up"
        )
    _, singular, basis = np.linalg.svd(local, full_matrices=False)
    if singular[1] <= singular[0] * _GEOMETRY_TOLERANCE:
        raise ValidationError("Planar pose needs non-collinear object points")
    if singular[2] > singular[0] * _GEOMETRY_TOLERANCE:
        raise ValidationError("Planar pose requires coplanar object points")
    normalized_pixels, _, _ = _normalize(xy, "Image points")
    singular_pixels = np.linalg.svd(normalized_pixels, compute_uv=False)
    if singular_pixels[1] <= singular_pixels[0] * _GEOMETRY_TOLERANCE:
        raise ValidationError("Planar pose needs non-collinear image points")
    if len(np.unique(xy, axis=0)) != len(xy) or len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValidationError("Planar correspondences need distinct object and image points")
    normal = basis[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return local, center, scale, normal


def _candidate(index, rv, tv, local, xy, k, d, scale, limit):
    measure = {"solver_index": index, "rms_px": None, "max_error_px": None, "min_depth": None}
    rv, tv = rv.reshape(3), tv.reshape(3)
    if not np.isfinite(rv).all() or not np.isfinite(tv).all():
        return {**measure, "status": "nonfinite_solution"}, None
    rotation = cv2.Rodrigues(rv)[0]
    depth = float(np.min(local @ rotation[2] + tv[2]) * scale)
    if not np.isfinite(depth):
        return {**measure, "status": "nonfinite_solution"}, None
    measure["min_depth"] = depth
    if depth <= 0:
        return {**measure, "status": "nonpositive_depth"}, None
    projected = cv2.projectPoints(local, rv, tv, k, d)[0].reshape(-1, 2)
    with np.errstate(over="ignore", invalid="ignore"):
        errors_sq = np.sum((projected - xy) ** 2, axis=1)
    if not np.isfinite(errors_sq).all():
        return {**measure, "status": "nonfinite_projection"}, None
    rms = float(np.sqrt(np.mean(errors_sq)))
    peak = float(np.sqrt(errors_sq.max()))
    if not np.isfinite(rms):
        return {**measure, "status": "nonfinite_projection"}, None
    measure.update(rms_px=rms, max_error_px=peak)
    if rms > limit:
        return {**measure, "status": "reprojection_limit"}, None
    return {**measure, "status": "accepted"}, (rotation, rv, tv)


def _unique(ranked):
    selected = []
    for measure, solution in ranked:
        if solution is None:
            continue
        rotation, _, translation = solution
        if any(
            np.linalg.norm(rotation - other[0]) <= 1e-6
            and np.linalg.norm(translation - other[2]) <= 1e-6
            for _, other in selected
        ):
            measure["status"] = "duplicate_solution"
        else:
            selected.append((measure, solution))
    return selected


def _normal_drawing(normal, length, selected, k, d):
    line = np.asarray([[0.0, 0.0, 0.0], normal * length])
    layers = {}
    for index, (measure, (rotation, rv, tv)) in enumerate(selected):
        if np.any(line @ rotation[2] + tv[2] <= 0):
            continue
        uv = cv2.projectPoints(line, rv, tv, k, d)[0].reshape(2, 2)
        if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1_000_000:
            continue
        color = _COLORS[index]
        layers[f"Candidate {index + 1} plane normal"] = GraphicDrawing(
            (
                graphic_object("polyline", uv, color, thickness=3),
                graphic_object("marker", [uv[-1]], color),
                graphic_object(
                    "label", [uv[-1]], color, text=f"H{index + 1}: {measure['rms_px']:.2f} px"
                ),
            )
        )
    return layers


class PlanarPoseHypotheses(Tool3D):
    type_id, title = "3d.planar_pose_hypotheses", "Planar pose hypotheses"
    subtype = "Geometry"
    inputs = (
        Port("object_points", "data"),
        Port("image_points", "data"),
        Port("camera", "data"),
        Port("image", "image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("hypotheses", "data"),
        Port("ambiguous", "boolean"),
        Port("report", "data"),
        Port("image", "image", required=False),
    )
    defaults = {
        "method": "ippe",
        "source_frame": "object",
        "max_rms_px": 3.0,
        "ambiguity_gap_px": 0.20,
        "normal_length": 0.0,
    }
    parameter_choices = {"method": tuple(_METHODS)}
    parameter_ranges = {
        "max_rms_px": NumericRange(0),
        "ambiguity_gap_px": NumericRange(0),
        "normal_length": NumericRange(0),
    }
    parameter_help = {
        "method": "IPPE for coplanar points. IPPE square for four centered marker corners.",
        "source_frame": "Coordinate frame of the supplied object points.",
        "max_rms_px": "Maximum distortion-aware pixel RMS for each plausible candidate.",
        "ambiguity_gap_px": "Largest best-to-second RMS gap still considered ambiguous.",
        "normal_length": "Preview normal length in object units. Zero scales it to the target.",
    }
    input_help = {
        "object_points": "At least four coplanar XYZ points in camera length units.",
        "image_points": "Corresponding distorted XY pixels in the calibrated pinhole image.",
        "image": "Optional source image for drawing the candidates' plane normals.",
    }
    payload_reference = "image"
    composes_result_image = True
    preview_safe = True
    parallel_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_frame"], str) or not p["source_frame"].strip():
            raise ValidationError("Use a named object source frame", "source_frame")

    def process(self, inputs):
        p = self.parameters
        model = inputs["camera"]
        k, d, size = require_pinhole(model)
        camera_frame, units = model.get("frame", "camera"), model.get("units", "m")
        if camera_frame == p["source_frame"]:
            raise ValidationError("Object and camera frames must be distinct", "source_frame")
        xyz, xy = points(inputs["object_points"], 3, 4), points(inputs["image_points"], 2, 4)
        if len(xyz) != len(xy) or len(xyz) > 4096:
            raise ValidationError("Use 4 to 4096 paired object and image points")
        obj_units = self.input_payloads.get("object_points", {}).get("object_point_units")
        if obj_units is not None and obj_units != units:
            raise ValidationError("Object point units must match the camera model")
        image_context = self.input_payloads.get("image_points", {})
        spatial = image_context.get("spatial")
        if spatial and spatial["image_size"] != list(size):
            raise ValidationError("Image point grid must match the calibrated camera")
        image = inputs.get("image")
        if image is not None:
            camera_for_image(model, image)
            require_same_frame(
                [image, ValuePacket(xy, image_context)],
                message="Image and point payload must describe the same calibrated image frame",
            )
        local, center, scale, normal = _planar_geometry(xyz, xy, p["method"])
        count, rotations, translations, _ = cv2.solvePnPGeneric(
            local, xy, k, d, flags=_METHODS[p["method"]]
        )
        if count != len(rotations) or count != len(translations) or count > 2:
            raise ValueError("Unexpected native IPPE candidate count")
        limit = float(p["max_rms_px"])
        ranked = sorted(
            (
                _candidate(i, rv, tv, local, xy, k, d, scale, limit)
                for i, (rv, tv) in enumerate(zip(rotations, translations, strict=True))
            ),
            key=lambda item: float("inf") if item[0]["rms_px"] is None else item[0]["rms_px"],
        )
        # Reject solutions that cannot be represented in the caller's original frame.
        for i, (measure, solution) in enumerate(ranked):
            if solution is None:
                continue
            rotation, _, tv = solution
            with np.errstate(over="ignore", invalid="ignore"):
                translation = tv * scale - rotation @ center
            if not np.isfinite(translation).all():
                measure["status"] = "nonfinite_solution"
                ranked[i] = (measure, None)
        selected = _unique(ranked)
        hypotheses = [
            pose_record(
                pose_matrix(rotation, tv * scale - rotation @ center),
                p["source_frame"],
                camera_frame,
                units,
                solver_index=measure["solver_index"],
                rms_pixels=measure["rms_px"],
            )
            for measure, (rotation, _, tv) in selected
        ]
        gap = (
            hypotheses[1]["rms_pixels"] - hypotheses[0]["rms_pixels"]
            if len(hypotheses) > 1
            else None
        )
        ambiguity_limit = float(p["ambiguity_gap_px"])
        ambiguous = gap is not None and gap <= ambiguity_limit
        report = {
            "model": "planar_pose_hypotheses_v1",
            "method": p["method"],
            "source_frame": p["source_frame"],
            "target_frame": camera_frame,
            "units": units,
            "max_rms_px": limit,
            "ambiguity_gap_px": ambiguity_limit,
            "native_count": int(count),
            "plausible_count": len(hypotheses),
            "rms_gap_px": gap,
            "ambiguous": ambiguous,
            "candidates": [measure for measure, _ in ranked],
        }
        successful = bool(hypotheses)
        chosen = hypotheses[0] if successful and not ambiguous else NO_VALUE
        drawings = {}
        if image is not None and self.generate_graphics:
            normal_length = float(p["normal_length"])
            length = normal_length / scale if normal_length else 0.6
            if np.isfinite(length):
                drawings = _normal_drawing(normal, length, selected, k, d)
        if ambiguous:
            message = f"Ambiguous planar pose · RMS gap {gap:.4g} px"
        elif successful:
            message = f"Selected planar pose · RMS {chosen['rms_pixels']:.4g} px"
        else:
            message = "No planar pose meets the depth and reprojection checks"
        return Result(
            {
                "pose": chosen,
                "hypotheses": hypotheses,
                "ambiguous": ambiguous,
                "report": report,
                "image": image if image is not None else NO_VALUE,
            },
            drawings=drawings,
            passed=successful,
            fail_reasons=[] if successful else ["planar_pose_no_candidate"],
            message=message,
        )
