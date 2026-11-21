"""Recover rejected ArUco markers against calibrated planar board geometry."""

from collections import Counter
from typing import NamedTuple

import cv2
import numpy as np

from .aruco_dictionary import MAX_MARKERS
from .core import Port, Result
from .fiducial_detection import (
    MARKER_CHOICES,
    MARKER_OPTIONS,
    MARKER_RANGES,
    detector_parameters,
    planar_board_geometry,
    projective_center,
)
from .geometry3d_common import (
    Tool3D,
    camera_for_image,
    pose_matrix,
    pose_record,
    require_pinhole,
)
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image, integer
from .result_graphics import ACCEPTED, REJECTED
from .selection import SELECTION_INPUTS, contains, selection_mask

_RECOVERED = (35, 180, 255)
_DETECTOR_OPTIONS = {key: value for key, value in MARKER_OPTIONS.items() if key != "border_bits"}


class _Fit(NamedTuple):
    rvec: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    rms: float


def _ids(numbers):
    return np.asarray(numbers, dtype=np.int32).reshape(-1, 1)


def _pairs(board, observations):
    xyz, xy = board.matchImagePoints(list(observations.values()), _ids(list(observations)))
    return xyz.reshape(-1, 3).astype(float), xy.reshape(-1, 2).astype(float)


def _project(xyz, fit, k, d):
    return cv2.projectPoints(xyz, fit.rvec, fit.translation, k, d)[0].reshape(-1, 2)


def _fit_board(xyz, xy, layout_points, k, d, seed=None):
    if seed is None:
        ok, rvec, translation = cv2.solvePnP(xyz, xy, k, d)
    else:
        ok, rvec, translation = cv2.solvePnP(
            xyz, xy, k, d, seed.rvec.copy(), seed.translation.copy(), useExtrinsicGuess=True
        )
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(translation).all():
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    if np.any(layout_points @ rotation[2] + translation.item(2) <= 0):
        return None
    fit = _Fit(rvec, translation, rotation, 0.0)
    residuals = _project(xyz, fit, k, d) - xy
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return fit._replace(rms=rms) if np.isfinite(rms) else None


def _recover(detector, gray, board, seed, rejected, geometry, fit, k, d, corner_limit):
    corners, ids, _, recovered = detector.refineDetectedMarkers(
        gray,
        board,
        list(seed.values()),
        _ids(list(seed)),
        [quad for _, quad in rejected],
        k,
        d,
    )
    if recovered is None:
        return {}, {}
    count = len(seed)
    native_ids = ids.ravel().tolist()
    if native_ids[:count] != list(seed) or len(native_ids) - count != recovered.size:
        raise RuntimeError("ArUco recovery returned inconsistent candidate indices")
    accepted, excluded = {}, {}
    for tag, quad, index in zip(
        native_ids[count:], corners[count:], recovered.ravel(), strict=True
    ):
        original_index, _ = rejected[int(index)]
        if tag in seed or tag in accepted or tag in excluded or tag not in geometry:
            raise RuntimeError("ArUco recovery returned an unexpected board marker")
        predicted = _project(geometry[tag], fit, k, d)
        error = float(np.max(np.linalg.norm(predicted - quad.reshape(4, 2), axis=1)))
        if np.isfinite(error) and error <= corner_limit:
            accepted[tag] = (quad, original_index, error)
        else:
            excluded[tag] = original_index
    return accepted, excluded


def _drawing(rows, color):
    return GraphicDrawing(
        tuple(
            graphic
            for tag, quad in rows
            for graphic in (
                graphic_object("polyline", quad.reshape(4, 2).tolist(), color, closed=True),
                graphic_object(
                    "label", [projective_center(quad.reshape(4, 2)).tolist()], color, text=str(tag)
                ),
            )
        )
    )


class RecoverArucoBoard(Tool3D):
    type_id, title = "3d.recover_aruco_board", "Recover partially visible ArUco board"
    subtype = "Fiducials"
    inputs = (*SELECTION_INPUTS, Port("target", "data"), Port("camera", "data"))
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("matches", "data"),
        Port("count", "number"),
        Port("report", "data"),
        Port("image"),
    )
    defaults = {
        **_DETECTOR_OPTIONS,
        "source_frame": "board",
        "minimum_seed_markers": 2,
        "minimum_markers": 2,
        "recover_rejected": True,
        "recovery_search_px": 10.0,
        "recovery_error_correction_rate": 1.0,
        "check_all_orders": True,
        "max_recovered_corner_error_px": 4.0,
        "max_pose_rms_px": 2.0,
    }
    parameter_choices = {"corner_refinement": MARKER_CHOICES["corner_refinement"]}
    parameter_ranges = {
        **{key: MARKER_RANGES[key] for key in _DETECTOR_OPTIONS if key in MARKER_RANGES},
        "minimum_seed_markers": NumericRange(2, MAX_MARKERS, whole=True),
        "minimum_markers": NumericRange(2, MAX_MARKERS, whole=True),
        "recovery_search_px": NumericRange(0.01, 1000),
        "recovery_error_correction_rate": NumericRange(0, 1),
        "max_recovered_corner_error_px": NumericRange(0, 1000),
        "max_pose_rms_px": NumericRange(0, 1000),
    }
    parameter_help = {
        "source_frame": "Name for the coordinate frame of the board layout.",
        "minimum_seed_markers": "Number of unique board IDs that must decode directly.",
        "minimum_markers": "Total number of matched board IDs needed to output a pose.",
        "recover_rejected": "Use the seed pose to look for visible markers that failed decoding.",
        "recovery_search_px": "OpenCV search radius for associating rejected corners.",
        "recovery_error_correction_rate": "Fraction of the dictionary bit correction budget.",
        "check_all_orders": "Check all four rotations of each rejected quadrilateral.",
        "max_recovered_corner_error_px": "Max corner error against the seed pose in pixels.",
        "max_pose_rms_px": "Maximum distorted pixel RMS for the seed and final fits.",
    }
    input_help = {
        "target": "Planar fiducial_target_v1 geometry and dictionary.",
        "camera": "Matching calibrated pinhole image grid. Use the board's metric units.",
        "mask": "Limit eligible detected and rejected quadrilaterals by their center.",
        "region": "Limit eligible detected and rejected quadrilaterals by their center.",
    }
    payload_reference = "image"
    composes_result_image = True
    parallel_safe = True
    preview_safe = True
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["source_frame"], str) or not p["source_frame"].strip():
            raise ValidationError("Use a named board source frame", "source_frame")
        for key in ("recover_rejected", "check_all_orders"):
            if not isinstance(p[key], bool):
                raise ValidationError("Use a boolean setting", key)
        for lower, upper in (
            ("adaptive_min", "adaptive_max"),
            ("min_perimeter_rate", "max_perimeter_rate"),
        ):
            if float(p[lower]) > float(p[upper]):
                raise ValidationError("Maximum must be at least the minimum", upper)

    def process(self, inputs):
        source, target, camera = inputs["image"], inputs["target"], inputs["camera"]
        p = self.parameters
        dictionary, units, board_ids, layout = planar_board_geometry(target)
        k, d, _ = require_pinhole(camera)
        camera_for_image(camera, source)
        board_frame, camera_frame = p["source_frame"].strip(), camera.get("frame", "camera")
        if board_frame == camera_frame:
            raise ValidationError("Board and camera frames must be distinct", "source_frame")
        if units != camera.get("units", "m"):
            raise ValidationError("Board and camera must use the same metric length units")
        for name in ("minimum_seed_markers", "minimum_markers"):
            if int(p[name]) > len(board_ids):
                raise ValidationError("Marker minimum exceeds the board's marker count", name)
        border = integer(target.get("border_bits", 1), "Board marker border bits", 1, 10)
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        mask = selection_mask(inputs, gray)
        native_params = detector_parameters({**p, "border_bits": border})
        refinement = cv2.aruco.RefineParameters(
            float(p["recovery_search_px"]),
            float(p["recovery_error_correction_rate"]),
            p["check_all_orders"],
        )
        detector = cv2.aruco.ArucoDetector(dictionary, native_params, refinement)
        geometry = dict(zip(board_ids, layout, strict=True))
        board = cv2.aruco.Board(layout.astype(np.float32), dictionary, _ids(board_ids))

        def selected(quad):
            return mask is None or contains(mask, projective_center(quad.reshape(4, 2)))

        corners, ids, rejected_corners = detector.detectMarkers(gray)
        detected = (
            []
            if ids is None
            else [
                (int(tag), quad)
                for tag, quad in zip(ids.ravel(), corners, strict=True)
                if selected(quad)
            ]
        )
        foreign = [(tag, quad) for tag, quad in detected if tag not in geometry]
        counts = Counter(tag for tag, _ in detected if tag in geometry)
        duplicates = [tag for tag in board_ids if counts[tag] > 1]
        unique_detections = {tag: quad for tag, quad in detected if counts[tag] == 1}
        seed = {tag: unique_detections[tag] for tag in board_ids if tag in unique_detections}
        recovered, excluded = {}, {}
        fit = seed_fit = None
        pairs = None
        limit = float(p["max_pose_rms_px"])
        if duplicates:
            reason = "aruco_board_duplicate_ids"
        elif len(seed) < int(p["minimum_seed_markers"]):
            reason = "aruco_board_insufficient_seed"
        else:
            xyz, xy = _pairs(board, seed)
            layout_points = layout.reshape(-1, 3)
            seed_fit = _fit_board(xyz, xy, layout_points, k, d)
            if seed_fit is None or seed_fit.rms > limit:
                reason = "aruco_board_seed_fit_failed"
            else:
                if p["recover_rejected"] and len(seed) < len(board_ids):
                    eligible = [
                        (i, quad) for i, quad in enumerate(rejected_corners) if selected(quad)
                    ]
                    if eligible:
                        recovered, excluded = _recover(
                            detector,
                            gray,
                            board,
                            seed,
                            eligible,
                            geometry,
                            seed_fit,
                            k,
                            d,
                            float(p["max_recovered_corner_error_px"]),
                        )
                if recovered:
                    observations = {**seed, **{tag: row[0] for tag, row in recovered.items()}}
                    pairs = _pairs(
                        board, {tag: observations[tag] for tag in board_ids if tag in observations}
                    )
                    fit = _fit_board(*pairs, layout_points, k, d, seed=seed_fit)
                else:
                    fit, pairs = seed_fit, (xyz, xy)
                if len(seed) + len(recovered) < int(p["minimum_markers"]):
                    reason = "aruco_board_insufficient_matches"
                elif fit is None or fit.rms > limit:
                    reason = "aruco_board_final_fit_failed"
                else:
                    reason = None

        pose = (
            pose_record(
                pose_matrix(fit.rotation, fit.translation),
                board_frame,
                camera_frame,
                units,
                reprojection_rms_px=fit.rms,
            )
            if reason is None
            else NO_VALUE
        )
        matches = []
        per_corner = (
            dict(
                zip(
                    (tag for tag in board_ids if tag in seed or tag in recovered),
                    np.linalg.norm(_project(pairs[0], fit, k, d) - pairs[1], axis=1).reshape(-1, 4),
                    strict=True,
                )
            )
            if fit is not None
            else {}
        )
        for tag in board_ids:
            if tag in recovered:
                quad, index, error = recovered[tag]
                row = {"origin": "recovered", "rejected_index": index, "seed_max_error_px": error}
            elif tag in seed:
                quad, row = seed[tag], {"origin": "detected"}
            else:
                continue
            row.update(
                id=tag,
                object_corners=geometry[tag].tolist(),
                image_corners=quad.reshape(4, 2).tolist(),
                reprojection_errors_px=per_corner[tag].tolist() if tag in per_corner else None,
            )
            matches.append(row)
        report = {
            "model": "aruco_board_recovery_v1",
            "units": units,
            "source_frame": board_frame,
            "target_frame": camera_frame,
            "board_ids": board_ids,
            "decoded_board_ids": list(seed),
            "recovered_ids": [tag for tag in board_ids if tag in recovered],
            "missing_ids": [tag for tag in board_ids if tag not in seed and tag not in recovered],
            "foreign_ids": sorted({tag for tag, _ in foreign}),
            "duplicate_ids": duplicates,
            "rejected_candidate_count": len(rejected_corners),
            "excluded_recovery_indices": excluded,
            "seed_rms_px": seed_fit.rms if seed_fit is not None else None,
            "final_rms_px": fit.rms if fit is not None else None,
        }
        drawings = {}
        if self.generate_graphics:
            if seed:
                drawings["Decoded board markers"] = _drawing(seed.items(), ACCEPTED)
            if recovered:
                drawings["Recovered board markers"] = _drawing(
                    ((tag, row[0]) for tag, row in recovered.items()), _RECOVERED
                )
            if foreign:
                drawings["Ignored foreign markers"] = _drawing(foreign, REJECTED)
        return Result(
            {
                "pose": pose,
                "matches": matches,
                "count": len(matches),
                "report": report,
                "image": source,
            },
            drawings=drawings,
            passed=reason is None,
            fail_reasons=[] if reason is None else [reason],
            message=f"{len(matches)}/{len(board_ids)} board markers · {len(recovered)} recovered",
        )
