"""Robust translation-only registration from paired image points."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .result_graphics import ACCEPTED, REJECTED
from .sample_tables import finite_features

_METHODS = {"ransac": cv2.RANSAC, "lmeds": cv2.LMEDS}
_PREVIEW_LIMIT = 512


class EstimateTranslation(Block):
    type_id, title = "inspect.translation", "Estimate image translation"
    category, block_type, subtype = "Inspection", "Inspection", "Registration"
    inputs = (Port("matches", "data"), Port("train_image", required=False))
    outputs = (
        Port("transform", "data", required=False),
        Port("inliers", "number"),
        Port("fit", "data"),
    )
    payload_reference = "matches"
    process_inputs_readonly = True
    defaults = {
        "method": "ransac",
        "reprojection_pixels": 3.0,
        "max_iterations": 2000,
        "confidence": 0.99,
        "refine_iterations": 0,
        "min_inliers": 3,
        "min_inlier_ratio": 0.5,
    }
    parameter_choices = {"method": tuple(_METHODS)}
    parameter_ranges = {
        "reprojection_pixels": NumericRange(1e-9, 1e12),
        "max_iterations": NumericRange(1, 100_000, whole=True),
        "confidence": NumericRange(1e-9, 1 - 1e-9),
        "refine_iterations": NumericRange(0, 10_000, whole=True),
        "min_inliers": NumericRange(1, 100_000, whole=True),
        "min_inlier_ratio": NumericRange(0, 1),
    }
    parameter_help = {
        "method": "RANSAC uses a pixel tolerance. LMeDS assumes a majority of correct pairs.",
        "reprojection_pixels": "RANSAC inlier distance in pixels. LMeDS uses its own cutoff.",
        "max_iterations": "Maximum iterations for robust consensus.",
        "confidence": "Requested confidence for robust consensus.",
        "refine_iterations": "Optional native refinement. Zero uses the closed-form inlier fit.",
        "min_inliers": "Minimum number of inliers required to emit the transform.",
        "min_inlier_ratio": "Minimum fraction of inliers required to emit the transform.",
    }
    input_help = {
        "matches": "Matched query and train point lists, paired in their original order.",
        "train_image": "Optional image from the train branch for green and red match markers.",
    }

    @staticmethod
    def _unavailable(status, count=0):
        fit = {
            "status": status,
            "shift_pixels": None,
            "inlier_mask": [False] * count,
            "inlier_ratio": 0.0,
            "rms_pixels": None,
            "residuals_pixels": [],
        }
        return Result(
            {"transform": NO_VALUE, "inliers": 0, "fit": fit},
            passed=False,
            score=0.0,
            fail_reasons=[status],
            message="No usable translation estimate",
        )

    def process(self, inputs):
        data = inputs["matches"]
        if not isinstance(data, dict) or not {"query_points", "train_points"} <= data.keys():
            raise ValidationError("Connect matches with query_points and train_points")
        query, train = data["query_points"], data["train_points"]
        if query == [] and train == []:
            return self._unavailable("translation_no_matches")
        query, train = (finite_features(points, max_columns=2) for points in (query, train))
        if query.shape != train.shape or query.shape[1] != 2:
            raise ValidationError("Translation needs equal lists of paired [x, y] image points")
        preview = inputs.get("train_image")
        if preview is not None and data.get("train_size") not in (None, list(preview.shape[1::-1])):
            raise ValidationError("Preview image dimensions must match the train image")

        # Pure translation depends only on displacements. Subtract in float64 before OpenCV's
        # native float32 point conversion so a large image origin does not erase small shifts.
        displacements = train - query
        p = self.parameters
        shift, mask = cv2.estimateTranslation2D(
            np.zeros(displacements.shape, dtype=np.float32),
            displacements.astype(np.float32),
            method=_METHODS[p["method"]],
            ransacReprojThreshold=float(p["reprojection_pixels"]),
            maxIters=int(p["max_iterations"]),
            confidence=float(p["confidence"]),
            refineIters=int(p["refine_iterations"]),
        )
        count = len(query)
        if mask is None or mask.size != count or not np.isfinite(shift).all() or not mask.any():
            return self._unavailable("translation_no_consensus", count)

        accepted = mask.reshape(-1).astype(bool)
        inliers = int(accepted.sum())
        score = inliers / count
        residuals = np.linalg.norm(displacements - shift, axis=1)
        rms = float(np.sqrt(np.mean(residuals[accepted] ** 2)))
        reasons = []
        if inliers < int(p["min_inliers"]):
            reasons.append("translation_below_min_inliers")
        if score < float(p["min_inlier_ratio"]):
            reasons.append("translation_below_min_inlier_ratio")
        dx, dy = shift
        fit = {
            "status": "below_acceptance" if reasons else "ok",
            "shift_pixels": [dx, dy],
            "inlier_mask": accepted.tolist(),
            "inlier_ratio": score,
            "rms_pixels": rms,
            "residuals_pixels": residuals.tolist(),
        }
        transform = {
            "model": "image_translation_v1",
            "source_to_target": [[1, 0, dx], [0, 1, dy], [0, 0, 1]],
            "source_size": data.get("query_size"),
            "target_size": data.get("train_size"),
        }
        result = Result(
            {"transform": NO_VALUE if reasons else transform, "inliers": inliers, "fit": fit},
            passed=not reasons,
            score=score,
            fail_reasons=reasons,
            message=(
                f"Shift ({dx:+.3g}, {dy:+.3g}) px · {inliers}/{count} inliers · RMS {rms:.3g} px"
            ),
        )
        if preview is not None and self.generate_graphics:
            markers = tuple(
                graphic_object("marker", (point,), ACCEPTED if good else REJECTED)
                for point, good in zip(train[:_PREVIEW_LIMIT], accepted[:_PREVIEW_LIMIT])
            )
            result.drawings["translation inliers"] = GraphicDrawing(markers, source="train_image")
        return result
