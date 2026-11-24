"""OpenCV sparse-to-dense flow with checked interpolation support."""

import cv2
import numpy as np

from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .sequence_tools import _DenseFlowSequence

_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_MAX_WORKSPACE_BYTES = 512 * 1024**2


class SparseToDenseSequence(_DenseFlowSequence):
    type_id, title = "sequence.sparse_to_dense", "Sparse-to-dense optical flow"
    description = "Interpolate sparse Lucas–Kanade matches into an edge-aware dense motion field."
    process_inputs_readonly = True
    defaults = {
        **_DenseFlowSequence.defaults,
        "grid_step": 8,
        "neighbors": 128,
        "sigma": 0.05,
        "post_smoothing": True,
        "fgs_lambda": 500.0,
        "fgs_sigma": 1.5,
    }
    parameter_ranges = {
        **_DenseFlowSequence.parameter_ranges,
        "max_gap_ms": NumericRange(0),
        "visual_max_motion": NumericRange(1e-6),
        "grid_step": NumericRange(2, 512, whole=True),
        "neighbors": NumericRange(4, 512, whole=True),
        "sigma": NumericRange(0.000101, 1),
        "fgs_lambda": NumericRange(1.000001, 100000),
        "fgs_sigma": NumericRange(0.010001, 255),
    }
    parameter_help = {
        **_DenseFlowSequence.parameter_help,
        "grid_step": "Pixel spacing of the candidate points along both axes.",
        "neighbors": "Number of valid sparse matches in each local affine fit.",
        "sigma": "Distance weight falloff for the edge-aware affine fits.",
        "post_smoothing": "Apply the native fast global smoother after interpolation.",
        "fgs_lambda": "Regularization weight for post-smoothing.",
        "fgs_sigma": "Sensitivity to guides' color or intensity changes during smoothing.",
        "visual_max_motion": "Motion in pixels mapped to full preview brightness.",
    }
    input_help = {
        "image": "Images converted to uint8. Color also guides the edge-aware interpolation.",
        "mask": "Select published flow pixels on the previous image grid.",
        "region": "Intersect this region with the optional output mask.",
    }
    output_help = {
        "flow": "Float32 vectors in pixels, from the previous frame to the current frame.",
        "consistency_mask": "White where the forward and reverse estimates agree.",
        "consistency_error": "Round-trip distance in pixels. NaN for unevaluated positions.",
    }

    def working_image(self, image):
        if image.dtype == np.float64 and image_format(image) in ("hsv", "lab"):
            image = ImageArray(image.astype(np.float32), image_format(image))
        native = image_value(image, _DEPTHS, canonical=True)
        if native.ndim == 3:
            native = native[..., :3]
        return native if native.dtype == np.uint8 else convert_depth(native, "uint8")

    def estimate_state_bytes(self, image):
        channels = 1 if image.ndim == 2 else 3
        return image.shape[0] * image.shape[1] * channels + 4096

    def flow_calculator(self, shape):
        height, width = shape[:2]
        if min(height, width) < 2 or max(height, width) > 8192:
            raise ValidationError("Sparse-to-dense flow requires 2–8192 pixels per axis")
        p = self.parameters
        step, neighbors = int(p["grid_step"]), int(p["neighbors"])
        rows, columns = len(range(0, height, step)), len(range(0, width, step))
        seeds = rows * columns
        if min(rows, columns) < 2 or seeds < neighbors or seeds >= 32767:
            raise ValidationError(
                "Grid needs at least two points per axis and enough candidates for the "
                "requested neighbors, with fewer than 32,767 total points",
                "grid_step",
                "neighbors",
            )
        if height * width * 112 + seeds * (64 + 8 * neighbors) > _MAX_WORKSPACE_BYTES:
            raise ValidationError(
                "Sparse-to-dense workspace exceeds 512 MiB. Crop or space the grid farther apart.",
                "grid_step",
                "neighbors",
            )
        create = getattr(getattr(cv2, "ximgproc", None), "createEdgeAwareInterpolator", None)
        if create is None:
            raise ValidationError("Sparse-to-dense flow requires OpenCV contrib ximgproc")

        def calculate(previous, current, _initial):
            # Mirror calcOpticalFlowSparseToDense with a check between PyrLK
            # and interpolation, where rejected matches would be invisible.
            x, y = np.meshgrid(
                np.arange(0, width, step, np.float32), np.arange(0, height, step, np.float32)
            )
            points = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
            prev_gray, cur_gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                for image in (previous, current)
            )
            targets, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, cur_gray, points, None, winSize=(21, 21)
            )
            valid = status.ravel() != 0
            count = int(np.count_nonzero(valid))
            if count < neighbors:
                raise ValidationError(
                    f"Only {count} valid sparse matches for {neighbors} neighbors. "
                    "Use a finer grid, fewer neighbors, or more image texture.",
                    "grid_step",
                    "neighbors",
                )
            matched = targets[valid]
            if not np.isfinite(matched).all():
                raise ValidationError("Sparse motion produced nonfinite matches")

            solver = create()
            solver.setK(neighbors)
            solver.setSigma(float(p["sigma"]))
            solver.setUsePostProcessing(p["post_smoothing"])
            solver.setFGSLambda(float(p["fgs_lambda"]))
            solver.setFGSSigma(float(p["fgs_sigma"]))
            return solver.interpolate(previous, points[valid], current, matched), None

        return calculate
