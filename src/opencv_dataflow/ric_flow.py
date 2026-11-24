"""Robust interpolation of sparse image correspondences using OpenCV RIC."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .edge_aware_flow import MatchedPointFlow
from .managed_process import ManagedProcess
from .parameter_validation import NumericRange, ValidationError

MAX_PIXELS = 1_048_576
MAX_MATCHES = 8_192
WORKER_MEMORY = 1024**3
WORKER_SCRIPT = Path(__file__).with_name("ric_worker.py").resolve()

NATIVE_DEFAULTS = {
    "neighbors": 32,
    "alpha": 0.7,
    "superpixel_mode": "slic",
    "superpixel_size": 15,
    "superpixel_neighbors": 150,
    "superpixel_ruler": 15.0,
    "model_iterations": 4,
    "refine_models": True,
    "max_flow": 250.0,
    "variational_refinement": False,
    "post_smoothing": True,
    "fgs_lambda": 500.0,
    "fgs_sigma": 1.5,
}


class RobustCorrespondenceFlow(MatchedPointFlow):
    type_id, title = "cv.ric_flow", "Interpolate motion with RIC"
    description = "Fit robust piecewise affine motion to supplied image correspondences."
    seed_offset = 0.0
    process_inputs_readonly = True
    defaults = {
        **NATIVE_DEFAULTS,
        "support_radius": 0.0,
        "timeout_seconds": 30.0,
        **MatchedPointFlow.defaults,
    }
    parameter_choices = {"superpixel_mode": ("slic", "slico", "mslic")}
    parameter_ranges = {
        "neighbors": NumericRange(4, 256, whole=True),
        "alpha": NumericRange(1e-4, 10),
        "superpixel_size": NumericRange(6, 512, whole=True),
        "superpixel_neighbors": NumericRange(4, 512, whole=True),
        "superpixel_ruler": NumericRange(1e-4, 255),
        "model_iterations": NumericRange(1, 12, whole=True),
        "max_flow": NumericRange(1e-4, 8192),
        "fgs_lambda": NumericRange(0, 100000),
        "fgs_sigma": NumericRange(1e-6, 255),
        "support_radius": NumericRange(0, 8192),
        "timeout_seconds": NumericRange(0.1, 120),
        **MatchedPointFlow.parameter_ranges,
    }
    parameter_help = {
        "neighbors": "Native neighbor limit for the match and superpixel graphs.",
        "alpha": "Controls how quickly the weight of geodesically distant matches falls off.",
        "superpixel_mode": "SLIC variant used to partition the source image.",
        "superpixel_size": "Nominal source superpixel size in pixels.",
        "superpixel_neighbors": "Distinct matches requested per superpixel for model estimation.",
        "superpixel_ruler": "Native compactness or smoothness factor for superpixels.",
        "model_iterations": "Number of native model propagation passes.",
        "refine_models": "Refit affine models with enough inliers.",
        "max_flow": "If a predicted component exceeds this limit, RIC uses a local translation.",
        "variational_refinement": "Refine the field using the source and target images.",
        "post_smoothing": "Apply the native fast global smoother to the flow.",
        "fgs_lambda": "Regularization weight for post-smoothing.",
        "fgs_sigma": "Sensitivity to guide intensity during post-smoothing.",
        "support_radius": "Limit support to this distance from a source seed in pixels. Zero disables.",
        "timeout_seconds": "Time budget for the isolated OpenCV worker.",
        **MatchedPointFlow.parameter_help,
    }
    output_help = {
        **MatchedPointFlow.output_help,
        "support": "Selected in-bounds destinations within the optional source-seed radius.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name in ("refine_models", "variational_refinement", "post_smoothing"):
            if type(p[name]) is not bool:
                raise ValidationError(f"{name.replace('_', ' ').capitalize()} must be boolean", name)

    def validate_workload(self, size, count):
        p = self.parameters
        width, height = size
        if width * height > MAX_PIXELS:
            raise ValidationError(f"RIC source raster is limited to {MAX_PIXELS:,} pixels", "image")
        if min(size) < 2 * int(p["superpixel_size"]):
            raise ValidationError(
                "Both raster dimensions must span at least two superpixels", "superpixel_size"
            )
        if count > MAX_MATCHES:
            raise ValidationError(f"RIC is limited to {MAX_MATCHES:,} distinct source seeds", "matches")
        fields = ("neighbors", "superpixel_neighbors")
        if count < max(int(p[name]) for name in fields):
            raise ValidationError(
                f"Only {count} distinct source seeds for the requested RIC neighbor counts", *fields
            )

    def limit_support(self, valid, source_points):
        radius = float(self.parameters["support_radius"])
        if radius > 0:
            seeds = np.ones(valid.shape, np.uint8)
            x, y = source_points.astype(np.int32).T
            seeds[y, x] = 0
            distance = cv2.distanceTransform(seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            valid &= distance <= radius

    def interpolate(self, guide, source_points, target_guide, target_points, cost):
        if self._cancelled():
            raise InterruptedError("RIC interpolation cancelled")
        if not hasattr(getattr(cv2, "ximgproc", None), "createRICInterpolator"):
            raise ValidationError("RIC interpolation requires OpenCV contrib ximgproc")

        with TemporaryDirectory(prefix="dataflow-ric-") as temporary:
            work = Path(temporary)
            np.savez(
                work / "input.npz", source=guide, target=target_guide, source_points=source_points,
                target_points=target_points, cost=cost if cost is not None else np.empty(0, np.float32),
            )
            settings = {name: self.parameters[name] for name in NATIVE_DEFAULTS}
            worker = ManagedProcess(
                [sys.executable, "-I", str(WORKER_SCRIPT), json.dumps(settings, allow_nan=False)],
                work, os.environ.copy(), memory_limit=WORKER_MEMORY,
            )
            try:
                code = worker.wait(float(self.parameters["timeout_seconds"]), self._cancelled)
                if code:
                    self.logger.error("RIC worker exit %s\n%s", code, worker.tail()[-4096:])
                    raise ValidationError(
                        "RIC worker failed. Check match coverage or reduce superpixel neighbors.",
                        "superpixel_neighbors",
                    )
            except TimeoutError as exc:
                raise TimeoutError("RIC interpolation timed out") from exc
            finally:
                worker.close()
            output = work / "flow.npy"
            height, width = guide.shape[:2]
            if output.stat().st_size > height * width * 8 + 512:
                raise ValidationError("RIC worker returned an oversized flow field")
            flow = np.load(output, allow_pickle=False)
        return flow, "ric"
