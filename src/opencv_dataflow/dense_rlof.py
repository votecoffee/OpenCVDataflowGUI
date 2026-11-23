"""OpenCV contrib Dense RLOF with ticketed previous-frame coordinates."""

import cv2
import numpy as np

from .images import ImageArray, convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value
from .segmentation import bgr8
from .sequence import SEQUENCE_CHOICES
from .sequence_tools import _DenseFlowSequence

_DEPTHS = (np.uint8, np.uint16, np.float32, np.float64)
_SUPPORT = {"fixed": "SR_FIXED", "cross": "SR_CROSS"}
_SOLVER = {"standard": "ST_STANDART", "bilinear": "ST_BILINEAR"}
_INTERPOLATION = {"geodesic": "INTERP_GEO", "epic": "INTERP_EPIC", "ric": "INTERP_RIC"}


class DenseRLOFSequence(_DenseFlowSequence):
    type_id, title = "sequence.dense_rlof", "Dense RLOF optical flow"
    description = "Measure dense motion with robust local flow and optional seed interpolation."
    defaults = {
        **_DenseFlowSequence.defaults,
        "support": "cross",
        "solver": "bilinear",
        "small_window": 9,
        "large_window": 21,
        "cross_threshold": 25,
        "max_level": 4,
        "iterations": 30,
        "min_eigenvalue": 0.0001,
        "illumination_model": True,
        "global_motion_prior": True,
        "global_motion_percentile": 10.0,
        "m_estimator": False,
        "norm_sigma0": 3.2,
        "norm_sigma1": 7.0,
        "seed_consistency_threshold": 0.0,
        "grid_x": 6,
        "grid_y": 6,
        "interpolation": "epic",
        "epic_neighbors": 128,
        "epic_sigma": 0.05,
        "epic_lambda": 100.0,
        "ric_neighbors": 16,
        "ric_superpixel_size": 15,
        "ric_superpixel_mode": "SLIC",
        "post_smoothing": True,
        "fgs_lambda": 500.0,
        "fgs_sigma": 1.5,
        "variational_refinement": False,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "support": tuple(_SUPPORT),
        "solver": tuple(_SOLVER),
        "interpolation": tuple(_INTERPOLATION),
        "ric_superpixel_mode": ("SLIC", "SLICO", "MSLIC"),
    }
    parameter_ranges = {
        **_DenseFlowSequence.parameter_ranges,
        "max_gap_ms": NumericRange(0),
        "visual_max_motion": NumericRange(1e-6),
        "small_window": NumericRange(3, 63, whole=True, odd=True),
        "large_window": NumericRange(3, 63, whole=True, odd=True),
        "cross_threshold": NumericRange(0, 255, whole=True),
        "max_level": NumericRange(0, 8, whole=True),
        "iterations": NumericRange(1, 100, whole=True),
        "min_eigenvalue": NumericRange(0, 1),
        "global_motion_percentile": NumericRange(0, 99.99),
        "norm_sigma0": NumericRange(1e-6, 255),
        "norm_sigma1": NumericRange(1e-6, 255),
        "seed_consistency_threshold": NumericRange(0, 100),
        "grid_x": NumericRange(1, 512, whole=True),
        "grid_y": NumericRange(1, 512, whole=True),
        "epic_neighbors": NumericRange(3, 256, whole=True),
        "epic_sigma": NumericRange(1e-6, 1),
        "epic_lambda": NumericRange(0, 10000),
        "ric_neighbors": NumericRange(3, 32, whole=True),
        "ric_superpixel_size": NumericRange(8, 128, whole=True),
        "fgs_lambda": NumericRange(0, 10000),
        "fgs_sigma": NumericRange(1e-6, 255),
    }
    parameter_help = {
        **_DenseFlowSequence.parameter_help,
        "support": "Fixed windows or color-adaptive cross support.",
        "solver": "Standard or bilinear local motion solver.",
        "small_window": "Minimum local support window side in pixels.",
        "large_window": "Maximum local support window side in pixels.",
        "cross_threshold": "Maximum 8-bit channel difference when extending cross support.",
        "max_level": "Maximum zero-based pyramid level.",
        "iterations": "Maximum local iterations per level.",
        "min_eigenvalue": "Reject local solves below this structure tensor eigenvalue.",
        "illumination_model": "Use an illumination variation model in local solves.",
        "global_motion_prior": "Homography motion prior. Requires 40 × 30 pixels.",
        "global_motion_percentile": "Motion percentile used for the homography RANSAC threshold.",
        "m_estimator": "Enable the robust Hampel estimator. Off uses least squares.",
        "norm_sigma0": "First Hampel residual cutoff when the M-estimator is enabled.",
        "norm_sigma1": "Second Hampel residual cutoff when the M-estimator is enabled.",
        "seed_consistency_threshold": "Geodesic only. Reject seed round trips at this pixel error.",
        "grid_x": "Horizontal seed spacing. A 1 × 1 grid without seed rejection is direct.",
        "grid_y": "Vertical seed spacing in pixels. A 1 × 1 grid without seed rejection is direct.",
        "interpolation": "Geodesic, edge-aware (EPIC), or superpixel-based (RIC) interpolation.",
        "epic_neighbors": "Neighbor count for the EPIC affine fit. Must fit the seed grid.",
        "epic_sigma": "Distance falloff for EPIC interpolation weights.",
        "epic_lambda": "Image-edge penalty for EPIC interpolation.",
        "ric_neighbors": "Maximum neighbor count used to connect RIC seeds and superpixels.",
        "ric_superpixel_size": "Approximate square side of a RIC superpixel in pixels.",
        "ric_superpixel_mode": "SLIC, SLICO, or MSLIC superpixel construction for RIC.",
        "post_smoothing": "Apply fast global smoothing after interpolation.",
        "fgs_lambda": "Fast global smoother regularization weight.",
        "fgs_sigma": "Fast global smoother color sensitivity.",
        "variational_refinement": "Apply grayscale variational refinement after interpolation.",
        "visual_max_motion": "Displacement in pixels mapped to full preview brightness.",
    }
    input_help = {
        "image": "Intensity or color image. Converted to 8-bit BGR for native RLOF.",
        "mask": "Select published flow pixels on the previous image grid.",
        "region": "Select published pixels on the previous image grid, intersected with any mask.",
    }
    output_help = {
        "flow": "Float32 displacement in pixels from the previous frame to the current frame.",
        "consistency_mask": "White where the interpolated forward and reverse estimates agree.",
        "consistency_error": "Round-trip distance in pixels. NaN for unevaluated pixels.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["small_window"] > p["large_window"]:
            raise ValidationError(
                "Small window must be no larger than the large window",
                "small_window",
                "large_window",
            )
        if p["m_estimator"] and float(p["norm_sigma0"]) >= float(p["norm_sigma1"]):
            raise ValidationError(
                "The first Hampel cutoff must be smaller than the second",
                "norm_sigma0",
                "norm_sigma1",
            )
        if float(p["seed_consistency_threshold"]) and p["interpolation"] != "geodesic":
            raise ValidationError(
                "Seed rejection requires geodesic interpolation. EPIC and RIC require a full grid.",
                "seed_consistency_threshold",
                "interpolation",
            )

    def working_image(self, image):
        native = image_value(image, _DEPTHS)
        encoding = image_format(image)
        if native.dtype == np.float64:
            image = ImageArray(convert_depth(native, "float32"), encoding)
        if image.dtype == np.uint8:
            return bgr8(image)
        canonical = image_value(image, _DEPTHS, canonical=True)
        return bgr8(convert_depth(canonical, "uint8"))

    def estimate_state_bytes(self, image):
        return image.shape[0] * image.shape[1] * 3 + 4096

    def flow_calculator(self, shape):
        p = self.parameters
        height, width = shape[:2]
        if min(height, width) < int(p["large_window"]):
            raise ValidationError("Dense RLOF image dimensions must fit the large window")
        if p["global_motion_prior"] and (height < 30 or width < 40):
            # The upstream homography sampling stride is integer size divided by 30 or 40.
            raise ValidationError("Global motion prior requires at least 40 × 30 pixels")
        if max(height, width) > 8192 or height * width > 2_097_152:
            raise ValidationError("Dense RLOF allows up to 2,097,152 pixels and 8,192 per axis")
        gx, gy = int(p["grid_x"]), int(p["grid_y"])
        seed_count = len(range(gx // 2, width - gx // 2, gx)) * len(
            range(gy // 2, height - gy // 2, gy)
        )
        seed_threshold = float(p["seed_consistency_threshold"])
        direct = gx == gy == 1 and seed_threshold == 0
        mode = p["interpolation"]
        neighbors = int(p["ric_neighbors"] if mode == "ric" else p["epic_neighbors"])
        if not 1 <= seed_count <= (262_144 if direct else 32_766):
            raise ValidationError("Dense RLOF seed grid is empty or exceeds the seed limit")
        if not direct and mode in ("epic", "ric") and seed_count < neighbors:
            raise ValidationError("Interpolation neighbors exceed the number of grid seeds")
        if not direct and mode == "ric" and 2 * int(p["ric_superpixel_size"]) > min(height, width):
            raise ValidationError("RIC requires room for at least two superpixels along each axis")

        optflow = getattr(cv2, "optflow", None)
        calc = getattr(optflow, "calcOpticalFlowDenseRLOF", None)
        create = getattr(optflow, "RLOFOpticalFlowParameter_create", None)
        if calc is None or create is None:
            raise ValidationError("Dense RLOF requires OpenCV contrib optflow")
        local = create()
        local.setSupportRegionType(getattr(optflow, _SUPPORT[p["support"]]))
        local.setSolverType(getattr(optflow, _SOLVER[p["solver"]]))
        local.setSmallWinSize(int(p["small_window"]))
        local.setLargeWinSize(int(p["large_window"]))
        local.setCrossSegmentationThreshold(int(p["cross_threshold"]))
        local.setMaxLevel(int(p["max_level"]))
        local.setMaxIteration(int(p["iterations"]))
        local.setMinEigenValue(float(p["min_eigenvalue"]))
        local.setUseIlluminationModel(p["illumination_model"])
        local.setUseGlobalMotionPrior(p["global_motion_prior"])
        local.setGlobalMotionRansacThreshold(float(p["global_motion_percentile"]))
        local.setUseMEstimator(p["m_estimator"])
        if p["m_estimator"]:
            local.setNormSigma0(float(p["norm_sigma0"]))
            local.setNormSigma1(float(p["norm_sigma1"]))
        interpolation = getattr(optflow, _INTERPOLATION[mode])
        slic = getattr(cv2.ximgproc, p["ric_superpixel_mode"])

        def calculate(previous, current, _initial):
            flow = calc(
                previous,
                current,
                None,
                local,
                seed_threshold,
                (gx, gy),
                interpolation,
                neighbors,
                float(p["epic_sigma"]),
                float(p["epic_lambda"]),
                int(p["ric_superpixel_size"]),
                slic,
                p["post_smoothing"],
                float(p["fgs_lambda"]),
                float(p["fgs_sigma"]),
                p["variational_refinement"],
            )
            return flow, None

        return calculate
