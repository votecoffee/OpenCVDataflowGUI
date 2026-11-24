"""Native local-motion controls shared by dense and sparse RLOF."""

from .parameter_validation import NumericRange, ValidationError

_SUPPORT = {"fixed": "SR_FIXED", "cross": "SR_CROSS"}
_SOLVER = {"standard": "ST_STANDART", "bilinear": "ST_BILINEAR"}
RLOF_CHOICES = {"support": tuple(_SUPPORT), "solver": tuple(_SOLVER)}
RLOF_DEFAULTS = {
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
}
RLOF_RANGES = {
    "small_window": NumericRange(3, 63, whole=True, odd=True),
    "large_window": NumericRange(3, 63, whole=True, odd=True),
    "cross_threshold": NumericRange(0, 255, whole=True),
    "max_level": NumericRange(0, 8, whole=True),
    "iterations": NumericRange(1, 100, whole=True),
    "min_eigenvalue": NumericRange(0, 1),
    "global_motion_percentile": NumericRange(0, 99.99),
    "norm_sigma0": NumericRange(1e-6, 255),
    "norm_sigma1": NumericRange(1e-6, 255),
}
RLOF_HELP = {
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
}


def validate_rlof(p):
    if p["small_window"] > p["large_window"]:
        raise ValidationError(
            "Small window must be no larger than the large window", "small_window", "large_window"
        )
    if p["m_estimator"] and float(p["norm_sigma0"]) >= float(p["norm_sigma1"]):
        raise ValidationError(
            "The first Hampel cutoff must be smaller than the second", "norm_sigma0", "norm_sigma1"
        )


def rlof_settings(optflow, create, p, *, initial=False):
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
    local.setUseInitialFlow(initial)
    local.setUseMEstimator(p["m_estimator"])
    if p["m_estimator"]:
        local.setNormSigma0(float(p["norm_sigma0"]))
        local.setNormSigma1(float(p["norm_sigma1"]))
    return local
