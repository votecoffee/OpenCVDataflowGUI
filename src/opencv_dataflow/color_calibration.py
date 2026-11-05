"""Detect color charts, fit OpenCV CCMs, and apply portable correction models."""

import copy
import json

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .images import image_format
from .payloads import NO_VALUE
from .processing import encoded_result, finite, image_value, integer
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask

CHARTS = {
    "macbeth": (cv2.mcc.MCC24, cv2.ccm.COLORCHECKER_MACBETH, 24),
    "digital_sg": (cv2.mcc.SG140, cv2.ccm.COLORCHECKER_DIGITAL_SG, 140),
    "vinyl": (cv2.mcc.VINYL18, cv2.ccm.COLORCHECKER_VINYL, 18),
}
MATRICES = {"linear": cv2.ccm.CCM_LINEAR, "affine": cv2.ccm.CCM_AFFINE}
LINEARIZATION = {
    "identity": cv2.ccm.LINEARIZATION_IDENTITY,
    "gamma": cv2.ccm.LINEARIZATION_GAMMA,
    "color_polynomial": cv2.ccm.LINEARIZATION_COLORPOLYFIT,
}
DISTANCES = {
    "ciede2000": cv2.ccm.DISTANCE_CIE2000,
    "cie76": cv2.ccm.DISTANCE_CIE76,
    "cie94_graphic_arts": cv2.ccm.DISTANCE_CIE94_GRAPHIC_ARTS,
    "cie94_textiles": cv2.ccm.DISTANCE_CIE94_TEXTILES,
    "cmc_1_to_1": cv2.ccm.DISTANCE_CMC_1TO1,
    "cmc_2_to_1": cv2.ccm.DISTANCE_CMC_2TO1,
    "rgb": cv2.ccm.DISTANCE_RGB,
    "linear_rgb": cv2.ccm.DISTANCE_RGBL,
}


class DetectColorChart(Block):
    type_id, title = "calibration.detect_color_chart", "Detect color chart"
    category, block_type, subtype = "Calibration", "Calibration", "Color"
    inputs = SELECTION_INPUTS
    outputs = (Port("charts", "data"), Port("count", "number"), Port("overlay", required=False))
    defaults = {
        "chart": "macbeth",
        "maximum_charts": 1,
        "minimum_charts": 1,
        "maximum_detection_error": 0.1,
        "minimum_image_size": 1000,
        "draw_overlay": True,
    }
    parameter_choices = {"chart": tuple(CHARTS)}
    parameter_help = {
        "maximum_detection_error": "Maximum matching error accepted by the OpenCV chart detector.",
        "minimum_image_size": "Small inputs are enlarged to this minimum size for detection.",
    }

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image = image_value(source, (np.uint8,), canonical=True)
        if image.ndim != 3:
            raise ValueError("Color chart detection requires a uint8 color image")
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        maximum = integer(p["maximum_charts"], "Maximum charts", 1, 16)
        minimum = integer(p["minimum_charts"], "Minimum charts", 0, maximum)
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        selection = selection_mask(inputs, image)
        items = []
        if selection is None or selection.any():
            origin = np.zeros(2)
            if selection is not None:
                x, y, w, h = cv2.boundingRect(selection)
                origin = np.array([x, y])
                image = image[y : y + h, x : x + w]
            if min(image.shape[:2]) >= 12:
                detector = cv2.mcc.CCheckerDetector.create()
                detector.setColorChartType(CHARTS[p["chart"]][0])
                options = cv2.mcc.DetectorParametersMCC()
                options.maxError = finite(
                    p["maximum_detection_error"], "Maximum detection error", 0
                )
                options.minImageSize = integer(
                    p["minimum_image_size"], "Minimum image size", 64, 4096
                )
                detector.setDetectionParams(options)
                if detector.process(image, maximum):
                    for checker in detector.getListColorChecker():
                        corners = np.asarray(checker.getBox()) + origin
                        patches = np.asarray(checker.getColorCharts()).reshape(-1, 4, 2) + origin
                        if any(not contains(selection, xy) for xy in patches.mean(axis=1)):
                            continue
                        rgb = checker.getChartsRGB(False).reshape(-1, 3) / 255.0
                        items.append(
                            {
                                "chart": p["chart"],
                                "rgb": rgb.tolist(),
                                "corners": corners.tolist(),
                                "patch_corners": patches.tolist(),
                                "detection_cost": float(checker.getCost()),
                            }
                        )
                        if overlay is not NO_VALUE:
                            cv2.polylines(
                                overlay,
                                [np.rint(corners).astype(np.int32)],
                                True,
                                (70, 220, 140),
                                2,
                            )
        items.sort(key=lambda item: item["detection_cost"])
        passed = len(items) >= minimum
        return Result(
            {
                "charts": {"model": "detected_color_charts_v1", "items": items},
                "count": len(items),
                "overlay": overlay,
            },
            graphics={"color charts": overlay} if overlay is not NO_VALUE else {},
            passed=passed,
            message=f"Detected {len(items)} color charts",
            fail_reasons=[] if passed else ["color_chart_not_found"],
        )


def rgb_patches(value):
    rgb = np.asarray(value, np.float64)
    if (
        rgb.ndim != 2
        or rgb.shape[1] != 3
        or not 3 <= len(rgb) <= 1024
        or not np.isfinite(rgb).all()
        or ((rgb < 0) | (rgb > 1)).any()
    ):
        raise ValueError("Provide 3–1024 RGB patch triplets normalized to [0, 1]")
    return rgb


class FitColorCorrection(Block):
    type_id, title = "calibration.fit_ccm", "Fit color correction model"
    category, block_type, subtype = "Calibration", "Calibration", "Color"
    inputs = (Port("patches", "data"), Port("reference", "data", required=False))
    outputs = (Port("model", "data"), Port("matrix", "data"), Port("statistics", "data"))
    defaults = {
        "chart_index": 0,
        "reference_chart": "auto",
        "reference_rgb": [],
        "reference_encoding": "srgb",
        "matrix_type": "linear",
        "distance": "ciede2000",
        "linearization": "gamma",
        "gamma": 2.2,
        "polynomial_degree": 3,
        "initial": "least_squares",
        "saturation_low": 0.0,
        "saturation_high": 0.98,
        "weights": [],
        "lightness_weight_exponent": 0.0,
        "maximum_iterations": 500,
        "epsilon": 1e-4,
    }
    parameter_choices = {
        "reference_chart": ("auto", *CHARTS, "custom"),
        "reference_encoding": ("srgb", "linear_srgb"),
        "matrix_type": tuple(MATRICES),
        "distance": tuple(DISTANCES),
        "linearization": tuple(LINEARIZATION),
        "initial": ("least_squares", "white_balance"),
    }
    parameter_help = {
        "chart_index": "Zero-based index of a detected chart, ordered by detection cost.",
        "reference_chart": "Auto uses the detected chart type, or Macbeth for raw patches.",
        "reference_rgb": "Custom [R,G,B] values in [0, 1], in the same order as the samples.",
        "weights": "Optional weights, one per patch. Zero excludes a patch.",
        "lightness_weight_exponent": "Native weighting by reference Lab lightness.",
        "saturation_low": "Ignore patches with an RGB channel below this value.",
        "saturation_high": "Ignore patches with an RGB channel above this value.",
    }

    def process(self, inputs):
        p, patches = self.parameters, inputs["patches"]
        chart = "macbeth"
        if isinstance(patches, dict):
            items = patches.get("items")
            if patches.get("model") != "detected_color_charts_v1" or not isinstance(items, list):
                raise ValueError("Connect detected charts or provide RGB patches")
            index = integer(p["chart_index"], "Chart index", 0, len(items) - 1)
            chart, patches = items[index]["chart"], items[index]["rgb"]
        samples = rgb_patches(patches)
        source = samples[:, None, :]
        reference = inputs.get("reference", p["reference_rgb"])
        chart = chart if p["reference_chart"] == "auto" else p["reference_chart"]
        if "reference" in inputs or p["reference_chart"] == "custom":
            colors = rgb_patches(reference)
            if colors.shape != samples.shape:
                raise ValueError("Custom reference must have one RGB triplet per sample")
            encoding = (
                cv2.ccm.COLOR_SPACE_SRGBL
                if p["reference_encoding"] == "linear_srgb"
                else cv2.ccm.COLOR_SPACE_SRGB
            )
            model = cv2.ccm.ColorCorrectionModel(source, colors[:, None, :], encoding)
        else:
            if chart not in CHARTS or len(samples) != CHARTS[chart][2]:
                raise ValueError("Patch count must match the selected chart reference")
            model = cv2.ccm.ColorCorrectionModel(source, CHARTS[chart][1])
        low, high = (finite(p[k], k, 0) for k in ("saturation_low", "saturation_high"))
        if not low < high <= 1:
            raise ValueError("Saturation bounds must satisfy 0 <= low < high <= 1")
        weight_exponent = finite(p["lightness_weight_exponent"], "Lightness weight exponent")
        if p["matrix_type"] == "affine" and (p["weights"] or weight_exponent):
            raise ValueError("The OpenCV 5 affine fit requires unweighted patches")
        retained = np.all((samples >= low) & (samples <= high), axis=1)
        if p["weights"]:
            weights = np.asarray(p["weights"], np.float64)
            if (
                weights.shape != (len(samples),)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or weights.max() <= 0
            ):
                raise ValueError("Provide one finite nonnegative weight per patch")
            retained &= weights > 0
            model.setWeightsList((weights / weights.max())[:, None])
        design = samples[retained]
        if p["matrix_type"] == "affine":
            design = np.column_stack((design, np.ones(len(design))))
        if len(design) < design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("Retained chart patches do not span the selected correction model")
        model.setCcmType(MATRICES[p["matrix_type"]])
        model.setColorSpace(cv2.ccm.COLOR_SPACE_SRGB)
        model.setDistance(DISTANCES[p["distance"]])
        model.setLinearization(LINEARIZATION[p["linearization"]])
        model.setLinearizationGamma(finite(p["gamma"], "Gamma", 0.001))
        model.setLinearizationDegree(integer(p["polynomial_degree"], "Polynomial degree", 1, 8))
        model.setInitialMethod(
            cv2.ccm.INITIAL_METHOD_WHITE_BALANCE
            if p["initial"] == "white_balance"
            else cv2.ccm.INITIAL_METHOD_LEAST_SQUARE
        )
        model.setSaturatedThreshold(low, high)
        model.setWeightCoeff(weight_exponent)
        model.setMaxCount(integer(p["maximum_iterations"], "Maximum iterations", 1, 100_000))
        model.setEpsilon(finite(p["epsilon"], "Epsilon", 1e-12))
        matrix = model.compute()
        loss = float(model.getLoss())
        if matrix is None or not np.isfinite(matrix).all() or not np.isfinite(loss):
            raise ValueError("Color correction fit produced nonfinite values")
        storage = cv2.FileStorage("ccm.json", cv2.FileStorage_WRITE | cv2.FileStorage_MEMORY)
        model.write(storage)
        native = json.loads(storage.releaseAndGetString())["ColorCorrectionModel"]
        used = int(np.count_nonzero(model.getMask()))
        return Result(
            {
                "model": {"model": "color_correction_v1", "opencv": native},
                "matrix": matrix.tolist(),
                "statistics": {"loss": loss, "used_patches": used, "sample_patches": len(samples)},
            },
            message=f"Fitted CCM using {used} patches, loss {loss:.5g}",
        )


def valid_native_matrix(matrix, rows, columns):
    if (
        not isinstance(matrix, dict)
        or set(matrix) != {"type_id", "rows", "cols", "dt", "data"}
        or (matrix["type_id"], matrix["rows"], matrix["cols"], matrix["dt"])
        != ("opencv-matrix", rows, columns, "d")
    ):
        return False
    values = np.asarray(matrix["data"], np.float64)
    return values.shape == (rows * columns,) and np.isfinite(values).all()


def load_ccm(value):
    if not isinstance(value, dict) or value.get("model") != "color_correction_v1":
        raise ValueError("Connect or configure a fitted color correction model")
    native = value.get("opencv")
    if not isinstance(native, dict) or native.get("csEnum") != cv2.ccm.COLOR_SPACE_SRGB:
        raise ValueError("Color correction model must target sRGB")
    matrix = native.get("ccm", {})
    expected_rows = {cv2.ccm.CCM_LINEAR: 3, cv2.ccm.CCM_AFFINE: 4}.get(native.get("ccm_type"))
    linear_type = native.get("linear_type")
    if (
        expected_rows is None
        or linear_type not in LINEARIZATION.values()
        or not valid_native_matrix(matrix, expected_rows, 3)
    ):
        raise ValueError("Invalid color correction model matrix or encoding")
    gamma = finite(native.get("gamma"), "Model gamma", 0.001)
    degree = integer(native.get("deg"), "Model polynomial degree", 1, 8)
    linear = native.get("linear")
    if linear_type == LINEARIZATION["gamma"]:
        linear = {"gamma": finite(linear["gamma"], "Model gamma", 0.001)}
    elif linear_type == LINEARIZATION["color_polynomial"]:
        polynomial = {"deg": degree}
        for channel in ("pr", "pg", "pb"):
            record = linear[channel]
            if record.get("deg") != degree or not valid_native_matrix(
                record.get("p"), degree + 1, 1
            ):
                raise ValueError("Invalid color correction polynomial")
            polynomial[channel] = {"deg": degree, "p": record["p"]}
        linear = polynomial
    else:
        linear = {}
    distance = native.get("distance")
    low, high = native.get("saturated_threshold", (0, 0.98))
    low, high = finite(low, "Model saturation low", 0), finite(high, "Model saturation high", 0)
    if distance not in DISTANCES.values() or not low < high <= 1:
        raise ValueError("Invalid color correction distance or saturation bounds")
    safe = {
        "ccm": matrix,
        "loss": finite(native.get("loss"), "Model loss", 0),
        "csEnum": cv2.ccm.COLOR_SPACE_SRGB,
        "ccm_type": native["ccm_type"],
        "shape": expected_rows * 3,
        "linear": linear,
        "distance": distance,
        "linear_type": linear_type,
        "gamma": gamma,
        "deg": degree,
        "saturated_threshold": [low, high],
    }
    encoded = json.dumps({"ColorCorrectionModel": safe}, allow_nan=False)
    if len(encoded) > 16_384:
        raise ValueError("Color correction model exceeds its size limit")
    storage = cv2.FileStorage(encoded, cv2.FileStorage_READ | cv2.FileStorage_MEMORY)
    model = cv2.ccm.ColorCorrectionModel()
    try:
        model.read(storage.getNode("ColorCorrectionModel"))
    finally:
        storage.release()
    if model.getColorCorrectionMatrix() is None:
        raise ValueError("OpenCV could not restore the fitted color correction model")
    model.setRGB(False)
    return model


class ApplyColorCorrection(RegionalOperation, Block):
    type_id, title = "calibration.apply_ccm", "Apply color correction model"
    category, block_type, subtype = "Calibration", "Calibration", "Color"
    inputs = (*SELECTION_INPUTS, Port("model", "data", required=False))
    outputs = (Port("image"),)
    defaults = {"model": {}}
    parameter_help = {"model": "Stored color correction JSON. A wired model takes precedence."}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._key, self._model = None, None

    close = reset_runtime

    def process(self, inputs):
        source = inputs["image"]
        encoding = image_format(source)
        image = image_value(source, (np.uint8, np.uint16, np.float32))
        if encoding not in ("bgr", "rgb", "bgra", "rgba"):
            raise ValueError("Color correction requires BGR, RGB, BGRA or RGBA input")
        if image.dtype == np.float32 and ((image < 0) | (image > 1)).any():
            raise ValueError("Floating-point color correction requires normalized channels")
        value = inputs.get("model", self.parameters["model"])
        if value != self._key:
            self._model = load_ccm(value)
            self._key = copy.deepcopy(value)
        # Supply RGB to the matrix directly and handle channel order at the block boundary.
        rgb = image[..., :3] if encoding.startswith("rgb") else image[..., 2::-1]
        corrected = self._model.correctImage(np.ascontiguousarray(rgb))
        if not np.isfinite(corrected).all():
            raise ValueError("Color correction produced nonfinite pixels")
        if corrected.dtype == np.float32:
            corrected = np.clip(corrected, 0, 1)
        if encoding.startswith("bg"):
            corrected = corrected[..., ::-1].copy()
        if image.shape[2] == 4:
            corrected = np.concatenate((corrected, image[..., 3:]), axis=2)
        return encoded_result(corrected, source)


COLOR_CALIBRATION_BLOCKS = (DetectColorChart, FitColorCorrection, ApplyColorCorrection)
