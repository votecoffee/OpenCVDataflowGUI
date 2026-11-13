"""Calibrated channel alignment on the green pixel grid using OpenCV 5."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value
from .selection import selection_mask

MODEL = "opencv_chromatic_aberration_v1"
MAX_PIXELS = 2_100_000
MAX_DEGREE = 11
ROWS = ("blue_dx", "blue_dy", "red_dx", "red_dy")


def make_model(size, degree, coefficients, camera_label="", lens_label=""):
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(v) is not int or not 3 <= v <= 16384 for v in size)
        or size[0] * size[1] > MAX_PIXELS
    ):
        raise ValidationError(
            "Calibration size must be integer [width,height], 3..16384, up to 2.1M pixels"
        )
    if type(degree) is not int or not 0 <= degree <= MAX_DEGREE:
        raise ValidationError("Calibration degree must be an integer from 0 to 11")
    for value in (camera_label, lens_label):
        if not isinstance(value, str) or len(value) > 128:
            raise ValidationError("Camera/lens labels must be text up to 128 characters")
    terms = (degree + 1) * (degree + 2) // 2
    try:
        matrix = np.asarray(coefficients, np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Coefficients must be four numeric rows") from exc
    if matrix.shape != (4, terms) or not np.isfinite(matrix).all() or np.max(np.abs(matrix)) > 1e6:
        raise ValidationError(f"Use 4 x {terms} finite coefficients within +/-1e6")
    record = {
        "model": MODEL,
        "image_size": list(size),
        "degree": degree,
        "coefficients": matrix.astype(np.float32).tolist(),
        "row_order": list(ROWS),
        "camera_label": camera_label,
        "lens_label": lens_label,
    }
    record["id"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return record


def validate_model(model):
    if (
        not isinstance(model, dict)
        or model.get("model") != MODEL
        or model.get("row_order") != list(ROWS)
    ):
        raise ValidationError("Connect a chromatic calibration model with native Bx,By,Rx,Ry order")
    record = make_model(
        model.get("image_size"),
        model.get("degree"),
        model.get("coefficients"),
        model.get("camera_label", ""),
        model.get("lens_label", ""),
    )
    if model.get("id") != record["id"]:
        raise ValidationError(
            "Chromatic calibration fingerprint differs; recreate or reload the model"
        )
    return record


def donor_coordinates(x, y, model):
    """Native polynomial convention, evaluated in bounded strips by the caller.

    Terms are 1,y,x,y²,xy,x²,... on u=(x-W/2)/(W/2), v=(y-H/2)/(H/2).
    Native displacement and final donor planes are rounded to float32 separately.
    This computes support/probes, not a substitute for the native image correction.
    """
    width, height = model["image_size"]
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    u = (x - width * 0.5) * (1.0 / (width * 0.5))
    v = (y - height * 0.5) * (1.0 / (height * 0.5))
    powers_u, powers_v = [np.ones_like(u)], [np.ones_like(v)]
    for _ in range(model["degree"]):
        powers_u.append(powers_u[-1] * u)
        powers_v.append(powers_v[-1] * v)
    delta = np.zeros((4, *x.shape), np.float64)
    coefficients = np.asarray(model["coefficients"], np.float32)
    index = 0
    for order in range(model["degree"] + 1):
        for px in range(order + 1):
            term = powers_u[px] * powers_v[order - px]
            for row in range(4):
                delta[row] += float(coefficients[row, index]) * term
            index += 1
    maps = delta.astype(np.float32)
    for row, coordinate in enumerate((x, y, x, y)):
        maps[row] = coordinate.astype(np.float32) - maps[row]
    return maps


def support(model, bayer, maximum):
    width, height = model["image_size"]
    valid = np.zeros((height, width), np.uint8)
    peak = [0.0, 0.0]
    margin = int(bayer)  # Bilinear demosaicing synthesizes the outer sensor border.
    rows = max(1, min(64, 131072 // width))
    x = np.arange(width, dtype=np.float64)[None, :]
    for top in range(0, height, rows):
        y = np.arange(top, min(height, top + rows), dtype=np.float64)[:, None]
        maps = donor_coordinates(x, y, model)
        good = np.ones(maps.shape[1:], bool)
        for channel in (0, 1):
            mx, my = maps[2 * channel : 2 * channel + 2]
            peak[channel] = max(peak[channel], float(np.hypot(mx - x, my - y).max()))
            good &= (mx >= margin) & (mx <= width - 1 - margin)
            good &= (my >= margin) & (my <= height - 1 - margin)
        if max(peak) > maximum:
            raise ValidationError("Calibration exceeds maximum_displacement in the current grid")
        if bayer:
            good &= (x >= margin) & (x <= width - 1 - margin)
            good &= (y >= margin) & (y <= height - 1 - margin)
        valid[top : top + len(y)] = good.astype(np.uint8) * 255
    return valid, peak


class ChromaticModel(Block):
    type_id, title = "calibration.chromatic_model", "Define chromatic calibration"
    category = block_type = "Calibration"
    subtype = "Chromatic aberration"
    outputs = (Port("calibration", "data"),)
    process_inputs_readonly = parallel_safe = True
    defaults = {
        "width": 640,
        "height": 480,
        "degree": 1,
        "coefficients": [[0.0, 0.0, 0.0] for _ in range(4)],
        "camera_label": "",
        "lens_label": "",
    }
    parameter_ranges = {
        "width": NumericRange(3, 16384, whole=True),
        "height": NumericRange(3, 16384, whole=True),
        "degree": NumericRange(0, 11, whole=True),
    }
    parameter_help = {
        "width": "Width of the exact pixel grid used for this calibration, not a resized image.",
        "height": "Height of that grid; up to 2.1M pixels in total.",
        "degree": "Polynomial degree 0..11; each row needs (degree+1)*(degree+2)/2 terms.",
        "coefficients": "Rows B dx, B dy, R dx, R dy in pixels relative to green. "
        "Terms 1,y,x,y²,xy,x²,... on coordinates normalized to the image half-size. "
        "Correction samples (x-dx,y-dy). Values are stored as native float32.",
        "camera_label": "Optional camera identity label for identifying the calibration setup.",
        "lens_label": "Optional lens/focus/aperture label. Retain those settings at runtime.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        make_model(
            [p["width"], p["height"]],
            p["degree"],
            p["coefficients"],
            p["camera_label"],
            p["lens_label"],
        )

    def process(self, inputs):
        p = self.parameters
        return Result(
            {
                "calibration": make_model(
                    [p["width"], p["height"]],
                    p["degree"],
                    p["coefficients"],
                    p["camera_label"],
                    p["lens_label"],
                )
            }
        )


class LoadChromaticModel(Block):
    type_id, title = "input.chromatic_calibration", "Load chromatic calibration"
    category = block_type = "Input"
    subtype = "Calibration profiles"
    outputs = (Port("calibration", "data"), Port("info", "data"))
    process_inputs_readonly = parallel_safe = True
    defaults = {
        "path": "",
        "path_base": "project",
        "node": "",
        "camera_label": "",
        "lens_label": "",
    }
    parameter_choices = {"path_base": ("project", "working_directory")}
    parameter_help = {
        "path": "OpenCV UTF-8 YAML/XML/JSON calibration (up to 256 KiB). "
        "Reads red_channel/blue_channel coeffs_x/coeffs_y and image_width/image_height.",
        "path_base": "Resolve relative paths beside the saved project or working directory.",
        "node": "Optional literal root mapping key; empty uses the storage root.",
        "camera_label": ChromaticModel.parameter_help["camera_label"],
        "lens_label": ChromaticModel.parameter_help["lens_label"],
    }

    def process(self, inputs):
        p = self.parameters
        if not isinstance(p["path"], str) or not p["path"].strip():
            raise ValidationError("Select a calibration file", "path")
        path = Path(p["path"])
        if p["path_base"] == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError(
                    "Save/open the project before using a relative calibration path"
                )
            path = self.project_dir / path
        with path.open("rb") as stream:
            content = stream.read(262145)
        if not 0 < len(content) <= 262144:
            raise ValidationError("Calibration file must contain 1..262144 bytes")
        if not isinstance(p["node"], str) or len(p["node"]) > 128:
            raise ValidationError("Node must be a root mapping key up to 128 characters", "node")
        storage = cv2.FileStorage(
            content.decode("utf-8-sig"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
        )
        try:
            if not storage.isOpened():
                raise ValidationError("Unable to read OpenCV calibration storage")
            node = storage.getNode(p["node"]) if p["node"] else storage.root()
            if not node.isMap():
                raise ValidationError("Calibration node must be a mapping")
            size = []
            for key in ("image_width", "image_height"):
                value = node.getNode(key)
                if not value.isInt() or not 3 <= value.real() <= 16384:
                    raise ValidationError(
                        "Calibration requires valid integer image_width/image_height"
                    )
                size.append(int(value.real()))
            # Check sequence sizes before asking native code to pack any coefficients.
            lengths = []
            for channel in ("blue_channel", "red_channel"):
                for key in ("coeffs_x", "coeffs_y"):
                    seq = node.getNode(channel).getNode(key)
                    if not seq.isSeq() or not 1 <= seq.size() <= 78:
                        raise ValidationError("Each channel needs 1..78 coefficients as a sequence")
                    lengths.append(seq.size())
            if len(set(lengths)) != 1:
                raise ValidationError("All four coefficient sequences must have the same length")
            matrix, native_size, degree = cv2.loadChromaticAberrationParams(node)
            model = make_model(
                list(native_size), degree, matrix, p["camera_label"], p["lens_label"]
            )
            return Result(
                {
                    "calibration": model,
                    "info": {
                        "filename": path.name,
                        "file_sha256": hashlib.sha256(content).hexdigest(),
                        "calibration_id": model["id"],
                        "image_size": size,
                        "degree": degree,
                    },
                }
            )
        finally:
            storage.release()


class CorrectChromaticAberration(Block):
    type_id, title = "cv.correct_chromatic_aberration", "Correct chromatic aberration"
    category = block_type = "Image processing"
    subtype = "Lens correction"
    inputs = (
        Port("image"),
        Port("calibration", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("valid_mask", visual_type="mask"),
        Port("processed_mask", visual_type="mask"),
        Port("report", "data"),
        Port("valid_fraction", "number"),
    )
    payload_reference = "image"
    process_inputs_readonly = parallel_safe = True
    defaults = {
        "bayer_pattern": "none",
        "calibration_grid": "original",
        "invalid_border": "replicate",
        "maximum_displacement": 128.0,
        "minimum_valid_fraction": 0.5,
        "maximum_work": 250_000_000,
    }
    parameter_choices = {
        "bayer_pattern": ("none", "RGGB", "BGGR", "GRBG", "GBRG"),
        "calibration_grid": ("original", "current"),
        "invalid_border": ("replicate", "zero"),
    }
    parameter_ranges = {
        "maximum_displacement": NumericRange(0, 16384),
        "minimum_valid_fraction": NumericRange(0, 1),
        "maximum_work": NumericRange(1, 1_000_000_000, whole=True),
    }
    parameter_help = {
        "bayer_pattern": "None requires RGB/BGR(A). Otherwise a raw uint8/uint16 mosaic is "
        "bilinearly demosaiced using the explicit current-grid 2x2 tile. Output is BGR. "
        "For other algorithms use Demosaic Bayer image before this block.",
        "calibration_grid": "Original rejects known crop/resize/pyramid grid transforms. "
        "Current accepts a calibration made for the current grid. Size must always match.",
        "invalid_border": "Native linear interpolation replicates edges; zero instead clears "
        "unsupported corrected RGB pixels. Alpha and unselected pixels are retained.",
        "maximum_displacement": "Reject a model exceeding this pixel distance in the grid.",
        "minimum_valid_fraction": "Minimum supported fraction of selected pixels. Failure emits "
        "diagnostic masks/report but withholds the corrected image.",
        "maximum_work": "Bound pixel-count * polynomial-terms * 4. Increase explicitly for "
        "high-degree calibrations; hard image limit is 2.1M pixels.",
    }
    input_help = {
        "image": "Original-size RGB/BGR(A) with uint8/uint16/int16/float32/float64 samples. "
        "Depth/order/alpha are preserved. Green defines the unchanged output coordinate grid.",
        "calibration": "Wire Load chromatic calibration or Define chromatic calibration. "
        "Use the calibrated camera, lens, focus and aperture; labels do not verify hardware.",
        "mask": "Correct selected output pixels only; source donors may lie outside this mask.",
        "region": "Intersected with the mask. Calibration always applies to the full image grid.",
    }

    def process(self, inputs):
        source = inputs["image"]
        array = image_value(source)
        model = validate_model(inputs["calibration"])
        p = self.parameters
        width, height = model["image_size"]
        if array.shape[:2] != (height, width):
            raise ValidationError("Image dimensions must exactly match the calibration grid")
        if width * height * len(model["coefficients"][0]) * 4 > p["maximum_work"]:
            raise ValidationError("Chromatic correction exceeds maximum_work")
        incoming = copy.deepcopy(self.input_payloads["image"])
        spatial = incoming.get("spatial", {})
        if p["calibration_grid"] == "original" and not np.allclose(
            spatial.get("to_root", np.eye(3)), np.eye(3), atol=1e-9, rtol=0
        ):
            raise ValidationError(
                "Use an original calibration grid, or explicitly calibrate the current grid"
            )
        if incoming.get("alpha_representation", "straight") != "straight":
            raise ValidationError("Chromatic correction requires straight alpha")
        bayer = array.ndim == 2
        encoding = image_format(source)
        if bayer:
            if p["bayer_pattern"] == "none" or array.dtype not in (np.uint8, np.uint16):
                raise ValidationError(
                    "Grayscale input needs an explicit uint8/uint16 Bayer pattern"
                )
            raw = incoming.get("raw_sensor")
            if isinstance(raw, dict):
                from .raw_sensor import bayer_pattern_from_sensor

                if bayer_pattern_from_sensor(incoming) != p["bayer_pattern"]:
                    raise ValidationError(
                        "Bayer pattern differs from current sensor phase metadata"
                    )
            code = getattr(cv2, f"COLOR_Bayer{p['bayer_pattern']}2BGR")
            base, native, encoding = cv2.demosaicing(array, code), array, "bgr"
            if isinstance(raw, dict):
                incoming["raw_sensor"]["mosaic"] = False
                incoming["raw_sensor"]["demosaicing"] = {
                    "pattern": p["bayer_pattern"],
                    "algorithm": "bilinear",
                    "color_format": "bgr",
                }
        else:
            if encoding not in ("rgb", "bgr", "rgba", "bgra"):
                raise ValidationError("Convert HSV/Lab to RGB/BGR before chromatic correction")
            base = array[..., :3]
            native = base[..., ::-1] if encoding.startswith("rgb") else base
            code = -1
        scope = selection_mask(inputs, source)
        selected = np.ones((height, width), bool) if scope is None else scope != 0
        if not selected.any():
            raise ValidationError("Chromatic correction selection is empty")
        valid, peak = support(model, bayer, float(p["maximum_displacement"]))
        valid[~selected] = 0
        fraction = float(np.count_nonzero(valid) / np.count_nonzero(selected))
        passed = fraction >= p["minimum_valid_fraction"]
        output = NO_VALUE
        if passed:
            corrected = cv2.correctChromaticAberration(
                native,
                np.asarray(model["coefficients"], np.float32),
                (width, height),
                model["degree"],
                bayer_pattern=code,
            )
            if not np.isfinite(corrected).all():
                raise ValidationError("Native chromatic correction produced nonfinite samples")
            if encoding.startswith("rgb"):
                corrected = corrected[..., ::-1]
            corrected[~selected] = base[~selected]
            if p["invalid_border"] == "zero":
                corrected[selected & (valid == 0)] = 0
            if array.ndim == 3 and array.shape[-1] == 4:
                corrected = np.concatenate((corrected, array[..., 3:]), axis=2)
            output = ImageArray(corrected, encoding)
        report = {
            "model": "chromatic_correction_report_v1",
            "calibration_id": model["id"],
            "camera_label": model["camera_label"],
            "lens_label": model["lens_label"],
            "image_size": [width, height],
            "degree": model["degree"],
            "selected_pixels": int(selected.sum()),
            "valid_pixels": int(np.count_nonzero(valid)),
            "valid_fraction": fraction,
            "max_displacement_blue": peak[0],
            "max_displacement_red": peak[1],
            "coordinate_reference": "green_input_grid",
            "interpolation": "linear",
            "native_border": "replicate",
            "invalid_border": p["invalid_border"],
            "bayer_pattern": p["bayer_pattern"] if bayer else None,
            "calibration_fit": "supplied_not_estimated",
            "passed": passed,
        }
        incoming["chromatic_correction"] = {
            "calibration_id": model["id"],
            "reference": "green_input_grid",
        }
        outputs = {
            "image": output,
            "valid_mask": valid,
            "processed_mask": selected.astype(np.uint8) * 255,
            "report": report,
            "valid_fraction": fraction,
        }
        payloads = {key: copy.deepcopy(incoming) for key in outputs}
        for key in ("valid_mask", "processed_mask"):
            payloads[key].pop("alpha_representation", None)
            payloads[key].pop("raw_sensor", None)
        drawings = {}
        if self.generate_graphics:
            drawings["chromatic support"] = GraphicDrawing(
                (
                    graphic_object(
                        "text",
                        [(5, min(18, height - 1))],
                        (60, 220, 120) if passed else (60, 160, 240),
                        text=f"CA support {fraction:.1%}",
                    ),
                ),
                "image",
                passed,
            )
        return Result(
            outputs,
            payloads=payloads,
            passed=passed,
            score=fraction,
            drawings=drawings,
            message=f"Chromatic correction: {fraction:.1%} of selection supported",
        )
