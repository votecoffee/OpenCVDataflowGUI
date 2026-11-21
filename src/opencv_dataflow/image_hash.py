"""Versioned OpenCV image hashes and compatible comparisons."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import image_value

METHODS = ("average", "perceptual", "block_mean", "marr_hildreth")
DESCRIPTOR_METHODS = ("color_moment", "radial_variance")
DESCRIPTOR_MODEL = "opencv_perceptual_descriptor_v1"
MARR_RANGES = {"marr_alpha": NumericRange(0.5, 4), "marr_scale": NumericRange(0, 2)}
HASH_MODEL = "opencv_binary_image_hash_v1"
RADIAL_RANGES = {
    "radial_sigma": NumericRange(1, 32),
    "radial_angle_lines": NumericRange(40, 1440, whole=True),
}
DESCRIPTOR_LAYOUTS = {"color_moment": (42, np.float64), "radial_variance": (40, np.uint8)}
_MAX_RADIAL_PROJECTION_SAMPLES = 8_388_608


def hash_options(method, settings):
    if method == "block_mean":
        mode = str(settings.get("block_mode"))
        if mode not in ("0", "1"):
            raise ValidationError("Block mean mode must be 0 or 1", "block_mode")
        return {"block_mode": mode}
    limits = {"marr_hildreth": MARR_RANGES, "radial_variance": RADIAL_RANGES}.get(method, {})
    if limits:
        return {
            name: limit.check(settings.get(name), name.replace("_", " "), name)
            for name, limit in limits.items()
        }
    return {}


def hash_bits(method, options):
    if method == "block_mean":
        return 961 if options["block_mode"] == "1" else 256
    return 576 if method == "marr_hildreth" else 64


def read_hash(record, field):
    if not isinstance(record, dict) or record.get("model") != HASH_MODEL:
        raise ValidationError("Use a binary image hash from Compute image hash", field)
    method, settings, encoded = (record.get(k) for k in ("method", "options", "hex"))
    if method not in METHODS or not isinstance(settings, dict):
        raise ValidationError("Invalid image hash method or options", field)
    options = hash_options(method, settings)
    if options != settings:
        raise ValidationError("Invalid options for the image hash method", field)
    bits = hash_bits(method, options)
    size = (bits + 7) // 8
    if not isinstance(encoded, str) or len(encoded) != 2 * size:
        raise ValidationError("Invalid image hash length", field)
    try:
        packed = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValidationError("Image hash must contain hexadecimal bytes", field) from exc
    if len(packed) != size or (bits % 8 and packed[-1] >= 1 << (bits % 8)):
        raise ValidationError("Invalid image hash bytes or padding", field)
    return method, options, np.frombuffer(packed, np.uint8), bits


def read_descriptor(record, field):
    if not isinstance(record, dict) or record.get("model") != DESCRIPTOR_MODEL:
        raise ValidationError("Use a Color Moment or Radial Variance image hash", field)
    method, settings, values = (record.get(k) for k in ("method", "options", "values"))
    if method not in DESCRIPTOR_METHODS or not isinstance(settings, dict):
        raise ValidationError("Invalid perceptual hash method or options", field)
    options = hash_options(method, settings)
    count, dtype = DESCRIPTOR_LAYOUTS[method]
    if options != settings or not isinstance(values, list) or len(values) != count:
        raise ValidationError("Invalid perceptual hash options or coefficient count", field)
    uniform = False
    if method == "radial_variance":
        uniform = record.get("uniform_input")
        if type(uniform) is not bool or any(
            type(v) is not int or not 0 <= v <= 255 for v in values
        ):
            raise ValidationError(
                "Radial hash requires byte coefficients and a uniform-input flag", field
            )
    elif any(type(v) not in (int, float) for v in values):
        raise ValidationError("Color moment hash requires floating-point coefficients", field)
    try:
        coefficients = np.asarray(values, dtype=dtype).reshape(1, count)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(
            "Perceptual hash coefficients are outside the native range", field
        ) from exc
    if not np.isfinite(coefficients).all():
        raise ValidationError("Perceptual hash coefficients must be finite", field)
    return method, options, coefficients, uniform


class ComputeImageHash(Block):
    type_id, title = "cv.image_hash", "Compute image hash"
    category, block_type, subtype = "Image processing", "Image processing", "Image hashes"
    inputs = (Port("image"),)
    outputs = (Port("hash", "data"),)
    defaults = {
        "method": "perceptual",
        "block_mode": "0",
        "marr_alpha": 2.0,
        "marr_scale": 1.0,
        "radial_sigma": 1.0,
        "radial_angle_lines": 180,
    }
    parameter_choices = {"method": (*METHODS, *DESCRIPTOR_METHODS), "block_mode": ("0", "1")}
    parameter_ranges = {**MARR_RANGES, **RADIAL_RANGES}
    parameter_help = {
        "method": "For Color Moment and Radial Variance use Compare perceptual descriptors.",
        "block_mode": "0 uses disjoint blocks. 1 uses overlapping blocks.",
        "marr_alpha": "Marr–Hildreth wavelet scale factor.",
        "marr_scale": "Marr–Hildreth wavelet scale level.",
        "radial_sigma": "Gaussian blur standard deviation in pixels for Radial Variance.",
        "radial_angle_lines": "Number of projection angles for Radial Variance.",
    }
    input_help = {"image": "8-bit grayscale or color. Crop upstream to hash a selected window."}
    output_help = {"hash": "Versioned record for the selected method and its active settings."}
    preview_safe = True

    def _descriptor(self, image, method, options):
        record = {"model": DESCRIPTOR_MODEL, "method": method, "options": options}
        if method == "color_moment":
            algorithm = cv2.img_hash.ColorMomentHash_create()
        else:
            lines = options["radial_angle_lines"]
            if lines * max(image.shape[:2]) > _MAX_RADIAL_PROJECTION_SAMPLES:
                raise ValidationError(
                    "Reduce projection angle count or resize the image before hashing"
                )
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            low, high, _, _ = cv2.minMaxLoc(image)
            record["uniform_input"] = low == high
            algorithm = cv2.img_hash.RadialVarianceHash_create(options["radial_sigma"], lines)
        digest = algorithm.compute(image)
        count, dtype = DESCRIPTOR_LAYOUTS[method]
        if digest.shape != (1, count) or digest.dtype != dtype or not np.isfinite(digest).all():
            raise RuntimeError("OpenCV returned an invalid perceptual hash descriptor")
        record["values"] = digest.ravel().tolist()
        return Result(
            {"hash": record},
            message=f"{method.replace('_', ' ').title()} · {count} coefficients",
        )

    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8,), canonical=True)
        method = self.parameters["method"]
        options = hash_options(method, self.parameters)
        if method in DESCRIPTOR_METHODS:
            return self._descriptor(image, method, options)
        if method == "block_mean":
            algorithm = cv2.img_hash.BlockMeanHash_create(int(options["block_mode"]))
        elif method == "marr_hildreth":
            algorithm = cv2.img_hash.MarrHildrethHash_create(
                options["marr_alpha"], options["marr_scale"]
            )
        elif method == "average":
            algorithm = cv2.img_hash.AverageHash_create()
        else:
            algorithm = cv2.img_hash.PHash_create()
        packed = algorithm.compute(image)
        bits = hash_bits(method, options)
        return Result(
            {
                "hash": {
                    "model": HASH_MODEL,
                    "method": method,
                    "options": options,
                    "hex": packed.tobytes().hex(),
                }
            },
            message=f"{method.replace('_', ' ').title()} · {bits} bits",
        )


class CompareImageHashes(Block):
    type_id, title = "inspect.compare_image_hashes", "Compare image hashes"
    description = "Count differing bits between binary hashes made with matching settings."
    category, block_type, subtype = "Inspection", "Inspection", "Image hashes"
    inputs = (Port("sample", "data"), Port("reference", "data"))
    outputs = (Port("distance", "integer"), Port("bit_fraction", "float"))
    defaults = {"check_threshold": False, "maximum_distance": 8}
    parameter_ranges = {"maximum_distance": NumericRange(0, 961, whole=True)}
    parameter_help = {
        "maximum_distance": "Maximum number of differing bits when threshold checking is enabled.",
    }
    payload_reference = "sample"
    parallel_safe = True

    def process(self, inputs):
        method, options, sample, bits = read_hash(inputs["sample"], "sample")
        ref_method, ref_options, reference, _ = read_hash(inputs["reference"], "reference")
        if (method, options) != (ref_method, ref_options):
            raise ValidationError("Image hashes must use the same method and settings")
        distance = int(cv2.norm(sample, reference, cv2.NORM_HAMMING))
        fraction = distance / bits
        p = self.parameters
        passed = not p["check_threshold"] or distance <= int(p["maximum_distance"])
        return Result(
            {"distance": distance, "bit_fraction": fraction},
            passed=passed,
            score=1 - fraction,
            message=f"{distance} / {bits} bits differ",
            fail_reasons=[] if passed else ["image_hash_distance_above_limit"],
        )


class ComparePerceptualDescriptors(Block):
    type_id, title = "inspect.compare_perceptual_descriptors", "Compare perceptual descriptors"
    category, block_type, subtype = "Inspection", "Inspection", "Image hashes"
    description = "Measure Color Moment distance or Radial Variance similarity."
    inputs = (Port("sample", "data"), Port("reference", "data"))
    outputs = (
        Port("distance", "float", required=False),
        Port("similarity", "float", required=False),
        Port("measurements", "data"),
    )
    defaults = {
        "check_threshold": False,
        "maximum_color_distance": 1.0,
        "minimum_radial_similarity": 0.8,
    }
    parameter_ranges = {
        "maximum_color_distance": NumericRange(0),
        "minimum_radial_similarity": NumericRange(0, 1),
    }
    parameter_help = {
        "maximum_color_distance": "Pass at or below this native Color Moment distance.",
        "minimum_radial_similarity": "Pass at or above this Radial Variance correlation.",
    }
    output_help = {
        "distance": "Color Moment distance, with smaller values indicating a closer match.",
        "similarity": "Radial Variance correlation in [0, 1], with higher values closer.",
        "measurements": "Native score, comparison direction, method, settings and threshold.",
    }
    payload_reference = "sample"
    preview_safe = parallel_safe = True

    def process(self, inputs):
        method, options, sample, uniform = read_descriptor(inputs["sample"], "sample")
        ref_method, ref_options, reference, ref_uniform = read_descriptor(
            inputs["reference"], "reference"
        )
        if (method, options) != (ref_method, ref_options):
            raise ValidationError("Image hashes must use the same method and settings")
        p = self.parameters
        output = {"distance": NO_VALUE, "similarity": NO_VALUE}
        metrics = {"method": method, "options": options, "status": "ok"}
        output["measurements"] = metrics
        if method == "color_moment":
            native = cv2.img_hash.ColorMomentHash_create().compare(sample, reference)
            if not math.isfinite(native):
                raise ValidationError("The Color Moment distance exceeds the native range")
            limit = float(p["maximum_color_distance"])
            value, terminal = native, "distance"
            label, direction = "Color Moment distance", "lower"
            in_range = value <= limit
        else:
            if uniform or ref_uniform or not np.ptp(sample) or not np.ptp(reference):
                metrics["status"] = "insufficient_radial_variation"
                reason = "Radial comparison needs two images and descriptors with variation"
                return Result(
                    output,
                    passed=False,
                    message=reason,
                    fail_reasons=[{"code": "radial_hash_degenerate", "message": reason}],
                )
            limit = float(p["minimum_radial_similarity"])
            algorithm = cv2.img_hash.RadialVarianceHash_create(
                options["radial_sigma"], options["radial_angle_lines"]
            )
            native = algorithm.compare(sample, reference)
            if not math.isfinite(native):
                raise ValidationError("OpenCV returned a nonfinite Radial Variance correlation")
            # Native FP accumulation can return slightly more or less than one.
            value = 1.0 if np.array_equal(sample, reference) else min(1.0, max(0.0, native))
            terminal = "similarity"
            label, direction = "Radial Variance similarity", "higher"
            in_range = value >= limit

        output[terminal] = value
        passed = not p["check_threshold"] or in_range
        metrics.update(
            value=value,
            direction=direction,
            threshold=limit if p["check_threshold"] else None,
            native_value=native,
        )
        message = f"{label} {value:.6g}"
        return Result(
            output,
            passed=passed,
            score=value if terminal == "similarity" else None,
            message=message,
            fail_reasons=[]
            if passed
            else [
                {
                    "code": f"perceptual_hash_{terminal}_outside_limit",
                    "message": f"{message} is outside the configured limit {limit:.6g}",
                }
            ],
        )
