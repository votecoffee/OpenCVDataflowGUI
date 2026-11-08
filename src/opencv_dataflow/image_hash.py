"""Versioned OpenCV binary image hashes and compatible comparisons."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value

METHODS = ("average", "perceptual", "block_mean", "marr_hildreth")
MARR_RANGES = {"marr_alpha": NumericRange(0.5, 4), "marr_scale": NumericRange(0, 2)}
HASH_MODEL = "opencv_binary_image_hash_v1"


def hash_options(method, settings):
    if method == "block_mean":
        mode = str(settings.get("block_mode"))
        if mode not in ("0", "1"):
            raise ValidationError("Block mean mode must be 0 or 1", "block_mode")
        return {"block_mode": mode}
    if method == "marr_hildreth":
        return {
            name: limit.check(settings.get(name), name.replace("_", " "), name)
            for name, limit in MARR_RANGES.items()
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


class ComputeImageHash(Block):
    type_id, title = "cv.image_hash", "Compute image hash"
    category, block_type, subtype = "Image processing", "Image processing", "Image hashes"
    inputs = (Port("image"),)
    outputs = (Port("hash", "data"),)
    defaults = {"method": "perceptual", "block_mode": "0", "marr_alpha": 2.0, "marr_scale": 1.0}
    parameter_choices = {"method": METHODS, "block_mode": ("0", "1")}
    parameter_ranges = MARR_RANGES
    parameter_help = {
        "method": "OpenCV binary hash. Compare only hashes made with the same method and settings.",
        "block_mode": "0 uses disjoint blocks. 1 uses overlapping blocks.",
        "marr_alpha": "Marr–Hildreth wavelet scale factor.",
        "marr_scale": "Marr–Hildreth wavelet scale level.",
    }
    input_help = {"image": "8-bit grayscale or color. Crop upstream to hash a selected window."}
    process_inputs_readonly = True
    parallel_safe = True

    def process(self, inputs):
        image = image_value(inputs["image"], (np.uint8,), canonical=True)
        method = self.parameters["method"]
        options = hash_options(method, self.parameters)
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
