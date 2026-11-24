"""Compute native patch descriptors at externally supplied keypoints."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .processing import ImageTool, gray_image
from .result_graphics import ACCEPTED
from .selection import contains, selection_mask
from .tickets import ticket_key


def _keypoint_rows(value, size):
    if isinstance(value, dict):
        if value.get("image_size") != size:
            raise ValidationError("Keypoint image_size must match the descriptor image")
        value = value.get("keypoints")
    if not isinstance(value, list) or len(value) > 10_000:
        raise ValidationError("Supply a list of up to 10,000 keypoints or a feature record")
    return value


def _native_keypoint(row, index, size):
    record = row if isinstance(row, dict) else {"point": row}
    try:
        xy = np.asarray(record["point"], dtype=float)
        diameter = float(record.get("size", 7))
        angle = float(record.get("angle", -1))
        response = float(record.get("response", 0))
        octave = record.get("octave", 0)
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or (xy < 0).any()
            or (xy >= size).any()
            or not 0.001 <= diameter <= 1_000_000
            or not (angle == -1 or 0 <= angle <= 360)
            or not -3e38 <= response <= 3e38
            or type(octave) is not int
            or not -(2**31) <= octave < 2**31
        ):
            raise ValueError
    except KeyError, TypeError, ValueError, OverflowError:
        raise ValidationError(
            f"Keypoint {index} needs an in-image [x, y] point and valid size, angle, "
            "response and octave"
        ) from None
    return cv2.KeyPoint(
        *xy, diameter, angle % 360 if angle >= 0 else angle, response, octave, index
    )


class KeypointDescriptor(ImageTool, ABC):
    category = block_type = "Inspection"
    subtype = "Features"
    inputs = (
        Port("image"),
        Port("keypoints", "data"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("features", "data"),
        Port("points", "data"),
        Port("count", "number"),
    )
    input_help = {
        "image": "Source intensity image. Descriptor computation uses 8-bit grayscale.",
        "keypoints": "Feature record, keypoint records, or [x, y] rows. Bare points use size 7 "
        "and unknown orientation.",
        "mask": "Describe keypoints whose centers are inside the mask.",
        "region": "Describe keypoints whose centers are inside the region and any wired mask.",
    }
    output_help = {
        "image": "Original image for overlaying drawable keypoint markers.",
        "features": "Aligned keypoints and descriptor rows, with original row indices.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    @abstractmethod
    def extractor(self):
        """Return a native extractor and the normalized settings identifying its pattern."""

    def sample_keypoints(self, inputs, size, selection):
        image_payload = self.input_payloads["image"]
        point_payload = self.input_payloads["keypoints"]
        try:
            require_same_frame([ValuePacket(None, image_payload), ValuePacket(None, point_payload)])
        except ValueError as exc:
            raise ValidationError("Keypoints and image must use the same coordinate frame") from exc
        if (
            "ticket" in image_payload
            and "ticket" in point_payload
            and ticket_key(image_payload["ticket"]) != ticket_key(point_payload["ticket"])
        ):
            raise ValidationError("Keypoints and image have different acquisition tickets")
        keypoints = []
        for index, row in enumerate(_keypoint_rows(inputs["keypoints"], size)):
            key = _native_keypoint(row, index, size)
            if contains(selection, key.pt):
                keypoints.append(key)
        if self.parameters.get("use_orientation") and any(key.angle < 0 for key in keypoints):
            raise ValidationError(
                f"Oriented {self.algorithm} needs a measured angle for each selected keypoint"
            )
        return keypoints

    def compute_descriptors(self, gray, keypoints):
        extractor, settings = self.extractor()
        keys, descriptors = extractor.compute(gray, keypoints) if keypoints else ((), None)
        return keys, descriptors, extractor.descriptorSize(), settings

    def process(self, inputs):
        source = inputs["image"]
        height, width = source.shape[:2]
        size = [width, height]
        if not 0 < width * height <= self.maximum_pixels:
            raise ValidationError(
                f"{self.algorithm} requires 1 to {self.maximum_pixels:,} image pixels"
            )
        selection = selection_mask(inputs, source)
        keypoints = self.sample_keypoints(inputs, size, selection)
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        keys, descriptors, descriptor_size, settings = self.compute_descriptors(gray, keypoints)
        points = [list(key.pt) for key in keys]
        features = {
            "algorithm": self.algorithm,
            "descriptor_size": descriptor_size,
            "descriptor_settings": settings,
            "image_size": size,
            "keypoints": [
                {
                    "point": point,
                    "size": key.size,
                    "response": key.response,
                    "octave": key.octave,
                    "angle": key.angle % 360 if self.algorithm == "FREAK" else key.angle,
                    "source_index": key.class_id,
                }
                for key, point in zip(keys, points, strict=True)
            ],
            "descriptors": descriptors.tolist() if descriptors is not None else [],
        }
        result = Result(
            {"image": source, "features": features, "points": points, "count": len(points)},
            message=f"{len(points)} {self.algorithm} descriptors from "
            f"{len(keypoints)} selected keypoints",
        )
        if self.generate_graphics and points:
            result.drawings[self.algorithm] = GraphicDrawing(
                tuple(graphic_object("marker", [point], ACCEPTED, thickness=1) for point in points)
            )
        return result

    def factory(self, name):
        factory = getattr(getattr(cv2, "xfeatures2d", None), name, None)
        if factory is None:
            raise ValidationError(f"{self.algorithm} requires OpenCV contrib xfeatures2d")
        return factory


class BriefDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.brief_descriptor", "BRIEF descriptors", "BRIEF"
    description = "Compute fixed-length binary intensity descriptors at wired keypoints."
    maximum_pixels = 8_388_608
    defaults = {"descriptor_bytes": "32", "use_orientation": False}
    parameter_choices = {"descriptor_bytes": ("16", "32", "64")}
    parameter_help = {
        "descriptor_bytes": "Number of bytes per Hamming descriptor.",
        "use_orientation": "Rotate the sampling pattern using the wired keypoint angles.",
    }

    def extractor(self):
        settings = {"use_orientation": self.parameters["use_orientation"]}
        factory = self.factory("BriefDescriptorExtractor_create")
        return factory(
            int(self.parameters["descriptor_bytes"]), settings["use_orientation"]
        ), settings


class FreakDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.freak_descriptor", "FREAK descriptors", "FREAK"
    description = "Compute 64-byte retinal sampling descriptors at wired keypoints."
    maximum_pixels = 16_777_216
    defaults = {
        "orientation_normalized": True,
        "scale_normalized": True,
        "pattern_scale": 22.0,
        "octaves": 4,
        "selected_pairs": [],
    }
    parameter_ranges = {
        "pattern_scale": NumericRange(1, 128),
        "octaves": NumericRange(1, 8, whole=True),
    }
    parameter_help = {
        "orientation_normalized": "Estimate a local angle and rotate the sampling pattern.",
        "scale_normalized": "Scale the pattern using the wired keypoint diameter.",
        "pattern_scale": "Base size of the retinal sampling pattern.",
        "octaves": "Number of octaves covered by the descriptor scales.",
        "selected_pairs": "Empty uses OpenCV's defaults. Otherwise supply 512 ordered pair indices "
        "from 0 to 902. Query and train must use the same list.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        pairs = p["selected_pairs"]
        if not isinstance(pairs, list) or (
            pairs
            and (len(pairs) != 512 or any(type(i) is not int or not 0 <= i <= 902 for i in pairs))
        ):
            raise ValidationError(
                "Selected pairs must be empty or 512 integer indices in 0..902", "selected_pairs"
            )

    def extractor(self):
        p = self.parameters
        settings = {
            "orientation_normalized": p["orientation_normalized"],
            "scale_normalized": p["scale_normalized"],
            "pattern_scale": float(np.float32(p["pattern_scale"])),
            "octaves": int(p["octaves"]),
            "selected_pairs": p["selected_pairs"],
        }
        factory = self.factory("FREAK_create")
        return factory(
            orientationNormalized=settings["orientation_normalized"],
            scaleNormalized=settings["scale_normalized"],
            patternScale=settings["pattern_scale"],
            nOctaves=settings["octaves"],
            selectedPairs=settings["selected_pairs"],
        ), settings


class LatchDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.latch_descriptor", "LATCH descriptors", "LATCH"
    description = "Compare learned triplets of intensity patches at wired keypoints."
    maximum_pixels = 8_388_608
    defaults = {
        "descriptor_bytes": "32",
        "use_orientation": True,
        "half_ssd_size": 3,
        "sigma": 2.0,
    }
    parameter_choices = {"descriptor_bytes": ("1", "2", "4", "8", "16", "32", "64")}
    parameter_ranges = {
        "half_ssd_size": NumericRange(0, 32, whole=True),
        "sigma": NumericRange(0, 32),
    }
    parameter_help = {
        "descriptor_bytes": "Number of bytes in the native binary descriptor.",
        "use_orientation": "Rotate the patch triplets by the measured keypoint angle.",
        "half_ssd_size": "Radius of each square patch used for squared intensity differences.",
        "sigma": "Gaussian smoothing sigma. Zero skips smoothing.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["use_orientation"]) is not bool:
            raise ValidationError("Use orientation must be on or off", "use_orientation")

    def extractor(self):
        p = self.parameters
        settings = {
            "use_orientation": p["use_orientation"],
            "half_ssd_size": int(p["half_ssd_size"]),
            "sigma": float(p["sigma"]),
        }
        return self.factory("LATCH_create")(
            int(p["descriptor_bytes"]),
            settings["use_orientation"],
            settings["half_ssd_size"],
            settings["sigma"],
        ), settings


class _LearnedPatchDescriptor(KeypointDescriptor):
    # The native integral-image implementation uses signed 32-bit sums and differences.
    maximum_pixels = 4_194_304
    parameter_choices = {"descriptor_bits": ("256", "512")}
    parameter_ranges = {"scale_factor": NumericRange(0.01, 32)}
    parameter_help = {
        "descriptor_bits": "Select the native 256-bit or 512-bit pattern.",
        "scale_factor": "Scale the patch relative to the keypoint's size. Use 1 for ORB, "
        "6.75 for SIFT, 6.25 for KAZE, or 5 for FAST and BRISK.",
    }

    def extractor(self):
        p = self.parameters
        scale = float(np.float32(p["scale_factor"]))
        factory = self.factory(f"{self.algorithm}_create")
        bits = getattr(cv2.xfeatures2d, f"{self.algorithm}_SIZE_{p['descriptor_bits']}_BITS")
        return factory(scale, bits), {"scale_factor": scale}


class BeblidDescriptor(_LearnedPatchDescriptor):
    type_id, title, algorithm = "inspect.beblid_descriptor", "BEBLID descriptors", "BEBLID"
    description = "Compute boosted binary descriptors from local intensity averages."
    defaults = {"descriptor_bits": "512", "scale_factor": 1.0}


class TeblidDescriptor(_LearnedPatchDescriptor):
    type_id, title, algorithm = "inspect.teblid_descriptor", "TEBLID descriptors", "TEBLID"
    description = "Compute triplet-trained binary descriptors from local intensity averages."
    defaults = {"descriptor_bits": "256", "scale_factor": 1.0}


# OpenCV's Python BoostDesc factory takes the public C++ enum values, but the
# current binding does not export their names.
BOOST_DESC_VARIANTS = {
    "BGM": (100, 32, cv2.NORM_HAMMING),
    "BGM_HARD": (101, 16, cv2.NORM_HAMMING),
    "BGM_BILINEAR": (102, 32, cv2.NORM_HAMMING),
    "LBGM": (200, 64, cv2.NORM_L2),
    "BINBOOST_64": (300, 8, cv2.NORM_HAMMING),
    "BINBOOST_128": (301, 16, cv2.NORM_HAMMING),
    "BINBOOST_256": (302, 32, cv2.NORM_HAMMING),
}
BOOST_DESC_MIN_SCALE = float(np.float32(0.01))
BOOST_DESC_MAX_SCALE = 32.0


class BoostDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.boost_descriptor", "BoostDesc descriptors", "BoostDesc"
    description = "Compute learned gradient descriptors for Hamming or L2 matching."
    maximum_pixels = 16_777_216
    defaults = {"variant": "BINBOOST_256", "use_orientation": True, "scale_factor": 6.25}
    parameter_choices = {"variant": tuple(BOOST_DESC_VARIANTS)}
    parameter_ranges = {"scale_factor": NumericRange(0.01, BOOST_DESC_MAX_SCALE)}
    parameter_help = {
        "variant": "BGM and BINBOOST use Hamming distance. LBGM uses L2.",
        "use_orientation": "Rotate the sampling window by each measured keypoint angle.",
        "scale_factor": "Sampling diameter multiplier. Use 0.75 for ORB, 6.75 for SIFT, "
        "6.25 for KAZE, or 5 for AKAZE and BRISK.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["use_orientation"]) is not bool:
            raise ValidationError("Use orientation must be on or off", "use_orientation")

    def extractor(self):
        p = self.parameters
        scale = float(np.float32(p["scale_factor"]))
        variant = p["variant"]
        number, width, norm = BOOST_DESC_VARIANTS[variant]
        api = getattr(cv2, "xfeatures2d", None)
        factory = getattr(getattr(api, "BoostDesc", None), "create", None)
        if factory is None:
            factory = self.factory("BoostDesc_create")
        try:
            extractor = factory(number, p["use_orientation"], scale)
        except cv2.error as exc:
            if exc.code == cv2.Error.StsNotImplemented:
                raise ValidationError(
                    "BoostDesc needs OpenCV contrib with BoostDesc model data"
                ) from exc
            raise
        if extractor.descriptorSize() != width or extractor.defaultNorm() != norm:
            raise ValidationError("OpenCV returned an unexpected BoostDesc layout")
        return extractor, {
            "variant": variant, "scale_factor": scale, "use_orientation": p["use_orientation"],
        }

    def compute_descriptors(self, gray, keypoints):
        if max(gray.shape) >= 32767:
            raise ValidationError("BoostDesc requires image axes below 32,767 pixels")
        scale = self.parameters["scale_factor"]
        if any(key.size * scale > 16384 for key in keypoints):
            raise ValidationError(
                "BoostDesc sampling diameter must be at most 16,384 pixels", "scale_factor"
            )
        return super().compute_descriptors(gray, keypoints)
