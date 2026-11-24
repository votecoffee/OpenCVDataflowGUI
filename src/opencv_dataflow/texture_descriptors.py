"""Native VGG intensity and LUCID color texture descriptors."""

import cv2
import numpy as np

from .compact_descriptors import KeypointDescriptor, check_scaled_patch_support
from .images import convert_depth, image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value

# Values from the public VGG enum. Python builds can omit the symbolic names.
VGG_MODES = {
    "VGG_120": (100, 120),
    "VGG_80": (101, 80),
    "VGG_64": (102, 64),
    "VGG_48": (103, 48),
}
VGG_BOOLEAN_SETTINGS = ("normalize_image", "use_orientation", "normalize_descriptor")
TEXTURE_MIN_SCALE = float(np.float32(0.01))
VGG_MAX_SCALE = 32.0
VGG_MAX_SIGMA = 16.0
LUCID_MAX_RADIUS = 6
LUCID_MAX_BLUR = 32


class VggDescriptor(KeypointDescriptor):
    type_id, title, algorithm = "inspect.vgg_descriptor", "VGG texture descriptors", "VGG"
    description = "Compute learned local intensity descriptors for L2 matching."
    maximum_pixels = 16_777_216
    defaults = {
        "variant": "VGG_120", "sigma": 1.4, "normalize_image": True,
        "use_orientation": True, "scale_factor": 6.25, "normalize_descriptor": False,
    }
    parameter_choices = {"variant": tuple(VGG_MODES)}
    parameter_ranges = {
        "sigma": NumericRange(0.01, VGG_MAX_SIGMA),
        "scale_factor": NumericRange(0.01, VGG_MAX_SCALE),
    }
    parameter_help = {
        "variant": "Number of values in the learned descriptor.",
        "sigma": "Standard deviation of the native Gaussian prefilter.",
        "normalize_image": "Normalize the intensities inside each sampled patch.",
        "use_orientation": "Rotate by measured keypoint angles.",
        "scale_factor": "Multiply the keypoint diameter to set the patch diameter. "
        "Use 6.75 for SIFT or 0.75 for ORB.",
        "normalize_descriptor": "Scale the entire descriptor table to 0–255 and emit bytes. "
        "Distance remains L2.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in VGG_BOOLEAN_SETTINGS:
            if type(p[key]) is not bool:
                raise ValidationError("Expected an on or off setting", key)

    def extractor(self):
        p = self.parameters
        settings = {
            "variant": p["variant"],
            "sigma": float(np.float32(p["sigma"])),
            "scale_factor": float(np.float32(p["scale_factor"])),
            **{name: p[name] for name in VGG_BOOLEAN_SETTINGS},
        }
        code, width = VGG_MODES[settings["variant"]]
        factory = self.factory("VGG_create")
        try:
            extractor = factory(
                code, settings["sigma"], settings["normalize_image"], settings["use_orientation"],
                settings["scale_factor"], settings["normalize_descriptor"],
            )
        except cv2.error as exc:
            if exc.code == cv2.Error.StsNotImplemented:
                raise ValidationError(
                    "VGG needs OpenCV contrib built with the VGG model data"
                ) from exc
            raise
        if extractor.descriptorSize() != width or extractor.defaultNorm() != cv2.NORM_L2:
            raise ValidationError("OpenCV returned an unexpected VGG descriptor layout")
        return extractor, settings

    def compute_descriptors(self, gray, keypoints):
        check_scaled_patch_support(gray, keypoints, self.parameters["scale_factor"], self.algorithm)
        return super().compute_descriptors(gray, keypoints)


class LucidDescriptor(KeypointDescriptor):
    type_id = "inspect.lucid_descriptor"
    title, algorithm = "LUCID color texture descriptors", "LUCID"
    description = "Compute sorted local color intensities for Hamming matching."
    maximum_pixels = 16_777_216
    defaults = {"patch_radius": 1, "blur_radius": 2}
    parameter_ranges = {
        "patch_radius": NumericRange(0, LUCID_MAX_RADIUS, whole=True),
        "blur_radius": NumericRange(0, LUCID_MAX_BLUR, whole=True),
    }
    parameter_help = {
        "patch_radius": "Half-width in pixels of the color patch. Radius 6 produces 507 bytes.",
        "blur_radius": "Half-width of the box blur. Zero samples the original color.",
    }
    input_help = {
        **KeypointDescriptor.input_help,
        "image": "RGB or BGR image. LUCID uses 8-bit color samples and ignores alpha.",
        "keypoints": "Feature record or x/y centers. LUCID uses position only.",
    }

    def descriptor_image(self, source):
        if image_format(source) not in ("rgb", "bgr", "rgba", "bgra"):
            raise ValidationError("LUCID needs RGB or BGR color")
        raw = image_value(source, (np.uint8, np.uint16, np.float32, np.float64), canonical=True)
        if raw.ndim != 3:
            raise ValidationError("LUCID needs three color channels")
        color = raw[..., :3]
        if min(color.shape[:2]) <= self.parameters["patch_radius"]:
            raise ValidationError(
                "LUCID patch radius must be smaller than both image axes", "patch_radius"
            )
        return color if color.dtype == np.uint8 else convert_depth(color, "uint8")

    def extractor(self):
        p = self.parameters
        patch, blur = int(p["patch_radius"]), int(p["blur_radius"])
        extractor = self.factory("LUCID_create")(patch, blur)
        return extractor, {"patch_radius": patch, "blur_radius": blur}


def texture_descriptor_layout(data):
    """Validate the wire metadata and state the native distance and numeric range."""
    family = data["algorithm"]
    width, settings = data.get("descriptor_size"), data.get("descriptor_settings")
    if type(width) is not int or not isinstance(settings, dict):
        raise ValidationError(f"Invalid {family} descriptor width or settings")
    if family == "VGG":
        mode, scale, sigma = (settings.get(name) for name in ("variant", "scale_factor", "sigma"))
        spec = VGG_MODES.get(mode) if isinstance(mode, str) else None
        if (
            spec is None or width != spec[1]
            or any(type(settings.get(name)) is not bool for name in VGG_BOOLEAN_SETTINGS)
            or type(scale) not in (float, int) or not TEXTURE_MIN_SCALE <= scale <= VGG_MAX_SCALE
            or type(sigma) not in (float, int) or not TEXTURE_MIN_SCALE <= sigma <= VGG_MAX_SIGMA
        ):
            raise ValidationError("Invalid VGG variant, width or sampling settings")
        if settings["normalize_descriptor"]:
            return 0, 255, True, np.float32, cv2.NORM_L2
        return -1_000_000, 1_000_000, False, np.float32, cv2.NORM_L2
    patch, blur = settings.get("patch_radius"), settings.get("blur_radius")
    if (
        type(patch) is not int or not 0 <= patch <= LUCID_MAX_RADIUS
        or type(blur) is not int or not 0 <= blur <= LUCID_MAX_BLUR
        or width != 3 * (2 * patch + 1) ** 2
    ):
        raise ValidationError("Invalid LUCID width or kernel radii")
    return 0, 255, True, np.uint8, cv2.NORM_HAMMING
