"""Explicit, seeded output-to-input distortion models for image regression."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import frame
from .processing import image_value
from .registration import BORDERS, INTERPOLATION
from .remap_maps import COORDINATE_MAP_TAG
from .selection import SELECTION_INPUTS, selection_mask

MAX_FIXTURE_PIXELS = 4_194_304
COEFFICIENTS = ("shear_x", "shear_y", "perspective_x", "perspective_y", "k1", "k2", "p1", "p2")
MODEL = "synthetic_distortion_v1"
_SEED = NumericRange(0, 2**32 - 1, whole=True)
_HALO = {"nearest": 0, "linear": 1, "cubic": 2, "lanczos4": 4}


def evaluate_distortion(x, y, coefficients, center, radius):
    """Analytic donor coordinates and local Jacobian, before float32 map rounding.

    Coordinates are normalized around center by radius. Projective/shear mapping
    precedes polynomial radial/tangential mapping. No inverse is inferred.
    """
    c = coefficients
    u, v = (x - center[0]) / radius, (y - center[1]) / radius
    denominator = 1 + c["perspective_x"] * u + c["perspective_y"] * v
    if np.any(denominator <= 0.05):
        raise ValidationError("Distortion approaches a projective horizon; reduce perspective")
    a = (u + c["shear_x"] * v) / denominator
    b = (v + c["shear_y"] * u) / denominator
    r2 = a * a + b * b
    radial = 1 + c["k1"] * r2 + c["k2"] * r2 * r2
    derivative = c["k1"] + 2 * c["k2"] * r2
    donor_x = a * radial + 2 * c["p1"] * a * b + c["p2"] * (r2 + 2 * a * a)
    donor_y = b * radial + c["p1"] * (r2 + 2 * b * b) + 2 * c["p2"] * a * b
    xx = radial + 2 * a * a * derivative + 2 * c["p1"] * b + 6 * c["p2"] * a
    xy = 2 * a * b * derivative + 2 * c["p1"] * a + 2 * c["p2"] * b
    yy = radial + 2 * b * b * derivative + 6 * c["p1"] * b + 2 * c["p2"] * a
    jacobian = (xx * yy - xy * xy) * (1 - c["shear_x"] * c["shear_y"]) / denominator**3
    return donor_x * radius + center[0], donor_y * radius + center[1], jacobian


class DistortImageFixture(Block):
    type_id, title = "cv.distortion_fixture", "Generate image distortion fixture"
    category, block_type, subtype = "Image processing", "Image processing", "Test fixtures"
    process_inputs_readonly = True
    payload_reference = "image"
    inputs = (
        *SELECTION_INPUTS,
        Port("seed", "number", required=False, required_when_connected=True),
        Port("points", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("map_x"),
        Port("map_y"),
        Port("valid_mask", visual_type="mask"),
        Port("mapping", "data"),
        Port("point_pairs", "data"),
        Port("seed", "number"),
        Port("valid_fraction", "number"),
    )
    defaults = {
        "coefficient_mode": "fixed",
        "seed": 0,
        **dict.fromkeys(COEFFICIENTS, 0.0),
        "center_x": 0.5,
        "center_y": 0.5,
        "radius_scale": 0.5,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
        "probe_points": [],
    }
    parameter_choices = {
        "coefficient_mode": ("fixed", "uniform_bounds"),
        "interpolation": tuple(INTERPOLATION),
        "border": tuple(BORDERS),
    }
    parameter_ranges = {
        "seed": _SEED,
        **{key: NumericRange(-0.75, 0.75) for key in COEFFICIENTS[:2]},
        **{key: NumericRange(-0.5, 0.5) for key in COEFFICIENTS[2:4]},
        **{key: NumericRange(-1, 1) for key in COEFFICIENTS[4:]},
        "center_x": NumericRange(0, 1),
        "center_y": NumericRange(0, 1),
        "radius_scale": NumericRange(0.1, 4),
        "border_value": NumericRange(),
    }
    parameter_help = {
        "coefficient_mode": (
            "Fixed uses signed coefficients. Uniform bounds draws each coefficient once "
            "from [-value,+value]; configured bounds must be nonnegative."
        ),
        "seed": "Private PCG64 generator. Same seed repeats the draw; wired Seed takes precedence.",
        "shear_x": "Normalized donor X includes shear_x times destination Y before perspective.",
        "shear_y": "Normalized donor Y includes shear_y times destination X before perspective.",
        "perspective_x": "Denominator is 1 + perspective_x*u + perspective_y*v.",
        "perspective_y": "Denominator coefficient for normalized destination Y.",
        "k1": "Radial multiplier includes k1*r^2 after projective/shear mapping.",
        "k2": "Radial multiplier includes k2*r^4 after projective/shear mapping.",
        "p1": "Tangential coefficient: X adds 2*p1*a*b; Y adds p1*(r^2+2*b^2).",
        "p2": "Tangential coefficient: X adds p2*(r^2+2*a^2); Y adds 2*p2*a*b.",
        "center_x": "Center X as a fraction of width-1. 0.5 is the image center.",
        "center_y": "Center Y as a fraction of height-1. 0.5 is the image center.",
        "radius_scale": "Normalization radius = radius_scale * max(width,height), in pixels.",
        "probe_points": "Output-grid [x,y] pairs, at most 256. Empty chooses nine grid probes.",
        "interpolation": (
            "Validity reserves the full source halo. "
            "Float64 cubic is rejected because of native boundary truncation."
        ),
        "border_value": "Constant border sample in every channel, including alpha.",
    }
    input_help = {
        "mask": "Optional source support; filters validity, not the full-image warp.",
        "region": "Optional source support; filters validity, not the full-image warp.",
        "points": "Optional output-grid probes override Probe points; wired empty means none.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["coefficient_mode"] == "uniform_bounds" and any(p[k] < 0 for k in COEFFICIENTS):
            raise ValidationError("Uniform coefficient bounds must be nonnegative")

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        image = image_value(source)
        if image.dtype == np.float64 and p["interpolation"] == "cubic":
            raise ValidationError(
                "Float64 cubic remap corrupts boundary samples; use linear or float32"
            )
        height, width = image.shape[:2]
        if width * height > MAX_FIXTURE_PIXELS or max(width, height) >= 32767:
            raise ValidationError(
                "Distortion fixtures are limited to 4,194,304 pixels and axes below 32767"
            )
        seed = _SEED.check(inputs.get("seed", p["seed"]), "Seed", "seed")
        coefficients = {name: float(p[name]) for name in COEFFICIENTS}
        if p["coefficient_mode"] == "uniform_bounds":
            rng = np.random.Generator(np.random.PCG64(seed))
            coefficients = {
                name: float(rng.uniform(-value, value)) for name, value in coefficients.items()
            }
        center = [(width - 1) * float(p["center_x"]), (height - 1) * float(p["center_y"])]
        radius = float(p["radius_scale"]) * max(width, height)
        yy, xx = np.indices((height, width), np.float64)
        donor_x, donor_y, jacobian = evaluate_distortion(xx, yy, coefficients, center, radius)
        if not np.isfinite(jacobian).all() or jacobian.min() <= 1e-6:
            raise ValidationError("Distortion has a nonpositive or near-singular sampled Jacobian")
        if any(not np.isfinite(a).all() or np.abs(a).max() > 32760 for a in (donor_x, donor_y)):
            raise ValidationError("Distortion donor coordinates exceed the safe remap range")
        mx, my = donor_x.astype(np.float32), donor_y.astype(np.float32)
        support = selection_mask(inputs, source)
        if support is None:
            support = np.full((height, width), 255, np.uint8)
        interpolation = p["interpolation"]
        halo = _HALO[interpolation]
        if halo:
            kernel = np.ones((2 * halo + 1, 2 * halo + 1), np.uint8)
            support = cv2.erode(support, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.remap(support, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid[(mx < 0) | (mx > width - 1) | (my < 0) | (my > height - 1)] = 0
        fill = (
            (float(p["border_value"]),) * image.shape[2]
            if image.ndim == 3
            else float(p["border_value"])
        )
        warped = cv2.remap(
            image,
            mx,
            my,
            INTERPOLATION[interpolation],
            borderMode=BORDERS[p["border"]],
            borderValue=fill,
        )
        if not np.isfinite(warped).all():
            raise ValidationError("Warped samples overflowed; rescale input or border values")

        incoming = copy.deepcopy(self.input_payloads["image"])
        source_context = {k: incoming[k] for k in ("spatial", "pyramid") if k in incoming}
        identity = {
            "model": MODEL,
            "source_size": [width, height],
            "center": center,
            "radius": radius,
            "coefficients": coefficients,
            "source_context": source_context,
        }
        if incoming["spatial"]["root_id"] is None:
            identity["anonymous_source_owner"] = self.id
        signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        output_spatial = frame([width, height], f"distortion-fixture:{signature}")
        output_payload = {k: v for k, v in incoming.items() if k not in ("spatial", "pyramid")}
        output_payload["spatial"] = output_spatial
        map_context = {
            "model": "image_coordinate_map_v1",
            "id": signature,
            "source_size": [width, height],
            "output_size": [width, height],
            "source_spatial": incoming["spatial"],
            "output_spatial": output_spatial,
        }
        mapping = {
            **identity,
            "output_spatial": output_spatial,
            "id": signature,
            "direction": "output_to_input",
            "minimum_sampled_jacobian": float(jacobian.min()),
            "seed": seed,
            "coefficient_mode": p["coefficient_mode"],
            "normalization": "u=(x-center_x)/radius; v=(y-center_y)/radius",
            "composition": "projective/shear then radial/tangential, then pixel denormalization",
            "map_quantization_max_pixels": float(
                max(np.abs(mx - donor_x).max(), np.abs(my - donor_y).max())
            ),
            "validity_halo_pixels": halo,
            "global_invertibility": "not_established",
        }
        probes = inputs.get("points", p["probe_points"])
        if "points" not in inputs and probes == []:
            probes = [
                [x, y]
                for y in (0, (height - 1) / 2, height - 1)
                for x in (0, (width - 1) / 2, width - 1)
            ]
        probes = np.asarray(probes, np.float64)
        if probes.size == 0:
            probes = probes.reshape(0, 2)
        if (
            probes.ndim != 2
            or probes.shape[1] != 2
            or len(probes) > 256
            or not np.isfinite(probes).all()
        ):
            raise ValidationError("Probe points must contain at most 256 finite [x,y] pairs")
        if len(probes) and (np.any(probes < 0) or np.any(probes > [width - 1, height - 1])):
            raise ValidationError("Probe points must be within the output pixel grid")
        px, py, _ = evaluate_distortion(probes[:, 0], probes[:, 1], coefficients, center, radius)
        pairs = [
            {
                "output": point.tolist(),
                "input": [float(x), float(y)],
                "input_center_in_bounds": bool(0 <= x <= width - 1 and 0 <= y <= height - 1),
            }
            for point, x, y in zip(probes, px, py, strict=True)
        ]
        outputs = {
            "image": ImageArray(warped, image_format(source)),
            "map_x": ImageArray(mx, "gray"),
            "map_y": ImageArray(my, "gray"),
            "valid_mask": ImageArray(valid, "gray"),
            "mapping": mapping,
            "point_pairs": pairs,
            "seed": seed,
            "valid_fraction": float(np.count_nonzero(valid) / valid.size),
        }
        payloads = {name: copy.deepcopy(output_payload) for name in outputs}
        for axis in ("x", "y"):
            payloads[f"map_{axis}"].pop("alpha_representation", None)
            payloads[f"map_{axis}"][COORDINATE_MAP_TAG] = {**map_context, "axis": axis}
        payloads["valid_mask"].pop("alpha_representation", None)
        drawings = {}
        if self.generate_graphics and pairs:
            objects = []
            for index, pair in enumerate(pairs):
                objects.append(graphic_object("marker", (pair["output"],), (50, 220, 120)))
                objects.append(
                    graphic_object("text", (pair["output"],), (50, 220, 120), text=str(index + 1))
                )
            drawings["output sampling probes"] = GraphicDrawing(tuple(objects), "image", True)
        return Result(
            outputs,
            payloads=payloads,
            drawings=drawings,
            message=f"{width} x {height} synthetic warp; {outputs['valid_fraction']:.1%} supported",
        )
