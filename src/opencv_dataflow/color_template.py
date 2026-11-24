"""Fixed-scale color phase correlation with source-grid template positions."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_box
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, moved_payload
from .poses import POSE_HELP, POSE_OFFSETS, local_offset, make_pose
from .processing import image_value, refine_template_peak, template_candidate_mask
from .result_graphics import ACCEPTED, REJECTED

# Native color correlation keeps several four-channel float64 spectra alive.
MAX_DFT_PIXELS = 1_048_576


def working_color(image, field):
    pixels = image_value(image)
    encoding = image_format(image)
    if pixels.ndim != 3 or encoding not in ("bgr", "rgb", "bgra", "rgba", "hsv", "lab"):
        raise ValidationError(f"{field.title()} must contain color channels", field)
    if encoding in ("hsv", "lab"):
        if pixels.dtype not in (np.uint8, np.float32):
            raise ValidationError("HSV and Lab require uint8 or float32 color data", field)
        pixels = image_value(image, canonical=True)
    else:
        pixels = pixels[..., :3]
        if encoding in ("rgb", "rgba"):
            pixels = pixels[..., ::-1]
    return np.ascontiguousarray(pixels)


class MatchColorTemplate(Block):
    type_id, title = "inspect.match_color_template", "Match color template"
    category = block_type = "Inspection"
    subtype = "Matching"
    description = "Locate a fixed-size color template with joint quaternion phase correlation."
    inputs = (
        Port("image"),
        Port("template"),
        Port("search_mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("score", "number"),
        Port("location", "data"),
        Port("center", "data"),
        Port("pose", "data", required=False),
        Port("response"),
    )
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {"min_score": 0.25, "subpixel": False, **POSE_OFFSETS}
    parameter_ranges = {
        "min_score": NumericRange(0, 1),
        "pose_offset_x": NumericRange(-1_000_000, 1_000_000),
        "pose_offset_y": NumericRange(-1_000_000, 1_000_000),
        "pose_offset_angle": NumericRange(),
    }
    parameter_help = {
        **POSE_HELP,
        "min_score": "Minimum native phase-correlation peak. Tune for the actual scene and template.",
        "subpixel": "Refine an interior peak by a local quadratic fit when its neighbors are eligible.",
    }
    input_help = {
        "image": "Color search image at the same scale as the template.",
        "template": "Tightly cropped color patch. RGB, BGR, HSV and Lab are converted to BGR working pixels.",
        "search_mask": "Accept peaks whose template center lies on a nonzero search-image mask pixel.",
        "region": "Restrict eligible template centers to this region in the source grid.",
    }
    output_help = {
        "image": "Original search image with drawable match bounds in the result view.",
        "score": "Native phase-correlation magnitude at the selected peak.",
        "location": "Best integer top-left and bounds in source pixels, plus a refined position when available.",
        "center": "Single best template center in source pixels before the optional pose offset.",
        "pose": "Source-grid pose at the accepted template center, composed with the configured local offset.",
        "response": "Float64 scores for complete placements. Each pixel indexes a template top-left.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", "template"]
        self.preview_output = "image"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["subpixel"]) is not bool:
            raise ValidationError("Subpixel must be boolean", "subpixel")

    def process(self, inputs):
        source, template = inputs["image"], inputs["template"]
        h, w = source.shape[:2]
        th, tw = template.shape[:2]
        if min(th, tw) < 1 or th > h or tw > w:
            raise ValidationError("Color template must be nonempty and fit the search image", "template")
        padded_h, padded_w = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
        if min(padded_h, padded_w) < 1 or padded_h * padded_w > MAX_DFT_PIXELS:
            raise ValidationError("Color correlation exceeds the padded transform limit. Crop the search image.")
        response_shape = (h - th + 1, w - tw + 1)
        candidates = template_candidate_mask(inputs, source, (th, tw), response_shape)

        native_match = getattr(getattr(cv2, "ximgproc", None), "colorMatchTemplate", None)
        if native_match is None:
            raise ValidationError("Color template matching requires OpenCV contrib ximgproc")
        raw = native_match(working_color(source, "image"), working_color(template, "template"))
        if raw is None or raw.dtype != np.float64 or raw.shape != (padded_h, padded_w):
            raise ValidationError("OpenCV returned an unexpected color correlation surface")
        if not np.isfinite(raw).all():
            raise ValidationError(
                "Color correlation is undefined for this spectrum. Use a textured template and search image."
            )
        # Zero lag is at (0,0). Native correlation stores a positive placement
        # at the wrapped negative lag on each padded axis.
        rows, cols = response_shape
        response = raw[np.ix_(-np.arange(rows) % padded_h, -np.arange(cols) % padded_w)]
        _, score, _, (x, y) = cv2.minMaxLoc(response, mask=candidates)
        return self.match_result(source, response, candidates, (x, y), (tw, th), score)

    def match_result(self, source, response, candidates, location, template_size, score):
        p = self.parameters
        x, y = location
        tw, th = template_size
        source_size = [source.shape[1], source.shape[0]]
        refined = refine_template_peak(response, candidates, x, y) if p["subpixel"] else None
        px, py = refined if refined is not None else (x, y)
        center = [px + (tw - 1) / 2, py + (th - 1) / 2]
        passed = score >= float(p["min_score"])
        pose = NO_VALUE
        payload = self.input_payloads["image"]
        if passed:
            matrix = np.eye(3)
            matrix[:2, 2] = center
            pose = make_pose(matrix @ local_offset(*(p[k] for k in POSE_OFFSETS)), source_size, payload)
        details = {
            "top_left": [x, y],
            "bounds": [x, y, tw, th],
            "method": "quaternion_phase_correlation",
        }
        if refined is not None:
            details["subpixel_top_left"] = refined
        drawings, graphics = {}, {}
        if self.generate_graphics:
            color = ACCEPTED if passed else REJECTED
            drawings["match"] = GraphicDrawing((graphic_box(x, y, tw, th, color),))
            gray = np.rint(np.clip(response, 0, 1) * 255).astype(np.uint8)
            graphics["phase response"] = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        response_payload = moved_payload(payload, np.eye(3), [response.shape[1], response.shape[0]])
        for field in ("alpha_representation", "raw_sensor"):
            response_payload.pop(field, None)
        return Result(
            {
                "image": source,
                "score": score,
                "location": details,
                "center": [center],
                "pose": pose,
                "response": response,
            },
            payloads={"response": response_payload},
            passed=passed,
            score=min(1.0, score),
            message=f"Color peak {score:.4f} at ({x}, {y})",
            fail_reasons=[] if passed else ["color_template_score_below_minimum"],
            graphics=graphics,
            drawings=drawings,
        )
