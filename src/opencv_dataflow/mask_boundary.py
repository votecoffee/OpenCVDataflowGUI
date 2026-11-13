"""Source-grid binary mask comparison with directed boundary and area measurements."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, require_same_frame
from .processing import image_value
from .selection import selection_mask
from .tickets import ticket_key

DISTANCES = {"euclidean": cv2.DIST_L2, "manhattan": cv2.DIST_L1, "chessboard": cv2.DIST_C}


def boundary(mask, connectivity, image_border):
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        if connectivity == "4"
        else np.ones((3, 3), np.uint8)
    )
    interior = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0 if image_border == "background" else 1,
    )
    return mask & (interior == 0)


def distance_to(points, parameters):
    if not points.any():
        return None
    aperture = parameters["l2_mask"]
    size = 0 if aperture == "precise" else int(aperture)
    if parameters["distance_type"] != "euclidean":
        size = 3
    return cv2.distanceTransform(np.uint8(~points), DISTANCES[parameters["distance_type"]], size)


def statistics(values, percentile):
    if values.size == 0:
        return None
    values = values.astype(np.float64)
    return {
        "count": int(values.size),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "percentile": float(np.percentile(values, percentile)),
    }


def outlines(mask, color, budget):
    """Bound diagnostic geometry independently of full-resolution measurements."""
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    objects, used = [], 0
    for contour in sorted(contours, key=len, reverse=True):
        if used >= budget or len(objects) >= 128:
            break
        points = contour.reshape(-1, 2)
        count = min(len(points), budget - used)
        points = points[np.linspace(0, len(points) - 1, count, dtype=int)]
        kind = "polyline" if len(points) > 1 else "marker"
        objects.append(graphic_object(kind, points, color, closed=True, thickness=1))
        used += count
    return tuple(objects)


class CompareMaskBoundary(Block):
    type_id, title = "inspect.mask_boundary", "Compare mask boundaries and area"
    category = block_type = "Inspection"
    subtype = "Measurements"
    parallel_safe = process_inputs_readonly = True
    payload_reference = "sample"
    inputs = (
        Port("reference"),
        Port("sample"),
        Port("image", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("distance", "number", required=False),
        Port("measurements", "data"),
        Port("sample_distance", required=False),
        Port("reference_distance", required=False),
        Port("sample_boundary"),
        Port("reference_boundary"),
        Port("tolerance_band"),
        Port("missing"),
        Port("extra"),
        Port("missing_beyond_tolerance"),
        Port("extra_beyond_tolerance"),
        Port("preview", required=False),
    )
    defaults = {
        "alignment": "same_frame",
        "tickets": "sample",
        "foreground": "nonzero",
        "connectivity": "8",
        "image_border": "background",
        "distance_type": "euclidean",
        "l2_mask": "precise",
        "direction": "both",
        "statistic": "maximum",
        "percentile": 95.0,
        "tolerance": 2.0,
        "check_boundary": True,
        "check_area": False,
        "area_basis": "beyond_tolerance",
        "maximum_missing_pixels": 0,
        "maximum_extra_pixels": 0,
        "empty_policy": "fail",
        "graphics_points": 4096,
    }
    parameter_choices = {
        "alignment": ("same_frame", "pixel_grid"),
        "tickets": ("sample", "require_match"),
        "foreground": ("nonzero", "zero"),
        "connectivity": ("4", "8"),
        "image_border": ("background", "ignore"),
        "distance_type": tuple(DISTANCES),
        "l2_mask": ("precise", "3", "5"),
        "direction": ("both", "sample_to_reference", "reference_to_sample"),
        "statistic": ("maximum", "percentile", "mean", "rms"),
        "area_basis": ("raw", "beyond_tolerance"),
        "empty_policy": ("fail", "both_empty_pass"),
    }
    parameter_ranges = {
        "percentile": NumericRange(0, 100),
        "tolerance": NumericRange(0),
        "maximum_missing_pixels": NumericRange(0, whole=True),
        "maximum_extra_pixels": NumericRange(0, whole=True),
        "graphics_points": NumericRange(64, 20000, whole=True),
    }
    parameter_help = {
        "alignment": "Require matching known frames, or compare already aligned pixels by index.",
        "tickets": "Sample owns results. Require match also checks a ticketed reference.",
        "foreground": "Interpret both monochrome masks using nonzero or zero samples.",
        "connectivity": "Inner boundary uses cross (4) or square (8) neighborhood erosion.",
        "image_border": "Assume background outside the image, or ignore unknown exterior pixels.",
        "distance_type": "Pixel-center Euclidean, Manhattan, or chessboard distance.",
        "l2_mask": "Precise L2 or approximate 3/5 chamfer mask. Other metrics force size 3.",
        "direction": "Select boundary queries; target boundaries always use the whole image.",
        "statistic": "Compare this statistic with tolerance; Both combines directed samples.",
        "percentile": "Percentile of directed/combined pixel-center boundary distances.",
        "tolerance": "Inclusive distance limit in current-image pixels, also used for area bands.",
        "check_boundary": "Judge the distance statistic and require valid boundary queries.",
        "check_area": "Also judge missing/extra pixel counts using the chosen area basis.",
        "area_basis": "Count all area differences, or only those farther than tolerance.",
        "maximum_missing_pixels": "Inclusive limit on expected foreground absent from the sample.",
        "maximum_extra_pixels": "Inclusive limit on unexpected sample foreground.",
        "empty_policy": "Fail absent selected foreground, or explicitly accept both empty masks.",
        "graphics_points": "Maximum outline vertices per layer; measurements use all pixels.",
    }
    input_help = {
        "reference": "Expected uint8/uint16 monochrome mask; nonzero is foreground by default.",
        "sample": "Observed mask of equal dimensions; owns output coordinates, ticket and payload.",
        "image": "Optional same-grid display image; preserves original pixels, depth and encoding.",
        "mask": "Measurement-center selection, not a crop or a cut in the target boundaries.",
        "region": "Source-grid region intersected with the optional measurement mask.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "distance", "measurements", "missing", "extra"]
        self.documentation = (
            "<p>Measure directed inner-boundary pixel-center distances and selected missing/extra "
            "foreground area. Holes and disconnected components participate. Selection restricts "
            "queries and area counts, while distance targets retain the full uncut boundaries. "
            "Distance maps are zero outside their companion boundary masks; missing targets "
            "withhold distance maps and statistics rather than inventing finite distances.</p>"
            "<p>Both combines the two directed sample sets, so mean/RMS/percentile are weighted "
            "by boundary pixel counts. Tolerance is inclusive. Beyond-tolerance area excludes "
            "differences within that distance of the opposite boundary. Image borders are an "
            "explicit exterior assumption. All measurements use current-grid pixels, not physical "
            "or automatically rescaled pyramid units. Float32 distances use a four-epsilon "
            "relative roundoff guard (at least four float32 epsilons) for all tolerance tests; "
            "tolerances of one million pixels or more need no guard. "
            "No registration is performed.</p>"
            '<p><a href="https://docs.opencv.org/5.0/main_modules/imgproc_misc.html">'
            "OpenCV distance transform reference</a>.</p>"
        )

    def process(self, inputs):
        p = self.parameters
        masks = []
        for name in ("reference", "sample"):
            raw = image_value(inputs[name], (np.uint8, np.uint16))
            if raw.ndim != 2 or raw.size == 0:
                raise ValidationError("Reference and sample must be nonempty monochrome masks")
            if raw.size > 4_194_304 or max(raw.shape) > 16384:
                raise ValidationError(
                    "Mask comparison is limited to 4,194,304 pixels and 16384 per axis"
                )
            masks.append(raw != 0 if p["foreground"] == "nonzero" else raw == 0)
        reference, sample = masks
        if reference.shape != sample.shape:
            raise ValidationError("Reference and sample must have matching pixel dimensions")
        shape = sample.shape
        source = inputs.get("image", inputs["sample"])
        image_value(source)
        if source.shape[:2] != shape:
            raise ValidationError("Display image must match the sample pixel dimensions")
        if p["alignment"] == "same_frame":
            try:
                require_same_frame([inputs["reference"], inputs["sample"], source])
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        payloads = self.input_payloads
        keys = ["sample", "image", "mask", "region"]
        if p["tickets"] == "require_match":
            keys.append("reference")
        tickets = [payloads[name]["ticket"] for name in keys if "ticket" in payloads.get(name, {})]
        if len({ticket_key(ticket) for ticket in tickets}) > 1:
            raise ValidationError("Sample, selection and display image must use the same ticket")
        selected = selection_mask({**inputs, "image": inputs["sample"]}, inputs["sample"])
        selected = np.ones(shape, bool) if selected is None else selected != 0
        if not selected.any():
            raise ValidationError("Select at least one measurement pixel")
        ref_edge, sample_edge = (
            boundary(mask, p["connectivity"], p["image_border"]) for mask in masks
        )
        to_reference, to_sample = distance_to(ref_edge, p), distance_to(sample_edge, p)
        ref_query, sample_query = ref_edge & selected, sample_edge & selected
        missing, extra = reference & ~sample & selected, sample & ~reference & selected
        tol = float(p["tolerance"])
        # Native distance maps are float32. Use the same roundoff guard for
        # scalar judgments, per-pixel violations and the displayed tolerance band.
        epsilon = 4 * np.finfo(np.float32).eps * max(1, tol) if tol < 1e6 else 0.0
        # No metric can exceed H+W on this grid. Capping the comparison threshold
        # also avoids narrowing a very large finite user limit to float32 infinity.
        limit = min(tol + epsilon, float(sum(shape)))
        missing_far = missing if to_sample is None else missing & (to_sample > limit)
        extra_far = extra if to_reference is None else extra & (to_reference > limit)
        queries = {
            "sample_to_reference": (sample_query, to_reference),
            "reference_to_sample": (ref_query, to_sample),
        }
        reports, samples, outputs = {}, {}, {}
        for name, (query, target) in queries.items():
            values = target[query] if target is not None else np.empty(0, np.float32)
            samples[name] = values
            reports[name] = statistics(values, p["percentile"])
            if reports[name] is not None:
                reports[name]["violation_count"] = int(np.count_nonzero(values > limit))
            port = "sample_distance" if name == "sample_to_reference" else "reference_distance"
            outputs[port] = NO_VALUE if target is None else np.where(query, target, 0)
        names = list(queries) if p["direction"] == "both" else [p["direction"]]
        combined = statistics(np.concatenate([samples[name] for name in names]), p["percentile"])
        valid = all(reports[name] is not None for name in names)
        value = combined[p["statistic"]] if valid else NO_VALUE
        ref_count, sample_count = (int(np.count_nonzero(mask & selected)) for mask in masks)
        intersection = int(np.count_nonzero(reference & sample & selected))
        union = int(np.count_nonzero((reference | sample) & selected))
        counts = {
            "missing_pixels": int(missing.sum()),
            "extra_pixels": int(extra.sum()),
            "missing_beyond_tolerance_pixels": int(missing_far.sum()),
            "extra_beyond_tolerance_pixels": int(extra_far.sum()),
        }
        reasons = []
        accepted_empty = ref_count == sample_count == 0 and p["empty_policy"] == "both_empty_pass"
        if not accepted_empty:
            if not ref_count:
                reasons.append("mask_reference_empty")
            if not sample_count:
                reasons.append("mask_sample_empty")
            if p["check_boundary"]:
                if not valid:
                    reasons.append("mask_boundary_unavailable")
                elif value > limit:
                    reasons.append("mask_boundary_out_of_tolerance")
        if p["check_area"]:
            suffix = (
                "_beyond_tolerance_pixels" if p["area_basis"] == "beyond_tolerance" else "_pixels"
            )
            for kind in ("missing", "extra"):
                if counts[kind + suffix] > p[f"maximum_{kind}_pixels"]:
                    reasons.append(f"mask_{kind}_area_above_limit")
        info = {
            "model": "mask_boundary_comparison_v1",
            "units": "current_pixel",
            "area_units": "current_pixel_squared",
            "image_size": list(shape[::-1]),
            "selected_pixels": int(selected.sum()),
            "reference_pixels": ref_count,
            "sample_pixels": sample_count,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": intersection / union if union else 1.0,
            **counts,
            "missing_reference_fraction": counts["missing_pixels"] / ref_count
            if ref_count
            else None,
            "extra_sample_fraction": counts["extra_pixels"] / sample_count
            if sample_count
            else None,
            "directed": reports,
            "combined": combined,
            "distance_valid": valid,
            "direction": p["direction"],
            "statistic": p["statistic"],
            "percentile": p["percentile"],
            "tolerance": tol,
            "comparison_epsilon": float(epsilon),
            "distance_type": p["distance_type"],
            "l2_mask": p["l2_mask"],
            "connectivity": p["connectivity"],
            "image_border": p["image_border"],
            "accepted_empty": accepted_empty,
        }
        outputs.update(
            {
                "image": source,
                "distance": value,
                "measurements": info,
                "sample_boundary": np.uint8(sample_query) * 255,
                "reference_boundary": np.uint8(ref_query) * 255,
                "tolerance_band": np.uint8(selected & (to_reference <= limit)) * 255
                if to_reference is not None
                else np.zeros(shape, np.uint8),
                **{
                    name: np.uint8(mask) * 255
                    for name, mask in (
                        ("missing", missing),
                        ("extra", extra),
                        ("missing_beyond_tolerance", missing_far),
                        ("extra_beyond_tolerance", extra_far),
                    )
                },
                "preview": NO_VALUE,
            }
        )
        payload = copy.deepcopy(payloads["sample"])
        payload.pop("alpha_representation", None)
        payload["mask_comparison_reference"] = {
            key: copy.deepcopy(payloads["reference"][key])
            for key in ("spatial", "pyramid", "ticket")
            if key in payloads["reference"]
        }
        graphics, drawings = {}, {}
        if self.generate_graphics:
            preview = overlay_image(source)
            preview[ref_query] = (230, 190, 30)
            preview[sample_query] = (60, 230, 120)
            preview[missing] = (30, 80, 240)
            preview[extra] = (220, 80, 220)
            outputs["preview"] = ImageArray(preview, "bgr")
            graphics["mask comparison"] = outputs["preview"]
            for name, mask, color in (
                ("reference", reference, (230, 190, 30)),
                ("sample", sample, (60, 230, 120)),
                ("missing", missing, (30, 80, 240)),
                ("extra", extra, (220, 80, 220)),
            ):
                drawings[name] = GraphicDrawing(
                    outlines(mask, color, p["graphics_points"]), "image", True
                )
        output_payloads = {name: payload for name in outputs}
        display_payload = payloads.get("image", payloads["sample"])
        if "alpha_representation" in display_payload:
            output_payloads["image"] = {
                **payload,
                "alpha_representation": display_payload["alpha_representation"],
            }
        return Result(
            outputs,
            payloads=output_payloads,
            graphics=graphics,
            drawings=drawings,
            passed=not reasons,
            score=info["iou"],
            fail_reasons=reasons,
            message=f"Missing {counts['missing_pixels']} px / extra {counts['extra_pixels']} px",
        )
