"""Bounded neighboring-frame border fills with explicitly unchanged measured support."""

import copy

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .core import Block, Port, Result
from .flow_warp import sample_mask
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray
from .motion_trajectory import affine
from .offline_stabilization import _frames
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, check_spatial, metadata
from .processing import image_value
from .selection import selection_mask
from .tickets import ticket_key


def eligible_holes(coverage, mode):
    holes = ~coverage
    if mode == "all":
        return holes
    _, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return holes & np.isin(labels, border[border != 0])


def remap_neighbor(pixels, support, pull, x, y, method, scales):
    if np.allclose(pull, np.eye(3), atol=1e-12, rtol=0):
        return pixels, support
    mx = (pull[0, 0] * x + pull[0, 1] * y + pull[0, 2]).astype(np.float32)
    my = (pull[1, 0] * x + pull[1, 1] * y + pull[1, 2]).astype(np.float32)
    valid = sample_mask(support, mx, my, method)
    interpolation = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
    values = pixels
    normalized = method == "linear" and pixels.dtype.kind == "f"
    if normalized:
        values = pixels.astype(np.float64) / scales
    warped = cv2.remap(values, mx, my, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if normalized:
        warped = (warped * scales).astype(pixels.dtype)
    if not np.isfinite(warped[valid]).all():
        raise ValidationError("Neighbor sampling produced nonfinite pixels")
    return warped, valid


class FillStabilizationBorders(Block):
    type_id, title = "sequence.fill_stabilization_borders", "Fill stabilization borders for review"
    category = block_type = "Image processing"
    subtype = "Video restoration"
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "frames"
    inputs = (
        Port("frames", "any"),
        Port("coverage", "any"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"),
        Port("display"),
        Port("coverage", visual_type="region"),
        Port("filled_mask", visual_type="region"),
        Port("remaining_mask", visual_type="region"),
        Port("display_support", visual_type="region"),
        Port("donor_index"),
        Port("contributors"),
        Port("report", "data"),
        Port("preview", required=False),
    )
    defaults = {
        "target_index": 0,
        "temporal_radius": 2,
        "alignment": "spatial",
        "different_segment": "skip",
        "coverage_tickets": "matching_if_known",
        "interpolation": "linear",
        "holes": "border_only",
        "blend": "nearest",
        "consistency": "overlap",
        "maximum_error": 0.03,
        "minimum_overlap_pixels": 64,
        "minimum_consistent_fraction": 0.8,
        "change_radius": 3,
        "maximum_fill_distance": 32.0,
        "minimum_fill_fraction": 0.0,
        "unfilled": "allow",
        "maximum_working_bytes": 512 * 1024**2,
        "maximum_graphics_points": 256,
    }
    parameter_choices = {
        "alignment": ("spatial", "pixel_grid"),
        "different_segment": ("skip", "error"),
        "coverage_tickets": ("matching_if_known", "ignore"),
        "interpolation": ("nearest", "linear"),
        "holes": ("border_only", "all"),
        "blend": ("nearest", "mean"),
        "consistency": ("overlap", "off"),
        "unfilled": ("allow", "fail"),
    }
    parameter_ranges = {
        "target_index": NumericRange(0, 63, whole=True),
        "temporal_radius": NumericRange(1, 63, whole=True),
        "maximum_error": NumericRange(0, 2),
        "minimum_overlap_pixels": NumericRange(1, 16_777_216, whole=True),
        "minimum_consistent_fraction": NumericRange(0, 1),
        "change_radius": NumericRange(0, 63, whole=True),
        "maximum_fill_distance": NumericRange(0, 4096),
        "minimum_fill_fraction": NumericRange(0, 1),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
        "maximum_graphics_points": NumericRange(0, 4096, whole=True),
    }
    parameter_help = {
        "target_index": "Current capture index; retain its ticket, timing and coordinate frame.",
        "temporal_radius": "Only use neighbors within this many ordered batch positions.",
        "alignment": "Spatial uses a shared known root; pixel grid asserts registration.",
        "different_segment": "Skip another known root, or reject it. Unknown spatial roots reject.",
        "coverage_tickets": "Check known per-image/mask acquisition tickets, or explicitly ignore.",
        "interpolation": "Resample neighboring measured pixels with full kernel support.",
        "holes": "Only border-connected geometric holes, or all geometric holes.",
        "blend": "Nearest capture wins (earlier index breaks ties), or mean accepted samples.",
        "consistency": "Require measured overlap, or explicitly disable the photometric check.",
        "maximum_error": "Maximum channel error divided by shared integer/full-batch float scale.",
        "minimum_overlap_pixels": "Minimum shared measured pixels for overlap consistency.",
        "minimum_consistent_fraction": "Minimum fraction of shared pixels within Maximum error.",
        "change_radius": "Exclude this halo around detected overlap changes before filling.",
        "maximum_fill_distance": "Distance to current measured support; zero means unlimited.",
        "minimum_fill_fraction": "Required filled fraction of selected holes; no holes means 1.",
        "unfilled": "Allow partial review fills, or fail if any selected eligible hole remains.",
        "maximum_working_bytes": "Array estimate before preparation; excludes graph/native copies.",
        "maximum_graphics_points": "Bound fill/missing contour vertices and review annotation.",
    }
    input_help = {
        "frames": "2-64 corrected gray/RGB(A)/BGR(A) images, same dimensions and pixel depth.",
        "coverage": "Required ordered group of original measured uint8 masks, one per frame.",
        "mask": "Optional current-grid selection of fill centers; measured pixels never change.",
        "region": "Optional current-grid region intersected with Mask.",
    }
    output_help = {
        "image": "Current measured image; zero outside original measured Coverage. Never filled.",
        "display": "Review-only image, with neighboring samples in originally uncovered pixels.",
        "coverage": "Original current support, intersected with positive alpha; never expanded.",
        "filled_mask": "Review pixels filled from other captures, disjoint from Coverage.",
        "remaining_mask": "All still unsupported current pixels, including unselected holes.",
        "display_support": "Review availability only; union of measured and filled pixels.",
        "donor_index": "First contributing batch index, or -1. Nearest mode uses it alone.",
        "contributors": "Number of eligible donors per pixel; nearest mode uses one.",
        "report": "Current identity, selections, geometry, overlap checks and donor provenance.",
        "preview": "Optional fill/support colors on the current grid; omitted with graphics off.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "display", "coverage", "filled_mask", "report"]
        self.description = (
            self.description or "Fill uncovered review pixels without inventing coverage."
        )
        self.documentation = self.documentation or (
            "<p>Transfer measured samples from bounded neighboring corrected frames onto the "
            "current grid. Optional overlap checks reject lighting/scene changes and exclude "
            "a local change halo. Only Display is filled; Image/Coverage stay current measured "
            "data. Neighbors cannot prove the current scene in an unseen area. "
            "See docs/stabilization-fill.md.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or (limits.whole and type(p[name]) is not int):
                raise ValidationError("Use typed numeric border-fill parameters", name)

    def process(self, inputs):
        p, group = self.parameters, inputs["frames"]
        if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 64:
            raise ValidationError("Use 2-64 corrected image packets")
        raw = [entry.value for entry in group.items]
        if any(not isinstance(a, np.ndarray) or a.ndim not in (2, 3) for a in raw):
            raise ValidationError("Every corrected entry must be an image")
        h, w = raw[0].shape[:2]
        if max(h, w) > 16384:
            raise ValidationError("Border fill supports at most 16384 pixels per axis")
        channels = 1 if raw[0].ndim == 2 else raw[0].shape[2]
        estimate = sum(a.nbytes for a in raw) + h * w * (len(raw) * 2 + 160 + channels * 64)
        if estimate > p["maximum_working_bytes"]:
            raise ValidationError(
                "Border fill exceeds the working array budget", "maximum_working_bytes"
            )
        images, contexts, encoding = _frames(group)
        n, target = len(images), p["target_index"]
        if target >= n:
            raise ValidationError("Target index is outside the corrected batch", "target_index")
        alpha = encoding in ("rgba", "bgra")
        if images[0].shape[2:] != (() if encoding == "gray" else (4,) if alpha else (3,)):
            raise ValidationError("Channel shape must match the declared image encoding")
        for image, context in zip(images, contexts, strict=True):
            image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
            if context.get("stabilization_border_fill", {}).get("review_only"):
                raise ValidationError(
                    "Use measured image inputs, not previously filled review images"
                )
            if alpha:
                maximum = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1
                if context.get("alpha_representation", "straight") != "straight":
                    raise ValidationError("Unassociate premultiplied image alpha first")
                if np.any(image[..., 3] < 0) or np.any(image[..., 3] > maximum):
                    raise ValidationError(
                        "Alpha must be within integer full scale or floating [0,1]"
                    )
        coverage = inputs["coverage"]
        if not isinstance(coverage, FlowGroup) or coverage.labels != group.labels:
            raise ValidationError("Coverage labels/order must match corrected Frames")
        support, geometric = [], []
        for i, entry in enumerate(coverage.items):
            mask, context = entry.value, metadata(entry)
            if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != (h, w):
                raise ValidationError("Coverage must be a matching uint8 mask per corrected image")
            check_spatial(context, [w, h])
            a, b = contexts[i].get("spatial"), context.get("spatial")
            if b is not None and (a is None or b != a):
                raise ValidationError("Coverage and corresponding corrected image grids differ")
            expected, actual = contexts[i].get("ticket"), context.get("ticket")
            if (
                p["coverage_tickets"] == "matching_if_known"
                and expected is not None
                and actual is not None
                and ticket_key(expected) != ticket_key(actual)
            ):
                raise ValidationError("Coverage and corrected image tickets differ")
            geometric.append(mask != 0)
            support.append((mask != 0) & (images[i][..., 3] > 0) if alpha else mask != 0)
        source, context = np.asarray(images[target]), contexts[target]
        selected = selection_mask({**inputs, "image": images[target]}, images[target])
        selected = np.ones((h, w), bool) if selected is None else selected != 0
        if not selected.any():
            raise ValidationError("Select at least one current-grid pixel")
        measured = support[target]
        requested = eligible_holes(geometric[target], p["holes"]) & selected
        holes = requested.copy()
        if p["maximum_fill_distance"] > 0:
            distance = cv2.distanceTransform(
                (~measured).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            holes &= distance <= p["maximum_fill_distance"]
        # Keep the denominator before distance filtering so limits cannot hide missing support.
        dtype = source.dtype
        color_planes = [np.asarray(a)[..., :3] if alpha else np.asarray(a) for a in images]
        scale = (
            float(np.iinfo(dtype).max)
            if dtype.kind == "u"
            else max(float(np.max(np.abs(a))) for a in color_planes) or 1.0
        )
        scales = (
            np.array(
                [scale] * (channels - 1)
                + [float(np.iinfo(dtype).max) if dtype.kind == "u" else 1.0]
            )
            if alpha
            else scale
        )
        current = source.astype(np.float64) / scales
        output = source.copy()
        output[~measured] = 0
        display = output.copy()
        sums = np.zeros_like(current) if p["blend"] == "mean" else None
        count = np.zeros((h, w), np.uint8)
        winner = np.full((h, w), -1, np.int16)
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        records = [
            {
                "index": i,
                "label": group.labels[i],
                "ticket": contexts[i].get("ticket"),
                "state": "target" if i == target else "outside_radius",
                "used_pixels": 0,
            }
            for i in range(n)
        ]
        target_grid = context.get("spatial")
        if p["alignment"] == "spatial" and (not target_grid or target_grid.get("root_id") is None):
            raise ValidationError("Spatial alignment requires a known current coordinate root")
        for i in sorted(range(n), key=lambda i: (abs(i - target), i)):
            if i == target or abs(i - target) > p["temporal_radius"]:
                continue
            row, grid = records[i], contexts[i].get("spatial")
            pull = np.eye(3)
            if p["alignment"] == "spatial":
                if not grid or grid.get("root_id") is None:
                    raise ValidationError("Spatial alignment requires known neighbor roots")
                if grid["root_id"] != target_grid["root_id"]:
                    if p["different_segment"] == "error":
                        raise ValidationError("Neighbor belongs to another coordinate segment")
                    row["state"] = "different_segment"
                    continue
                if grid["root_size"] != target_grid["root_size"]:
                    raise ValidationError("Shared coordinate root has inconsistent extents")
                pull = affine(
                    np.linalg.solve(affine(grid["to_root"]), affine(target_grid["to_root"]))
                )
            row["target_to_neighbor"] = pull.tolist()
            warped, valid = remap_neighbor(
                np.asarray(images[i]), support[i], pull, x, y, p["interpolation"], scales
            )
            overlap = measured & valid
            row["overlap_pixels"] = int(overlap.sum())
            changed = np.zeros((h, w), bool)
            if p["consistency"] == "overlap":
                error = np.abs(warped.astype(np.float64) / scales - current)
                if error.ndim == 3:
                    error = error.max(axis=2)
                good = overlap & (error <= p["maximum_error"])
                fraction = float(good.sum() / overlap.sum()) if overlap.any() else 0.0
                row["consistent_fraction"] = fraction
                row["maximum_overlap_error"] = (
                    float(error[overlap].max()) if overlap.any() else None
                )
                if (
                    overlap.sum() < p["minimum_overlap_pixels"]
                    or fraction < p["minimum_consistent_fraction"]
                ):
                    row["state"] = "inconsistent_overlap"
                    continue
                radius = p["change_radius"]
                changed = (
                    cv2.dilate(
                        (overlap & ~good).astype(np.uint8),
                        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                    )
                    != 0
                )
            accepted = holes & valid & ~changed
            row.update(
                state="considered",
                eligible_pixels=int(accepted.sum()),
                changed_halo_pixels=int(np.count_nonzero(holes & changed)),
            )
            first = accepted & (count == 0)
            winner[first] = i
            if sums is None:
                display[first] = warped[first]
                row["used_pixels"] = int(first.sum())
            else:
                old = count[accepted].astype(np.float64)
                weight = old / (old + 1)
                values = warped[accepted].astype(np.float64) / scales
                sums[accepted] = sums[accepted] * (weight[:, None] if source.ndim == 3 else weight)
                sums[accepted] += values / ((old + 1)[:, None] if source.ndim == 3 else old + 1)
                row["used_pixels"] = int(accepted.sum())
            count += accepted
        filled = count > 0
        if sums is not None:
            values = cast_intensity(sums * scales, dtype)
            display[filled] = values[filled]
        available = measured | filled
        remaining = ~available
        requested_count = int(requested.sum())
        fraction = float(filled.sum() / requested_count) if requested_count else 1.0
        passed = bool(
            measured.any()
            and fraction >= p["minimum_fill_fraction"]
            and (p["unfilled"] == "allow" or not np.any(requested & ~filled))
        )
        status = (
            "filled" if filled.any() else "no_fill_needed" if not requested_count else "unfilled"
        )
        report = {
            "model": "stabilization_border_fill_v1",
            "status": status,
            "target_index": target,
            "target_ticket": context.get("ticket"),
            "alignment": p["alignment"],
            "measured_pixels": int(measured.sum()),
            "requested_pixels": requested_count,
            "filled_pixels": int(filled.sum()),
            "fill_fraction": fraction,
            "remaining_pixels": int(remaining.sum()),
            "complete": not bool(np.any(requested & ~filled)),
            "scale": scale,
            "neighbors": records,
            "estimated_working_bytes": estimate,
        }
        payload = copy.deepcopy(context)
        payload.pop("outcome", None)
        payload["stabilization_border_fill"] = {
            "target_index": target,
            "requires_coverage": True,
            "review_only": False,
        }
        review = copy.deepcopy(payload)
        review["stabilization_border_fill"]["review_only"] = True
        map_payload = copy.deepcopy(payload)
        map_payload.pop("alpha_representation", None)
        outputs = {
            "image": ImageArray(output, encoding),
            "display": ImageArray(display, encoding),
            "coverage": np.uint8(measured) * 255,
            "filled_mask": np.uint8(filled) * 255,
            "remaining_mask": np.uint8(remaining) * 255,
            "display_support": np.uint8(available) * 255,
            "donor_index": winner,
            "contributors": count,
            "report": report,
            "preview": NO_VALUE,
        }
        drawings = {}
        if self.generate_graphics:
            preview = np.zeros((h, w, 3), np.uint8)
            preview[measured], preview[filled], preview[remaining] = (
                (80, 130, 70),
                (40, 180, 240),
                (65, 65, 160),
            )
            outputs["preview"] = ImageArray(preview, "bgr")
            limit, objects = p["maximum_graphics_points"], []
            for mask, color in ((filled, (40, 180, 240)), (remaining, (65, 65, 220))):
                if limit < 3:
                    break
                contours, _ = cv2.findContours(
                    np.uint8(mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    points = cv2.approxPolyDP(contour, 1, True).reshape(-1, 2)
                    if 3 <= len(points) <= limit:
                        objects.append(graphic_object("polygon", points, color))
                        limit -= len(points)
            if limit:
                objects.append(
                    graphic_object(
                        "text",
                        [[5, 16]],
                        (40, 180, 240),
                        text=f"Review fill {int(filled.sum())} pixels",
                    )
                )
            if objects:
                drawings["Review border fills"] = GraphicDrawing(tuple(objects), "display", True)
        return Result(
            outputs,
            payloads={
                **{name: map_payload for name in outputs},
                "image": payload,
                "display": review,
            },
            drawings=drawings,
            passed=passed,
            score=fraction,
            fail_reasons=[] if passed else ["stabilization_fill.insufficient_review_support"],
            message=f"Review filled {int(filled.sum())}/{requested_count} requested pixels",
        )
