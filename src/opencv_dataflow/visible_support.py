"""Qualify projected template appearance before image-plane pick selection."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .flow_warp import sample_mask
from .graphics_layers import GraphicDrawing, graphic_object
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .poses import local_offset, make_pose, pose_for_image, validate_pose
from .processing import image_value
from .result_graphics import ACCEPTED, REJECTED
from .selection import region_matrix, selection_mask
from .tickets import ticket_key


def comparison_pixels(image, mode):
    values = image_value(image)
    encoding = image_format(image)
    if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Convert comparison images to grayscale, RGB or BGR first")
    if values.ndim == 3:
        values = values[..., :3]  # Alpha is not appearance evidence.
        if encoding in ("rgb", "rgba"):
            values = values[..., ::-1]
    values = np.asarray(values, dtype=np.float64, order="C")
    if values.ndim == 3 and mode == "grayscale":
        values = values @ np.array([0.114, 0.587, 0.299])
    return values


def same_ticket(source, target):
    a, b = source.get("ticket"), target.get("ticket")
    if a is not None and b is not None and ticket_key(a) != ticket_key(b):
        raise ValidationError("Candidate and current image tickets differ")


class VisibleSupport(Block):
    type_id, title = "inspect.visible_support", "Filter candidates by visible support"
    category, block_type, subtype = "Inspection", "Inspection", "Pick selection"
    inputs = (
        Port("image"),
        Port("template"),
        Port("matches", "data"),
        Port("mask", required=False, visual_type="region"),
        Port("region", "data", required=False, visual_type="region"),
        Port("template_mask", required=False, visual_type="region"),
        Port("template_region", "data", required=False, visual_type="region"),
    )
    outputs = (
        Port("image"),
        Port("matches", "data"),
        Port("candidates", "data"),
        Port("count", "number"),
        Port("report", "data"),
    )
    defaults = {
        "transform_mode": "auto",
        "template_grid_policy": "matching_frames",
        "template_origin": [],
        "comparison": "grayscale",
        "interpolation": "linear",
        "absolute_tolerance": 20.0,
        "minimum_visible_fraction": 0.85,
        "minimum_observed_fraction": 0.95,
        "minimum_matches": 1,
        "maximum_candidates": 64,
        "maximum_source_pixels": 8_000_000,
        "maximum_template_pixels": 262144,
        "maximum_sample_count": 16_777_216,
        "maximum_working_bytes": 256 * 1024**2,
        "maximum_graphics_candidates": 64,
    }
    parameter_choices = {
        "transform_mode": ("auto", "transform", "bounds", "pose"),
        "template_grid_policy": ("matching_frames", "assert_pixel_alignment"),
        "comparison": ("grayscale", "all_channels"),
        "interpolation": ("nearest", "linear"),
    }
    parameter_ranges = {
        "absolute_tolerance": NumericRange(0),
        "minimum_visible_fraction": NumericRange(0, 1),
        "minimum_observed_fraction": NumericRange(0, 1),
        "minimum_matches": NumericRange(0, 64, whole=True),
        "maximum_candidates": NumericRange(1, 64, whole=True),
        "maximum_source_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_template_pixels": NumericRange(1, 1_048_576, whole=True),
        "maximum_sample_count": NumericRange(1, 67_108_864, whole=True),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
        "maximum_graphics_candidates": NumericRange(0, 64, whole=True),
    }
    parameter_help = {
        "transform_mode": "Auto prefers the record's transform, then bounds, then pose.",
        "template_grid_policy": "Match known frames, or assert template pixel alignment.",
        "template_origin": "Pose mode: template [x,y] at the pose origin; empty uses pixel center.",
        "comparison": "Grayscale BGR weights, or agreement in every BGR channel; alpha is ignored.",
        "interpolation": "Sample the scene on the template grid; border fill never adds support.",
        "absolute_tolerance": "Maximum absolute channel difference, in raw image units.",
        "minimum_visible_fraction": "Matching observed samples / all selected template samples.",
        "minimum_observed_fraction": "Fully observed samples / all selected template samples.",
        "minimum_matches": "Minimum accepted candidates for the block to pass; zero permits none.",
        "maximum_candidates": "Reject oversized lists; never silently truncate candidates.",
        "maximum_source_pixels": "Source image pixel limit before comparison allocations.",
        "maximum_template_pixels": "Full template-grid pixel limit, including masked-out pixels.",
        "maximum_sample_count": "Full template pixels times candidate count, before sampling.",
        "maximum_working_bytes": "Estimated arrays; excludes native workspaces and runtime copies.",
        "maximum_graphics_candidates": "Limit diagnostic outlines/labels; zero omits drawings.",
    }
    input_help = {
        "image": "Current inspection image; unchanged on output.",
        "template": "Expected template on the candidate transform's original source grid.",
        "matches": "Template-instance list or pattern_matches_v1; original scores are retained.",
        "mask": "Optional observed source-image support, intersected with Region.",
        "region": "Optional source-image support; excluded samples still count in the denominator.",
        "template_mask": "Optional uint8 expected-object selection; excludes template background.",
        "template_region": "Optional template-grid region, intersected with Template mask.",
    }
    output_help = {
        "matches": "Accepted annotated records, preserving list or pattern_matches_v1 format.",
        "candidates": "All records with support fractions, counts, validity and rejection reasons.",
        "report": "Template appearance evidence only; not true occlusion or grasp safety proof.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = self.description or "Check expected appearance before choosing a pick."
        self.documentation = self.documentation or (
            "<p>Sample the current scene on each candidate's template grid. Report absolute "
            "appearance agreement separately from detector score and real-source support. "
            "Lighting, registration and lookalike objects can confound this visibility proxy. "
            "See docs/visible-support.md for transforms, masks, units and limitations.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limit in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or (limit.whole and type(p[name]) is not int):
                raise ValidationError("Use typed numeric support parameters", name)
        origin = p["template_origin"]
        if not isinstance(origin, list) or (
            origin
            and (
                len(origin) != 2
                or any(type(v) not in (int, float) for v in origin)
                or not np.isfinite(origin).all()
                or np.max(np.abs(origin)) > 1_000_000
            )
        ):
            raise ValidationError(
                "Template origin must be empty or finite [x,y]", "template_origin"
            )
        if p["minimum_matches"] > p["maximum_candidates"]:
            raise ValidationError("Minimum matches exceeds Maximum candidates", "minimum_matches")

    def candidate_matrix(self, match, image, template):
        if not isinstance(match, dict):
            raise ValidationError("Candidates must be match records")
        p, size = self.parameters, list(template.shape[1::-1])
        mode = p["transform_mode"]
        if mode == "auto":
            mode = (
                "transform" if "transform" in match else "bounds" if "bounds" in match else "pose"
            )
        image_context = self.input_payloads["image"]
        if mode == "transform":
            transform = match.get("transform")
            if (
                not isinstance(transform, dict)
                or transform.get("model") != "image_pattern_pose_v1"
                or transform.get("source_size") != size
            ):
                raise ValidationError("Candidate transform must match the full template grid")
            source_context = transform.get("source_payload", {})
            if p["template_grid_policy"] == "matching_frames":
                require_same_frame(
                    (template, ValuePacket(template, source_context)),
                    message="Template and trained source grids differ",
                )
            same_ticket(transform.get("target_payload", {}), image_context)
            pose = make_pose(
                transform.get("source_to_target"),
                transform.get("target_size"),
                transform.get("target_payload", {}),
            )
        elif mode == "bounds":
            bounds = np.asarray(match.get("bounds"), np.float64)
            if bounds.shape != (4,) or not np.isfinite(bounds).all() or list(bounds[2:]) != size:
                raise ValidationError("Bounds must describe the unscaled template dimensions")
            top_left = np.asarray(match.get("subpixel_top_left", bounds[:2]), np.float64)
            if top_left.shape != (2,) or not np.isfinite(top_left).all():
                raise ValidationError("Subpixel top-left must be finite [x,y]")
            target = (
                validate_pose(match["pose"])
                if "pose" in match
                else {
                    "target_size": list(image.shape[1::-1]),
                    "target_payload": self.input_payloads["matches"],
                }
            )
            same_ticket(target["target_payload"], image_context)
            pose = make_pose(
                local_offset(*top_left, 0), target["target_size"], target["target_payload"]
            )
        else:
            pose = validate_pose(match.get("pose"))
            same_ticket(pose["target_payload"], image_context)
            origin = p["template_origin"] or [(size[0] - 1) / 2, (size[1] - 1) / 2]
            pose = make_pose(
                np.asarray(pose["source_to_target"]) @ local_offset(-origin[0], -origin[1], 0),
                pose["target_size"],
                pose["target_payload"],
            )
        return region_matrix(pose_for_image(pose, image)["source_to_target"]), mode

    def process(self, inputs):
        p, image, template = self.parameters, inputs["image"], inputs["template"]
        records = inputs["matches"]
        wrapped = isinstance(records, dict) and records.get("model") == "pattern_matches_v1"
        items = records.get("items") if wrapped else records
        if (
            not isinstance(items, list)
            or len(items) > p["maximum_candidates"]
            or (
                wrapped
                and (type(records.get("count")) is not int or records["count"] != len(items))
            )
        ):
            raise ValidationError("Use a bounded candidate list or consistent pattern_matches_v1")
        same_ticket(self.input_payloads["matches"], self.input_payloads["image"])
        source_pixels, template_pixels = np.prod(image.shape[:2]), np.prod(template.shape[:2])
        if (
            min(*image.shape[:2], *template.shape[:2]) < 1
            or max(*image.shape[:2], *template.shape[:2]) > 16384
            or source_pixels > p["maximum_source_pixels"]
            or template_pixels > p["maximum_template_pixels"]
        ):
            raise ValidationError("Source/template exceeds pixel or axis limits")
        if template_pixels * len(items) > p["maximum_sample_count"]:
            raise ValidationError(
                "Candidate template grids exceed Maximum sample count", "maximum_sample_count"
            )
        estimate = int(source_pixels * 40 + template_pixels * 224)
        if estimate > p["maximum_working_bytes"]:
            raise ValidationError(
                "Support arrays exceed Maximum working bytes", "maximum_working_bytes"
            )
        source, expected = (
            comparison_pixels(inputs[name], p["comparison"]) for name in ("image", "template")
        )
        if source.ndim != expected.ndim or (
            source.ndim == 3 and source.shape[2] != expected.shape[2]
        ):
            raise ValidationError("All channels comparison requires matching channel layouts")
        source_selection = selection_mask(inputs, image)
        source_selection = (
            np.ones(image.shape[:2], np.uint8)
            if source_selection is None
            else (source_selection != 0).astype(np.uint8)
        )
        expected_selection = selection_mask(
            {key: inputs.get("template_" + key) for key in ("mask", "region")}, template
        )
        expected_selection = (
            np.ones(template.shape[:2], bool)
            if expected_selection is None
            else expected_selection != 0
        )
        denominator = int(expected_selection.sum())
        if not denominator:
            raise ValidationError("Select at least one expected template pixel")
        yy, xx = np.indices(template.shape[:2], np.float64)
        real_source = np.ones(image.shape[:2], np.uint8)
        local = np.array(
            [
                [0, 0],
                [template.shape[1] - 1, 0],
                [template.shape[1] - 1, template.shape[0] - 1],
                [0, template.shape[0] - 1],
            ]
        )
        candidates, accepted, objects = [], [], []
        for index, match in enumerate(items):
            matrix, mode = self.candidate_matrix(match, image, template)
            if np.any(np.c_[local, np.ones(4)] @ matrix[2] <= 1e-9):
                raise ValidationError("Template projection crosses the perspective horizon")
            div = matrix[2, 0] * xx + matrix[2, 1] * yy + matrix[2, 2]
            x = ((matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]) / div).astype(np.float32)
            y = ((matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]) / div).astype(np.float32)
            if (
                not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or max(np.max(abs(x)), np.max(abs(y))) > 1_000_000
            ):
                raise ValidationError("Projected sample coordinates exceed finite bounds")
            if p["interpolation"] == "linear":
                # Float64 native remap uses 1/32 phases on the qualified build.
                # Make the sampling grid explicit for both appearance and support.
                x, y = np.rint(x * 32) / 32, np.rint(y * 32) / 32
            real = sample_mask(real_source, x, y, p["interpolation"])
            observed = (
                sample_mask(source_selection, x, y, p["interpolation"]) & real & expected_selection
            )
            sampled = cv2.remap(
                source,
                x,
                y,
                cv2.INTER_LINEAR if p["interpolation"] == "linear" else cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(sampled - expected)
            error = difference.max(axis=2) if difference.ndim == 3 else difference
            if not np.isfinite(error).all():
                raise ValidationError("Image values exceed finite comparison arithmetic")
            supported = observed & (error <= p["absolute_tolerance"])
            n_observed, n_supported = int(observed.sum()), int(supported.sum())
            fraction, observation = n_supported / denominator, n_observed / denominator
            reasons = []
            if not n_observed:
                reasons.append("no_observed_support")
            if observation < p["minimum_observed_fraction"]:
                reasons.append("insufficient_observed_support")
            if fraction < p["minimum_visible_fraction"]:
                reasons.append("appearance_support_below_minimum")
            evidence = {
                "model": "visible_support_v1",
                "measure": "template_appearance_agreement",
                "index": index,
                "transform_mode": mode,
                "template_to_image": matrix.tolist(),
                "expected_samples": denominator,
                "in_frame_samples": int((real & expected_selection).sum()),
                "observed_samples": n_observed,
                "matching_samples": n_supported,
                "visible_fraction": fraction,
                "observed_fraction": observation,
                "observed_agreement_fraction": n_supported / n_observed if n_observed else None,
                "mean_observed_error": float((error[observed] / n_observed).sum())
                if n_observed
                else None,
                "accepted": not reasons,
                "reasons": reasons,
            }
            annotated = {**copy.deepcopy(match), "support": evidence}
            candidates.append(annotated)
            if not reasons:
                accepted.append(annotated)
            if self.generate_graphics and index < p["maximum_graphics_candidates"]:
                corners = cv2.perspectiveTransform(local.astype(np.float64)[None], matrix)[0]
                color = REJECTED if reasons else ACCEPTED
                objects.extend(
                    (
                        graphic_object("polyline", corners, color, closed=True),
                        graphic_object(
                            "text", [corners[0]], color, text=f"{index}: {fraction:.1%}"
                        ),
                    )
                )
        output_matches = (
            {**copy.deepcopy(records), "items": accepted, "count": len(accepted)}
            if wrapped
            else accepted
        )
        passed = len(accepted) >= p["minimum_matches"]
        return Result(
            {
                "image": image,
                "matches": output_matches,
                "candidates": candidates,
                "count": len(accepted),
                "report": {
                    "model": "visible_support_report_v1",
                    "measure": "template_appearance_agreement",
                    "candidate_count": len(items),
                    "accepted_count": len(accepted),
                    "expected_samples": denominator,
                    "interpolation": p["interpolation"],
                    "linear_phase_steps": 32 if p["interpolation"] == "linear" else None,
                    "comparison": p["comparison"],
                    "template_grid_policy": p["template_grid_policy"],
                    "image_size": list(image.shape[1::-1]),
                    "template_size": list(template.shape[1::-1]),
                    "minimum_visible_fraction": p["minimum_visible_fraction"],
                    "minimum_observed_fraction": p["minimum_observed_fraction"],
                    "minimum_matches": p["minimum_matches"],
                    "absolute_tolerance": p["absolute_tolerance"],
                    "estimated_working_bytes": estimate,
                    "true_occlusion_measurement": False,
                },
            },
            passed=passed,
            score=max((m["support"]["visible_fraction"] for m in accepted), default=0.0),
            fail_reasons=[] if passed else ["visible_support.too_few_candidates"],
            drawings={"Candidate visible support": GraphicDrawing(tuple(objects))}
            if objects
            else {},
            message=f"{len(accepted)} of {len(items)} candidates have sufficient support",
        )
