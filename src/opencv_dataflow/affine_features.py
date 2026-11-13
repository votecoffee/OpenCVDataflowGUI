"""Bounded affine-view features with source-space geometry and reusable descriptors."""

import math

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, draw_object, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import ImageTool, gray_image
from .registration import ORBFeatures, SIFTFeatures
from .selection import SELECTION_INPUTS, contains, selection_mask

MAX_VIEWS = 64
MAX_SAMPLED_PIXELS = 64_000_000


def affine_views(p):
    """Resolve a finite schedule before native allocation; tilt indices are exponents."""
    if p["view_mode"] == "custom":
        raw = p["custom_views"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_VIEWS:
            raise ValidationError("Custom views require 1..64 [tilt, roll_degrees] pairs")
        views = []
        for pair in raw:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValidationError("Each custom view must be [tilt, roll_degrees]")
            for value in pair:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValidationError("View tilt and roll must be finite numbers")
            tilt, roll = map(float, pair)
            if not math.isfinite(tilt) or not math.isfinite(roll):
                raise ValidationError("View tilt and roll must be finite numbers")
            if not 1 <= tilt <= 8 or not -180 <= roll <= 180:
                raise ValidationError("View tilt must be 1..8 and roll -180..180 degrees")
            views.append((tilt, roll))
        if len(set(views)) != len(views):
            raise ValidationError("Custom views must not contain duplicate pairs")
        return views
    if p["min_tilt"] > p["max_tilt"]:
        raise ValidationError("Minimum tilt index must not exceed maximum", "min_tilt")
    views = []
    for index in range(int(p["min_tilt"]), int(p["max_tilt"]) + 1):
        tilt = float(p["tilt_step"]) ** index
        if tilt > 8 + 1e-6:
            raise ValidationError("Generated tilt factor exceeds eight; reduce tilt index/step")
        step = float(p["rotate_step_base"]) / tilt
        count = 1 if index == 0 else math.ceil(180 / step - 1e-7)
        if len(views) + count > MAX_VIEWS:
            raise ValidationError("Sampling exceeds 64 views; increase roll step or reduce tilts")
        views.extend((tilt, step * j) for j in range(count))
    return views


def source_keypoint(key, views):
    """OpenCV maps centers back, but its native size/angle still belong to the view."""
    tilt, roll = views[key.class_id]
    radians = math.radians(roll)
    c, s = math.cos(radians), math.sin(radians)
    # The view's linear map is diag(1/tilt, 1) @ rotation(roll).
    inverse = np.array([[c * tilt, s], [-s * tilt, c]], np.float64)
    angle = key.angle
    orientation_radius = None
    if angle >= 0:
        direction = inverse @ [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 360
        orientation_radius = float(np.linalg.norm(direction)) * key.size / 2
    return {
        "point": list(key.pt),
        "size": key.size * math.sqrt(tilt),
        "angle": angle,
        "response": key.response,
        "octave": key.octave,
        "view_index": key.class_id,
        "view_size": key.size,
        "view_angle": key.angle,
        "orientation_radius": orientation_radius,
        "support_ellipse": {
            "major_diameter": key.size * tilt,
            "minor_diameter": key.size,
            "angle_degrees": (-roll) % 180,
        },
    }


class AffineFeatures(ImageTool):
    type_id, title = "inspect.affine_features", "Affine-invariant image features"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    process_inputs_readonly = True
    inputs = SELECTION_INPUTS
    outputs = (
        Port("image"),
        Port("features", "data"),
        Port("points", "data"),
        Port("count", "number"),
        Port("sampling", "data"),
        Port("overlay", required=False),
    )
    defaults = {
        "backend": "sift",
        "view_mode": "generated",
        "min_tilt": 0,
        "max_tilt": 2,
        "tilt_step": math.sqrt(2),
        "rotate_step_base": 72.0,
        "custom_views": [[1.0, 0.0], [2.0, 0.0], [2.0, 90.0]],
        "features_per_view": 400,
        "max_features": 4000,
        **{f"sift_{k}": v for k, v in SIFTFeatures.defaults.items() if k != "max_features"},
        **{f"orb_{k}": v for k, v in ORBFeatures.defaults.items() if k != "max_features"},
    }
    parameter_choices = {
        "backend": ("sift", "orb"),
        "view_mode": ("generated", "custom"),
        **{f"sift_{k}": v for k, v in SIFTFeatures.parameter_choices.items()},
        **{f"orb_{k}": v for k, v in ORBFeatures.parameter_choices.items()},
    }
    parameter_ranges = {
        "min_tilt": NumericRange(0, 6, whole=True),
        "max_tilt": NumericRange(0, 6, whole=True),
        "tilt_step": NumericRange(1.01, 2),
        "rotate_step_base": NumericRange(1, 180),
        "features_per_view": NumericRange(1, 2000, whole=True),
        "max_features": NumericRange(1, 10000, whole=True),
        **{f"sift_{k}": v for k, v in SIFTFeatures.parameter_ranges.items() if k != "max_features"},
        "orb_scale_factor": NumericRange(1.000001, 4),
        "orb_levels": NumericRange(1, 32, whole=True),
        "orb_fast_threshold": NumericRange(0, 255, whole=True),
        "orb_edge_threshold": NumericRange(0, 1024, whole=True),
        "orb_first_level": NumericRange(0, 31, whole=True),
        "orb_patch_size": NumericRange(3, 255, whole=True, odd=True),
    }
    parameter_help = {
        "backend": "SIFT outputs match existing SIFT matchers; ORB outputs match ORB matchers.",
        "view_mode": "Generate tilt/roll samples or use the explicit custom view list.",
        "min_tilt": "First exponent of tilt_step; zero includes the unwarped image once.",
        "max_tilt": "Last exponent of tilt_step, inclusive. Tilt factors are limited to eight.",
        "tilt_step": "Multiplicative tilt factor between generated levels; not an angle.",
        "rotate_step_base": "Generated roll spacing in degrees divided by each tilt factor.",
        "custom_views": "Explicit [tilt_factor, roll_degrees] pairs; 1..64 views, tilt 1..8.",
        "features_per_view": "Requested detector limit per sampled view, before combining results.",
        "max_features": "Keep strongest selected results; descriptor rows follow the same order.",
        **{f"sift_{k}": v for k, v in SIFTFeatures.parameter_help.items()},
        "sift_octave_layers": "SIFT sublevels per octave; used only with SIFT.",
        "sift_sigma": "Initial SIFT Gaussian sigma; used only with SIFT.",
        "sift_descriptor_depth": "SIFT descriptor storage; uint8 still uses L1/L2 matching.",
        "orb_scale_factor": "ORB pyramid scale ratio; used only with ORB.",
        "orb_levels": "ORB pyramid level count; first level must be smaller.",
        "orb_fast_threshold": "ORB FAST detector intensity threshold.",
        "orb_edge_threshold": "ORB image-border exclusion in sampled-view pixels.",
        "orb_first_level": "ORB pyramid level containing the input sampled view.",
        "orb_wta_k": "ORB descriptor comparisons; 3/4 require Hamming2 matching.",
        "orb_score_type": "ORB feature ranking method.",
        "orb_patch_size": "Odd ORB descriptor patch width in sampled-view pixels.",
    }
    input_help = {
        "image": "Original image passes through unchanged; detection uses an 8-bit gray copy.",
        "mask": "Optional uint8 mask intersected with region, in the input image's frame.",
        "region": "Optional circle, rectangle or polygon in the input image's frame.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        views = affine_views(p)
        if len(views) * int(p["features_per_view"]) > 64000:
            raise ValidationError("Requested features across views exceed 64,000")
        if p["backend"] == "orb" and p["orb_first_level"] >= p["orb_levels"]:
            raise ValidationError("ORB first level must be smaller than level count")

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        views = affine_views(p)
        h, w = source.shape[:2]
        # Bound rotation canvases too: compression does not avoid the rotated allocation.
        work = sum(
            math.ceil(
                w * abs(math.cos(math.radians(roll))) + h * abs(math.sin(math.radians(roll))) + 2
            )
            * math.ceil(
                w * abs(math.sin(math.radians(roll))) + h * abs(math.cos(math.radians(roll))) + 2
            )
            for _, roll in views
        )
        if w * h > 4_000_000 or work > MAX_SAMPLED_PIXELS:
            raise ValidationError(
                "Affine sampling exceeds the 4 MP input / 64 MP canvas-work limit"
            )
        gray = gray_image(source, (np.uint8,), auto_depth=True)
        mask = selection_mask(inputs, gray)
        count = int(p["features_per_view"])
        if p["backend"] == "sift":
            backend = cv2.SIFT_create(
                count,
                int(p["sift_octave_layers"]),
                float(p["sift_contrast_threshold"]),
                float(p["sift_edge_threshold"]),
                float(p["sift_sigma"]),
                cv2.CV_8U if p["sift_descriptor_depth"] == "uint8" else cv2.CV_32F,
                p["sift_precise_upscale"],
            )
            details = {"descriptor_depth": p["sift_descriptor_depth"]}
        else:
            backend = cv2.ORB_create(
                nfeatures=count,
                scaleFactor=float(p["orb_scale_factor"]),
                nlevels=int(p["orb_levels"]),
                edgeThreshold=int(p["orb_edge_threshold"]),
                firstLevel=int(p["orb_first_level"]),
                WTA_K=int(p["orb_wta_k"]),
                scoreType=cv2.ORB_HARRIS_SCORE
                if p["orb_score_type"] == "harris"
                else cv2.ORB_FAST_SCORE,
                patchSize=int(p["orb_patch_size"]),
                fastThreshold=int(p["orb_fast_threshold"]),
            )
            details = {"wta_k": int(p["orb_wta_k"])}
        detector = cv2.AffineFeature_create(backend, maxTilt=1)
        detector.setViewParams([v[0] for v in views], [v[1] for v in views])
        keys, descriptors = detector.detectAndCompute(gray, mask)
        keep = [
            i
            for i, key in enumerate(keys)
            if 0 <= key.pt[0] < w
            and 0 <= key.pt[1] < h
            and (mask is None or contains(mask, key.pt))
        ]
        keep.sort(key=lambda i: -keys[i].response)
        keep = keep[: int(p["max_features"])]
        records = [source_keypoint(keys[i], views) for i in keep]
        sampling = {
            "model": "affine_views_v1",
            "views": [{"tilt": t, "roll_degrees": r} for t, r in views],
            "view_count": len(views),
            "detected_count": len(keys),
            "selected_count": len(records),
            "canvas_work_pixels": work,
        }
        features = {
            "algorithm": p["backend"].upper(),
            **details,
            "descriptor_size": backend.descriptorSize(),
            "image_size": [w, h],
            "extractor": "AffineFeature",
            "sampling": sampling,
            "size_basis": "source-image equal-area diameter; exact ellipse stored per keypoint",
            "angle_basis": "source-image direction in degrees; -1 means unknown",
            "octave_basis": "backend octave in the sampled virtual view",
            "keypoints": records,
            "descriptors": descriptors[keep].tolist() if keep else [],
        }
        result = Result(
            {
                "image": source,
                "features": features,
                "points": [k["point"] for k in records],
                "count": len(records),
                "sampling": sampling,
                "overlay": NO_VALUE,
            }
        )
        if self.generate_graphics:
            objects = []
            circle = np.array(
                [
                    [math.cos(a), math.sin(a)]
                    for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)
                ]
            )
            for record in records:
                ellipse = record["support_ellipse"]
                angle = math.radians(ellipse["angle_degrees"])
                rotation = np.array(
                    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
                )
                radii = [ellipse["major_diameter"] / 2, ellipse["minor_diameter"] / 2]
                ring = (circle * radii) @ rotation.T + record["point"]
                objects.append(
                    graphic_object("polyline", ring, (70, 220, 140), closed=True, thickness=1)
                )
                if record["angle"] >= 0:
                    radians = math.radians(record["angle"])
                    end = np.array(record["point"]) + record["orientation_radius"] * np.array(
                        [math.cos(radians), math.sin(radians)]
                    )
                    objects.append(
                        graphic_object(
                            "polyline", [record["point"], end], (230, 180, 60), thickness=1
                        )
                    )
            result.drawings["Affine features"] = GraphicDrawing(tuple(objects))
            preview = overlay_image(source)
            for obj in objects:
                draw_object(preview, obj, np.eye(3))
            result.outputs["overlay"] = preview
            result.graphics["Affine features"] = preview
        return result
