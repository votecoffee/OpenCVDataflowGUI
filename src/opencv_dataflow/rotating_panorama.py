"""Unordered rotating-camera registration with independent geometric checks."""

import copy
import math
from itertools import combinations

import cv2
import numpy as np

from .analysis_tools import cast_intensity
from .core import Port, Result
from .flow_warp import sample_mask
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .panorama_fitting import CHOICES as FIT_CHOICES
from .panorama_fitting import DEFAULTS as FIT_DEFAULTS
from .panorama_fitting import HELP as FIT_HELP
from .panorama_fitting import RANGES as FIT_RANGES
from .panorama_fitting import explicit_fit, validate_settings
from .panorama_geometry import (
    PROJECTIONS,
    camera_homography,
    intrinsics,
    layout_frame,
    mapped_error,
    project_points,
    rotation,
    source_boundary,
)
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, frame, require_same_frame
from .processing import ImageTool
from .scan_stitching import _features, _pairs, _unpack
from .selection import selection_mask

INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
STATUS = {
    0: "ok",
    1: "need_more_images",
    2: "homography_estimation_failed",
    3: "camera_adjustment_failed",
}
WAVE = {
    "horizontal": cv2.detail.WAVE_CORRECT_HORIZ,
    "vertical": cv2.detail.WAVE_CORRECT_VERT,
    "auto": cv2.detail.WAVE_CORRECT_AUTO,
}


def connected(indices, edges, anchor):
    indices = set(indices)
    found = {anchor}
    while True:
        before = len(found)
        for a, b in edges:
            if a not in indices or b not in indices:
                continue
            if a in found or b in found:
                found.update((a, b))
        if len(found) == before:
            return sorted(found.intersection(indices))


def verification(cameras, features, scales, p):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    records, edges, inconsistent = [], [], False
    for a, b in combinations(cameras, 2):
        i, j = a["index"], b["index"]
        x, y = _pairs(
            matcher, features[i], features[j], p["match_ratio"], p["maximum_descriptor_distance"]
        )
        x = x / np.asarray(scales[i])
        y = y / np.asarray(scales[j])
        row = {
            "source_index": i,
            "target_index": j,
            "matches": len(x),
            "status": "insufficient_matches",
        }
        records.append(row)
        if len(x) < p["minimum_inliers"]:
            continue
        H, mask = cv2.findHomography(
            x,
            y,
            cv2.RANSAC,
            p["inlier_pixels"],
            maxIters=p["ransac_iterations"],
            confidence=p["ransac_confidence"],
        )
        if H is None or mask is None:
            row["status"] = "no_homography"
            continue
        valid = mask.ravel() != 0
        n = int(valid.sum())
        ratio = n / len(x)
        row.update(inliers=n, inlier_ratio=ratio)
        if n < p["minimum_inliers"] or ratio < p["minimum_inlier_ratio"]:
            row["status"] = "insufficient_inliers"
            continue
        local = mapped_error(H, x[valid], y[valid])
        native = camera_homography(a, b)
        residual = max(
            mapped_error(native, x[valid], y[valid]),
            mapped_error(np.linalg.inv(native), y[valid], x[valid]),
        )
        row.update(
            local_rms_pixels=local if np.isfinite(local) else None,
            camera_rms_pixels=residual if np.isfinite(residual) else None,
        )
        if not np.isfinite(local) or local > p["inlier_pixels"]:
            row["status"] = "local_fit_error"
            continue
        if not np.isfinite(residual) or residual > p["maximum_camera_error"]:
            row["status"] = "inconsistent_rotation_model"
            inconsistent = True
            continue
        row["status"] = "validated"
        edges.append((i, j))
    return records, edges, inconsistent


def diagnostic(labels, report):
    image = np.full((320, 640, 3), 25, np.uint8)
    n = len(labels)
    centers = [
        (
            round(320 + 235 * math.cos(i * 2 * np.pi / n)),
            round(170 + 105 * math.sin(i * 2 * np.pi / n)),
        )
        for i in range(n)
    ]
    for row in report.get("pairs", []):
        color = (70, 180, 90) if row["status"] == "validated" else (65, 70, 150)
        cv2.line(image, centers[row["source_index"]], centers[row["target_index"]], color, 1)
    for i, (label, point) in enumerate(zip(labels, centers, strict=True)):
        color = (70, 220, 140) if i in report.get("retained_indices", []) else (60, 90, 240)
        cv2.circle(image, point, 14, color, 2)
        cv2.putText(
            image,
            f"{i}: {label[:22]}",
            (max(1, point[0] - 70), point[1] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )
    cv2.putText(image, report["status"], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return image


class StitchRotatingPanorama(ImageTool):
    type_id, title, subtype = (
        "cv.stitch_rotating_panorama",
        "Stitch rotating-camera views",
        "Panoramas",
    )
    inputs = (
        Port("views", "any"),
        Port("masks", "any", required=False, required_when_connected=True),
        Port("regions", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("coverage", required=False, visual_type="region"),
        Port("contributors", required=False),
        Port("layout", "data", required=False),
        Port("report", "data"),
        Port("diagnostic", required=False),
    )
    defaults = {
        **FIT_DEFAULTS,
        "reference_index": 0,
        "anchor_label": "",
        "source_labels": [],
        "component_policy": "reject",
        "projection": "spherical",
        "projection_scale": 0.0,
        "wave_correction": "none",
        "registration_megapixels": 0.2,
        "native_confidence": 0.5,
        "verification_features": 1000,
        "maximum_verification_pairs": 32_000_000,
        "match_ratio": 0.75,
        "maximum_descriptor_distance": 400.0,
        "minimum_inliers": 12,
        "minimum_inlier_ratio": 0.3,
        "inlier_pixels": 3.0,
        "maximum_camera_error": 3.0,
        "ransac_iterations": 2000,
        "ransac_confidence": 0.995,
        "minimum_focal_ratio": 0.2,
        "maximum_focal_ratio": 5.0,
        "interpolation": "linear",
        "blend": "average",
        "minimum_covered_fraction": 0.0,
        "maximum_input_pixels": 16_777_216,
        "maximum_output_pixels": 8_388_608,
        "maximum_working_bytes": 512 * 1024**2,
    }
    parameter_choices = {
        **FIT_CHOICES,
        "component_policy": ("reject", "fail_partial", "allow_partial"),
        "projection": PROJECTIONS,
        "wave_correction": ("none", *WAVE),
        "interpolation": tuple(INTERPOLATION),
        "blend": ("average", "first"),
    }
    parameter_ranges = {
        **FIT_RANGES,
        "reference_index": NumericRange(0, 7, whole=True),
        "projection_scale": NumericRange(0, 16384),
        "registration_megapixels": NumericRange(0.01, 4),
        "native_confidence": NumericRange(0.01, 5),
        "verification_features": NumericRange(32, 4000, whole=True),
        "maximum_verification_pairs": NumericRange(1, 256_000_000, whole=True),
        "match_ratio": NumericRange(0.01, 0.99),
        "maximum_descriptor_distance": NumericRange(1, 1000),
        "minimum_inliers": NumericRange(4, 1000, whole=True),
        "minimum_inlier_ratio": NumericRange(0.01, 1),
        "inlier_pixels": NumericRange(0.1, 20),
        "maximum_camera_error": NumericRange(0.1, 30),
        "ransac_iterations": NumericRange(1, 10000, whole=True),
        "ransac_confidence": NumericRange(0.01, 0.999999),
        "minimum_focal_ratio": NumericRange(0.05, 20),
        "maximum_focal_ratio": NumericRange(0.1, 20),
        "minimum_covered_fraction": NumericRange(0, 1),
        "maximum_input_pixels": NumericRange(64, 67_108_864, whole=True),
        "maximum_output_pixels": NumericRange(1, 64_000_000, whole=True),
        "maximum_working_bytes": NumericRange(1, 2 * 1024**3, whole=True),
    }
    parameter_help = {
        **FIT_HELP,
        "reference_index": "Anchor index when Anchor label is empty; not the assumed view order.",
        "anchor_label": "Stable source label; must survive component selection.",
        "source_labels": "Unique labels in batch order; empty uses labels or view indices.",
        "component_policy": "Reject missing views, fail a partial output, or allow partial output.",
        "projection": "Spherical, cylindrical or plane projection on a new image grid.",
        "projection_scale": (
            "Projection pixels per radian; zero uses median estimated focal length."
        ),
        "wave_correction": "Optional native global horizon correction after anchoring rotations.",
        "registration_megapixels": (
            "Work-image megapixels per source; render original-depth pixels."
        ),
        "native_confidence": "Native Stitcher panorama-component confidence threshold.",
        "verification_features": "Automatic solver: bound separate SIFT validation keypoints.",
        "maximum_verification_pairs": "Fit and validation each obey this descriptor-product cap.",
        "match_ratio": "Independent mutual descriptor nearest/second-nearest distance ratio.",
        "maximum_descriptor_distance": "Maximum independent SIFT match distance.",
        "minimum_inliers": "Minimum independent homography inliers for a validated overlap edge.",
        "minimum_inlier_ratio": "Minimum inlier fraction after mutual descriptor matching.",
        "inlier_pixels": "Independent homography RANSAC threshold, in original source pixels.",
        "maximum_camera_error": (
            "Maximum bidirectional camera-model RMS, in original source pixels."
        ),
        "ransac_iterations": "Independent validation homography iteration limit.",
        "ransac_confidence": "Independent validation homography confidence.",
        "minimum_focal_ratio": "Minimum estimated focal / longest source side.",
        "maximum_focal_ratio": "Maximum estimated focal / longest source side.",
        "interpolation": "Image sampling; cubic/Lanczos use conservative support halos.",
        "blend": "Average supported samples or prefer the first retained input.",
        "minimum_covered_fraction": "Minimum covered fraction of the rectangular panorama canvas.",
        "maximum_input_pixels": "Total source pixel cap before registration preparation.",
        "maximum_output_pixels": "Canvas and total local warp-map pixel cap before rendering.",
        "maximum_working_bytes": (
            "Estimated arrays, excluding runtime copies and native workspaces."
        ),
    }
    input_help = {
        "views": "Unordered 2–8 gray/RGB/BGR images; repeated labels use view indices.",
        "masks": "Optional uint8 mask batch, matching source labels/order and source frames.",
        "regions": "Optional region batch, intersected with source masks on each source grid.",
    }
    output_help = {
        "image": "Depth/encoding-preserving panorama; zero outside Coverage.",
        "coverage": (
            "Binary supported pixels, excluding padding and conservative kernel boundaries."
        ),
        "layout": "Pinhole cameras, rotations, projection, provenance and output grid.",
        "report": "Native status, source selection and independent camera residuals.",
        "diagnostic": "Connectivity chart on its own grid; omitted when graphics are off.",
    }
    preview_safe = parallel_safe = process_inputs_readonly = True
    payload_reference = "views"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "coverage", "layout", "report"]
        self.description = self.description or "Register unordered views from a rotating camera."
        self.documentation = self.documentation or (
            "<p>Native panorama camera estimation plus independent overlap/rotation-model checks. "
            "Masks restrict registration and supported rendering. Source depth is retained; "
            "panorama coordinates are a new nonlinear grid. See docs/rotating-panorama.md. "
            "Rotation/static-scene assumptions require camera qualification.</p>"
        )

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        validate_settings(p)
        for key, bounds in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or (bounds.whole and type(p[key]) is not int):
                raise ValidationError("Use typed numeric panorama parameters", key)
        if not isinstance(p["anchor_label"], str) or len(p["anchor_label"]) > 1024:
            raise ValidationError("Anchor label must be text up to 1024 characters", "anchor_label")
        names = p["source_labels"]
        if (
            not isinstance(names, list)
            or len(names) > 8
            or any(not isinstance(x, str) or not 1 <= len(x) <= 1024 for x in names)
            or len(set(names)) != len(names)
        ):
            raise ValidationError(
                "Source labels must be an empty or unique text list", "source_labels"
            )
        if p["minimum_focal_ratio"] >= p["maximum_focal_ratio"]:
            raise ValidationError(
                "Focal limits must increase", "minimum_focal_ratio", "maximum_focal_ratio"
            )
        limit = "verification_features" if p["camera_solver"] == "automatic" else "camera_features"
        if p["minimum_inliers"] > p[limit]:
            raise ValidationError(
                "Minimum inliers exceeds available feature limit", "minimum_inliers", limit
            )

    def finish(self, report, layout=None, image=None, coverage=None, counts=None):
        chart = diagnostic(report["labels"], report) if self.generate_graphics else NO_VALUE
        context = copy.deepcopy(self.input_payloads["views"])
        context.pop("pyramid", None)
        payloads = {"report": copy.deepcopy(context)}
        outputs = {
            "image": NO_VALUE,
            "coverage": NO_VALUE,
            "contributors": NO_VALUE,
            "layout": layout if layout is not None else NO_VALUE,
            "report": report,
            "diagnostic": chart,
        }
        drawings = {}
        if image is not None:
            context["spatial"] = layout["spatial"]
            context["panorama"] = {
                "model": layout["model"],
                "projection": layout["projection"],
                "anchor_label": layout["anchor_label"],
                "requires_coverage": True,
            }
            outputs.update(image=image, coverage=coverage, contributors=counts)
            payloads.update(
                {
                    key: copy.deepcopy(context)
                    for key in ("image", "coverage", "contributors", "layout", "report")
                }
            )
            if self.generate_graphics:
                objects = []
                for camera in layout["cameras"]:
                    points, valid = project_points(
                        source_boundary(camera["source_size"]),
                        camera,
                        layout["projection"],
                        layout["scale"],
                        layout["origin"],
                    )
                    if valid.all():
                        objects.extend(
                            (
                                graphic_object("polyline", points, (70, 220, 140), closed=True),
                                graphic_object(
                                    "text", [points[0]], (230, 220, 70), text=camera["label"][:40]
                                ),
                            )
                        )
                drawings["Panorama source fields"] = GraphicDrawing(
                    tuple(objects), "image", from_output=True
                )
        if chart is not NO_VALUE:
            payloads["diagnostic"] = copy.deepcopy(self.input_payloads["views"])
            payloads["diagnostic"]["spatial"] = frame([640, 320], f"panorama-diagnostic:{self.id}")
            payloads["diagnostic"].pop("pyramid", None)
        passed = report["status"] == "ok"
        return Result(
            outputs,
            payloads=payloads,
            passed=passed,
            score=report.get("coverage", 0.0),
            fail_reasons=[] if passed else ["panorama." + report["status"]],
            drawings=drawings,
            message=report["status"],
        )

    def process(self, inputs):
        p = self.parameters
        group = inputs["views"]
        if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 8:
            raise ValidationError("Use a batch of 2–8 panorama views")
        raw = [entry.value for entry in group.items]
        if any(not isinstance(value, np.ndarray) or value.ndim not in (2, 3) for value in raw):
            raise ValidationError("Every panorama view must be an image")
        total = sum(value.shape[0] * value.shape[1] for value in raw)
        if total > p["maximum_input_pixels"] or total * 40 > p["maximum_working_bytes"]:
            raise ValidationError(
                "Panorama sources exceed pixel/working limits",
                "maximum_input_pixels",
                "maximum_working_bytes",
            )
        images, pixels, masks, contexts = _unpack(group, inputs.get("masks"))
        input_labels = list(group.labels)
        labels = p["source_labels"] or (
            input_labels
            if len(set(input_labels)) == len(input_labels)
            else [f"view_{i}" for i in range(len(images))]
        )
        n = len(images)
        if (
            len(labels) != n
            or len(set(labels)) != n
            or any(not isinstance(x, str) or not x or len(x) > 1024 for x in labels)
        ):
            raise ValidationError("Panorama source labels must match the view count")
        if inputs.get("masks") is not None and inputs["masks"].labels != group.labels:
            raise ValidationError("Mask labels/order must match panorama views")
        regions = inputs.get("regions")
        if regions is not None:
            if (
                not isinstance(regions, FlowGroup)
                or regions.labels != group.labels
                or len(regions.items) != n
            ):
                raise ValidationError("Region batch labels/order must match panorama views")
            for i, packet in enumerate(regions.items):
                require_same_frame(
                    (group.items[i], packet), message="Panorama source/region frames differ"
                )
                region = selection_mask({"region": packet.value}, images[i])
                masks[i] = region if masks[i] is None else cv2.bitwise_and(masks[i], region)
        if any(image_format(x) not in ("gray", "rgb", "bgr") for x in images):
            raise ValidationError("Use gray, RGB or BGR panorama sources")
        if pixels[0].dtype == np.float64 and p["interpolation"] == "cubic":
            raise ValidationError("Float64 cubic remap is unreliable; use linear or float32")
        anchor = (
            labels.index(p["anchor_label"])
            if p["anchor_label"] in labels
            else p["reference_index"]
            if not p["anchor_label"]
            else -1
        )
        if not 0 <= anchor < n:
            raise ValidationError(
                "Anchor is not in the input batch", "anchor_label", "reference_index"
            )
        masks = [
            np.full(x.shape[:2], 255, np.uint8) if m is None else m
            for x, m in zip(images, masks, strict=True)
        ]
        grays = []
        for image, values in zip(images, pixels, strict=True):
            values = values.astype(np.float64)
            if values.ndim == 3:
                if image_format(image) == "rgb":
                    values = values[..., ::-1]
                values = values @ np.array([0.114, 0.587, 0.299])
            grays.append(values)
        selected = [gray[mask != 0] for gray, mask in zip(grays, masks, strict=True)]
        active = [
            i
            for i, values in enumerate(selected)
            if len(values) and float(values.min()) < float(values.max())
        ]
        report = {
            "model": "rotation_panorama_report_v1",
            "status": "insufficient_texture",
            "labels": labels,
            "input_labels": input_labels,
            "native_status": None,
            "retained_indices": [],
            "dropped_labels": labels.copy(),
            "anchor_label": labels[anchor],
            "pairs": [],
            "component_policy": p["component_policy"],
            "assumption": "Static pinhole views; rotation about one optical center",
        }
        if len(active) < 2 or anchor not in active:
            return self.finish(report)
        low = min(float(selected[i].min()) for i in active)
        high = max(float(selected[i].max()) for i in active)
        center, half_span = low / 2 + high / 2, high / 2 - low / 2
        report["registration_luminance_range"] = [low, high]
        factor = min(
            1.0, math.sqrt(p["registration_megapixels"] * 1e6 / max(x.size for x in grays))
        )
        work, work_masks, scales = [], [], []
        for gray, mask in zip(grays, masks, strict=True):
            size = (max(8, round(gray.shape[1] * factor)), max(8, round(gray.shape[0] * factor)))
            converted = np.rint(
                np.clip((gray - center) / half_span * 127.5 + 127.5, 0, 255)
            ).astype(np.uint8)
            work.append(
                cv2.resize(converted, size, interpolation=cv2.INTER_AREA)
                if size != gray.shape[::-1]
                else converted
            )
            work_masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
            scales.append([size[0] / gray.shape[1], size[1] / gray.shape[0]])
        validation_features = None
        if p["camera_solver"] == "explicit":
            status, component, native, validation_features, audit = explicit_fit(
                work, work_masks, scales, active, anchor, p
            )
            report.update(native_status=status, camera_fit=audit)
            if status != "ok":
                report["status"] = status
                return self.finish(report)
        else:
            solver = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            solver.setRegistrationResol(-1)
            solver.setPanoConfidenceThresh(p["native_confidence"])
            solver.setWaveCorrection(False)
            status = int(
                solver.estimateTransform([work[i] for i in active], [work_masks[i] for i in active])
            )
            report["native_status"] = STATUS.get(status, "unknown_native_status")
            report["camera_fit"] = {
                "solver": "automatic",
                "validation_basis": "separate_detection",
                "validation_excluded_from_camera_fit": False,
            }
            if status != cv2.Stitcher_OK:
                report["status"] = report["native_status"]
                return self.finish(report)
            indices = list(map(int, solver.component()))
            if len(set(indices)) != len(indices) or any(not 0 <= i < len(active) for i in indices):
                raise ValidationError("Native component indices are invalid")
            component = [active[i] for i in indices]
            native = list(solver.cameras())
        if (
            len(component) != len(native)
            or len(set(component)) != len(component)
            or len(native) < 2
        ):
            raise ValidationError("Native panorama returned inconsistent cameras/components")
        report["native_component_indices"] = component
        if anchor not in component:
            report["status"] = "anchor_disconnected"
            return self.finish(report)
        reference = rotation(native[component.index(anchor)].R)
        cameras = []
        for i, camera in zip(component, native, strict=True):
            K = intrinsics(np.diag([1 / scales[i][0], 1 / scales[i][1], 1]) @ camera.K())
            R = rotation(reference.T @ rotation(camera.R))
            focal = math.sqrt(K[0, 0] * K[1, 1]) / max(pixels[i].shape[:2])
            if not p["minimum_focal_ratio"] <= focal <= p["maximum_focal_ratio"]:
                report["status"] = "focal_out_of_range"
                report.update(rejected_camera_index=i, focal_ratio=focal)
                return self.finish(report)
            cameras.append(
                {
                    "index": i,
                    "label": labels[i],
                    "input_label": input_labels[i],
                    "source_size": list(pixels[i].shape[1::-1]),
                    "source_spatial": contexts[i].get("spatial"),
                    "source_ticket": contexts[i].get("ticket"),
                    "K": K.tolist(),
                    "camera_to_anchor": R.tolist(),
                    "camera_to_panorama": R.tolist(),
                    "registration_scale": scales[i],
                    "selected_pixels": int(np.count_nonzero(masks[i])),
                }
            )
        if p["wave_correction"] != "none":
            corrected = cv2.detail.waveCorrect(
                [np.asarray(c["camera_to_anchor"], np.float32) for c in cameras],
                WAVE[p["wave_correction"]],
            )
            for camera, R in zip(cameras, corrected, strict=True):
                camera["camera_to_panorama"] = rotation(R).tolist()
        report["camera_estimates"] = copy.deepcopy(cameras)
        if validation_features is None:
            detector = cv2.SIFT_create(p["verification_features"])
            features = {
                i: _features(detector, work[i], work_masks[i], p["verification_features"])
                for i in component
            }
        else:
            features = {i: validation_features[i] for i in component}
        comparisons = sum(
            len(features[a][0]) * len(features[b][0]) for a, b in combinations(component, 2)
        )
        if comparisons > p["maximum_verification_pairs"]:
            raise ValidationError(
                "Descriptor verification budget exceeded", "maximum_verification_pairs"
            )
        report["verification_descriptor_pairs"] = comparisons
        pairs, edges, inconsistent = verification(cameras, features, scales, p)
        report["pairs"] = pairs
        if inconsistent:
            report["status"] = "inconsistent_rotation_model"
            return self.finish(report)
        retained = connected(component, edges, anchor)
        report["retained_indices"] = retained
        report["dropped_labels"] = [label for i, label in enumerate(labels) if i not in retained]
        if len(retained) < 2 or (len(retained) != n and p["component_policy"] == "reject"):
            report["status"] = "disconnected_views"
            return self.finish(report)
        cameras = sorted((c for c in cameras if c["index"] in retained), key=lambda c: c["index"])
        scale = p["projection_scale"] or float(
            np.median([intrinsics(c["K"])[0, 0] for c in cameras])
        )
        if not 1 <= scale <= 16384:
            raise ValidationError("Set a projection scale between 1 and 16384", "projection_scale")
        warper = cv2.PyRotationWarper(p["projection"], scale)
        rois = []
        for camera in cameras:
            K = intrinsics(camera["K"])
            R = rotation(camera["camera_to_panorama"])
            if p["projection"] == "plane":
                corners = source_boundary(camera["source_size"], 2)
                if not project_points(corners, camera, "plane", scale)[1].all():
                    report["status"] = "plane_projection_horizon"
                    return self.finish(report)
            roi = tuple(
                map(
                    int,
                    warper.warpRoi(
                        tuple(camera["source_size"]), K.astype(np.float32), R.astype(np.float32)
                    ),
                )
            )
            if (
                min(roi[2:]) < 1
                or max(roi[2:]) > 16384
                or max(abs(roi[0]), abs(roi[1])) > 1_000_000
            ):
                raise ValidationError("Native panorama warp bounds exceed limits")
            rois.append(roi)
        origin = [min(r[0] for r in rois), min(r[1] for r in rois)]
        size = [
            max(r[0] + r[2] for r in rois) - origin[0],
            max(r[1] + r[3] for r in rois) - origin[1],
        ]
        canvas_pixels = math.prod(size)
        map_pixels = sum(r[2] * r[3] for r in rois)
        channels = 1 if pixels[0].ndim == 2 else pixels[0].shape[2]
        estimate = (
            total * 40
            + canvas_pixels * (channels * 16 + 8)
            + max(r[2] * r[3] for r in rois) * (channels * 16 + 32)
        )
        if (
            max(size) > 16384
            or max(canvas_pixels, map_pixels) > p["maximum_output_pixels"]
            or estimate > p["maximum_working_bytes"]
        ):
            raise ValidationError(
                "Panorama output/map/working budget exceeded",
                "maximum_output_pixels",
                "maximum_working_bytes",
            )
        output = np.zeros((*size[::-1], channels), np.float64)
        counts = np.zeros(size[::-1], np.uint8)
        method = p["interpolation"]
        halo = {"cubic": 2, "lanczos4": 4}.get(method, 0)
        for camera, roi in zip(cameras, rois, strict=True):
            i = camera["index"]
            K = intrinsics(camera["K"]).astype(np.float32)
            R = rotation(camera["camera_to_panorama"]).astype(np.float32)
            native_roi, x, y = warper.buildMaps(tuple(camera["source_size"]), K, R)
            if tuple(native_roi[:2]) != roi[:2] or x.shape != roi[2:][::-1] or y.shape != x.shape:
                raise ValidationError("Native warp map bounds differ from preflight")
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = np.where(finite, x, -1), np.where(finite, y, -1)
            valid = (
                sample_mask(
                    masks[i] != 0, x, y, "nearest" if method == "nearest" else "linear", halo
                )
                & finite
            )
            warped = cv2.remap(
                pixels[i],
                x,
                y,
                INTERPOLATION[method],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if warped.ndim == 2:
                warped = warped[..., None]
            if not np.isfinite(warped[valid]).all():
                raise ValidationError("Panorama sampling produced nonfinite pixels")
            left, top = roi[0] - origin[0], roi[1] - origin[1]
            target = output[top : top + roi[3], left : left + roi[2]]
            covered = counts[top : top + roi[3], left : left + roi[2]]
            if p["blend"] == "first":
                target[valid & (covered == 0)] = warped[valid & (covered == 0)]
            else:
                old = covered[valid].astype(float)
                new = old + 1
                target[valid] = target[valid] * (old / new)[:, None] + warped[valid] / new[:, None]
            covered[valid] += 1
            camera["panorama_roi"] = [left, top, roi[2], roi[3]]
        mask = np.where(counts > 0, 255, 0).astype(np.uint8)
        fraction = float(np.count_nonzero(mask) / mask.size)
        report.update(
            status="ok",
            coverage=fraction,
            covered_pixels=int(np.count_nonzero(mask)),
            image_size=size,
            projection=p["projection"],
            projection_scale=scale,
            estimated_working_bytes=estimate,
            wave_correction=p["wave_correction"],
            complete=len(retained) == n,
            overlap_pixels=int(np.count_nonzero(counts >= 2)),
        )
        if fraction < p["minimum_covered_fraction"] or not mask.any():
            report["status"] = "insufficient_coverage"
        elif len(retained) != n and p["component_policy"] == "fail_partial":
            report["status"] = "partial_component"
        layout = layout_frame(
            cameras, p["projection"], scale, origin, size, labels[anchor], self.id
        )
        result = cast_intensity(output[..., 0] if channels == 1 else output, pixels[0].dtype)
        if not np.isfinite(result).all():
            raise ValidationError("Panorama accumulation exceeded output range")
        return self.finish(
            report, layout, ImageArray(result, image_format(images[0])), mask, counts
        )
