"""Fourier/log-polar similarity proposals with spatial verification and ambiguity gates."""

import copy
import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .processing import image_value
from .selection import selection_mask

INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
BORDERS = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect_101": cv2.BORDER_REFLECT_101,
}


def gray_samples(image):
    raw = image_value(image, (np.uint8, np.uint16, np.float32, np.float64))
    encoding = image_format(image)
    alpha = encoding in ("rgba", "bgra")
    expected = () if encoding == "gray" else (4,) if alpha else (3,)
    if encoding not in ("gray", "rgb", "bgr", "rgba", "bgra") or raw.shape[2:] != expected:
        raise ValidationError("Use grayscale or RGB(A)/BGR(A) intensity images")
    if (
        alpha
        and getattr(image, "payload", {}).get("alpha_representation", "straight") != "straight"
    ):
        raise ValidationError("Unassociate premultiplied images before registration")
    values = (raw[..., :3] if alpha else raw).astype(np.float64)
    scale = float(np.max(np.abs(values))) or 1.0
    values /= scale
    if values.ndim == 3:
        weights = (0.299, 0.587, 0.114) if encoding.startswith("rgb") else (0.114, 0.587, 0.299)
        values = np.sum(values * weights, axis=2)
    valid = np.ones(raw.shape[:2], np.uint8)
    if alpha:
        maximum = np.iinfo(raw.dtype).max if raw.dtype.kind == "u" else 1.0
        if np.any(raw[..., 3] < 0) or np.any(raw[..., 3] > maximum):
            raise ValidationError("Alpha must be within integer full scale or floating [0,1]")
        valid &= raw[..., 3] > 0
    return np.where(valid, values, 0), valid


def spectrum_polar(gray, p):
    n = gray.shape[0]
    data = gray - gray.mean()
    if p["spatial_window"]:
        data *= cv2.createHanningWindow((n, n), cv2.CV_64F)
    values = cv2.dft(data, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude = np.fft.fftshift(np.hypot(values[..., 0], values[..., 1]))
    magnitude = np.log1p(magnitude) if p["spectrum"] == "log" else magnitude
    radius = n * p["radius_fraction"]
    minimum = float(p["minimum_radius"])
    if radius <= minimum + 1:
        raise ValidationError("Frequency radius must exceed Minimum radius by at least one")
    width, height = int(p["radial_bins"]), int(p["angular_bins"])
    polar = cv2.warpPolar(
        magnitude,
        (width, height),
        (n // 2, n // 2),
        radius,
        cv2.WARP_POLAR_LOG | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR,
    )[: height // 2]
    klog = width / math.log(radius)
    cut = int(math.ceil(math.log(minimum) * klog))
    polar = polar[:, cut:].copy()
    if polar.shape[1] < 16:
        raise ValidationError("The selected log-radius band needs at least 16 bins")
    if p["angular_centering"]:
        polar -= polar.mean(axis=0, keepdims=True)
    if p["radial_window"]:
        polar *= np.hanning(polar.shape[1])[None]
    width = cv2.getOptimalDFTSize(polar.shape[1])
    polar = np.pad(polar, ((0, 0), (0, width - polar.shape[1])))
    return polar, klog


def phase_peaks(a, b, count, radius):
    first, second = (cv2.dft(value, flags=cv2.DFT_COMPLEX_OUTPUT) for value in (a, b))
    cross = cv2.mulSpectrums(first, second, 0, conjB=True)
    norm = np.hypot(cross[..., 0], cross[..., 1])
    np.divide(cross, norm[..., None], out=cross, where=norm[..., None] > 0)
    cross[norm == 0] = 0
    response = cv2.idft(cross, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    search = response.copy()
    peaks = []
    h, w = response.shape
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(search), search.shape)
        if not np.isfinite(search[y, x]):
            break
        peaks.append((float(x if x <= w / 2 else x - w), float(y if y <= h / 2 else y - h)))
        ys = (np.arange(y - radius, y + radius + 1) % h).astype(int)
        xs = (np.arange(x - radius, x + radius + 1) % w).astype(int)
        search[np.ix_(ys, xs)] = -np.inf
    return peaks


def angle_scale(matrix):
    scale = float(np.linalg.norm(matrix[:2, 0]))
    angle = (math.degrees(math.atan2(matrix[0, 1], matrix[0, 0])) + 180) % 360 - 180
    return angle, scale


def in_search(matrix, center, q, original_size, p):
    angle, scale = angle_scale(matrix)
    shift = matrix[:2] @ np.array([*center, 1.0]) - center
    maximum = p["maximum_translation"] or 0.4 * max(original_size)
    return (
        p["minimum_angle"] <= angle <= p["maximum_angle"]
        and p["minimum_scale"] <= scale <= p["maximum_scale"]
        and np.linalg.norm(shift) <= maximum * q
    )


def spatial_score(reference, moving, support, selected, matrix, p):
    n = reference.shape[0]
    warped = cv2.warpAffine(moving, matrix[:2], (n, n), flags=cv2.INTER_LINEAR)
    covered = (
        cv2.warpAffine(support.astype(np.float64), matrix[:2], (n, n), flags=cv2.INTER_LINEAR)
        >= 1 - 1e-6
    )
    valid = selected & covered
    fraction = float(valid.sum() / max(1, selected.sum()))
    if valid.sum() < p["minimum_pixels"] or fraction < p["minimum_overlap"]:
        return None, fraction
    a, b = reference[valid], warped[valid]
    a, b = a - a.mean(), b - b.mean()
    na, nb = float(cv2.norm(a, cv2.NORM_L2)), float(cv2.norm(b, cv2.NORM_L2))
    if min(na, nb) / math.sqrt(len(a)) < p["minimum_stddev"] or min(na, nb) == 0:
        return None, fraction
    return float(np.clip(np.sum((a / na) * (b / nb)), -1, 1)), fraction


def refine_similarity(matrix, reference, moving, selected, support, p):
    if not p["ecc_iterations"]:
        return matrix, "disabled"
    inverse = np.linalg.inv(matrix)[:2].astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_COUNT | (cv2.TERM_CRITERIA_EPS if p["ecc_epsilon"] else 0),
        int(p["ecc_iterations"]),
        p["ecc_epsilon"],
    )
    _, inverse = cv2.findTransformECCWithMask(
        reference.astype(np.float32),
        moving.astype(np.float32),
        np.uint8(selected) * 255,
        np.uint8(support) * 255,
        inverse,
        cv2.MOTION_AFFINE,
        criteria,
        int(p["ecc_gaussian"]),
    )
    if not np.isfinite(inverse).all() or np.linalg.det(inverse[:2, :2]) <= 0:
        raise ValueError("ECC did not produce an orientation-preserving transform")
    u, singular, vt = np.linalg.svd(inverse[:2, :2].astype(np.float64))
    if singular[-1] <= 0 or singular[0] / singular[-1] > p["maximum_anisotropy"]:
        raise ValueError("ECC affine proposal exceeds the similarity anisotropy limit")
    rotation = u @ vt
    center = np.array([(reference.shape[0] - 1) / 2] * 2)
    affine = np.eye(3)
    affine[:2, :2] = singular.mean() * rotation
    affine[:2, 2] = inverse[:2, :2] @ center + inverse[:2, 2] - affine[:2, :2] @ center
    return np.linalg.inv(affine), "projected_similarity"


class LogPolarRegistration(Block):
    type_id, title = "inspect.logpolar_registration", "Register rotation and scale (log-polar)"
    category = block_type = "Inspection"
    subtype = "Registration"
    payload_reference = "image"
    parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("image"),
        Port("reference"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
        Port("moving_mask", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image", required=False),
        Port("transform", "data", required=False),
        Port("geometry", "data", required=False),
        Port("valid_mask", visual_type="region", required=False),
        Port("angle", "number", required=False),
        Port("scale", "number", required=False),
        Port("similarity", "number", required=False),
        Port("info", "data"),
        Port("candidates", "data"),
        Port("source"),
        Port("reference"),
    )
    defaults = {
        "work_size": 256,
        "angular_bins": "720",
        "radial_bins": "256",
        "minimum_radius": 3.0,
        "radius_fraction": 0.45,
        "spectrum": "log",
        "spatial_window": True,
        "angular_centering": True,
        "radial_window": True,
        "polar_peaks": 3,
        "peak_radius": 7,
        "minimum_scale": 0.5,
        "maximum_scale": 2.0,
        "minimum_angle": -180.0,
        "maximum_angle": 180.0,
        "maximum_translation": 0.0,
        "minimum_polar_response": 0.05,
        "minimum_score": 0.8,
        "minimum_margin": 0.03,
        "pose_separation": 3.0,
        "minimum_overlap": 0.5,
        "minimum_pixels": 64,
        "minimum_stddev": 1e-5,
        "ecc_iterations": 60,
        "ecc_epsilon": 1e-6,
        "ecc_gaussian": 5,
        "maximum_anisotropy": 1.05,
        "refinement_failure": "reject",
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
    }
    parameter_choices = {
        "angular_bins": ("360", "720", "1440"),
        "radial_bins": ("128", "256", "512"),
        "spectrum": ("log", "magnitude"),
        "refinement_failure": ("reject", "coarse"),
        "interpolation": tuple(INTERPOLATION),
        "border": tuple(BORDERS),
    }
    parameter_ranges = {
        "work_size": NumericRange(64, 1024, whole=True),
        "minimum_radius": NumericRange(1, 256),
        "radius_fraction": NumericRange(0.1, 0.49),
        "polar_peaks": NumericRange(1, 6, whole=True),
        "peak_radius": NumericRange(1, 31, whole=True),
        "minimum_scale": NumericRange(0.25, 4),
        "maximum_scale": NumericRange(0.25, 4),
        "minimum_angle": NumericRange(-180, 180),
        "maximum_angle": NumericRange(-180, 180),
        "maximum_translation": NumericRange(0, 32768),
        "minimum_polar_response": NumericRange(0, 1),
        "minimum_score": NumericRange(0, 1),
        "minimum_margin": NumericRange(0, 1),
        "pose_separation": NumericRange(0.1, 1000),
        "minimum_overlap": NumericRange(0, 1),
        "minimum_pixels": NumericRange(16, 1_048_576, whole=True),
        "minimum_stddev": NumericRange(0, 1),
        "ecc_iterations": NumericRange(0, 500, whole=True),
        "ecc_epsilon": NumericRange(0, 0.1),
        "ecc_gaussian": NumericRange(1, 31, whole=True, odd=True),
        "maximum_anisotropy": NumericRange(1, 2),
        "border_value": NumericRange(),
    }
    parameter_help = {
        "work_size": "Maximum square canvas; isotropic sampling maps back to original pixels.",
        "angular_bins": "Full-circle polar rows; correlation uses the conjugate-symmetric half.",
        "radial_bins": "Log-radius columns before the inner-frequency band is removed.",
        "minimum_radius": "Inner frequency radius in working DFT bins, at least one.",
        "radius_fraction": "Outer frequency radius as a fraction of the square working canvas.",
        "spectrum": "Log-compressed or linear DFT magnitude for the polar proposal.",
        "spatial_window": "Apply Hann taper before DFT to reduce image-edge effects.",
        "angular_centering": "Subtract each radius column's angular mean.",
        "radial_window": "Apply radial Hann taper; angular coordinates remain periodic.",
        "polar_peaks": "Additional polar peaks; each also tests its 180-degree alternative.",
        "peak_radius": "Wrapped nonmaximum suppression radius in polar correlation bins.",
        "minimum_scale": "Minimum moving-to-reference scale; must be below Maximum scale.",
        "maximum_scale": "Maximum moving-to-reference isotropic scale.",
        "minimum_angle": "Minimum moving-to-reference angle in OpenCV degrees, within [-180,180].",
        "maximum_angle": "Must exceed Minimum angle; no wrapping interval is inferred.",
        "maximum_translation": "Center displacement in original pixels; zero uses 40% of size.",
        "minimum_polar_response": "Minimum native polar phase response; not a probability.",
        "minimum_score": "Minimum supported-pixel normalized correlation.",
        "minimum_margin": "Required best-minus-runner-up score for distinct eligible poses.",
        "pose_separation": "Merge poses within this mapped-corner RMS distance in original pixels.",
        "minimum_overlap": "Minimum supported fraction of the selected reference scoring region.",
        "minimum_pixels": "Minimum supported working-grid pixels for a spatial score.",
        "minimum_stddev": "Minimum working luminance variation after common-channel scaling.",
        "ecc_iterations": "Affine ECC iterations before similarity projection; zero disables it.",
        "ecc_epsilon": "ECC convergence epsilon; zero uses iteration count only.",
        "ecc_gaussian": "Odd Gaussian kernel for native ECC refinement.",
        "maximum_anisotropy": "Maximum singular-value ratio of the ECC affine proposal.",
        "refinement_failure": "Reject failed refinement, or allow verified coarse output.",
        "interpolation": "Final resampling; spatial verification always uses linear sampling.",
        "border": "Final image extrapolation; the validity mask excludes extrapolated pixels.",
        "border_value": "Final constant border sample value; ignored by other border modes.",
    }
    input_help = {
        "image": "Moving gray/RGB(A)/BGR(A) uint8/16/float32/64 image; owns the result ticket.",
        "reference": "Reference of equal pixel dimensions; owns the aligned coordinate grid.",
        "mask": "Reference-grid verification selection; does not crop the Fourier proposal.",
        "region": "Reference-grid scoring region intersected with the mask.",
        "moving_mask": "Optional captured support on the moving grid for spatial verification.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "transform", "info"]
        self.documentation = (
            "<p>Estimate an orientation-preserving moving-to-reference similarity using Fourier "
            "magnitude, semilog polar remapping and phase proposals, then translation and spatial "
            "verification. Both 180-degree alternatives and additional polar peaks are evaluated. "
            "Optional affine ECC is projected to isotropic scale/rotation before verification. "
            "Distinct near-equal verified poses withhold primary geometry and aligned images.</p>"
            "<p>Selections affect reference-grid verification. The Fourier seed sees the whole "
            "image; use upstream cropping/masking for unrelated clutter. Candidate uniqueness is "
            "conditional on the searched ranges and finite proposal budget. This is whole-image "
            "registration. It does not perform multi-part matching. Outputs keep moving "
            "depth/encoding and use the reference frame. The moving ticket and both source grids "
            "remain explicit. Scores are correlations, not probabilities.</p>"
        )

    def process(self, inputs):
        p, source, reference = self.parameters, inputs["image"], inputs["reference"]
        shape = source.shape[:2]
        if (
            reference.shape[:2] != shape
            or min(shape) < 32
            or max(shape) > 16384
            or math.prod(shape) > 4_194_304
        ):
            raise ValidationError(
                "Use matching axes of 32–16384 pixels and at most 4,194,304 pixels"
            )
        if p["minimum_scale"] >= p["maximum_scale"] or p["minimum_angle"] >= p["maximum_angle"]:
            raise ValidationError(
                "Search minimum must be below maximum",
                "minimum_scale",
                "maximum_scale",
                "minimum_angle",
                "maximum_angle",
            )
        moving, moving_support = gray_samples(source)
        target, target_support = gray_samples(reference)
        moving_mask = selection_mask({"image": source, "mask": inputs.get("moving_mask")}, source)
        if moving_mask is not None:
            moving_support &= moving_mask != 0
        selected = selection_mask({**inputs, "image": reference}, reference)
        selected = (
            target_support.astype(bool)
            if selected is None
            else (selected != 0) & (target_support != 0)
        )
        if not selected.any() or not moving_support.any():
            raise ValidationError("Selections must include captured pixels")
        h, w = shape
        n = min(int(p["work_size"]), max(shape))
        q = n / max(shape)
        center = np.array([(n - 1) / 2] * 2)
        work = np.eye(3)
        work[:2, :2] *= q
        work[:2, 2] = center - q * np.array([(w - 1) / 2, (h - 1) / 2])
        target_work, moving_work = (
            cv2.warpAffine(value, work[:2], (n, n)) for value in (target, moving)
        )
        selected_work = (
            cv2.warpAffine(np.uint8(selected), work[:2], (n, n), flags=cv2.INTER_NEAREST) != 0
        )
        support_work = cv2.warpAffine(moving_support.astype(float), work[:2], (n, n)) >= 1 - 1e-6
        selected_work &= cv2.warpAffine(target_support.astype(float), work[:2], (n, n)) >= 1 - 1e-6
        if min(selected_work.sum(), support_work.sum()) < p["minimum_pixels"]:
            raise ValidationError(
                "Too few captured working pixels; enlarge the region or Work size"
            )
        reports, distinct = [], []
        status, polar_response = "registration_uninformative", None
        if min(float(target_work.std()), float(moving_work.std())) >= p["minimum_stddev"]:
            polar_a, klog = spectrum_polar(target_work, p)
            polar_b, _ = spectrum_polar(moving_work, p)
            if min(float(polar_a.std()), float(polar_b.std())) > 1e-12:
                shift, polar_response = cv2.phaseCorrelate(polar_a.copy(), polar_b.copy())
                if (
                    np.isfinite([*shift, polar_response]).all()
                    and abs(shift[0]) <= polar_a.shape[1] / 2 + 1
                    and abs(shift[1]) <= polar_a.shape[0] / 2 + 1
                    and polar_response >= p["minimum_polar_response"]
                ):
                    seeds = [(float(shift[0]), float(shift[1]))]
                    seeds += [
                        (-x, -y)
                        for x, y in phase_peaks(
                            polar_a, polar_b, int(p["polar_peaks"]), int(p["peak_radius"])
                        )
                    ]
                    for dx, dy in seeds:
                        scale = math.exp(float(np.clip(dx / klog, -10, 10)))
                        for extra in (0, 180):
                            angle = (dy * 360 / int(p["angular_bins"]) + extra + 180) % 360 - 180
                            row = {
                                "seed_angle": angle,
                                "seed_scale": scale,
                                "score": None,
                                "reason": "outside_search",
                            }
                            matrix = np.eye(3)
                            matrix[:2] = cv2.getRotationMatrix2D(tuple(center), angle, scale)
                            if not (
                                p["minimum_scale"] <= scale <= p["maximum_scale"]
                                and p["minimum_angle"] <= angle <= p["maximum_angle"]
                            ):
                                reports.append(row)
                                continue
                            rotated = cv2.warpAffine(moving_work, matrix[:2], (n, n))
                            translation, response = cv2.phaseCorrelate(target_work.copy(), rotated)
                            if (
                                not np.isfinite([*translation, response]).all()
                                or max(map(abs, translation)) > n / 2 + 1
                            ):
                                row["reason"] = "nonfinite_translation"
                                reports.append(row)
                                continue
                            matrix[:2, 2] -= translation
                            try:
                                matrix, refinement = refine_similarity(
                                    matrix, target_work, moving_work, selected_work, support_work, p
                                )
                            except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
                                refinement = str(exc).splitlines()[0][:240]
                                if p["refinement_failure"] == "reject":
                                    row.update(reason="refinement_rejected", refinement=refinement)
                                    reports.append(row)
                                    continue
                            row["refinement"] = refinement
                            if not in_search(matrix, center, q, (w, h), p):
                                reports.append(row)
                                continue
                            score, overlap = spatial_score(
                                target_work, moving_work, support_work, selected_work, matrix, p
                            )
                            original = np.linalg.inv(work) @ matrix @ work
                            angle, scale = angle_scale(original)
                            row.update(
                                score=score,
                                overlap=overlap,
                                angle=angle,
                                scale=scale,
                                source_to_target=original.tolist(),
                                reason="eligible"
                                if score is not None
                                else "insufficient_spatial_evidence",
                            )
                            reports.append(row)
                    corners = np.array(
                        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]
                    ).T
                    for row in sorted(
                        (r for r in reports if r["score"] is not None),
                        key=lambda r: r["score"],
                        reverse=True,
                    ):
                        mapped = (np.asarray(row["source_to_target"]) @ corners)[:2]
                        if all(
                            np.sqrt(np.mean(np.sum((mapped - other) ** 2, axis=0)))
                            > p["pose_separation"]
                            for _, other in distinct
                        ):
                            distinct.append((row, mapped))
                    status = "registration_no_verified_pose"
                else:
                    status = "registration_low_polar_response"
        best = distinct[0][0] if distinct else None
        margin = best["score"] - distinct[1][0]["score"] if len(distinct) > 1 else None
        if best is not None:
            status = (
                "registration_low_similarity"
                if best["score"] < p["minimum_score"]
                else "registration_ambiguous"
                if margin is not None and margin < p["minimum_margin"]
                else "ok"
            )
        accepted = status == "ok"
        source_payload, ref_payload = self.input_payloads["image"], self.input_payloads["reference"]
        payload = copy.deepcopy(source_payload)
        for key in ("spatial", "pyramid"):
            payload.pop(key, None)
            if key in ref_payload:
                payload[key] = copy.deepcopy(ref_payload[key])
        payload.setdefault("spatial", frame([w, h]))
        payload["registration_source"] = {
            key: copy.deepcopy(source_payload[key])
            for key in ("spatial", "pyramid")
            if key in source_payload
        }
        payload["registration_reference"] = {
            key: copy.deepcopy(ref_payload[key])
            for key in ("spatial", "pyramid", "ticket")
            if key in ref_payload
        }
        map_payload = copy.deepcopy(payload)
        map_payload.pop("alpha_representation", None)
        source_mapping = copy.deepcopy(source_payload)
        source_mapping.pop("alpha_representation", None)
        outputs = {
            name: NO_VALUE
            for name in (
                "image",
                "transform",
                "geometry",
                "valid_mask",
                "angle",
                "scale",
                "similarity",
            )
        }
        if accepted:
            matrix = np.asarray(best["source_to_target"])
            color = image_format(source)
            fill = (
                tuple([p["border_value"]] * source.shape[2])
                if source.ndim == 3
                else p["border_value"]
            )
            aligned = cv2.warpAffine(
                np.asarray(source),
                matrix[:2],
                (w, h),
                flags=INTERPOLATION[p["interpolation"]],
                borderMode=BORDERS[p["border"]],
                borderValue=fill,
            )
            if not np.isfinite(aligned).all():
                raise ValidationError(
                    "Final warp exceeds image precision; rescale or change interpolation"
                )
            radius = {"nearest": 0, "linear": 1, "cubic": 2, "lanczos4": 4}[p["interpolation"]]
            support = (
                cv2.erode(
                    moving_support,
                    np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if radius
                else moving_support
            )
            support = cv2.warpAffine(support, matrix[:2], (w, h), flags=cv2.INTER_NEAREST) != 0
            outputs.update(
                image=ImageArray(aligned, color),
                transform={
                    "model": "image_logpolar_similarity_v1",
                    "source_to_target": matrix.tolist(),
                    "source_size": [w, h],
                    "target_size": [w, h],
                    "source_payload": source_mapping,
                    "target_payload": map_payload,
                },
                geometry={
                    "output_to_input": np.linalg.inv(matrix).tolist(),
                    "input_size": [w, h],
                    "output_size": [w, h],
                    "source_payload": map_payload,
                    "target_payload": source_mapping,
                },
                valid_mask=np.uint8(support & selected) * 255,
                angle=best["angle"],
                scale=best["scale"],
                similarity=best["score"],
            )
        info = {
            "model": "logpolar_registration_v1",
            "status": status,
            "polar_response": float(polar_response)
            if polar_response is not None and np.isfinite(polar_response)
            else None,
            "working_size": n,
            "working_to_original": np.linalg.inv(work).tolist(),
            "distinct_poses": len(distinct),
            "score": best["score"] if best else None,
            "margin": margin,
            "angle_direction": "moving to reference; OpenCV degrees",
            "scale_direction": "moving to reference",
            "selection_scope": "spatial verification",
            "candidate_budget": 2 * (int(p["polar_peaks"]) + 1),
        }
        outputs.update(info=info, candidates=reports, source=source, reference=reference)
        contexts = {name: map_payload for name in outputs}
        contexts.update(image=payload, source=source_payload)
        drawings = {}
        if self.generate_graphics:
            text = f"{best['angle']:.2f} deg / {best['scale']:.4f}" if accepted else status
            label = graphic_object(
                "text", [[5, 16]], (60, 230, 120) if accepted else (60, 90, 240), text=text
            )
            drawings["registration"] = GraphicDrawing((label,), "reference", True)
            for i, (candidate, _) in enumerate(distinct[:1] if accepted else distinct[:2]):
                matrix = np.asarray(candidate["source_to_target"])
                points = np.array(
                    [
                        [(w - 1) / 2, (h - 1) / 2, 1],
                        [(w - 1) / 2 + min(w, h) * 0.15, (h - 1) / 2, 1],
                    ]
                ).T
                mapped = (matrix @ points)[:2].T
                arrow = graphic_object(
                    "arrow", mapped, (60, 230, 120) if accepted else (80, 90 + 80 * i, 240)
                )
                drawings[f"candidate axis {i + 1}"] = GraphicDrawing((arrow,), "reference", True)
        return Result(
            outputs,
            payloads=contexts,
            drawings=drawings,
            passed=accepted,
            score=float(np.clip(best["score"], 0, 1)) if best else 0.0,
            fail_reasons=[] if accepted else [status],
            message=status,
        )
