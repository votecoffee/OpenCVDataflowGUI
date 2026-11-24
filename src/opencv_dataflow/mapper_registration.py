"""OpenCV reg image alignment with explicit sampling geometry and quality checks."""

import cv2
import numpy as np

from .core import Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import metadata
from .processing import ImageTool, gray_image
from .registration import matrix_value
from .selection import SELECTION_INPUTS, selection_mask

_MAPPERS = {
    "translation": "MapperGradShift",
    "euclidean": "MapperGradEuclid",
    "similarity": "MapperGradSimilar",
    "affine": "MapperGradAffine",
    "projective": "MapperGradProj",
}
_MAX_PIXELS = 2_100_000


def _checked_map(transform, motion, size):
    matrix = matrix_value(transform, (3, 3))
    if matrix[2, 2] == 0:
        raise ValidationError("The map has a projective pole at the origin")
    matrix = matrix / matrix[2, 2]
    width, height = size
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [0, height - 1, 1], [width - 1, height - 1, 1]])
    mapped = corners @ matrix.T
    if not np.isfinite(mapped).all() or (mapped[:, 2] <= 1e-6).any():
        raise ValidationError("The map has a projective pole on or too close to the image grid")
    if motion == "projective":
        return matrix
    if not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-9):
        raise ValidationError(f"The {motion} model needs an affine-form transform")
    matrix[2] = [0, 0, 1]
    linear = matrix[:2, :2]
    if motion == "translation":
        if not np.allclose(linear, np.eye(2), rtol=0, atol=1e-6):
            raise ValidationError("A translation transform can only shift the image")
        linear[:] = np.eye(2)
    if motion in ("euclidean", "similarity"):
        gram = linear.T @ linear
        squared_scale = 1.0 if motion == "euclidean" else float(np.trace(gram) / 2)
        if np.linalg.det(linear) <= 0 or not np.allclose(
            gram, squared_scale * np.eye(2), rtol=0, atol=squared_scale * 1e-6
        ):
            raise ValidationError(f"The {motion} transform needs rotation without shear")
        # Remove numerical drift in transforms fitted from float32 control points.
        a = (linear[0, 0] + linear[1, 1]) / 2
        b = (linear[1, 0] - linear[0, 1]) / 2
        linear[:] = [[a, -b], [b, a]]
        if motion == "euclidean":
            linear /= np.hypot(a, b)
    return matrix


def _seed_map(value, motion, size, source_context, target_context):
    if isinstance(value, dict):
        for key, context in (("source", source_context), ("target", target_context)):
            if key + "_size" in value and value[key + "_size"] != list(size):
                raise ValidationError(f"The seed's {key} dimensions differ from the input image")
            declared = value.get(key + "_context")
            if (isinstance(declared, dict) and declared.get("root_id") is not None
                    and context.get("root_id") is not None and declared != context):
                raise ValidationError(f"The seed's {key} frame differs from the input image")
        value = value.get("source_to_target")
    if np.shape(value) == (2, 3):
        value = np.vstack((value, [0, 0, 1]))
    source_to_target = _checked_map(value, motion, size)
    return _checked_map(np.linalg.inv(source_to_target), motion, size)


def _native_matrix(api, result, motion):
    if motion == "projective":
        return api.MapTypeCaster.toProjec(result).getProjTr()
    matrix = np.eye(3)
    if motion == "translation":
        matrix[:2, 2] = api.MapTypeCaster.toShift(result).getShift().ravel()
    else:
        affine = api.MapTypeCaster.toAffine(result)
        matrix[:2, :2] = affine.getLinTr()
        matrix[:2, 2] = affine.getShift().ravel()
    return matrix


def _sample(image, inverse, size, interpolation=cv2.INTER_LINEAR):
    return cv2.warpPerspective(
        image, inverse, size, flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _has_two_directional_gradients(image):
    dx = image[1:-1, 2:] - image[1:-1, :-2]
    dy = image[2:, 1:-1] - image[:-2, 1:-1]
    xx = float(np.einsum("ij,ij->", dx, dx))
    yy = float(np.einsum("ij,ij->", dy, dy))
    xy = float(np.einsum("ij,ij->", dx, dy))
    return xx * yy - xy * xy > 1e-8 * (xx + yy) ** 2


class AlignMapper(ImageTool):
    type_id, title, subtype = "cv.align_mapper", "Align images (multiscale mapper)", "Registration"
    inputs = (
        *SELECTION_INPUTS, Port("reference"),
        Port("reference_mask", required=False, required_when_connected=True),
        Port("reference_region", "data", required=False, required_when_connected=True),
        Port("initial_transform", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("image"), Port("transform", "data"), Port("geometry", "data"),
        Port("coverage", visual_type="mask"), Port("residual"), Port("report", "data"),
    )
    defaults = {
        "motion": "affine", "pyramid_levels": 3, "iterations_per_level": 6,
        "minimum_coverage": 0.5, "maximum_normalized_rmse": 0.12,
    }
    parameter_choices = {"motion": tuple(_MAPPERS)}
    parameter_ranges = {
        "pyramid_levels": NumericRange(1, 8, whole=True),
        "iterations_per_level": NumericRange(1, 25, whole=True),
        "minimum_coverage": NumericRange(0, 1),
        "maximum_normalized_rmse": NumericRange(0, 1),
    }
    parameter_help = {
        "motion": "Select the least flexible motion model that fits the scene.",
        "pyramid_levels": "Total levels including full resolution. One uses only full resolution.",
        "iterations_per_level": "Native gradient refinements per level.",
        "minimum_coverage": "Minimum fraction of selected reference pixels with valid donors.",
        "maximum_normalized_rmse": "Gray RMSE limit relative to the reference intensity range.",
    }
    input_help = {
        "image": "Moving image. Use matching dimensions and sample depth for both images.",
        "mask": "Optional moving-image donor mask for measuring quality and coverage.",
        "region": "Optional moving-image donor region, intersected with mask.",
        "reference": "Fixed reference image and output coordinate grid.",
        "reference_mask": "Optional quality measurement selection in the reference grid.",
        "reference_region": "Optional reference region, intersected with reference_mask.",
        "initial_transform": "Optional moving-to-reference matrix or transform record.",
    }
    output_help = {
        "image": "Moving image resampled once onto the reference grid with linear interpolation.",
        "transform": "Moving-to-reference and reference-to-moving 3 × 3 pixel maps.",
        "geometry": "Output-to-input sampling transform and image dimensions.",
        "coverage": "255 where all weighted interpolation donors and the reference are selected.",
        "residual": "Absolute gray error divided by reference range. Zero outside coverage.",
        "report": "Model, coverage and initial/final RMSE on the same set of valid pixels.",
    }
    payload_reference = "image"
    process_inputs_readonly = True

    def process(self, inputs):
        moving, reference = inputs["image"], inputs["reference"]
        height, width = reference.shape[:2]
        size = (width, height)
        p = self.parameters
        levels, iterations = int(p["pyramid_levels"]), int(p["iterations_per_level"])
        if moving.shape[:2] != reference.shape[:2] or moving.dtype != reference.dtype:
            raise ValidationError("Use images with matching dimensions and sample depth")
        if not 0 < width * height <= _MAX_PIXELS or max(size) >= 32767:
            raise ValidationError("Use images up to 2.1 million pixels and axes below 32,767")
        divisor = 2 ** (levels - 1)
        if (min(size) + divisor - 1) // divisor < 8:
            raise ValidationError(
                "Keep at least 8 × 8 pixels at the coarsest level", "pyramid_levels"
            )
        depths = (np.uint8, np.uint16, np.float32)
        gray, ref = (gray_image(image, depths).astype(np.float64) for image in (moving, reference))
        low = min(gray.min(), ref.min())
        span = max(gray.max(), ref.max()) - low
        ref_span = float(np.ptp(ref))
        if span == 0 or min(np.ptp(gray), ref_span) <= 1e-9 * span:
            raise ValidationError("Both input images need intensity contrast for registration")
        gray, ref = (gray - low) / span, (ref - low) / span
        if not _has_two_directional_gradients(ref) or not _has_two_directional_gradients(gray):
            raise ValidationError("The images need gradients in two directions to constrain motion")
        moving_mask = selection_mask(inputs, moving)
        reference_mask = selection_mask(
            {"image": reference, "mask": inputs.get("reference_mask"),
             "region": inputs.get("reference_region")}, reference
        )
        donors = (
            np.ones(gray.shape, np.float32)
            if moving_mask is None else moving_mask.astype(np.float32) / 255
        )
        selected = np.ones(ref.shape, bool) if reference_mask is None else reference_mask != 0
        if not donors.any() or not selected.any():
            raise ValidationError("Registration quality selections must contain pixels")
        api = getattr(cv2, "reg", None)
        motion = p["motion"]
        if api is None or not hasattr(api, _MAPPERS[motion]) or not hasattr(api, "MapperPyramid"):
            raise ValidationError("Multiscale registration requires OpenCV contrib's reg module")
        source_payload, target_payload = metadata(moving), metadata(reference)
        source_context, target_context = source_payload["spatial"], target_payload["spatial"]
        seed, estimation_image = np.eye(3), gray
        if (initial := inputs.get("initial_transform")) is not None:
            seed = _seed_map(initial, motion, size, source_context, target_context)
            estimation_image = _sample(gray, seed, size, cv2.INTER_CUBIC)
            if np.ptp(estimation_image) <= 1e-9:
                raise ValidationError("The initial transform leaves no usable moving-image texture")
        gradient = getattr(api, _MAPPERS[motion])()  # MapperPyramid stores a reference to it.
        pyramid = api.MapperPyramid(gradient)
        pyramid.numLev_, pyramid.numIterPerScale_ = levels, iterations
        delta = _native_matrix(api, pyramid.calculate(ref, estimation_image), motion)
        # Read the prewarped image at delta(x), which reads the original at seed(delta(x)).
        inverse = _checked_map(seed @ delta, motion, size)
        forward = _checked_map(np.linalg.inv(inverse), motion, size)
        seed_valid = _sample(donors, seed, size) >= 1 - 1e-6
        covered = (_sample(donors, inverse, size) >= 1 - 1e-6) & selected
        common = covered & seed_valid
        error = np.abs(_sample(gray, inverse, size) - ref) * (span / ref_span)
        rmse = float(np.sqrt(np.mean(error[covered] ** 2))) if covered.any() else None
        before, after = None, None
        if common.any():
            seed_error = (_sample(gray, seed, size)[common] - ref[common]) * (span / ref_span)
            before = float(np.sqrt(np.mean(seed_error ** 2)))
            after = float(np.sqrt(np.mean(error[common] ** 2)))
        coverage = float(covered.sum() / selected.sum())
        reasons = []
        if coverage < p["minimum_coverage"] or rmse is None:
            reasons.append("mapper_insufficient_coverage")
        if rmse is not None and rmse > p["maximum_normalized_rmse"]:
            reasons.append("mapper_residual_too_high")

        context = {key: value for key, value in source_payload.items()
                   if key not in ("spatial", "pyramid", "alpha_representation")}
        context.update({key: target_payload[key] for key in ("spatial", "pyramid")
                        if key in target_payload})
        image_context = dict(context)
        if "alpha_representation" in source_payload:
            image_context["alpha_representation"] = source_payload["alpha_representation"]
        transform = {
            "model": "image_mapper_registration_v1", "motion": motion,
            "source_size": list(size), "target_size": list(size),
            "source_context": source_context, "target_context": target_context,
            "source_to_target": forward.tolist(), "target_to_source": inverse.tolist(),
        }
        status = f"Coverage {coverage:.1%} · normalized RMSE "
        status += f"{rmse:.4f}" if rmse is not None else "unavailable"
        drawings = {}
        if self.generate_graphics:
            color = (20, 70, 240) if reasons else (65, 190, 85)
            label = f"{coverage:.1%} overlap"
            drawings["Registration coverage"] = GraphicDrawing(
                (graphic_object("text", [(5, min(18, height - 1))], color, text=label),),
                "image",
                from_output=True,
            )
        return Result(
            {
                "image": ImageArray(_sample(moving, inverse, size), image_format(moving)),
                "transform": transform,
                "geometry": {"input_size": list(size), "output_size": list(size),
                             "output_to_input": transform["target_to_source"]},
                "coverage": covered.astype(np.uint8) * 255,
                "residual": np.where(covered, error, 0).astype(np.float32),
                "report": {
                    "motion": motion, "seeded": initial is not None, "pyramid_levels": levels,
                    "iterations_per_level": iterations, "coverage_fraction": coverage,
                    "valid_pixels": int(covered.sum()), "comparison_pixels": int(common.sum()),
                    "normalized_rmse": rmse, "initial_comparison_rmse": before,
                    "final_comparison_rmse": after,
                },
            },
            payloads={"image": image_context, "coverage": context, "residual": context,
                      "transform": source_payload, "geometry": source_payload, "report": context},
            passed=not reasons, fail_reasons=reasons,
            score=max(0.0, 1.0 - rmse) if rmse is not None else 0.0,
            message=status, drawings=drawings,
        )
