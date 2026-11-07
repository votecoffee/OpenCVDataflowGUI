"""Geometric transforms, feature registration and additional native filters."""

from itertools import combinations

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS as FILTER_BORDERS
from .parameters import OUTPUT_DEPTHS, anchor_value, optional_mask
from .payloads import metadata, moved_payload
from .processing import (
    ImageTool,
    encoded_result,
    finite,
    geometry_result,
    gray_image,
    image_value,
    integer,
)
from .selection import SELECTION_INPUTS, RegionalOperation, contains

BORDERS = {
    **FILTER_BORDERS,
    "wrap": cv2.BORDER_WRAP,
}
INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def matrix_value(value, shape):
    if isinstance(value, dict):
        value = value.get("source_to_target")
    matrix = np.asarray(value, np.float64)
    if shape == (2, 3) and matrix.shape == (3, 3) and np.allclose(matrix[2], [0, 0, 1]):
        matrix = matrix[:2]
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"Transform must be a finite {shape[0]}x{shape[1]} matrix")
    homogeneous = np.vstack((matrix, [0, 0, 1])) if shape == (2, 3) else matrix
    if np.linalg.matrix_rank(homogeneous) != 3:
        raise ValueError("Transform matrix must be invertible")
    return matrix


def output_size(parameters, image):
    width = integer(parameters["width"], "Width", 0, 16384) or image.shape[1]
    height = integer(parameters["height"], "Height", 0, 16384) or image.shape[0]
    if width * height > 64_000_000:
        raise ValueError("Output is limited to 64 million pixels")
    return width, height


def warped(source, matrix, size, interpolation, border, value, inverse_map=False):
    image = image_value(source)
    value = finite(value, "Border value")
    fill = tuple([value] * image.shape[2]) if image.ndim == 3 else value
    function = cv2.warpAffine if matrix.shape == (2, 3) else cv2.warpPerspective
    result = function(
        image,
        matrix,
        size,
        flags=INTERPOLATION[interpolation] | (cv2.WARP_INVERSE_MAP if inverse_map else 0),
        borderMode=BORDERS[border],
        borderValue=fill,
    )
    homogeneous = np.vstack((matrix, [0, 0, 1])) if matrix.shape == (2, 3) else matrix
    if not np.isfinite(result).all():
        raise ValueError("Warp produced nonfinite samples")
    return geometry_result(
        result, source, homogeneous if inverse_map else np.linalg.inv(homogeneous)
    )


def _normalized_control_points(value, count, field):
    message = f"Provide {count} finite [x, y] points within ±10 million pixels"
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(message, field) from exc
    if points.shape != (count, 2) or not np.isfinite(points).all() or np.abs(points).max() > 1e7:
        raise ValidationError(message, field)

    center = points.mean(axis=0)
    centered = points - center
    span = float(np.abs(centered).max())
    if span < 1e-6:
        raise ValidationError("Control points must span at least 0.000001 pixels", field)
    local = centered / span
    for a, b, c in combinations(local, 3):
        ab, ac = b - a, c - a
        if abs(ab[0] * ac[1] - ab[1] * ac[0]) <= 1e-6:
            raise ValidationError("Control points must be distinct and non-collinear", field)
    basis = np.eye(3)
    basis[:2, :2] /= span
    basis[:2, 2] = -center / span
    return local, basis


class ControlPointTransform(Block):
    type_id, title = "cv.control_point_transform", "Transform from control points"
    category, block_type, subtype = "Image processing", "Image processing", "Geometry"
    inputs = (
        Port("source_points", "data", required=False, required_when_connected=True),
        Port("target_points", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("transform", "data"), Port("rms_pixels", "number"))
    defaults = {"model": "affine", "source_points": [], "target_points": [], "solver": "lu"}
    parameter_choices = {"model": ("affine", "perspective"), "solver": ("lu", "svd")}
    parameter_help = {
        "model": "Affine uses three pairs. Perspective uses four pairs.",
        "source_points": "Ordered source [x, y] coordinates. A wired list takes precedence.",
        "target_points": "Corresponding destination [x, y] coordinates in the same pair order.",
        "solver": "OpenCV linear solver for the perspective model.",
    }
    input_help = {
        "source_points": "Three or four ordered source points.",
        "target_points": "Corresponding points in the destination image.",
    }

    def process(self, inputs):
        p = self.parameters
        affine = p["model"] == "affine"
        count = 3 if affine else 4
        (source, source_basis), (target, target_basis) = (
            _normalized_control_points(inputs.get(field, p[field]), count, field)
            for field in ("source_points", "target_points")
        )
        a, b = source.astype(np.float32), target.astype(np.float32)
        if affine:
            local_matrix = np.vstack((cv2.getAffineTransform(a, b), [0, 0, 1]))
        else:
            solver = cv2.DECOMP_LU if p["solver"] == "lu" else cv2.DECOMP_SVD
            local_matrix = cv2.getPerspectiveTransform(a, b, solver)
        if not np.isfinite(local_matrix).all() or np.linalg.matrix_rank(local_matrix) != 3:
            raise ValidationError("Control points do not define an invertible mapping")
        w = np.c_[source, np.ones(count)] @ local_matrix[2]
        if np.min(np.abs(w)) <= 1e-8 or np.min(w) < 0 < np.max(w):
            raise ValidationError("The mapping crosses infinity within the selected points")
        projected = cv2.perspectiveTransform(source[:, None, :], local_matrix).reshape(-1, 2)
        error = np.linalg.norm(projected - target, axis=1)
        if not np.isfinite(error).all() or error.max() > 1e-4:
            raise ValidationError("Cannot solve the control-point mapping accurately")
        rms = float(np.sqrt(np.mean(error**2)) / target_basis[0, 0])
        matrix = matrix_value(np.linalg.inv(target_basis) @ local_matrix @ source_basis, (3, 3))
        transform = {
            "model": "image_affine_v1" if affine else "image_homography_v1",
            "source_to_target": matrix.tolist(),
            "rms_pixels": rms,
        }
        return Result(
            {"transform": transform, "rms_pixels": rms},
            message=f"{count} pairs · RMS {rms:.3g} px",
        )


class WarpAffine(ImageTool):
    type_id, title, subtype = "cv.warp_affine", "Warp affine", "Geometry"
    inputs = (
        Port("image"),
        Port("transform", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "width": 0,
        "height": 0,
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
        "inverse_map": False,
    }
    parameter_choices = {"interpolation": tuple(INTERPOLATION), "border": tuple(BORDERS)}

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        matrix = matrix_value(inputs.get("transform", p["matrix"]), (2, 3))
        return warped(
            inputs["image"],
            matrix,
            output_size(p, image),
            p["interpolation"],
            p["border"],
            p["border_value"],
            p["inverse_map"],
        )


class WarpPerspective(WarpAffine):
    type_id, title = "cv.warp_perspective", "Warp perspective"
    defaults = {
        **WarpAffine.defaults,
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    def process(self, inputs):
        p = self.parameters
        image = image_value(inputs["image"])
        matrix = matrix_value(inputs.get("transform", p["matrix"]), (3, 3))
        return warped(
            inputs["image"],
            matrix,
            output_size(p, image),
            p["interpolation"],
            p["border"],
            p["border_value"],
            p["inverse_map"],
        )


class Rotate(ImageTool):
    type_id, title, subtype = "cv.rotate", "Rotate arbitrary angle", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {
        "angle_degrees": 15.0,
        "scale": 1.0,
        "center": [],
        "interpolation": "linear",
        "border": "constant",
        "border_value": 0.0,
    }
    parameter_choices = WarpAffine.parameter_choices

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        p = self.parameters
        angle, scale = finite(p["angle_degrees"], "Angle"), finite(p["scale"], "Scale", 0)
        if scale <= 0:
            raise ValueError("Scale must be positive")
        center = p["center"] or [(image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2]
        if np.asarray(center).shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Center must be a finite [x,y] pair or an empty list")
        matrix = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, scale)
        return warped(
            source,
            matrix,
            (image.shape[1], image.shape[0]),
            p["interpolation"],
            p["border"],
            p["border_value"],
        )


class CopyBorder(ImageTool):
    type_id, title, subtype = "cv.copy_border", "Add image border", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {
        "top": 10,
        "bottom": 10,
        "left": 10,
        "right": 10,
        "border": "constant",
        "value": 0.0,
    }
    parameter_choices = {"border": tuple(BORDERS)}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        top, bottom, left, right = (
            integer(self.parameters[k], k, 0, 16384) for k in ("top", "bottom", "left", "right")
        )
        if (image.shape[0] + top + bottom) * (image.shape[1] + left + right) > 64_000_000:
            raise ValueError("Border output is limited to 64 million pixels")
        value = finite(self.parameters["value"], "Value")
        fill = tuple([value] * image.shape[2]) if image.ndim == 3 else value
        output = cv2.copyMakeBorder(
            image, top, bottom, left, right, BORDERS[self.parameters["border"]], value=fill
        )
        return geometry_result(output, source, [[1, 0, -left], [0, 1, -top], [0, 0, 1]])


class Pyramid(ImageTool):
    direction = "down"
    type_id, title, subtype = "cv.pyramid", "Gaussian pyramid up/down", "Geometry"
    outputs = (Port("image"), Port("geometry", "data"))
    defaults = {
        "direction": "down",
        "levels": 1,
        "border": "reflect",
        "output_width": 0,
        "output_height": 0,
    }
    parameter_choices = {
        "direction": ("down", "up"),
        "border": ("replicate", "reflect", "reflect_101", "reflect_edge"),
    }
    parameter_help = {
        "output_width": "0 uses native sizing. Explicit dimensions apply to the final level.",
        "border": "PyrUp supports only reflect/reflect_101; PyrDown also supports other choices.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        levels = integer(self.parameters["levels"], "Levels", 1, 6)
        direction = self.parameters.get("direction", self.direction)
        if direction not in ("down", "up"):
            raise ValueError("Pyramid direction must be up or down")
        if direction == "up" and image.shape[0] * image.shape[1] * 4**levels > 64_000_000:
            raise ValueError("Pyramid output is limited to 64 million pixels")
        border = BORDERS[self.parameters["border"]]
        if direction == "up" and border != cv2.BORDER_REFLECT_101:
            raise ValueError("PyrUp supports only reflect_101 border")
        width, height = (
            integer(self.parameters[k], k, 0, 16384) for k in ("output_width", "output_height")
        )
        for level in range(levels):
            default = (
                ((image.shape[1] + 1) // 2, (image.shape[0] + 1) // 2)
                if direction == "down"
                else (image.shape[1] * 2, image.shape[0] * 2)
            )
            size = (width or default[0], height or default[1]) if level == levels - 1 else default
            if size[0] * size[1] > 64_000_000:
                raise ValueError("Pyramid output exceeds 64 million pixels")
            if direction == "down":
                image = cv2.pyrDown(image, dstsize=size, borderType=border)
            else:
                image = cv2.pyrUp(image, dstsize=size, borderType=border)
        factor = 2.0**levels if direction == "down" else 0.5**levels
        matrix = np.diag([factor, factor, 1.0])
        result = geometry_result(image, source, matrix)
        payload = moved_payload(metadata(source), matrix, [image.shape[1], image.shape[0]])
        payload["pyramid_operation"] = {"direction": direction, "levels": levels}
        if "pyramid" not in payload:
            payload["pyramid"] = {
                "set_id": self.id,
                "level": levels if direction == "down" else 0,
                "largest_size": [source.shape[1], source.shape[0]]
                if direction == "down"
                else [image.shape[1], image.shape[0]],
                "to_largest": matrix.tolist() if direction == "down" else np.eye(3).tolist(),
            }
        result.payloads = {"image": payload, "geometry": payload}
        return result


class PyramidDown(Pyramid):
    type_id, title, subtype = "cv.pyramid_down", "Pyramid down", "Pyramids"
    defaults = {key: value for key, value in Pyramid.defaults.items() if key != "direction"}
    parameter_choices = {"border": Pyramid.parameter_choices["border"]}


class PyramidUp(PyramidDown):
    type_id, title = "cv.pyramid_up", "Pyramid up"
    direction = "up"
    parameter_choices = {"border": ("reflect", "reflect_101")}


class Filter2D(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.filter2d", "Custom 2D filter", "Filtering"
    defaults = {
        "kernel": [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        "delta": 0.0,
        "border": "reflect",
        "anchor": [-1, -1],
        "output_depth": "same",
    }
    parameter_choices = {"border": tuple(FILTER_BORDERS), "output_depth": tuple(OUTPUT_DEPTHS)}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        kernel = np.asarray(self.parameters["kernel"], np.float64)
        if (
            kernel.ndim != 2
            or not kernel.size
            or max(kernel.shape) > 31
            or not np.isfinite(kernel).all()
        ):
            raise ValueError("Kernel must be a finite nonempty matrix with at most 31 rows/columns")
        output = cv2.filter2D(
            image,
            OUTPUT_DEPTHS[self.parameters["output_depth"]],
            kernel,
            anchor=anchor_value(self.parameters["anchor"], kernel.shape[1], kernel.shape[0]),
            delta=finite(self.parameters["delta"], "Delta"),
            borderType=BORDERS[self.parameters["border"]],
        )
        if not np.isfinite(output).all():
            raise ValueError("Filter overflowed the output depth")
        return encoded_result(output, source)


class UnsharpMask(RegionalOperation, ImageTool):
    type_id, title, subtype = "cv.unsharp", "Unsharp mask", "Filtering"
    defaults = {
        "kernel": 5,
        "sigma": 1.0,
        "amount": 1.0,
        "kernel_y": 0,
        "sigma_y": 0.0,
        "border": "reflect",
    }
    parameter_choices = {"border": tuple(FILTER_BORDERS)}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source)
        size = integer(self.parameters["kernel"], "Kernel", 3, 255, odd=True)
        sigma = finite(self.parameters["sigma"], "Sigma", 0)
        amount = finite(self.parameters["amount"], "Amount", 0)
        ky = integer(self.parameters["kernel_y"], "Kernel Y", 0, 255) or size
        if ky % 2 == 0:
            raise ValueError("Kernel Y must be odd")
        sy = finite(self.parameters["sigma_y"], "Sigma Y", 0)
        output = cv2.addWeighted(
            image,
            1 + amount,
            cv2.GaussianBlur(
                image,
                (size, ky),
                sigma,
                sigmaY=sy,
                borderType=FILTER_BORDERS[self.parameters["border"]],
            ),
            -amount,
            0,
        )
        if not np.isfinite(output).all():
            raise ValueError("Unsharp mask overflowed the output depth")
        return encoded_result(output, source)


class FASTFeatures(ImageTool):
    type_id, title, subtype = "inspect.fast", "FAST corners", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs = SELECTION_INPUTS
    outputs = (Port("points", "data"), Port("count", "number"), Port("overlay"))
    defaults = {"threshold": 10, "nonmax_suppression": True, "pattern": "9_16"}
    parameter_choices = {"pattern": ("5_8", "7_12", "9_16")}

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,))
        pattern = {
            "5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
            "7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            "9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        }[self.parameters["pattern"]]
        detector = cv2.FastFeatureDetector_create(
            integer(self.parameters["threshold"], "Threshold", 0, 255),
            nonmaxSuppression=self.parameters["nonmax_suppression"],
            type=pattern,
        )
        keypoints = detector.detect(gray, optional_mask(inputs, gray))
        overlay = cv2.drawKeypoints(
            overlay_image(inputs["image"]), keypoints, None, color=(70, 220, 140)
        )
        return Result(
            {
                "points": [list(k.pt) for k in keypoints],
                "count": len(keypoints),
                "overlay": overlay,
            },
            graphics={"FAST": overlay},
        )


class ORBFeatures(ImageTool):
    type_id, title, subtype = "inspect.orb", "ORB keypoints and descriptors", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("features", "data"),
        Port("points", "data"),
        Port("count", "number"),
        Port("overlay"),
    )
    defaults = {
        "max_features": 500,
        "scale_factor": 1.2,
        "levels": 8,
        "fast_threshold": 20,
        "edge_threshold": 31,
        "first_level": 0,
        "wta_k": "2",
        "score_type": "harris",
        "patch_size": 31,
    }
    parameter_choices = {"wta_k": ("2", "3", "4"), "score_type": ("harris", "fast")}

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,))
        p = self.parameters
        scale = finite(p["scale_factor"], "Scale factor", 1)
        if scale <= 1:
            raise ValueError("ORB scale factor must exceed one")
        detector = cv2.ORB_create(
            nfeatures=integer(p["max_features"], "Maximum features", 1, 10000),
            scaleFactor=scale,
            nlevels=integer(p["levels"], "Levels", 1, 32),
            fastThreshold=integer(p["fast_threshold"], "FAST threshold", 0, 255),
            edgeThreshold=integer(p["edge_threshold"], "Edge threshold", 0, 1024),
            firstLevel=integer(p["first_level"], "First level", 0, int(p["levels"]) - 1),
            WTA_K=int(p["wta_k"]),
            scoreType={"harris": cv2.ORB_HARRIS_SCORE, "fast": cv2.ORB_FAST_SCORE}[p["score_type"]],
            patchSize=integer(p["patch_size"], "Patch size", 3, 255, odd=True),
        )
        return feature_result(
            inputs, gray, detector, "ORB", {"descriptor_size": 32, "wta_k": int(p["wta_k"])}
        )


def feature_result(inputs, gray, detector, algorithm, details, *, limit=None):
    mask = optional_mask(inputs, gray)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    if mask is not None and keypoints:
        keep = [index for index, point in enumerate(keypoints) if contains(mask, point.pt)]
        keypoints = [keypoints[index] for index in keep]
        descriptors = descriptors[keep] if descriptors is not None else None
    if limit is not None and len(keypoints) > limit:
        keep = sorted(range(len(keypoints)), key=lambda index: -keypoints[index].response)[:limit]
        keypoints = [keypoints[index] for index in keep]
        descriptors = descriptors[keep]
    points = [list(k.pt) for k in keypoints]
    features = {
        "algorithm": algorithm,
        **details,
        "image_size": [gray.shape[1], gray.shape[0]],
        "keypoints": [
            {
                "point": point,
                "size": k.size,
                "angle": k.angle,
                "response": k.response,
                "octave": k.octave,
            }
            for k, point in zip(keypoints, points, strict=True)
        ],
        "descriptors": descriptors.tolist() if descriptors is not None else [],
    }
    overlay = cv2.drawKeypoints(
        overlay_image(inputs["image"]),
        keypoints,
        None,
        color=(70, 220, 140),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return Result(
        {"features": features, "points": points, "count": len(points), "overlay": overlay},
        graphics={algorithm: overlay},
    )


class SIFTFeatures(ImageTool):
    type_id, title, subtype = "inspect.sift", "SIFT keypoints and descriptors", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs, outputs = SELECTION_INPUTS, ORBFeatures.outputs
    defaults = {
        "max_features": 500,
        "octave_layers": 3,
        "contrast_threshold": 0.04,
        "edge_threshold": 10.0,
        "sigma": 1.6,
        "descriptor_depth": "float32",
        "precise_upscale": False,
    }
    parameter_choices = {"descriptor_depth": ("float32", "uint8")}
    parameter_ranges = {
        "max_features": NumericRange(1, 10000, whole=True),
        "octave_layers": NumericRange(1, 16, whole=True),
        "contrast_threshold": NumericRange(0, 1),
        "edge_threshold": NumericRange(0.01, 1000),
        "sigma": NumericRange(0.01, 100),
    }
    parameter_help = {
        "contrast_threshold": "Larger values reject more low-contrast keypoints.",
        "edge_threshold": "Larger values keep more edge-like keypoints.",
        "precise_upscale": "Use OpenCV's precise initial upscale to reduce localization bias.",
    }

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        p = self.parameters
        detector = cv2.SIFT_create(
            int(p["max_features"]),
            int(p["octave_layers"]),
            float(p["contrast_threshold"]),
            float(p["edge_threshold"]),
            float(p["sigma"]),
            cv2.CV_8U if p["descriptor_depth"] == "uint8" else cv2.CV_32F,
            p["precise_upscale"],
        )
        result = feature_result(
            inputs,
            gray,
            detector,
            "SIFT",
            {"descriptor_size": 128, "descriptor_depth": p["descriptor_depth"]},
        )
        if self.generate_graphics and result.outputs["points"]:
            result.drawings["SIFT"] = GraphicDrawing(
                tuple(
                    graphic_object("marker", [point], (70, 220, 140))
                    for point in result.outputs["points"]
                )
            )
        return result


def orb_data(data):
    if data.get("algorithm") != "ORB" or data.get("descriptor_size") != 32:
        raise ValueError("Matcher requires 32-byte ORB descriptors")
    if data.get("wta_k", 2) not in (2, 3, 4):
        raise ValueError("Invalid ORB WTA_K metadata")
    keys = data["keypoints"]
    raw = np.asarray(data["descriptors"])
    if not keys and raw.size == 0:
        return np.empty((0, 32), np.uint8), np.empty((0, 2), np.float64)
    points = np.asarray([k["point"] for k in keys], np.float64)
    if (
        raw.shape != (len(keys), 32)
        or not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
        or np.any(raw < 0)
        or np.any(raw > 255)
        or np.any(raw != np.rint(raw))
        or points.shape != (len(keys), 2)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Invalid descriptor bytes or keypoint coordinates")
    return raw.astype(np.uint8), points


class MatchORB(Block):
    type_id, title, category = "inspect.match_orb", "Match ORB descriptors", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs = (Port("query", "data"), Port("train", "data"))
    outputs = (Port("matches", "data"), Port("count", "number"))
    defaults = {"ratio": 0.75, "max_distance": 64.0, "mode": "ratio", "norm": "auto"}
    parameter_choices = {"mode": ("ratio", "cross_check"), "norm": ("auto", "hamming", "hamming2")}

    def process(self, inputs):
        query, qpoints = orb_data(inputs["query"])
        train, tpoints = orb_data(inputs["train"])
        ratio = finite(self.parameters["ratio"], "Ratio", 0)
        maximum = finite(self.parameters["max_distance"], "Maximum distance", 0)
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be strictly between zero and one")
        wta = inputs["query"].get("wta_k", 2)
        if inputs["train"].get("wta_k", 2) != wta:
            raise ValueError("ORB descriptor WTA_K values must match")
        expected_norm = "hamming" if wta == 2 else "hamming2"
        norm_name = expected_norm if self.parameters["norm"] == "auto" else self.parameters["norm"]
        if norm_name != expected_norm:
            raise ValueError("Matcher norm is incompatible with ORB WTA_K")
        norm = cv2.NORM_HAMMING if norm_name == "hamming" else cv2.NORM_HAMMING2
        return descriptor_match_result(
            inputs, query, train, qpoints, tpoints, norm, self.parameters["mode"], ratio, maximum
        )


def descriptor_match_result(inputs, query, train, qpoints, tpoints, norm, mode, ratio, maximum):
    selected = []
    if mode == "cross_check" and len(query) and len(train):
        selected = [
            m
            for m in cv2.BFMatcher(norm, crossCheck=True).match(query, train)
            if m.distance <= maximum
        ]
    elif len(query) and len(train) >= 2:
        pairs = cv2.BFMatcher(norm).knnMatch(query, train, k=2)
        selected = ratio_matches(pairs, ratio, maximum)
    return paired_feature_result(inputs, qpoints, tpoints, selected)


def ratio_matches(pairs, ratio, maximum):
    return [
        pair[0]
        for pair in pairs
        if len(pair) == 2
        and pair[0].distance < ratio * pair[1].distance
        and pair[0].distance <= maximum
    ]


def paired_feature_result(inputs, qpoints, tpoints, selected):
    data = {
        "query_points": [qpoints[m.queryIdx].tolist() for m in selected],
        "train_points": [tpoints[m.trainIdx].tolist() for m in selected],
        "matches": [
            {"query_index": m.queryIdx, "train_index": m.trainIdx, "distance": m.distance}
            for m in selected
        ],
        "query_size": inputs["query"]["image_size"],
        "train_size": inputs["train"]["image_size"],
    }
    return Result(
        {"matches": data, "count": len(selected)},
        message=f"{len(selected)} descriptor matches",
    )


def sift_data(data):
    if (
        not isinstance(data, dict)
        or data.get("algorithm") != "SIFT"
        or data.get("descriptor_size") != 128
        or data.get("descriptor_depth") not in ("uint8", "float32")
    ):
        raise ValidationError("Connect SIFT keypoints and descriptors")
    byte_storage = data["descriptor_depth"] == "uint8"
    raw, points = descriptor_arrays(data, 128, 0, 255 if byte_storage else 512, byte_storage)
    return raw.astype(np.float32), points


def descriptor_arrays(data, width, minimum, maximum, integer_only):
    try:
        size = data["image_size"]
        keypoints = data["keypoints"]
        raw = np.asarray(data["descriptors"])
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
            or not isinstance(keypoints, list)
            or raw.dtype.kind not in "uif"
        ):
            raise ValueError("Invalid image size, keypoints or descriptor type")
        if not keypoints and raw.size == 0:
            return np.empty((0, width)), np.empty((0, 2))
        points = np.asarray([key["point"] for key in keypoints], np.float64)
        if (
            raw.shape != (len(keypoints), width)
            or not np.isfinite(raw).all()
            or (raw < minimum).any()
            or (raw > maximum).any()
            or (integer_only and np.any(raw != np.rint(raw)))
            or points.shape != (len(keypoints), 2)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Invalid descriptor values or keypoint coordinates")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"Invalid feature data: {exc}") from exc
    return raw, points


class MatchSIFT(Block):
    type_id, title, category = "inspect.match_sift", "Match SIFT descriptors", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs, outputs = MatchORB.inputs, MatchORB.outputs
    defaults = {"ratio": 0.75, "max_distance": 250.0, "mode": "ratio", "norm": "l2"}
    parameter_choices = {"mode": ("ratio", "cross_check"), "norm": ("l1", "l2")}
    parameter_ranges = {
        "ratio": NumericRange(0.000001, 0.999999),
        "max_distance": NumericRange(0),
    }
    parameter_help = {
        "ratio": "Reject a nearest neighbor too similar in distance to the runner-up.",
        "max_distance": "Maximum descriptor distance using the selected norm.",
    }

    def process(self, inputs):
        query, qpoints = sift_data(inputs["query"])
        train, tpoints = sift_data(inputs["train"])
        p = self.parameters
        norm = cv2.NORM_L2 if p["norm"] == "l2" else cv2.NORM_L1
        return descriptor_match_result(
            inputs,
            query,
            train,
            qpoints,
            tpoints,
            norm,
            p["mode"],
            float(p["ratio"]),
            float(p["max_distance"]),
        )


AKAZE_MODES = ("mldb", "mldb_upright", "kaze", "kaze_upright")
KAZE_DIFFUSIVITY = ("pm_g1", "pm_g2", "weickert", "charbonnier")


class KAZEFeatures(ImageTool):
    type_id, title, subtype = "inspect.kaze", "KAZE keypoints and descriptors", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs, outputs = SELECTION_INPUTS, ORBFeatures.outputs
    defaults = {
        "extended": False,
        "upright": False,
        "threshold": 0.001,
        "octaves": 4,
        "octave_layers": 4,
        "diffusivity": "pm_g2",
        "max_features": 500,
    }
    parameter_choices = {"diffusivity": KAZE_DIFFUSIVITY}
    parameter_ranges = {
        "threshold": NumericRange(1e-9, 1),
        "octaves": NumericRange(1, 8, whole=True),
        "octave_layers": NumericRange(1, 8, whole=True),
        "max_features": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        "extended": "Use 128 floating-point values per descriptor instead of 64.",
        "upright": "Use a fixed descriptor orientation for scenes without rotation.",
        "threshold": "Minimum detector response for accepting a keypoint.",
        "octaves": "Maximum number of scales in the nonlinear scale space.",
        "octave_layers": "Number of sublevels within each scale.",
        "diffusivity": "Diffusion model used to build the nonlinear scale space.",
        "max_features": "Keep up to this many of the strongest selected keypoints.",
    }

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        p = self.parameters
        detector = cv2.xfeatures2d.KAZE_create(
            p["extended"],
            p["upright"],
            float(p["threshold"]),
            int(p["octaves"]),
            int(p["octave_layers"]),
            getattr(cv2.xfeatures2d, f"KAZE_DIFF_{p['diffusivity'].upper()}"),
        )
        result = feature_result(
            inputs,
            gray,
            detector,
            "KAZE",
            {
                "descriptor_size": detector.descriptorSize(),
                "extended": p["extended"],
                "upright": p["upright"],
            },
            limit=int(p["max_features"]),
        )
        if self.generate_graphics and result.outputs["points"]:
            result.drawings["KAZE"] = GraphicDrawing(
                tuple(
                    graphic_object("marker", [point], (70, 220, 140))
                    for point in result.outputs["points"]
                )
            )
        return result


def akaze_descriptor_width(mode, bits, channels):
    if not mode.startswith("mldb"):
        return 64
    if bits == 0 and channels != 3:
        raise ValidationError("Full-length MLDB requires three channels", "descriptor_channels")
    if bits > 162 * channels:
        raise ValidationError("MLDB supports at most 162 bits per channel", "descriptor_bits")
    return ((bits or 486) + 7) // 8


class AKAZEFeatures(ImageTool):
    type_id, title, subtype = "inspect.akaze", "AKAZE keypoints and descriptors", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs, outputs = SELECTION_INPUTS, ORBFeatures.outputs
    defaults = {
        "descriptor_mode": "mldb",
        "descriptor_bits": 0,
        "descriptor_channels": "3",
        "threshold": 0.001,
        "octaves": 4,
        "octave_layers": 4,
        "diffusivity": "pm_g2",
        "max_features": 500,
    }
    parameter_choices = {
        "descriptor_mode": AKAZE_MODES,
        "descriptor_channels": ("1", "2", "3"),
        "diffusivity": KAZE_DIFFUSIVITY,
    }
    parameter_ranges = {
        "descriptor_bits": NumericRange(0, 486, whole=True),
        "threshold": NumericRange(1e-9, 1),
        "octaves": NumericRange(1, 8, whole=True),
        "octave_layers": NumericRange(1, 8, whole=True),
        "max_features": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        "descriptor_bits": "MLDB only. Zero selects full length with three channels.",
        "descriptor_channels": "Use three channels on the current OpenCV build.",
        "descriptor_mode": "Upright modes keep a fixed orientation. KAZE descriptors are floats.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        # Reduced-channel MLDB kernels can corrupt the heap in the OpenCV 5.0 wheel.
        if p["descriptor_mode"].startswith("mldb") and p["descriptor_channels"] != "3":
            raise ValidationError("MLDB requires three channels", "descriptor_channels")
        akaze_descriptor_width(
            p["descriptor_mode"], int(p["descriptor_bits"]), int(p["descriptor_channels"])
        )

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        p = self.parameters
        mode = p["descriptor_mode"]
        binary = mode.startswith("mldb")
        bits = int(p["descriptor_bits"]) if binary else 0
        channels = int(p["descriptor_channels"]) if binary else 3
        native = cv2.xfeatures2d
        detector = cv2.xfeatures2d.AKAZE_create(
            getattr(native, f"AKAZE_DESCRIPTOR_{mode.upper()}"),
            bits,
            channels,
            float(p["threshold"]),
            int(p["octaves"]),
            int(p["octave_layers"]),
            getattr(native, f"KAZE_DIFF_{p['diffusivity'].upper()}"),
            int(p["max_features"]),
        )
        result = feature_result(
            inputs,
            gray,
            detector,
            "AKAZE",
            {
                "descriptor_size": detector.descriptorSize(),
                "descriptor_mode": mode,
                "descriptor_bits": bits,
                "descriptor_channels": channels,
            },
        )
        if self.generate_graphics and result.outputs["points"]:
            result.drawings["AKAZE"] = GraphicDrawing(
                tuple(
                    graphic_object("marker", [point], (70, 220, 140))
                    for point in result.outputs["points"]
                )
            )
        return result


def akaze_data(data):
    if not isinstance(data, dict) or data.get("algorithm") != "AKAZE":
        raise ValidationError("Connect AKAZE keypoints and descriptors")
    mode = data.get("descriptor_mode")
    bits, channels = data.get("descriptor_bits"), data.get("descriptor_channels")
    if (
        mode not in AKAZE_MODES
        or type(bits) is not int
        or not 0 <= bits <= 486
        or type(channels) is not int
        or channels not in (1, 2, 3)
    ):
        raise ValidationError("Invalid AKAZE descriptor settings")
    binary = mode.startswith("mldb")
    width = akaze_descriptor_width(mode, bits, channels)
    if width != data.get("descriptor_size"):
        raise ValidationError("AKAZE descriptor width does not match its settings")
    minimum, maximum = (0, 255) if binary else (-2, 2)
    raw, points = descriptor_arrays(data, width, minimum, maximum, binary)
    return raw.astype(np.uint8 if binary else np.float32), points, (mode, bits, channels)


class MatchAKAZE(Block):
    type_id, title, category = "inspect.match_akaze", "Match AKAZE descriptors", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs, outputs = MatchSIFT.inputs, MatchSIFT.outputs
    defaults = {"ratio": 0.75, "max_distance": 64.0, "mode": "ratio"}
    parameter_choices = {"mode": ("ratio", "cross_check")}
    parameter_ranges = MatchSIFT.parameter_ranges
    parameter_help = MatchSIFT.parameter_help

    def process(self, inputs):
        query, qpoints, settings = akaze_data(inputs["query"])
        train, tpoints, train_settings = akaze_data(inputs["train"])
        if settings != train_settings:
            raise ValidationError("AKAZE descriptor modes, bit counts and channels must match")
        p = self.parameters
        norm = cv2.NORM_HAMMING if query.dtype == np.uint8 else cv2.NORM_L2
        return descriptor_match_result(
            inputs,
            query,
            train,
            qpoints,
            tpoints,
            norm,
            p["mode"],
            float(p["ratio"]),
            float(p["max_distance"]),
        )


FLANN_INDEX_KDTREE = 1


def kaze_data(data):
    extended, upright = data.get("extended"), data.get("upright")
    if type(extended) is not bool or type(upright) is not bool:
        raise ValidationError("KAZE descriptors require extended and upright settings")
    width = 128 if extended else 64
    if data.get("descriptor_size") != width:
        raise ValidationError(f"KAZE descriptor size must be {width} for this mode")
    descriptors, points = descriptor_arrays(data, width, -2, 2, False)
    return descriptors.astype(np.float32), points, ("KAZE", extended, upright)


def flann_data(data):
    if isinstance(data, dict) and data.get("algorithm") == "SIFT":
        descriptors, points = sift_data(data)
        return descriptors, points, ("SIFT",)
    if isinstance(data, dict) and data.get("algorithm") == "AKAZE":
        descriptors, points, settings = akaze_data(data)
        if descriptors.dtype == np.float32:
            return descriptors, points, ("AKAZE", *settings)
    if isinstance(data, dict) and data.get("algorithm") == "KAZE":
        return kaze_data(data)
    raise ValidationError("FLANN KD-tree requires SIFT, KAZE, or float AKAZE descriptors")


class MatchFLANN(Block):
    type_id, title, category = "inspect.match_flann", "Match descriptors with FLANN", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs, outputs = MatchSIFT.inputs, MatchSIFT.outputs
    defaults = {
        "mode": "ratio",
        "ratio": 0.75,
        "max_distance": 250.0,
        "trees": 5,
        "checks": 50,
    }
    parameter_choices = {"mode": ("ratio", "cross_check")}
    parameter_ranges = {
        **MatchSIFT.parameter_ranges,
        "trees": NumericRange(1, 32, whole=True),
        "checks": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        **MatchSIFT.parameter_help,
        "trees": "Number of randomized KD-trees in the search index.",
        "checks": "Search budget. Higher values can improve recall at greater cost.",
    }

    def process(self, inputs):
        query, qpoints, layout = flann_data(inputs["query"])
        train, tpoints, train_layout = flann_data(inputs["train"])
        if layout != train_layout:
            raise ValidationError("Query and train need the same FLANN descriptor family and mode")
        p = self.parameters
        maximum = float(p["max_distance"])
        selected = []
        if len(query) and len(train):
            matcher = cv2.FlannBasedMatcher(
                {"algorithm": FLANN_INDEX_KDTREE, "trees": int(p["trees"])},
                {"checks": int(p["checks"])},
            )
            if p["mode"] == "cross_check":
                forward = matcher.match(query, train)
                reverse = {m.queryIdx: m.trainIdx for m in matcher.match(train, query)}
                selected = [
                    m
                    for m in forward
                    if m.distance <= maximum and reverse.get(m.trainIdx) == m.queryIdx
                ]
            elif len(train) >= 2:
                selected = ratio_matches(
                    matcher.knnMatch(query, train, k=2), float(p["ratio"]), maximum
                )
        return paired_feature_result(inputs, qpoints, tpoints, selected)


class BRISKFeatures(ImageTool):
    type_id, title, subtype = "inspect.brisk", "BRISK keypoints and descriptors", "Features"
    category, block_type = "Inspection", "Inspection"
    inputs, outputs = SELECTION_INPUTS, ORBFeatures.outputs
    defaults = {"threshold": 30, "octaves": 3, "pattern_scale": 1.0, "max_features": 500}
    parameter_ranges = {
        "threshold": NumericRange(1, 255, whole=True),
        "octaves": NumericRange(0, 8, whole=True),
        "pattern_scale": NumericRange(0.5, 4),
        "max_features": NumericRange(1, 10000, whole=True),
    }
    parameter_help = {
        "threshold": "AGAST response threshold.",
        "octaves": "Number of detection octaves. Zero uses a single scale.",
        "pattern_scale": "Scale the standard BRISK keypoint sampling pattern.",
        "max_features": "Keep up to this many of the strongest detected features.",
    }

    def process(self, inputs):
        gray = gray_image(inputs["image"], (np.uint8,), auto_depth=True)
        p = self.parameters
        scale = float(p["pattern_scale"])
        detector = cv2.xfeatures2d.BRISK_create(int(p["threshold"]), int(p["octaves"]), scale)
        result = feature_result(
            inputs,
            gray,
            detector,
            "BRISK",
            {"descriptor_size": 64, "pattern_scale": scale},
            limit=int(p["max_features"]),
        )
        if self.generate_graphics and result.outputs["points"]:
            result.drawings["BRISK"] = GraphicDrawing(
                tuple(
                    graphic_object("marker", [point], (70, 220, 140))
                    for point in result.outputs["points"]
                )
            )
        return result


def brisk_data(data):
    if (
        not isinstance(data, dict)
        or data.get("algorithm") != "BRISK"
        or data.get("descriptor_size") != 64
    ):
        raise ValidationError("Connect standard 64-byte BRISK descriptors")
    scale = data.get("pattern_scale")
    if type(scale) not in (int, float) or not 0.5 <= scale <= 4:
        raise ValidationError("Invalid BRISK sampling pattern scale")
    raw, points = descriptor_arrays(data, 64, 0, 255, True)
    return raw.astype(np.uint8), points, scale


class MatchBRISK(Block):
    type_id, title, category = "inspect.match_brisk", "Match BRISK descriptors", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs, outputs = MatchSIFT.inputs, MatchSIFT.outputs
    defaults = {"ratio": 0.75, "max_distance": 100.0, "mode": "ratio"}
    parameter_choices = MatchAKAZE.parameter_choices
    parameter_ranges = MatchSIFT.parameter_ranges
    parameter_help = MatchSIFT.parameter_help

    def process(self, inputs):
        query, qpoints, query_scale = brisk_data(inputs["query"])
        train, tpoints, train_scale = brisk_data(inputs["train"])
        if query_scale != train_scale:
            raise ValidationError("BRISK sampling pattern scales must match")
        p = self.parameters
        return descriptor_match_result(
            inputs,
            query,
            train,
            qpoints,
            tpoints,
            cv2.NORM_HAMMING,
            p["mode"],
            float(p["ratio"]),
            float(p["max_distance"]),
        )


def radius_data(data):
    family = data.get("algorithm") if isinstance(data, dict) else None
    if family == "ORB":
        descriptors, points = orb_data(data)
        wta = data.get("wta_k", 2)
        norm = cv2.NORM_HAMMING if wta == 2 else cv2.NORM_HAMMING2
        return descriptors, points, (family, wta), norm
    if family == "BRISK":
        descriptors, points, scale = brisk_data(data)
        return descriptors, points, (family, scale), cv2.NORM_HAMMING
    if family == "AKAZE":
        descriptors, points, settings = akaze_data(data)
        norm = cv2.NORM_HAMMING if descriptors.dtype == np.uint8 else cv2.NORM_L2
        return descriptors, points, (family, *settings), norm
    if family == "SIFT":
        descriptors, points = sift_data(data)
        return descriptors, points, (family,), cv2.NORM_L2
    if family == "KAZE":
        descriptors, points, layout = kaze_data(data)
        return descriptors, points, layout, cv2.NORM_L2
    raise ValidationError("Connect ORB, BRISK, AKAZE, SIFT, or KAZE feature descriptors")


class MatchDescriptorRadius(Block):
    type_id = "inspect.match_descriptor_radius"
    title, category = "Match descriptors within radius", "Inspection"
    block_type, subtype = "Inspection", "Matching"
    inputs, outputs = MatchSIFT.inputs, MatchSIFT.outputs
    defaults = {"max_distance": 32.0, "max_neighbors": 4}
    parameter_ranges = {
        "max_distance": NumericRange(0, 65536),
        "max_neighbors": NumericRange(1, 32, whole=True),
    }
    parameter_help = {
        "max_distance": "Inclusive descriptor distance, in Hamming, Hamming2, or L2 units.",
        "max_neighbors": "Maximum candidates per query keypoint, ordered by distance.",
    }

    def process(self, inputs):
        query, qpoints, layout, norm = radius_data(inputs["query"])
        train, tpoints, train_layout, _ = radius_data(inputs["train"])
        if layout != train_layout:
            raise ValidationError("Query and train need the same descriptor family and settings")
        if max(len(query), len(train)) > 10_000:
            raise ValidationError("Radius matching supports up to 10,000 descriptors per input")
        if len(query) * len(train) > 10_000_000:
            raise ValidationError("Radius matching supports up to 10 million descriptor pairs")
        selected = []
        if len(query) and len(train):
            radius = float(self.parameters["max_distance"])
            limit = int(self.parameters["max_neighbors"])
            matcher = cv2.BFMatcher(norm)
            # Bound the number of candidates OpenCV can materialize per call.
            batch_size = max(1, 100_000 // len(train))
            for start in range(0, len(query), batch_size):
                rows = matcher.radiusMatch(
                    query[start : start + batch_size], train, max(radius, 1e-6)
                )
                for row in rows:
                    candidates = sorted(row, key=lambda match: (match.distance, match.trainIdx))
                    for match in candidates[:limit]:
                        if match.distance <= radius:
                            match.queryIdx += start
                            selected.append(match)
        return paired_feature_result(inputs, qpoints, tpoints, selected)


class EstimateHomography(Block):
    type_id, title, category = "inspect.homography", "Estimate image homography", "Inspection"
    block_type, subtype = "Inspection", "Registration"
    inputs = (Port("matches", "data"),)
    outputs = (Port("transform", "data"), Port("inliers", "number"))
    defaults = {
        "ransac_pixels": 3.0,
        "min_inliers": 4,
        "min_inlier_ratio": 0.5,
        "method": "ransac",
        "max_iterations": 2000,
        "confidence": 0.995,
    }
    parameter_choices = {"method": ("direct", "ransac", "lmeds", "rho")}

    def process(self, inputs):
        data = inputs["matches"]
        query, train = (np.asarray(data[k], np.float64) for k in ("query_points", "train_points"))
        if (
            query.ndim != 2
            or query.shape[1] != 2
            or len(query) < 4
            or train.shape != query.shape
            or not np.isfinite(query).all()
            or not np.isfinite(train).all()
        ):
            raise ValueError("Homography needs at least four finite paired image points")
        threshold = finite(self.parameters["ransac_pixels"], "RANSAC threshold", 0)
        fraction = finite(self.parameters["min_inlier_ratio"], "Minimum inlier ratio", 0)
        required = integer(self.parameters["min_inliers"], "Minimum inliers", 4, 100000)
        if threshold <= 0 or fraction > 1:
            raise ValueError("RANSAC threshold must be positive and inlier ratio at most one")
        confidence = finite(self.parameters["confidence"], "Confidence", 0)
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be strictly between zero and one")
        method = {"direct": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "rho": cv2.RHO}[
            self.parameters["method"]
        ]
        matrix, mask = cv2.findHomography(
            query,
            train,
            method,
            threshold,
            maxIters=integer(self.parameters["max_iterations"], "Maximum iterations", 1, 100000),
            confidence=confidence,
        )
        if matrix is None or mask is None:
            raise ValueError("Cannot estimate a nondegenerate image homography")
        matrix = matrix_value(matrix, (3, 3))
        valid = mask.reshape(-1).astype(bool)
        count = int(valid.sum())
        if count < 4:
            raise ValueError("Homography requires at least four RANSAC inliers")
        projected = cv2.perspectiveTransform(query[valid, None, :], matrix).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - train[valid]) ** 2, axis=1))))
        if not np.isfinite(error):
            raise ValueError("Homography produced a nonfinite reprojection error")
        score = count / len(query)
        result = {
            "model": "image_homography_v1",
            "source_to_target": matrix.tolist(),
            "source_image_size": data.get("query_size"),
            "target_image_size": data.get("train_size"),
            "inlier_mask": valid.tolist(),
            "rms_pixels": error,
            "inlier_ratio": score,
        }
        return Result(
            {"transform": result, "inliers": count},
            passed=count >= required and score >= fraction,
            score=score,
            message=f"{count}/{len(query)} inliers · RMS {error:.3g} px",
        )


class TransformPoints(Block):
    type_id, title, category = "cv.transform_points", "Transform image points", "Image processing"
    block_type, subtype = "Image processing", "Geometry"
    inputs = (
        Port("transform", "data", required=False, required_when_connected=True),
        Port("points", "data", required=False),
    )
    outputs = (Port("points", "data"),)
    defaults = {
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pixel_points": [[0.0, 0.0]],
        "invert": False,
    }

    def process(self, inputs):
        payload = inputs.get("transform", self.parameters["matrix"])
        if isinstance(payload, dict) and "output_to_input" in payload:
            payload = payload["output_to_input"]
        matrix = matrix_value(payload, (3, 3))
        if self.parameters["invert"]:
            matrix = np.linalg.inv(matrix)
        points = np.asarray(inputs.get("points", self.parameters["pixel_points"]), np.float64)
        if points.size == 0:
            points = points.reshape(0, 2)
        elif points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Points must be finite [x,y] pairs")
        denominators = np.c_[points, np.ones(len(points))] @ matrix[2]
        if np.any(np.abs(denominators) < 1e-12):
            raise ValueError("A point maps to infinity")
        output = (
            cv2.perspectiveTransform(points[:, None, :], matrix).reshape(-1, 2)
            if len(points)
            else np.empty((0, 2))
        )
        if not np.isfinite(output).all():
            raise ValueError("Point transform produced nonfinite coordinates")
        context = self.input_payloads.get("points", {})
        target_size = None
        if isinstance(inputs.get("transform"), dict):
            transform = inputs["transform"]
            if "output_to_input" in transform:
                target_size = transform.get(
                    "output_size" if self.parameters["invert"] else "input_size"
                )
            else:
                target_size = transform.get(
                    "source_size" if self.parameters["invert"] else "target_size"
                )
        if context.get("spatial") and target_size is not None:
            context = moved_payload(context, np.linalg.inv(matrix), target_size)
        elif context.get("spatial"):
            context = {
                "coordinate_mapping": {
                    "source": context["spatial"],
                    "source_to_target": matrix.tolist(),
                }
            }
        if isinstance(inputs.get("transform"), dict):
            target_payload = inputs["transform"].get(
                "source_payload" if self.parameters["invert"] else "target_payload"
            )
            if target_payload is not None:
                context = target_payload
        return Result({"points": output.tolist()}, payloads={"points": context})


REGISTRATION_BLOCKS = (
    ControlPointTransform,
    WarpAffine,
    WarpPerspective,
    Rotate,
    CopyBorder,
    Pyramid,
    PyramidDown,
    PyramidUp,
    Filter2D,
    UnsharpMask,
    FASTFeatures,
    ORBFeatures,
    SIFTFeatures,
    AKAZEFeatures,
    KAZEFeatures,
    BRISKFeatures,
    MatchORB,
    MatchSIFT,
    MatchAKAZE,
    MatchFLANN,
    MatchBRISK,
    MatchDescriptorRadius,
    EstimateHomography,
    TransformPoints,
)
