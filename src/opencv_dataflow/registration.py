"""Geometric transforms, feature registration and additional native filters."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .parameters import BORDERS as FILTER_BORDERS
from .parameters import OUTPUT_DEPTHS, anchor_value, optional_mask
from .processing import (
    ImageTool,
    encoded_result,
    finite,
    geometry_result,
    gray_image,
    image_value,
    integer,
)

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


class WarpAffine(ImageTool):
    type_id, title, subtype = "cv.warp_affine", "Warp affine", "Geometry"
    inputs = (Port("image"), Port("transform", "data", required=False))
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
        direction = self.parameters["direction"]
        if direction not in self.parameter_choices["direction"]:
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
        return geometry_result(image, source, [[factor, 0, 0], [0, factor, 0], [0, 0, 1]])


class Filter2D(ImageTool):
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


class UnsharpMask(ImageTool):
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
    inputs = (Port("image"), Port("mask", required=False))
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
    inputs = (Port("image"), Port("mask", required=False))
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
        keypoints, descriptors = detector.detectAndCompute(gray, optional_mask(inputs, gray))
        points = [list(k.pt) for k in keypoints]
        features = {
            "algorithm": "ORB",
            "descriptor_size": 32,
            "wta_k": int(p["wta_k"]),
            "image_size": [gray.shape[1], gray.shape[0]],
            "keypoints": [
                {
                    "point": list(k.pt),
                    "size": k.size,
                    "angle": k.angle,
                    "response": k.response,
                    "octave": k.octave,
                }
                for k in keypoints
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
            graphics={"ORB": overlay},
        )


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
        selected = []
        wta = inputs["query"].get("wta_k", 2)
        if inputs["train"].get("wta_k", 2) != wta:
            raise ValueError("ORB descriptor WTA_K values must match")
        expected_norm = "hamming" if wta == 2 else "hamming2"
        norm_name = expected_norm if self.parameters["norm"] == "auto" else self.parameters["norm"]
        if norm_name != expected_norm:
            raise ValueError("Matcher norm is incompatible with ORB WTA_K")
        norm = cv2.NORM_HAMMING if norm_name == "hamming" else cv2.NORM_HAMMING2
        if self.parameters["mode"] == "cross_check" and len(query) and len(train):
            selected = [
                m
                for m in cv2.BFMatcher(norm, crossCheck=True).match(query, train)
                if m.distance <= maximum
            ]
        elif len(query) and len(train) >= 2:
            pairs = cv2.BFMatcher(norm).knnMatch(query, train, k=2)
            selected = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < ratio * pair[1].distance
                and pair[0].distance <= maximum
            ]
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
            message=f"{len(selected)} ratio-filtered matches",
        )


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
    inputs = (Port("transform", "data", required=False), Port("points", "data", required=False))
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
            return Result({"points": []})
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Points must be finite [x,y] pairs")
        denominators = np.c_[points, np.ones(len(points))] @ matrix[2]
        if np.any(np.abs(denominators) < 1e-12):
            raise ValueError("A point maps to infinity")
        output = cv2.perspectiveTransform(points[:, None, :], matrix).reshape(-1, 2)
        if not np.isfinite(output).all():
            raise ValueError("Point transform produced nonfinite coordinates")
        return Result({"points": output.tolist()})


REGISTRATION_BLOCKS = (
    WarpAffine,
    WarpPerspective,
    Rotate,
    CopyBorder,
    Pyramid,
    Filter2D,
    UnsharpMask,
    FASTFeatures,
    ORBFeatures,
    MatchORB,
    EstimateHomography,
    TransformPoints,
)
