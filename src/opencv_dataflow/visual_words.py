"""Portable image-level visual vocabularies and exact, bounded gallery retrieval."""

import copy
import json

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    model_revision,
    prediction_drawing,
    validate_prediction_image,
)
from .clustering import cluster_colors
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_circle
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, metadata
from .processing import gray_image
from .projection import profile_plot
from .selection import contains, selection_mask

CODEBOOK = "opencv_visual_words_v1"
GALLERY = "opencv_visual_gallery_v1"
FEATURES = "opencv_visual_features_v1"
MAX_BYTES = 8 * 1024**2
FEATURE_DEFAULTS = {
    "method": "sift",
    "max_features": 512,
    "sift_octave_layers": 3,
    "sift_contrast_threshold": 0.04,
    "sift_edge_threshold": 10.0,
    "sift_sigma": 1.6,
    "sift_precise_upscale": False,
    "orb_scale_factor": 1.2,
    "orb_levels": 8,
    "orb_edge_threshold": 31,
    "orb_first_level": 0,
    "orb_wta_k": "2",
    "orb_score": "harris",
    "orb_patch_size": 31,
    "orb_fast_threshold": 20,
}
LIMITS = {"max_work": 128_000_000, "max_bytes_mb": 128.0}
LIMIT_RANGES = {
    "max_work": NumericRange(1, 1_000_000_000, whole=True),
    "max_bytes_mb": NumericRange(0.01, 4096),
}
LIMIT_HELP = {
    "max_work": "Estimated descriptor comparisons/fit work before native calls.",
    "max_bytes_mb": "Estimated numeric workspace allowance, not process RSS.",
}


def contract(p):
    return {
        "model": "opencv_visual_extractor_v1",
        "method": p["method"],
        "settings": {
            key: value
            for key, value in p.items()
            if key == "max_features" or key.startswith(p["method"] + "_")
        },
        "opencv_version": cv2.__version__,
    }


def read_contract(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "method", "settings", "opencv_version"}
        or value.get("method") not in ("sift", "orb")
        or not isinstance(value.get("settings"), dict)
    ):
        raise ValidationError("Invalid visual descriptor contract")
    p = {**FEATURE_DEFAULTS, **value["settings"], "method": value["method"]}
    validator = VisualFeatures(parameters=p)
    try:
        validator.validate_parameters()
        if contract(p) != value:
            raise ValidationError("Visual descriptor settings or OpenCV version differ")
    finally:
        validator.close()
    return p


def descriptor_spec(extractor):
    if extractor["method"] == "sift":
        return 128, np.float32, cv2.NORM_L2, "l2"
    two = extractor["settings"]["orb_wta_k"] == "2"
    return (
        32,
        np.uint8,
        cv2.NORM_HAMMING if two else cv2.NORM_HAMMING2,
        "hamming" if two else "hamming2",
    )


def allowance(p, work, bytes_required):
    if work > p["max_work"]:
        raise ValidationError("Visual-word operation exceeds Max work", "max_work")
    if bytes_required > p["max_bytes_mb"] * 1024**2:
        raise ValidationError("Visual-word operation exceeds workspace allowance", "max_bytes_mb")


def check_feature_context(value, context):
    identity = context.get("feature_contract")
    if identity is not None and identity != value["extractor"]:
        raise ValidationError("Descriptor packet identity differs from its feature record")


def read_features(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "extractor", "image_size", "points", "descriptors"}
        or value.get("model") != FEATURES
    ):
        raise ValidationError("Connect visual local features")
    read_contract(value["extractor"])
    d, dtype, _, _ = descriptor_spec(value["extractor"])
    size = value["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(x) is not int or not 1 <= x <= 16384 for x in size)
        or size[0] * size[1] > 1_000_000
        or not isinstance(value["points"], list)
        or len(value["points"]) > 4096
        or not isinstance(value["descriptors"], list)
        or len(value["descriptors"]) != len(value["points"])
        or any(not isinstance(row, list) or len(row) != d for row in value["descriptors"])
        or any(not isinstance(row, list) or len(row) != 2 for row in value["points"])
    ):
        raise ValidationError("Visual feature image/row bounds exceeded")
    try:
        points = np.asarray(value["points"], np.float64).reshape(-1, 2)
        wide = np.asarray(value["descriptors"], np.float64).reshape(-1, d)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Invalid visual feature arrays") from exc
    if (
        len(points) != len(wide)
        or len(wide) > 4096
        or not np.isfinite(points).all()
        or not np.isfinite(wide).all()
        or np.any(points < 0)
        or np.any(points >= np.array(size))
        or np.any(wide < 0)
        or np.any(wide > (255 if dtype == np.uint8 else 1e6))
    ):
        raise ValidationError("Visual descriptor values or point coordinates are invalid")
    descriptors = np.ascontiguousarray(wide, dtype=dtype)
    if not np.array_equal(wide, descriptors.astype(np.float64)):
        raise ValidationError("Descriptors must retain exact native float32 or byte values")
    return points, descriptors


def read_model(value):
    if not isinstance(value, dict) or value.get("model") not in (CODEBOOK, GALLERY):
        raise ValidationError("Connect a visual-word codebook or gallery model")
    if len(canonical_json(value).encode("utf-8")) > MAX_BYTES:
        raise ValidationError("Visual-word model exceeds 8 MiB")
    if value.get("revision") != model_revision(value):
        raise ValidationError("Visual-word model revision differs")
    if value["model"] == CODEBOOK:
        if set(value) != {"model", "extractor", "centers", "metric", "training", "revision"}:
            raise ValidationError("Visual-word codebook fields differ")
        read_contract(value["extractor"])
        d, dtype, _, metric = descriptor_spec(value["extractor"])
        try:
            wide = np.asarray(value["centers"], np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValidationError("Visual words must be numeric") from exc
        if (
            wide.ndim != 2
            or not 2 <= len(wide) <= 128
            or wide.shape[1] != d
            or not np.isfinite(wide).all()
            or np.any(wide < 0)
            or np.any(wide > (255 if dtype == np.uint8 else 1e6))
            or value["metric"] != metric
        ):
            raise ValidationError("Visual word dimensions, values or metric differ")
        centers = np.ascontiguousarray(wide, dtype=dtype)
        if not np.array_equal(wide, centers.astype(np.float64)) or len(
            np.unique(centers, axis=0)
        ) != len(centers):
            raise ValidationError("Visual words must be distinct exact native descriptors")
        training = value["training"]
        if (
            not isinstance(training, dict)
            or set(training) != {"images", "descriptors", "fit"}
            or type(training["images"]) is not int
            or not 1 <= training["images"] <= 64
            or type(training["descriptors"]) is not int
            or not len(centers) <= training["descriptors"] <= 8192
            or training["fit"]
            != ("native_kmeans" if dtype == np.float32 else "farthest_first_medoids")
        ):
            raise ValidationError("Invalid visual-word training metadata")
        return centers
    if set(value) != {"model", "feature_contract", "entries", "revision"}:
        raise ValidationError("Visual-word gallery fields differ")
    read_histogram_contract(value["feature_contract"])
    entries = value["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1024:
        raise ValidationError("Gallery needs 1..1024 entries")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"label", "histogram", "source"}
            or not isinstance(entry["label"], str)
            or not 1 <= len(entry["label"]) <= 256
            or not isinstance(entry["source"], dict)
            or len(canonical_json(entry["source"]).encode("utf-8")) > 16384
        ):
            raise ValidationError("Invalid gallery label/source metadata")
        read_histogram(entry["histogram"], value["feature_contract"], allow_empty=False)
    return None


def seal(value):
    value["revision"] = model_revision(value)
    read_model(value)
    return value


def read_histogram_contract(value):
    if (
        not isinstance(value, dict)
        or set(value)
        != {"model", "codebook_revision", "words", "normalization", "max_word_distance"}
        or value.get("model") != "opencv_visual_histogram_v1"
        or not isinstance(value["codebook_revision"], str)
        or len(value["codebook_revision"]) != 64
        or any(c not in "0123456789abcdef" for c in value["codebook_revision"])
        or type(value["words"]) is not int
        or not 2 <= value["words"] <= 128
        or value["normalization"] not in ("l1", "l2", "hellinger")
        or type(value["max_word_distance"]) not in (int, float)
        or not np.isfinite(value["max_word_distance"])
        or value["max_word_distance"] < -1
    ):
        raise ValidationError("Invalid visual histogram contract")
    return value


def read_histogram(value, identity, *, allow_empty=True):
    read_histogram_contract(identity)
    try:
        array = np.asarray(value, np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Visual histogram must be numeric") from exc
    if (
        array.shape != (identity["words"],)
        or not np.isfinite(array).all()
        or np.any(array < 0)
        or np.any(array > 1 + 1e-6)
    ):
        raise ValidationError("Visual histogram has invalid dimensions or values")
    total = (
        float(array.sum()) if identity["normalization"] == "l1" else float(np.linalg.norm(array))
    )
    if total == 0 and allow_empty:
        return array
    if not np.isclose(total, 1, rtol=1e-6, atol=1e-8):
        raise ValidationError("Gallery/query histogram must have the declared unit norm")
    return array


def assignments(rows, centers, norm):
    matches = cv2.BFMatcher(norm).match(rows, centers)
    if len(matches) != len(rows):
        raise ValidationError("Native matcher did not assign every descriptor")
    ids, distances = np.empty(len(rows), np.int32), np.empty(len(rows), np.float64)
    for match in matches:
        ids[match.queryIdx], distances[match.queryIdx] = match.trainIdx, match.distance
    return ids, distances


class _Words(Block):
    category, block_type, subtype = "ML", "ML", "Image retrieval"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True


class VisualFeatures(_Words):
    type_id, title = "ml.visual_features", "Visual local features"
    inputs = (
        Port("image"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("features", "data"), Port("count", "number"), Port("image"))
    defaults = FEATURE_DEFAULTS
    parameter_choices = {
        "method": ("sift", "orb"),
        "orb_wta_k": ("2", "3", "4"),
        "orb_score": ("harris", "fast"),
    }
    parameter_ranges = {
        "max_features": NumericRange(8, 4096, whole=True),
        "sift_octave_layers": NumericRange(1, 8, whole=True),
        "sift_contrast_threshold": NumericRange(0, 1),
        "sift_edge_threshold": NumericRange(0.1, 1000),
        "sift_sigma": NumericRange(0.1, 10),
        "orb_scale_factor": NumericRange(1.05, 2),
        "orb_levels": NumericRange(1, 16, whole=True),
        "orb_first_level": NumericRange(0, 15, whole=True),
        "orb_edge_threshold": NumericRange(0, 127, whole=True),
        "orb_patch_size": NumericRange(3, 127, whole=True, odd=True),
        "orb_fast_threshold": NumericRange(0, 255, whole=True),
    }
    parameter_help = {
        "method": "SIFT float descriptors use L2; ORB bytes use Hamming (Hamming2 for WTA 3/4).",
        "max_features": "Hard retained feature bound; points are sorted by response and geometry.",
        "sift_precise_upscale": "Native precise upscale; saved in descriptor identity.",
        "orb_first_level": "Must be smaller than ORB levels.",
        "orb_patch_size": "Odd patch size; selection limits centers, not patch support.",
    }
    input_help = {
        "image": "Up to one million pixels. Fixed-scale gray conversion; original image retained.",
        "mask": "Selected keypoint centers only; descriptor patches may read beyond selection.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if type(p["sift_precise_upscale"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "sift_precise_upscale")
        if p["orb_first_level"] >= p["orb_levels"]:
            raise ValidationError(
                "First level must be smaller than levels", "orb_first_level", "orb_levels"
            )

    def process(self, inputs):
        image, p = inputs["image"], self.parameters
        if max(image.shape[:2]) > 16384 or image.shape[0] * image.shape[1] > 1_000_000:
            raise ValidationError("Visual features are limited to one million image pixels")
        if image.dtype.kind == "f" and (np.any(image < 0) or np.any(image > 1)):
            raise ValidationError("Floating images must use fixed [0,1] intensity units")
        gray = gray_image(image, (np.uint8,), auto_depth=True)
        if p["method"] == "orb":
            pyramid_pixels = gray.size * sum(
                p["orb_scale_factor"] ** (2 * (p["orb_first_level"] - level))
                for level in range(p["orb_levels"])
            )
            if pyramid_pixels > 16_000_000:
                raise ValidationError("ORB pyramid exceeds 16 million estimated pixels")
        mask = selection_mask(inputs, gray)
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = np.uint8(image[..., 3] > 0) * 255
            mask = alpha if mask is None else cv2.bitwise_and(mask, alpha)
        if p["method"] == "sift":
            detector = cv2.SIFT_create(
                nfeatures=p["max_features"],
                nOctaveLayers=p["sift_octave_layers"],
                contrastThreshold=p["sift_contrast_threshold"],
                edgeThreshold=p["sift_edge_threshold"],
                sigma=p["sift_sigma"],
                enable_precise_upscale=p["sift_precise_upscale"],
            )
        else:
            detector = cv2.ORB_create(
                nfeatures=p["max_features"],
                scaleFactor=p["orb_scale_factor"],
                nlevels=p["orb_levels"],
                edgeThreshold=p["orb_edge_threshold"],
                firstLevel=p["orb_first_level"],
                WTA_K=int(p["orb_wta_k"]),
                scoreType=cv2.ORB_HARRIS_SCORE
                if p["orb_score"] == "harris"
                else cv2.ORB_FAST_SCORE,
                patchSize=p["orb_patch_size"],
                fastThreshold=p["orb_fast_threshold"],
            )
        keys, descriptors = detector.detectAndCompute(gray, mask)
        selected = sorted(
            (i for i, k in enumerate(keys) if contains(mask, k.pt)),
            key=lambda i: (
                -keys[i].response,
                *keys[i].pt,
                keys[i].size,
                keys[i].angle,
                keys[i].octave,
            ),
        )[: p["max_features"]]
        identity = contract(p)
        d, dtype, _, _ = descriptor_spec(identity)
        rows = descriptors[selected] if selected else np.empty((0, d), dtype)
        features = {
            "model": FEATURES,
            "extractor": identity,
            "image_size": [gray.shape[1], gray.shape[0]],
            "points": [list(keys[i].pt) for i in selected],
            "descriptors": rows.tolist(),
        }
        read_features(features)
        drawings = {}
        if self.generate_graphics:
            drawings["Local feature centers"] = GraphicDrawing(
                tuple(
                    graphic_circle(*point, 2, (40, 210, 250)) for point in features["points"][:256]
                )
            )
        return Result(
            {"features": features, "count": len(rows), "image": image},
            drawings=drawings,
            payloads={"features": {**self.input_payloads["image"], "feature_contract": identity}},
            message=f"{len(rows)} {p['method'].upper()} descriptors; selection limits centers",
        )


class TrainVisualWords(_Words):
    type_id, title = "ml.bow_train", "Train visual words"
    inputs, outputs = (Port("features", "any"),), (Port("model", "data"), Port("report", "data"))
    defaults = {
        "words": 16,
        "max_per_image": 512,
        "max_samples": 2048,
        "iterations": 20,
        "epsilon": 0.01,
        **LIMITS,
    }
    parameter_ranges = {
        "words": NumericRange(2, 128, whole=True),
        "max_per_image": NumericRange(8, 4096, whole=True),
        "max_samples": NumericRange(8, 8192, whole=True),
        "iterations": NumericRange(1, 100, whole=True),
        "epsilon": NumericRange(0),
        **LIMIT_RANGES,
    }
    parameter_help = {
        "words": "Requires at least this many distinct sampled descriptors.",
        "max_per_image": "Even deterministic row sampling before global Max samples.",
        "iterations": "SIFT native k-means after deterministic initialization. "
        "ORB retains farthest-first actual byte descriptors.",
        "epsilon": "SIFT native k-means center-movement tolerance; unused for ORB.",
        **LIMIT_HELP,
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["words"] > p["max_samples"]:
            raise ValidationError("Words cannot exceed Max samples", "words", "max_samples")

    def process(self, inputs):
        raw, p = inputs["features"], self.parameters
        items = raw.items if isinstance(raw, FlowGroup) else (raw,)
        if not 1 <= len(items) <= 64:
            raise ValidationError("Train with 1..64 image-feature records")
        rows, identity, seen = [], None, 0
        for item in items:
            value = item.value if hasattr(item, "value") else item
            _, table = read_features(value)
            check_feature_context(
                value,
                metadata(item) if isinstance(raw, FlowGroup) else self.input_payloads["features"],
            )
            current = value["extractor"]
            if identity is not None and identity != current:
                raise ValidationError("Training descriptor settings differ")
            identity = current
            seen += len(table)
            if seen > 32768:
                raise ValidationError("Training input exceeds 32,768 descriptors")
            if len(table):
                rows.append(
                    table[
                        np.linspace(
                            0, len(table) - 1, min(len(table), p["max_per_image"]), dtype=int
                        )
                    ]
                )
        if not rows:
            raise ValidationError("Training images contain no descriptors")
        values = np.concatenate(rows)
        values = values[
            np.linspace(0, len(values) - 1, min(len(values), p["max_samples"]), dtype=int)
        ]
        count, words, d = len(values), p["words"], values.shape[1]
        _, dtype, norm, metric = descriptor_spec(identity)
        steps = p["iterations"] + 1 if dtype == np.float32 else 1
        allowance(p, count * words * d * steps, values.nbytes * 4 + count * words * 8)
        centers, closest = [values[0]], np.full(count, np.inf)
        for _ in range(words):
            _, distances = assignments(values, np.asarray(centers[-1:], dtype=dtype), norm)
            closest = np.minimum(closest, distances)
            if len(centers) == words:
                break
            index = int(np.argmax(closest))
            if closest[index] == 0:
                raise ValidationError("Need more distinct descriptors or fewer visual words")
            centers.append(values[index])
        centers = np.ascontiguousarray(centers, dtype=dtype)
        compactness = None
        if dtype == np.float32:
            ids, _ = assignments(values, centers, norm)
            compactness, _, centers = cv2.kmeans(
                values,
                words,
                ids[:, None],
                (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, p["iterations"], p["epsilon"]),
                1,
                cv2.KMEANS_USE_INITIAL_LABELS,
            )
        centers = centers[sorted(range(words), key=lambda i: tuple(centers[i]))]
        fit = "native_kmeans" if dtype == np.float32 else "farthest_first_medoids"
        model = seal(
            {
                "model": CODEBOOK,
                "extractor": identity,
                "centers": centers.tolist(),
                "metric": metric,
                "training": {"images": len(items), "descriptors": count, "fit": fit},
            }
        )
        return Result(
            {
                "model": model,
                "report": {
                    "images": len(items),
                    "input_descriptors": seen,
                    "sampled_descriptors": count,
                    "words": words,
                    "metric": metric,
                    "fit": fit,
                    "compactness": compactness,
                },
            },
            message=f"Trained {words} {metric} visual words from {count} sampled descriptors",
        )


class EncodeVisualWords(_Words):
    type_id, title = "ml.bow_encode", "Image visual-word histogram"
    inputs = (
        Port("model", "data"),
        Port("features", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("histogram", "data"),
        Port("word_ids", "data"),
        Port("distances", "data"),
        Port("count", "number"),
        Port("image", required=False),
        Port("report", "data"),
    )
    defaults = {"normalization": "l1", "max_word_distance": -1.0, **LIMITS}
    parameter_choices = {"normalization": ("l1", "l2", "hellinger")}
    parameter_ranges = {"max_word_distance": NumericRange(-1), **LIMIT_RANGES}
    parameter_help = {
        "normalization": "L1 frequency, L2 unit norm, or square-root L1 (Hellinger) signature.",
        "max_word_distance": "-1 disables rejection; otherwise use the inclusive native distance "
        "limit. Rejected IDs are -1.",
        **LIMIT_HELP,
    }
    payload_reference = "features"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if -1 < p["max_word_distance"] < 0:
            raise ValidationError("Use -1 or a nonnegative distance", "max_word_distance")

    def process(self, inputs):
        model, features, p = inputs["model"], inputs["features"], self.parameters
        centers = read_model(model)
        if model["model"] != CODEBOOK:
            raise ValidationError("Encoding requires a codebook, not a gallery")
        points, values = read_features(features)
        check_feature_context(features, self.input_payloads["features"])
        if features["extractor"] != model["extractor"]:
            raise ValidationError("Image descriptor settings differ from the codebook")
        image = validate_prediction_image(self, inputs, 1, samples_port="features")
        if image is not NO_VALUE and [image.shape[1], image.shape[0]] != features["image_size"]:
            raise ValidationError("Review image size differs from feature image")
        allowance(p, values.size * len(centers), values.nbytes * 4 + len(values) * len(centers) * 8)
        _, _, norm, _ = descriptor_spec(model["extractor"])
        ids, distances = (
            assignments(values, centers, norm)
            if len(values)
            else (np.empty(0, np.int32), np.empty(0))
        )
        accepted = (
            np.ones(len(ids), bool)
            if p["max_word_distance"] < 0
            else distances <= p["max_word_distance"]
        )
        ids[~accepted] = -1
        counts = np.bincount(ids[accepted], minlength=len(centers)).astype(np.float64)
        if counts.sum():
            counts /= counts.sum() if p["normalization"] != "l2" else np.linalg.norm(counts)
            if p["normalization"] == "hellinger":
                counts = np.sqrt(counts)
        identity = {
            "model": "opencv_visual_histogram_v1",
            "codebook_revision": model["revision"],
            "words": len(centers),
            "normalization": p["normalization"],
            "max_word_distance": p["max_word_distance"],
        }
        payload = {**self.input_payloads["features"], "feature_contract": identity}
        graphics, drawings = {}, {}
        if self.generate_graphics:
            graphics["Visual word frequency"] = profile_plot(
                counts[:, None],
                np.ones(len(counts), bool),
                ("frequency",),
                "visual word",
                p["normalization"],
            )
            if image is not NO_VALUE:
                palette = cluster_colors(len(centers))
                drawings["Assigned feature centers"] = GraphicDrawing(
                    tuple(
                        graphic_circle(*point, 3, tuple(map(int, palette[word])))
                        for point, word in zip(points[:256], ids[:256], strict=True)
                        if word >= 0
                    )
                )
        payloads = {"histogram": payload}
        if image is not NO_VALUE:
            payloads["image"] = copy.deepcopy(self.input_payloads["image"])
        return Result(
            {
                "histogram": counts.tolist(),
                "word_ids": ids.tolist(),
                "distances": distances.tolist(),
                "count": int(accepted.sum()),
                "image": image,
                "report": {
                    "descriptors": len(ids),
                    "accepted": int(accepted.sum()),
                    "rejected": int((~accepted).sum()),
                    "codebook_revision": model["revision"],
                },
            },
            payloads=payloads,
            graphics=graphics,
            drawings=drawings,
            passed=bool(accepted.any()),
            fail_reasons=[] if accepted.any() else ["no_visual_words"],
            message=f"{int(accepted.sum())}/{len(ids)} descriptors, {len(centers)} visual words",
        )


class BuildVisualGallery(_Words):
    type_id, title = "ml.bow_gallery", "Build visual-word gallery"
    inputs, outputs = (Port("histograms", "any"),), (Port("model", "data"), Port("count", "number"))
    defaults = {"labels": []}
    parameter_help = {
        "labels": "Optional label per signature; otherwise batch labels or entry numbers. "
        "Duplicate labels can represent multiple views."
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        labels = p["labels"]
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or not 1 <= len(label) <= 256 for label in labels
        ):
            raise ValidationError(
                "Labels must be nonempty strings of at most 256 characters", "labels"
            )

    def process(self, inputs):
        raw = inputs["histograms"]
        items = raw.items if isinstance(raw, FlowGroup) else (raw,)
        if not 1 <= len(items) <= 1024:
            raise ValidationError("Gallery needs 1..1024 signatures")
        labels = self.parameters["labels"]
        if not isinstance(labels, list) or (labels and len(labels) != len(items)):
            raise ValidationError("Provide one label per signature or an empty list", "labels")
        identity, entries = None, []
        for index, item in enumerate(items):
            context = (
                metadata(item) if isinstance(raw, FlowGroup) else self.input_payloads["histograms"]
            )
            current = context.get("feature_contract")
            read_histogram_contract(current)
            if identity is not None and current != identity:
                raise ValidationError("Gallery histogram/codebook contracts differ")
            identity = current
            value = item.value if isinstance(raw, FlowGroup) else item
            histogram = read_histogram(value, identity, allow_empty=False)
            label = (
                labels[index]
                if labels
                else raw.labels[index]
                if isinstance(raw, FlowGroup) and raw.labels
                else str(index)
            )
            entries.append(
                {
                    "label": label,
                    "histogram": histogram.tolist(),
                    "source": {
                        key: copy.deepcopy(context[key])
                        for key in ("ticket", "spatial", "pyramid", "recipe_ref")
                        if key in context
                    },
                }
            )
        model = seal({"model": GALLERY, "feature_contract": identity, "entries": entries})
        return Result(
            {"model": model, "count": len(entries)},
            message=f"Built gallery with {len(entries)} image signatures",
        )


class RankVisualWords(_Words):
    type_id, title = "ml.bow_rank", "Rank visual-word images"
    inputs = (
        Port("query", "data"),
        Port("gallery", "data"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("results", "data"),
        Port("label", "text"),
        Port("distance", "number", required=False),
        Port("count", "number"),
        Port("image", required=False),
        Port("report", "data"),
    )
    defaults = {"metric": "cosine", "top_k": 5, "max_distance": -1.0, "minimum_results": 1}
    parameter_choices = {"metric": ("cosine", "l2", "chi_square", "bhattacharyya")}
    parameter_ranges = {
        "top_k": NumericRange(1, 1024, whole=True),
        "max_distance": NumericRange(-1),
        "minimum_results": NumericRange(1, 1024, whole=True),
    }
    parameter_help = {
        "max_distance": "-1 disables rejection; otherwise an inclusive metric-specific limit. "
        "Distance is not confidence.",
        "minimum_results": "Required retained results after distance filtering and Top k.",
    }
    payload_reference = "query"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_results"] > p["top_k"]:
            raise ValidationError("Minimum results cannot exceed Top k", "minimum_results", "top_k")
        if -1 < p["max_distance"] < 0:
            raise ValidationError("Use -1 or a nonnegative distance", "max_distance")

    def process(self, inputs):
        model, p = inputs["gallery"], self.parameters
        read_model(model)
        if model["model"] != GALLERY:
            raise ValidationError("Ranking requires a gallery, not a codebook")
        identity = self.input_payloads["query"].get("feature_contract")
        if identity != model["feature_contract"]:
            raise ValidationError("Query histogram/codebook contract differs from gallery")
        query = read_histogram(inputs["query"], identity)
        image = validate_prediction_image(self, inputs, 1, samples_port="query")
        ranked = []
        if np.any(query):
            for index, entry in enumerate(model["entries"]):
                reference = np.asarray(entry["histogram"], np.float64)
                if p["metric"] == "cosine":
                    distance = float(
                        np.clip(
                            1
                            - (query @ reference)
                            / (np.linalg.norm(query) * np.linalg.norm(reference)),
                            0,
                            1,
                        )
                    )
                elif p["metric"] == "l2":
                    distance = float(cv2.norm(query, reference, cv2.NORM_L2))
                else:
                    mode = (
                        cv2.HISTCMP_CHISQR_ALT
                        if p["metric"] == "chi_square"
                        else cv2.HISTCMP_BHATTACHARYYA
                    )
                    distance = float(
                        cv2.compareHist(
                            query.astype(np.float32), reference.astype(np.float32), mode
                        )
                    )
                if not np.isfinite(distance):
                    raise ValidationError("Histogram distance overflowed")
                if p["max_distance"] < 0 or distance <= p["max_distance"]:
                    ranked.append(
                        {
                            "index": index,
                            "label": entry["label"],
                            "distance": distance,
                            "source": copy.deepcopy(entry["source"]),
                        }
                    )
        ranked.sort(key=lambda row: (row["distance"], row["index"]))
        ranked = ranked[: p["top_k"]]
        passed = len(ranked) >= p["minimum_results"]
        label, distance = (ranked[0]["label"], ranked[0]["distance"]) if ranked else ("", NO_VALUE)
        text = f"{label}: distance {distance:.4g}" if ranked else "No retained image matches"
        return Result(
            {
                "results": ranked,
                "label": label,
                "distance": distance,
                "count": len(ranked),
                "image": image,
                "report": {
                    "gallery_revision": model["revision"],
                    "metric": p["metric"],
                    "gallery_entries": len(model["entries"]),
                    "empty_query": not bool(np.any(query)),
                    "distance_is_probability": False,
                },
            },
            passed=passed,
            fail_reasons=[] if passed else ["insufficient_visual_matches"],
            message=text,
            drawings=prediction_drawing(self, image, text, passed, "Image retrieval"),
            payloads={"image": copy.deepcopy(self.input_payloads["image"])}
            if image is not NO_VALUE
            else {},
        )


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate visual-word model key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValidationError("Nonfinite visual-word model number: " + value)


class SaveVisualWords(_Words):
    type_id, title = "ml.bow_save", "Save visual-word model"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "visual-words.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_model(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result(
            {"path": str(path)}, message="Saved visual-word model; existing files preserved"
        )


class LoadVisualWords(_Words):
    type_id, title = "ml.bow_load", "Load visual-word model"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        with local_path(self, self.parameters["path"]).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("Visual-word model exceeds 8 MiB")
        model = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )
        read_model(model)
        return Result({"model": model}, message="Loaded " + model["model"])


VISUAL_WORD_BLOCKS = (
    VisualFeatures,
    TrainVisualWords,
    EncodeVisualWords,
    BuildVisualGallery,
    RankVisualWords,
    SaveVisualWords,
    LoadVisualWords,
)
