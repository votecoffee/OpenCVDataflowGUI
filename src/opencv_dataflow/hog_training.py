"""Train an oriented native HOG detector with a bounded hard-negative pass."""

import hashlib
from collections import defaultdict

import cv2
import numpy as np

from .classifier_common import canonical_json
from .core import Block, Port, Result
from .hog import HOGFeatures
from .hog_detection import geometry, linear_detector_coefficients, make_model, native_hog
from .images import image_format
from .native_runtime import cpu_execution
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup
from .processing import gray_image, image_value

MAX_CROPS = 512
MAX_BACKGROUNDS = 32
MAX_IMAGE_PIXELS = 1_048_576
MAX_PIXELS = 8_388_608
MAX_TRAIN_WORK = 256_000_000
MAX_MINING_WORK = 128_000_000
SOLVER_FIELDS = ("c", "max_iterations", "epsilon", "positive_weight", "negative_weight")


def images(value, name, maximum, minimum):
    group = value.items if isinstance(value, FlowGroup) else None
    count = len(group) if group is not None else 1
    if not minimum <= count <= maximum:
        raise ValidationError(f"{name} requires {minimum}..{maximum} image entries")
    entries = [packet.value for packet in group] if group is not None else [value]
    for image in entries:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim not in (2, 3)
            or image.dtype != np.uint8
            or not 1 <= image.shape[0] * image.shape[1] <= MAX_IMAGE_PIXELS
            or max(image.shape[:2]) > 8192
            or image_format(image) not in ("gray", "bgr", "rgb")
        ):
            raise ValidationError(f"{name} requires bounded uint8 mono, BGR or RGB images")
    return entries


def working_image(image, encoding):
    if encoding == "gray":
        return np.ascontiguousarray(gray_image(image, (np.uint8,)))
    result = image_value(image, (np.uint8,), canonical=True)
    return (
        cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        if result.ndim == 2
        else np.ascontiguousarray(result)
    )


def pixel_hash(image):
    digest = hashlib.sha256(canonical_json(list(image.shape)).encode("ascii"))
    digest.update(memoryview(image).cast("B"))
    return digest.hexdigest()


def identities(groups):
    """Check duplicate working pixels before fitting or allocating native descriptors."""
    seen = set()
    result = {}
    for name, group in groups.items():
        digests = [pixel_hash(image) for image in group]
        if len(set(digests)) != len(digests) or seen.intersection(digests):
            raise ValidationError("Duplicate images within or across HOG training inputs")
        seen.update(digests)
        result[name] = digests
    return result


def descriptors(hog, image, count, length, *, stride=None, locations=None):
    result = (
        hog.compute(image)
        if locations is None
        else hog.compute(image, winStride=stride, padding=(0, 0), locations=locations)
    )
    if result is None or result.size != count * length or not np.isfinite(result).all():
        raise ValidationError("Native HOG returned invalid descriptor rows")
    return np.ascontiguousarray(result.reshape(count, length), dtype=np.float32)


def margins(rows, vector):
    return rows.astype(np.float64) @ vector[:-1].astype(np.float64) + float(vector[-1])


def fit_detector(g, pos, neg, parameters, source):
    examples = np.ascontiguousarray(np.concatenate((pos, neg), axis=0), dtype=np.float32)
    truth = np.repeat([0, 1], [len(pos), len(neg)]).astype(np.int32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(float(parameters["c"]))
    svm.setClassWeights(
        np.array([parameters["positive_weight"], parameters["negative_weight"]], np.float32)
    )
    svm.setTermCriteria(
        (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            parameters["max_iterations"],
            float(parameters["epsilon"]),
        )
    )
    if not svm.train(examples, cv2.ml.ROW_SAMPLE, truth):
        raise ValidationError("OpenCV could not train the linear HOG classifier")
    coefficients = linear_detector_coefficients(svm, 0).astype(np.float32)
    if not np.any(coefficients[:-1]):
        raise ValidationError("Training crops produced a detector with no feature weights")
    raw = svm.predict(examples, flags=cv2.ml.StatModel_RAW_OUTPUT)[1].ravel()
    labels = svm.predict(examples)[1].ravel()
    derived = margins(examples, coefficients)
    clear = np.abs(raw) > 1e-5
    if (
        not np.any(clear)
        or not np.isin(labels, (0, 1)).all()
        or not np.array_equal(raw[clear] > 0, labels[clear] == 0)
        or not np.allclose(raw, derived, atol=1e-4, rtol=1e-4)
    ):
        raise ValidationError("Exported HOG margins disagree with the native binary SVM")
    model = make_model(g, coefficients, parameters["label"], source)
    return model, {
        "positive_correct": int(np.count_nonzero(labels[: len(pos)] == 0)),
        "negative_correct": int(np.count_nonzero(labels[len(pos) :] == 1)),
        "maximum_margin_difference": float(np.max(np.abs(raw - derived))),
    }


def mining_windows(backgrounds, g, parameters, length):
    stride = parameters["mining_stride"]
    width, height = g["window_size"]
    windows = []
    for image in backgrounds:
        h, w = image.shape[:2]
        if h < height or w < width:
            raise ValidationError("Mining backgrounds must fit the complete HOG window")
        windows.append(((w - width) // stride[0] + 1) * ((h - height) // stride[1] + 1))
    if sum(windows) > parameters["maximum_mining_windows"]:
        raise ValidationError("HOG mining exceeds the window allowance", "maximum_mining_windows")
    if 2 * sum(windows) * length > MAX_MINING_WORK:
        raise ValidationError("HOG mining exceeds the feature-work allowance")
    return windows


def scan(hog, backgrounds, vector, stride, threshold, window, limits):
    hog.setSVMDetector(vector)
    candidates, counts = [], []
    w, h = window
    for index, (image, limit) in enumerate(zip(backgrounds, limits, strict=True)):
        points, weights = hog.detect(
            image, hitThreshold=threshold, winStride=stride, padding=(0, 0)
        )
        score = np.asarray(weights, np.float64).reshape(-1)
        if len(score) != len(points) or len(score) > limit or not np.isfinite(score).all():
            raise ValidationError("Native HOG mining returned invalid candidates")
        counts.append(len(score))
        seen = set()
        for (x, y), value in zip(points, score, strict=True):
            x, y = int(x), int(y)
            if (
                x < 0
                or y < 0
                or x + w > image.shape[1]
                or y + h > image.shape[0]
                or x % stride[0]
                or y % stride[1]
                or (x, y) in seen
            ):
                raise ValidationError("Native HOG mining returned an invalid source location")
            seen.add((x, y))
            candidates.append((float(value), index, y, x))
    candidates.sort(key=lambda row: (-row[0], *row[1:]))
    return candidates, counts


def hard_descriptors(hog, backgrounds, vector, candidates, known, length, parameters):
    # Compute a bounded prefix in per-image batches. Keep original search context at window edges.
    limit = parameters["maximum_hard_negatives"]
    prefix = candidates[: 4 * limit]
    by_image = defaultdict(list)
    for position, (_, index, y, x) in enumerate(prefix):
        by_image[index].append((position, (x, y)))
    features = {}
    stride = tuple(parameters["mining_stride"])
    for index, locations in by_image.items():
        rows = descriptors(
            hog,
            backgrounds[index],
            len(locations),
            length,
            stride=stride,
            locations=[point for _, point in locations],
        )
        features.update({position: row for (position, _), row in zip(locations, rows, strict=True)})
    distinct = {row.tobytes() for row in known}
    selected, report = [], []
    max_delta = 0.0
    for position, (score, index, y, x) in enumerate(prefix):
        row = features[position]
        delta = abs(float(margins(row, vector)) - score)
        if delta > 1e-4 + abs(score) * 1e-4:
            raise ValidationError("Mined descriptors disagree with native HOG search margins")
        max_delta = max(delta, max_delta)
        fingerprint = row.tobytes()
        if fingerprint in distinct:
            continue
        distinct.add(fingerprint)
        selected.append(row)
        report.append({"background": index, "origin_xy": [x, y], "baseline_margin": score})
        if len(selected) == limit:
            break
    return selected, report, max_delta


class TrainHOGDetector(Block):
    type_id, title = "ml.hog_detector_train", "Train HOG detector with hard negatives"
    category, block_type, subtype = "ML", "ML", "HOG detection"
    inputs = (
        Port("positives", "any"),
        Port("negatives", "any"),
        Port("backgrounds", "any", required=False, required_when_connected=True),
    )
    outputs = (
        Port("model", "data"),
        Port("baseline_model", "data"),
        Port("report", "data"),
    )
    defaults = {
        **HOGFeatures.defaults,
        "window_size": [32, 32],
        "working_encoding": "gray",
        "label": "part",
        "c": 2.0,
        "max_iterations": 5000,
        "epsilon": 1e-6,
        "positive_weight": 1.0,
        "negative_weight": 1.0,
        "mining_stride": [8, 8],
        "mining_threshold": 0.0,
        "maximum_hard_negatives": 64,
        "maximum_mining_windows": 20000,
    }
    parameter_choices = {"working_encoding": ("gray", "bgr")}
    parameter_ranges = {
        **HOGFeatures.parameter_ranges,
        "c": NumericRange(1e-6, 1e6),
        "max_iterations": NumericRange(1, 100_000, whole=True),
        "epsilon": NumericRange(1e-12, 1),
        "positive_weight": NumericRange(1e-6, 1000),
        "negative_weight": NumericRange(1e-6, 1000),
        "mining_threshold": NumericRange(-1e6, 1e6),
        "maximum_hard_negatives": NumericRange(1, 256, whole=True),
        "maximum_mining_windows": NumericRange(1, 100_000, whole=True),
    }
    parameter_help = {
        **HOGFeatures.parameter_help,
        "window_size": "Exact crop [width,height] in pixels. Resize or crop upstream.",
        "working_encoding": "Canonical grayscale or BGR used in both training and detection.",
        "label": "Object class name emitted by the exported detector.",
        "c": "Penalty for the native linear C-SVC fit.",
        "max_iterations": "Maximum native fitting iterations.",
        "epsilon": "Native solver termination tolerance.",
        "positive_weight": "Training cost multiplier for labeled positives.",
        "negative_weight": "Training cost multiplier for both original and mined negatives.",
        "mining_stride": "Fixed-scale background scan stride. Must be a multiple of block stride.",
        "mining_threshold": "Native margin threshold when mining a background window.",
        "maximum_hard_negatives": "Maximum distinct mined descriptors retained for one refit.",
        "maximum_mining_windows": "Combined native window count limit across separate backgrounds.",
    }
    input_help = {
        "positives": "Batch at least two target crops of exactly the configured window size.",
        "negatives": "Batch at least two negative crops of the same window size.",
        "backgrounds": "Optional target-free scene or batch, separate from the crop and test sets.",
    }
    process_inputs_readonly = True
    requires_opencl_context = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        g, _ = geometry(p)
        stride = p["mining_stride"]
        if (
            not isinstance(stride, list)
            or len(stride) != 2
            or any(
                type(v) is not int or not b <= v <= 2048 or v % b
                for v, b in zip(stride, g["block_stride"], strict=True)
            )
        ):
            raise ValidationError(
                "Mining stride must use positive multiples of block stride", "mining_stride"
            )
        for key, rule in self.parameter_ranges.items():
            if (rule.whole and type(p[key]) is not int) or type(p[key]) not in (int, float):
                raise ValidationError(f"Use a numeric value for {key}", key)
        if not isinstance(p["label"], str) or not 1 <= len(p["label"]) <= 256:
            raise ValidationError("Label needs 1..256 characters", "label")

    def process(self, inputs):
        p = self.parameters
        g, length = geometry(p)
        groups = {
            "positive": images(inputs["positives"], "Positives", MAX_CROPS, 2),
            "negative": images(inputs["negatives"], "Negatives", MAX_CROPS, 2),
            "background": images(inputs["backgrounds"], "Backgrounds", MAX_BACKGROUNDS, 1)
            if "backgrounds" in inputs
            else [],
        }
        crop_count = len(groups["positive"]) + len(groups["negative"])
        if crop_count > MAX_CROPS:
            raise ValidationError("Use up to 512 total training crops")
        if any(
            list(im.shape[1::-1]) != g["window_size"]
            for name in ("positive", "negative")
            for im in groups[name]
        ):
            raise ValidationError("All labeled crops must have the exact HOG window geometry")
        if sum(im.shape[0] * im.shape[1] for group in groups.values() for im in group) > MAX_PIXELS:
            raise ValidationError("HOG training images exceed the aggregate pixel allowance")
        windows = mining_windows(groups["background"], g, p, length)
        # Bound the maximum refit before issuing either native SVM call.
        extra = min(p["maximum_hard_negatives"], sum(windows))
        work = ((crop_count + extra) ** 2 + crop_count**2) * length
        if work > MAX_TRAIN_WORK:
            raise ValidationError("HOG fitting exceeds the feature-work allowance")
        groups = {
            name: [working_image(im, g["working_encoding"]) for im in group]
            for name, group in groups.items()
        }
        hashes = identities(groups)
        source = {
            "method": "native_hog_linear_c_svc",
            "stage": "baseline",
            "solver": {key: p[key] for key in SOLVER_FIELDS},
            "positive_sha256": hashes["positive"],
            "negative_sha256": hashes["negative"],
        }
        # Store one aggregate hash per class in the portable model, retaining per-image hashes
        # in the block report. This keeps the model source metadata below 16 KiB.
        source = {
            key: hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
            if key.endswith("_sha256")
            else value
            for key, value in source.items()
        }
        with cpu_execution():
            hog = native_hog(g, levels=1)
            pos, neg = (
                np.concatenate([descriptors(hog, image, 1, length) for image in groups[name]])
                for name in ("positive", "negative")
            )
            baseline, baseline_fit = fit_detector(g, pos, neg, p, source)
            vector = np.asarray(baseline["coefficients"], np.float32)
            stride = tuple(p["mining_stride"])
            backgrounds = groups["background"]
            candidates, before = scan(
                hog, backgrounds, vector, stride, p["mining_threshold"], g["window_size"], windows
            )
            mined, hits, error = hard_descriptors(
                hog, backgrounds, vector, candidates, np.concatenate((pos, neg)), length, p
            )
            model, fitted, after = baseline, baseline_fit, before
            if mined:
                final_source = {
                    **source,
                    "stage": "hard_negative_refit",
                    "baseline_revision": baseline["revision"],
                    "background_sha256": hashes["background"],
                    "mining_stride": list(stride),
                    "mining_threshold": p["mining_threshold"],
                    "mined_sha256": hashlib.sha256(
                        canonical_json(hits).encode("ascii")
                    ).hexdigest(),
                    "mined_count": len(mined),
                }
                model, fitted = fit_detector(g, pos, np.concatenate((neg, mined)), p, final_source)
                _, after = scan(
                    hog,
                    backgrounds,
                    np.asarray(model["coefficients"], np.float32),
                    stride,
                    p["mining_threshold"],
                    g["window_size"],
                    windows,
                )
        return Result(
            {
                "model": model,
                "baseline_model": baseline,
                "report": {
                    "positive_crops": len(pos),
                    "negative_crops": len(neg),
                    "image_hashes": hashes,
                    "mining_windows": windows,
                    "mining_candidates_before": before,
                    "mining_candidates_after": after,
                    "selected_mined_negatives": hits,
                    "max_mining_margin_difference": error,
                    "baseline_fit": baseline_fit,
                    "final_fit": fitted,
                    "baseline_revision": baseline["revision"],
                    "model_revision": model["revision"],
                },
            },
            message=f"Trained HOG on {len(pos)} positive, {len(neg)} negative "
            f"and {len(mined)} mined windows",
        )
