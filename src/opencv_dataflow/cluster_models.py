"""Portable color palettes and image-descriptor vocabularies with fixed label IDs."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .clustering import KMeans, cluster_colors
from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import ImageArray, image_format
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup, metadata
from .processing import image_value
from .sample_tables import read_sample_table
from .selection import SELECTION_INPUTS, selection_mask

MAX_MODEL_BYTES = 8 * 1024**2
MAX_CENTER_VALUES = 131_072
MAX_IMAGE_PIXELS = 1_000_000
MAX_DISTANCE_WORK = 64_000_000
SPACES = ("bgr", "lab", "gray")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def revision(model):
    return hashlib.sha256(
        _json({k: v for k, v in model.items() if k != "revision"}).encode()
    ).hexdigest()


def color_contract(space):
    return {
        "model": "opencv_color_features_v1",
        "space": space,
        "scaling": "unit_bgr_gray;Lab_L/100_ab/128",
        "opencv_version": cv2.__version__,
    }


def _contract(value):
    if value is not None and (not isinstance(value, dict) or len(_json(value).encode()) > 16384):
        raise ValidationError("Feature contract must be a JSON dictionary of at most 16 KiB")
    return copy.deepcopy(value)


def _rows(value):
    contracts = None
    if isinstance(value, FlowGroup):
        contracts = [metadata(item).get("feature_contract") for item in value.items]
        value = [item.value for item in value.items]
        if contracts and any(c != contracts[0] for c in contracts):
            raise ValidationError("Batched descriptors have different preprocessing contracts")
    array = np.asarray(value)
    if array.ndim == 1 and array.size:
        array = array[None, :]
    array = read_sample_table(array, max_columns=65_536)
    if not np.isfinite(array).all() or np.any(np.abs(array.astype(np.float64)) > 1e15):
        raise ValidationError("Cluster samples must be finite and within ±1e15")
    values = np.ascontiguousarray(array, np.float32)
    if np.any((array != 0) & (values == 0)):
        raise ValidationError("Cluster samples underflow float32; rescale the descriptors")
    return values, _contract(contracts[0]) if contracts else None


def _normalize(values, mode):
    if mode == "none":
        return values
    wide = values.astype(np.float64)
    peak = np.max(np.abs(wide), axis=1, keepdims=True)
    if np.any(peak == 0):
        raise ValidationError("Cannot L2 normalize a zero descriptor")
    wide /= peak
    return np.ascontiguousarray(wide / np.linalg.norm(wide, axis=1, keepdims=True), np.float32)


def read_model(model):
    if (
        not isinstance(model, dict)
        or set(model)
        != {"model", "kind", "centers", "normalization", "feature_contract", "revision"}
        or model.get("model") != "opencv_cluster_centers_v1"
    ):
        raise ValidationError("Connect a portable cluster-center model")
    if len(_json(model).encode()) > MAX_MODEL_BYTES or model["revision"] != revision(model):
        raise ValidationError("Cluster model exceeds 8 MiB or its revision differs")
    centers, _ = _rows(model["centers"])
    if len(centers) > 32 or centers.size > MAX_CENTER_VALUES:
        raise ValidationError("Cluster centers exceed 32 rows or 131,072 values")
    # Files retain the exact learned float32 centers, not silently narrowed replacements.
    if not np.array_equal(np.asarray(model["centers"], np.float64), centers.astype(np.float64)):
        raise ValidationError("Cluster centers must retain their exact float32 values")
    contract = _contract(model["feature_contract"])
    if model["kind"] not in ("colors", "descriptors") or model["normalization"] not in (
        "none",
        "l2",
    ):
        raise ValidationError("Invalid cluster kind or normalization")
    if model["kind"] == "colors":
        if (
            not isinstance(contract, dict)
            or contract.get("space") not in SPACES
            or contract != color_contract(contract["space"])
            or model["normalization"] != "none"
            or centers.shape[1] != (1 if contract["space"] == "gray" else 3)
        ):
            raise ValidationError("Color palette preprocessing or OpenCV version differs")
        low = -1.0 if contract["space"] == "lab" else 0.0
        if np.any(centers < low - 1e-6) or np.any(centers > 1 + 1e-6):
            raise ValidationError("Palette centers exceed the saved color-space range")
        if contract["space"] == "lab" and np.any(centers[:, 0] < -1e-6):
            raise ValidationError("Palette Lab lightness must be nonnegative")
    return centers


def nearest(values, centers):
    """Direct float64 differences avoid cancellation in expanded squared norms."""
    if values.shape[1] != centers.shape[1]:
        raise ValidationError("Prediction dimensions differ from the trained centers")
    if values.size * len(centers) > MAX_DISTANCE_WORK:
        raise ValidationError("Nearest-center work exceeds 64 million feature comparisons")
    labels = np.zeros(len(values), np.int32)
    distances = np.empty(len(values), np.float64)
    chunk = max(1, min(4096, 1_000_000 // values.shape[1]))
    for start in range(0, len(values), chunk):
        batch = values[start : start + chunk].astype(np.float64)
        best = np.full(len(batch), np.inf)
        ids = np.zeros(len(batch), np.int32)
        for index, center in enumerate(centers):
            delta = batch - center.astype(np.float64)
            squared = np.einsum("ij,ij->i", delta, delta)
            update = squared < best  # An exact tie keeps the lowest stored cluster ID.
            best[update], ids[update] = squared[update], index
        labels[start : start + len(batch)] = ids
        distances[start : start + len(batch)] = np.sqrt(best)
    return labels, distances


def _bundle(kind, centers, normalization, contract):
    model = {
        "model": "opencv_cluster_centers_v1",
        "kind": kind,
        "centers": centers.tolist(),
        "normalization": normalization,
        "feature_contract": _contract(contract),
    }
    model["revision"] = revision(model)
    read_model(model)
    return model


class _Clusters(Block):
    category, block_type, subtype = "ML", "ML", "Clustering"
    process_inputs_readonly = True
    preview_safe = True
    parallel_safe = True


class TrainClusters(KMeans, _Clusters):
    type_id, title = "ml.kmeans_train", "Train descriptor clusters"
    inputs = (
        Port("samples", "any"),
        Port("initial_labels", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("model", "data"), Port("labels", "data"), Port("report", "data"))
    defaults = {**KMeans.defaults, "normalization": "none"}
    parameter_choices = {**KMeans.parameter_choices, "normalization": ("none", "l2")}
    parameter_help = {
        **KMeans.parameter_help,
        "normalization": "Per-row L2 or none; stored and reused at prediction.",
    }
    payload_reference = "samples"
    parallel_safe = False  # Native kmeans owns RNG state; no global seed changes here.
    input_help = {
        "samples": "Numeric rows or a batch of image-descriptor vectors with matching "
        "feature_contract payloads.",
        "initial_labels": "Optional cluster ID per fitted row; used on the first attempt.",
    }

    def process(self, inputs):
        values, contract = _rows(inputs["samples"])
        if not isinstance(inputs["samples"], FlowGroup):
            contract = _contract(self.input_payloads["samples"].get("feature_contract"))
        count = int(self.parameters["clusters"])
        if values.shape[1] * count > MAX_CENTER_VALUES:
            raise ValidationError("Requested centers exceed 131,072 values")
        values = _normalize(values, self.parameters["normalization"])
        initial = np.asarray(inputs["initial_labels"]) if "initial_labels" in inputs else None
        compactness, labels, centers = self.fit(values, initial)
        model = _bundle("descriptors", centers, self.parameters["normalization"], contract)
        return Result(
            {
                "model": model,
                "labels": labels.tolist(),
                "report": {
                    "samples": len(values),
                    "dimensions": values.shape[1],
                    "clusters": len(centers),
                    "counts": np.bincount(labels, minlength=len(centers)).tolist(),
                    "compactness": compactness,
                    "revision": model["revision"],
                    "label_order": "lexicographic_centers_at_training",
                },
            },
            message=f"Trained {len(centers)} fixed descriptor clusters",
        )


class PredictClusters(_Clusters):
    type_id, title = "ml.kmeans_predict", "Predict descriptor clusters"
    inputs = (Port("model", "data"), Port("samples", "data"))
    outputs = (
        Port("labels", "data"),
        Port("distances", "data"),
        Port("accepted", "data"),
        Port("histogram", "data"),
        Port("report", "data"),
    )
    defaults = {
        "contract_policy": "require_match",
        "max_distance": 0.0,
        "max_unknown_fraction": 0.0,
        "accepted_clusters": [],
    }
    parameter_choices = {"contract_policy": ("require_match", "dimensions_only")}
    parameter_ranges = {
        "max_distance": NumericRange(0, 1e15),
        "max_unknown_fraction": NumericRange(0, 1),
    }
    parameter_help = {
        "contract_policy": "Require the saved preprocessing contract, or explicitly check "
        "dimensions only. Stored normalization always applies.",
        "max_distance": "Euclidean distance in the saved working feature space; 0 disables "
        "the cutoff. Positive limits are inclusive. This is not a probability.",
        "max_unknown_fraction": "Allowed fraction rejected by distance or accepted-cluster "
        "filters. Exceeding it is an inspection failure, not a processing exception.",
        "accepted_clusters": "Stored zero-based IDs to accept; [] accepts all. IDs do not "
        "have semantic names and stay fixed only within the same model revision.",
    }
    payload_reference = "samples"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        ids = p["accepted_clusters"]
        if (
            not isinstance(ids, list)
            or len(ids) > 32
            or any(type(i) is not int or not 0 <= i < 32 for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValidationError(
                "Accepted clusters must be distinct integer IDs in [0,31]", "accepted_clusters"
            )

    def assign_rows(self, model, values):
        centers = read_model(model)
        labels, distances = nearest(_normalize(values, model["normalization"]), centers)
        p = self.parameters
        if any(i >= len(centers) for i in p["accepted_clusters"]):
            raise ValidationError(
                "Accepted cluster ID is outside the trained model", "accepted_clusters"
            )
        accepted = np.ones(len(values), bool)
        if p["max_distance"] > 0:
            accepted &= distances <= p["max_distance"]
        if p["accepted_clusters"]:
            accepted &= np.isin(labels, p["accepted_clusters"])
        counts = np.bincount(labels[accepted], minlength=len(centers))
        histogram = (counts / len(values)).tolist()
        fraction = float(np.count_nonzero(~accepted) / len(values))
        report = {
            "revision": model["revision"],
            "samples": len(values),
            "counts": counts.tolist(),
            "unknown_count": int(np.count_nonzero(~accepted)),
            "unknown_fraction": fraction,
            "max_distance": float(distances.max()),
            "mean_distance": float(distances.mean()),
            "distance_metric": "euclidean",
            "histogram_denominator": "all_selected_samples",
            "probabilities_available": False,
        }
        return labels, distances, accepted, histogram, report

    def histogram_contract(self, model):
        return {
            "model": "cluster_histogram_v1",
            "revision": model["revision"],
            "normalization": "accepted_counts/all_selected_samples",
            "max_distance": self.parameters["max_distance"],
            "accepted_clusters": sorted(self.parameters["accepted_clusters"]),
        }

    def process(self, inputs):
        model = inputs["model"]
        read_model(model)
        if model["kind"] != "descriptors":
            raise ValidationError("Use a descriptor model for descriptor prediction")
        values, _ = _rows(inputs["samples"])
        contract = _contract(self.input_payloads["samples"].get("feature_contract"))
        if (
            self.parameters["contract_policy"] == "require_match"
            and contract != model["feature_contract"]
        ):
            raise ValidationError(
                "Prediction preprocessing differs from the trained feature contract"
            )
        labels, distances, accepted, histogram, report = self.assign_rows(model, values)
        passed = report["unknown_fraction"] <= self.parameters["max_unknown_fraction"]
        return Result(
            {
                "labels": labels.tolist(),
                "distances": distances.tolist(),
                "accepted": accepted.tolist(),
                "histogram": histogram,
                "report": report,
            },
            payloads={
                "histogram": {
                    **self.input_payloads[self.payload_reference],
                    "feature_contract": self.histogram_contract(model),
                }
            },
            passed=passed,
            fail_reasons=[] if passed else ["cluster_unknown_samples"],
            message=f"{len(values)} descriptors · {report['unknown_count']} unknown",
        )


def color_values(inputs, space):
    source = inputs["image"]
    if source.shape[0] * source.shape[1] > MAX_IMAGE_PIXELS:
        raise ValidationError("Color clustering is limited to 1 million image pixels")
    encoding = image_format(source)
    if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Color clusters require tagged gray/BGR/RGB/BGRA/RGBA inputs")
    expected_channels = {"gray": 1, "bgr": 3, "rgb": 3, "bgra": 4, "rgba": 4}[encoding]
    if source.ndim != (2 if expected_channels == 1 else 3) or (
        expected_channels != 1 and source.shape[2] != expected_channels
    ):
        raise ValidationError("Image encoding does not match its channel layout")
    image = image_value(source, (np.uint8, np.uint16, np.float32, np.float64), canonical=True)
    if image.dtype.kind == "f" and (np.any(image < 0) or np.any(image > 1)):
        raise ValidationError("Float color samples must be in [0,1]; no per-image rescaling")
    unit = image.astype(np.float32)
    if image.dtype.kind == "u":
        unit /= np.iinfo(image.dtype).max
    mask = selection_mask(inputs, source)
    selected = np.ones(image.shape[:2], bool) if mask is None else mask != 0
    if image.ndim == 3 and image.shape[2] == 4:
        selected &= image[..., 3] > 0
        unit = unit[..., :3]
    if not selected.any():
        raise ValidationError("Color selection contains no visible pixels")
    if unit.ndim == 2:
        unit = cv2.cvtColor(unit, cv2.COLOR_GRAY2BGR)
    if space == "gray":
        working = cv2.cvtColor(unit, cv2.COLOR_BGR2GRAY)[..., None]
    elif space == "lab":
        working = cv2.cvtColor(unit, cv2.COLOR_BGR2Lab) / np.float32([100, 128, 128])
    else:
        working = unit
    return np.ascontiguousarray(working[selected]), selected


class TrainPalette(TrainClusters):
    type_id, title = "ml.palette_train", "Train image color palette"
    inputs = tuple(
        Port(p.name, p.kind, p.required, p.visual_type, p.name != "image") for p in SELECTION_INPUTS
    )
    outputs = (Port("model", "data"), Port("image"), Port("report", "data"))
    defaults = {**KMeans.defaults, "color_space": "lab", "max_samples": 20_000}
    parameter_choices = {**KMeans.parameter_choices, "color_space": SPACES}
    parameter_ranges = {
        **KMeans.parameter_ranges,
        "max_samples": NumericRange(1, 100_000, whole=True),
    }
    parameter_help = {
        **KMeans.parameter_help,
        "color_space": "Fixed normalized BGR/gray or Lab (L/100, a/128, b/128). No automatic "
        "white balance. Stored and reused on other depths/encodings.",
        "max_samples": "Evenly spaced selected pixels in row-major order, not random sampling. "
        "Raise for rare colors. Transparent pixels are excluded.",
    }
    payload_reference = "image"
    input_help = {
        "image": "uint8/uint16 or normalized float gray/BGR/RGB/BGRA/RGBA; "
        "primary pixels and alpha remain unchanged."
    }

    def process(self, inputs):
        values, selected = color_values(inputs, self.parameters["color_space"])
        total = len(values)
        limit = int(self.parameters["max_samples"])
        if total > limit:
            values = values[np.linspace(0, total - 1, limit, dtype=np.int64)]
        compactness, labels, centers = self.fit(values)
        model = _bundle("colors", centers, "none", color_contract(self.parameters["color_space"]))
        return Result(
            {
                "model": model,
                "image": inputs["image"],
                "report": {
                    "selected_pixels": int(selected.sum()),
                    "fitted_samples": len(values),
                    "sampling": "even_selected_row_major",
                    "compactness": compactness,
                    "counts": np.bincount(labels, minlength=len(centers)).tolist(),
                    "revision": model["revision"],
                    "color_space": self.parameters["color_space"],
                },
            },
            message=f"Trained {len(centers)} colors from {len(values)} pixels",
        )


class ApplyPalette(PredictClusters):
    type_id, title = "ml.palette_predict", "Apply image color palette"
    inputs = (
        Port("model", "data"),
        *tuple(
            Port(p.name, p.kind, p.required, p.visual_type, p.name != "image")
            for p in SELECTION_INPUTS
        ),
    )
    outputs = (
        Port("image"),
        Port("labels"),
        Port("distances"),
        Port("unknown_mask"),
        Port("quantized"),
        Port("cluster_map", required=False),
        Port("histogram", "data"),
        Port("report", "data"),
    )
    defaults = {k: v for k, v in PredictClusters.defaults.items() if k != "contract_policy"}
    parameter_choices = {}
    parameter_help = {
        k: v for k, v in PredictClusters.parameter_help.items() if k != "contract_policy"
    }
    payload_reference = "image"

    def process(self, inputs):
        model = inputs["model"]
        centers = read_model(model)
        if model["kind"] != "colors":
            raise ValidationError("Use a trained image color palette")
        space = model["feature_contract"]["space"]
        values, selected = color_values(inputs, space)
        labels, distances, accepted, histogram, report = self.assign_rows(model, values)
        shape = selected.shape
        label_image = np.full(shape, -1, np.int16)
        label_image[selected] = labels
        distance_image = np.full(shape, -1, np.float32)
        distance_image[selected] = distances
        unknown = np.zeros(shape, np.uint8)
        unknown[selected] = (~accepted).astype(np.uint8) * 255
        source = inputs["image"]
        quantized = self.quantized_image(source, selected, centers[labels], space)
        mapped, drawings = NO_VALUE, {}
        if self.generate_graphics:
            mapped = np.zeros((*shape, 3), np.uint8)
            mapped[selected] = cluster_colors(len(centers))[labels]
            mapped[unknown != 0] = (40, 40, 240)
            objects = []
            contours, _ = cv2.findContours(unknown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vertices = 0
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:128]:
                if vertices + len(contour) > 8192:
                    break
                vertices += len(contour)
                objects.append(
                    graphic_object("polyline", contour.reshape(-1, 2), (40, 40, 240), closed=True)
                )
            objects.append(
                graphic_object(
                    "label",
                    [(8, 8)],
                    (40, 40, 240) if unknown.any() else (80, 220, 80),
                    text=f"Unknown: {report['unknown_fraction']:.1%}",
                )
            )
            drawings["Palette inspection"] = GraphicDrawing(tuple(objects))
        passed = report["unknown_fraction"] <= self.parameters["max_unknown_fraction"]
        report["color_space"] = space
        return Result(
            {
                "image": source,
                "labels": label_image,
                "distances": distance_image,
                "unknown_mask": unknown,
                "quantized": quantized,
                "cluster_map": mapped,
                "histogram": histogram,
                "report": report,
            },
            drawings=drawings,
            payloads={
                "histogram": {
                    **self.input_payloads[self.payload_reference],
                    "feature_contract": self.histogram_contract(model),
                }
            },
            passed=passed,
            fail_reasons=[] if passed else ["palette_unknown_pixels"],
            message=f"{int(selected.sum())} pixels · {report['unknown_count']} unknown",
        )

    @staticmethod
    def quantized_image(source, selected, values, space):
        if space == "lab":
            colors = cv2.cvtColor((values * np.float32([100, 128, 128]))[None], cv2.COLOR_Lab2BGR)[
                0
            ]
        elif space == "gray":
            colors = np.repeat(values, 3, axis=1)
        else:
            colors = values
        if image_format(source) in ("rgb", "rgba"):
            colors = colors[:, ::-1]
        if source.ndim == 2:
            # The trained color palette may have been captured from a color image.
            colors = (colors @ np.float32([0.114, 0.587, 0.299]))[:, None]
        colors = np.clip(colors, 0, 1)
        if source.dtype.kind == "u":
            colors = np.rint(colors.astype(np.float64) * np.iinfo(source.dtype).max)
        output = np.array(source, copy=True)
        if source.ndim == 2:
            output[selected] = colors[:, 0]
        else:
            output[..., :3][selected] = colors
        return ImageArray(output, image_format(source), metadata=getattr(source, "metadata", {}))


class SaveClusters(_Clusters):
    type_id, title = "ml.kmeans_save", "Save cluster model"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs, outputs = (Port("model", "data"),), (Port("path", "data"),)
    defaults = {"path": "clusters.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_model(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(model) + "\n")
        return Result(
            {"path": str(path)}, message="Saved fixed cluster model; existing files preserved"
        )


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate cluster model JSON key: " + key)
        result[key] = value
    return result


def _constant(token):
    raise ValidationError("Nonfinite cluster model JSON number: " + token)


class LoadClusters(_Clusters):
    type_id, title = "ml.kmeans_load", "Load cluster model"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("Cluster model exceeds 8 MiB")
        model = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        read_model(model)
        return Result({"model": model}, message=f"Loaded {len(model['centers'])} fixed clusters")


CLUSTER_MODEL_BLOCKS = (
    TrainClusters,
    PredictClusters,
    TrainPalette,
    ApplyPalette,
    SaveClusters,
    LoadClusters,
)
