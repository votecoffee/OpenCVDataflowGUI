"""Bounded native ANNIndex image retrieval with stable IDs and self-contained bundles."""

import base64
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .classifier_common import (
    canonical_json,
    descriptor_matrix,
    feature_contract,
    label_strings,
    model_revision,
    normalize_features,
    prediction_drawing,
    validate_prediction_image,
)
from .core import Block, Port, Result
from .image_hash import HASH_MODEL, read_hash
from .ml_common import local_path
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, FlowGroup

MAX_MODEL_BYTES = 64 * 1024**2
MAX_INDEX_BYTES = 32 * 1024**2
MAX_ELEMENTS = 2_000_000
MAX_ROWS = 4096
MAX_WORK = 64_000_000
METRICS = {
    "euclidean": cv2.ANNINDEX_DIST_EUCLIDEAN,
    "manhattan": cv2.ANNINDEX_DIST_MANHATTAN,
    "angular": cv2.ANNINDEX_DIST_ANGULAR,
    "hamming": cv2.ANNINDEX_DIST_EUCLIDEAN,
    "dot": cv2.ANNINDEX_DIST_DOTPRODUCT,
}
MODEL_KEYS = {
    "model",
    "metric",
    "normalization",
    "samples",
    "item_ids",
    "labels",
    "feature_contract",
    "settings",
    "native",
    "revision",
}


def abi():
    return {
        "opencv": cv2.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
    }


def rows(value):
    def hashed(record):
        method, options, packed, bits = read_hash(record, "samples")
        contract = {
            "model": "opencv_image_hash_features_v1",
            "method": method,
            "options": options,
            "bits": bits,
            "opencv": cv2.__version__,
        }
        return packed.astype(np.float32), contract

    if isinstance(value, dict) and value.get("model") == HASH_MODEL:
        packed, contract = hashed(value)
        return packed[None], contract
    if isinstance(value, FlowGroup) and all(
        isinstance(item.value, dict) and item.value.get("model") == HASH_MODEL
        for item in value.items
    ):
        packed = [hashed(item.value) for item in value.items]
        if any(contract != packed[0][1] for _, contract in packed):
            raise ValidationError("Image hashes have different methods/options")
        return np.stack([row for row, _ in packed]), packed[0][1]
    return descriptor_matrix(value, max_rows=MAX_ROWS, max_elements=MAX_ELEMENTS, max_abs=1e6)


def native_vectors(values, metric, normalization):
    if metric == "hamming":
        if (
            normalization != "none"
            or np.any(values < 0)
            or np.any(values > 255)
            or np.any(values != np.rint(values))
        ):
            raise ValidationError("Hamming needs packed byte rows and normalization=None")
        if values.size * 8 > MAX_ELEMENTS or values.shape[1] * 8 > 65_536:
            raise ValidationError("Expanded Hamming bits exceed native element/dimension limits")
        # The installed uchar Hamming backend returns incorrect zero distances.
        # Euclidean on unpacked bits preserves Hamming ranking exactly: L2^2=popcount.
        return np.ascontiguousarray(np.unpackbits(values.astype(np.uint8), axis=1), np.float32)
    if metric == "angular" and np.any(np.linalg.norm(values.astype(np.float64), axis=1) == 0):
        raise ValidationError("Angular search needs nonzero descriptors")
    return np.ascontiguousarray(values, np.float32)


def distances(values, query, metric):
    a, b = values.astype(np.float64), query.astype(np.float64)
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=1)
    if metric == "manhattan":
        return np.abs(a - b).sum(axis=1)
    if metric == "angular":
        cosine = (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))
        return np.sqrt(np.maximum(0, 2 - 2 * np.clip(cosine, -1, 1)))
    if metric == "dot":
        return a @ b
    return (
        np.unpackbits(np.bitwise_xor(values.astype(np.uint8), query.astype(np.uint8)), axis=1)
        .sum(axis=1)
        .astype(np.float64)
    )


def read_model(model):
    if (
        not isinstance(model, dict)
        or set(model) != MODEL_KEYS
        or model.get("model") != "opencv_ann_index_v1"
    ):
        raise ValidationError("Connect a self-contained native ANN index model")
    if len(canonical_json(model).encode()) > MAX_MODEL_BYTES or model["revision"] != model_revision(
        model
    ):
        raise ValidationError("ANN model exceeds 64 MiB or its revision differs")
    if model["metric"] not in METRICS or model["normalization"] not in ("none", "l2"):
        raise ValidationError("Unknown ANN metric or normalization")
    values, _ = rows(model["samples"])
    if not np.array_equal(np.asarray(model["samples"], np.float64), values.astype(np.float64)):
        raise ValidationError("Stored ANN descriptors must retain exact float32 values")
    for name in ("item_ids", "labels"):
        strings = label_strings(model[name], name, maximum=MAX_ROWS)
        if len(strings) != len(values) or name == "item_ids" and len(set(strings)) != len(strings):
            raise ValidationError("ANN needs one unique item ID and one label per descriptor")
    if model["normalization"] == "l2" and not np.allclose(
        np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0
    ):
        raise ValidationError("Stored L2 descriptors must have unit length")
    feature_contract(model["feature_contract"])
    settings = model["settings"]
    if (
        not isinstance(settings, dict)
        or set(settings) != {"trees", "seed", "build_mode"}
        or type(settings["trees"]) is not int
        or not 1 <= settings["trees"] <= 64
        or type(settings["seed"]) is not int
        or not 0 <= settings["seed"] <= 2**32 - 1
        or settings["build_mode"] not in ("memory", "disk")
    ):
        raise ValidationError("Invalid native ANN build settings")
    native = model["native"]
    if (
        not isinstance(native, dict)
        or set(native) != {"abi", "sha256", "bytes", "data", "trees", "dimensions"}
        or native["abi"] != abi()
        or type(native["bytes"]) is not int
        or not 1 <= native["bytes"] <= MAX_INDEX_BYTES
        or type(native["trees"]) is not int
        or not 1 <= native["trees"] <= 2 * settings["trees"]
        or not isinstance(native["data"], str)
        or len(native["data"]) > (MAX_INDEX_BYTES + 2) // 3 * 4
    ):
        raise ValidationError(
            "Invalid/incompatible native ANN bytes or ABI; rebuild from source descriptors"
        )
    prepared = native_vectors(values, model["metric"], model["normalization"])
    if type(native["dimensions"]) is not int or native["dimensions"] != prepared.shape[1]:
        raise ValidationError("Native ANN dimensions differ from its saved descriptors")
    try:
        raw = base64.b64decode(native["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid base64 native ANN data") from exc
    if len(raw) != native["bytes"] or hashlib.sha256(raw).hexdigest() != native["sha256"]:
        raise ValidationError("Native ANN byte count/hash mismatch")
    return values, raw


class NativeHandle:
    """Own the mmap and its private verified file together; no native pointer travels on a wire."""

    def __init__(self, model, raw, prefault):
        self.index, self.directory = None, None
        try:
            self.directory = tempfile.TemporaryDirectory(prefix="dataflow-ann-load-")
            path = Path(self.directory.name) / "index.ann"
            path.write_bytes(raw)
            self.index = cv2.ANNIndex.create(
                model["native"]["dimensions"], METRICS[model["metric"]]
            )
            self.index.load(str(path), prefault)
            if (
                self.index.getItemNumber() != len(model["samples"])
                or self.index.getTreeNumber() != model["native"]["trees"]
            ):
                raise ValidationError("Native ANN item/tree counts differ from its manifest")
        except Exception:
            self.close()
            raise

    def close(self):
        self.index = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None


class _ANNBlock(Block):
    category, block_type, subtype = "ML", "ML", "Image retrieval"
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for name, limits in self.parameter_ranges.items():
            if type(p[name]) not in (int, float) or limits.whole and type(p[name]) is not int:
                raise ValidationError("Use typed numeric ANN controls", name)


class BuildNativeANN(_ANNBlock):
    type_id, title = "ml.native_ann_build", "Build native image embedding index"
    preview_safe = True
    inputs = (
        Port("samples", "any"),
        Port("item_ids", "data", required=False, required_when_connected=True),
        Port("labels", "data", required=False, required_when_connected=True),
    )
    outputs = (Port("model", "data"), Port("count", "integer"), Port("report", "data"))
    defaults = {
        "metric": "euclidean",
        "normalization": "none",
        "trees": 10,
        "seed": 0,
        "build_mode": "memory",
        "item_ids": [],
        "labels": [],
        "maximum_index_bytes": 16 * 1024**2,
    }
    parameter_choices = {
        "metric": tuple(METRICS),
        "normalization": ("none", "l2"),
        "build_mode": ("memory", "disk"),
    }
    parameter_ranges = {
        "trees": NumericRange(1, 64, whole=True),
        "seed": NumericRange(0, 2**32 - 1, whole=True),
        "maximum_index_bytes": NumericRange(1, MAX_INDEX_BYTES, whole=True),
    }
    parameter_help = {
        "metric": (
            "Native L2/L1/angular/dot. Hamming uses exact byte-to-bit expansion and native L2."
        ),
        "normalization": (
            "Saved per-row L2 or None. Hamming requires None; zero rows cannot use L2/angular."
        ),
        "trees": "Explicit bounded forest count. Native automatic tree count is not used.",
        "seed": "Full uint32 seed; converted to the signed native API without changing its bits.",
        "build_mode": (
            "RAM build/native save, or temporary native setOnDiskBuild before adding rows."
        ),
        "item_ids": "Unique strings per image; blank uses batch member IDs or sequential strings.",
        "labels": "Optional labels per image; blank uses item IDs. Duplicate labels are allowed.",
        "maximum_index_bytes": (
            "Reject conservative node-size estimate before native build and actual bytes afterward."
        ),
    }
    input_help = {
        "samples": (
            "Image descriptor rows or a FlowGroup of descriptors with consistent feature_contract."
        ),
        "item_ids": (
            "Optional stable unique image identifiers, one per row; overrides configured IDs."
        ),
        "labels": "Optional display/class labels, one per row; overrides configured labels.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._model, self._key = None, None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("item_ids", "labels"):
            label_strings(p[key], key, maximum=MAX_ROWS)
        if p["metric"] == "hamming" and p["normalization"] != "none":
            raise ValidationError("Hamming requires normalization=None", "normalization")

    def process(self, inputs):
        p = self.parameters
        values, grouped_contract = rows(inputs["samples"])
        contract = feature_contract(
            grouped_contract
            if grouped_contract is not None
            else self.input_payloads.get("samples", {}).get("feature_contract")
        )
        values = normalize_features(values, p["normalization"])
        prepared = native_vectors(values, p["metric"], p["normalization"])
        ids = inputs.get("item_ids", p["item_ids"])
        if not ids:
            ids = (
                list(inputs["samples"].labels)
                if isinstance(inputs["samples"], FlowGroup)
                else list(map(str, range(len(values))))
            )
        labels = inputs.get("labels", p["labels"]) or ids
        for key, data in (("item_ids", ids), ("labels", labels)):
            label_strings(data, key, maximum=MAX_ROWS)
            if len(data) != len(values) or key == "item_ids" and len(set(data)) != len(data):
                raise ValidationError(
                    "One unique item ID and one label per training row are required", key
                )
        model = {
            "model": "opencv_ann_index_v1",
            "metric": p["metric"],
            "normalization": p["normalization"],
            "samples": values.tolist(),
            "item_ids": list(ids),
            "labels": list(labels),
            "feature_contract": contract,
            "settings": {k: p[k] for k in ("trees", "seed", "build_mode")},
        }
        key = model_revision(model)
        reused = self._model is not None and self._key == (key, p["maximum_index_bytes"])
        if not reused:
            self.reset_runtime()
            stride = 4 * (prepared.shape[1] + (6 if p["metric"] == "dot" else 4))
            estimate = (len(values) * (1 + 2 * p["trees"]) + 2 * p["trees"]) * stride
            if estimate > p["maximum_index_bytes"]:
                raise ValidationError(
                    "Native ANN conservative build estimate exceeds byte budget",
                    "maximum_index_bytes",
                )
            native = None
            with tempfile.TemporaryDirectory(prefix="dataflow-ann-build-") as directory:
                path = Path(directory) / "index.ann"
                try:
                    native = cv2.ANNIndex.create(prepared.shape[1], METRICS[p["metric"]])
                    native.setSeed(p["seed"] if p["seed"] < 2**31 else p["seed"] - 2**32)
                    if p["build_mode"] == "disk" and not native.setOnDiskBuild(str(path)):
                        raise ValidationError("Native ANN refused on-disk build")
                    native.addItems(prepared)
                    native.build(p["trees"])
                    trees = native.getTreeNumber()
                    if native.getItemNumber() != len(values) or trees != p["trees"]:
                        raise ValidationError(
                            "Native ANN build returned inconsistent item/tree counts"
                        )
                    if p["build_mode"] == "memory":
                        native.save(str(path))
                finally:
                    native = None
                with path.open("rb") as stream:
                    raw = stream.read(p["maximum_index_bytes"] + 1)
                if not raw or len(raw) > p["maximum_index_bytes"]:
                    raise ValidationError(
                        "Native ANN file exceeds byte budget", "maximum_index_bytes"
                    )
                try:
                    native = cv2.ANNIndex.create(prepared.shape[1], METRICS[p["metric"]])
                    native.load(str(path))
                    trees = native.getTreeNumber()
                    if native.getItemNumber() != len(values) or not 1 <= trees <= 2 * p["trees"]:
                        raise ValidationError("Reloaded native ANN item/tree counts differ")
                finally:
                    native = None
            model["native"] = {
                "abi": abi(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "data": base64.b64encode(raw).decode("ascii"),
                "trees": trees,
                "dimensions": prepared.shape[1],
            }
            model["revision"] = model_revision(model)
            read_model(model)
            self._model, self._key = model, (key, p["maximum_index_bytes"])
        model = self._model
        return Result(
            {
                "model": copy.deepcopy(model),
                "count": len(values),
                "report": {
                    "revision": model["revision"],
                    "items": len(values),
                    "dimensions": values.shape[1],
                    "native_dimensions": prepared.shape[1],
                    "index_bytes": model["native"]["bytes"],
                    "build_mode": p["build_mode"],
                    "requested_trees": p["trees"],
                    "reloaded_native_trees": model["native"]["trees"],
                    "native_build_reused": reused,
                    "metric": p["metric"],
                    "effective_native_metric": "euclidean_bits"
                    if p["metric"] == "hamming"
                    else p["metric"],
                },
            },
            message=f"Indexed {len(values)} image descriptors ({p['metric']})",
        )


class SearchNativeANN(_ANNBlock):
    type_id, title = "ml.native_ann_search", "Search native image embeddings"
    preview_safe = True
    payload_reference = "samples"
    inputs = (
        Port("model", "data"),
        Port("samples", "any"),
        Port("image", required=False, required_when_connected=True),
    )
    outputs = (
        Port("neighbors", "data"),
        Port("item_ids", "data"),
        Port("labels", "data"),
        Port("item_id", "text", required=False),
        Port("label", "text", required=False),
        Port("accepted", "data"),
        Port("image", required=False),
        Port("count", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "neighbors": 5,
        "candidates": 32,
        "search_k": -1,
        "minimum_results": 1,
        "filter_results": False,
        "maximum_distance": 1.0,
        "minimum_similarity": 0.0,
        "contract_policy": "require_match",
        "prefault": False,
    }
    parameter_choices = {"contract_policy": ("require_match", "dimensions_only")}
    parameter_ranges = {
        "neighbors": NumericRange(1, MAX_ROWS, whole=True),
        "candidates": NumericRange(1, MAX_ROWS, whole=True),
        "search_k": NumericRange(-1, 10_000_000, whole=True),
        "minimum_results": NumericRange(1, MAX_ROWS, whole=True),
        "maximum_distance": NumericRange(0, 1e18),
        "minimum_similarity": NumericRange(-1e18, 1e18),
    }
    parameter_help = {
        "neighbors": (
            "Returned neighbors per query, limited to gallery size after candidate "
            "reranking/filtering."
        ),
        "candidates": (
            "Native candidate count per query, capped at gallery size. More "
            "candidates may improve recall."
        ),
        "search_k": (
            "-1 uses native default; otherwise positive effort limit. Approximate "
            "search may return fewer rows."
        ),
        "minimum_results": "Minimum qualifying returned neighbors per query for a passing result.",
        "filter_results": "Apply the inclusive distance limit, or minimum dot similarity.",
        "maximum_distance": (
            "Used for L2/L1/angular/Hamming only when filtering is enabled; native "
            "L2 is not squared."
        ),
        "minimum_similarity": "Minimum dot product when filtering is enabled; may be negative.",
        "contract_policy": (
            "Require the saved descriptor preprocessing contract, or explicitly "
            "check dimensions only."
        ),
        "prefault": (
            "Ask native mmap to pre-read pages when supported; this platform may ignore it."
        ),
    }
    input_help = {
        "model": "Checked native index bundle; never a pointer, pickle or bare .ann path.",
        "samples": "One query descriptor or rows, using the gallery's encoding and normalization.",
        "image": "Optional unchanged review image for exactly one query; known context must agree.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._handle, self._key = None, None

    def reset_runtime(self):
        if self._handle is not None:
            self._handle.close()
        self._handle, self._key = None, None

    close = reset_runtime

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["search_k"] == 0 or p["search_k"] < -1:
            raise ValidationError("Search effort must be -1 or positive", "search_k")
        if p["neighbors"] > p["candidates"] or p["minimum_results"] > p["neighbors"]:
            raise ValidationError(
                "Require minimum results <= neighbors <= candidates",
                "neighbors",
                "candidates",
                "minimum_results",
            )
        for key in ("filter_results", "prefault"):
            if type(p[key]) is not bool:
                raise ValidationError("Use a checkbox", key)

    def process(self, inputs):
        model, p = inputs["model"], self.parameters
        values, raw = read_model(model)
        query, grouped_contract = rows(inputs["samples"])
        contract = feature_contract(
            grouped_contract
            if grouped_contract is not None
            else self.input_payloads.get("samples", {}).get("feature_contract")
        )
        if p["contract_policy"] == "require_match" and contract != model["feature_contract"]:
            raise ValidationError(
                "Query descriptor preprocessing differs from the gallery contract",
                "contract_policy",
            )
        if query.shape[1] != values.shape[1]:
            raise ValidationError("Query descriptor dimensions differ from the gallery")
        query = normalize_features(query, model["normalization"])
        prepared = native_vectors(query, model["metric"], model["normalization"])
        k = min(p["candidates"], len(values))
        if len(query) * k * prepared.shape[1] > MAX_WORK:
            raise ValidationError("ANN query/candidate work exceeds 64 million feature comparisons")
        effort = p["search_k"] if p["search_k"] > 0 else model["native"]["trees"] * k
        if len(query) * effort > MAX_WORK:
            raise ValidationError("ANN batch search effort exceeds 64 million visits", "search_k")
        image = validate_prediction_image(self, inputs, len(query))
        key = (model["revision"], p["prefault"])
        reused = self._key == key and self._handle is not None
        if not reused:
            self.reset_runtime()
            self._handle = NativeHandle(model, raw, p["prefault"])
            self._key = key
        try:
            ids = np.full((len(query), k), -1, np.int32)
            native_distances = np.full((len(query), k), np.nan, np.float32)
            ids, native_distances = self._handle.index.knnSearch(
                prepared, k, indices=ids, dists=native_distances, search_k=p["search_k"]
            )
            if ids.shape != (len(query), k) or native_distances.shape != ids.shape:
                raise ValidationError("Native ANN returned invalid neighbor shapes")
            results, missing = [], []
            dot = model["metric"] == "dot"
            for row, q, raw_distances in zip(ids, query, native_distances, strict=True):
                valid = row >= 0
                if (
                    np.any(row < -1)
                    or np.any(row[valid] >= len(values))
                    or len(set(row[valid])) != int(valid.sum())
                    or not np.isfinite(raw_distances[valid]).all()
                ):
                    raise ValidationError("Native ANN returned invalid candidate IDs/distances")
                candidate_ids = row[valid]
                measured = distances(values[candidate_ids], q, model["metric"])
                ranked = sorted(
                    zip(candidate_ids, measured, raw_distances[valid], strict=True),
                    key=lambda item: (-item[1] if dot else item[1], model["item_ids"][item[0]]),
                )
                current = []
                for index, d, raw_distance in ranked:
                    if p["filter_results"] and (
                        (dot and d < p["minimum_similarity"])
                        or (not dot and d > p["maximum_distance"])
                    ):
                        continue
                    current.append(
                        {
                            "index": int(index),
                            "item_id": model["item_ids"][index],
                            "label": model["labels"][index],
                            "distance": None if dot else float(d),
                            "similarity": float(d) if dot else None,
                            "native_distance": float(raw_distance),
                        }
                    )
                results.append(current[: p["neighbors"]])
                missing.append(k - int(valid.sum()))
        except Exception:
            self.reset_runtime()
            raise
        accepted = [len(row) >= p["minimum_results"] for row in results]
        chosen_ids = [
            row[0]["item_id"] if good else None for row, good in zip(results, accepted, strict=True)
        ]
        chosen_labels = [
            row[0]["label"] if good else None for row, good in zip(results, accepted, strict=True)
        ]
        passed = all(accepted)
        label = chosen_labels[0] if len(results) == 1 and accepted[0] else NO_VALUE
        item_id = chosen_ids[0] if len(results) == 1 and accepted[0] else NO_VALUE
        return Result(
            {
                "neighbors": results,
                "item_ids": chosen_ids,
                "labels": chosen_labels,
                "item_id": item_id,
                "label": label,
                "accepted": accepted,
                "image": image,
                "count": sum(map(len, results)),
                "report": {
                    "revision": model["revision"],
                    "metric": model["metric"],
                    "effective_native_metric": "euclidean_bits"
                    if model["metric"] == "hamming"
                    else model["metric"],
                    "native_index_reused": reused,
                    "candidate_count": k,
                    "missing_candidates": missing,
                    "result_counts": [len(row) for row in results],
                    "approximate": True,
                    "ranking": "exact metric rerank of returned candidates, then stable item ID",
                },
            },
            passed=passed,
            message=f"ANN retrieval: {sum(accepted)}/{len(results)} queries accepted",
            fail_reasons=[] if passed else ["native_ann.insufficient_neighbors"],
            drawings=prediction_drawing(
                self,
                image,
                "ANN: " + str(label if label is not NO_VALUE else "unknown"),
                passed,
                "ANN retrieval",
            ),
        )


class SaveNativeANN(_ANNBlock):
    type_id, title = "ml.native_ann_save", "Save native image embedding index"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs = (Port("model", "data"),)
    outputs = (Port("path", "text"),)
    defaults = {"path": "image-index.json"}
    parameter_help = {
        "path": (
            "New self-contained JSON file retaining exact native bytes, descriptors "
            "and stable IDs. Existing files are preserved."
        )
    }
    project_path_parameters = ("path",)

    def process(self, inputs):
        read_model(inputs["model"])
        raw = (canonical_json(inputs["model"]) + "\n").encode()
        path = local_path(self, self.parameters["path"])
        with tempfile.TemporaryDirectory(
            prefix=".dataflow-ann-save-", dir=path.parent
        ) as directory:
            temporary = Path(directory) / "bundle.json"
            temporary.write_bytes(raw)
            os.link(temporary, path)
        return Result({"path": str(path)}, message="Saved checked native ANN bundle")


class LoadNativeANN(_ANNBlock):
    type_id, title = "ml.native_ann_load", "Load native image embedding index"
    outputs = (Port("model", "data"), Port("count", "integer"))
    defaults = {"path": "", "prefault": False}
    project_asset_parameters = ("recipe_path", "path")
    parameter_help = {
        "path": "Self-contained native ANN JSON bundle with matching hashes and native ABI.",
        "prefault": (
            "Ask native mmap to pre-read pages when supported; unsupported platforms may ignore it."
        ),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["path"], str) or type(p["prefault"]) is not bool:
            raise ValidationError("Choose a path and use the prefault checkbox", "path", "prefault")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValidationError("ANN JSON file exceeds 64 MiB")

        def pairs(items):
            data = {}
            for key, value in items:
                if key in data:
                    raise ValidationError("Duplicate ANN JSON field: " + key)
                data[key] = value
            return data

        def bad_constant(value):
            raise ValidationError("Nonfinite ANN JSON value: " + value)

        model = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
        values, binary = read_model(model)
        handle = NativeHandle(model, binary, self.parameters["prefault"])
        handle.close()
        return Result({"model": model, "count": len(values)}, message="Verified native ANN bundle")


NATIVE_INDEX_BLOCKS = (BuildNativeANN, SearchNativeANN, SaveNativeANN, LoadNativeANN)
