"""Annoy vector indexes with portable training data and checked on-disk bundles."""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .core import Block, Port, Result
from .ml_common import array, digest_file, file_key, local_path, normalized_vectors
from .processing import finite, integer

METRICS = ("angular", "euclidean", "manhattan", "hamming", "dot")


def annoy_class():
    try:
        from annoy import AnnoyIndex
    except ImportError as exc:
        raise RuntimeError("Annoy is optional. Install this project's [annoy] extra.") from exc
    return AnnoyIndex


def index_metadata(model):
    if not isinstance(model, dict) or model.get("model") not in ("annoy_index_v1", "annoy_file_v1"):
        raise ValueError("Connect an Annoy index model")
    if model.get("metric") not in METRICS or model.get("normalize") not in ("none", "l2"):
        raise ValueError("Invalid index metric/normalization")
    count = integer(model.get("count"), "Index count", 1, 100_000)
    dimensions = integer(model.get("dimensions"), "Index dimensions", 1, 65536)
    labels = model.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) != count
        or any(not isinstance(s, str) or len(s) > 1024 for s in labels)
    ):
        raise ValueError("Index requires one string label per item")
    return count, dimensions


def index_vectors(model):
    count, dimensions = index_metadata(model)
    values = array(model.get("vectors"), "Index vectors")
    if values.shape != (count, dimensions):
        raise ValueError("Index vectors do not match count/dimensions")
    if model["metric"] == "angular" and np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Angular indexes require nonzero vectors")
    if model["metric"] == "hamming" and (
        model["normalize"] != "none" or np.any((values != 0) & (values != 1))
    ):
        raise ValueError("Hamming requires binary vectors and normalize=none")
    return values


def native_index(model):
    count, dimensions = index_metadata(model)
    index = annoy_class()(dimensions, model["metric"])
    if model["model"] == "annoy_file_v1":
        path = Path(model["index_path"])
        if path.stat().st_size < 12 or digest_file(path) != model.get("sha256"):
            raise ValueError("Annoy index checksum mismatch or truncated file")
        if not index.load(str(path), prefault=bool(model.get("prefault", False))):
            raise ValueError("Unable to load Annoy index")
        if index.get_n_items() != count:
            index.unload()
            raise ValueError("Annoy item count differs from metadata")
    else:
        values = index_vectors(model)
        index.set_seed(integer(model.get("seed"), "Index seed", 0, 2**31 - 1))
        for i, value in enumerate(values):
            index.add_item(i, value.tolist())
        index.build(integer(model.get("trees"), "Trees", 1, 1000), n_jobs=1)
    return index


class VectorData(Block):
    type_id, title = "ml.vector_data", "Vector data"
    category, block_type, subtype = "ML", "ML", "Embeddings"
    outputs = (Port("vectors", "data"),)
    defaults = {"values": [[1.0, 0.0], [0.0, 1.0]]}

    def process(self, inputs):
        values = array(self.parameters["values"], "Vector data")
        if values.ndim not in (1, 2):
            raise ValueError("Vector data must be one vector or a row matrix")
        return Result({"vectors": values.tolist()})


class BuildAnnoy(Block):
    type_id, title = "ml.annoy_build", "Build Annoy index"
    category, block_type, subtype = "ML", "ML", "Vector search"
    inputs = (Port("vectors", "data"), Port("labels", "data", required=False))
    outputs = (Port("model", "data"), Port("count", "number"))
    defaults = {"metric": "angular", "trees": 10, "seed": 0, "normalize": "none", "labels": []}
    parameter_choices = {"metric": METRICS, "normalize": ("none", "l2")}
    parameter_help = {
        "trees": "Build-time recall/storage tradeoff; search_k controls query effort.",
        "labels": "One string label per vector; empty generates sequential IDs.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._key = None

    def reset_runtime(self):
        self._key = None

    def process(self, inputs):
        p = self.parameters
        vectors = normalized_vectors(inputs["vectors"], p["normalize"])
        if vectors.ndim != 2:
            raise ValueError("Index training vectors must be a row matrix")
        labels = inputs.get("labels", p["labels"])
        if labels == []:
            labels = list(map(str, range(len(vectors))))
        model = {
            "model": "annoy_index_v1",
            "vectors": vectors.tolist(),
            "labels": labels,
            "metric": p["metric"],
            "normalize": p["normalize"],
            "trees": p["trees"],
            "seed": p["seed"],
            "count": len(vectors),
            "dimensions": vectors.shape[1],
        }
        index_vectors(model)
        key = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        if key != self._key:
            native_index(model).unload()
            self._key = key
        return Result(
            {"model": model, "count": len(vectors)}, message=f"{len(vectors)} indexed vectors"
        )


class SearchAnnoy(Block):
    type_id, title = "ml.annoy_search", "Annoy nearest neighbors"
    category, block_type, subtype = "ML", "ML", "Vector search"
    inputs = (Port("model", "data"), Port("vector", "data"))
    outputs = (Port("neighbors", "data"), Port("label", "data"), Port("count", "number"))
    defaults = {
        "neighbors": 5,
        "search_k": -1,
        "max_distance": -1.0,
        "min_similarity": -1.0,
        "minimum_results": 1,
    }
    parameter_help = {
        "search_k": "-1 uses Annoy's default; larger values favor recall over latency.",
        "max_distance": "Negative disables filtering; not used for dot metric.",
        "min_similarity": "Dot metric only: minimum inner product; -1 disables.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index, self._key = None, None

    def reset_runtime(self):
        if self._index is not None:
            self._index.unload()
        self._index, self._key = None, None

    close = reset_runtime

    def process(self, inputs):
        model, p = inputs["model"], self.parameters
        count, dimensions = index_metadata(model)
        query = normalized_vectors(inputs["vector"], model["normalize"])
        if query.shape != (dimensions,):
            raise ValueError(f"Query must be a vector with {dimensions} dimensions")
        if model["metric"] == "angular" and np.linalg.norm(query) == 0:
            raise ValueError("Angular search requires a nonzero query")
        if model["metric"] == "hamming" and np.any((query != 0) & (query != 1)):
            raise ValueError("Hamming query must be binary")
        key = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        if model["model"] == "annoy_file_v1":
            key = (key, file_key(Path(model["index_path"])))
        if key != self._key:
            self.reset_runtime()
            self._index = native_index(model)
            self._key = key
        n = min(count, integer(p["neighbors"], "Neighbors", 1, 10000))
        search_k = integer(p["search_k"], "Search K", -1, 10_000_000)
        if search_k == 0:
            raise ValueError("Search K must be -1 or positive")
        maximum = finite(p["max_distance"], "Maximum distance")
        minimum = finite(p["min_similarity"], "Minimum similarity")
        ids, distances = self._index.get_nns_by_vector(
            query.tolist(), n, search_k=search_k, include_distances=True
        )
        dot = model["metric"] == "dot"
        neighbors = [
            {
                "id": i,
                "label": model["labels"][i],
                "distance": None if dot else float(d),
                "similarity": float(d) if dot else None,
            }
            for i, d in zip(ids, distances)
            if (
                (dot and (minimum == -1 or d >= minimum))
                or (not dot and (maximum < 0 or d <= maximum))
            )
        ]
        passed = len(neighbors) >= integer(p["minimum_results"], "Minimum results", 1, 10000)
        return Result(
            {
                "neighbors": neighbors,
                "label": neighbors[0]["label"] if neighbors else None,
                "count": len(neighbors),
            },
            passed=passed,
            message=f"{len(neighbors)} neighbors ({model['metric']})",
            fail_reasons=[] if passed else ["nearest_neighbors"],
        )


class SaveAnnoy(Block):
    type_id, title = "ml.annoy_save", "Save Annoy index"
    category, block_type, subtype = "Output", "Output", "ML models"
    inputs = (Port("model", "data"),)
    outputs = (Port("path", "data"),)
    defaults = {"path": "neighbors.json"}

    def process(self, inputs):
        model = inputs["model"]
        count, dimensions = index_metadata(model)
        path = local_path(self, self.parameters["path"])
        index_path = path.with_suffix(".ann")
        if path == index_path or path.exists() or index_path.exists():
            raise FileExistsError(
                "Choose a new .json bundle path; existing artifacts are preserved"
            )
        index = native_index(model)
        fd, temporary = tempfile.mkstemp(prefix=".annoy-", suffix=".ann", dir=path.parent)
        os.close(fd)
        try:
            if not index.save(temporary):
                raise ValueError("Unable to save Annoy index")
            metadata = {
                "model": "annoy_file_v1",
                "index_path": index_path.name,
                "sha256": digest_file(Path(temporary)),
                "count": count,
                "dimensions": dimensions,
                "metric": model["metric"],
                "normalize": model["normalize"],
                "labels": model["labels"],
            }
            # Hard-link publication refuses replacement and keeps the native bytes intact.
            os.link(temporary, index_path)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, allow_nan=False)
        finally:
            index.unload()
            Path(temporary).unlink(missing_ok=True)
        return Result({"path": str(path)}, message=f"Saved {count} vectors")


class LoadAnnoy(Block):
    type_id, title = "ml.annoy_load", "Load Annoy index"
    category, block_type, subtype = "ML", "ML", "Vector search"
    outputs = (Port("model", "data"),)
    defaults = {"path": "", "prefault": False}

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        model = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(model, dict) or model.get("model") != "annoy_file_v1":
            raise ValueError("Select an Annoy JSON bundle, not a bare .ann file")
        model = copy.deepcopy(model)
        index_path = Path(model["index_path"])
        model["index_path"] = str((path.parent / index_path).resolve())
        model["prefault"] = self.parameters["prefault"]
        # Validate metadata here; the search block verifies binary integrity before mmap.
        index_metadata(model)
        return Result({"model": model})


ANNOY_BLOCKS = (VectorData, BuildAnnoy, SearchAnnoy, SaveAnnoy, LoadAnnoy)
