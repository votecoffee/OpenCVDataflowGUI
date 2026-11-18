"""Local ONNX feature grids and sealed known-good nearest-reference anomaly banks."""

import copy
import hashlib
import json

import cv2
import numpy as np

from .classifier_common import canonical_json, model_revision, normalize_features
from .core import Block, Port, Result
from .detection_decode import check_context, preprocessing_geometry
from .graphics_layers import GraphicDrawing, graphic_object
from .ml_common import local_path, selected_tensor
from .parameter_validation import NumericRange, ValidationError
from .payloads import FlowGroup, ValuePacket, require_same_frame
from .processing import image_value
from .regression import receipt_identity
from .selection import SELECTION_INPUTS, selection_mask
from .semantic_segmentation import source_coordinates
from .tickets import ticket_key

MAX_ELEMENTS = 1_048_576
MAX_BYTES = 32 * 1024**2
GRID = "local_feature_grid_v1"
BANK = "local_anomaly_bank_v1"


def sha(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def hash_image(image):
    header = canonical_json([str(image.dtype), list(image.shape)]).encode()
    return hashlib.sha256(header + np.ascontiguousarray(image).tobytes()).hexdigest()


def is_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def matrix(value, maximum_rows, maximum_dimensions=256):
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= maximum_rows
        or not isinstance(value[0], list)
        or not 1 <= len(value[0]) <= maximum_dimensions
    ):
        raise ValidationError("Use bounded nonempty local feature rows")
    width = len(value[0])
    if len(value) * width > MAX_ELEMENTS or any(
        not isinstance(row, list)
        or len(row) != width
        or any(type(v) not in (int, float) for v in row)
        for row in value
    ):
        raise ValidationError("Local features exceed rectangular numeric/element limits")
    wide = np.asarray(value, np.float64)
    if not np.isfinite(wide).all() or np.any(np.abs(wide) > 1e6):
        raise ValidationError("Local features must be finite within +/-1e6")
    result = np.ascontiguousarray(wide, np.float32)
    if not np.array_equal(wide, result.astype(np.float64)):
        raise ValidationError("Local features must retain exact float32 values")
    return result


def validate_contract(contract, values):
    if (
        not isinstance(contract, dict)
        or len(canonical_json(contract)) > 16384
        or type(contract.get("dimensions")) is not int
        or contract["dimensions"] != values.shape[1]
        or contract.get("normalization") not in ("none", "l2")
        or contract.get("layout") not in ("NCHW", "NHWC")
        or contract.get("grid_convention") != "uniform_full_network_pixel_edges"
        or not isinstance(contract.get("preprocessing"), dict)
        or not is_hash(contract.get("model_revision"))
        and not (
            isinstance(contract.get("model_revision"), str)
            and contract["model_revision"].startswith("sha256:")
            and is_hash(contract["model_revision"][7:])
        )
    ):
        raise ValidationError("Invalid local feature/model contract")
    shape = contract.get("grid_size")
    name = contract.get("output_name")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(v) is not int or not 1 <= v <= 4096 for v in shape)
        or not 2 <= shape[0] * shape[1] <= 4096
        or not isinstance(name, str)
        or not 1 <= len(name) <= 128
    ):
        raise ValidationError("Invalid local feature grid/head contract")
    if contract["normalization"] == "l2" and not np.allclose(
        np.linalg.norm(values.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0
    ):
        raise ValidationError("Local L2 vectors must have unit length")


def validate_grid(grid):
    if not isinstance(grid, dict) or grid.get("model") != GRID:
        raise ValidationError("Connect a decoded local feature grid")
    values = matrix(grid.get("features"), 4096)
    shape, valid = grid.get("grid_size"), grid.get("valid")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(v) is not int or not 1 <= v <= 4096 for v in shape)
        or shape[0] * shape[1] != len(values)
        or not isinstance(valid, list)
        or len(valid) != len(values)
        or any(type(v) is not bool for v in valid)
        or not any(valid)
    ):
        raise ValidationError("Feature grid dimensions/selected cells differ or selection is empty")
    contract = grid.get("feature_contract")
    validate_contract(contract, values)
    if contract.get("grid_size") != shape or not is_hash(grid.get("source_sha256")):
        raise ValidationError("Local feature grid/source identity differs")
    return values, np.array(valid), contract


def read_bank(model):
    keys = {
        "model",
        "samples",
        "feature_contract",
        "metric",
        "training_sources",
        "candidate_rows",
        "sampling",
        "revision",
    }
    if not isinstance(model, dict) or set(model) != keys or model.get("model") != BANK:
        raise ValidationError("Connect a sealed local anomaly reference bank")
    samples = matrix(model["samples"], 8192)
    contract = model["feature_contract"]
    if (
        not isinstance(contract, dict)
        or len(canonical_json(contract)) > 16384
        or contract.get("dimensions") != samples.shape[1]
        or model["metric"] not in ("euclidean", "manhattan")
        or model["sampling"] not in ("all", "even_stride")
        or type(model["candidate_rows"]) is not int
        or not len(samples) <= model["candidate_rows"] <= 32768
    ):
        raise ValidationError("Reference-bank feature/distance/sampling contract differs")
    sources = model["training_sources"]
    if (
        not isinstance(sources, list)
        or not 1 <= len(sources) <= 64
        or any(not is_hash(s) for s in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValidationError("Reference-bank training source hashes differ")
    validate_contract(contract, samples)
    if len(canonical_json(model).encode()) > MAX_BYTES or model["revision"] != model_revision(
        model
    ):
        raise ValidationError("Reference-bank byte limit or content revision differs")
    return samples


def associated(block, grid):
    context = grid.get("context", {})
    image_context = block.input_payloads.get("image", {})
    require_same_frame(
        [ValuePacket(None, context), ValuePacket(None, image_context)],
        message="Local feature grid belongs to another image frame",
    )
    for key in ("ticket", "recipe_ref", "pyramid"):
        a, b = context.get(key), image_context.get(key)
        if (
            a is not None
            and b is not None
            and (ticket_key(a) != ticket_key(b) if key == "ticket" else a != b)
        ):
            raise ValidationError("Local feature grid has another " + key)


class _LocalAnomaly(Block):
    category = block_type = "ML"
    subtype = "Local anomalies"
    preview_safe = parallel_safe = process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key, rule in self.parameter_ranges.items():
            if type(p[key]) not in (int, float) or rule.whole and type(p[key]) is not int:
                raise ValidationError("Use an integer" if rule.whole else "Use numeric data", key)
        for key, default in self.defaults.items():
            if type(default) is bool and type(p[key]) is not bool:
                raise ValidationError("Use a checkbox boolean", key)


class DecodeLocalFeatures(_LocalAnomaly):
    type_id, title = "ml.local_features", "Decode local feature grid"
    inputs = (
        *SELECTION_INPUTS,
        Port("tensors", "data"),
        Port("preprocessing", "data"),
        Port("receipt", "data"),
    )
    outputs = (Port("image"), Port("features", "data"), Port("report", "data"))
    defaults = {
        "output_name": "",
        "layout": "NCHW",
        "normalization": "none",
        "maximum_image_pixels": 4_194_304,
        "maximum_feature_elements": 262144,
    }
    parameter_choices = {"layout": ("NCHW", "NHWC"), "normalization": ("none", "l2")}
    parameter_ranges = {
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "maximum_feature_elements": NumericRange(1, MAX_ELEMENTS, whole=True),
    }
    parameter_help = {
        "layout": "One image with uniformly spaced cells across the full network field. "
        "Declare the exported feature layout; arbitrary receptive-field offsets are not inferred.",
        "normalization": "Per-cell L2 or unchanged feature units; zero L2 vectors are invalid.",
        "output_name": "Spatial embedding output; pooled descriptors are not local features.",
    }
    input_help = {
        "mask": "Select training/query cell centers; neighborhoods may still see excluded pixels.",
        "receipt": "Required execution identity; detailed single-image output hashes are checked.",
    }

    def process(self, inputs):
        p, source = self.parameters, inputs["image"]
        if source.shape[0] * source.shape[1] > p["maximum_image_pixels"]:
            raise ValidationError("Image exceeds the pixel budget", "maximum_image_pixels")
        image = image_value(source)
        checks = check_context(self, kind="Local feature")
        size, reverse = preprocessing_geometry(inputs["preprocessing"], image)
        tensors = inputs["tensors"]
        name = p["output_name"] or next(iter(tensors))
        record = tensors.get(name)
        if not isinstance(record, dict) or record.get("dtype") != "float32":
            raise ValidationError("Use a float32 tensor_v1 spatial embedding")
        flat = record.get("values")
        if (
            not isinstance(flat, list)
            or not 1 <= len(flat) <= p["maximum_feature_elements"]
            or any(type(v) not in (int, float) for v in flat)
        ):
            raise ValidationError("Use bounded flat numeric feature values")
        raw = selected_tensor(tensors, name, max_elements=p["maximum_feature_elements"])
        if raw.ndim != 4 or raw.shape[0] != 1:
            raise ValidationError("Local features require one NCHW/NHWC spatial image", "layout")
        values = raw[0] if p["layout"] == "NCHW" else raw[0].transpose(2, 0, 1)
        c, h, w = values.shape
        if not 1 <= c <= 256 or h * w > 4096 or h * w < 2 or np.any(np.abs(values) > 1e6):
            raise ValidationError("Use 2..4096 local cells, 1..256 channels within +/-1e6")
        identity = receipt_identity(inputs["receipt"], name, raw, inputs["preprocessing"])
        receipt_context = self.input_payloads.get("receipt", {})
        require_same_frame(
            [ValuePacket(None, receipt_context), source], message="Receipt frame differs"
        )
        a, b = receipt_context.get("ticket"), self.input_payloads["image"].get("ticket")
        if a is not None and b is not None and ticket_key(a) != ticket_key(b):
            raise ValidationError("Receipt belongs to another ticket")
        rows = normalize_features(np.ascontiguousarray(values.reshape(c, -1).T), p["normalization"])
        sx = ((np.arange(w) + 0.5) * size[0] / w) * reverse[0, 0] + reverse[0, 2]
        sy = ((np.arange(h) + 0.5) * size[1] / h) * reverse[1, 1] + reverse[1, 2]
        # Continuous pixel-edge coordinates: padded cells outside the source are excluded.
        valid = (
            (sy[:, None] >= 0)
            & (sy[:, None] < image.shape[0])
            & (sx[None, :] >= 0)
            & (sx[None, :] < image.shape[1])
        )
        selection = selection_mask(inputs, image)
        if selection is not None:
            xi = np.floor(sx).astype(int).clip(0, image.shape[1] - 1)
            yi = np.floor(sy).astype(int).clip(0, image.shape[0] - 1)
            valid &= selection[yi[:, None], xi] != 0
        contract = {
            "model_revision": identity["model_revision"],
            "output_name": name,
            "layout": p["layout"],
            "normalization": p["normalization"],
            "dimensions": c,
            "grid_size": [w, h],
            "preprocessing": copy.deepcopy(inputs["preprocessing"].get("preprocessing_contract")),
            "grid_convention": "uniform_full_network_pixel_edges",
        }
        grid = {
            "model": GRID,
            "features": rows.tolist(),
            "grid_size": [w, h],
            "valid": valid.ravel().tolist(),
            "feature_contract": contract,
            "source_sha256": hash_image(source),
            "context": copy.deepcopy(self.input_payloads["image"]),
            "preprocessing": copy.deepcopy(inputs["preprocessing"]),
        }
        validate_grid(grid)
        return Result(
            {
                "image": source,
                "features": grid,
                "report": {
                    "identity": identity,
                    "context_checks": checks,
                    "selected_cells": int(valid.sum()),
                },
            },
            message=f"{int(valid.sum())} local feature cells, {c} dimensions",
        )


class TrainLocalAnomaly(_LocalAnomaly):
    type_id, title = "ml.train_local_anomaly", "Train local anomaly reference"
    inputs = (Port("features", "any"),)
    outputs = (Port("model", "data"), Port("report", "data"))
    defaults = {
        "metric": "euclidean",
        "sampling": "all",
        "bank_limit": 2048,
        "maximum_candidate_rows": 32768,
    }
    parameter_choices = {"metric": ("euclidean", "manhattan"), "sampling": ("all", "even_stride")}
    parameter_ranges = {
        "bank_limit": NumericRange(1, 8192, whole=True),
        "maximum_candidate_rows": NumericRange(1, 32768, whole=True),
    }
    parameter_help = {
        "sampling": "All rejects overflow. Even stride explicitly discards reference cells "
        "in deterministic source/raster order; not a representative coreset.",
        "metric": "Raw nearest-reference distance; not a calibrated probability.",
    }
    input_help = {
        "features": "One grid, a list, or FlowGroup of known-good grids; no defect labels "
        "are inferred. Duplicate source pixel hashes are rejected."
    }

    def process(self, inputs):
        p, data = self.parameters, inputs["features"]
        grids = (
            [item.value for item in data.items]
            if isinstance(data, FlowGroup)
            else [data]
            if isinstance(data, dict)
            else data
        )
        if not isinstance(grids, list) or not 1 <= len(grids) <= 64:
            raise ValidationError("Supply 1..64 known-good feature grids")
        candidates, sources, contracts = [], [], []
        total = 0
        for grid in grids:
            rows, valid, contract = validate_grid(grid)
            total += int(valid.sum())
            if total > p["maximum_candidate_rows"] or total * rows.shape[1] > MAX_ELEMENTS:
                raise ValidationError("Training candidates exceed row/element budgets")
            candidates.append(rows[valid])
            sources.append(grid["source_sha256"])
            contracts.append(contract)
        if len(set(sources)) != len(sources):
            raise ValidationError("Duplicate known-good source image bytes")
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValidationError("Training model/preprocessing/local feature contracts differ")
        rows = np.concatenate(candidates)
        if len(rows) > p["bank_limit"]:
            if p["sampling"] == "all":
                raise ValidationError(
                    "Reference cells exceed Bank limit; choose sampling explicitly", "bank_limit"
                )
            rows = rows[np.linspace(0, len(rows) - 1, p["bank_limit"]).astype(int)]
        model = {
            "model": BANK,
            "samples": rows.tolist(),
            "feature_contract": copy.deepcopy(contracts[0]),
            "metric": p["metric"],
            "training_sources": sources,
            "candidate_rows": total,
            "sampling": p["sampling"],
        }
        model["revision"] = model_revision(model)
        read_bank(model)
        return Result(
            {
                "model": model,
                "report": {
                    "training_images": len(sources),
                    "candidate_rows": total,
                    "retained_rows": len(rows),
                    "revision": model["revision"],
                    "known_good_labels_verified": False,
                    "capture_independence_verified": False,
                },
            },
            message=f"Trained {len(rows)} reference cells from {len(sources)} images",
        )


class InspectLocalAnomaly(_LocalAnomaly):
    type_id, title = "ml.local_anomaly", "Inspect local image anomalies"
    inputs = (*SELECTION_INPUTS, Port("features", "data"), Port("model", "data"))
    outputs = (
        Port("image"),
        Port("distance"),
        Port("support"),
        Port("mask"),
        Port("maximum_distance", "number"),
        Port("anomaly_pixels", "integer"),
        Port("report", "data"),
    )
    defaults = {
        "distance_threshold": 0.2,
        "allowed_anomaly_pixels": 0,
        "minimum_support_fraction": 0.95,
        "reject_training_image": False,
        "maximum_distance_work": 64_000_000,
        "query_batch_size": 64,
        "maximum_image_pixels": 4_194_304,
        "graphics_maximum_contours": 200,
        "graphics_maximum_points": 100000,
    }
    parameter_ranges = {
        "distance_threshold": NumericRange(0, 512e6),
        "allowed_anomaly_pixels": NumericRange(0, 16_777_216, whole=True),
        "minimum_support_fraction": NumericRange(0, 1),
        "maximum_distance_work": NumericRange(1, 256_000_000, whole=True),
        "query_batch_size": NumericRange(1, 512, whole=True),
        "maximum_image_pixels": NumericRange(1, 16_777_216, whole=True),
        "graphics_maximum_contours": NumericRange(1, 2000, whole=True),
        "graphics_maximum_points": NumericRange(3, 100000, whole=True),
    }
    parameter_help = {
        "distance_threshold": "Anomalous when distance > threshold; equality is accepted.",
        "allowed_anomaly_pixels": "Pass when source-grid anomalous pixels <= this count.",
        "minimum_support_fraction": "Reject insufficient scored support in the selected region.",
        "reject_training_image": "Reject identical source pixel bytes used for training; "
        "different bytes do not establish an independent acquisition.",
        "maximum_distance_work": "Maximum query cells x bank rows x dimensions before search.",
    }

    def process(self, inputs):
        p, source, grid, model = (
            self.parameters,
            inputs["image"],
            inputs["features"],
            inputs["model"],
        )
        image = image_value(source)
        if image.shape[0] * image.shape[1] > p["maximum_image_pixels"]:
            raise ValidationError("Image exceeds the pixel budget", "maximum_image_pixels")
        rows, valid, contract = validate_grid(grid)
        bank = read_bank(model)
        associated(self, grid)
        if contract != model["feature_contract"]:
            raise ValidationError("Query local feature/model/preprocessing contract differs")
        if hash_image(source) != grid["source_sha256"]:
            raise ValidationError("Feature source pixel bytes differ from the inspection image")
        reused = grid["source_sha256"] in model["training_sources"]
        if reused and p["reject_training_image"]:
            raise ValidationError(
                "Query image is identical to a training image", "reject_training_image"
            )
        query = rows[valid]
        if query.size * len(bank) > p["maximum_distance_work"]:
            raise ValidationError(
                "Nearest-reference work exceeds the configured budget", "maximum_distance_work"
            )
        distances, nearest = [], []
        norm = cv2.NORM_L2 if model["metric"] == "euclidean" else cv2.NORM_L1
        for start in range(0, len(query), p["query_batch_size"]):
            d, index = cv2.batchDistance(
                query[start : start + p["query_batch_size"]], bank, cv2.CV_32F, normType=norm, K=1
            )
            distances.extend(d[:, 0].tolist())
            nearest.extend(index[:, 0].tolist())
        if not np.isfinite(distances).all():
            raise ValidationError("Nearest-reference search produced nonfinite distances")
        w, h = grid["grid_size"]
        native = np.zeros(len(rows), np.float32)
        native[valid] = distances
        size, reverse = preprocessing_geometry(grid["preprocessing"], image)
        x, y = source_coordinates(size, reverse, image.shape, (h, w))
        xi, yi = np.floor(x).astype(int).clip(0, w - 1), np.floor(y).astype(int).clip(0, h - 1)
        mapped = native.reshape(h, w)[yi[:, None], xi]
        selection = selection_mask(inputs, image)
        selected = np.ones(image.shape[:2], bool) if selection is None else selection != 0
        support = valid.reshape(h, w)[yi[:, None], xi] & selected
        count = int(support.sum())
        fraction = count / max(1, int(selected.sum()))
        if count == 0 or fraction < p["minimum_support_fraction"]:
            raise ValidationError(
                "No/insufficient scored source support", "minimum_support_fraction"
            )
        mapped[~support] = 0
        mask = (mapped > p["distance_threshold"]) & support
        pixels, maximum = int(mask.sum()), float(mapped[support].max())
        passed = pixels <= p["allowed_anomaly_pixels"]
        report = {
            "model": "local_anomaly_report_v1",
            "reference_revision": model["revision"],
            "decision_revision": sha(
                {
                    "reference": model["revision"],
                    "distance_threshold": p["distance_threshold"],
                    "allowed_anomaly_pixels": p["allowed_anomaly_pixels"],
                    "minimum_support_fraction": p["minimum_support_fraction"],
                }
            ),
            "metric": model["metric"],
            "score_is_probability": False,
            "selected_pixels": int(selected.sum()),
            "scored_pixels": count,
            "support_fraction": fraction,
            "maximum_distance": maximum,
            "anomaly_pixels": pixels,
            "native_distances": distances,
            "nearest_reference_rows": nearest,
            "training_image_reused": reused,
            "capture_independence_verified": False,
            "resampling": "nearest_full_network_cells",
        }
        drawings, graphics = {}, {}
        if self.generate_graphics:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            color = (80, 210, 100) if passed else (70, 70, 235)
            objects = [
                graphic_object(
                    "label",
                    [(8, 18)],
                    color,
                    text=f"Anomaly pixels {pixels}; max distance {maximum:.4g}",
                )
            ]
            remaining = p["graphics_maximum_points"]
            for contour in contours[: p["graphics_maximum_contours"]]:
                if remaining < 3:
                    break
                points = contour.reshape(-1, 2)
                if len(points) > remaining:
                    points = points[np.linspace(0, len(points) - 1, remaining).astype(int)]
                remaining -= len(points)
                if len(contour) >= 2:
                    objects.append(graphic_object("polyline", points.tolist(), color, closed=True))
            drawings["local anomalies"] = GraphicDrawing(tuple(objects))
            # Fixed threshold-relative colors make runs comparable; no per-image autoscaling.
            heat = np.uint8(np.clip(mapped / max(p["distance_threshold"] * 2, 1e-12), 0, 1) * 255)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            heat[~support] = 0
            graphics["anomaly distance"] = heat
        context = copy.deepcopy(self.input_payloads["image"])
        return Result(
            {
                "image": source,
                "distance": mapped,
                "support": support.astype(np.uint8) * 255,
                "mask": mask.astype(np.uint8) * 255,
                "maximum_distance": maximum,
                "anomaly_pixels": pixels,
                "report": report,
            },
            payloads={k: copy.deepcopy(context) for k in ("image", "distance", "support", "mask")},
            passed=passed,
            drawings=drawings,
            graphics=graphics,
            fail_reasons=[]
            if passed
            else [
                {"code": "anomaly.area", "message": "Anomalous area exceeds the configured limit"}
            ],
            message=f"{pixels} anomaly pixels; maximum distance {maximum:.5g}",
        )


class SaveLocalAnomaly(_LocalAnomaly):
    type_id, title = "ml.save_local_anomaly", "Save local anomaly reference"
    category = block_type = "Output"
    inputs, outputs = (Port("model", "data"),), (Port("path", "text"),)
    defaults = {"path": "local-anomaly.json"}
    preview_safe = False

    def process(self, inputs):
        model = inputs["model"]
        read_bank(model)
        path = local_path(self, self.parameters["path"])
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(model) + "\n")
        return Result({"path": str(path)}, message="Saved reference bank; existing files preserved")


class LoadLocalAnomaly(_LocalAnomaly):
    type_id, title = "ml.load_local_anomaly", "Load local anomaly reference"
    outputs = (Port("model", "data"),)
    defaults = {"path": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")

    def process(self, inputs):
        path = local_path(self, self.parameters["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("Reference-bank file exceeds 32 MiB")

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValidationError("Duplicate reference-bank JSON key: " + key)
                result[key] = value
            return result

        def invalid(value):
            raise ValidationError("Nonfinite reference-bank JSON number: " + value)

        model = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
        read_bank(model)
        return Result({"model": model}, message=f"Loaded {len(model['samples'])} reference cells")


LOCAL_ANOMALY_BLOCKS = (
    DecodeLocalFeatures,
    TrainLocalAnomaly,
    InspectLocalAnomaly,
    SaveLocalAnomaly,
    LoadLocalAnomaly,
)
