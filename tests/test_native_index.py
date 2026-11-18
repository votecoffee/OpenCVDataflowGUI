"""Native image retrieval metrics, bounded mmap lifecycle and portable bundle contracts."""

import base64
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.image_hash import ComputeImageHash
from opencv_dataflow.native_index import (
    BuildNativeANN,
    LoadNativeANN,
    NativeHandle,
    SaveNativeANN,
    SearchNativeANN,
    read_model,
)
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock, ticket_key

METRICS = ("euclidean", "manhattan", "angular", "dot", "hamming")


def execute(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def build(metric="euclidean", mode="memory", values=None, **settings):
    values = [[1, 0], [0, 1], [1, 1], [3, 1]] if values is None else values
    block = BuildNativeANN(
        parameters={
            "metric": metric,
            "build_mode": mode,
            "item_ids": [f"image-{i}" for i in range(len(values))],
            **settings,
        }
    )
    try:
        return execute(block, {"samples": values}).outputs["model"]
    finally:
        block.close()


def exact(values, q, metric):
    # Independent scalar references; do not use the application's reranking helpers.
    if metric == "hamming":
        return [sum((int(a) ^ int(b)).bit_count() for a, b in zip(row, q)) for row in values]
    if metric == "dot":
        return [sum(a * b for a, b in zip(row, q)) for row in values]
    if metric == "manhattan":
        return [sum(abs(a - b) for a, b in zip(row, q)) for row in values]
    if metric == "euclidean":
        return [sum((a - b) ** 2 for a, b in zip(row, q)) ** 0.5 for row in values]
    return [
        max(
            0,
            2
            - 2
            * sum(a * b for a, b in zip(row, q))
            / (sum(a * a for a in row) ** 0.5 * sum(b * b for b in q) ** 0.5),
        )
        ** 0.5
        for row in values
    ]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_native_all_candidate_rankings_match_independent_metrics_and_file_reload(
    metric, mode, tmp_path
):
    model = build(metric, mode)
    search = SearchNativeANN(parameters={"neighbors": 4, "candidates": 4, "search_k": 10000})
    try:
        result = execute(search, {"model": model, "samples": [[0, 1], [1, 1]]})
        for query, neighbors in zip([[0, 1], [1, 1]], result.outputs["neighbors"], strict=True):
            distances = exact(model["samples"], query, metric)
            order = sorted(
                range(4),
                key=lambda i: (
                    -distances[i] if metric == "dot" else distances[i],
                    model["item_ids"][i],
                ),
            )
            assert [n["index"] for n in neighbors] == order
            np.testing.assert_allclose(
                [n["similarity"] if metric == "dot" else n["distance"] for n in neighbors],
                np.array(distances)[order],
                atol=1e-12,
            )
        path = tmp_path / "index.json"
        execute(SaveNativeANN(parameters={"path": str(path)}), {"model": model})
        loaded = execute(LoadNativeANN(parameters={"path": str(path)}), {}).outputs["model"]
        assert loaded == model
        assert (
            execute(search, {"model": loaded, "samples": [0, 1]}).outputs["neighbors"][0]
            == result.outputs["neighbors"][0]
        )
        handle, directory = search._handle, Path(search._handle.directory.name)
        search.close()
        assert handle.index is None and not directory.exists()
    finally:
        search.close()


def test_full_256_bit_descriptors_use_correct_native_euclidean_bit_embedding():
    values = [[0] * 32, [255] * 32, [15] * 32]
    model = build("hamming", values=values)
    block = SearchNativeANN(parameters={"neighbors": 3, "candidates": 3, "search_k": 10000})
    try:
        result = execute(block, {"model": model, "samples": [0] * 32})
        assert [n["distance"] for n in result.outputs["neighbors"][0]] == [0, 128, 256]
        assert result.outputs["report"]["effective_native_metric"] == "euclidean_bits"
        assert model["native"]["dimensions"] == 256
    finally:
        block.close()


def test_native_small_gallery_tree_count_after_reload_is_recorded_not_assumed():
    model = build()
    assert model["settings"]["trees"] == 10
    assert 10 <= model["native"]["trees"] <= 20
    handle = NativeHandle(model, read_model(model)[1], False)
    try:
        assert handle.index.getTreeNumber() == model["native"]["trees"]
    finally:
        handle.close()


@pytest.mark.parametrize(
    "bad",
    [
        "revision",
        "native_hash",
        "native_size",
        "base64",
        "abi",
        "ids",
        "dimensions",
        "normalization",
        "contract",
        "byte_values",
    ],
)
def test_model_corruption_rejects_before_native_load(monkeypatch, bad):
    model = copy.deepcopy(build())
    if bad == "revision":
        model["revision"] = "0" * 64
    elif bad == "native_hash":
        model["native"]["sha256"] = "0" * 64
    elif bad == "native_size":
        model["native"]["bytes"] += 1
    elif bad == "base64":
        model["native"]["data"] = "!not base64!"
    elif bad == "abi":
        model["native"]["abi"]["opencv"] = "other"
    elif bad == "ids":
        model["item_ids"][1] = model["item_ids"][0]
    elif bad == "dimensions":
        model["native"]["dimensions"] += 1
    elif bad == "normalization":
        model["normalization"] = "l2"
    elif bad == "contract":
        model["feature_contract"] = "not a contract"
    else:
        model["metric"] = "hamming"
        model["samples"][0][0] = 0.5
    if bad != "revision":
        model["revision"] = model_revision(model)
    monkeypatch.setattr(
        cv2.ANNIndex, "create", lambda *a: pytest.fail("Bad model reached native import")
    )
    block = SearchNativeANN()
    assert (
        block.execute({"model": model, "samples": [0, 1]}) is None and not block.current_emissions
    )
    assert block._handle is None


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (BuildNativeANN, {"trees": 0}),
        (BuildNativeANN, {"seed": True}),
        (BuildNativeANN, {"metric": "hamming", "normalization": "l2"}),
        (BuildNativeANN, {"item_ids": [""]}),
        (SearchNativeANN, {"search_k": 0}),
        (SearchNativeANN, {"candidates": 1}),
        (SearchNativeANN, {"filter_results": 1}),
        (SearchNativeANN, {"minimum_results": 6}),
        (LoadNativeANN, {"prefault": 1}),
    ],
)
def test_parameter_validation_names_invalid_controls_without_execution(cls, parameters):
    with pytest.raises(ValidationError):
        cls(parameters=parameters).validate_parameters()


def test_contracts_batch_ids_normalization_and_stable_mapping_after_reorder():
    contract = {"model": "inspection_descriptors", "revision": 1}
    group = FlowGroup(
        tuple(ValuePacket(v, {"feature_contract": contract}) for v in [[3, 0], [0, 2], [2, 2]]),
        ("part-z", "part-a", "part-m"),
    )
    trainer = BuildNativeANN(parameters={"normalization": "l2"})
    model = execute(trainer, {"samples": group}).outputs["model"]
    assert model["item_ids"] == list(group.labels) and model["feature_contract"] == contract
    query = ValuePacket([3, 0], {"feature_contract": contract})
    search = SearchNativeANN(parameters={"neighbors": 3, "candidates": 3})
    try:
        first = execute(search, {"model": model, "samples": query})
        assert first.outputs["item_id"] == "part-z"
        reverse = FlowGroup(group.items[::-1], group.labels[::-1])
        second_model = execute(trainer, {"samples": reverse}).outputs["model"]
        assert (
            execute(search, {"model": second_model, "samples": query}).outputs["item_id"]
            == "part-z"
        )
        assert search.execute({"model": model, "samples": [3, 0]}) is None
        search.parameters["contract_policy"] = "dimensions_only"
        assert execute(search, {"model": model, "samples": [3, 0]}).passed
    finally:
        search.close()
        trainer.close()


def test_query_judgment_current_image_context_graphics_lean_and_cache():
    model = build()
    block = SearchNativeANN(parameters={"neighbors": 4, "candidates": 4})
    context = {
        "ticket": TicketClock().issue("current"),
        "spatial": frame([12, 10], "camera"),
        "feature_contract": None,
    }
    inputs = {
        "model": model,
        "samples": ValuePacket([0, 1], context),
        "image": ValuePacket(np.full((10, 12), 77, np.uint16), context),
    }
    try:
        result = execute(block, inputs)
        assert result.drawings and result.outputs["image"].dtype == np.uint16
        assert ticket_key(result.payloads["neighbors"]["ticket"]) == ticket_key(context["ticket"])
        handle = block._handle
        assert (
            execute(block, inputs).outputs["report"]["native_index_reused"]
            and block._handle is handle
        )
        block.run_profile = profile("lean")
        assert not execute(block, inputs).drawings
        block.parameters.update(filter_results=True, maximum_distance=0, minimum_results=2)
        failed = execute(block, inputs)
        assert not failed.passed and failed.outputs["item_id"] is NO_VALUE
        inputs["image"].payload = {**context, "ticket": TicketClock().issue("wrong")}
        assert block.execute(inputs) is None and not block.current_emissions
    finally:
        block.close()


def test_native_short_results_are_prefilled_not_read_from_uninitialized_slots():
    model = build()
    block = SearchNativeANN(parameters={"neighbors": 4, "candidates": 4})
    try:
        execute(block, {"model": model, "samples": [0, 1]})

        class Short:
            def knnSearch(self, query, k, *, indices, dists, search_k):
                indices[0, 0] = 1
                dists[0, 0] = 0
                return indices, dists

        block._handle.index = Short()
        result = execute(block, {"model": model, "samples": [0, 1]})
        assert result.outputs["report"]["missing_candidates"] == [3]
        assert result.outputs["count"] == 1 and result.outputs["item_id"] == "image-1"
    finally:
        block.close()


def test_build_and_query_budgets_native_failure_and_recovery(monkeypatch):
    tiny = BuildNativeANN(parameters={"maximum_index_bytes": 1})
    assert tiny.execute({"samples": [[1, 2], [2, 3]]}) is None
    assert tiny._model is None
    for samples in ([[True, False]], [[float("nan")]], [[1e8]], [[]]):
        assert BuildNativeANN().execute({"samples": samples}) is None
    model = build()
    block = SearchNativeANN()
    create = cv2.ANNIndex.create
    excessive = SearchNativeANN(parameters={"search_k": 10_000_000})
    assert excessive.execute({"model": model, "samples": [[0, 1]] * 7}) is None
    assert excessive._handle is None

    def broken(*args):
        raise RuntimeError("native failure")

    monkeypatch.setattr(cv2.ANNIndex, "create", broken)
    assert block.execute({"model": model, "samples": [0, 1]}) is None
    assert block._handle is None
    monkeypatch.setattr(cv2.ANNIndex, "create", create)
    try:
        assert execute(block, {"model": model, "samples": [0, 1]}).passed
    finally:
        block.close()


def test_hash_records_connect_directly_and_preserve_method_contract():
    hasher = ComputeImageHash(parameters={"method": "average"})
    images = [
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        np.arange(63, -1, -1, dtype=np.uint8).reshape(8, 8),
    ]
    hashes = [execute(hasher, {"image": image}).outputs["hash"] for image in images]
    group = FlowGroup(tuple(ValuePacket(h, {}) for h in hashes), ("ascending", "descending"))
    trainer = BuildNativeANN(parameters={"metric": "hamming"})
    model = execute(trainer, {"samples": group}).outputs["model"]
    block = SearchNativeANN(parameters={"neighbors": 2, "candidates": 2})
    try:
        result = execute(block, {"model": model, "samples": hashes[1]})
        assert result.outputs["item_id"] == "descending"
        difference = (int(hashes[0]["hex"], 16) ^ int(hashes[1]["hex"], 16)).bit_count()
        assert [n["distance"] for n in result.outputs["neighbors"][0]] == [0, difference]
        other = ComputeImageHash().execute({"image": images[0]}).outputs["hash"]
        assert block.execute({"model": model, "samples": other}) is None
    finally:
        trainer.close()
        block.close()


def test_bundle_load_rejects_duplicate_keys_and_preserves_existing_save(tmp_path):
    path = tmp_path / "index.json"
    model = build()
    save = SaveNativeANN(parameters={"path": str(path)})
    execute(save, {"model": model})
    before = path.read_bytes()
    assert save.execute({"model": model}) is None and path.read_bytes() == before
    path.write_text('{"model":1,"model":2}', encoding="utf-8")
    assert LoadNativeANN(parameters={"path": str(path)}).execute({}) is None
    assert base64.b64decode(model["native"]["data"])
    assert json.loads(before)["revision"] == model["revision"]
