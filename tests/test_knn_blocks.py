"""Native K-nearest parity, unknown-class policy and portable image-feature models."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.knn_blocks import DISTANCE_POLICIES
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock

X = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]
LABELS = ["small"] * 3 + ["large"] * 3


def execute(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def trained(samples=X, labels=LABELS, **parameters):
    return execute("ml.knn_train", {"samples": samples, "labels": labels}, **parameters)[1].outputs[
        "model"
    ]


@pytest.mark.parametrize("k", [1, 2, 3, 6])
@pytest.mark.parametrize("normalization", ["none", "l2"])
def test_classes_neighbors_votes_and_squared_distances_match_independent_native(k, normalization):
    samples = np.float32([[1, 0], [1, 0.1], [1, 0.2], [0, 1], [0.1, 1], [0.2, 1]])
    queries = np.float32([[0.99, 0.15], [0.15, 0.99]])
    model = trained(samples.tolist(), normalization=normalization, k=k)
    if normalization == "l2":
        samples /= np.linalg.norm(samples.astype(np.float64), axis=1, keepdims=True)
        queries /= np.linalg.norm(queries.astype(np.float64), axis=1, keepdims=True)
    native = cv2.ml.KNearest_create()
    native.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)
    native.setIsClassifier(True)
    native.setDefaultK(k)
    assert native.train(samples, cv2.ml.ROW_SAMPLE, np.float32([0, 0, 0, 1, 1, 1]))
    _, labels, neighbors, squared = native.findNearest(queries, k)
    _, result = execute(
        "ml.knn_predict",
        {"model": json.loads(json.dumps(model)), "samples": [[0.99, 0.15], [0.15, 0.99]]},
        k=k,
        reject_vote_ties=False,
        distance_policy="nearest",
    )
    assert result.outputs["labels"] == [model["class_labels"][int(i)] for i in labels.ravel()]
    assert result.outputs["label"] is NO_VALUE
    for row, expected, distances in zip(
        result.outputs["neighbors"], neighbors, squared, strict=True
    ):
        assert row["labels"] == [model["class_labels"][int(i)] for i in expected]
        np.testing.assert_allclose(row["squared_distances"], distances, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(
            row["distances"], np.sqrt(distances.astype(float)), rtol=2e-6, atol=1e-7
        )
    assert result.outputs["report"]["probabilities_available"] is False
    assert result.final_judgment is None and result.score is None


@pytest.mark.parametrize("policy", DISTANCE_POLICIES)
def test_distance_statistic_is_explicit_with_minority_closest_neighbor(policy):
    model = trained([[0], [0.5], [0.9], [0.05]], ["a", "a", "a", "b"], k=3)
    _, result = execute(
        "ml.knn_predict",
        {"model": model, "samples": [0.04]},
        distance_policy=policy,
        max_distance=0.02,
    )
    assert result.outputs["candidate_labels"] == ["a"]
    distances = result.outputs["neighbors"][0]["distances"]
    expected = {
        "winning_class_nearest": distances[1],
        "nearest": distances[0],
        "mean": sum(distances) / 3,
        "kth": distances[-1],
    }[policy]
    assert result.outputs["distances"] == pytest.approx([expected])
    assert result.passed is (policy == "nearest")
    if not result.passed:
        assert result.outputs["labels"] == [None] and result.outputs["label"] is NO_VALUE
        assert result.outputs["report"]["unknown_reasons"] == [["knn_distance"]]
        assert result.fail_reasons[0]["code"] == "knn_unknown_class"


def test_vote_cutoff_ties_and_native_first_appearance_class_order():
    model = trained([[0], [2]], ["z-first", "a-second"], k=2)
    block, result = execute("ml.knn_predict", {"model": model, "samples": [1]})
    assert result.outputs["candidate_labels"] == ["z-first"]
    assert result.outputs["labels"] == [None] and not result.passed
    assert result.outputs["report"]["unknown_reasons"] == [["knn_vote_tie"]]
    assert not block.current_emissions["label"]
    _, result = execute(
        "ml.knn_predict",
        {"model": model, "samples": [1]},
        reject_vote_ties=False,
        min_vote_fraction=0.5,
        max_distance=1,
    )
    assert result.passed and result.outputs["label"] == "z-first"
    _, result = execute(
        "ml.knn_predict",
        {"model": model, "samples": [1]},
        reject_vote_ties=False,
        min_vote_fraction=0.500001,
    )
    assert not result.passed and result.outputs["report"]["unknown_reasons"] == [
        ["knn_vote_fraction"]
    ]
    _, result = execute("ml.knn_predict", {"model": model, "samples": [1]}, k=1)
    assert result.outputs["label"] == "z-first"  # Equal-distance K-boundary keeps the earlier row.


def test_accepted_class_failure_is_still_a_known_label_and_single_class_is_supported():
    _, result = execute(
        "ml.knn_predict", {"model": trained(), "samples": [10.1, 10.1]}, accepted_labels=["small"]
    )
    assert not result.passed and result.outputs["label"] == "large"
    assert result.outputs["report"]["unknown_count"] == 0
    assert result.fail_reasons[0]["code"] == "knn_unaccepted_label"
    model = trained([[2, 3]], ["prototype"], k=1)
    _, result = execute("ml.knn_predict", {"model": model, "samples": [2, 3]}, max_distance=0.1)
    assert result.passed and result.outputs["label"] == "prototype"


def test_batched_contracts_current_ticket_image_graphics_and_lean():
    context = {"feature_contract": {"model": "test-image-feature", "settings": {"gray": True}}}
    rows = FlowGroup(tuple(ValuePacket(row, context) for row in X), tuple(map(str, range(6))))
    model = trained(rows)
    context = {
        **context,
        "ticket": TicketClock().issue("current"),
        "spatial": frame([32, 32], "camera"),
        "recipe_ref": {"id": "r", "revision": 1, "digest": "1" * 64},
    }
    image = np.full((32, 32), 100, np.uint16)
    inputs = {
        "samples": ValuePacket([0.1, 0.1], context),
        "image": ValuePacket(image, context),
        "model": model,
    }
    block, result = execute("ml.knn_predict", inputs)
    assert result.outputs["label"] == "small"
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert "small" in result.drawings["K-nearest class"].objects[0].text
    for port in ("label", "neighbors", "image", "report"):
        assert result.payloads[port]["ticket"] == context["ticket"]
        assert result.payloads[port]["spatial"] == context["spatial"]
        assert result.payloads[port]["recipe_ref"] == context["recipe_ref"]
    block.run_profile = profile("lean")
    assert not block.execute(inputs).drawings
    changed = copy.deepcopy(context)
    changed["feature_contract"]["settings"]["gray"] = False
    assert block.execute({**inputs, "samples": ValuePacket([0.1, 0.1], changed)}) is None
    assert "preprocessing differs" in block.error and not block.current_emissions
    block.parameters["contract_policy"] = "dimensions_only"
    assert block.execute({**inputs, "samples": ValuePacket([0.1, 0.1], changed)}) is not None, (
        block.error
    )
    bad = FlowGroup((ValuePacket([1, 0], context), ValuePacket([0, 1], changed)), ("a", "b"))
    block = builtin_registry().create("ml.knn_train", parameters={"labels": ["a", "b"], "k": 1})
    assert block.execute({"samples": bad}) is None and "different preprocessing" in block.error


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref", "pyramid"])
def test_optional_image_context_conflicts_are_rejected(field, monkeypatch):
    context = {
        "ticket": TicketClock().issue("a"),
        "spatial": frame([32, 32], "a"),
        "recipe_ref": {"id": "r", "revision": 1, "digest": "1" * 64},
        "pyramid": {"level": 0, "largest_size": [32, 32], "to_largest": np.eye(3).tolist()},
    }
    changed = copy.deepcopy(context)
    if field == "ticket":
        changed[field] = TicketClock().issue("b")
    elif field == "spatial":
        changed[field]["root_id"] = "b"
    elif field == "recipe_ref":
        changed[field]["revision"] = 2
    else:
        changed[field]["level"] = 1
    model = trained()

    def unexpected(*args):
        raise AssertionError("Invalid image context reached native work")

    monkeypatch.setattr("opencv_dataflow.knn_blocks.native_knn", unexpected)
    block = builtin_registry().create("ml.knn_predict")
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([0.1, 0.1], context),
                "image": ValuePacket(np.zeros((32, 32), np.uint8), changed),
            }
        )
        is None
    )
    assert not block.current_emissions
    assert (
        "different tickets"
        if field == "ticket"
        else "frame differs"
        if field == "spatial"
        else "recipe revisions"
        if field == "recipe_ref"
        else "conflicting " + field
    ) in block.error


@pytest.mark.parametrize(
    "samples", [[], [[1, 2], [3]], [[np.nan]], [[np.inf]], [[True]], [["a"]], [[1e16]], [[1e-300]]]
)
def test_invalid_descriptors_emit_no_model(samples):
    block = builtin_registry().create("ml.knn_train", parameters={"labels": ["a"], "k": 1})
    assert block.execute({"samples": samples}) is None and not block.current_emissions


@pytest.mark.parametrize(
    "field,value",
    [
        ("default_k", 7),
        ("dimensions", 3),
        ("responses", [0, 0, 0, 1, 1, True]),
        ("class_labels", ["a", "a"]),
        ("algorithm", "kdtree"),
        ("normalization", "auto"),
        ("opencv_major", 0),
        ("samples", [[0.1, 0.2]]),
        ("feature_contract", []),
    ],
)
def test_rehashed_incompatible_models_remain_invalid(field, value):
    model = copy.deepcopy(trained())
    model[field] = value
    model["revision"] = model_revision(model)
    block = builtin_registry().create("ml.knn_predict")
    assert (
        block.execute({"model": model, "samples": [0, 0]}) is None and not block.current_emissions
    )


def test_budgets_and_k_validation_precede_native_prediction(monkeypatch):
    model = trained()
    block = builtin_registry().create("ml.knn_predict", parameters={"k": 7})
    assert block.execute({"model": model, "samples": X}) is None
    assert "K exceeds" in block.error
    block.parameters["k"] = 0
    monkeypatch.setattr("opencv_dataflow.knn_blocks.MAX_WORK", 1)
    assert block.execute({"model": model, "samples": X}) is None and "64 million" in block.error
    block = builtin_registry().create("ml.knn_train", parameters={"k": 7, "labels": LABELS})
    assert block.execute({"samples": X}) is None and "K exceeds" in block.error
    block.parameters.update(k=1, normalization="l2")
    assert block.execute({"samples": X}) is None and "zero feature vector" in block.error


def test_file_roundtrip_model_revision_strict_json_and_existing_files(tmp_path, monkeypatch):
    model = trained()
    path = tmp_path / "model.json"
    execute("ml.knn_save", {"model": model}, path=str(path))
    before = path.read_bytes()
    block = builtin_registry().create("ml.knn_save", parameters={"path": str(path)})
    assert not block.preview_safe and block.execute({"model": model}) is None
    assert "FileExistsError" in block.error and path.read_bytes() == before
    _, result = execute("ml.knn_load", {}, path=str(path))
    assert result.outputs["model"] == model
    changed = copy.deepcopy(model)
    changed["samples"][0][0] += 1
    block = builtin_registry().create("ml.knn_predict")
    assert block.execute({"model": changed, "samples": X}) is None
    assert "revision differs" in block.error
    for raw in (b'{"model":1,"model":2}', b'{"value":NaN}'):
        path.write_bytes(raw)
        block = builtin_registry().create("ml.knn_load", parameters={"path": str(path)})
        assert block.execute({}) is None and not block.current_emissions
    path.write_bytes(b" " * 101)
    monkeypatch.setattr("opencv_dataflow.knn_blocks.MAX_MODEL_BYTES", 100)
    block = builtin_registry().create("ml.knn_load", parameters={"path": str(path)})
    assert block.execute({}) is None and "8 MiB" in block.error


def test_native_exception_is_contained_then_recovers(monkeypatch):
    model = trained()
    native = cv2.ml.KNearest_create

    def fail():
        raise RuntimeError("deliberate native K-nearest failure")

    monkeypatch.setattr("opencv_dataflow.knn_blocks.cv2.ml.KNearest_create", fail)
    block = builtin_registry().create("ml.knn_predict")
    assert block.execute({"model": model, "samples": [0.1, 0.1]}) is None
    assert "deliberate native" in block.error and not block.current_emissions
    monkeypatch.setattr("opencv_dataflow.knn_blocks.cv2.ml.KNearest_create", native)
    assert block.execute({"model": model, "samples": [0.1, 0.1]}) is not None, block.error


BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/knn_classification"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_saved_native_hog_graphs_train_reload_and_classify_disjoint_images(
    case, run_profile, tmp_path
):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        training = expected["target"] == "save"
        if training:
            graph.blocks["save"].parameters["path"] = str(tmp_path / "new-model.json")
        for index in range(expected["runs"]):
            graph.run()
            block = graph.blocks[expected["target"]]
            if training and index < 5:
                assert block.state == graph.blocks["train"].state == "inactive"
                continue
            assert block.state == expected["states"][index], block.error
            if training:
                model = graph.blocks["train"].last_result.outputs["model"]
                assert json.loads((tmp_path / "new-model.json").read_bytes()) == model
                assert model["class_labels"] == ["vertical", "horizontal"]
                assert model["feature_contract"]["descriptor_length"] == model["dimensions"] == 324
                _, loaded = execute("ml.knn_load", {}, path=str(tmp_path / "new-model.json"))
                assert loaded.outputs["model"] == model
            elif case == "wrong_contract":
                assert not block.current_emissions and "preprocessing differs" in block.error
                assert graph.blocks["overlay"].state == "inactive"
            else:
                result = block.last_result
                assert result.outputs["labels"] == [expected["labels"][index]]
                source = graph.blocks["source"].last_result.outputs["image"]
                np.testing.assert_array_equal(result.outputs["image"], source)
                filename = (
                    "heldout-diagonal.png"
                    if case == "reject_diagonal"
                    else "heldout-vertical.png"
                    if index == 0
                    else "heldout-horizontal.png"
                )
                assert result.payloads["report"]["ticket"]["value"] == filename
                if expected["labels"][index] is None:
                    assert (
                        result.outputs["label"] is NO_VALUE and not block.current_emissions["label"]
                    )
                else:
                    assert result.outputs["label"] == expected["labels"][index]
                if case == "reject_diagonal":
                    assert result.outputs["report"]["unknown_reasons"] == [["knn_distance"]]
                    assert result.outputs["distances"][0] > 2.8
                if case == "classify":
                    assert result.outputs["distances"][0] < 0.28
                assert graph.blocks["filmstrip"].state == "pass"
                if run_profile == "debug":
                    assert result.drawings
                    output = graph.blocks["overlay"].last_result.outputs["image"]
                    assert np.any(output != source[..., None])
                else:
                    assert not result.drawings
    finally:
        graph.close()


def test_json_xml_save_as_and_generator_preserve_model_and_source_bytes(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "classify.json", builtin_registry())
    try:
        graph.save(tmp_path / "copied.json")
        restored = Graph.load(tmp_path / "copied.json", builtin_registry())
        try:
            model = tmp_path / restored.blocks["model"].parameters["path"]
            assert model.read_bytes() == (BASE / "assets/classifier-none.json").read_bytes()
            restored.run()
            assert restored.blocks["predict"].last_result.outputs["label"] == "vertical"
            export_project_xml(restored.to_dict(), tmp_path / "copied.xml")
            from_xml = Graph.from_dict(
                read_project_xml(tmp_path / "copied.xml"), builtin_registry(), project_dir=tmp_path
            )
            try:
                from_xml.run()
                assert from_xml.blocks["predict"].last_result.outputs["label"] == "vertical"
            finally:
                from_xml.close()
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("knn_examples", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "generated"
    module.HERE.mkdir()
    module.build()
    for path in module.HERE.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(module.HERE)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        module.build()
    for name, digest in before.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    pixels = [cv2.imread(str(p), 0).tobytes() for p in (BASE / "assets").glob("*.png")]
    assert len(pixels) == len(set(pixels)) == 9


def test_failed_input_admission_and_missing_wired_image_prevent_work(monkeypatch):
    graph = Graph.load(BASE / "classify.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["predict"]
        before = block.last_inputs["image"].copy()
        count = block.timing.count
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def rejected(inputs):
            result = original(inputs)
            result.outputs["image"] = np.zeros_like(result.outputs["image"])
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", rejected)
        graph.run()
        assert block.input_rejected and block.timing.count == count and not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], before)
        monkeypatch.setattr(
            graph.blocks["source"],
            "process",
            lambda inputs: Result({"image": NO_VALUE, "info": NO_VALUE, "metadata": NO_VALUE}),
        )
        graph.run()
        assert block.timing.count == count and not block.current_emissions
        assert not graph.blocks["source"].error
    finally:
        graph.close()


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("ml.knn_train", {"k": 0}),
        ("ml.knn_train", {"k": 65}),
        ("ml.knn_train", {"normalization": "auto"}),
        ("ml.knn_train", {"labels": [""]}),
        ("ml.knn_predict", {"distance_policy": "squared"}),
        ("ml.knn_predict", {"min_vote_fraction": 1.1}),
        ("ml.knn_predict", {"max_distance": -1}),
        ("ml.knn_predict", {"accepted_labels": [True]}),
    ],
)
def test_parameter_ranges_enums_and_class_fields_are_validated(kind, parameters):
    with pytest.raises(ValueError):
        builtin_registry().create(kind, parameters=parameters).validate_parameters()
