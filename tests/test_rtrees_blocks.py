"""Native forest votes, saved structure and reusable image classifier contracts."""

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
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.rtrees_blocks import read_forest
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock

X = [[i / 10, j / 10] for i in range(5) for j in range(5)] + [
    [5 + i / 10, 5 + j / 10] for i in range(5) for j in range(5)
]
LABELS = ["small"] * 25 + ["large"] * 25


def execute(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def trained(**parameters):
    p = {"labels": LABELS, "max_trees": 12, **parameters}
    return execute("ml.rtrees_train", {"samples": X}, **p)[1].outputs["model"]


def json_native(model):
    native, _ = read_forest(model)
    storage = cv2.FileStorage(
        "", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_JSON
    )
    native.write(storage, "forest")
    return json.loads(storage.releaseAndGetString())


def controlled_ensemble(ids):
    # Explicit constant native trees isolate vote-gate behavior from stochastic fitting.
    model = trained()
    data = json_native(model)
    data["forest"]["ntrees"] = len(ids)
    data["forest"]["trees"] = [
        {"nodes": [{"depth": 0, "value": float(i), "norm_class_idx": i}]} for i in ids
    ]
    model["forest_yaml"] = json.dumps(data)
    model["revision"] = model_revision(model)
    return model


@pytest.mark.parametrize("normalization", ["none", "l2"])
@pytest.mark.parametrize("importance", [False, True])
@pytest.mark.parametrize("termination", ["tree_limit", "tree_limit_or_oob_error"])
def test_saved_forest_predictions_votes_importance_and_controls_match_native(
    normalization, importance, termination, tmp_path
):
    samples = (np.float32(X) + 1).tolist()  # L2 vectors are nonzero.
    block, result = execute(
        "ml.rtrees_train",
        {"samples": samples, "labels": LABELS},
        max_trees=16,
        max_depth=4,
        active_variables=1,
        min_sample_count=2,
        normalization=normalization,
        calculate_importance=importance,
        termination=termination,
        oob_error=0.02,
        class_priors=[1, 2],
        sample_weights=[1, 2] * 25,
    )
    model = json.loads(json.dumps(result.outputs["model"]))
    path = tmp_path / "native.yml"
    path.write_bytes(model["forest_yaml"].encode())
    native = cv2.ml.RTrees_load(str(path), "forest")
    assert native.isClassifier() and native.getVarCount() == 2
    assert native.getMaxDepth() == 4 and native.getActiveVarCount() == 1
    np.testing.assert_array_equal(native.getPriors(), np.float32([1, 2]))
    query = np.float32([[1.1, 1.2], [6.1, 6.2]])
    if normalization == "l2":
        query /= np.linalg.norm(query.astype(np.float64), axis=1, keepdims=True)
    expected_votes = native.getVotes(query, 0)
    expected_classes = native.predict(query)[1].ravel().astype(int)
    _, predicted = execute(
        "ml.rtrees_predict",
        {"model": model, "samples": [[1.1, 1.2], [6.1, 6.2]]},
        reject_vote_ties=False,
    )
    assert expected_votes[0].tolist() == [0, 1]
    np.testing.assert_array_equal(predicted.outputs["votes"], expected_votes[1:])
    assert predicted.outputs["labels"] == [model["class_labels"][i] for i in expected_classes]
    trees = predicted.outputs["report"]["trees"]
    np.testing.assert_allclose(predicted.outputs["vote_fractions"], expected_votes[1:] / trees)
    assert 1 <= trees <= 16
    if termination == "tree_limit":
        assert trees == 16
    assert predicted.score is None and predicted.final_judgment is None
    if importance:
        np.testing.assert_allclose(
            result.outputs["importance"], native.getVarImportance(), atol=1e-7
        )
        assert result.graphics["Feature importance"].shape == (400, 680, 3)
    else:
        assert (
            result.outputs["importance"] is NO_VALUE and not block.current_emissions["importance"]
        )
    assert result.outputs["report"]["oob_computation_requested"] is (
        importance or termination != "tree_limit"
    )
    if not importance and termination == "tree_limit":
        assert result.outputs["report"]["native_oob_statistic"] is None


def test_vote_fraction_margin_tie_boundaries_and_withheld_recognition():
    model = controlled_ensemble([0, 0, 0, 1])
    _, result = execute(
        "ml.rtrees_predict",
        {"model": model, "samples": [1, 1]},
        min_vote_fraction=0.75,
        min_vote_margin=0.5,
    )
    assert result.passed and result.outputs["label"] == "small"
    assert result.outputs["votes"] == [[3, 1]]
    for parameter, value, code in [
        ("min_vote_fraction", 0.75001, "rtrees_vote_fraction"),
        ("min_vote_margin", 0.50001, "rtrees_vote_margin"),
    ]:
        block, result = execute(
            "ml.rtrees_predict", {"model": model, "samples": [1, 1]}, **{parameter: value}
        )
        assert not result.passed and result.outputs["labels"] == [None]
        assert result.outputs["candidate_labels"] == ["small"]
        assert result.outputs["report"]["unknown_reasons"] == [[code]]
        assert result.outputs["label"] is NO_VALUE and not block.current_emissions["label"]
    tie = controlled_ensemble([0, 0, 1, 1])
    _, result = execute("ml.rtrees_predict", {"model": tie, "samples": [1, 1]})
    assert not result.passed and result.outputs["report"]["unknown_reasons"] == [
        ["rtrees_vote_tie"]
    ]
    _, result = execute(
        "ml.rtrees_predict", {"model": tie, "samples": [1, 1]}, reject_vote_ties=False
    )
    assert result.passed and result.outputs["label"] == "small"
    _, result = execute(
        "ml.rtrees_predict", {"model": model, "samples": [1, 1]}, accepted_labels=["large"]
    )
    assert not result.passed and result.outputs["label"] == "small"
    assert result.outputs["report"]["unknown_count"] == 0
    assert result.fail_reasons[0]["code"] == "rtrees_unaccepted_label"


def test_feature_batches_context_graphics_and_lean():
    metadata = {"feature_contract": {"source": "image measurements", "settings": {"units": "px"}}}
    group = FlowGroup(tuple(ValuePacket(x, metadata) for x in X), tuple(map(str, range(50))))
    block, result = execute(
        "ml.rtrees_train",
        {"samples": group, "labels": LABELS},
        max_trees=12,
        calculate_importance=True,
    )
    model = result.outputs["model"]
    assert model["feature_contract"] == metadata["feature_contract"]
    context = {
        **metadata,
        "ticket": TicketClock().issue("current"),
        "spatial": frame([32, 32], "camera"),
        "recipe_ref": {"id": "r", "revision": 1, "digest": "1" * 64},
    }
    image = np.full((32, 32), 12345, np.uint16)
    inputs = {
        "model": model,
        "samples": ValuePacket([0.1, 0.2], context),
        "image": ValuePacket(image, context),
    }
    block, result = execute("ml.rtrees_predict", inputs)
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.outputs["label"] == "small" and result.drawings
    for port in ("label", "votes", "report", "image"):
        assert result.payloads[port]["ticket"] == context["ticket"]
        assert result.payloads[port]["spatial"] == context["spatial"]
        assert result.payloads[port]["recipe_ref"] == context["recipe_ref"]
    block.run_profile = profile("lean")
    assert not block.execute(inputs).drawings
    wrong = copy.deepcopy(context)
    wrong["feature_contract"]["settings"]["units"] = "mm"
    assert block.execute({**inputs, "samples": ValuePacket([0.1, 0.2], wrong)}) is None
    assert "preprocessing differs" in block.error and not block.current_emissions
    block.parameters["contract_policy"] = "dimensions_only"
    assert block.execute({**inputs, "samples": ValuePacket([0.1, 0.2], wrong)}) is not None, (
        block.error
    )
    mixed = FlowGroup((ValuePacket([1, 0], context), ValuePacket([0, 1], wrong)), ("a", "b"))
    block = builtin_registry().create("ml.rtrees_train", parameters={"labels": ["a", "b"]})
    assert block.execute({"samples": mixed}) is None and "different preprocessing" in block.error


@pytest.mark.parametrize("field", ["ticket", "spatial", "recipe_ref", "pyramid"])
def test_image_context_is_rejected_before_native_loading(field, monkeypatch):
    model = trained()
    context = {
        "ticket": TicketClock().issue("a"),
        "spatial": frame([32, 32], "a"),
        "recipe_ref": {"id": "r", "revision": 1, "digest": "1" * 64},
        "pyramid": {"level": 0, "largest_size": [32, 32], "to_largest": np.eye(3).tolist()},
    }
    wrong = copy.deepcopy(context)
    if field == "ticket":
        wrong[field] = TicketClock().issue("b")
    elif field == "spatial":
        wrong[field]["root_id"] = "b"
    elif field == "recipe_ref":
        wrong[field]["revision"] = 2
    else:
        wrong[field]["level"] = 1

    def unexpected():
        raise AssertionError("Invalid context reached native model loading")

    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.cv2.ml.RTrees_create", unexpected)
    block = builtin_registry().create("ml.rtrees_predict")
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([1, 1], context),
                "image": ValuePacket(np.zeros((32, 32), np.uint8), wrong),
            }
        )
        is None
    )
    assert not block.current_emissions
    expected = {
        "ticket": "different tickets",
        "spatial": "frame differs",
        "recipe_ref": "recipe revisions",
        "pyramid": "conflicting pyramid",
    }[field]
    assert expected in block.error


@pytest.mark.parametrize(
    "change",
    [
        "classes",
        "dimensions",
        "tree_count",
        "depth",
        "variable",
        "missing_child",
        "class_value",
        "categorical",
        "unknown_field",
    ],
)
def test_rehashed_malformed_native_structure_rejects_before_native_read(change, monkeypatch):
    model = trained()
    data = json_native(model)
    forest = data["forest"]
    if change == "classes":
        forest["class_labels"] = [1, 0]
    elif change == "dimensions":
        forest["var_count"] = 3
    elif change == "tree_count":
        forest["ntrees"] = 999
    elif change == "depth":
        forest["trees"][0]["nodes"][0]["depth"] = 2
    elif change == "variable":
        forest["trees"][0]["nodes"][0]["splits"][0]["var"] = 99
    elif change == "missing_child":
        forest["trees"][0]["nodes"].pop()
    elif change == "class_value":
        forest["trees"][0]["nodes"][0]["value"] = 99
    elif change == "categorical":
        forest["var_type"][0] = 1
    else:
        forest["unsupported"] = [1]
    model["forest_yaml"] = json.dumps(data)
    model["revision"] = model_revision(model)

    def unexpected():
        raise AssertionError("Malformed tree reached native model loading")

    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.cv2.ml.RTrees_create", unexpected)
    block = builtin_registry().create("ml.rtrees_predict")
    assert (
        block.execute({"model": model, "samples": [1, 1]}) is None and not block.current_emissions
    )
    assert "Malformed tree" not in block.error


@pytest.mark.parametrize(
    "samples", [[], [[1, 2], [3]], [[True]], [["1"]], [[np.nan]], [[np.inf]], [[1e16]], [[1e-300]]]
)
def test_invalid_training_features_have_no_model_event(samples):
    block = builtin_registry().create("ml.rtrees_train", parameters={"labels": ["a", "b"]})
    assert block.execute({"samples": samples}) is None and not block.current_emissions


def test_native_failure_recovery_work_guards_and_strict_file_roundtrip(tmp_path, monkeypatch):
    model = trained()
    file = tmp_path / "forest.json"
    execute("ml.rtrees_save", {"model": model}, path=str(file))
    raw = file.read_bytes()
    block = builtin_registry().create("ml.rtrees_save", parameters={"path": str(file)})
    assert not block.preview_safe and block.execute({"model": model}) is None
    assert "FileExistsError" in block.error and file.read_bytes() == raw
    _, loaded = execute("ml.rtrees_load", {}, path=str(file))
    assert loaded.outputs["model"] == model
    for text in ('{"model":1,"model":2}', '{"bad":NaN}'):
        file.write_text(text, encoding="utf-8")
        block = builtin_registry().create("ml.rtrees_load", parameters={"path": str(file)})
        assert block.execute({}) is None and not block.current_emissions
    original = cv2.ml.RTrees_create

    def fail():
        raise RuntimeError("Deliberate forest native failure")

    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.cv2.ml.RTrees_create", fail)
    block = builtin_registry().create("ml.rtrees_predict")
    assert block.execute({"model": model, "samples": [1, 1]}) is None
    assert "Deliberate forest" in block.error and not block.current_emissions
    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.cv2.ml.RTrees_create", original)
    assert block.execute({"model": model, "samples": [1, 1]}) is not None, block.error
    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.MAX_WORK", 1)
    assert (
        block.execute({"model": model, "samples": [1, 1]}) is None and "64 million" in block.error
    )
    train = builtin_registry().create("ml.rtrees_train", parameters={"labels": LABELS})
    assert train.execute({"samples": X}) is None and "64 million" in train.error
    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.MAX_WORK", 64_000_000)
    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.MAX_NODES", 1)
    assert block.execute({"model": model, "samples": [1, 1]}) is None and "50,000" in block.error


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_depth": 0},
        {"max_trees": 257},
        {"min_sample_count": 0},
        {"termination": "accuracy_only"},
        {"termination": "tree_limit_or_oob_error", "oob_error": 0},
        {"class_priors": [1, -1]},
        {"sample_weights": [True]},
        {"normalization": "auto"},
    ],
)
def test_training_parameter_constraints(parameters):
    with pytest.raises(ValueError):
        builtin_registry().create("ml.rtrees_train", parameters=parameters).validate_parameters()


BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/rtrees_classification"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_saved_hog_forest_graphs_training_reload_votes_and_graphics(case, run_profile, tmp_path):
    expected = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    try:
        training = expected["target"] == "save"
        if training:
            graph.blocks["save"].parameters["path"] = str(tmp_path / "new-forest.json")
        for index in range(expected["runs"]):
            graph.run()
            block = graph.blocks[expected["target"]]
            if training and index < 5:
                assert block.state == graph.blocks["train"].state == "inactive"
                continue
            assert block.state == expected["states"][index], block.error
            if training:
                result = graph.blocks["train"].last_result
                model = result.outputs["model"]
                assert json.loads((tmp_path / "new-forest.json").read_bytes()) == model
                assert model["class_labels"] == ["vertical", "horizontal"]
                assert model["dimensions"] == model["feature_contract"]["descriptor_length"] == 324
                native, info = read_forest(model)
                assert 1 <= info["trees"] <= 64
                if case != "train_oob":
                    assert info["trees"] == 64
                if case == "train_importance":
                    assert result.outputs["importance"] is not NO_VALUE
                    assert bool(result.graphics) is (run_profile == "debug")
                if case == "train_weighted":
                    np.testing.assert_array_equal(native.getPriors(), [1, 2])
                    assert model["training_summary"]["sample_weights_sha256"] is not None
            elif case == "wrong_contract":
                assert not block.current_emissions and "preprocessing differs" in block.error
            else:
                result = block.last_result
                assert result.outputs["labels"] == [expected["labels"][index]]
                image = graph.blocks["source"].last_result.outputs["image"]
                np.testing.assert_array_equal(result.outputs["image"], image)
                name = (
                    "diagonal"
                    if case == "vote_threshold"
                    else "vertical"
                    if index == 0
                    else "horizontal"
                )
                assert result.payloads["votes"]["ticket"]["value"] == f"heldout-{name}.png"
                if case != "normalized":
                    assert result.outputs["votes"] == [TRUTH["expected_none_votes"][name]]
                if expected["labels"][index] is None:
                    assert (
                        result.outputs["label"] is NO_VALUE and not block.current_emissions["label"]
                    )
                    assert result.outputs["report"]["unknown_reasons"] == [["rtrees_vote_fraction"]]
                if run_profile == "debug":
                    assert result.drawings and graph.blocks["filmstrip"].state == "pass"
                    assert np.any(
                        graph.blocks["overlay"].last_result.outputs["image"] != image[..., None]
                    )
                else:
                    assert not result.drawings
    finally:
        graph.close()


def test_json_xml_portable_assets_and_exact_generation_in_isolated_seed_process(tmp_path):
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
            assert model.read_bytes() == (BASE / "assets/forest-none.json").read_bytes()
            restored.run()
            assert restored.blocks["predict"].last_result.outputs["label"] == "vertical"
            export_project_xml(restored.to_dict(), tmp_path / "copied.xml")
            xml = Graph.from_dict(
                read_project_xml(tmp_path / "copied.xml"), builtin_registry(), project_dir=tmp_path
            )
            try:
                xml.run()
                assert xml.blocks["predict"].last_result.outputs["label"] == "vertical"
            finally:
                xml.close()
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("forest_examples", BASE / "build_example.py")
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


def test_runtime_training_does_not_set_seed_and_rejects_invalid_weights_or_model_limits(
    monkeypatch,
):
    def unexpected(*args):
        raise AssertionError("Runtime changed the caller RNG seed")

    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.cv2.setRNGSeed", unexpected)
    model = trained()
    block = builtin_registry().create("ml.rtrees_train", parameters={"labels": LABELS})
    for weights in ([1], [0] * 50, [float("nan")] * 50):
        assert block.execute({"samples": X, "sample_weights": weights}) is None
        assert not block.current_emissions
        assert (
            "JSON compliant" if not np.isfinite(weights).all() else "sample_weights"
        ) in block.error
    block.parameters["active_variables"] = 3
    assert block.execute({"samples": X}) is None and "descriptor length" in block.error
    changed = copy.deepcopy(model)
    changed["class_labels"][0] = "changed without revision"
    block = builtin_registry().create("ml.rtrees_predict")
    assert (
        block.execute({"model": changed, "samples": [1, 1]}) is None
        and "revision differs" in block.error
    )
    monkeypatch.setattr("opencv_dataflow.rtrees_blocks.MAX_MODEL_BYTES", 10)
    assert block.execute({"model": model, "samples": [1, 1]}) is None and "8 MiB" in block.error


def test_failed_admission_and_missing_connected_image_preserve_last_input(monkeypatch):
    graph = Graph.load(BASE / "classify.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["predict"]
        count, image = block.timing.count, block.last_inputs["image"].copy()
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
        np.testing.assert_array_equal(block.last_inputs["image"], image)
        monkeypatch.setattr(
            graph.blocks["source"],
            "process",
            lambda inputs: Result({"image": NO_VALUE, "info": NO_VALUE, "metadata": NO_VALUE}),
        )
        graph.run()
        assert not graph.blocks["source"].error
        assert block.timing.count == count and not block.current_emissions
    finally:
        graph.close()
