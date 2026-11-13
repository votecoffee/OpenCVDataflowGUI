"""Native classifier parity, portable model contracts and held-out image evidence."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.svm_blocks import KERNELS, _revision, read_svm
from opencv_dataflow.tickets import TicketClock

X = [[0, 0], [0, 1], [1, 0], [8, 8], [8, 9], [9, 8]]
LABELS = ["small"] * 3 + ["large"] * 3


def execute(type_id, inputs, **parameters):
    block = builtin_registry().create(type_id, parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def trained(**parameters):
    return execute("ml.svm_train", {"samples": X, "labels": LABELS}, **parameters)[1].outputs[
        "model"
    ]


@pytest.mark.parametrize("kernel", tuple(KERNELS))
def test_each_kernel_matches_independent_native_c_svc_and_roundtrips(kernel):
    model = trained(kernel=kernel, class_weights=[1, 2])
    native = cv2.ml.SVM_create()
    native.setType(cv2.ml.SVM_C_SVC)
    native.setKernel(KERNELS[kernel])
    native.setC(2)
    native.setGamma(0.1)
    native.setDegree(3)
    native.setCoef0(0)
    native.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 1e-6))
    native.setClassWeights(np.float32([1, 2]))
    assert native.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.int32([0, 0, 0, 1, 1, 1]))
    query = [[0.1, 0.1], [8.2, 8.3]]
    ids = native.predict(np.float32(query))[1].reshape(-1).astype(int)
    block, result = execute("ml.svm_predict", {"model": model, "samples": query})
    assert result.outputs["labels"] == [model["class_labels"][i] for i in ids]
    assert result.outputs["label"] is NO_VALUE and not block.current_emissions["label"]
    assert result.outputs["report"]["probabilities_available"] is False
    copy_model = json.loads(json.dumps(model))
    np.testing.assert_array_equal(
        read_svm(copy_model).predict(np.float32(query))[1], native.predict(np.float32(query))[1]
    )


def test_model_files_preserve_mapping_revision_and_existing_bytes(tmp_path):
    model = execute(
        "ml.svm_train",
        {"samples": [[1, 0], [1, 0.1], [1, 0.2], [0, 1], [0.1, 1], [0.2, 1]], "labels": LABELS},
        kernel="linear",
        normalization="l2",
    )[1].outputs["model"]
    save = builtin_registry().create("ml.svm_save", parameters={"path": "classifier.json"})
    save.project_dir = tmp_path
    assert not save.preview_safe
    result = save.execute({"model": model})
    assert result is not None, save.error
    raw = (tmp_path / "classifier.json").read_bytes()
    assert raw.endswith(b"\n") and not raw.startswith(b"\xef\xbb\xbf")
    assert save.execute({"model": model}) is None and "FileExistsError" in save.error
    assert (tmp_path / "classifier.json").read_bytes() == raw
    load = builtin_registry().create("ml.svm_load", parameters={"path": "classifier.json"})
    load.project_dir = tmp_path
    result = load.execute({})
    assert result is not None, load.error
    assert result.outputs["model"] == model
    _, prediction = execute("ml.svm_predict", {"model": model, "samples": [0.1, 1]})
    assert prediction.outputs["label"] == "large"


def test_hog_contract_travels_in_batches_and_rejects_same_length_wrong_settings():
    registry = builtin_registry()
    hog = registry.create("inspect.hog")
    entries = []
    for angle in range(4):
        image = np.zeros((32, 32), np.uint8)
        cv2.line(image, (4, 4 + angle * 3), (27, 26 - angle * 3), 255, 2)
        result = hog.execute({"image": image})
        assert result is not None, hog.error
        assert result.payloads["descriptor"]["feature_contract"] == result.outputs["info"]
        entries.append(ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"]))
    group = FlowGroup(tuple(entries), tuple(str(i) for i in range(4)))
    _, trained_result = execute("ml.svm_train", {"samples": group, "labels": ["a", "a", "b", "b"]})
    model = trained_result.outputs["model"]
    assert model["feature_contract"]["descriptor_length"] == model["dimensions"]
    _, result = execute("ml.svm_predict", {"model": model, "samples": entries[0]})
    assert result.outputs["label"] in ("a", "b")
    wrong = copy.deepcopy(entries[0].payload)
    wrong["feature_contract"]["settings"]["signed_gradient"] = True
    bad = ValuePacket(entries[0].value, wrong)
    block = registry.create("ml.svm_predict")
    assert block.execute({"model": model, "samples": bad}) is None
    assert "preprocessing differs" in block.error and not block.current_emissions
    block.parameters["contract_policy"] = "dimensions_only"
    assert block.execute({"model": model, "samples": bad}) is not None, block.error
    altered = FlowGroup((entries[0], bad), ("a", "b"))
    block = registry.create("ml.svm_train", parameters={"labels": ["a", "b"]})
    assert block.execute({"samples": altered}) is None
    assert "different preprocessing" in block.error
    color = np.repeat(image[..., None], 3, axis=2)
    result = hog.execute({"image": color})
    color_packet = ValuePacket(result.outputs["descriptor"], result.payloads["descriptor"])
    block = registry.create("ml.svm_predict")
    assert block.execute({"model": model, "samples": color_packet}) is None
    assert "preprocessing differs" in block.error


def test_image_prediction_owns_current_ticket_and_optional_graphics():
    model = trained(kernel="linear")
    clock = TicketClock()
    context = {"spatial": frame([32, 32], "camera"), "ticket": clock.issue("camera")}
    image = np.full((32, 32), 80, np.uint8)
    block, result = execute(
        "ml.svm_predict",
        {
            "model": model,
            "samples": ValuePacket([8, 8], context),
            "image": ValuePacket(image, context),
        },
        accepted_labels=["small"],
    )
    assert not result.passed and result.final_judgment is None and result.score is None
    assert result.outputs["label"] == "large"
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.payloads["label"]["ticket"] == context["ticket"]
    assert result.drawings["SVM class"].objects[0].text == "large"
    assert result.fail_reasons[0]["code"] == "svm_unaccepted_label"
    other = {**context, "ticket": clock.issue("camera")}
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([8, 8], context),
                "image": ValuePacket(image, other),
            }
        )
        is None
    )
    assert "different tickets" in block.error
    other = {**context, "spatial": frame([32, 32], "other")}
    assert (
        block.execute(
            {
                "model": model,
                "samples": ValuePacket([8, 8], context),
                "image": ValuePacket(image, other),
            }
        )
        is None
    )
    assert "frame differs" in block.error


@pytest.mark.parametrize(
    "samples",
    [[], [[1, 2], [3]], [[np.nan, 0]], [[np.inf, 1]], [[1e40, 1]], [[1e-300, 1]], [["a", "b"]]],
)
def test_invalid_features_fail_without_native_training(samples):
    block = builtin_registry().create("ml.svm_train", parameters={"labels": ["a", "b"]})
    assert block.execute({"samples": samples}) is None
    assert block.state == "error" and not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"c": 0},
        {"gamma": -1},
        {"degree": 0},
        {"max_iterations": 0},
        {"epsilon": 0},
        {"kernel": "custom"},
        {"class_weights": [0, 1]},
        {"class_weights": [1]},
        {"normalization": "unknown"},
    ],
)
def test_invalid_training_parameters_are_reported(parameters):
    block = builtin_registry().create("ml.svm_train", parameters=parameters)
    assert block.execute({"samples": X, "labels": LABELS}) is None
    assert block.state == "error"


@pytest.mark.parametrize("labels", [["only"] * 6, ["a", "b"], [1, 1, 1, 2, 2, 2]])
def test_training_requires_an_explicit_complete_string_mapping(labels):
    block = builtin_registry().create("ml.svm_train")
    assert block.execute({"samples": X, "labels": labels}) is None
    assert block.state == "error"


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "tampered"),
        ("dimensions", 3),
        ("opencv_major", 4),
        ("class_labels", ["x", "x"]),
    ],
)
def test_model_tampering_or_invalid_metadata_withholds_predictions(field, value):
    model = trained()
    model[field] = value
    block = builtin_registry().create("ml.svm_predict")
    assert block.execute({"model": model, "samples": [0, 0]}) is None
    assert not block.current_emissions
    if field != "revision":
        model["revision"] = _revision(model)
        assert block.execute({"model": model, "samples": [0, 0]}) is None


@pytest.mark.parametrize("text", ['{"model":1,"model":2}', '{"value":NaN}'])
def test_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path, text):
    file = tmp_path / "bad.json"
    file.write_text(text, encoding="utf-8")
    block = builtin_registry().create("ml.svm_load", parameters={"path": str(file)})
    assert block.execute({}) is None
    assert not block.current_emissions


def test_native_training_exception_is_contained(monkeypatch):
    def fail():
        raise RuntimeError("Native training failed")

    monkeypatch.setattr("opencv_dataflow.svm_blocks.cv2.ml.SVM_create", fail)
    block = builtin_registry().create("ml.svm_train")
    assert block.execute({"samples": X, "labels": LABELS}) is None
    assert "Native training failed" in block.error and not block.current_emissions


def test_xml_preserves_kernel_parameters_and_class_labels(tmp_path):
    graph = Graph()
    block = graph.add(
        builtin_registry().create(
            "ml.svm_train",
            parameters={
                "kernel": "polynomial",
                "degree": 2,
                "coef0": 1,
                "labels": LABELS,
            },
        )
    )
    block.position = [140, 80]
    block.description = "Part descriptor classifier"
    block.node_size = "large"
    try:
        data = graph.to_dict()
        export_project_xml(data, tmp_path / "svm.xml")
        restored = Graph.from_dict(read_project_xml(tmp_path / "svm.xml"), builtin_registry())
        try:
            assert restored.to_dict() == data
        finally:
            restored.close()
    finally:
        graph.close()


def test_float32_underflow_and_element_budgets_are_explicit(monkeypatch):
    block = builtin_registry().create("ml.svm_train", parameters={"labels": ["a", "b"]})
    assert block.execute({"samples": [[1e-300, 1], [1, 0]]}) is None
    assert "underflow float32" in block.error
    monkeypatch.setattr("opencv_dataflow.svm_blocks.MAX_ELEMENTS", 3)
    assert block.execute({"samples": [[1, 0], [0, 1]]}) is None
    assert "element limits" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_local_hog_svm_retrains_reloads_and_predicts_disjoint_images(
    profile, tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[1] / "examples/private/learnopencv/75_hog_svm"
    if not (source / "train.json").exists():
        pytest.skip("Private LearnOpenCV HOG/SVM study is absent")
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    # These user-output model files are not part of the shipped fixture.
    target = relocated / "trained-classifier.json"
    assert not target.exists()
    monkeypatch.chdir(tmp_path)
    truth = json.loads((relocated / "assets/truth.json").read_text(encoding="utf-8"))
    hashes = []
    for path, expected in truth["pixel_sha256"].items():
        pixels = cv2.imread(str(relocated / path), cv2.IMREAD_UNCHANGED)
        actual = hashlib.sha256(pixels.tobytes()).hexdigest()
        assert actual == expected
        hashes.append(actual)
    assert len(set(hashes)) == len(hashes) == 36
    graph = Graph.load(relocated / "train.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        for index in range(len(truth["training_paths"])):
            graph.run()
            if index < 26:
                assert graph.blocks["train"].state == graph.blocks["save"].state == "inactive"
        train = graph.blocks["train"]
        assert train.state == graph.blocks["save"].state == "pass", train.error
        report = train.last_result.outputs["report"]
        assert report["class_counts"] == [9, 9, 9] and report["dimensions"] == 324
        assert report["training_accuracy"] == 1
        assert report["revision"] == truth["model_revision"]
        assert json.loads(target.read_text(encoding="utf-8")) == train.last_result.outputs["model"]
    finally:
        graph.close()
    # Select the model produced by the saved training graph, not the bundled model.
    graph = Graph.load(relocated / "predict.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.blocks["model"].parameters["path"] = "trained-classifier.json"
    try:
        for path, expected in truth["heldout"].items():
            graph.run()
            block = graph.blocks["predict"]
            assert block.state == "pass", block.error
            result = block.last_result
            assert result.outputs["label"] == expected
            assert result.outputs["report"]["revision"] == truth["model_revision"]
            assert result.payloads["label"]["ticket"]["value"] == Path(path).name
            assert graph.blocks["overlay"].state == graph.blocks["review"].state == "pass"
            original = graph.blocks["source"].last_result.outputs["image"]
            np.testing.assert_array_equal(result.outputs["image"], original)
            assert np.any(
                graph.blocks["overlay"].last_result.outputs["image"] != original[..., None]
            )
        assert graph.sources_exhausted()
    finally:
        graph.close()
    graph = Graph.load(relocated / "wrong_features.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["predict"]
        assert block.state == "error" and "feature contract" in block.error
        assert not block.current_emissions and graph.blocks["review"].state == "inactive"
    finally:
        graph.close()
