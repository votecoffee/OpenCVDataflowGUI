"""Fixed-palette/visual-vocabulary contracts and independent held-out inspection checks."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cluster_models import nearest, revision
from opencv_dataflow.core import Graph, Result
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock

COLORS = np.uint8([[24, 32, 200], [40, 180, 36], [210, 70, 25]])
X = [[0, 0], [2, 0], [10, 10], [12, 10]]


def run(type_id, inputs, **parameters):
    block = builtin_registry().create(type_id, parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def trained():
    return run(
        "ml.kmeans_train", {"samples": X, "initial_labels": [1, 1, 0, 0]}, clusters=2, attempts=1
    )[1].outputs["model"]


def palette_image():
    return np.repeat(np.repeat(COLORS[None], 4, axis=0), 5, axis=1)


@pytest.mark.parametrize("initialization", ["random", "kmeans_plus_plus"])
@pytest.mark.parametrize("termination", ["both", "iterations", "epsilon"])
def test_training_matches_native_centers_then_reuses_labels_on_independent_rows(
    initialization, termination
):
    flags = cv2.KMEANS_RANDOM_CENTERS if initialization == "random" else cv2.KMEANS_PP_CENTERS
    criteria = {"both": 3, "iterations": 1, "epsilon": 2}[termination]
    compactness, _, centers = cv2.kmeans(
        np.float32(X),
        2,
        np.int32([1, 1, 0, 0]),
        (criteria, 50, 0.01),
        1,
        flags | cv2.KMEANS_USE_INITIAL_LABELS,
    )
    centers = np.array(sorted(centers.tolist()))
    _, result = run(
        "ml.kmeans_train",
        {"samples": X, "initial_labels": [1, 1, 0, 0]},
        clusters=2,
        attempts=1,
        initialization=initialization,
        termination=termination,
    )
    model = result.outputs["model"]
    np.testing.assert_array_equal(model["centers"], centers)
    assert result.outputs["report"]["compactness"] == pytest.approx(compactness)
    heldout = [[0.5, 1], [11, 12], [6, 5]]
    _, result = run("ml.kmeans_predict", {"samples": heldout, "model": model}, max_distance=3)
    expected = np.linalg.norm(np.array(heldout)[:, None] - centers[None], axis=2)
    np.testing.assert_allclose(result.outputs["distances"], expected.min(axis=1))
    assert result.outputs["labels"] == [0, 1, 0]  # Third row is an exact tie.
    assert result.outputs["accepted"] == [True, True, False] and not result.passed
    assert result.outputs["histogram"] == pytest.approx([1 / 3, 1 / 3])
    assert result.fail_reasons[0]["code"] == "cluster_unknown_samples"
    assert result.final_judgment is None and result.score is None


def test_distance_cutoff_fraction_boundary_cluster_filter_and_histogram_contract():
    model = trained()
    inputs = {"model": model, "samples": [[1, 0], [11, 10], [1, 2]]}
    _, result = run(
        "ml.kmeans_predict",
        inputs,
        max_distance=2,
        max_unknown_fraction=1 / 3,
        accepted_clusters=[0],
    )
    assert result.passed and result.outputs["accepted"] == [True, False, True]
    assert result.outputs["distances"] == [0, 0, 2]
    first = result.payloads["histogram"]["feature_contract"]
    _, changed = run("ml.kmeans_predict", inputs, max_distance=1.99, accepted_clusters=[0])
    assert changed.payloads["histogram"]["feature_contract"] != first
    assert changed.outputs["accepted"] == [True, False, False]
    block = builtin_registry().create("ml.kmeans_predict", parameters={"accepted_clusters": [2]})
    assert block.execute(inputs) is None and not block.current_emissions
    assert "outside the trained model" in block.error


def test_direct_distances_avoid_cancellation_and_single_center_vector_work():
    values = np.float32([[1e12, 1e12], [1e12 + 131072, 1e12]])
    centers = values.copy()
    labels, distances = nearest(values, centers)
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_array_equal(distances, [0, 0])
    _, result = run("ml.kmeans_train", {"samples": [[3, 4]]}, clusters=1)
    _, result = run("ml.kmeans_predict", {"model": result.outputs["model"], "samples": [3, 4]})
    assert result.outputs["labels"] == [0] and result.outputs["distances"] == [0]


def test_l2_descriptor_batches_contracts_and_current_ticket_ownership():
    contract = {"source": "image embedding", "settings": {"resize": [32, 32]}}
    context = {"feature_contract": contract}
    group = FlowGroup(
        tuple(ValuePacket(row, context) for row in [[3, 0], [6, 0], [0, 4], [0, 8]]),
        ("a", "b", "c", "d"),
    )
    _, trained_result = run("ml.kmeans_train", {"samples": group}, clusters=2, normalization="l2")
    model = json.loads(json.dumps(trained_result.outputs["model"]))
    assert model["feature_contract"] == contract and model["centers"] == [[0, 1], [1, 0]]
    ticket = TicketClock().issue("new acquisition")
    context.update(
        ticket=ticket,
        spatial=frame([32, 32], "current"),
        recipe_ref={"id": "recipe", "revision": 1, "digest": "1" * 64},
    )
    _, prediction = run(
        "ml.kmeans_predict", {"model": model, "samples": ValuePacket([30, 0], context)}
    )
    assert prediction.outputs["labels"] == [1] and prediction.outputs["distances"] == [0]
    assert prediction.payloads["histogram"]["ticket"] == ticket
    assert prediction.payloads["labels"]["spatial"] == context["spatial"]
    assert prediction.payloads["report"]["recipe_ref"] == context["recipe_ref"]
    context["feature_contract"] = {"source": "another same-length embedding"}
    block = builtin_registry().create("ml.kmeans_predict")
    inputs = {"model": model, "samples": ValuePacket([30, 0], context)}
    assert block.execute(inputs) is None and "preprocessing differs" in block.error
    block.parameters["contract_policy"] = "dimensions_only"
    assert block.execute(inputs) is not None, block.error
    group = FlowGroup(
        (ValuePacket([1, 0], {"feature_contract": contract}), ValuePacket([0, 1], context)),
        ("a", "b"),
    )
    block = builtin_registry().create("ml.kmeans_train", parameters={"clusters": 2})
    assert block.execute({"samples": group}) is None and "different preprocessing" in block.error


@pytest.mark.parametrize("space", ["bgr", "lab", "gray"])
@pytest.mark.parametrize("depth", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("encoding", ["bgr", "rgb", "rgba", "gray"])
def test_palette_cross_depth_encoding_selection_alpha_and_unchanged_primary(space, depth, encoding):
    reference = palette_image()
    _, result = run("ml.palette_train", {"image": reference}, clusters=3, color_space=space)
    model = json.loads(json.dumps(result.outputs["model"]))
    query = reference[:, ::-1].copy()  # Held-out arrangement, fixed trained IDs.
    if encoding in ("rgb", "rgba"):
        query = query[..., ::-1]
    if encoding == "gray":
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    if depth == "uint16":
        query = query.astype(np.uint16) * 257
    elif depth == "float32":
        query = query.astype(np.float32) / 255
    if encoding == "rgba":
        alpha = np.full(
            query.shape[:2], 1 if depth == "float32" else np.iinfo(depth).max, dtype=depth
        )
        alpha[1, 2] = 0
        query = np.dstack((query, alpha))
    image = ImageArray(query, encoding, metadata={"custom": "retained"})
    mask = np.full(query.shape[:2], 255, np.uint8)
    mask[:, 0] = 0
    ticket = TicketClock().issue("heldout")
    context = {"ticket": ticket, "spatial": frame([15, 4], "camera")}
    block, result = run(
        "ml.palette_predict", {"model": model, "image": ValuePacket(image, context), "mask": mask}
    )
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert result.outputs["image"].dtype == image.dtype
    quantized = result.outputs["quantized"]
    assert quantized.dtype == image.dtype and image_format(quantized) == encoding
    assert quantized.metadata["custom"] == "retained"
    np.testing.assert_array_equal(quantized[:, 0], image[:, 0])
    assert np.all(result.outputs["labels"][:, 0] == -1)
    if encoding == "rgba":
        np.testing.assert_array_equal(quantized[..., 3], image[..., 3])
        np.testing.assert_array_equal(quantized[1, 2], image[1, 2])
        assert result.outputs["labels"][1, 2] == -1
    if space == "bgr" and encoding != "gray":
        assert result.outputs["labels"][0, [1, 6, 11]].tolist() == [2, 1, 0]
        np.testing.assert_allclose(result.outputs["distances"][0, [1, 6, 11]], 0, atol=2e-6)
    for output in ("labels", "distances", "unknown_mask", "image", "quantized", "histogram"):
        assert result.payloads[output]["ticket"] == ticket
        assert result.payloads[output]["spatial"] == context["spatial"]
    assert result.drawings and result.outputs["cluster_map"] is not NO_VALUE
    block.run_profile = profile("lean")
    result = block.execute({"model": model, "image": image, "mask": mask})
    assert result is not None, block.error
    assert not result.drawings and result.outputs["cluster_map"] is NO_VALUE
    assert not block.current_emissions["cluster_map"]


def test_palette_independent_defect_region_mask_and_sampling():
    reference = palette_image()
    _, trained_result = run(
        "ml.palette_train", {"image": reference}, clusters=3, color_space="bgr", max_samples=9
    )
    assert trained_result.outputs["report"]["fitted_samples"] == 9
    assert trained_result.outputs["report"]["selected_pixels"] == 60
    query = reference.copy()
    query[1:3, 6:9] = [240, 240, 240]
    _, result = run(
        "ml.palette_predict",
        {"model": trained_result.outputs["model"], "image": query},
        max_distance=0.05,
        max_unknown_fraction=0.1,
    )
    assert result.passed  # Exactly six of sixty at the inclusive fraction boundary.
    expected = np.zeros((4, 15), np.uint8)
    expected[1:3, 6:9] = 255
    np.testing.assert_array_equal(result.outputs["unknown_mask"], expected)
    assert result.outputs["report"]["unknown_count"] == 6
    assert sum(result.outputs["histogram"]) == pytest.approx(0.9)
    assert result.drawings["Palette inspection"].objects[0].kind == "polyline"
    block = builtin_registry().create("ml.palette_predict", parameters={"max_distance": 0.05})
    assert not block.execute({"model": trained_result.outputs["model"], "image": query}).passed
    assert block.state == "fail" and not block.error


@pytest.mark.parametrize(
    "samples", [[], [[1, 2], [3]], [[True]], [["1"]], [[np.nan]], [[np.inf]], [[1e20]], [[1e-300]]]
)
def test_invalid_training_values_produce_no_event(samples):
    block = builtin_registry().create("ml.kmeans_train", parameters={"clusters": 1})
    assert block.execute({"samples": samples}) is None and not block.current_emissions


@pytest.mark.parametrize(
    "image",
    [
        np.full((2, 2), -0.01, np.float32),
        np.full((2, 2), 1.001, np.float64),
        np.full((2, 2), np.nan, np.float32),
        ImageArray(np.ones((2, 2, 3), np.uint8), "hsv"),
        ImageArray(np.ones((2, 2, 3), np.uint8), "gray"),
    ],
)
def test_invalid_image_encoding_or_range_is_not_silently_rescaled(image):
    block = builtin_registry().create("ml.palette_train", parameters={"clusters": 1})
    assert block.execute({"image": image}) is None and not block.current_emissions


def test_training_and_prediction_resource_guards_precede_native_work(monkeypatch):
    block = builtin_registry().create("ml.palette_train", parameters={"clusters": 1})
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_IMAGE_PIXELS", 2)
    assert block.execute({"image": palette_image()}) is None and "1 million" in block.error
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_CENTER_VALUES", 1)
    block = builtin_registry().create("ml.kmeans_train", parameters={"clusters": 2})
    assert block.execute({"samples": X}) is None and "131,072" in block.error
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_CENTER_VALUES", 131072)
    model = trained()
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_DISTANCE_WORK", 1)
    block = builtin_registry().create("ml.kmeans_predict")
    assert block.execute({"samples": X, "model": model}) is None and "64 million" in block.error


@pytest.mark.parametrize(
    "field,value",
    [
        ("centers", [[0.1, 0.2]]),
        ("normalization", "auto"),
        ("feature_contract", []),
        ("kind", "other"),
    ],
)
def test_rehashed_incompatible_models_still_reject(field, value):
    model = trained()
    model[field] = value
    model["revision"] = revision(model)
    block = builtin_registry().create("ml.kmeans_predict")
    assert block.execute({"samples": X, "model": model}) is None and not block.current_emissions


def test_save_load_preserves_model_existing_bytes_and_strict_json(tmp_path):
    model = trained()
    _, result = run("ml.kmeans_save", {"model": model}, path=str(tmp_path / "model.json"))
    target = tmp_path / "model.json"
    raw = target.read_bytes()
    block = builtin_registry().create("ml.kmeans_save", parameters={"path": str(target)})
    assert not block.preview_safe and block.execute({"model": model}) is None
    assert "FileExistsError" in block.error and target.read_bytes() == raw
    _, result = run("ml.kmeans_load", {}, path=str(target))
    assert result.outputs["model"] == model
    block = builtin_registry().create("ml.kmeans_predict")
    changed = copy.deepcopy(model)
    changed["centers"][0][0] += 1
    assert block.execute({"model": changed, "samples": X}) is None
    assert "revision differs" in block.error
    for text in ('{"model":1,"model":2}', '{"bad":NaN}'):
        target.write_text(text, encoding="utf-8")
        block = builtin_registry().create("ml.kmeans_load", parameters={"path": str(target)})
        assert block.execute({}) is None and not block.current_emissions


def test_native_exception_is_contained_and_recovers_without_partial_emission(monkeypatch):
    native = cv2.kmeans

    def fail(*args):
        raise RuntimeError("deliberate native fit failure")

    monkeypatch.setattr("opencv_dataflow.clustering.cv2.kmeans", fail)
    block = builtin_registry().create("ml.kmeans_train", parameters={"clusters": 2})
    assert block.execute({"samples": X}) is None
    assert "deliberate native fit failure" in block.error and not block.current_emissions
    monkeypatch.setattr("opencv_dataflow.clustering.cv2.kmeans", native)
    assert block.execute({"samples": X}) is not None, block.error


BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/reusable_clusters"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
@pytest.mark.parametrize("case", tuple(TRUTH["cases"]))
def test_saved_graphs_train_reload_and_inspect_disjoint_images(case, run_profile, tmp_path):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(run_profile)
    expected = TRUTH["cases"][case]
    try:
        if case.startswith("train_"):
            graph.blocks["save"].parameters["path"] = str(tmp_path / "new-model.json")
        for index in range(expected["runs"]):
            graph.run()
            if case == "train_vocabulary" and index < 5:
                assert graph.blocks["train"].state == graph.blocks["save"].state == "inactive"
        block = graph.blocks[expected["target"]]
        assert block.state == expected["expected_state"], block.error
        if case == "vocabulary_wrong_contract":
            assert not block.current_emissions and "preprocessing differs" in block.error
        elif case.startswith("train_"):
            model = graph.blocks["train"].last_result.outputs["model"]
            assert json.loads((tmp_path / "new-model.json").read_bytes()) == model
            _, result = run("ml.kmeans_load", {}, path=str(tmp_path / "new-model.json"))
            assert result.outputs["model"] == model
        else:
            result = block.last_result
            assert result.passed is expected["passed"]
            assert result.payloads["histogram"]["ticket"]["value"] == (
                "heldout-horizontal.png"
                if case == "vocabulary_query"
                else "defect.png"
                if case in ("palette_defect", "palette_selection")
                else "query16.png"
                if case == "palette_rgb16"
                else "query.png"
            )
            if case.startswith("palette_"):
                source = graph.blocks["source"].last_result.outputs["image"]
                np.testing.assert_array_equal(result.outputs["image"], source)
                if case in ("palette_ok", "palette_rgb16"):
                    np.testing.assert_array_equal(
                        result.outputs["labels"],
                        np.broadcast_to(TRUTH["palette_query_label_columns"], (96, 144)),
                    )
                if case == "palette_defect":
                    mask = np.zeros((96, 144), np.uint8)
                    mask[20:40, 12:28] = 255
                    np.testing.assert_array_equal(result.outputs["unknown_mask"], mask)
                    assert result.outputs["report"]["unknown_count"] == 320
                if case == "palette_selection":
                    assert result.outputs["report"]["unknown_count"] == 0
                    assert result.outputs["report"]["samples"] == 96 * 96
                if run_profile == "debug":
                    assert graph.blocks["filmstrip"].state == "pass"
                    assert np.any(graph.blocks["overlay"].last_result.outputs["image"] != source)
                else:
                    assert not result.drawings and result.outputs["cluster_map"] is NO_VALUE
    finally:
        graph.close()


def test_json_xml_save_as_relocates_sealed_models_and_generator_preserves_assets(tmp_path):
    before = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    graph = Graph.load(BASE / "palette_ok.json", builtin_registry())
    try:
        graph.save(tmp_path / "copied.json")
        restored = Graph.load(tmp_path / "copied.json", builtin_registry())
        try:
            model_path = tmp_path / restored.blocks["model"].parameters["path"]
            assert (
                model_path.is_file()
                and model_path.read_bytes() == (BASE / "assets/palette-bgr.json").read_bytes()
            )
            restored.run()
            assert restored.blocks["predict"].last_result.passed
            export_project_xml(restored.to_dict(), tmp_path / "copied.xml")
            from_xml = Graph.from_dict(
                read_project_xml(tmp_path / "copied.xml"), builtin_registry(), project_dir=tmp_path
            )
            try:
                from_xml.run()
                assert from_xml.blocks["predict"].last_result.passed
            finally:
                from_xml.close()
        finally:
            restored.close()
    finally:
        graph.close()
    spec = importlib.util.spec_from_file_location("cluster_examples", BASE / "build_example.py")
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


def test_failed_admission_and_missing_connected_selection_prevent_new_work(monkeypatch):
    graph = Graph.load(BASE / "palette_selection.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["predict"]
        count, previous = block.timing.count, block.last_inputs["image"].copy()
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed_source(inputs):
            result = original(inputs)
            result.outputs["image"] = np.zeros_like(result.outputs["image"])
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed_source)
        graph.run()
        assert block.input_rejected and block.timing.count == count
        assert not block.current_emissions
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
        monkeypatch.setattr(graph.blocks["source"], "process", original)
        monkeypatch.setattr(
            graph.blocks["selection"],
            "process",
            lambda inputs: Result({"mask": NO_VALUE, "region": NO_VALUE, "overlay": NO_VALUE}),
        )
        graph.run()
        assert not graph.blocks["selection"].error
        assert block.timing.count == count and not block.current_emissions
    finally:
        graph.close()


def test_empty_selection_frame_mismatch_and_wrong_model_kind_are_explicit():
    image = palette_image()
    block = builtin_registry().create("ml.palette_train", parameters={"clusters": 1})
    assert block.execute({"image": image, "mask": np.zeros(image.shape[:2], np.uint8)}) is None
    assert "no visible pixels" in block.error
    first = {"spatial": frame([15, 4], "camera")}
    other = {"spatial": frame([15, 4], "other")}
    assert (
        block.execute(
            {
                "image": ValuePacket(image, first),
                "mask": ValuePacket(np.ones(image.shape[:2], np.uint8), other),
            }
        )
        is None
    )
    assert not block.current_emissions
    _, model = run("ml.palette_train", {"image": image}, clusters=3)
    block = builtin_registry().create("ml.kmeans_predict")
    assert block.execute({"model": model.outputs["model"], "samples": [[1, 2, 3]]}) is None
    assert "descriptor model" in block.error
    block = builtin_registry().create("ml.palette_predict")
    assert block.execute({"image": image, "model": trained()}) is None
    assert "image color palette" in block.error


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("ml.kmeans_train", {"clusters": 0}),
        ("ml.palette_train", {"max_samples": 0}),
        ("ml.palette_train", {"color_space": "hsv"}),
        ("ml.kmeans_predict", {"max_distance": -1}),
        ("ml.kmeans_predict", {"max_unknown_fraction": 1.1}),
        ("ml.kmeans_predict", {"accepted_clusters": [True]}),
        ("ml.kmeans_predict", {"accepted_clusters": [0, 0]}),
        ("ml.kmeans_predict", {"accepted_clusters": [32]}),
    ],
)
def test_parameter_constraints_reject_before_execution(kind, settings):
    block = builtin_registry().create(kind, parameters=settings)
    with pytest.raises(ValueError):
        block.validate_parameters()


def test_native_lab_distances_and_quantization_match_independent_full_precision_reference():
    image = palette_image()
    unit = image.astype(np.float32) / 255
    work = (cv2.cvtColor(unit, cv2.COLOR_BGR2Lab) / np.float32([100, 128, 128])).reshape(-1, 3)
    initial = np.broadcast_to(np.repeat([0, 1, 2], 5), (4, 15)).astype(np.int32).reshape(-1)
    _, _, native_centers = cv2.kmeans(
        work, 3, initial, (3, 50, 0.01), 1, cv2.KMEANS_PP_CENTERS | cv2.KMEANS_USE_INITIAL_LABELS
    )
    native_centers = np.array(sorted(native_centers.tolist()), np.float32)
    _, result = run("ml.palette_train", {"image": image}, color_space="lab", clusters=3, attempts=1)
    model = result.outputs["model"]
    np.testing.assert_allclose(model["centers"], native_centers, atol=1e-7, rtol=0)
    query = image.astype(np.uint16) * 257
    query[1:3, 1:4] = np.uint16([21000, 27000, 51000])
    native_work = cv2.cvtColor(query.astype(np.float32) / 65535, cv2.COLOR_BGR2Lab)
    native_work /= np.float32([100, 128, 128])
    centers = np.asarray(model["centers"], np.float32)
    distances = np.linalg.norm(native_work.astype(np.float64)[..., None, :] - centers, axis=3)
    labels = distances.argmin(axis=2)
    expected_bgr = cv2.cvtColor(
        (centers[labels] * np.float32([100, 128, 128])).astype(np.float32), cv2.COLOR_Lab2BGR
    )
    expected_pixels = np.rint(np.clip(expected_bgr, 0, 1).astype(np.float64) * 65535).astype(
        np.uint16
    )
    _, result = run("ml.palette_predict", {"model": model, "image": query}, max_distance=0.02)
    np.testing.assert_array_equal(result.outputs["labels"], labels)
    np.testing.assert_allclose(result.outputs["distances"], distances.min(axis=2), atol=1e-7)
    np.testing.assert_array_equal(result.outputs["quantized"], expected_pixels)
    np.testing.assert_array_equal(
        result.outputs["unknown_mask"], (distances.min(axis=2) > 0.02) * 255
    )


def test_saved_hog_vocabulary_centers_and_heldout_identity_have_independent_native_reference():
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
        _winSigma=-1,
        _L2HysThreshold=0.2,
        _gammaCorrection=True,
        _signedGradient=False,
    )
    expected = []
    for direction in ("vertical", "horizontal"):
        rows = [
            hog.compute(cv2.imread(str(BASE / f"assets/train-{direction}-{i}.png"), 0)).ravel()
            for i in range(3)
        ]
        expected.append(np.mean(rows, axis=0))
    order = sorted(range(2), key=lambda i: tuple(expected[i]))
    model = json.loads((BASE / "assets/vocabulary.json").read_bytes())
    np.testing.assert_allclose(model["centers"], np.asarray(expected)[order], atol=1e-7, rtol=0)
    for index, direction in enumerate(("vertical", "horizontal")):
        query = hog.compute(cv2.imread(str(BASE / f"assets/heldout-{direction}.png"), 0)).ravel()
        _, result = run(
            "ml.kmeans_predict",
            {"model": model, "samples": query.tolist()},
            contract_policy="dimensions_only",
        )
        assert result.outputs["labels"] == [order.index(index)]
        distances = np.linalg.norm(
            query.astype(np.float64)[None] - np.asarray(model["centers"]), axis=1
        )
        assert result.outputs["distances"] == pytest.approx([min(distances)])


def test_loader_read_budget_and_color_contract_are_enforced(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 101)
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_MODEL_BYTES", 100)
    block = builtin_registry().create("ml.kmeans_load", parameters={"path": str(path)})
    assert block.execute({}) is None and "8 MiB" in block.error
    monkeypatch.setattr("opencv_dataflow.cluster_models.MAX_MODEL_BYTES", 8 * 1024**2)
    _, result = run("ml.palette_train", {"image": palette_image()}, clusters=3)
    model = result.outputs["model"]
    for field, value in (("opencv_version", "0.0.0"), ("scaling", "unscaled")):
        changed = copy.deepcopy(model)
        changed["feature_contract"][field] = value
        changed["revision"] = revision(changed)
        block = builtin_registry().create("ml.palette_predict")
        assert block.execute({"model": changed, "image": palette_image()}) is None
        assert "preprocessing or OpenCV version differs" in block.error
    changed = copy.deepcopy(model)
    changed["centers"][0][0] = -100.0
    changed["revision"] = revision(changed)
    block = builtin_registry().create("ml.palette_predict")
    assert block.execute({"model": changed, "image": palette_image()}) is None
    assert "color-space range" in block.error
