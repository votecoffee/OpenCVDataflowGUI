"""Native ONNX inference, Annoy search/persistence, and optional learned LightGlue checks."""

import builtins
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QFileDialog

from opencv_dataflow.annoy_blocks import BuildAnnoy, LoadAnnoy, SaveAnnoy, SearchAnnoy
from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import ClassifyTensor, DNNInference, TensorVector, preprocess
from opencv_dataflow.examples import annoy_graph, dnn_graph, lightglue_graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.lightglue_blocks import LightGlueFeatures, LightGlueMatch, validate_features
from opencv_dataflow.ml_common import selected_tensor, tensor_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.poses import validate_pose
from opencv_dataflow.tickets import TicketClock

ONNX = str(Path("examples/assets/rgb_average.onnx").resolve())
WEIGHTS = Path(".local/models/sift_lightglue.pth")


def execute(block, inputs, expected="pass"):
    block.execute(inputs)
    assert block.state == expected, block.error
    return block.last_result


def inference(**parameters):
    return DNNInference(parameters={"model_path": ONNX, "width": 32, "height": 32, **parameters})


@pytest.mark.parametrize("engine", ["auto", "classic", "new"])
@pytest.mark.parametrize("profile", ["debug", "balanced", "lean"])
def test_real_onnx_channels_normalization_ticket_and_profiles(engine, profile):
    graph = Graph()
    block = graph.add(inference(engine=engine))
    graph.set_run_profile(profile)
    ticket = TicketClock().issue("camera")
    image = ImageArray(np.full((40, 50, 3), [0, 0, 255], np.uint8), "bgr")
    result = execute(block, {"image": ValuePacket(image, {"ticket": ticket})})
    np.testing.assert_allclose(selected_tensor(result.outputs["tensors"]), [[1, 0, 0]], atol=1e-6)
    assert result.payloads["tensors"]["ticket"] == ticket
    assert block.timing.count == 1
    first = block._net
    execute(block, {"image": image})
    assert block._net is first
    block.close()
    assert block._net is None


def test_preprocessing_preserves_declared_rgb_and_uint16_units():
    image = ImageArray(np.full((3, 4, 3), [65535, 0, 0], np.uint16), "rgb")
    parameters = {**DNNInference.defaults, "width": 4, "height": 3, "scale": 1 / 65535}
    blob, _ = preprocess({"image": image}, parameters)
    np.testing.assert_allclose(blob[0, :, 0, 0], [1, 0, 0], atol=1e-6)
    parameters.update(mean=[65535, 0, 0], std=[2, 1, 1])
    blob, _ = preprocess({"image": image}, parameters)
    np.testing.assert_allclose(blob, 0, atol=1e-6)


def test_letterbox_transform_mask_and_model_paths(tmp_path):
    source = np.full((10, 20, 3), 255, np.uint8)
    mask = np.zeros((10, 20), np.uint8)
    mask[:, :10] = 255
    p = {**DNNInference.defaults, "width": 32, "height": 32, "resize_mode": "letterbox"}
    blob, geometry = preprocess({"image": source, "mask": mask}, p)
    assert blob.shape == (1, 3, 32, 32)
    np.testing.assert_allclose(geometry["input_to_network"], [[1.6, 0, 0], [0, 1.6, 8], [0, 0, 1]])
    np.testing.assert_allclose(
        np.asarray(geometry["network_to_input"]) @ geometry["input_to_network"], np.eye(3)
    )
    assert np.all(blob[:, :, :8] == 0) and np.all(blob[:, :, :, 20:] == 0)
    path = tmp_path / "模型.onnx"
    path.write_bytes(Path(ONNX).read_bytes())
    block = inference(model_path=path.name)
    block.project_dir = tmp_path
    execute(block, {"image": source})


@pytest.mark.parametrize(
    "parameters,fragment",
    [
        ({"std": [0, 1, 1]}, "std"),
        ({"mean": [0]}, "Mean"),
        ({"width": 0}, "Width"),
        ({"max_output_elements": 2}, "limit"),
        ({"output_names": ["scores", "scores"]}, "unique"),
        ({"engine": "auto", "fusion": False}, "classic"),
        ({"target": "cuda"}, "CUDA"),
        ({"color_format": "wrong"}, "Color format"),
    ],
)
def test_invalid_dnn_settings_are_diagnostics(parameters, fragment):
    block = inference(**parameters)
    execute(block, {"image": np.zeros((8, 8, 3), np.uint8)}, "error")
    assert fragment.casefold() in block.error.casefold()


def test_tensor_classification_and_embedding_have_bounded_clear_contracts():
    tensors = {"scores": tensor_record(np.array([[1000.0, 1001.0, 999.0]]), 10)}
    classify = ClassifyTensor(parameters={"labels": ["A", "B", "C"], "accepted_labels": ["A"]})
    result = execute(classify, {"tensors": tensors}, "fail")
    assert result.outputs["label"] == "B" and result.fail_reasons[0]["code"] == "classification"
    assert sum(v["confidence"] for v in result.outputs["predictions"]) == pytest.approx(1)
    embed = TensorVector()
    result = execute(embed, {"tensors": tensors})
    assert np.linalg.norm(result.outputs["vector"]) == pytest.approx(1)
    tensor = tensor_record(np.arange(12).reshape(1, 3, 2, 2), 12)
    embed.parameters.update(pooling="global_average", normalize="none")
    assert execute(embed, {"tensors": {"values": tensor}}).outputs["vector"] == [1.5, 5.5, 9.5]
    execute(embed, {"tensors": {"values": {**tensor, "shape": [1, 2]}}}, "error")


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan", "hamming", "dot"])
def test_native_annoy_metrics_and_dot_similarity(metric):
    pytest.importorskip("annoy")
    build = BuildAnnoy(parameters={"metric": metric, "labels": ["X", "Y", "XY"]})
    model = execute(build, {"vectors": [[1, 0], [0, 1], [1, 1]]}).outputs["model"]
    query = SearchAnnoy(parameters={"neighbors": 3, "search_k": 10000})
    result = execute(query, {"model": model, "vector": [0, 1]})
    rows = result.outputs["neighbors"]
    if metric == "dot":
        assert rows[0]["similarity"] == 1 and rows[-1]["similarity"] == 0
        assert all(row["distance"] is None for row in rows)
    else:
        assert rows[0]["label"] == "Y" and rows[0]["distance"] == pytest.approx(0)
    index = query._index
    execute(query, {"model": model, "vector": [1, 0]})
    assert query._index is index
    query.close()
    assert query._index is None


def test_annoy_save_reload_integrity_and_no_overwrite(tmp_path):
    pytest.importorskip("annoy")
    build = BuildAnnoy(parameters={"metric": "euclidean", "normalize": "l2"})
    model = execute(build, {"vectors": [[3, 0], [0, 4]]}).outputs["model"]
    saver = SaveAnnoy(parameters={"path": str(tmp_path / "items.json")})
    execute(saver, {"model": model})
    before = (tmp_path / "items.ann").read_bytes()
    execute(saver, {"model": model}, "error")
    assert (tmp_path / "items.ann").read_bytes() == before
    load = LoadAnnoy(parameters={"path": "items.json"})
    load.project_dir = tmp_path
    loaded = execute(load, {}).outputs["model"]
    search = SearchAnnoy()
    assert execute(search, {"model": loaded, "vector": [10, 0]}).outputs["label"] == "0"
    search.close()
    (tmp_path / "items.ann").write_bytes(before + b"changed")
    execute(search, {"model": loaded, "vector": [10, 0]}, "error")
    assert "checksum" in search.error


@pytest.mark.parametrize(
    "vectors,parameters",
    [
        ([[0, 0]], {}),
        ([[1, 0]], {"labels": ["A", "B"]}),
        ([[1, 2]], {"metric": "hamming"}),
        ([[0, 0]], {"normalize": "l2"}),
        ([[1, 0]], {"trees": 0}),
    ],
)
def test_bad_annoy_training_does_not_create_native_handles(vectors, parameters):
    pytest.importorskip("annoy")
    block = BuildAnnoy(parameters=parameters)
    execute(block, {"vectors": vectors}, "error")


def test_annoy_distance_filter_is_a_failed_inspection_not_exception():
    pytest.importorskip("annoy")
    model = execute(BuildAnnoy(parameters={"metric": "euclidean"}), {"vectors": [[0, 0]]}).outputs[
        "model"
    ]
    result = execute(
        SearchAnnoy(parameters={"max_distance": 0.1}), {"model": model, "vector": [10, 10]}, "fail"
    )
    assert result.outputs["neighbors"] == [] and result.outputs["label"] is None


def textured_pair():
    rng = np.random.default_rng(4)
    image = cv2.GaussianBlur(rng.integers(0, 255, (160, 200), np.uint8), (3, 3), 0)
    query = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 6]]), (200, 160))
    return image, query


def test_sift_reference_features_mask_and_serializable_coordinates():
    image, _ = textured_pair()
    mask = np.zeros(image.shape, np.uint8)
    mask[:, :100] = 255
    block = LightGlueFeatures(parameters={"max_features": 128})
    result = execute(block, {"image": image, "mask": mask})
    features = result.outputs["features"]
    assert 0 < result.outputs["count"] <= 128
    assert all(point[0] < 100 for point in features["points"])
    points, descriptors, _, _ = validate_features(json.loads(json.dumps(features)))
    assert len(points) == len(descriptors)
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1, atol=1e-5)


def test_empty_lightglue_match_emits_no_fake_pose():
    result = execute(
        LightGlueMatch(),
        {"image": np.zeros((30, 30), np.uint8), "reference": np.zeros((30, 30), np.uint8)},
        "fail",
    )
    assert result.outputs["pose"] is NO_VALUE and result.outputs["transform"] is NO_VALUE
    assert result.outputs["count"] == 0


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Optional official LightGlue weights not prepared")
def test_real_lightglue_local_weights_geometry_cache_and_no_download(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightglue")
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *a, **k: pytest.fail("Inference must not download weights"),
    )
    image, query = textured_pair()
    block = LightGlueMatch(parameters={"weights_path": str(WEIGHTS), "max_features": 64})
    result = execute(block, {"image": query, "reference": image})
    assert result.outputs["count"] >= 20 and result.outputs["inliers"] >= 20
    matrix = np.asarray(result.outputs["transform"]["source_to_target"])
    np.testing.assert_allclose(matrix[:2, 2], [8, 6], atol=0.2)
    validate_pose(result.outputs["pose"])
    cached = block._matcher
    reference = execute(
        LightGlueFeatures(parameters={"max_features": 64}), {"image": image}
    ).outputs["features"]
    execute(block, {"image": query, "reference_features": reference})
    assert block._matcher is cached
    block.close()
    assert block._matcher is None


@pytest.mark.parametrize(
    "filename,factory",
    [
        ("dnn_embeddings.json", dnn_graph),
        ("annoy_search.json", annoy_graph),
        ("lightglue_match.json", lightglue_graph),
    ],
)
def test_ml_examples_roundtrip_and_registry(filename, factory):
    r = builtin_registry()
    graph = Graph.load(Path("examples") / filename, r)
    assert graph.to_dict() == factory(r).to_dict()
    json.dumps(graph.to_dict(), allow_nan=False)


def test_ml_inspector_exposes_enums_paths_and_library(app, monkeypatch):
    window = MainWindow()
    window.load_ml_example(dnn_graph)
    window.scene.nodes["ml_net"].setSelected(True)
    for name in ("engine", "backend", "target", "color_format", "resize_mode"):
        assert isinstance(window.inspector.fields[name], QComboBox)
    assert "model_path" in window.inspector.fields
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (ONNX, ""))
    window.inspector.browse_path(window.inspector.fields["model_path"])
    assert window.inspector.fields["model_path"].text() == ONNX
    assert window.scene.nodes["ml_net"].category == "ML"
    window.load_ml_example(lightglue_graph)
    window.scene.nodes["lg_match"].setSelected(True)
    assert isinstance(window.inspector.fields["device"], QComboBox)
    assert "weights_path" in window.inspector.fields
    window.dirty = False
    window.close()


@pytest.mark.parametrize("missing", ["annoy", "torch"])
def test_optional_dependencies_are_lazy_and_fail_with_install_guidance(monkeypatch, missing):
    original = builtins.__import__

    def import_without_optional(name, *args, **kwargs):
        if name.split(".")[0] == missing:
            raise ImportError(f"Simulated missing {missing}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optional)
    registry = builtin_registry()
    assert "ml.lightglue" in registry.types and "ml.annoy_search" in registry.types
    if missing == "annoy":
        block = BuildAnnoy()
        execute(block, {"vectors": [[1, 0]]}, "error")
        assert "optional" in block.error
    else:
        image, query = textured_pair()
        block = LightGlueMatch(parameters={"weights_path": ONNX, "max_features": 32})
        execute(block, {"image": query, "reference": image}, "error")
        assert "Install PyTorch" in block.error
