import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import lena_features_graph, lena_graph, lena_registration_graph
from opencv_dataflow.toolblocks import encapsulate


def make_bundle(directory):
    directory.mkdir()
    assets = directory / "assets"
    assets.mkdir()
    image = np.random.default_rng(17).integers(0, 256, (512, 512, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    (assets / "lena.jpg").write_bytes(encoded.tobytes())
    project = directory / "getting_started" / "lena" / "lena.json"
    project.parent.mkdir(parents=True)
    lena_graph(builtin_registry()).save(project)
    return project


def test_project_relative_image_survives_move_clone_and_nested_graph(tmp_path, monkeypatch):
    registry = builtin_registry()
    project = make_bundle(tmp_path / "bundle")
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(project, registry)
    graph.run()
    assert all(b.state == "pass" for b in graph.processing_blocks())
    assert graph.blocks["lena_match"].last_result.outputs["location"]["top_left"] == [220, 200]
    assert graph.blocks["lena_source"].parameters["path"] == "../../assets/lena.jpg"
    clone = Graph.from_dict(graph.to_dict(), registry, project_dir=graph.project_dir)
    clone.run()
    assert clone.blocks["lena_source"].state == "pass"
    composite = encapsulate(graph, ["lena_source"], "Image input", registry)
    graph.run()
    assert graph.blocks[composite.id].state == "pass", graph.blocks[composite.id].error
    graph.save(project)
    restored = Graph.load(project, registry)
    restored.run()
    assert all(b.state == "pass" for b in restored.processing_blocks())


def test_unsaved_project_relative_input_is_explicit_and_legacy_default_unchanged():
    registry = builtin_registry()
    block = registry.create(
        "input.file", parameters={"path": "assets/lena.jpg", "path_base": "project"}
    )
    block.execute({})
    assert "Save or open the project" in block.error
    assert registry.create("input.file").parameters["path_base"] == "working_directory"


@pytest.mark.skipif(
    not Path("examples/assets/lena.jpg").exists(), reason="Optional Lena asset not fetched"
)
def test_real_lena_matches_expected_asset_and_repeated_results():
    data = Path("examples/assets/lena.jpg").read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07"
    )
    registry = builtin_registry()
    graph = Graph.load("examples/opencv tools/lena/lena.json", registry)
    assert graph.to_dict() == lena_graph(registry).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        assert graph.blocks["lena_source"].last_output_images["image"].shape == (512, 512, 3)
        assert graph.blocks["lena_match"].last_result.outputs["score"] > 0.999
        assert graph.blocks["lena_match"].last_result.outputs["location"]["top_left"] == [220, 200]
        histogram = graph.blocks["lena_color_hist"].last_result.outputs["histogram"]
        assert len(histogram["channels"]) == 3
        assert all(sum(c["counts"]) == 512 * 512 for c in histogram["channels"])
        assert graph.blocks["lena_edges"].last_output_images["image"].any()


def test_lena_downloader_preserves_mismatched_existing_asset(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_lena", "scripts/prepare_lena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "existing.jpg"
    path.write_bytes(b"preserve this file")
    with pytest.raises(ValueError, match="preserved"):
        module.prepare(path)
    assert path.read_bytes() == b"preserve this file"


@pytest.mark.skipif(
    not Path("examples/assets/lena.jpg").exists(), reason="Optional Lena asset not fetched"
)
def test_lena_feature_pipeline_and_saved_graph():
    registry = builtin_registry()
    graph = Graph.load("examples/opencv tools/lena_features/lena_features.json", registry)
    assert graph.to_dict() == lena_features_graph(registry).to_dict()
    graph.run()
    assert all(b.state == "pass" for b in graph.processing_blocks())
    assert 0 < graph.blocks["feature_corners"].last_result.outputs["count"] <= 100
    assert graph.blocks["feature_lines"].last_result.outputs["count"] > 0
    histogram = graph.blocks["feature_histogram"].last_result.outputs["histogram"]
    assert sum(histogram["channels"][0]["counts"]) == 512 * 512


@pytest.mark.skipif(
    not Path("examples/assets/lena.jpg").exists(), reason="Optional Lena asset not fetched"
)
def test_lena_registration_matches_known_rotation():
    registry = builtin_registry()
    graph = Graph.load(
        "examples/opencv tools/lena_registration/lena_registration.json", registry
    )
    assert graph.to_dict() == lena_registration_graph(registry).to_dict()
    graph.run()
    assert all(b.state == "pass" for b in graph.processing_blocks()), [
        (b.name, b.error) for b in graph.processing_blocks()
    ]
    known = np.linalg.inv(
        np.array(
            graph.blocks["registration_rotate"].last_result.outputs["geometry"]["output_to_input"]
        )
    )
    found = np.array(
        graph.blocks["registration_estimate"].last_result.outputs["transform"]["source_to_target"]
    )
    points = np.float64([[[100, 100], [200, 150], [300, 300], [400, 350]]])
    expected = cv2.perspectiveTransform(points, known)
    actual = cv2.perspectiveTransform(points, found)
    np.testing.assert_allclose(actual, expected, atol=3.0)
    assert graph.blocks["registration_estimate"].last_result.outputs["inliers"] >= 20
    assert graph.blocks["registration_difference"].last_output_images["image"].shape == (
        512,
        512,
        3,
    )
