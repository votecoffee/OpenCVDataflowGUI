"""Save As keeps existing assets accessible without copying or modifying them."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph
from opencv_dataflow.toolblocks import encapsulate


@pytest.fixture
def assets(tmp_path):
    original = tmp_path / "original"
    images = original / "images"
    images.mkdir(parents=True)
    pixels = np.arange(48, dtype=np.uint8).reshape(6, 8)
    assert cv2.imwrite(str(images / "part.png"), pixels)
    return original, pixels


@pytest.mark.parametrize("mode", ["single", "files", "directory"])
@pytest.mark.parametrize("nested", [False, True])
def test_save_as_reopens_same_images_without_touching_originals(tmp_path, assets, mode, nested):
    original, pixels = assets
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.file",
            block_id="source",
            parameters={
                "path": "images/part.png",
                "paths": ["images/part.png"],
                "directory": "images",
                "source_mode": mode,
                "path_base": "project",
                "color_format": "unchanged",
            },
        )
    )
    if nested:
        tool = encapsulate(graph, [source.id], "Nested acquisition", registry)
        source = tool.graph.blocks[source.id]
    project = original / "project.json"
    graph.save(project)
    original_bytes = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
    graph.run()
    np.testing.assert_array_equal(source.last_output_images["image"], pixels)
    destination = tmp_path / "saved elsewhere"
    destination.mkdir()
    saved = destination / "program.json"
    graph.save(saved)
    assert source.resolve_path(source.parameters["path"]).resolve() == original / "images/part.png"
    # Saving again is stable; opening the new program can acquire the same pixels.
    saved_bytes = saved.read_bytes()
    graph.save(saved)
    assert saved.read_bytes() == saved_bytes
    reopened = Graph.load(saved, registry)
    reopened.run()
    inner = next(iter(reopened.blocks.values())).graph if nested else reopened
    np.testing.assert_array_equal(inner.blocks["source"].last_output_images["image"], pixels)
    assert {
        path: path.read_bytes() for path in original.rglob("*") if path.is_file()
    } == original_bytes
    assert list(destination.iterdir()) == [saved]
    reopened.close()
    graph.close()


def test_first_save_and_working_directory_and_output_paths_keep_their_meaning(tmp_path, assets):
    original, _ = assets
    registry = builtin_registry()
    graph = Graph()
    relative = graph.add(
        registry.create(
            "input.file",
            parameters={
                "path": "images/part.png",
                "path_base": "project",
            },
        )
    )
    working = graph.add(registry.create("input.file", parameters={"path": "working.png"}))
    absolute = graph.add(
        registry.create(
            "input.file",
            parameters={
                "path": str(original / "images/part.png"),
                "path_base": "project",
            },
        )
    )
    output = graph.add(
        registry.create(
            "output.lens_maps",
            parameters={
                "path": "results/maps.npz",
                "path_base": "project",
            },
        )
    )
    graph.save(original / "program.json")
    assert relative.parameters["path"] == "images/part.png"
    preserved = {b.id: copy.deepcopy(b.parameters) for b in (working, absolute, output)}
    graph.save(tmp_path / "relocated.json")
    assert {b.id: b.parameters for b in (working, absolute, output)} == preserved
    assert (graph.project_dir / relative.parameters["path"]).resolve().is_file()
    graph.close()


class PluginAssets(Block):
    type_id = "test.plugin_assets"
    defaults = {
        "recipe_path": "recipes/model.json",
        "training_images": ["images/part.png"],
        "documentation": "images/part.png",
        "endpoint": "https://example.org/model",
    }
    project_path_parameters = ("recipe_path", "training_images", "endpoint")


def test_save_as_keeps_declared_plugin_calibration_and_profile_assets(tmp_path, assets):
    original, _ = assets
    registry = builtin_registry()
    registry.register(PluginAssets)
    graph = Graph()
    plugin = graph.add(PluginAssets(block_id="plugin"))
    calibration = graph.add(
        registry.create(
            "calibration.lens",
            parameters={
                "image_paths": ["images/part.png"],
                "path_base": "project",
            },
        )
    )
    metadata = graph.add(registry.create("cv.image_metadata", parameters={"icc_path": "rgb.icc"}))
    graph.save(original / "program.json")
    graph.save(tmp_path / "relocated.json")
    reopened = Graph.load(tmp_path / "relocated.json", registry)
    assert (graph.project_dir / plugin.parameters["recipe_path"]).resolve() == (
        original / "recipes/model.json"
    )
    assert (graph.project_dir / plugin.parameters["training_images"][0]).resolve().is_file()
    assert plugin.parameters["documentation"] == "images/part.png"
    assert plugin.parameters["endpoint"] == "https://example.org/model"
    assert (graph.project_dir / calibration.parameters["image_paths"][0]).resolve().is_file()
    assert (graph.project_dir / metadata.parameters["icc_path"]).resolve() == original / "rgb.icc"
    assert reopened.to_dict() == graph.to_dict()
    reopened.close()
    graph.close()


@pytest.mark.parametrize("failure", [None, "serialize", "write"])
def test_validation_assets_and_failed_save_preserve_references(
    tmp_path, assets, monkeypatch, failure
):
    original, _ = assets
    graph = Graph()
    source = graph.add(
        builtin_registry().create(
            "input.file",
            parameters={
                "path": "images/part.png",
                "path_base": "project",
            },
        )
    )
    graph.validation_sets = [
        {
            "id": "validation",
            "name": "Parts",
            "source": source.id,
            "judgment": source.id,
            "cases": [{"id": "part", "path": "images/part.png", "expectation": "unknown"}],
        }
    ]
    graph.save(original / "program.json")
    before = copy.deepcopy(graph.to_dict())
    parameters = source.parameters
    if failure == "serialize":
        monkeypatch.setattr(graph, "to_dict", lambda: {"invalid": float("nan")})
    elif failure == "write":

        def blocked(*args, **kwargs):
            raise OSError("Destination unavailable")

        monkeypatch.setattr(Path, "write_text", blocked)
    target = tmp_path / "relocated.json"
    if failure:
        with pytest.raises((ValueError, OSError)):
            graph.save(target)
        monkeypatch.undo()
        assert graph.to_dict() == before
        assert source.parameters is parameters
        assert graph.project_dir == source.project_dir == original
        assert not target.exists()
    else:
        graph.save(target)
        assert (
            (graph.project_dir / graph.validation_sets[0]["cases"][0]["path"]).resolve().is_file()
        )
        assert Graph.load(target, builtin_registry()).validation_sets == graph.validation_sets
    graph.close()
