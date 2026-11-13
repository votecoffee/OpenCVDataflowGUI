"""Keep sealed recipes and their dependencies usable after a project Save As."""

import copy
import hashlib
from pathlib import Path

import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/Simple Plugins"


class Recipe(Block):
    type_id = "test.sealed_recipe"
    defaults = {"recipe_path": "assets/recipe.bin"}


@pytest.mark.parametrize("conflict", [False, True])
def test_existing_recipe_is_reused_only_when_bytes_match(tmp_path, conflict):
    original, destination = tmp_path / "original", tmp_path / "new"
    for root in (original, destination):
        (root / "assets").mkdir(parents=True)
    (original / "assets/recipe.bin").write_bytes(b"original sealed model")
    incoming = b"another model" if conflict else b"original sealed model"
    existing = destination / "assets/recipe.bin"
    existing.write_bytes(incoming)
    stat = existing.stat()
    graph = Graph()
    block = graph.add(Recipe())
    graph.save(original / "project.json")
    before = copy.deepcopy(graph.to_dict())
    if conflict:
        with pytest.raises(FileExistsError, match="existing asset"):
            graph.save(destination / "project.json")
        assert graph.project_dir == original
        assert graph.to_dict() == before
        assert not (destination / "project.json").exists()
    else:
        graph.save(destination / "project.json")
        assert graph.project_dir == destination
        assert block.parameters["recipe_path"] == "assets/recipe.bin"
    assert existing.read_bytes() == incoming and existing.stat().st_mtime_ns == stat.st_mtime_ns
    graph.close()


def test_failed_save_removes_only_copies_created_by_that_save(tmp_path, monkeypatch):
    original, destination = tmp_path / "original", tmp_path / "new"
    (original / "assets").mkdir(parents=True)
    (original / "assets/recipe.bin").write_bytes(b"pinned model")
    destination.mkdir()
    retained = destination / "unrelated.txt"
    retained.write_bytes(b"keep this")
    graph = Graph()
    graph.add(Recipe())
    graph.save(original / "project.json")
    before = copy.deepcopy(graph.to_dict())
    monkeypatch.setattr(graph, "to_dict", lambda: {"invalid": float("nan")})
    with pytest.raises(ValueError):
        graph.save(destination / "project.json")
    monkeypatch.undo()
    assert list(destination.iterdir()) == [retained]
    assert retained.read_bytes() == b"keep this"
    assert graph.to_dict() == before and graph.project_dir == original
    assert (original / "assets/recipe.bin").read_bytes() == b"pinned model"
    graph.close()


@pytest.mark.parametrize("relative", ["../outside.bin", "program.json"])
def test_recipe_cannot_escape_or_replace_the_program_file(tmp_path, relative):
    original = tmp_path / "original"
    original.mkdir()
    graph = Graph()
    graph.add(Recipe(parameters={"recipe_path": relative}))
    graph.save(original / "program.json")
    with pytest.raises(ValueError, match="project folder|program filename"):
        graph.save(tmp_path / "program.json")
    assert not (tmp_path / "program.json").exists()
    assert graph.project_dir == original
    graph.close()


@pytest.mark.local_examples
def test_sealed_3d_measurement_recipe_survives_save_as_and_native_replay(tmp_path):
    pytest.importorskip("opencv_simple_3d_measure.dataflow")
    registry = builtin_registry()
    original = EXAMPLES / "3d_measure/generated/good/project.json"
    graph = Graph.load(original, registry)
    try:
        graph.run()
        assert graph.blocks["measure"].state == "pass", graph.blocks["measure"].error
        expected = copy.deepcopy(graph.blocks["measure"].last_result.outputs["measurements"])
        relative = graph.blocks["measure"].parameters["recipe_path"]
        recipe = (original.parent / relative).read_bytes()
        source_project = original.read_bytes()
        destination = tmp_path / "measurement.json"
        graph.save(destination)
        assert (tmp_path / relative).read_bytes() == recipe
        assert original.read_bytes() == source_project
        reopened = Graph.load(destination, registry)
        try:
            reopened.run()
            measured = reopened.blocks["measure"]
            assert measured.state == "pass", measured.error
            assert measured.last_result.outputs["measurements"] == expected
            assert (
                hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
                == (measured.parameters["sha256"])
            )
        finally:
            reopened.close()
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("project", ["project.json", "color_and_layout.json"])
def test_classifier_bundle_replays_and_can_review_training_images_after_save_as(tmp_path, project):
    from opencv_dataflow.simple_classifier.model import load_assignments

    registry = builtin_registry()
    graph = Graph.load(EXAMPLES / "image_classification/vegqual" / project, registry)
    try:
        graph.run()
        classifier = next(
            b for b in graph.blocks.values() if b.type_id == "inspect.simple_classify_image"
        )
        assert classifier.state == "pass", classifier.error
        expected = copy.deepcopy(classifier.last_result.outputs["probabilities"])
        manifest = classifier.parameters["model"]
        frozen = {p: (graph.project_dir / p).read_bytes() for p in classifier.project_assets()}
        destination = tmp_path / "classifier.json"
        graph.save(destination)
        assert {p: (tmp_path / p).read_bytes() for p in frozen} == frozen
        assignments, _ = load_assignments(tmp_path, manifest)
        assert all(e.path.is_file() for rows in assignments.values() for e in rows)
        reopened = Graph.load(destination, registry)
        try:
            reopened.run()
            result = reopened.blocks[classifier.id]
            assert result.state == "pass", result.error
            assert result.last_result.outputs["probabilities"] == expected
        finally:
            reopened.close()
    finally:
        graph.close()
