"""Replay the portable plugin template and the generated assistant example."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, block_schema, make_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/user_tools/assistant_workflow"
PLUGIN = ROOT / "skills/opencv-dataflow-plugins/assets/mean_inspection.py"


def module(path):
    spec = importlib.util.spec_from_file_location("assistant_helper", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_portable_plugin_matches_builder_pass_and_negative_example():
    registry = make_registry([PLUGIN])
    with ProjectSession(registry) as session:
        session.apply(module(EXAMPLE / "build_example.py").operations())
        saved = json.loads((EXAMPLE / "assistant_workflow.json").read_text())
        assert session.graph.to_dict() == saved
        assert session.run()["passed"]
        mean = session.graph.blocks["mean"].last_result
        assert 20 < mean.outputs["mean"] < 100
        assert session.graph.blocks["judgment"].last_result.final_judgment
        annotated = session.graph.blocks["review"].last_result.outputs["image"]
        original = session.graph.blocks["source"].last_result.outputs["image"]
        assert np.count_nonzero(annotated != original) > 0
        session.apply(json.loads((EXAMPLE / "fail_edits.json").read_text()))
        assert not session.run()["passed"]
        assert session.graph.blocks["mean"].last_result.fail_reasons[0]["code"] == "mean_intensity"
        assert session.graph.blocks["judgment"].last_result.final_judgment is False


def test_plugin_selection_graphics_and_original_pixels_are_preserved():
    block = make_registry([PLUGIN]).create("example.mean_inspection")
    image = np.full((12, 20, 3), 50, np.uint8)
    image[:, 10:] = 200
    mask = np.zeros((12, 20), np.uint8)
    mask[:, :10] = 255
    block.execute({"image": image, "mask": mask})
    assert block.state == "pass" and block.last_result.outputs["mean"] == 50
    np.testing.assert_array_equal(block.last_result.outputs["image"], image)
    assert block.last_result.drawings
    block.set_graphics_mode("off")
    block.execute({"image": image, "mask": mask})
    assert not block.last_result.drawings
    block.parameters["comparison"] = "outside"
    block.execute({"image": image, "mask": mask})
    assert block.state == "fail"
    block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum": -1},
        {"maximum": 256},
        {"minimum": 120, "maximum": 100},
        {"comparison": "typo"},
    ],
)
def test_template_invalid_controls_are_validation_errors(parameters):
    block = make_registry([PLUGIN]).create("example.mean_inspection", parameters=parameters)
    block.execute({"image": np.ones((5, 5), np.uint8)})
    assert block.state == "error" and block.error_kind == "invalid"


def test_installer_preserves_different_skills_and_excludes_bytecode(tmp_path):
    installer = module(ROOT / "scripts/install_assistant_skills.py")
    records = installer.install(tmp_path)
    assert len(records) == 2 and all(r["state"] == "installed" for r in records)
    assert all(not any("__pycache__" in path for path in r["files"]) for r in records)
    assert all(r["state"] == "unchanged" for r in installer.install(tmp_path))
    path = tmp_path / "opencv-dataflow-apps/SKILL.md"
    path.write_text("User-owned edit", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Preserved existing skill"):
        installer.install(tmp_path)
    assert path.read_text() == "User-owned edit"


def test_api_saved_example_opens_in_editor(app, tmp_path):
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.plugins import load_plugin

    window = MainWindow()
    try:
        load_plugin(PLUGIN, window.registry)
        with ProjectSession.open(EXAMPLE / "assistant_workflow.json", window.registry) as session:
            session.save(tmp_path / "inspection.json")
        window.replace_graph(Graph.load(tmp_path / "inspection.json", window.registry))
        window.scene.nodes["mean"].setSelected(True)
        assert set(window.inspector.fields) >= {"minimum", "maximum", "comparison"}
        schema = block_schema("example.mean_inspection", window.registry)
        assert schema["parameters"]["comparison"]["choices"] == ["inside", "outside"]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
