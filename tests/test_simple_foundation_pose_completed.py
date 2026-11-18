"""Portable wheel/asset/native replay checks; never run ONNX or select a runtime."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/foundation_pose/completed_runtime"
)


def test_delivered_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_explicit_loader_and_exact_recipe_save_as_without_heavy_imports(tmp_path):
    program = """
import sys
from pathlib import Path
from opencv_dataflow.assistant_api import ProjectSession,make_registry
root,destination=map(Path,sys.argv[1:])
registry=make_registry(plugins=[str(root/'plugin.py')])
block=registry.create('3d.simple_foundation_pose')
assert block.plugin_review_dialog_label and block.preview_output=='cloud'
assert 'onnxruntime' not in sys.modules and 'torch' not in sys.modules
for name in ('held_out_frame13','different_object_negative','global_cohort'):
    with ProjectSession.open(root/name/'run_project.json',registry) as session:
        assert session.inspect()['validation']['valid']
        target=destination/name/'project.json';target.parent.mkdir(parents=True)
        session.save(target)
    with ProjectSession.open(target,registry) as reopened:
        assert reopened.inspect()['validation']['valid']
        block=reopened.graph.blocks['pose']
        assert (target.parent/block.parameters['recipe_path']).read_bytes()==(root/name/'trained.sfp').read_bytes()
assert 'onnxruntime' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("case", ("held_out_frame13", "different_object_negative", "global_cohort"))
def test_native_recorded_cloud_replay(case):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from opencv_dataflow.assistant_api import ProjectSession
    from opencv_dataflow.cloud_viewer import CloudViewer

    app = QApplication.instance() or QApplication([])
    with ProjectSession.open(ROOT / case / "review_project.json") as session:
        assert session.inspect()["validation"]["valid"]
        session.run()
        count = 0
        for block in session.graph.blocks.values():
            assert block.state != "error", block.error
            if block.type_id == "input.cloud":
                cloud = block.last_result.outputs["cloud"]
                assert cloud.colors is not None and cloud.units == "m" and cloud.frame == "camera"
                view = CloudViewer(None)
                view.set_cloud(cloud)
                assert view.flyaround.isEnabled()
                view.advance_orbit()
                assert view._rendered is not None
                view.close()
                count += 1
        assert count == (1 if case == "global_cohort" else 2)
    app.processEvents()
