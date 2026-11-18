"""Qualify the explicit trainer wheel without changing the host installation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/pattern_matching/training_workflow"
)


def test_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    assert not any("__pycache__" in name for name in manifest["sha256"])
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("brackets", "visa_pcb2"))
def test_explicit_training_package_restores_and_relocates_exact_models(tmp_path, case):
    program = """
import hashlib,sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import ProjectSession,make_registry
from opencv_dataflow.plugin_dialog import PluginDialogContext
root,out,case=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
registry=make_registry(plugins=[str(root/'plugin.py')])
import opencv_geometric_pattern_match as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
for filename in ('project.json','refined_project.json'):
    with ProjectSession.open(root/case/filename,registry) as source:
        assert source.inspect()['validation']['valid']
        block=source.graph.blocks['matcher']
        context=PluginDialogContext(source.graph,block)
        dialog=block.create_plugin_dialog(None,block.parameters.copy(),context)
        assert dialog._reference is not None and 'Fine edges:' in dialog.health.text()
        assert dialog._refinement_comparison is None
        if filename.startswith('refined'):
            assert dialog._scene is not None
            if case=='visa_pcb2':
                assert 'Fine weights:' in dialog.health.text()
        dialog.reject();context._close()
        original=root/case/block.parameters['recipe_path']
        members=[Path(p) for p in block.project_assets()]
        expected={str(p):hashlib.sha256((root/case/p).read_bytes()).hexdigest() for p in members}
        assert any(str(p).endswith('.reference.png') for p in members)
        target=out/filename/'project.json';target.parent.mkdir(parents=True)
        source.save(target)
    with ProjectSession.open(target,registry) as moved:
        assert moved.inspect()['validation']['valid']
        for name,digest in expected.items():
            assert hashlib.sha256((target.parent/name).read_bytes()).hexdigest()==digest
        block=moved.graph.blocks['matcher']
        context=PluginDialogContext(moved.graph,block)
        dialog=block.create_plugin_dialog(None,block.parameters.copy(),context)
        assert dialog._reference is not None and dialog._pattern is not None
        assert 'Fine edges:' in dialog.health.text()
        dialog.reject();context._close()
app.processEvents()
"""
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path), case],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
