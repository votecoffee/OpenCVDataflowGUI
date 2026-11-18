"""Portable retained-source training checks without changing the host install."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/pixel_segmentation/training_workflow"
)


def test_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("pcb", "tote"))
def test_relocated_training_popup_restores_inputs_and_legacy_model_stays_supported(tmp_path, case):
    code = """
import hashlib,json,sys,time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import ProjectSession,make_registry
from opencv_dataflow.plugin_dialog import PluginDialogContext
root,out,case=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
registry=make_registry(plugins=[str(root/'plugin.py')])
import opencv_defect_segmentation as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
def wait(dialog):
    deadline=time.monotonic()+30
    while dialog._future is not None:
        app.processEvents();time.sleep(.005)
        assert time.monotonic()<deadline
def review(session,retained):
    block=session.graph.blocks['segment']
    context=PluginDialogContext(session.graph,block)
    parameters=block.parameters.copy()
    dialog=block.create_plugin_dialog(None,parameters,context)
    wait(dialog)
    if retained:
        assert 'Restored' in dialog.summary.text(),dialog.summary.text()
        assert 'No prediction' in dialog.preview_title.text()
        assert not dialog.preview.pixmap().isNull()
        manifest=Path(dialog.paths['training'].text())
        assert manifest.is_relative_to(context.project_dir/'.local')
        assert dialog._pinned_fingerprints
        dialog.labels.click()
        assert dialog._editor is not None
        expected=Path(json.loads((root/case/'training.json').read_bytes())[0]['image']).name
        assert dialog._editor.entries[0].name==expected
        assert not dialog._editor.preview.isChecked()
        dialog._editor.close()
    else:
        assert 'legacy model' in dialog.summary.text()
        assert not dialog.paths['training'].text()
    dialog.reject();context._close();app.processEvents()
    assert parameters==block.parameters
for filename in ('project.json','trained_project.json'):
    with ProjectSession.open(root/case/filename,registry) as session:
        assert session.inspect()['validation']['valid']
        review(session,filename.startswith('trained'))
        dependencies=session.graph.blocks['segment'].project_assets()
        expected={p:hashlib.sha256((root/case/p).read_bytes()).hexdigest() for p in dependencies}
        target=out/filename/'project.json';target.parent.mkdir(parents=True)
        session.save(target)
    with ProjectSession.open(target,registry) as moved:
        assert moved.inspect()['validation']['valid']
        for name,digest in expected.items():
            assert hashlib.sha256((target.parent/name).read_bytes()).hexdigest()==digest
        review(moved,filename.startswith('trained'))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(tmp_path), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=70,
    )
    assert result.returncode == 0, result.stderr
