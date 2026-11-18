"""Portable dataset-diagnosis trainer checks using the explicit bundled wheel."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/image_classification/dev/training_workflow"
)


def test_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("profile", ("color_and_layout", "surface_texture"))
def test_restored_dataset_checks_and_exact_model_save_as(tmp_path, profile):
    program = """
import hashlib,sys,time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import ProjectSession,make_registry
from opencv_dataflow.plugin_dialog import PluginDialogContext
root,out,profile=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
registry=make_registry(plugins=[str(root/'plugin.py')])
import opencv_simple_classifier as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
for prefix in ('','retrained_'):
    path=root/'vegqual'/f'{prefix}{profile}.json'
    with ProjectSession.open(path,registry) as session:
        assert session.inspect()['validation']['valid']
        block=session.graph.blocks['classifier']
        context=PluginDialogContext(session.graph,block)
        parameters=block.parameters.copy()
        dialog=block.create_plugin_dialog(None,parameters,context)
        assert dialog.tabs.count()==4 and dialog.report.rowCount()==4
        if prefix:
            assert len(dialog.predictions)==20
        dialog.audit_button.click()
        deadline=time.monotonic()+30
        while dialog.future is not None:
            app.processEvents();time.sleep(.005)
            assert time.monotonic()<deadline,dialog.status.text()
        assert dialog._audit.ready
        assert [c.training for c in dialog._audit.classes]==[24]*4
        assert [c.validation for c in dialog._audit.classes]==[5]*4
        assert 'assigned' in dialog.review_counts.text()
        dialog.reject();context._close()
        assert parameters==block.parameters
        members=block.project_assets()
        expected={p:hashlib.sha256((path.parent/p).read_bytes()).hexdigest() for p in members}
        target=out/(prefix or 'original')/'project.json';target.parent.mkdir(parents=True)
        session.save(target)
    with ProjectSession.open(target,registry) as copied:
        assert copied.inspect()['validation']['valid']
        for name,digest in expected.items():
            assert hashlib.sha256((target.parent/name).read_bytes()).hexdigest()==digest
app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path), profile],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=75,
    )
    assert result.returncode == 0, result.stderr
