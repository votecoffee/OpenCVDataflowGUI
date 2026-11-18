"""Pinned ReferenceInspect training workflow and portable assets."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/reference_inspect/dev/training_workflow"
)


def test_reference_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("generated", "visa_posed"))
def test_native_import_cancel_train_apply_reopen_and_save_as(tmp_path, case):
    program = """
import runpy,shutil,sys
from pathlib import Path
root,out,name=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
ns=runpy.run_path(str(root/'check_training.py'))
shutil.copytree(root/name,out/name,ignore=shutil.ignore_patterns('.local','__pycache__','review','staged_project.json','alignment_support.png'))
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import make_registry
import opencv_simple_reference_inspect as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
registry=make_registry(plugins=[str(root/'pattern_plugin.py'),str(root/'plugin.py')])
record=ns['verify_case'](out,name,registry,app)
assert record['cancelled_training_not_staged'] and record['exact_recipe_save_as']
assert record['trained_arrays_and_original_health_exact'] and record['original_assets_preserved']
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
