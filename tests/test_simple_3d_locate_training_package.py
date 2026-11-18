"""Portable native 3D teaching with an explicitly pinned wheel."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate/training_workflow"


def test_3d_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("real_positive", "real_no_match"))
def test_native_crop_undo_cancel_train_test_apply_and_save_as(tmp_path, case):
    program = """
import runpy,shutil,sys
from pathlib import Path
root,out,name=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
ns=runpy.run_path(str(root/'check_training.py'))
shutil.copytree(root/name,out/name,ignore=shutil.ignore_patterns('.local','__pycache__','staged_project.json','crop_preview.*','reviewed_fit.*'))
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import make_registry
import opencv_simple_3d_locate as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
record=ns['verify_case'](out,name,make_registry(plugins=[str(root/'plugin.py')]),app)
assert record['accepted_count']==(1 if name=='real_positive' else 0)
assert record['unchanged_training_arrays_exact'] and record['exact_recipe_save_as']
assert record['crop_undo_restores_cad'] and record['cancelled_job_not_staged']
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
