"""Portable measurement training with an explicitly pinned optional wheel."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/3d_measure/dev/training_workflow"
)


def test_measurement_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("real13", "real24", "missing"))
def test_native_measurement_edit_undo_train_apply_and_save_as(tmp_path, case):
    program = """
import runpy,sys
from pathlib import Path
root,out,case=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
ns=runpy.run_path(str(root/'check_training.py'))
import opencv_simple_3d_measure as package
assert '.whl' in package.__file__
ns['verify'](root,out,cases=(case,))
import json
record=json.loads((out/'validation_training.json').read_bytes())[0]
assert record['passed']==(case!='missing')
assert record['unavailable']==(3 if case=='missing' else 0)
assert record['reference_xyz_exact'] and record['exact_recipe_save_as']
assert record['numeric_edit_undo_redo'] and record['stage_preserves_active_until_apply']
assert record['cancel_preserves_active'] and record['independent_native_flyaround']
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path / "native"), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
