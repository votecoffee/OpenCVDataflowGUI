"""Pinned suction/jaw native teaching with explicit calibration and failed real cases."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/bin_picking/training_workflow"


def test_bin_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case,count", [("suction_good", 10), ("suction_real13", 0),
                                        ("jaw_clear", 1), ("jaw_real13", 0)])
def test_native_setup_edit_preview_stage_apply_cancel_and_save_as(tmp_path, case, count):
    program = """
import runpy,sys,json
from pathlib import Path
root,out,case,count=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
ns=runpy.run_path(str(root/'check_training.py'))
import opencv_simple_bin_picking as package
assert '.whl' in package.__file__
ns['verify'](root,out,cases=(case,))
record=json.loads((out/'validation_training.json').read_bytes())[0]
assert record['eligible_picks']==count
assert record['staged_decisions_agree_within_1e_12'] and record['runtime_clouds_exact']
assert record['cancel_preserves_active'] and record['stage_preserves_active_until_apply']
assert record['exact_recipe_save_as'] and record['retained_reference_reopens']
assert record['settings_exact'] and record['calibration_exact']
assert not record['robot_execution']
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path / "native"), case, str(count)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
