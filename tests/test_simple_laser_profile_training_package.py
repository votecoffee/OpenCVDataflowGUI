"""Pinned calibration fitting and separate reference/preview health through the inspector."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/laser_profile/dev/training_workflow"
)


def test_laser_training_artifact_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize(
    "case,count,passed",
    [
        ("calibrated_step", 480, True),
        ("calibrated_unavailable", 390, False),
        ("real_wli1", 72, False),
    ],
)
def test_native_calibration_fit_learn_stage_apply_and_exact_save_as(tmp_path, case, count, passed):
    program = """
import json,runpy,sys
from pathlib import Path
root,out,case,count,passed=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5]=='1'
ns=runpy.run_path(str(root/'check_training.py'))
import opencv_simple_laser_profile as package
assert '.whl' in package.__file__
ns['verify'](root,out,cases=(case,))
record=json.loads((out/'validation_training.json').read_bytes())[0]
assert record['runtime_count']==count and record['runtime_passed']==passed
assert record['reference_preview_separate'] and record['calibration_review_readonly']
assert record['stage_preserves_active_until_apply'] and record['cancel_preserves_active']
assert record['exact_recipe_save_as'] and record['original_reference_and_mask_exact']
assert record['metric']==(case!='real_wli1')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(ROOT),
            str(tmp_path / "native"),
            case,
            str(count),
            "1" if passed else "0",
        ],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
