"""Pinned PathWidth training workflows using the native inspector, without installation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/path_width/training_workflow"


def test_path_width_training_pins():
    manifest = json.loads((ROOT / "artifact_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        assert ".local" not in Path(name).parts and "__pycache__" not in name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("case", ("calibrated", "weld"))
def test_native_restored_path_alignment_preview_stage_and_save_as(tmp_path, case):
    program = """
import runpy,shutil,sys
from pathlib import Path
root,out,name=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
ns=runpy.run_path(str(root/'check_training.py'))
shutil.copytree(root/name,out/name,ignore=shutil.ignore_patterns('.local','__pycache__','preview','planned_guides.png','staged_project.json'))
from PySide6.QtWidgets import QApplication
from opencv_dataflow.assistant_api import make_registry
import opencv_simple_path_width as package
assert '.whl' in package.__file__
app=QApplication.instance() or QApplication([])
record=ns['verify_case'](out,name,make_registry(plugins=[str(root/'plugin.py')]),app)
assert record['exact_recipe_save_as'] and record['original_assets_preserved']
assert record['invalid_setup_and_stale_preview'] and record['linked_station_preview']
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(tmp_path), case],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
