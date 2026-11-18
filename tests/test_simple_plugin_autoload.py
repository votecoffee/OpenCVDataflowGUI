"""Open actual Simple Plugin projects through the default registry, without plugin flags."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_simple_integration
from opencv_dataflow.simple_plugin_bundles import BUNDLE_DIRECTORY, BUNDLES, prepare_bundle

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/Simple Plugins"


def child(program, *arguments, timeout=60):
    result = subprocess.run(
        [sys.executable, "-c", program, *map(str, arguments)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_shipped_integrations_are_pinned_and_retain_their_licenses():
    assert len(BUNDLES) == 13
    for package, (name, digest) in BUNDLES.items():
        path = BUNDLE_DIRECTORY / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with zipfile.ZipFile(path) as archive:
            assert package + "/__init__.py" in archive.namelist()
            assert any("/licenses/" in n and n.endswith("LICENSE") for n in archive.namelist())


@pytest.mark.parametrize("package", tuple(BUNDLES))
def test_missing_or_changed_bundle_fails_before_registration(monkeypatch, tmp_path, package):
    from opencv_dataflow import simple_plugin_bundles

    monkeypatch.setattr(simple_plugin_bundles, "BUNDLE_DIRECTORY", tmp_path)
    before = builtin_registry().types.copy()
    with pytest.raises(ValueError, match="missing or oversized"):
        prepare_bundle(package)
    (tmp_path / BUNDLES[package][0]).write_bytes(b"changed code")
    with pytest.raises(ValueError, match="checksum differs"):
        prepare_bundle(package)
    assert builtin_registry().types == before


def test_failed_provider_registration_is_atomic(monkeypatch):
    from opencv_dataflow import plugins
    from opencv_dataflow.core import Block

    class Uncommitted(Block):
        type_id = "test.uncommitted"

    def register(target):
        target.register(Uncommitted)
        raise RuntimeError("broken integration")

    monkeypatch.setattr(
        plugins.importlib, "import_module", lambda name: SimpleNamespace(register=register)
    )
    registry = builtin_registry()
    before = registry.types.copy()
    with pytest.raises(RuntimeError, match="broken integration"):
        load_simple_integration(registry, "3d.simple_laser_profile")
    assert registry.types == before


def test_all_optional_ids_are_discoverable_without_importing_their_packages():
    child("""
import sys
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import SIMPLE_EXTRA_PROVIDERS
from opencv_dataflow.toolbox import optional_simple_entries
registry=builtin_registry()
assert SIMPLE_EXTRA_PROVIDERS.keys() <= set(registry.available_type_ids())
entries={entry.type_id:entry for entry in optional_simple_entries(registry)}
assert SIMPLE_EXTRA_PROVIDERS.keys() <= entries.keys()
assert all(entries[t].path[0]=='Simple Plugins' for t in SIMPLE_EXTRA_PROVIDERS)
assert not any(name.startswith(('opencv_simple_', 'opencv_defect_', 'opencv_geometric_'))
               for name in sys.modules)
"""
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("reverse", (False, True))
def test_every_public_project_opens_without_plugin_loading(tmp_path, reverse):
    projects = []
    for path in EXAMPLES.rglob("*.json"):
        if ".local" in path.parts:
            continue
        data = json.loads(path.read_bytes())
        if isinstance(data, dict) and isinstance(data.get("blocks"), list):
            projects.append(str(path))
    assert len(projects) >= 280
    listing = tmp_path / "projects.json"
    listing.write_text(json.dumps(sorted(projects, reverse=reverse)), encoding="utf-8")
    child("""
import json,socket,subprocess,sys
from pathlib import Path
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
def forbidden(*args,**kwargs):raise AssertionError('Loading must not start a service or connection')
subprocess.Popen=forbidden
socket.socket.connect=forbidden
registry=builtin_registry()
for filename in json.loads(Path(sys.argv[1]).read_bytes()):
    with ProjectSession.open(filename,registry) as session:
        assert session.inspect()['validation']['valid'],filename
assert not any(name in sys.modules for name in ('torch','onnxruntime','timm'))
""", listing)


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "relative",
    (
        "bin_picking/dev/known_part_jaw/clear/project.json",
        "bin_picking/dev/known_part_jaw/obstacle/project.json",
        "bin_picking/dev/known_part_jaw/locator_chain/cad/project.json",
        "bin_picking/bin_to_grid/motion_review/baseline/project.json",
        "surface_enhance/inspection/connected_material.json",
        "surface_enhance/inspection/missing.json",
        "fringe_scan/inspection/connected_missing.json",
        "deflectometry/inspection/connected_change.json",
        "laser_profile/inspection/calibrated_step.json",
        "laser_profile/inspection/real_wli1.json",
        "pattern_matching/visa_pcb2/project.json",
        "pixel_segmentation/pcb_laminate/project.json",
        "path_width/calibrated_bead/project.json",
        "reference_inspect/armbench_tote/project.json",
        "3d_measure/dev/flatness/generated/good/project.json",
    ),
)
def test_representative_offline_projects_execute_without_plugin_flags(relative):
    child("""
import sys
from pathlib import Path
from opencv_dataflow.assistant_api import ProjectSession
with ProjectSession.open(sys.argv[1]) as session:
    for profile in ('debug','lean'):
        session.graph.set_run_profile(profile)
        session.run()
        for block in session.graph.blocks.values():
            assert block.state!='error',(sys.argv[1],block.id,block.error)
        if 'known_part_jaw' in Path(sys.argv[1]).parts:
            expected={'clear':'pass','obstacle':'fail'}.get(Path(sys.argv[1]).parent.name)
            if expected:assert session.graph.blocks['jaw'].state==expected
""", EXAMPLES / relative, timeout=90)


@pytest.mark.local_examples
def test_recorded_rgbd_source_runs_without_starting_a_pose_worker():
    child("""
import sys
from opencv_dataflow.assistant_api import ProjectSession
root=sys.argv[1]
with ProjectSession.open(root) as session:
    source=session.graph.blocks['capture']
    source.execute({})
    assert source.state=='pass',source.error
    assert source.last_result.outputs['image'].ndim==3
    assert source.last_result.outputs['depth'].ndim==2
    assert 'onnxruntime' not in sys.modules and 'torch' not in sys.modules
""", EXAMPLES / "foundation_pose/held_out_frame13/run_project.json")


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "relative",
    (
        "bin_picking/dev/known_part_jaw/clear/project.json",
        "foundation_pose/held_out_frame13/run_project.json",
        "surface_enhance/inspection/connected_material.json",
        "fringe_scan/inspection/connected_missing.json",
        "deflectometry/inspection/connected_change.json",
    ),
)
def test_recorded_sources_save_as_and_reopen_without_plugin_flags(tmp_path, relative):
    child("""
import hashlib,sys
from pathlib import Path
from opencv_dataflow.assistant_api import ProjectSession
source,target=map(Path,sys.argv[1:])
with ProjectSession.open(source) as session:
    fields={b.id:b.parameters.copy() for b in session.graph.blocks.values()
            if b.type_id.startswith('input.') and b.project_assets()}
    assets={str(p):hashlib.sha256((source.parent/p).read_bytes()).hexdigest()
            for b in session.graph.blocks.values() for p in b.project_assets()}
    assert fields and assets
    session.save(target)
with ProjectSession.open(target) as moved:
    assert moved.inspect()['validation']['valid']
    for name,parameters in fields.items():assert moved.graph.blocks[name].parameters==parameters
    for name,digest in assets.items():
        assert hashlib.sha256((target.parent/name).read_bytes()).hexdigest()==digest,name
""", EXAMPLES / relative, tmp_path / "project.json")


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "relative",
    (
        "surface_enhance/inspection/connected_material.json",
        "bin_picking/dev/known_part_jaw/clear/project.json",
    ),
)
def test_offscreen_editor_opens_and_runs_new_providers(relative):
    child("""
import sys,time
from PySide6.QtWidgets import QApplication
from opencv_dataflow.app import MainWindow
app=QApplication([])
window=MainWindow()
warnings=[]
window.warn=warnings.append
try:
    window.open_path(sys.argv[1])
    window.start_run(False)
    deadline=time.monotonic()+30
    while window.runner is not None and time.monotonic()<deadline:
        app.processEvents()
        # Release the Python GIL as the real Qt event loop does between callbacks.
        time.sleep(0.01)
    assert window.runner is None
    assert not warnings,warnings
    for block in window.graph.blocks.values():
        assert block.state!='error',(block.id,block.error)
finally:
    window.dirty=False
    window.close()
    app.processEvents()
""", EXAMPLES / relative)


def test_project_cannot_select_an_executable_plugin(tmp_path):
    sentinel = tmp_path / "executed"
    (tmp_path / "plugin.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n", encoding="utf-8"
    )
    child("""
import sys
from pathlib import Path
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
data={'schema_version':1,'plugins':['plugin.py'],
      'blocks':[{'type':'unknown.external','id':'unknown','parameters':{}}]}
try:Graph.from_dict(data,builtin_registry(),project_dir=Path(sys.argv[1]))
except ValueError as error:assert 'Load its plugin first' in str(error)
else:raise AssertionError('Unknown code must remain unavailable')
assert not (Path(sys.argv[1])/'executed').exists()
""", tmp_path)
