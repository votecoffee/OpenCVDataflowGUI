"""Native context lifetime, worker startup and local CPU preference checks."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest
from PySide6.QtCore import QThread

from opencv_dataflow.app import LivePreview, Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.hog_detection import cpu_search
from opencv_dataflow.native_runtime import prepare_native_runtime

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/hog_detection"


def test_loading_is_inert_and_worker_construction_primes_on_caller(app, monkeypatch):
    queried = []
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: queried.append(QThread.currentThread()))
    graph = Graph.load(BASE / "multiscale.json", builtin_registry())
    try:
        assert not queried
        Runner(graph, "once")
        assert queried == [app.thread()]
        graph.prepare_native_execution()
        assert len(queried) == 1
        graph.invalidate()
        graph.prepare_native_execution()
        assert queried == [app.thread()] * 2
        LivePreview(graph.blocks["detect"], {}, 1, None)
        assert queried == [app.thread()] * 3
    finally:
        graph.close()


def test_nested_toolblock_and_parallel_graphs_are_primed_without_changing_preference(monkeypatch):
    leaf = SimpleNamespace(type_id="ml.hog_detect", requires_opencl_context=True)
    nested = SimpleNamespace(blocks={"hog": leaf})
    parallel = SimpleNamespace(type_id="flow.parallel_inspect", graph=nested)
    child = SimpleNamespace(blocks={"parallel": parallel})
    tool = SimpleNamespace(type_id="flow.toolblock", graph=child)
    root = SimpleNamespace(blocks={"tool": tool})
    queried = []
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: queried.append(True))
    monkeypatch.setattr(cv2.ocl, "setUseOpenCL", lambda value: pytest.fail("Preference changed"))
    prepare_native_runtime(root)
    prepare_native_runtime(root)
    prepare_native_runtime(nested)
    assert queried == [True]
    assert all(g._native_runtime_ready for g in (root, child, nested))
    plain = SimpleNamespace(blocks={"block": SimpleNamespace(type_id="test.plain")})
    prepare_native_runtime(plain)
    assert queried == [True] and not getattr(plain, "_native_runtime_ready", False)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_cpu_search_restores_calling_thread_preference_even_on_error(monkeypatch, enabled, raises):
    state, changes = [enabled], []
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: state[0])

    def set_preference(value):
        state[0] = value
        changes.append(value)

    monkeypatch.setattr(cv2.ocl, "setUseOpenCL", set_preference)
    try:
        with cpu_search():
            assert state[0] is False
            if raises:
                raise RuntimeError("controlled native failure")
    except RuntimeError:
        assert raises
    assert state[0] is enabled and changes == [False, enabled]


@pytest.mark.parametrize("mode", ["parallel", "preview"])
def test_cold_native_execution_finishes_and_exits_without_device_abort(mode):
    script = r"""
import json,sys
from pathlib import Path
import cv2
from PySide6.QtCore import QCoreApplication
from opencv_dataflow.app import LivePreview
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
base=Path(sys.argv[1]); app=QCoreApplication([]); registry=builtin_registry()
graph=Graph.load(base/'multiscale.json',registry)
try:
 if sys.argv[2]=='parallel':
  second=graph.add(registry.create('ml.hog_detect',block_id='second',parameters=graph.blocks['detect'].parameters))
  graph.connect('source','image',second.id,'image');graph.connect('model','model',second.id,'model')
  graph.set_branch_workers(2);graph.run()
  assert second.last_result.outputs['count']==2,second.error
  assert graph.blocks['detect'].last_result.outputs['count']==2
 else:
  block=graph.blocks['detect']
  inputs={'image':cv2.imread(str(base/'assets/scene.png')),'model':json.loads((base/'assets/detector.json').read_bytes())}
  worker=LivePreview(block,inputs,1,None);worker.start()
  assert worker.wait(25000),'preview timeout'
  assert not worker.error,worker.error
  assert block.last_result.outputs['count']==2,block.error
finally:graph.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE), mode],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
