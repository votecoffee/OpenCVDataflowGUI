"""Image-dependent native instance masks, lossless replays and per-mask identity controls."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.selection import rasterize
from opencv_dataflow.tickets import ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/instance_tracking"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
BENCH = json.loads((BASE / "benchmark.json").read_bytes())


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_mask_and_identity_reports_match_independent_visible_truth(case, profile):
    evaluate = runpy.run_path(str(BASE / "benchmark.py"))["evaluate"]
    result = evaluate(case, profile)
    assert result == BENCH["cases"][case]
    for row in result["frames"]:
        if "matches" in row:
            assert all(match["iou"] == 1 for match in row["matches"])
    if case == "crossing":
        assert result["frames"][5]["observed"] == 1 and result["frames"][5]["predicted"] == 1
    if case == "transparent_overlap":
        assert any(row["overlap_pixels"] for row in result["frames"])
        assert result["frames"][5]["observed"] == 0 and result["frames"][5]["ambiguous"] == 2
    if case == "expiry":
        assert result["summary"]["id_switches"] == 1


def test_native_channel_masks_and_all_lossless_frames_match_independent_truth():
    assert runpy.run_path(str(BASE / "benchmark.py"))["native_reference"]() == {
        "native_frames": 38,
        "maximum_mask_difference": 0.0,
    }


def test_asset_pins_exact_regeneration_and_no_overwrite(tmp_path):
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    build(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(tmp_path)).read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        build(tmp_path)


def test_current_tickets_high_depth_drawings_and_selectable_mask_are_preserved():
    graph = Graph.load(BASE / "high_depth.json", builtin_registry())
    try:
        for _ in range(3):
            graph.run()
            source = graph.blocks["source"].last_result
            result = graph.blocks["track"].last_result
            assert result is not None and result.outputs["image"].dtype == np.uint16
            np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
            assert ticket_key(result.payloads["tracks"]["ticket"]) == ticket_key(
                source.outputs["ticket"]
            )
            assert result.drawings and result.graphics
            selected = graph.blocks["selected"].last_result
            assert selected.outputs["found"] and selected.outputs["item"]["track_id"] == 1
            np.testing.assert_array_equal(
                selected.outputs["mask"],
                rasterize(result.outputs["tracks"]["items"][0]["mask_region"], (40, 64)),
            )
            assert graph.blocks["filmstrip"].last_result.outputs["image"].shape[:2] == (40, 64)
    finally:
        graph.close()


def test_save_as_xml_and_reopen_do_not_persist_live_tracks(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(BASE / "crossing.json", registry)
    try:
        graph.run()
        graph.run()
        assert graph.blocks["track"].memory["streams"]
        graph.save(tmp_path / "saved.json")
    finally:
        graph.close()
    graph = Graph.load(tmp_path / "saved.json", registry)
    try:
        assert not graph.blocks["track"].memory["streams"]
        graph.run()
        assert [i["id"] for i in graph.blocks["track"].last_result.outputs["tracks"]["items"]] == [
            1,
            2,
        ]
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
    finally:
        graph.close()
    graph = Graph.from_dict(
        read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path
    )
    try:
        graph.run()
        assert graph.blocks["track"].last_result.outputs["count"] == 2
    finally:
        graph.close()


def test_failed_inputs_do_not_advance_tracks_or_replace_last_image(monkeypatch):
    graph = Graph.load(BASE / "crossing.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["track"]
        previous = block.last_inputs["image"].copy()
        activations = next(iter(block.memory["streams"].values()))["activations"]
        block.input_failure_policies = {"image": "reject_failed"}
        original = graph.blocks["source"].process

        def failed(inputs):
            result = original(inputs)
            result.passed = False
            return result

        monkeypatch.setattr(graph.blocks["source"], "process", failed)
        graph.run()
        assert block.input_rejected and not block.current_emissions
        assert next(iter(block.memory["streams"].values()))["activations"] == activations
        np.testing.assert_array_equal(block.last_inputs["image"], previous)
    finally:
        graph.close()


@pytest.mark.parametrize("case,count", [("crossing", 9), ("expiry", 11)])
def test_cold_run_all_drains_native_video_frames_and_closes_models(case, count):
    script = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import MainWindow,Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);r=builtin_registry();g=Graph.load(Path(sys.argv[1])/(sys.argv[2]+'.json'),r)
try:
 worker=Runner(g,MainWindow.run_all_mode(SimpleNamespace(root_graph=g)),r);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(30000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==int(sys.argv[3])
 assert g.blocks['track'].last_result.outputs['count']==2
finally:g.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BASE), case, str(count)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
