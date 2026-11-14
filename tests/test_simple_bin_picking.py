"""Saved-project replay and one-publication teaching for the optional suction tool."""

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

pytest.importorskip("opencv_simple_bin_picking.dataflow")
from opencv_simple_bin_picking.persistence import load_recipe
from opencv_simple_bin_picking.qt_training import BinPickingDialog

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/bin_picking"
CASES = ["generated/" + name for name in ("good", "empty", "missing", "holes", "narrow_bin")]
CASES += [f"real_tless/frame{frame:06}" for frame in (3, 13, 24)]


@pytest.mark.local_examples
@pytest.mark.parametrize("relative", CASES)
def test_saved_projects_replay_numeric_evidence_and_native_cloud(relative):
    registry = builtin_registry()
    assert "3d.simple_bin_picking" not in registry.types
    with ProjectSession.open(ROOT / relative / "project.json", registry) as session:
        assert session.inspect()["validation"]["valid"]
        session.run()
        block = session.graph.blocks["picker"]
        expected = json.loads((ROOT / relative / "expected.json").read_bytes())
        assert block.state == ("pass" if relative == "generated/good" else "fail"), block.error
        result = block.last_result
        assert result.outputs["report"] == expected
        assert block.preview_output == "graphics"
        cloud = result.outputs["graphics"]
        assert isinstance(cloud, PointCloud) and cloud.colors.shape == cloud.points.shape
        if not expected["eligible_picks"]:
            assert result.outputs["tcp"] is result.outputs["pregrasp"] is NO_VALUE
        else:
            assert result.outputs["tcp"]["target_frame"] == expected["frame"]
        for host_off in (False, True):
            block.parameters["render_graphics"] = host_off
            block.set_graphics_mode("off" if host_off else "inherit")
            with patch(
                "opencv_simple_bin_picking.dataflow.result_cloud", side_effect=AssertionError
            ):
                session.run()
            assert block.state in ("pass", "fail"), block.error
            assert block.last_result.outputs["graphics"] is NO_VALUE
            assert block.last_result.outputs["report"] == expected


@pytest.mark.local_examples
def test_example_hashes_and_raw_license_provenance():
    manifest = json.loads((ROOT / "sha256.json").read_bytes())
    assert not any("__pycache__" in p or p.endswith((".pyc", ".pyo")) for p in manifest)
    for relative, digest in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    source = ROOT / "real_tless/source"
    provenance = json.loads((source / "provenance.json").read_bytes())
    assert provenance["license"] == "CC-BY-4.0"
    for relative, digest in provenance["sha256"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == digest
    for path in ROOT.rglob("*.json"):
        assert "OpenCVSimpleTools" not in path.read_text(encoding="utf-8")


@pytest.mark.local_examples
def test_real_canonical_camera_preserves_first_returns_and_fixed_training():
    source = ROOT / "real_tless/source/test_primesense/000002"
    cameras = json.loads((source / "scene_camera.json").read_bytes())
    normalization = json.loads((ROOT / "real_tless/normalization.json").read_bytes())
    recipes = []
    for frame in (3, 13, 24):
        case = ROOT / f"real_tless/frame{frame:06}"
        raw = cv2.imread(str(source / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
        normalized = np.load(case / "assets/depth.npy", allow_pickle=False)
        dx, dy = normalization["pixel_shifts_xy"][str(frame)]
        y, x = np.indices(raw.shape)
        inside = (x + dx >= 0) & (x + dx < raw.shape[1]) & (y + dy >= 0) & (y + dy < raw.shape[0])
        expected = np.zeros_like(raw)
        expected[(y + dy)[inside], (x + dx)[inside]] = raw[inside]
        np.testing.assert_array_equal(normalized, expected)
        recipe = load_recipe(case / "assets/recipe.sbp")
        np.testing.assert_array_equal(
            recipe.camera.matrix, np.array(cameras["3"]["cam_K"]).reshape(3, 3)
        )
        assert recipe.depth_scale == cameras[str(frame)]["depth_scale"] == 0.1
        recipes.append((case / "assets/recipe.sbp").read_bytes())
    assert recipes[0] == recipes[1] == recipes[2]


@pytest.mark.local_examples
def test_save_as_keeps_sealed_recipe_bytes(tmp_path):
    original = ROOT / "generated/good"
    with ProjectSession.open(original / "project.json") as session:
        session.save(tmp_path / "project.json")
        path = session.graph.blocks["picker"].parameters["recipe_path"]
        assert (tmp_path / path).read_bytes() == (original / "assets/recipe.sbp").read_bytes()
        assert session.inspect()["validation"]["valid"]
        session.run()
        assert session.graph.blocks["picker"].state == "pass"


@pytest.mark.local_examples
def test_novel_block_teaches_from_host_connected_publication(app, tmp_path):
    recipe = load_recipe(ROOT / "generated/good/assets/recipe.sbp")

    class Source(Block):
        outputs = (Port("depth", "depth"), Port("camera", "data"))

        def process(self, inputs):
            return Result(
                {"depth": recipe.reference_depth.copy(), "camera": recipe.camera.record("mm")}
            )

    with ProjectSession(directory=tmp_path) as session:
        graph = session.graph
        source = graph.add(Source(block_id="source"))
        target = graph.add(session.registry.create("3d.simple_bin_picking", block_id="picker"))
        graph.connect(source.id, "depth", target.id, "depth")
        graph.connect(source.id, "camera", target.id, "camera")
        source.execute({})
        context = PluginDialogContext(graph, target)
        draft = target.parameters.copy()
        dialog = BinPickingDialog(None, draft, context)
        try:
            dialog.capture_reference()
            assert target.timing.count == 0
            assert dialog._reference_capture.publication_id
            dialog.fields["max_candidates"].setValue(64)
            dialog.preview_reference()
            deadline = time.monotonic() + 15
            while dialog._future is not None and time.monotonic() < deadline:
                app.processEvents()
                dialog.poll()
            assert dialog._future is None and dialog.session.trained
            dialog.stage()
            assert draft["recipe_path"] and target.parameters["recipe_path"] == ""
            restored = load_recipe(tmp_path / draft["recipe_path"], sha256=draft["sha256"])
            np.testing.assert_array_equal(restored.reference_depth, recipe.reference_depth)
        finally:
            dialog.done(0)
            context._close()
