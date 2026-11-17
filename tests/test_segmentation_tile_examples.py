"""Native CPU fixtures, independent full-frame labels, seams and portable project replay."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/segmentation_tile_fusion"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("name", list(TRUTH["cases"]))
@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_native_projects_match_independent_pixels_and_keep_current_failures(name, run_profile):
    graph = Graph.load(BASE / (name + ".json"), builtin_registry())
    try:
        graph.set_run_profile(run_profile)
        graph.run()
        assert all(block.state != "error" for block in graph.blocks.values()), {
            b.id: b.error for b in graph.blocks.values() if b.state == "error"
        }
        result = graph.blocks["fuse"].last_result
        assert result is not None
        assert result.passed == TRUTH["cases"][name].get("passed", True)
        image = cv2.imread(str(BASE / "assets" / ("black.png" if name == "empty" else "parts.png")))
        np.testing.assert_array_equal(result.outputs["image"], image)
        truth = (
            cv2.imread(str(BASE / "assets/truth_labels_plus_one.png"), cv2.IMREAD_GRAYSCALE).astype(
                np.int32
            )
            - 1
        )
        if name == "empty":
            truth[:] = 0
        full = graph.blocks["decode_full"].last_result
        np.testing.assert_array_equal(full.outputs["labels"], truth)
        if name == "selected_roi":
            roi = np.zeros(truth.shape, bool)
            roi[4:16, 20:76] = True
            truth[~roi] = -1
        if name == "coverage_failure":
            truth[:, 32:48] = -1
            truth[:, 80:] = -1
        np.testing.assert_array_equal(result.outputs["labels"], truth)
        selected = truth > 0
        np.testing.assert_array_equal(result.outputs["mask"], selected.astype(np.uint8) * 255)
        assert result.outputs["pixel_count"] == int(selected.sum())
        # The red object stays one component across x32/x64 tile seams.
        if name not in ("coverage_failure", "empty", "selected_roi"):
            assert cv2.connectedComponents((truth == 1).astype(np.uint8))[0] == 2
            assert np.all(result.outputs["mask"][6:14, 20:76] == 255)
        assert result.outputs["report"]["confidence_semantics"] == (
            "weighted_vote_fraction" if name == "label_vote" else "winning_probability"
        )
        if name == "failed_tile":
            assert any(r["code"] == "segmentation.tile_failed" for r in result.fail_reasons)
        if name == "coverage_failure":
            assert result.outputs["report"]["coverage_fraction"] == pytest.approx(2 / 3)
        assert bool(result.drawings) == (run_profile == "debug")
        film = graph.blocks["filmstrip"].last_result.outputs["image"]
        np.testing.assert_array_equal(film, graph.blocks["overlay"].last_result.outputs["image"])
        if run_profile == "debug":
            assert np.any(film != image)
        for packet in graph.blocks["join"].last_result.outputs["group"].items:
            assert packet.payload["ticket"] == result.payloads["image"]["ticket"]
    finally:
        graph.close()


def test_new_cpu_engine_matches_classic_labels():
    graph = Graph.load(BASE / "probability_mean.json", builtin_registry())
    try:
        for block in graph.blocks.values():
            if block.type_id == "ml.dnn":
                block.parameters["engine"] = "new"
        graph.run()
        assert not any(b.state == "error" for b in graph.blocks.values()), {
            b.id: b.error for b in graph.blocks.values() if b.state == "error"
        }
        np.testing.assert_array_equal(
            graph.blocks["fuse"].last_result.outputs["labels"],
            graph.blocks["decode_full"].last_result.outputs["labels"],
        )
    finally:
        graph.close()


def test_assets_regenerate_exactly_and_relocated_json_xml_replay(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "tile_segmentation_fixture", BASE / "build_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path / "regenerated"
    module.HERE.mkdir()
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert (BASE / "assets" / name).read_bytes() == (module.HERE / "assets" / name).read_bytes()
    for path in BASE.glob("*.json"):
        assert json.loads(path.read_bytes()) == json.loads((module.HERE / path.name).read_bytes())
    graph = Graph.load(BASE / "probability_mean.json", builtin_registry())
    try:
        saved = tmp_path / "portable/program.json"
        saved.parent.mkdir()
        graph.save(saved)
        export_project_xml(graph.to_dict(), tmp_path / "program.xml")
        for candidate in (
            Graph.load(saved, builtin_registry()),
            Graph.from_dict(
                read_project_xml(tmp_path / "program.xml"),
                builtin_registry(),
                project_dir=saved.parent,
            ),
        ):
            try:
                model_path = (
                    candidate.project_dir / candidate.blocks["infer_left"].parameters["model_path"]
                ).resolve()
                assert model_path == (BASE / "assets/pixel_color.onnx").resolve()
                candidate.run()
                assert candidate.blocks["fuse"].last_result.passed
                assert candidate.blocks["fuse"].last_result.outputs["pixel_count"] == 788
            finally:
                candidate.close()
    finally:
        graph.close()
    (module.HERE / "assets/pixel_color.onnx").write_bytes(b"preserve existing model")
    with pytest.raises(FileExistsError):
        module.build()
