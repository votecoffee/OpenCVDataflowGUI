"""Photo provenance, real showcase outcomes and retained synthetic references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.validation import run_set

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "examples/demos"
NAMES = (
    "demo",
    "starter_inspection",
    "regions",
    "object_collections",
    "overlay_result_graphics",
    "image_set_workspace",
    "validation_images",
    "sequence_memory",
    "rgbd_cloud",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.mark.parametrize("name", NAMES)
def test_demo_sources_are_real_and_credited_and_original_fixture_remains(name):
    path = DEMOS / name / (name + ".json")
    definition = json.loads(path.read_text())
    sources = [b for b in definition["blocks"] if b["type"].startswith("input.")]
    assert sources and all(
        b["type"] in ("input.file", "input.depth_file", "input.video_file") for b in sources
    )
    assert any(
        b["type"] == "layout.comment" and "CC BY 4.0" in b.get("description", "")
        for b in definition["blocks"]
    )
    for source in sources:
        p = source["parameters"]
        assert p["path_base"] == "project"
        assert (path.parent / p["path"]).is_file()
    fixture = ROOT / "examples/assets/synthetic_graphs" / (name + ".json")
    assert fixture.is_file()
    assert any(
        b["type"] in ("input.demo", "input.sequence_demo", "input.stereo_demo", "input.file")
        for b in json.loads(fixture.read_text())["blocks"]
    )


def test_photo_derivatives_and_upstream_inputs_match_recorded_hashes():
    folder = ROOT / "examples/assets/showcase"
    manifest = json.loads((folder / "sources.json").read_text())
    assert len(manifest["images"]) == 7
    for record in manifest["images"]:
        assert record["license"] == "CC-BY-4.0"
        assert digest(ROOT / record["source"]) == record["source_sha256"]
        assert digest(folder / record["file"]) == record["sha256"]
        image = cv2.imread(str(folder / record["file"]))
        assert image.shape == (320, 480, 3)
    for value in manifest["datasets"].values():
        assert value["publisher"].startswith("https://") and "CC BY 4.0" in value["credit"]


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name", NAMES)
def test_all_real_demo_graphs_run_and_expose_expected_results(name, profile):
    graph = Graph.load(DEMOS / name / (name + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        for _ in range(8 if name == "sequence_memory" else 1):
            graph.run()
            assert all(
                b.state in ("pass", "waiting", "inactive") for b in graph.processing_blocks()
            ), [(b.id, b.state, b.error) for b in graph.processing_blocks()]
        counts = {
            b.id: b.last_result.outputs["count"]
            for b in graph.processing_blocks()
            if b.last_result and "count" in b.last_result.outputs
        }
        if name == "demo":
            assert counts["demo_contours"] == 1
        if name == "starter_inspection":
            assert counts["starter_components"] == 1
        if name == "regions":
            assert (counts["roi_left"], counts["roi_right"]) == (2, 1)
        if name == "object_collections":
            assert counts["objects_blobs"] == 4 and counts["objects_blob_sort"] == 2
        if name == "sequence_memory":
            assert graph.blocks["sequence_source"].last_result.outputs["info"]["frame_index"] == 7
        if name == "rgbd_cloud":
            assert counts["project"] == 23921 and counts["sample"] > 10000
    finally:
        graph.close()


def test_real_rgbd_cloud_matches_publisher_depth_scale_intrinsics_and_color():
    graph = Graph.load(DEMOS / "rgbd_cloud/rgbd_cloud.json", builtin_registry())
    try:
        graph.run()
        cloud = graph.blocks["project"].last_result.outputs["cloud"]
        assert cloud.units == "m" and cloud.source_size == (720, 540)
        points, pixels, colors = cloud.points[::137], cloud.pixels[::137], cloud.colors[::137]
        folder = ROOT / "examples/Simple Plugins/3d_locate/real_tless/source/test_primesense/000001"
        camera = json.loads((folder / "scene_camera.json").read_text())["220"]
        depth = cv2.imread(str(folder / "depth/000220.png"), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(str(folder / "rgb/000220.png"))
        x, y = pixels.astype(int).T
        z = depth[y, x].astype(float) * camera["depth_scale"] / 1000
        k = camera["cam_K"]
        expected = np.column_stack(((x - k[2]) * z / k[0], (y - k[5]) * z / k[4], z))
        np.testing.assert_allclose(points, expected, atol=1e-6)
        np.testing.assert_equal(colors, image[y, x, ::-1])
    finally:
        graph.close()


@pytest.mark.parametrize("name", ["validation_images", "image_set_workspace"])
def test_real_occupancy_validation_scores_separate_empty_and_loaded_captures(name):
    registry = builtin_registry()
    graph = Graph.load(DEMOS / name / (name + ".json"), registry)
    try:
        report = run_set(graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir)
        assert report["summary"]["score_percent"] == 100
        assert report["summary"]["scored"] == 4 and report["summary"]["unknown"] == 1
        assert report["summary"]["execution_errors"] == 0
        assert [r["actual"] for r in report["cases"]] == [True, True, False, False, False]
    finally:
        graph.close()


@pytest.mark.parametrize("name", ["lab_color_match", "lab_color_classification"])
def test_real_color_applications_report_raw_photographic_comparisons(name):
    graph = Graph.load(ROOT / f"examples/applications/{name}/{name}.json", builtin_registry())
    try:
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        block = graph.blocks["compare" if name == "lab_color_match" else "classify"]
        assert block.last_result.outputs["delta_e"] == pytest.approx(5.352966, abs=1e-5)
        if name == "lab_color_classification":
            assert block.last_result.outputs["label"] == "blue PCB substrate"
    finally:
        graph.close()


def test_industrial_photo_template_is_disabled_and_offline_replay_has_no_controller_blocks():
    folder = ROOT / "examples/industrial integration/controllogix_trigger_and_writeback"
    registry = builtin_registry()
    template = Graph.load(folder / "controllogix_trigger_and_writeback.json", registry)
    try:
        assert template.blocks["read"].parameters["enabled"] is False
        assert template.blocks["write"].parameters["write_enabled"] is False
        assert template.blocks["read"].parameters["route"] == ""
        template.run()
        assert all(b.state in ("pass", "waiting", "inactive") for b in template.processing_blocks())
        assert template.blocks["capture"].last_result is None
    finally:
        template.close()
    replay = Graph.load(folder / "offline_photo_replay.json", registry)
    try:
        assert not any("logix" in b.type_id or "fanuc" in b.type_id for b in replay.blocks.values())
        replay.run()
        assert all(b.state == "pass" for b in replay.processing_blocks())
        assert replay.blocks["count"].last_result.outputs["count"] == 1
    finally:
        replay.close()
