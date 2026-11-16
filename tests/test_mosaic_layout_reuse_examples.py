"""Saved source-ID projects checked against independent integer-grid truth."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import tiled_mosaic
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/mosaic_layout_reuse"
TRUTH = json.loads((HERE / "truth.json").read_bytes())
IDS = ("left", "middle", "right")


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_reuse_projects_preserve_fixed_canvas_and_current_source_context(
    case, profile, monkeypatch
):
    graph = Graph.load(HERE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    expected = TRUTH["cases"][case]
    if case != "native_registration":
        monkeypatch.setattr(
            tiled_mosaic, "register_scan", lambda *a: pytest.fail("Frozen replay fitted")
        )
    try:
        for _ in range(expected["runs"]):
            graph.run()
        block = graph.blocks["render"]
        assert block.state == expected["expected_state"], block.error
        result = block.last_result
        if expected["expected_state"] == "error":
            assert block.error_kind == "invalid" and result is None
            assert not any(block.current_emissions.values())
            return
        assert result is not None
        tile = result.outputs["tile"]
        layout = graph.blocks["layout"].last_result.outputs["layout"]
        assert tile["layout_id"] == layout["layout_id"]
        spatial = result.payloads["image"]["spatial"]
        assert spatial["root_id"] == "mosaic:" + layout["layout_id"]
        assert spatial["root_size"] == layout["image_size"]
        order = expected["order"]
        assert tile["source_matching"]["missing_source_ids"] == [
            IDS[i] for i in range(3) if i not in order
        ]
        for record in tile["views"]:
            i = record["index"]
            assert record["source_id"] == IDS[i]
            assert record["input_index"] == order.index(i)
            assert record["source_ticket"]["value"] == IDS[i] + ".png"
            np.testing.assert_allclose(
                np.asarray(record["source_to_tile"])[0:2, 2],
                np.asarray(record["source_to_mosaic"])[0:2, 2] - tile["origin"],
            )
        if case == "native_registration":
            assert tile["source_matching"]["complete"]
            assert layout["source_identity"]["mode"] == "explicit"
            return
        output = np.zeros((128, 480), np.uint8)
        counts = np.zeros(output.shape, np.uint8)
        for i in sorted(order):
            image = cv2.imread(str(HERE / "assets" / (IDS[i] + ".png")), cv2.IMREAD_GRAYSCALE)
            area = np.s_[:, 144 * i : 144 * i + 192]
            mask = np.ones(image.shape, bool)
            if case == "multiband_tile":
                mask = graph.blocks["selection"].last_result.outputs["mask"] != 0
            use = mask & (counts[area] == 0)
            output[area][use] = image[use]
            counts[area] += mask
        x, y = tile["origin"]
        w, h = tile["image_size"]
        area = np.s_[y : y + h, x : x + w]
        if case != "multiband_tile":
            np.testing.assert_array_equal(result.outputs["image"], output[area])
        np.testing.assert_array_equal(
            result.outputs["coverage"], (counts[area] > 0).astype(np.uint8) * 255
        )
        np.testing.assert_array_equal(
            result.outputs["overlap"], (counts[area] > 1).astype(np.uint8) * 255
        )
        np.testing.assert_array_equal(spatial["to_root"], [[1, 0, x], [0, 1, y], [0, 0, 1]])
        if expected["expected_state"] == "fail":
            assert result.fail_reasons[0]["code"] == "mosaic_sources_missing"
    finally:
        graph.close()


def test_regeneration_reproduces_original_images_layout_and_every_project(tmp_path):
    spec = importlib.util.spec_from_file_location("mosaic_reuse_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
    assert (tmp_path / "assets/calibration.json").read_bytes() == (
        HERE / "assets/calibration.json"
    ).read_bytes()
