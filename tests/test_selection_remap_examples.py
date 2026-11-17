"""Saved nonlinear selections: independent distortion, masks and RGB16 measurements."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import image_format
from opencv_dataflow.selection import rasterize

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/selection_remap"
pytestmark = pytest.mark.local_examples


def fixture_module():
    spec = importlib.util.spec_from_file_location("selection_fixture", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", ["lens", "coverage", "packed", "wide", "wrong_pair"])
def test_saved_lens_selection_matches_independent_support_and_measurements(profile, case):
    truth = json.loads((BASE / "truth.json").read_bytes())["cases"][case]
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["remap"]
        if truth["expected_error"]:
            assert block.state == "error" and "matching X and Y" in block.error
            assert not any(block.current_emissions.values())
            assert all(
                graph.blocks[k].last_result is None for k in ("measure", "statistics", "filmstrip")
            )
            return
        assert all(b.state == "pass" for b in graph.processing_blocks()), [
            (b.id, b.state, b.error) for b in graph.processing_blocks()
        ]
        result = block.last_result
        mask, report = result.outputs["mask"], result.outputs["report"]
        assert hashlib.sha256(mask).hexdigest() == truth["mask_sha256"]
        assert report["selected_pixels"] == truth["selected_pixels"]
        assert report["valid_pixels"] == truth["valid_pixels"]
        np.testing.assert_array_equal(rasterize(result.outputs["region"], mask.shape), mask)
        assert result.outputs["image"].dtype == np.uint16
        assert image_format(result.outputs["image"]) == "rgb"
        measured = graph.blocks["measure"].last_result.outputs
        assert measured["area"] == truth["selected_pixels"]
        np.testing.assert_allclose(measured["centroid"], [truth["centroid"]], rtol=0, atol=1e-10)
        summary = graph.blocks["statistics"].last_result.outputs["summary"]
        assert [c["channel"] for c in summary["channels"]] == ["R", "G", "B"]
        np.testing.assert_allclose(
            [c["mean"] for c in summary["channels"]],
            truth["channel_means"],
            rtol=0,
            atol=truth["mean_absolute_tolerance"],
        )
        source = graph.blocks["source"].last_result
        for name in ("image", "mask", "region", "coverage", "valid_mask", "report"):
            assert result.payloads[name]["ticket"] == source.outputs["ticket"]
            assert result.payloads[name]["spatial"] == result.payloads["image"]["spatial"]
        assert result.payloads["image"]["spatial"] != source.payloads["image"]["spatial"]
        if profile == "debug":
            assert result.drawings
            assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
        else:
            assert not result.drawings
    finally:
        graph.close()


def test_fixture_regeneration_is_exact_without_changing_source_files(tmp_path):
    module = fixture_module()
    names = ["truth.json", "assets/lens.json", "assets/ramp_rgb16.png"] + [
        f"{case}.json" for case in module.CASES
    ]
    before = {name: (BASE / name).read_bytes() for name in names}
    module.HERE = tmp_path
    module.build()
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
    assert {name: (BASE / name).read_bytes() for name in names} == before
    truth = json.loads(before["truth.json"])
    assert hashlib.sha256(before["assets/ramp_rgb16.png"]).hexdigest() == truth["source_sha256"]


def test_float_and_packed_maps_each_match_their_own_native_phase_precision():
    module = fixture_module()
    x, y = module.analytic_maps([160, 120], module.PROFILE["camera_matrix"])
    yy, xx = np.indices((120, 160))
    selected = (xx - 68) ** 2 + (yy - 56) ** 2 <= 36**2
    selected[46:66, 64:82] = False
    registry = builtin_registry()
    image = np.zeros((120, 160), np.float32)
    for packed in (False, True):
        block = registry.create("cv.remap_selection_packed" if packed else "cv.remap_selection")
        inputs = {"image": image, "mask": selected.astype(np.uint8)}
        if packed:
            pack = registry.create("cv.pack_remap_map")
            inputs.update(pack.execute({"map_x": x, "map_y": y}).outputs)
        else:
            inputs.update(map_x=x, map_y=y)
        result = block.execute(inputs)
        assert result is not None, block.error
        np.testing.assert_allclose(
            result.outputs["coverage"],
            module.four_taps(selected, x, y, packed=packed),
            rtol=0,
            atol=2e-7,
        )
