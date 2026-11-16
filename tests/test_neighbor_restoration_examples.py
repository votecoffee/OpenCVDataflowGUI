"""Installed saved graph paths, independent pixels, private identity and regeneration."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/neighbor_restoration"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ("debug", "lean"))
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_examples_match_independent_pixels_and_protect_changed_parts(case, profile):
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        expected = TRUTH["cases"][case]
        block = graph.blocks["restore"]
        assert block.state == expected["state"], block.error
        result = block.last_result
        assert result is not None
        target = graph.blocks["frame1"].last_result
        source, output = target.outputs["image"], result.outputs["image"]
        restored = result.outputs["restored_mask"] != 0
        np.testing.assert_array_equal(output[~restored], source[~restored])
        assert result.payloads["image"]["ticket"] == target.payloads["image"]["ticket"]
        assert result.payloads["image"]["spatial"] == target.payloads["image"]["spatial"]
        assert all(b.state == "pass" for b in graph.processing_blocks() if b.id != "restore")
        if profile == "lean":
            assert (
                not result.drawings
                and not result.graphics
                and result.outputs["preview"] is NO_VALUE
            )
        if expected["unchanged"]:
            assert not restored.any() and result.outputs["info"]["state"] == "unchanged"
        else:
            assert restored.mean() > (0.25 if case == "highdepth" else 0.5)
            truth = graph.blocks["frame3"].last_result.outputs["image"]
            np.testing.assert_array_equal(output[restored], truth[restored])
            assert graph.blocks["verify"].last_result.outputs["value"] == 0
            assert graph.blocks["before"].last_result.outputs["value"] > 0
            before = np.mean((source.astype(float) - truth.astype(float)) ** 2)
            after = np.mean((output.astype(float) - truth.astype(float)) ** 2)
            assert after < before * (0.8 if case in ("highdepth", "moving_parts") else 0.4)
        if case == "moving_parts":
            for x, y, w, h in TRUTH["moving_rectangles_xywh"]:
                np.testing.assert_array_equal(
                    output[y : y + h, x : x + w], source[y : y + h, x : x + w]
                )
                assert not restored[y : y + h, x : x + w].any()
        if case == "highdepth":
            assert output.dtype == np.uint16 and output.color_format == "bgr"
            support = graph.blocks["support1"].last_result.outputs["mask"]
            np.testing.assert_array_equal(result.outputs["valid_mask"], support)
            assert not restored[:18].any() and not restored[:, :18].any()
        # No cursor, cache or image mutations survive activation/reset/reopen.
        block.reset_runtime()
        again = block.execute(
            {
                "frames": graph.blocks["frames"].last_result.outputs["group"],
                **(
                    {
                        "coverage": graph.blocks["coverage"].last_result.outputs["group"],
                        "region": graph.blocks["selection"].last_result.outputs["region"],
                    }
                    if case == "highdepth"
                    else {}
                ),
            }
        )
        np.testing.assert_array_equal(again.outputs["image"], output)
    finally:
        graph.close()


def test_public_assets_and_projects_regenerate_exactly_without_customer_files(tmp_path):
    spec = importlib.util.spec_from_file_location("neighbor_example", HERE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (HERE / path.relative_to(tmp_path)).read_bytes(), path.name
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == expected
