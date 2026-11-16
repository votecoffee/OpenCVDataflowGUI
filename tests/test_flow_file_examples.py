"""Saved .flo round trips and measured computed-vs-analytic motion, offscreen/headless."""

import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/flow_file_roundtrip"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_flow_roundtrip_and_negative_associations(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.save(tmp_path / "program.json")
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        warp = graph.blocks["warp"]
        if case in ("direction_mismatch", "wrong_image"):
            assert warp.state == "error" and warp.error_kind == "invalid", warp.error
            assert not warp.current_emissions
        else:
            assert warp.state == "pass", warp.error
            result = warp.last_result
            if case != "computed_motion":
                assert result.outputs["report"]["valid_pixels"] == TRUTH["exact_covered_pixels"]
                expected = cv2.imread(str(BASE / "assets/target.png"), cv2.IMREAD_GRAYSCALE)
                valid = result.outputs["coverage"] != 0
                np.testing.assert_array_equal(result.outputs["image"][valid], expected[valid])
            if case != "bare_file":
                assert result.outputs["report"]["file_association"]["checked_images"] == [
                    "source",
                    "target",
                ]
        if case != "bare_file":
            save, reload = graph.blocks["save"], graph.blocks["reload"]
            assert save.state == reload.state == "pass", (save.error, reload.error)
            assert reload.last_result.outputs["info"]["images_verified"]
            output = Path(save.last_result.outputs["path"])
            assert output.parent == tmp_path / "output"
            assert (
                output.read_bytes()
                == (
                    BASE / "assets" / ("computed.flo" if case == "computed_motion" else "exact.flo")
                ).read_bytes()
            )
            # A second activation writes a new pair rather than overwriting.
            graph.run()
            assert graph.blocks["save"].last_result.outputs["path"] != str(output)
        assert bool(graph.blocks["load"].last_result.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_fixed_motion_references_and_regeneration_preserve_every_asset():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    source = cv2.imread(str(BASE / "assets/source.png"), 0)
    target = cv2.imread(str(BASE / "assets/target.png"), 0)
    actual = cv2.calcOpticalFlowFarneback(source, target, None, 0.5, 3, 25, 5, 7, 1.5, 0)
    native = cv2.readOpticalFlow(str(BASE / "assets/computed.flo"))
    np.testing.assert_array_equal(actual, native)
    errors = np.linalg.norm(actual[16:-16, 16:-16] - np.array([3, -2]), axis=2)
    assert np.median(errors) < 0.01 and np.percentile(errors, 95) < 0.02
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
