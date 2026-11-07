import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/Simple Plugins/pattern_matching/asymmetric_brackets"
)


@pytest.mark.parametrize("filename", ["project.json", "wired_recipe.json"])
def test_public_pattern_example_matches_generated_poses_after_relocation(tmp_path, filename):
    pytest.importorskip("opencv_geometric_pattern_match.dataflow")
    root = shutil.copytree(EXAMPLE, tmp_path / "Moved inspection")
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    graph = Graph.load(root / filename, builtin_registry())
    try:
        assert graph.blocks["scene"].parameters["paths"] == [frame["path"] for frame in expected]
        for frame in expected:
            graph.run()
            block = graph.blocks["matcher"]
            assert block.state == "pass", block.error
            result = block.last_result.outputs
            records = result["matches"]["items"]
            assert result["count"] == len(records) == len(frame["poses"])
            unmatched = frame["poses"].copy()
            for record in records:
                matrix = np.asarray(record["pose"]["source_to_target"])
                point = matrix[:2, 2]
                distances = [np.linalg.norm(point - [pose["x"], pose["y"]]) for pose in unmatched]
                index = int(np.argmin(distances))
                pose = unmatched.pop(index)
                assert distances[index] < .5
                angle = np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0]))
                assert angle == pytest.approx(pose["angle_deg"], abs=.5)
                assert np.linalg.norm(matrix[0, :2]) == pytest.approx(pose["scale"], abs=.015)
            layers = {layer.name for layer in result[RESULT_GRAPHICS].layers}
            assert {"pose axes", "found edges"} <= layers
            if frame["path"].endswith("partial_tray.png"):
                assert "missing edges" in layers
                assert sum(record["coverage"] < .98 for record in records) == 1
        assert graph.sources_exhausted()
    finally:
        graph.close()
