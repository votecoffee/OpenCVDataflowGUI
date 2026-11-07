import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/pattern_matching/visa_pcb2"


def test_visa_pattern_projects_landmarks_into_a_separate_photograph(tmp_path):
    pytest.importorskip("opencv_geometric_pattern_match.dataflow")
    root = shutil.copytree(EXAMPLE, tmp_path / "Moved PCB example")
    frames = json.loads((root / "landmarks.json").read_text(encoding="utf-8"))
    reference = np.asarray(frames[0]["mounting_holes"], dtype=float)
    relative = np.column_stack((reference - reference[0], np.ones(len(reference))))
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        assert graph.blocks["scene"].parameters["paths"] == [frame["path"] for frame in frames]
        for frame in frames:
            graph.run()
            matcher = graph.blocks["matcher"]
            assert matcher.state == "pass", matcher.error
            result = matcher.last_result.outputs
            assert result["count"] == 1
            [match] = result["matches"]["items"]
            matrix = np.asarray(match["pose"]["source_to_target"])
            projected = relative @ matrix[:2].T
            distances = np.linalg.norm(projected - frame["mounting_holes"], axis=1)
            assert distances.max() < 3
            assert result["overlay"].shape == (535, 702, 3)
            layers = {layer.name for layer in result[RESULT_GRAPHICS].layers}
            assert {"pose axes", "found edges"} <= layers
        assert "missing edges" in layers
        assert graph.sources_exhausted()
    finally:
        graph.close()
