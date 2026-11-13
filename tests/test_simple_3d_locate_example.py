import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


@pytest.mark.local_examples
@pytest.mark.parametrize("variant", ["", "cad_surface"])
def test_simple_3d_locate_example_roundtrip(variant):
    pytest.importorskip("opencv_simple_3d_locate.dataflow")
    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate" / variant
    if variant:
        import opencv_simple_3d_locate as tool

        if not hasattr(tool, "read_stl"):
            pytest.skip("CAD example requires Simple3DLocate 0.1.0.dev1 or later")
    truth = [np.asarray(item) for item in json.loads((root / "assets/truth.json").read_text())]
    graph = Graph.load(root / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["locate"]
        assert block.state == "pass", block.error
        assert block.last_result.outputs["count"] == 2
        assert block.last_result.outputs["overlay"].shape == (640, 960, 3)
        assert block.filmstrip_output == "overlay"
        cached = block._locator
        graph.run()
        assert block.state == "pass", block.error
        assert block._locator is cached
        assert block.last_result.outputs["count"] == 2
        for pose in block.last_result.outputs["poses"]:
            assert pose["source_frame"] == "part" and pose["target_frame"] == "camera"
            matrix = np.asarray(pose["source_to_target"])
            target = min(truth, key=lambda item: np.linalg.norm(matrix[:3, 3] - item[:3, 3]))
            assert np.linalg.norm(matrix[:3, 3] - target[:3, 3]) < 0.5
            rotation_error = cv2.Rodrigues(matrix[:3, :3] @ target[:3, :3].T)[0]
            assert np.degrees(np.linalg.norm(rotation_error)) < 1
        restored = Graph.from_dict(graph.to_dict(), builtin_registry())
        try:
            assert restored.blocks["locate"].parameters == block.parameters
        finally:
            restored.close()
    finally:
        graph.close()
