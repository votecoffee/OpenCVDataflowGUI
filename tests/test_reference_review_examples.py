import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect"
GALLERY = ROOT / "result_review/samples"


def test_gallery_keeps_native_maps_grid_provenance_and_unevaluated_states():
    receipt = json.loads((GALLERY / "receipt.json").read_bytes())
    assert len(receipt) == 8
    assert {row["state"] for row in receipt.values()} == {"pass", "fail", "not_evaluated"}
    for name, context in receipt.items():
        folder = GALLERY / name
        report = json.loads((folder / "review.json").read_bytes())
        assert report["state"] == context["state"]
        assert report["input_pixels_sha256"] == context["input_pixels_sha256"]
        for key, array_context in report["arrays"].items():
            pixels = np.load(folder / f"{key}.npy")
            assert list(pixels.shape[::-1]) == report["reference_size"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == array_context["sha256"]
        if not report["inspection"]["evaluated"]:
            assert report["metrics"][3]["value"] is None
            assert not report["inspected_reference_pixels"]
    assert receipt["visa_pcb2_1"]["source_ground_truth"] == "normal"
    assert receipt["visa_pcb2_1"]["state"] == "not_evaluated"
    assert receipt["visa_pcb2_0"]["training_sample_count"] == 1


@pytest.mark.parametrize("real", [False, True])
def test_assessment_projects_relocate_and_show_pass_or_evaluation_failure(tmp_path, real):
    pytest.importorskip("opencv_simple_reference_inspect.result_review")
    folder = GALLERY / "visa_pcb2_project" if real else ROOT
    project_name = "project.json" if real else "assessment-project.json"
    target = tmp_path / "relocated"
    target.mkdir()
    shutil.copy2(folder / project_name, target / project_name)
    shutil.copytree(folder / "assets", target / "assets")
    graph = Graph.load(target / project_name, builtin_registry())
    try:
        for index in range(2):
            graph.run()
            block = graph.blocks["inspection"]
            assert block.state == ("pass" if not index else "fail"), block.error
            outputs = block.last_result.outputs
            assert outputs["assessment"].shape == (1050, 1500, 3)
            assert block.filmstrip_output == "assessment"
            assert outputs["inspection"]["evaluated"] is (not (real and index == 1))
    finally:
        graph.close()
