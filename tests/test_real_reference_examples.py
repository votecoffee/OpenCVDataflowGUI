import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect"


def test_real_tote_reference_relocates_reopens_and_replays_independent_captures(tmp_path):
    model_io = pytest.importorskip("opencv_simple_reference_inspect")
    source = ROOT / "armbench_tote"
    target = tmp_path / "tote"
    target.mkdir()
    shutil.copytree(source / "assets", target / "assets")
    shutil.copy2(source / "project.json", target / "project.json")
    expected = json.loads((source / "evaluation.json").read_bytes())
    provenance = json.loads((source / "provenance.json").read_bytes())
    assert provenance["reference"] not in {Path(row["image"]).stem for row in expected}
    for image_id, entry in provenance["images"].items():
        assert (
            hashlib.sha256((target / f"assets/{image_id}.jpg").read_bytes()).hexdigest()
            == entry["prepared_sha256"]
        )
    graph = Graph.load(target / "project.json", builtin_registry())
    try:
        block = graph.blocks["inspection"]
        model = model_io.load_model(
            target / block.parameters["recipe_path"], sha256=block.parameters["sha256"]
        )
        assert len(model.samples) == 1
        assert model.settings.max_total_defect_area == int(model.inspection_mask.sum() * 0.05)
        assert model.settings.min_defect_area == 500
        assert model.settings.normalization == "none" and model.settings.min_correlation is None
        for row in expected:
            graph.run()
            assert block.state == ("pass" if row["expected_occupancy"] == "empty" else "fail"), (
                block.error
            )
            output = block.last_result.outputs
            assert output["inspection"] == row["inspection"]
            assert output["assessment"].shape == (1050, 1500, 3)
            assert output["inspection"]["evaluated"]
            assert np.count_nonzero(output["inspected_mask"]) > 180000
    finally:
        graph.close()
