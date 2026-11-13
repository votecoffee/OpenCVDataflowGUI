import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "kind,index,count",
    [
        ("asymmetric_brackets", 0, 6),
        ("asymmetric_brackets", 1, 4),
        ("visa_pcb2", 0, 1),
        ("visa_pcb2", 1, 1),
    ],
)
def test_selected_review_fixtures_keep_counts_limits_and_image_grids(kind, index, count):
    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/pattern_matching"
    samples = root / "result_review/samples"
    receipt = json.loads((samples / "receipt.json").read_bytes())
    for status in ("accepted", "rejected"):
        name = f"{kind}_{index}_{status}"
        row = receipt[name]
        assert row["accepted_count"] == count
        project = json.loads((root / row["project"]).read_bytes())
        block = next(b for b in project["blocks"] if b["type"] == "pattern.ogpm_recipe")
        recipe = (root / row["project"]).parent / block["parameters"]["recipe_path"]
        assert hashlib.sha256(recipe.read_bytes()).hexdigest() == row["recipe_sha256"]
        folder = samples / name
        report = json.loads((folder / "review.json").read_bytes())
        assert report["accepted"] == (status == "accepted")
        assert report["metrics"][0]["minimum"] == 0.55
        assert report["metrics"][4]["units"] == "working pixels"
        if status == "rejected":
            assert report["reason"] == "score_cutoff"
            assert report["metrics"][0]["status"] == "below limit"
        edges = report["edges"]
        assert edges["found"] + edges["missing"] + edges["unmeasured"] == edges["trained"]
        fit = cv2.imdecode(
            np.frombuffer((folder / "fit.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        chart = cv2.imdecode(
            np.frombuffer((folder / "assessment.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        assert list(fit.shape[1::-1]) == report["working_image_size"] == row["working_size"]
        assert chart.shape == (640, 1120, 3)
