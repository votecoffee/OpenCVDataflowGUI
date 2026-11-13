import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.cloud_io import read_cloud


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "name,count", [("generated", 2), ("real_positive", 1), ("real_no_match", 0)]
)
def test_saved_result_views_preserve_known_outcomes_and_separate_rejected_clouds(name, count):
    root = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"
    folder = root / "result_review"
    receipt = json.loads((folder / "receipt.json").read_bytes())[name]
    report = json.loads((folder / f"{name}.json").read_bytes())
    assert receipt["accepted_count"] == len(report["matches"]) == count
    for relative, digest in receipt["input_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    pixels = np.frombuffer((folder / f"{name}.png").read_bytes(), np.uint8)
    image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    assert image.shape == (640, 960, 3)
    accepted = read_cloud(folder / f"{name}_accepted_fit.ply", units="mm", frame="camera")
    rejected = read_cloud(folder / f"{name}_REJECTED_candidate.ply", units="mm", frame="camera")
    assert len(accepted.points) >= report["sampled_scene_points"]
    assert np.any(np.all(rejected.colors == [255, 170, 35], axis=1))
    if count == 0:
        assert report["assessment"]["status"] == "verification_failed"
        assert report["execution"]["rejection_counts"] == {"geometry": 24}
        assert len(accepted.points) == report["sampled_scene_points"]
        assert np.all(accepted.colors == 115)
        review = report["execution"]["candidate_review"][0]
        assert review["failed_checks"] == ["coverage"]
        assert review["coverage"] < review["limits"]["min_coverage"] == .5
