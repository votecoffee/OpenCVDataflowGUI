import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.cloud_io import read_cloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins"
GALLERY = ROOT / "real_data"


def test_all_six_tools_have_real_projects_credit_and_unchanged_saved_result_images():
    catalog = json.loads((GALLERY / "catalog.json").read_bytes())
    receipt = json.loads((GALLERY / "samples/receipt.json").read_bytes())
    tools = {
        "SimplePatternMatch",
        "SimpleSegmentation",
        "SimpleClassify",
        "SimplePathWidth",
        "SimpleReferenceInspect",
        "Simple3DLocate",
    }
    assert {row["tool"] for row in catalog["examples"]} == tools == set(receipt)
    assert sum(len(row["frames"]) for row in receipt.values()) == 28
    for example in catalog["examples"]:
        assert (ROOT / example["project"]).is_file()
        assert (ROOT / example["attribution"]).is_file()
        assert example["license"] in {"MIT", "CC-BY-4.0"}
        evidence = receipt[example["tool"]]
        assert len(evidence["frames"]) == example["frames"]
        for row in evidence["frames"]:
            raw = (ROOT / row["input_file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["input_file_sha256"]
            path = GALLERY / "samples" / example["tool"] / f"{row['frame']:02}" / "result.png"
            pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
            assert list(pixels.shape[1::-1]) == row["result_size"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["result_pixels_sha256"]
    frames = receipt["SimpleClassify"]["frames"]
    assert sum(row["measurements"]["correct"] for row in frames) == 10
    assert receipt["SimpleSegmentation"]["frames"][1]["measurements"]["foreground_pixels"] == 0
    assert [row["state"] for row in receipt["SimpleReferenceInspect"]["frames"]] == [
        "pass",
        "fail",
        "fail",
        "fail",
    ]


def test_real_depth_example_exports_native_clouds_and_license_audit_covers_every_dataset():
    folder = GALLERY / "samples/Simple3DLocate/00"
    evidence = json.loads((folder / "result.json").read_bytes())
    assert evidence["source_points"] > 20000
    assert evidence["measurements"]["count"] == 1
    for name in ("fit", "aligned_model"):
        cloud = read_cloud(folder / f"{name}.ply")
        assert cloud.units == "mm" and cloud.frame == "camera"
        assert len(cloud.points) > 100
    audit = json.loads((GALLERY / "license_audit.json").read_bytes())
    assert set(audit["datasets"]) == {"VisA", "ARMBench", "VegQual v4", "Weld bead", "T-LESS"}
    assert audit["datasets"]["Weld bead"]["license"] == "MIT"
    assert all(
        row["license"] == "CC-BY-4.0"
        for name, row in audit["datasets"].items()
        if name != "Weld bead"
    )
