import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_GALLERY = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/path_width/station_review"


def test_saved_station_reviews_keep_source_grid_limits_and_license():
    receipt = json.loads((_GALLERY / "samples/receipt.json").read_bytes())
    assert len(receipt) == 11
    assert {item["path_passed"] for item in receipt.values()} == {True, False}
    for name, item in receipt.items():
        folder = _GALLERY / "samples" / name
        report = json.loads((folder / "review.json").read_bytes())
        assert report["group"] == item["group"]
        assert report["image_sha256"] == item["input_pixels_sha256"]
        assert report["station_number"] == item["station_number"]
        image = cv2.imdecode(np.frombuffer((folder / "overlay.png").read_bytes(), np.uint8), 1)
        assert list(image.shape[1::-1]) == report["image_size"]
        if report["group"] in ("missing", "ambiguous"):
            assert report["metrics"][0]["value"] is None
            assert report["trace"]["intensity"] is not None
        if report["physical_station"] is not None:
            assert report["metrics"][2]["units"] == "mm"
            assert (folder / "physical_trend.png").is_file()
    license_text = (_GALLERY.parent / "weld_bead/LICENSE-data.txt").read_text()
    assert "Copyright (c) 2024 Hyun Soo Kim" in license_text
    assert "MIT License" in license_text


def test_station_gallery_replays_completed_outputs_without_retuning(tmp_path):
    pytest.importorskip("opencv_simple_path_width.station_review")
    spec = importlib.util.spec_from_file_location("station_gallery", _GALLERY / "build_views.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "station-review"
    actual = module.build(target)
    expected = json.loads((_GALLERY / "samples/receipt.json").read_bytes())
    assert actual == expected
    for name in actual:
        for source in (_GALLERY / "samples" / name).iterdir():
            rebuilt = (target / name / source.name).read_bytes()
            if source.suffix == ".json":
                # Git may check text out with CRLF on Windows. Numeric records
                # must match without treating that as changed evidence.
                assert json.loads(source.read_bytes()) == json.loads(rebuilt)
            else:
                assert (
                    hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(rebuilt).digest()
                )
    with pytest.raises(FileExistsError):
        module.build(target)
