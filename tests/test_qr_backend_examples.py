"""Backend comparisons against known payload/module/placement references."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/qr_backends"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_comparisons_match_known_symbols_placements_and_rectification(case, profile):
    graph = Graph.load(HERE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    reference = TRUTH["cases"][case]
    truth = {code["text"]: code for code in reference["codes"]}
    try:
        graph.run()
        for backend in ("standard", "aruco"):
            block = graph.blocks[backend]
            result = block.last_result
            assert result is not None, block.error
            codes = result.outputs["codes"]
            expected = reference["decoded_by_backend"][backend]
            assert sorted(code["text"] for code in codes) == sorted(expected)
            assert result.passed == (case != "small" or backend != "standard")
            assert result.payloads["codes"]["ticket"]["value"] == f"{case}.png"
            assert result.payloads["codes"]["spatial"] == block.input_payloads["image"]["spatial"]
            for code in codes:
                error = np.linalg.norm(
                    np.asarray(code["corners"]) - truth[code["text"]]["corners"], axis=1
                )
                assert max(error) < 2.5, (case, backend, error)
                np.testing.assert_allclose(code["center"], np.mean(code["corners"], axis=0))
            patch = result.outputs["rectified"]
            if len(codes) == 1:
                wanted = truth[codes[0]["text"]]
                assert patch.shape == (wanted["module_size"],) * 2
                assert hashlib.sha256(patch.tobytes()).hexdigest() == wanted["module_sha256"]
                assert (
                    result.payloads["rectified"]["spatial"]["root_id"]
                    != result.payloads["codes"]["spatial"]["root_id"]
                )
            else:
                assert patch is NO_VALUE
    finally:
        graph.close()


def test_mask_region_intersection_preserves_original_camera_coordinates():
    from opencv_dataflow.code_readers import ReadQRCode

    image = cv2.imread(str(HERE / "assets/multiple.png"), cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(image)
    mask[:, :600] = 255
    region = {
        "model": "image_region_v1",
        "image_size": list(image.shape[::-1]),
        "shape": "polygon",
        "vertices": [[300, 50], [610, 50], [610, 380], [300, 380]],
    }
    reference = TRUTH["cases"]["multiple"]["codes"][1]
    for backend in ("standard", "aruco"):
        block = ReadQRCode(parameters={"backend": backend})
        result = block.execute({"image": image, "mask": mask, "region": region})
        assert result is not None, block.error
        (code,) = result.outputs["codes"]
        assert code["text"] == "CENTER-07"
        np.testing.assert_allclose(code["corners"], reference["corners"], atol=1)


def test_regeneration_matches_all_committed_pixels_and_projects(tmp_path):
    spec = importlib.util.spec_from_file_location("qr_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha


def test_old_projects_without_backend_fields_keep_standard_reader(tmp_path):
    path = HERE.parent / "qr_encoding/qr_roundtrip.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        reader = graph.blocks["read"]
        assert reader.parameters["backend"] == "standard" and reader.last_result.passed
        graph.save(tmp_path / "reopened.json")
    finally:
        graph.close()
