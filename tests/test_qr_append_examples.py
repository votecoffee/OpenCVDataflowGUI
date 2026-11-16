"""Saved native examples explicitly reject partial or ambiguous structured messages."""

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

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/qr_structured_append"
TRUTH = json.loads((HERE / "truth.json").read_bytes())
REASONS = {
    "missing": "missing_qr_fragments",
    "duplicate": "duplicate_qr_fragment_index",
    "mismatched": "mismatched_qr_fragment_set",
    "mixed_count": "mismatched_qr_fragment_set",
    "collision_digest": "qr_expected_digest_mismatch",
    "damaged_header": "unverified_qr_fragment",
}


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", TRUTH["cases"])
def test_saved_assembly_results_geometry_and_no_partial_payload_events(case, profile):
    graph = Graph.load(HERE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    block = graph.blocks["assemble"]
    try:
        for run in range(3 if case == "shuffled_group" else 1):
            graph.run()
            if case == "shuffled_group" and run < 2:
                assert block.last_result is None
        result = block.last_result
        assert result is not None, block.error
        expected = case in TRUTH["success_cases"]
        assert result.passed is expected and result.outputs["complete"] is expected
        if expected:
            assert result.outputs["text"] == TRUTH["text"]
            assert (
                hashlib.sha256(bytes.fromhex(result.outputs["bytes_hex"])).hexdigest()
                == TRUTH["sha256"]
            )
            assert not result.outputs["status"]["missing_indices"]
            assert not result.outputs["status"]["duplicate_indices"]
        else:
            assert result.outputs["text"] is result.outputs["bytes_hex"] is NO_VALUE
            assert not block.current_emissions.get("text") and not block.current_emissions.get(
                "bytes_hex"
            )
            assert REASONS[case] in result.outputs["status"]["reasons"]
            assert REASONS[case] in [reason["code"] for reason in result.fail_reasons]
        if case == "shuffled_group":
            assert [r["ticket"]["value"] for r in result.outputs["fragments"]] == [
                "part_2.png",
                "part_0.png",
                "part_1.png",
            ]
        else:
            assert result.payloads["fragments"]["ticket"]["value"] == case + ".png"
            actual = sorted(
                tuple(np.min(r["corners"], axis=0)) for r in result.outputs["fragments"]
            )
            np.testing.assert_allclose(
                actual, sorted(TRUTH["scene_core_top_lefts"][case]), atol=1.5
            )
            source = cv2.imread(str(HERE / "assets" / f"{case}.png"), cv2.IMREAD_GRAYSCALE)
            assert result.outputs["overlay"].shape == (*source.shape, 3)
            assert result.payloads["overlay"]["spatial"] == block.input_payloads["image"]["spatial"]
            if profile == "debug":
                objects = result.drawings["QR fragments"].objects
                assert len(objects) == 4 * len(result.outputs["fragments"])
                assert all(
                    obj.color == ((60, 230, 120) if expected else (0, 190, 255)) for obj in objects
                )
    finally:
        graph.close()


def test_native_decoder_repairs_damaged_fragment_while_strict_header_verifier_refuses_it():
    image = cv2.imread(str(HERE / "assets/damaged_header.png"), cv2.IMREAD_GRAYSCALE)
    decoded, texts, _, _ = cv2.QRCodeDetectorAruco().detectAndDecodeBytesMulti(image)
    assert decoded and TRUTH["text"].encode() in texts
    graph = Graph.load(HERE / "damaged_header.json", builtin_registry())
    try:
        graph.run()
        result = graph.blocks["assemble"].last_result
        assert not result.passed
        assert any("error correction" in f.get("error", "") for f in result.outputs["fragments"])
    finally:
        graph.close()


def test_xor_parity_collision_needs_the_expected_digest_to_distinguish_payload():
    graph = Graph.load(HERE / "collision_digest.json", builtin_registry())
    try:
        graph.blocks["assemble"].parameters["expected_sha256"] = ""
        graph.run()
        result = graph.blocks["assemble"].last_result
        assert result.passed and result.outputs["text"] != TRUTH["text"]
        assert result.outputs["text"].endswith("81")
    finally:
        graph.close()


def test_selection_can_remove_a_part_without_emitting_partial_text():
    from opencv_dataflow.qr_append import ReadQRAppend

    image = cv2.imread(str(HERE / "assets/shuffled_scene.png"), cv2.IMREAD_GRAYSCALE)
    mask = np.full_like(image, 255)
    mask[:, -210:] = 0
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": list(image.shape[::-1]),
        "rectangle": [0, 0, image.shape[1], image.shape[0]],
    }
    block = ReadQRAppend()
    result = block.execute({"image": image, "mask": mask, "region": region})
    assert result is not None, block.error
    assert not result.passed and result.outputs["text"] is NO_VALUE
    assert result.outputs["status"]["missing_indices"] == [1]
    assert not block.current_emissions.get("text")
    assert all(min(np.asarray(f["corners"])[:, 0]) >= 55 for f in result.outputs["fragments"])


def test_regeneration_matches_all_original_assets_and_saved_projects(tmp_path):
    spec = importlib.util.spec_from_file_location("append_example", HERE / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
