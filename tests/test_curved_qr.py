"""Native wrapped labels with independent pinhole geometry and original module bytes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.code_readers import ReadQRCode
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock, ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/curved_qr"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def image(case="wrapped"):
    return cv2.imread(str(BASE / "assets" / (case + ".png")), cv2.IMREAD_GRAYSCALE)


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_curved_labels_text_modules_source_coordinates_and_filmstrip(case, profile):
    reference = TRUTH["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["read"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed is reference["passed"]
        info = result.outputs["info"]
        assert info["backend"] == "standard" and info["mode"] == "curved"
        assert info["rectification"]["geometry"] == "native_curved"
        assert info["rectification"]["mapping_to_source"] is None
        assert info["rectification"]["available"] is reference["passed"]
        assert result.payloads["codes"]["ticket"]["value"] == case + ".png"
        source_frame = block.input_payloads["image"]["spatial"]
        assert result.payloads["codes"]["spatial"] == source_frame
        assert result.payloads["overlay"]["spatial"] == source_frame
        json.dumps(info, allow_nan=False)
        flat, _, _ = cv2.QRCodeDetector().detectAndDecode(image(case))
        assert bool(flat) is reference["flat_reads"]
        assert info["decoded_count"] == int(reference["passed"])
        if reference["passed"]:
            (code,) = result.outputs["codes"]
            assert code["text"] == TRUTH["text"]
            assert code["mode"] == "curved"
            assert code["corner_model"] == "native_detection_quadrilateral"
            indices = reference["anchor_corner_indices"]
            corners = np.array(code["corners"])
            surface = np.array(reference["surface_corners"])
            error = np.linalg.norm(corners[indices] - surface[indices], axis=1)
            assert max(error) < reference["maximum_anchor_error_pixels"]
            # The extrapolated fourth corner is NOT the actual curved surface corner.
            if case in ("wrapped", "selected", "vertical"):
                assert np.linalg.norm(corners[2] - surface[2]) > 40
            patch = result.outputs["rectified"]
            wanted = np.asarray(reference["modules"], np.uint8)
            assert patch.shape == wanted.shape
            assert set(np.unique(patch)) <= {0, 255}
            assert int(np.count_nonzero(patch != wanted)) <= reference["maximum_module_errors"]
            payload = result.payloads["rectified"]
            assert payload["spatial"]["root_id"] != source_frame["root_id"]
            assert payload["spatial"]["image_size"] == [len(wanted)] * 2
            assert payload["qr_observation"]["source_spatial"] == source_frame
            assert payload["qr_observation"]["source_corners"] == code["corners"]
            assert payload["qr_observation"]["mapping_to_source"] is None
            assert ticket_key(payload["ticket"]) == ticket_key(result.payloads["codes"]["ticket"])
            if profile == "debug":
                outline, axis_x, axis_y, label = result.drawings["decoded codes"].objects
                np.testing.assert_allclose(outline.points, corners)
                assert axis_x.kind == axis_y.kind == "arrow" and label.text == code["text"]
            np.testing.assert_array_equal(
                graph.blocks["filmstrip"].last_result.outputs["image"], result.outputs["overlay"]
            )
        else:
            assert result.outputs["codes"] == [] and result.outputs["rectified"] is NO_VALUE
            assert result.fail_reasons[0]["code"] == "code_read_count_below_minimum"
            assert not block.error
    finally:
        graph.close()


@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgra", "rgba"])
def test_curved_encoding_ticket_recipe_and_pyramid_provenance(encoding):
    pixels = image()
    if encoding != "gray":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        pixels = pixels // 2 + np.array([30, 60, 90], np.uint8)
        if encoding.startswith("rgb"):
            pixels = pixels[..., ::-1].copy()
        if encoding.endswith("a"):
            pixels = np.dstack((pixels, np.full(pixels.shape[:2], 255, np.uint8)))
    pixels = ImageArray(pixels, encoding)
    before = pixels.copy()
    spatial = frame([600, 600], "camera")
    spatial.update(root_size=[1200, 1200], to_root=np.diag([2, 2, 1]).tolist())
    pyramid = {"level": 1, "largest_size": [1200, 1200], "to_largest": np.diag([2, 2, 1]).tolist()}
    ticket = TicketClock().issue("camera", "external", "wrapped-part-42")
    context = {
        "spatial": spatial,
        "pyramid": pyramid,
        "ticket": ticket,
        "recipe_ref": {"id": "test-recipe", "revision": 3, "digest": "0" * 64},
    }
    block = ReadQRCode(parameters={"mode": "curved", "draw_overlay": False})
    result = block.execute({"image": ValuePacket(pixels, context)})
    assert result is not None, block.error
    assert result.passed
    np.testing.assert_array_equal(pixels, before)
    assert result.outputs["codes"][0]["text"] == TRUTH["text"]
    payload = result.payloads["rectified"]
    assert "pyramid" not in payload
    assert payload["qr_observation"]["source_pyramid"] == pyramid
    assert payload["qr_observation"]["source_spatial"] == spatial
    assert payload["recipe_ref"] == context["recipe_ref"]
    assert ticket_key(payload["ticket"]) == ticket_key(ticket)


def test_distinct_module_grids_and_no_stale_output_after_failed_read():
    block = ReadQRCode(parameters={"mode": "curved"})
    first = block.execute({"image": image()})
    failed = block.execute({"image": image("severe")})
    third = block.execute({"image": image()})
    assert first.passed and third.passed and not failed.passed
    assert failed.outputs["rectified"] is NO_VALUE and failed.outputs["codes"] == []
    assert not failed.outputs["info"]["rectification"]["available"]
    assert not block.error
    assert (
        first.payloads["rectified"]["spatial"]["root_id"]
        != third.payloads["rectified"]["spatial"]["root_id"]
    )


def test_curved_path_has_no_automatic_flat_fallback_and_contains_native_errors(monkeypatch):
    import opencv_dataflow.code_readers as readers

    original = readers.make_detector
    calls = []

    class Detector:
        def detectAndDecodeCurved(self, search):
            calls.append(search.shape)
            raise cv2.error("Controlled curved decode failure")

        def detectAndDecode(self, search):
            pytest.fail("Unexpected flat fallback")

    monkeypatch.setattr(readers, "make_detector", lambda _: Detector())
    block = ReadQRCode(parameters={"mode": "curved"})
    assert block.execute({"image": image()}) is None
    assert block.state == "error" and "Controlled curved" in block.error
    assert len(calls) == 1 and not any(block.current_emissions.values())
    monkeypatch.setattr(readers, "make_detector", original)
    assert block.execute({"image": image()}).passed


def test_empty_selection_reports_mode_and_does_not_call_native(monkeypatch):
    import opencv_dataflow.code_readers as readers

    monkeypatch.setattr(readers, "make_detector", lambda _: pytest.fail("Native empty selection"))
    pixels = image()
    block = ReadQRCode(parameters={"mode": "curved"})
    result = block.execute({"image": pixels, "mask": np.zeros_like(pixels)})
    assert not result.passed and result.outputs["count"] == 0
    assert result.outputs["info"]["mode"] == "curved"
    assert result.event_data["qr_mode"] == "curved"
    assert result.outputs["rectified"] is NO_VALUE


def test_nonempty_mask_region_intersection_can_exclude_the_label_without_moving_source():
    graph = Graph.load(BASE / "selected.json", builtin_registry())
    try:
        graph.run()
        region = graph.blocks["selection"].last_result.outputs["region"]
        pixels = image()
        before = pixels.copy()
        mask = np.zeros_like(pixels)
        mask[90:120, 90:120] = 255  # Inside the region, outside the projected label/quiet zone.
        block = ReadQRCode(parameters={"mode": "curved"})
        result = block.execute({"image": pixels, "mask": mask, "region": region})
        assert result is not None and not result.passed
        assert result.outputs["codes"] == [] and result.outputs["rectified"] is NO_VALUE
        np.testing.assert_array_equal(pixels, before)
        assert result.outputs["overlay"].shape[:2] == pixels.shape
    finally:
        graph.close()


@pytest.mark.parametrize("graphics", ["on", "off"])
@pytest.mark.parametrize("overlay", [True, False])
def test_curved_result_objects_and_overlay_can_be_disabled_independently(graphics, overlay):
    block = ReadQRCode(parameters={"mode": "curved", "draw_overlay": overlay})
    block.set_graphics_mode(graphics)
    result = block.execute({"image": image()})
    assert result.passed
    assert bool(result.drawings) is (graphics == "on")
    assert (result.outputs["overlay"] is NO_VALUE) is (not overlay)
    assert result.outputs["info"]["rectification"]["available"]


def test_curved_settings_and_optional_info_port_roundtrip_json_xml(tmp_path):
    graph = Graph.load(BASE / "wrapped.json", builtin_registry())
    try:
        block = graph.blocks["read"]
        assert "info" in block.exposed_outputs
        path, xml = tmp_path / "program.json", tmp_path / "program.xml"
        graph.save(path)
        export_project_xml(graph.to_dict(), xml)
        restored = Graph.from_dict(read_project_xml(xml), builtin_registry(), project_dir=tmp_path)
        try:
            assert restored.blocks[block.id].parameters == block.parameters
            assert restored.blocks[block.id].exposed_outputs == block.exposed_outputs
            assert not restored.blocks[block.id].last_result
        finally:
            restored.close()
    finally:
        graph.close()


def test_original_fixture_hashes_and_regeneration_preserve_source_collection(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    spec = importlib.util.spec_from_file_location("curved_qr_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((tmp_path / path.name).read_bytes()) == json.loads(path.read_bytes())
    for reference in TRUTH["cases"].values():
        modules = np.asarray(reference["modules"], np.uint8)
        assert hashlib.sha256(modules.tobytes()).hexdigest() == reference["module_sha256"]
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
