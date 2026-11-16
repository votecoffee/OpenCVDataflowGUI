"""First eligible masks, sealed camera references and bounded native replay lifecycle."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow import background_reference as assets
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.image_assets import seal
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock


def fixture(dtype=np.uint8, encoding="gray", root="capture-a"):
    y, x = np.mgrid[:48, :64]
    image = (40 + (x + y) % 40).astype(dtype)
    if encoding != "gray":
        image = np.dstack((image, image + 10, image + 20))
    payload = {
        "spatial": frame([64, 48], root),
        "ticket": TicketClock().issue("camera"),
        "custom": {"lot": "A"},
    }
    clean = ImageArray(image, encoding, payload)
    current = clean.copy()
    current[12:28, 18:35] = 220
    return clean, current


def saved_reference(tmp_path, clean):
    record = assets.make_reference(clean, "camera-A", "cal-7")
    path = tmp_path / "background.json"
    path.write_bytes(assets.reference_bytes(record))
    return record, path


def restored_block(kind, path, **parameters):
    return builtin_registry().create(
        kind,
        parameters={
            "reference_path": str(path),
            "reference_camera_id": "camera-A",
            "reference_calibration_id": "cal-7",
            "detect_shadows": False,
            **parameters,
        },
    )


@pytest.mark.parametrize("kind", ("sequence.mog2", "sequence.knn_background"))
@pytest.mark.parametrize("encoding", ("gray", "rgb", "bgr"))
def test_first_current_image_matches_native_clean_replay_and_independent_foreground_truth(
    kind, encoding, tmp_path
):
    clean, current = fixture(encoding=encoding)
    record, path = saved_reference(tmp_path, clean)
    block = restored_block(kind, path)
    result = block.execute({"image": current})
    assert result is not None, block.error
    assert result.outputs["ready"] and result.outputs["activations"] == 1
    expected = np.zeros((48, 64), np.uint8)
    expected[12:28, 18:35] = 255
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    native = block.new_model()
    working = np.asarray(clean)[..., ::-1] if encoding == "rgb" else np.asarray(clean)
    live = np.asarray(current)[..., ::-1] if encoding == "rgb" else np.asarray(current)
    for _ in range(32):
        native.apply(working, learningRate=0.5)
    native.apply(working, learningRate=0)
    np.testing.assert_array_equal(result.outputs["raw_mask"], native.apply(live, learningRate=0))
    np.testing.assert_array_equal(result.outputs["background"], working)
    assert (
        result.outputs["status"]["seeded"] and result.outputs["status"]["required_activations"] == 0
    )
    assert result.outputs["status"]["background_reference"]["id"] == record["id"]
    assert result.payloads["mask"]["ticket"] == current.metadata["ticket"]
    assert result.payloads["foreground"]["spatial"] == current.metadata["spatial"]
    # A fresh acquisition root does not invalidate unchanged camera-grid geometry.
    _, fresh = fixture(encoding=encoding, root="after-restart")
    block.reset_runtime()
    again = block.execute({"image": fresh})
    assert again.outputs["ready"] and again.outputs["activations"] == 1
    np.testing.assert_array_equal(again.outputs["mask"], expected)


def test_float_mog2_replays_without_quantization_and_learning_can_be_disabled_on_first_image(
    tmp_path,
):
    clean, current = fixture(np.float32, "rgb")
    clean += np.float32(0.125)
    current += np.float32(0.125)
    _, path = saved_reference(tmp_path, clean)
    block = restored_block("sequence.mog2", path)
    result = block.execute({"image": current, "learn": False})
    assert result is not None and result.outputs["ready"], block.error
    np.testing.assert_array_equal(result.outputs["background"], clean[..., ::-1])
    assert result.outputs["background"].dtype == np.float32
    assert result.outputs["status"]["learning_activations"] == 0


@pytest.mark.parametrize("kind", ("sequence.mog2", "sequence.knn_background"))
@pytest.mark.parametrize("policy", ("frozen", "guarded"))
def test_persistent_parts_are_not_absorbed_by_frozen_or_mask_guarded_learning(
    kind, policy, tmp_path
):
    clean, current = fixture()
    _, path = saved_reference(tmp_path, clean)
    block = restored_block(kind, path, reference_update=policy, learning_rate=0.5)
    known = np.zeros(clean.shape, np.uint8)
    known[12:28, 18:35] = 255
    for _ in range(80):
        result = block.execute(
            {"image": current, **({"known_foreground": known} if policy == "guarded" else {})}
        )
        assert result is not None, block.error
    assert result.outputs["mask"][16, 22] == 255
    np.testing.assert_array_equal(result.outputs["background"], clean)
    assert result.outputs["status"]["learning_activations"] == (80 if policy == "guarded" else 0)


@pytest.mark.parametrize(
    "case", ("camera", "calibration", "grid", "pyramid", "recipe", "depth", "shape", "encoding")
)
def test_mismatches_reject_before_allocating_or_mutating_a_native_model(
    case, tmp_path, monkeypatch
):
    clean, current = fixture()
    _, path = saved_reference(tmp_path, clean)
    block = restored_block("sequence.mog2", path)
    if case == "camera":
        block.parameters["reference_camera_id"] = "camera-B"
    elif case == "calibration":
        block.parameters["reference_calibration_id"] = "cal-8"
    elif case == "grid":
        current.metadata["spatial"]["to_root"][0][2] = 1
    elif case == "pyramid":
        current.metadata["pyramid"] = {
            "level": 1,
            "largest_size": [128, 96],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        }
    elif case == "recipe":
        current.metadata["recipe_ref"] = {"id": "another-recipe", "revision": 1, "digest": "a" * 64}
    elif case == "depth":
        current = current.astype(np.float32)
    elif case == "shape":
        current = current[:-1].copy()
    else:
        current = ImageArray(np.dstack((current, current, current)), "rgb", current.metadata)
    monkeypatch.setattr(block, "new_model", lambda: pytest.fail("Mismatch allocated native state"))
    assert block.execute({"image": current}) is None
    assert block.error_kind == "invalid" and not block.memory["streams"]


def test_capture_known_foreground_is_rejected_and_recipe_independent_binding_is_explicit(tmp_path):
    clean, current = fixture()
    block = builtin_registry().create(
        "sequence.capture_background_reference", parameters={"camera_id": "camera-A"}
    )
    mask = np.zeros(clean.shape, np.uint8)
    mask[4, 4] = 255
    assert block.execute({"image": current, "known_foreground": mask}) is None
    assert block.error_kind == "invalid" and not any(block.current_emissions.values())
    block.parameters["recipe_binding"] = "ignore"
    result = block.execute({"image": clean})
    assert result is not None, block.error
    record = result.outputs["reference"]
    path = tmp_path / "ignored.json"
    path.write_bytes(assets.reference_bytes(record))
    current.metadata["recipe_ref"] = {"id": "new-recipe", "revision": 1, "digest": "b" * 64}
    consumer = restored_block("sequence.mog2", path, reference_calibration_id="")
    outcome = consumer.execute({"image": current})
    assert outcome is not None, consumer.error
    assert outcome.outputs["ready"]


@pytest.mark.parametrize(
    "case",
    (
        "checksum",
        "image_checksum",
        "duplicate",
        "nan",
        "bom",
        "extra",
        "bad_grid",
        "depth",
        "range",
    ),
)
def test_corrupt_or_unsupported_reference_files_are_contained(case, tmp_path):
    clean, current = fixture()
    record, path = saved_reference(tmp_path, clean)
    if case == "checksum":
        record["camera_id"] = "tampered"
    elif case == "image_checksum":
        record["image"]["data"] = record["image"]["data"][:-4] + "AAAA"
        record["id"] = seal(record)
    elif case == "extra":
        record["unsupported"] = True
    elif case == "bad_grid":
        record["image"]["context"]["spatial"]["image_size"] = [1, 1]
        record["image"]["id"] = seal(record["image"])
        record["id"] = seal(record)
    elif case == "depth":
        record = assets.make_reference(clean, "camera-A", "cal-7")
        record["image"]["dtype"] = "uint16"
        record["image"]["id"] = seal(record["image"])
        record["id"] = seal(record)
    elif case == "range":
        invalid = clean.astype(np.float32)
        invalid[0, 0] = 256
        with pytest.raises(ValueError):
            assets.make_reference(invalid, "camera-A")
        return
    raw = (json.dumps(record) + "\n").encode()
    if case == "duplicate":
        raw = raw.replace(b"{", b'{"model":"duplicate",', 1)
    elif case == "nan":
        raw = b'{"model":NaN}'
    elif case == "bom":
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)
    block = restored_block("sequence.mog2", path)
    assert block.execute({"image": current}) is None
    assert block.error_kind == "invalid", block.error
    assert not any(block.current_emissions.values())


def test_file_pin_cache_refresh_reset_and_recovery_never_use_stale_native_state(
    tmp_path, monkeypatch
):
    from opencv_dataflow import native_background

    clean, current = fixture()
    _, path = saved_reference(tmp_path, clean)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    block = restored_block("sequence.mog2", path, reference_sha256=digest)
    calls = []
    actual = native_background.load_reference

    def observe(*args):
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(native_background, "load_reference", observe)
    first = block.execute({"image": current})
    for _ in range(3):
        assert block.execute({"image": current}).outputs["ready"]
    assert len(calls) == 1
    assert (
        first.outputs["status"]["generation"] == block.last_result.outputs["status"]["generation"]
    )
    assert block.execute({"image": current, "reset": True}).outputs["activations"] == 1
    assert len(calls) == 2
    path.write_bytes(path.read_bytes() + b" ")
    assert block.execute({"image": current}) is None and block.error_kind == "invalid"
    block.parameters["reference_sha256"] = ""
    assert block.execute({"image": current}).outputs["activations"] == 1
    assert len(calls) == 4


def test_guard_requirement_work_budget_and_incomplete_knn_replay_withhold_output(
    tmp_path, monkeypatch
):
    clean, current = fixture()
    _, path = saved_reference(tmp_path, clean)
    for parameters in (
        {"reference_update": "guarded"},
        {"maximum_reference_work": 1},
        {"reference_replay_passes": 1},
    ):
        block = restored_block("sequence.knn_background", path, **parameters)
        assert block.execute({"image": current}) is None
        assert block.error_kind == "invalid" and not any(block.current_emissions.values())
        assert not block.memory["streams"]
    block = restored_block("sequence.mog2", path)
    native = block.new_model
    monkeypatch.setattr(
        block, "new_model", lambda: (_ for _ in ()).throw(cv2.error("Controlled replay failure"))
    )
    assert block.execute({"image": current}) is None and block.error_kind == "exception"
    monkeypatch.setattr(block, "new_model", native)
    assert block.execute({"image": current}).outputs["ready"]


def test_file_publication_is_exclusive_atomic_and_program_xml_save_as_keeps_sealed_bytes(
    tmp_path, monkeypatch
):
    clean, _ = fixture()
    record, path = saved_reference(tmp_path, clean)
    block = builtin_registry().create("output.background_reference", parameters={"path": str(path)})
    original = path.read_bytes()
    assert block.execute({"reference": record}) is None
    assert path.read_bytes() == original
    block.parameters["write_mode"] = "replace"
    assert block.execute({"reference": record}) is not None

    def broken(*args):
        raise OSError("Controlled atomic publication failure")

    actual_replace = assets.os.replace
    monkeypatch.setattr(assets.os, "replace", broken)
    assert block.execute({"reference": record}) is None
    assert path.read_bytes() == original and not list(tmp_path.glob(".background-*.tmp"))
    monkeypatch.setattr(assets.os, "replace", actual_replace)
    graph, registry = Graph(), builtin_registry()
    try:
        consumer = graph.add(restored_block("sequence.mog2", path))
        expected = graph.to_dict()
        xml = tmp_path / "program.xml"
        export_project_xml(expected, xml)
        opened = Graph.from_dict(read_project_xml(xml), registry)
        assert opened.to_dict() == expected
        opened.close()
        graph.save(tmp_path / "program.json")
        consumer.parameters["reference_path"] = "background.json"
        other = tmp_path / "copy"
        other.mkdir()
        graph.save(other / "program.json")
        assert (other / "background.json").read_bytes() == original
    finally:
        graph.close()


def test_offscreen_reference_controls_do_not_load_files_or_create_models(
    app, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    clean, _ = fixture()
    _, path = saved_reference(tmp_path, clean)
    graph = Graph()
    block = graph.add(restored_block("sequence.knn_background", path))
    monkeypatch.setattr(block, "new_model", lambda: pytest.fail("Editing created native state"))
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        assert isinstance(window.inspector.fields["reference_update"], QComboBox)
        assert block._reference_cache is None and not block.memory["streams"]
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
