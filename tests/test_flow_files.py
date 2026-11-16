"""Native/independent .flo parity, image association and exclusive file publication."""

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.flow_files import MAX_PIXELS, LoadFlowFile, SaveFlowFile, publication
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.sequence_tools import DenseFlow
from opencv_dataflow.tickets import TicketClock, ticket_key


def execute(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


@pytest.fixture
def samples():
    y, x = np.indices((20, 28))
    source = ImageArray(
        ((x * 17 + y * 11) % 256).astype(np.uint8), "gray", {"spatial": frame([28, 20], "source")}
    )
    target = ImageArray(
        np.roll(source, (-2, 3), axis=(0, 1)), "gray", {"spatial": frame([28, 20], "target")}
    )
    vectors = np.dstack((x / 16 - 0.5, 1 - y / 8)).astype(np.float32)
    return source, target, vectors


def save(tmp_path, vectors, source=None, target=None, **parameters):
    block = SaveFlowFile(parameters={"path": str(tmp_path / "部品-é.flo"), **parameters})
    inputs = {"flow": vectors}
    if source is not None:
        inputs.update(source=source, target=target)
    return block, execute(block, inputs)


@pytest.mark.parametrize("direction", ["source_to_target", "target_to_source"])
def test_signed_samples_match_independent_and_native_bytes_and_preserve_pair(
    tmp_path, samples, direction
):
    source, target, vectors = samples
    original = vectors.copy()
    _, saved = save(
        tmp_path,
        vectors,
        source,
        target,
        direction=direction,
        metadata_mode="require_images",
        source_label="before",
        target_label="after",
    )
    path = Path(saved.outputs["path"])
    raw = path.read_bytes()
    assert raw[:12] == struct.pack("<4sii", b"PIEH", 28, 20)
    np.testing.assert_array_equal(np.frombuffer(raw[12:], "<f4").reshape(vectors.shape), original)
    # Native comparison uses an ASCII fixture name: this Windows binding lacks Unicode fopen.
    native_path = tmp_path / "native.flo"
    native_path.write_bytes(raw)
    np.testing.assert_array_equal(cv2.readOpticalFlow(str(native_path)), original)
    assert cv2.writeOpticalFlow(str(tmp_path / "reference.flo"), original)
    assert (tmp_path / "reference.flo").read_bytes() == raw
    reader = LoadFlowFile(
        parameters={
            "path": str(path),
            "metadata_mode": "verify_images",
            "expected_direction": direction,
        }
    )
    loaded = execute(reader, {"source": source, "target": target})
    np.testing.assert_array_equal(loaded.outputs["flow"].vectors, original)
    index = source if direction == "source_to_target" else target
    assert loaded.payloads["flow"]["spatial"] == index.metadata["spatial"]
    assert loaded.outputs["info"]["images_verified"] is True
    assert loaded.outputs["info"]["association"]["source"]["label"] == "before"
    np.testing.assert_array_equal(vectors, original)


def test_reload_rebinds_new_roots_but_never_restores_old_tickets(tmp_path, samples):
    source, target, vectors = samples
    ticket = TicketClock().issue("camera", "fixed", "original")
    _, result = save(
        tmp_path, ValuePacket(DenseFlow(vectors), {"ticket": ticket, "lot": "A"}), source, target
    )
    assert ticket_key(result.payloads["path"]["ticket"]) == ticket_key(ticket)
    manifest = json.loads(Path(result.outputs["path"] + ".json").read_bytes())
    assert "ticket" not in json.dumps(manifest)
    source.metadata["spatial"]["root_id"] = "reopened-source"
    target.metadata["spatial"]["root_id"] = "reopened-target"
    reader = LoadFlowFile(
        parameters={"path": result.outputs["path"], "metadata_mode": "verify_images"}
    )
    loaded = execute(reader, {"source": source, "target": target})
    assert loaded.payloads["flow"]["spatial"]["root_id"] == "reopened-source"
    assert ticket_key(loaded.payloads["flow"]["ticket"]) != ticket_key(ticket)


@pytest.mark.parametrize(
    "case",
    ["wrong_source", "swapped", "shape", "geometry", "encoding", "direction", "changed_flow"],
)
def test_wrong_association_is_rejected_by_reader_or_motion_consumer(tmp_path, samples, case):
    source, target, vectors = samples
    _, saved = save(tmp_path, vectors, source, target)
    reader = LoadFlowFile(
        parameters={"path": saved.outputs["path"], "metadata_mode": "verify_images"}
    )
    if case == "wrong_source":
        source[2, 3] ^= 1
    elif case == "swapped":
        source, target = target, source
    elif case == "shape":
        source = source[:-1]
    elif case == "geometry":
        source.metadata["spatial"]["to_root"][0][0] = 2
    elif case == "encoding":
        # Equal channel bytes, but RGB and BGR declarations are different identities.
        source = ImageArray(np.dstack([source] * 3), "rgb", source.metadata)
    elif case == "direction":
        reader.parameters["expected_direction"] = "target_to_source"
    if case != "changed_flow":
        assert reader.execute({"source": source, "target": target}) is None
        assert reader.error_kind == "invalid" and not reader.current_emissions
    else:
        loaded = execute(reader, {"source": source, "target": target})
        loaded.outputs["flow"].vectors[3, 4] += 0.25
        warp = builtin_registry().create(
            "cv.warp_flow", parameters={"direction": "source_to_target"}
        )
        assert (
            warp.execute({"image": source, "target": target, "flow": loaded.outputs["flow"]})
            is None
        )
        assert warp.error_kind == "invalid"


@pytest.mark.parametrize("block_type", ["cv.warp_flow", "cv.refine_flow"])
def test_consumers_check_pair_and_direction_even_when_loader_only_reads_metadata(
    tmp_path, samples, block_type
):
    source, target, vectors = samples
    _, saved = save(tmp_path, vectors, source, target, direction="target_to_source")
    loaded = execute(LoadFlowFile(parameters={"path": saved.outputs["path"]}), {})
    assert loaded.outputs["info"]["images_verified"] is False
    block = builtin_registry().create(
        block_type,
        parameters=(
            {"direction": "source_to_target"}
            if block_type == "cv.warp_flow"
            else {"alignment": "pixel_grid"}
        ),
    )
    assert (
        block.execute({"image": source, "target": target, "flow": loaded.outputs["flow"]}) is None
    )
    assert block.error_kind == "invalid" and "direction" in block.error.lower()


@pytest.mark.parametrize("mutation", ["hash", "shape", "model", "duplicate", "nan", "oversize"])
def test_damaged_or_stale_manifest_cannot_be_silently_used(tmp_path, samples, mutation):
    _, _, vectors = samples
    _, saved = save(tmp_path, vectors)
    path = Path(saved.outputs["path"])
    companion = Path(str(path) + ".json")
    data = json.loads(companion.read_bytes())
    if mutation == "duplicate":
        companion.write_text('{"model":1,"model":2}')
    elif mutation == "nan":
        companion.write_text('{"model":NaN}')
    elif mutation == "oversize":
        companion.write_bytes(b" " * 65537)
    else:
        data[{"hash": "sha256", "shape": "shape", "model": "model"}[mutation]] = "wrong"
        companion.write_text(json.dumps(data))
    reader = LoadFlowFile(parameters={"path": str(path)})
    assert reader.execute({}) is None and reader.error_kind == "invalid"
    reader.parameters["metadata_mode"] = "ignore"
    assert execute(reader, {}).outputs["info"]["association"].startswith("unspecified")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e10, -1e10])
def test_invalid_vector_policies_preserve_samples_without_making_them_valid(
    tmp_path, samples, value
):
    _, _, vectors = samples
    vectors[1, 2, 0] = value
    block = SaveFlowFile(parameters={"path": str(tmp_path / "bad.flo")})
    assert block.execute({"flow": vectors}) is None and block.error_kind == "invalid"
    assert not list(tmp_path.iterdir())
    block.parameters["invalid_vectors"] = "preserve"
    saved = execute(block, {"flow": vectors})
    reader = LoadFlowFile(parameters={"path": saved.outputs["path"]})
    loaded = execute(reader, {})
    assert loaded.outputs["flow"].vectors.tobytes() == vectors.tobytes()
    assert loaded.outputs["info"]["invalid_vectors"] == 1
    reader.parameters["invalid_vectors"] = "reject"
    assert reader.execute({}) is None and reader.error_kind == "invalid"


def test_float64_narrowing_is_explicit_and_reports_actual_error(tmp_path, samples):
    vectors = samples[2].astype(np.float64) + 1 / 7
    block = SaveFlowFile(parameters={"path": str(tmp_path / "rounded.flo")})
    assert block.execute({"flow": vectors}) is None and block.error_kind == "invalid"
    block.parameters["precision"] = "convert_float64"
    result = execute(block, {"flow": vectors})
    expected = np.max(np.abs(vectors - vectors.astype(np.float32)))
    assert result.outputs["info"]["maximum_rounding_error"] == expected > 0
    assert result.outputs["flow"].vectors.dtype == np.float32


@pytest.mark.parametrize("kind", ["integer", "channels", "empty", "overflow"])
def test_unsupported_arrays_do_not_create_files_or_directories(tmp_path, kind):
    values = np.zeros((8, 10, 2), np.float32)
    if kind == "integer":
        values = values.astype(np.int32)
    elif kind == "channels":
        values = values[..., :1]
    elif kind == "empty":
        values = values[:0]
    else:
        values = np.full((8, 10, 2), 1e100, np.float64)
    block = SaveFlowFile(
        parameters={
            "path": str(tmp_path / "new/out.flo"),
            "create_directories": True,
            "precision": "convert_float64",
        }
    )
    assert block.execute({"flow": values}) is None and block.error_kind == "invalid"
    assert not list(tmp_path.iterdir())


def test_pyramid_context_is_preserved_without_rescaling_vectors(tmp_path, samples):
    source, target, vectors = samples
    matrix = np.diag([2.0, 2.0, 1.0]).tolist()
    source.metadata["spatial"].update(root_size=[56, 40], to_root=matrix)
    source.metadata["pyramid"] = {"level": 1, "largest_size": [56, 40], "to_largest": matrix}
    _, saved = save(tmp_path, vectors, source, target)
    reader = LoadFlowFile(
        parameters={"path": saved.outputs["path"], "metadata_mode": "verify_images"}
    )
    loaded = execute(reader, {"source": source, "target": target})
    assert loaded.payloads["flow"]["pyramid"] == source.metadata["pyramid"]
    np.testing.assert_array_equal(loaded.outputs["flow"].vectors, vectors)


def test_required_manifest_and_both_anchors_are_enforced(tmp_path, samples):
    source, target, vectors = samples
    _, bare = save(tmp_path, vectors, metadata_mode="none")
    reader = LoadFlowFile(parameters={"path": bare.outputs["path"], "metadata_mode": "required"})
    assert reader.execute({}) is None and reader.error_kind == "invalid"
    reader.parameters["metadata_mode"] = "ignore"
    unbound = execute(reader, {"source": source, "target": target})
    assert unbound.outputs["info"]["images_verified"] is False
    assert "flow_association" not in unbound.payloads["flow"]
    block = SaveFlowFile(
        parameters={"path": str(tmp_path / "unpaired.flo"), "metadata_mode": "require_images"}
    )
    assert block.execute({"flow": vectors, "source": source}) is None
    assert not (tmp_path / "unpaired.flo").exists()


@pytest.mark.parametrize("operation", ["write_false", "write_error", "read_error"])
def test_native_failures_are_contained_and_staging_is_removed(
    tmp_path, samples, monkeypatch, operation
):
    import opencv_dataflow.flow_files as module

    paths = []
    original = module.TemporaryDirectory

    def track(*args, **kwargs):
        value = original(*args, **kwargs)
        paths.append(Path(value.name))
        return value

    monkeypatch.setattr(module, "TemporaryDirectory", track)

    def fail(*args):
        if operation == "write_false":
            return False
        raise cv2.error("controlled native failure")

    if operation.startswith("write"):
        monkeypatch.setattr(cv2, "writeOpticalFlow", fail)
        block = SaveFlowFile(parameters={"path": str(tmp_path / "output.flo")})
        result = block.execute({"flow": samples[2]})
        assert not (tmp_path / "output.flo").exists()
    else:
        path = tmp_path / "input.flo"
        path.write_bytes(struct.pack("<4sii", b"PIEH", 28, 20) + samples[2].tobytes())
        monkeypatch.setattr(cv2, "readOpticalFlow", fail)
        block = LoadFlowFile(parameters={"path": str(path)})
        result = block.execute({})
    assert result is None and block.state == "error" and not block.current_emissions
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"PIEH",
        struct.pack("<4sii", b"HEIP", 2, 2),
        struct.pack("<4sii", b"PIEH", -1, 2),
        struct.pack("<4sii", b"PIEH", MAX_PIXELS + 1, 1),
        struct.pack("<4sii", b"PIEH", 2, 2) + b"\0" * 31,
        struct.pack("<4sii", b"PIEH", 2, 2) + b"\0" * 33,
    ],
)
def test_invalid_file_is_rejected_before_native_decode(tmp_path, monkeypatch, data):
    def forbidden(*args):
        raise AssertionError("native decode must not run")

    monkeypatch.setattr(cv2, "readOpticalFlow", forbidden)
    path = tmp_path / "invalid.flo"
    path.write_bytes(data)
    block = LoadFlowFile(parameters={"path": str(path)})
    assert block.execute({}) is None and block.error_kind == "invalid"


def test_existing_files_and_handled_publication_races_preserve_other_owners(tmp_path, monkeypatch):
    import opencv_dataflow.flow_files as module

    path = tmp_path / "output.flo"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        publication(path, b"new", None)
    assert path.read_bytes() == b"original"
    path.unlink()
    original_link = module.os.link

    def race(source, target):
        if Path(target) == path:
            path.write_bytes(b"other owner")
            raise FileExistsError("controlled race")
        return original_link(source, target)

    monkeypatch.setattr(module.os, "link", race)
    with pytest.raises(FileExistsError):
        publication(path, b"new", {"test": True})
    assert path.read_bytes() == b"other owner"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.flo"]


def test_unique_names_preview_controls_xml_and_save_as(app, tmp_path, samples):
    from opencv_dataflow.app import MainWindow

    source, target, vectors = samples
    writer = SaveFlowFile(parameters={"path": str(tmp_path / "out.flo"), "unique_name": True})
    writer.set_graphics_mode("off")
    first = execute(writer, {"flow": vectors})
    assert first.outputs["visualization"] is NO_VALUE and not first.graphics
    writer.exposed_outputs.append("visualization")
    second = execute(writer, {"flow": vectors})
    assert first.outputs["path"] != second.outputs["path"]
    assert second.outputs["visualization"].shape == (20, 28, 3)
    reader = LoadFlowFile(parameters={"path": Path(first.outputs["path"]).name})
    graph = Graph()
    graph.add(reader)
    graph.bind_project_dir(tmp_path)
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    graph.save(destination / "project.json")
    assert execute(reader, {}).outputs["info"]["size"] == [28, 20]
    xml = tmp_path / "file.output.flow_file.xml"
    save_block_xml(writer, xml)
    assert read_block_xml(xml)["parameters"] == writer.parameters
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(reader)
        for name in reader.parameter_choices:
            assert isinstance(window.inspector.fields[name], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
