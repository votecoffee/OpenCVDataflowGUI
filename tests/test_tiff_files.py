"""Exercise multipage TIFF replay with native fixtures and saved projects."""

import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.runner import main


@pytest.fixture
def stack(tmp_path):
    path = tmp_path / "pÃ¡ginas.tiff"
    pages = [np.full((8 + i, 12), 1000 * (i + 1), np.uint16) for i in range(5)]
    ok, encoded = cv2.imencodemulti(".tiff", pages)
    assert ok
    path.write_bytes(encoded.tobytes())
    return path


def test_tiff_replay_resumes_unicode_path_and_preserves_depth(stack):
    registry = builtin_registry()
    parameters = {"path": str(stack), "start_page": 0, "stride": 2}
    reader = registry.create("input.tiff_file", parameters=parameters)
    resume = registry.create("input.tiff_file", parameters=parameters)
    first = reader.execute({})
    assert first is not None, reader.error
    reader.close()
    resume.copy_runtime_from(reader)
    for index, result in zip(
        (0, 2, 4), [first, resume.execute({}), resume.execute({})], strict=True
    ):
        assert result is not None, resume.error
        assert result.outputs["info"]["page_index"] == index
        pixels = result.outputs["image"]
        assert pixels.dtype == np.uint16 and pixels.shape == (8 + index, 12)
        assert np.all(pixels == (index + 1) * 1000)
        assert result.outputs["ticket"]["value"] == f"{stack.name}#page-{index:06d}"
        assert result.payloads["image"]["spatial"]["root_id"].endswith(f":page:{index}")
    assert resume.exhausted
    assert resume.execute({}).outputs["image"] is NO_VALUE
    resume.reset_runtime()
    assert resume.execute({}).outputs["info"]["page_index"] == 0


@pytest.mark.parametrize("end", [-1, 5])
def test_tiff_loop_wraps_and_assigns_distinct_ticket_instances(stack, end):
    reader = builtin_registry().create(
        "input.tiff_file",
        parameters={
            "path": str(stack),
            "start_page": 2,
            "end_page": end,
            "stride": 2,
            "repeat": "loop",
        },
    )
    results = [reader.execute({}) for _ in range(5)]
    assert all(result is not None for result in results), reader.error
    assert [result.outputs["info"]["page_index"] for result in results] == [2, 4, 2, 4, 2]
    assert len({result.outputs["ticket"]["session"] for result in results}) == 5
    reader.close()


def test_tiff_headless_all_images_excludes_an_empty_eof_iteration(stack, tmp_path):
    graph = Graph()
    graph.add(
        builtin_registry().create(
            "input.tiff_file", parameters={"path": stack.name, "path_base": "project"}
        )
    )
    project, receipt = tmp_path / "replay.json", tmp_path / "report.json"
    graph.save(project)
    assert main([str(project), "--all-images", "--report", str(receipt)]) == 0
    runs = json.loads(receipt.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 5
    assert [run["blocks"][0]["outputs"]["info"]["page_index"] for run in runs] == list(range(5))


def test_tiff_invalid_first_page_or_explicit_missing_page_reports_error(stack):
    reader = builtin_registry().create(
        "input.tiff_file", parameters={"path": str(stack), "start_page": 9}
    )
    assert reader.execute({}) is None
    assert "Cannot decode TIFF page 9" in reader.error
    reader.parameters.update(start_page=4, end_page=6)
    assert reader.execute({}).outputs["info"]["page_index"] == 4
    assert reader.execute({}) is None
    assert "Cannot decode TIFF page 5" in reader.error


def tiff_group(*images):
    return FlowGroup(
        tuple(ValuePacket(image, {}) for image in images),
        tuple(f"page_{i}" for i in range(len(images))),
    )


@pytest.mark.parametrize("compression", ["none", "lzw", "deflate", "packbits"])
def test_tiff_writer_round_trips_order_mixed_depth_and_color(tmp_path, compression):
    path = tmp_path / "récapitulatif.tiff"
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    rgb = ImageArray(np.tile(np.uint16([52000, 22000, 11000]), (5, 7, 1)), "rgb")
    block = builtin_registry().create(
        "output.tiff_file",
        parameters={
            "path": str(path),
            "compression": compression,
            "x_dpi": 300,
            "y_dpi": 300,
        },
    )
    result = block.execute({"pages": tiff_group(gray, rgb)})
    assert result is not None, block.error
    assert result.outputs["page_count"] == 2
    assert result.outputs["byte_count"] == path.stat().st_size
    ok, decoded = cv2.imdecodemulti(
        np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert ok and len(decoded) == 2
    np.testing.assert_array_equal(decoded[0], gray)
    np.testing.assert_array_equal(decoded[1], np.asarray(rgb)[..., ::-1])
    original = path.read_bytes()
    assert block.execute({"pages": tiff_group(gray)}) is None
    assert "existing TIFF files" in block.error and path.read_bytes() == original


def test_tiff_writer_refuses_replacement_even_if_path_appears_during_encoding(
    tmp_path, monkeypatch
):
    path = tmp_path / "contended.tif"
    native = cv2.imencodemulti

    def concurrent_writer(*args, **kwargs):
        path.write_bytes(b"other writer")
        return native(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencodemulti", concurrent_writer)
    block = builtin_registry().create("output.tiff_file", parameters={"path": str(path)})
    assert block.execute({"pages": tiff_group(np.zeros((3, 3), np.uint8))}) is None
    assert path.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [path]


def test_tiff_writer_replays_an_ordered_batch_through_normal_graph_execution(stack, tmp_path):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.tiff_file", parameters={"path": stack.name, "path_base": "project", "stride": 2}
        )
    )
    batch = graph.add(registry.create("flow.batch", parameters={"size": 3}))
    writer = graph.add(
        registry.create(
            "output.tiff_file", parameters={"path": "export.tiff", "path_base": "project"}
        )
    )
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.connect(batch.id, "group", writer.id, "pages")
    project = tmp_path / "write_tiff.json"
    graph.save(project)
    assert main([str(project), "--all-images"]) == 0
    readback = registry.create(
        "input.tiff_file", parameters={"path": str(tmp_path / "export.tiff")}
    )
    images = [readback.execute({}).outputs["image"] for _ in range(3)]
    assert [int(image[0, 0]) for image in images] == [1000, 3000, 5000]
    assert readback.exhausted


def test_tiff_writer_rejects_invalid_batch_and_predictor_before_publication(tmp_path):
    path = tmp_path / "invalid.tiff"
    block = builtin_registry().create("output.tiff_file", parameters={"path": str(path)})
    assert block.execute({"pages": tiff_group(np.ones((5, 5), np.float32))}) is None
    assert block.error_kind == "invalid"
    block.parameters.update(compression="packbits", predictor="horizontal")
    assert block.execute({"pages": tiff_group(np.ones((5, 5), np.uint8))}) is None
    assert "Horizontal prediction" in block.error
    assert not path.exists()
