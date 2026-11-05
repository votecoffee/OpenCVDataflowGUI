"""Verify real video decoding, graph completion, and output file ownership."""

import json

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.runner import main


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "source.avi"
    writer = builtin_registry().create(
        "output.video", parameters={"path": str(path), "backend": "opencv_mjpeg", "fps": 10}
    )
    try:
        for index in range(6):
            image = np.full((40, 60, 3), 20 + index * 30, np.uint8)
            result = writer.execute({"image": image})
            assert result is not None, writer.error
            assert result.outputs["info"]["frames_submitted"] == index + 1
    finally:
        writer.close()
    return path


def test_video_source_resumes_without_open_handle_and_stops_at_eof(clip):
    registry = builtin_registry()
    parameters = {"path": str(clip), "start_frame": 1, "stride": 2, "color_format": "gray"}
    source = registry.create("input.video_file", parameters=parameters)
    resume = registry.create("input.video_file", parameters=parameters)
    try:
        first = source.execute({})
        assert first is not None, source.error
        source.close()
        resume.copy_runtime_from(source)
        results = [first, resume.execute({}), resume.execute({})]
        for index, result in zip((1, 3, 5), results, strict=True):
            assert result is not None, resume.error
            assert result.outputs["image"].shape == (40, 60)
            assert result.outputs["image"].mean() == pytest.approx(20 + index * 30, abs=3)
            info = result.outputs["info"]
            assert info["frame_index"] == index
            assert info["timestamp_ms"] == pytest.approx(index * 100)
            assert result.payloads["image"]["video_frame"] == info
            assert result.payloads["image"]["spatial"]["image_size"] == [60, 40]
            assert result.outputs["ticket"]["value"] == f"source.avi#{index:06d}"
        assert resume.exhausted
        assert resume.execute({}).outputs["image"] is NO_VALUE
    finally:
        source.close()
        resume.close()


def test_loop_wraps_range_and_preserves_distinct_ticket_instances(clip):
    reader = builtin_registry().create(
        "input.video_file",
        parameters={"path": str(clip), "start_frame": 2, "end_frame": 4, "repeat": "loop"},
    )
    try:
        results = [reader.execute({}) for _ in range(5)]
        assert [result.outputs["info"]["frame_index"] for result in results] == [2, 3, 2, 3, 2]
        tickets = [result.outputs["ticket"] for result in results]
        assert len({ticket["session"] for ticket in tickets}) == 5
        assert not reader.finite_source
    finally:
        reader.close()


def test_video_writer_keeps_existing_path_and_rejects_frame_size_changes(tmp_path):
    path = tmp_path / "output.avi"
    path.write_bytes(b"existing content")
    block = builtin_registry().create(
        "output.video", parameters={"path": str(path), "backend": "opencv_mjpeg"}
    )
    image = np.zeros((40, 60), np.uint8)
    assert block.execute({"image": image}) is None
    assert "FileExistsError" in block.error
    assert path.read_bytes() == b"existing content"
    block.parameters["path"] = str(tmp_path / "new.avi")
    try:
        assert block.execute({"image": image}).passed
        assert block.execute({"image": image[:, :50]}) is None
        assert "frame dimensions changed" in block.error
        assert block.execute({"image": image}).outputs["info"]["frames_submitted"] == 2
    finally:
        block.close()


@pytest.mark.parametrize("queued", [False, True])
def test_headless_video_all_images_transcodes_exact_range(clip, tmp_path, queued):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.video_file",
            parameters={"path": clip.name, "path_base": "project", "end_frame": 5, "stride": 2},
        )
    )
    writer = graph.add(
        registry.create(
            "output.video",
            parameters={"path": "processed.avi", "path_base": "project", "backend": "opencv_mjpeg"},
        )
    )
    if queued:
        queue = graph.add(
            registry.create("flow.image_queue", parameters={"mode": "background_acquisition"})
        )
        graph.connect(source.id, "image", queue.id, "image")
        source = queue
    graph.connect(source.id, "image", writer.id, "image")
    project, receipt = tmp_path / "transcode.json", tmp_path / "receipt.json"
    graph.save(project)
    assert main([str(project), "--all-images", "--report", str(receipt)]) == 0
    assert len(json.loads(receipt.read_text(encoding="utf-8"))["runs"]) == 3

    readback = registry.create(
        "input.video_file", parameters={"path": str(tmp_path / "processed.avi")}
    )
    try:
        values = [readback.execute({}).outputs["image"].mean() for _ in range(3)]
        np.testing.assert_allclose(values, [20, 80, 140], atol=3)
        assert readback.exhausted
    finally:
        readback.close()
