"""Exercise multipage TIFF replay with native fixtures and saved projects."""

import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.runner import main


@pytest.fixture
def stack(tmp_path):
    path = tmp_path / "páginas.tiff"
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
