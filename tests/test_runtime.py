import csv
import json
import sqlite3
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.plugins import load_plugin


@pytest.fixture
def registry():
    return builtin_registry()


def test_demo_eager_opencv_and_retained_results(registry):
    graph = demo_graph(registry)
    graph.run()
    assert all(b.state == "pass" for b in graph.blocks.values())
    blocks = list(graph.blocks.values())
    original = blocks[0].last_output_images["image"]
    gray = blocks[1].last_output_images["image"]
    np.testing.assert_array_equal(gray, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
    np.testing.assert_array_equal(
        blocks[2].last_output_images["image"], cv2.GaussianBlur(gray, (5, 5), 1.2)
    )
    result = blocks[-1]
    assert result.last_result.outputs["count"] == 2
    assert result.last_result.score == 1
    assert set(result.last_output_images) == {"overlay", "mask"}
    assert "contours" in result.last_graphics
    assert "image" in result.last_inputs


def test_graph_rejects_invalid_wires_and_cycles(registry):
    graph = Graph()
    a, b = [graph.add(registry.create("gapi.blur")) for _ in range(2)]
    graph.connect(a.id, "image", b.id, "image")
    for source, output, target, input in (
        (b.id, "image", a.id, "image"),
        (a.id, "image", a.id, "image"),
        (a.id, "image", b.id, "image"),
        (a.id, "missing", b.id, "image"),
    ):
        with pytest.raises(ValueError):
            graph.connect(source, output, target, input)
    sink = graph.add(registry.create("output.results"))
    with pytest.raises(ValueError, match="image to data"):
        graph.connect(a.id, "image", sink.id, "data")
    assert len(graph.connections) == 1


def test_failure_clears_stale_results_and_skips_descendants(registry):
    graph = demo_graph(registry)
    graph.run()
    blocks = list(graph.blocks.values())
    blocks[2].parameters["kernel"] = 2
    graph.run()
    assert blocks[2].state == "error"
    assert "odd" in blocks[2].error
    assert blocks[-1].state == "skipped"
    assert blocks[-1].last_result is None
    assert not blocks[-1].last_output_images


def test_roundtrip_and_schema_rejection(tmp_path, registry):
    graph = demo_graph(registry)
    block = list(graph.blocks.values())[-1]
    block.name = "Inspection A"
    block.preview_output = "mask"
    block.exposed_outputs = ["overlay", "mask"]
    path = tmp_path / "project.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    assert restored.to_dict() == graph.to_dict()
    restored.run()
    assert restored.blocks[block.id].state == "pass"
    data = json.loads(path.read_text())
    data["schema_version"] = 42
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        Graph.load(path, registry)


def test_custom_plugin_and_branch_isolation(registry):
    load_plugin("examples/custom_invert.py", registry)
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    invert = graph.add(registry.create("example.invert"))
    graph.connect(source.id, "image", invert.id, "image")
    graph.run()
    image = source.last_output_images["image"]
    np.testing.assert_array_equal(invert.last_output_images["image"], cv2.bitwise_not(image))
    invert.last_inputs["image"][:] = 0
    assert image.any()
    assert invert.last_output_images["original"].any()


@pytest.mark.parametrize("format", ["csv", "xml", "sqlite"])
def test_result_sinks(tmp_path, registry, format):
    path = tmp_path / f"results.{format}"
    block = registry.create("output.results", parameters={"path": str(path), "format": format})
    value = {"count": 2, "areas": [10.0, 20.0]}
    block.execute({"data": value})
    assert block.state == "pass", block.error
    if format == "csv":
        with path.open(newline="") as stream:
            assert json.loads(list(csv.DictReader(stream))[0]["payload"]) == value
    elif format == "xml":
        assert json.loads(ET.parse(path).getroot().findtext("payload")) == value
        block.execute({"data": value})
        assert block.state == "error"  # never silently overwrite existing XML
    else:
        with sqlite3.connect(path) as db:
            assert json.loads(db.execute("SELECT payload FROM results").fetchone()[0]) == value


def test_native_canny(registry):
    image = np.zeros((64, 64), np.uint8)
    image[10:40, 10:40] = 255
    block = registry.create("gapi.canny")
    block.execute({"image": image})
    assert block.state == "pass", block.error
    np.testing.assert_array_equal(block.last_output_images["image"], cv2.Canny(image, 60, 150))


def test_bad_result_and_missing_input():
    class Bad(Block):
        inputs = (Port("image"),)

        def process(self, inputs):
            return Result(score=float("nan"))

    block = Bad()
    block.execute({})
    assert "Missing input" in block.error
    block.execute({"image": np.ones((2, 2), np.uint8)})
    assert "Score" in block.error


def test_stop_skips_unexecuted_blocks(registry):
    graph = demo_graph(registry)
    graph.run(cancelled=lambda: True)
    assert all(b.state == "skipped" for b in graph.blocks.values())


def test_plugin_dataclass_import_and_atomic_registration(tmp_path, registry):
    path = tmp_path / "extension.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from opencv_dataflow.core import Block\n"
        "@dataclass\n"
        "class Settings:\n"
        "    value: int = 1\n"
        "class Custom(Block):\n"
        "    type_id = 'test.dataclass'\n"
        "def register(registry):\n"
        "    registry.register(Custom)\n"
    )
    load_plugin(path, registry)
    assert "test.dataclass" in registry.types
    before = dict(registry.types)
    path.write_text(
        path.read_text().replace("test.dataclass", "test.partial")
        + "    raise ValueError('broken registration')\n"
    )
    with pytest.raises(ValueError, match="broken registration"):
        load_plugin(path, registry)
    assert registry.types == before


def test_image_writer_preserves_existing_file(tmp_path, registry):
    path = tmp_path / "capture.png"
    image = np.full((20, 30, 3), 123, np.uint8)
    block = registry.create("output.image", parameters={"path": str(path)})
    block.execute({"image": image})
    assert block.state == "pass", block.error
    before = path.read_bytes()
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(before, np.uint8), 1), image)
    block.execute({"image": np.zeros_like(image)})
    assert block.state == "error"
    assert path.read_bytes() == before


def test_camera_handle_reuse_and_release(monkeypatch, registry):
    handles = []

    class Capture:
        def __init__(self, device):
            self.device, self.released, self.reads = device, False, 0
            handles.append(self)

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            return True, np.full((4, 4, 3), self.device, np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    block = registry.create("input.camera")
    block.execute({})
    block.execute({})
    assert len(handles) == 1 and handles[0].reads == 2
    block.parameters["device"] = 1
    block.execute({})
    assert handles[0].released and len(handles) == 2
    assert not handles[1].released
    block.close()
    assert handles[1].released
