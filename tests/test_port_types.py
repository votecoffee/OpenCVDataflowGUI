"""Static compatibility and runtime conversions through actual graph wires."""

import csv

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.canvas import CanvasScene
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import DepthArray
from opencv_dataflow.toolblocks import Binding, ToolBlock


class ValueSource(Block):
    def __init__(self, kind, value):
        self.outputs = (Port("value", kind),)
        super().__init__()
        self.value = value

    def process(self, inputs):
        return Result({"value": self.value}, payloads={"value": {"fixture": "unchanged"}})


class Receiver(Block):
    def __init__(self, kind):
        self.inputs = (Port("value", kind),)
        self.outputs = (Port("result", "any"),)
        super().__init__()

    def process(self, inputs):
        return Result({"result": inputs["value"]})


@pytest.mark.parametrize(
    "source_kind,target_kind,value,expected",
    [
        ("integer", "number", 42, 42),
        ("integer", "float", np.uint64(2**53), float(2**53)),
        ("number", "integer", np.float32(-7.0), -7),
        ("number", "data", np.uint64(2**63 + 1), 2**63 + 1),
        ("data", "float", 4, 4.0),
        ("data", "boolean", True, True),
        ("boolean", "data", np.bool_(False), False),
        ("data", "text", "ABC", "ABC"),
    ],
)
def test_scalar_conversion_preserves_value_and_payload(source_kind, target_kind, value, expected):
    graph = Graph()
    source = graph.add(ValueSource(source_kind, value))
    receiver = graph.add(Receiver(target_kind))
    graph.connect(source.id, "value", receiver.id, "value")
    graph.run()
    assert receiver.state == "pass", receiver.error
    result = receiver.last_result.outputs["result"]
    assert type(result) is type(expected) and result == expected
    assert receiver.input_payloads["value"]["fixture"] == "unchanged"
    assert type(source.last_result.outputs["value"]) is type(value)


@pytest.mark.parametrize(
    "source_kind,target_kind,value,message",
    [
        ("number", "integer", 2.5, "conversion must be exact"),
        ("integer", "float", np.uint64(2**53 + 1), "exactly representable"),
        ("data", "number", {"count": 3}, "requires a number"),
        ("data", "integer", "42", "requires an integer"),
        ("data", "boolean", 1, "requires a boolean"),
    ],
)
def test_value_dependent_wire_errors_are_local(source_kind, target_kind, value, message):
    graph = Graph()
    source = graph.add(ValueSource(source_kind, value))
    receiver = graph.add(Receiver(target_kind))
    graph.connect(source.id, "value", receiver.id, "value")
    graph.run()
    assert source.state == "pass"
    assert receiver.state == "error" and message in receiver.error
    assert receiver.current_emissions == {}


def test_depth_image_connections_preserve_samples_and_check_plain_images():
    graph = Graph()
    depth = DepthArray([[1.25, np.nan]], units="mm")
    source = graph.add(ValueSource("depth", depth))
    image = graph.add(Receiver("image"))
    graph.connect(source.id, "value", image.id, "value")
    graph.run()
    observed = image.last_result.outputs["result"]
    assert isinstance(observed, DepthArray) and observed.units == "mm"
    np.testing.assert_array_equal(observed, depth)

    graph = Graph()
    source = graph.add(ValueSource("image", np.array([[4096]], dtype=np.uint16)))
    receiver = graph.add(Receiver("depth"))
    graph.connect(source.id, "value", receiver.id, "value")
    graph.run()
    assert receiver.state == "pass"
    assert receiver.last_result.outputs["result"][0, 0] == 4096
    source.value = np.zeros((3, 3, 3), np.uint8)
    graph.run()
    assert receiver.state == "error" and "requires a 2D" in receiver.error


def test_canvas_rejects_unrelated_kinds_and_describes_checked_wire(app):
    graph = Graph()
    source = graph.add(ValueSource("number", 4.0))
    accepted = graph.add(Receiver("integer"))
    rejected = graph.add(Receiver("image"))
    scene = CanvasScene(graph)
    warnings = []
    scene.warning.connect(warnings.append)
    start = scene.nodes[source.id].sockets[True, "value"]
    scene.start_wire(start)
    scene.finish_wire(scene.nodes[rejected.id].sockets[False, "value"])
    assert warnings == ["Cannot connect number to image"]
    assert not graph.connections

    scene.start_wire(scene.nodes[accepted.id].sockets[False, "value"])
    scene.finish_wire(start)
    assert len(graph.connections) == 1
    assert "number → integer · checked scalar conversion" in scene.wires[0].toolTip()
    assert scene.wires[0].visual_type == "number"


def test_toolblock_and_injected_inputs_use_the_same_conversion():
    graph = Graph()
    inner = Graph()
    receiver = inner.add(Receiver("integer"))
    tool = graph.add(ToolBlock(
        inner, {"input": Binding(receiver.id, "value")}, {"out": Binding(receiver.id, "result")}
    ))
    graph.run(injected={(tool.id, "input"): 9.0})
    assert tool.state == "pass", tool.error
    assert type(tool.last_result.outputs["out"]) is int
    assert tool.last_result.outputs["out"] == 9


def test_saved_builtin_number_to_data_wire_writes_a_scalar(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.demo"))
    inspect = graph.add(registry.create("inspect.contours"))
    path = tmp_path / "measurements.csv"
    writer = graph.add(registry.create("output.results", parameters={"path": str(path)}))
    graph.connect(source.id, "image", inspect.id, "image")
    graph.connect(inspect.id, "count", writer.id, "data")
    project = tmp_path / "job.json"
    graph.save(project)
    restored = Graph.load(project, registry)
    restored.run()
    assert restored.blocks[writer.id].state == "pass"
    with path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["payload"], [str(restored.blocks[inspect.id].last_result.outputs["count"])]
        ]
