"""GUI-independent typed graph contracts, execution, and project persistence."""

from __future__ import annotations

import copy
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class Port:
    name: str
    kind: str = "image"
    required: bool = True


@dataclass
class Result:
    outputs: dict[str, Any] = field(default_factory=dict)
    graphics: dict[str, np.ndarray] = field(default_factory=dict)
    passed: bool = True
    score: float | None = None
    message: str = ""


class Block:
    """Subclass and override process; runtime owns history and failure handling.

    Inputs are private snapshots: custom code may modify them without changing
    upstream history or a sibling branch. Keep only the latest run in memory.
    """

    type_id: ClassVar[str] = "base"
    title: ClassVar[str] = "Block"
    category: ClassVar[str] = "Custom"
    inputs: ClassVar[tuple[Port, ...]] = ()
    outputs: ClassVar[tuple[Port, ...]] = ()
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, block_id=None, parameters=None):
        self.id = block_id or uuid.uuid4().hex
        self.name = self.title
        self.parameters = copy.deepcopy(self.defaults)
        if parameters:
            unknown = parameters.keys() - self.defaults.keys()
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            self.parameters.update(parameters)
        self.exposed_inputs = [p.name for p in self.inputs]
        self.exposed_outputs = [p.name for p in self.outputs]
        self.position = [0.0, 0.0]
        self.thumbnail = True
        self.preview_output = ""
        self.clear_result()

    def clear_result(self):
        self.last_inputs: dict[str, Any] = {}
        self.last_result: Result | None = None
        self.last_output_images: dict[str, np.ndarray] = {}
        self.last_graphics: dict[str, np.ndarray] = {}
        self.state = "idle"
        self.error = ""
        self.elapsed_ms = 0.0

    def process(self, inputs: dict[str, Any]) -> Result:
        raise NotImplementedError

    def execute(self, values, cancelled=lambda: False):
        self.clear_result()
        self._cancelled = cancelled
        started = time.perf_counter()
        try:
            self.last_inputs = copy.deepcopy(values)
            for port in self.inputs:
                if port.required and port.name not in values:
                    raise ValueError(f"Missing input: {port.name}")
                if port.name in values:
                    validate_value(port, values[port.name])
            result = self.process(copy.deepcopy(self.last_inputs))
            if not isinstance(result, Result):
                raise TypeError("process() must return Result")
            if result.score is not None and (
                not math.isfinite(result.score) or not 0 <= result.score <= 1
            ):
                raise ValueError("Score must be finite and between 0 and 1")
            expected = {p.name for p in self.outputs}
            if result.outputs.keys() != expected:
                raise ValueError(f"Result outputs must match {sorted(expected)}")
            for port in self.outputs:
                validate_value(port, result.outputs[port.name])
            for name, value in result.graphics.items():
                validate_value(Port(name), value)
            self.last_result = copy.deepcopy(result)
            self.last_output_images = {
                p.name: self.last_result.outputs[p.name] for p in self.outputs if p.kind == "image"
            }
            self.last_graphics = self.last_result.graphics
            self.state = "pass" if result.passed else "fail"
        except Exception as exc:
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self.last_result

    def close(self):
        """Release optional native resources, on the execution thread."""

    def serialize_extra(self):
        return {}

    def copy_results_from(self, other):
        for name in (
            "last_inputs",
            "last_result",
            "last_output_images",
            "last_graphics",
            "state",
            "error",
            "elapsed_ms",
        ):
            setattr(self, name, copy.deepcopy(getattr(other, name)))


def validate_value(port, value):
    if port.kind == "image":
        if not isinstance(value, np.ndarray) or value.size == 0 or value.ndim not in (2, 3):
            raise TypeError(f"{port.name} requires a nonempty image array")
    elif port.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{port.name} requires a number")
    elif port.kind == "data":
        json.dumps(value, allow_nan=False)
    elif port.kind != "any":
        raise ValueError(f"Unknown port type: {port.kind}")


class Registry:
    def __init__(self):
        self.types: dict[str, type[Block]] = {}

    def register(self, block_type):
        if not issubclass(block_type, Block) or block_type.type_id in self.types:
            raise ValueError("Block must be a subclass with a unique type_id")
        self.types[block_type.type_id] = block_type

    def create(self, type_id, **kwargs):
        if type_id not in self.types:
            raise ValueError(f"Unavailable block type: {type_id}. Load its plugin first.")
        return self.types[type_id](**kwargs)


@dataclass(frozen=True)
class Connection:
    source: str
    output: str
    target: str
    input: str


class Graph:
    def __init__(self):
        self.blocks: dict[str, Block] = {}
        self.connections: list[Connection] = []

    def add(self, block):
        if block.id in self.blocks:
            raise ValueError("Duplicate block id")
        self.blocks[block.id] = block
        return block

    def remove(self, block_id):
        self.connections = [c for c in self.connections if block_id not in (c.source, c.target)]
        del self.blocks[block_id]
        self.invalidate()

    def invalidate(self):
        for block in self.blocks.values():
            block.clear_result()

    def connect(self, source, output, target, input):
        if source == target:
            raise ValueError("A block cannot connect to itself")
        src, dst = self.blocks[source], self.blocks[target]
        out = next((p for p in src.outputs if p.name == output), None)
        inp = next((p for p in dst.inputs if p.name == input), None)
        if out is None or inp is None:
            raise ValueError("Unknown port")
        if output not in src.exposed_outputs or input not in dst.exposed_inputs:
            raise ValueError("Expose both ports in the inspector before connecting")
        if out.kind != inp.kind and inp.kind != "any":
            raise ValueError(f"Cannot connect {out.kind} to {inp.kind}")
        if any(c.target == target and c.input == input for c in self.connections):
            raise ValueError("Input already connected; delete its wire first")
        connection = Connection(source, output, target, input)
        self.connections.append(connection)
        try:
            self.ordered()
        except ValueError:
            self.connections.remove(connection)
            raise
        self.invalidate()

    def ordered(self):
        pending = {key: set() for key in self.blocks}
        for c in self.connections:
            pending[c.target].add(c.source)
        order = []
        while pending:
            ready = [key for key, deps in pending.items() if not deps]
            if not ready:
                raise ValueError("Cycles are not supported")
            order.extend(ready)
            for key in ready:
                del pending[key]
            for deps in pending.values():
                deps.difference_update(ready)
        return order

    def run(self, cancelled=lambda: False, injected=None):
        self.invalidate()
        injected = injected or {}
        for (target, name), value in injected.items():
            block = self.blocks[target]
            port = next((p for p in block.inputs if p.name == name), None)
            if port is None or any(
                c.target == target and c.input == name for c in self.connections
            ):
                raise ValueError("Invalid or internally connected public input")
            validate_value(port, value)
        for key in self.ordered():
            block = self.blocks[key]
            if cancelled():
                block.state = "skipped"
                block.error = "Run stopped"
                continue
            incoming = [c for c in self.connections if c.target == key]
            if any(self.blocks[c.source].state in ("error", "skipped") for c in incoming):
                block.state = "skipped"
                block.error = "Upstream block unavailable"
                continue
            values = {
                c.input: self.blocks[c.source].last_result.outputs[c.output] for c in incoming
            }
            values.update(
                {name: value for (target, name), value in injected.items() if target == key}
            )
            block.execute(values, cancelled)

    def close(self):
        for block in self.blocks.values():
            block.close()

    def to_dict(self):
        return {
            "schema_version": 1,
            "blocks": [
                {
                    "id": b.id,
                    "type": b.type_id,
                    "name": b.name,
                    "parameters": b.parameters,
                    "position": b.position,
                    "exposed_inputs": b.exposed_inputs,
                    "exposed_outputs": b.exposed_outputs,
                    "thumbnail": b.thumbnail,
                    "preview_output": b.preview_output,
                    **b.serialize_extra(),
                }
                for b in self.blocks.values()
            ],
            "connections": [asdict(c) for c in self.connections],
        }

    def save(self, path):
        path = Path(path)
        content = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, registry):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, registry)

    @classmethod
    def from_dict(cls, data, registry, _depth=0):
        if _depth > 16:
            raise ValueError("ToolBlock nesting exceeds 16 levels")
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported project schema version")
        graph = cls()
        for item in data["blocks"]:
            if item["type"] == "core.toolblock":
                from .toolblocks import ToolBlock

                block = ToolBlock.from_definition(
                    item["toolblock"], registry, _depth + 1, block_id=item["id"]
                )
            else:
                block = registry.create(
                    item["type"], block_id=item["id"], parameters=item["parameters"]
                )
            block.name = item["name"]
            position = item["position"]
            if len(position) != 2 or not all(math.isfinite(v) for v in position):
                raise ValueError("Invalid node position")
            block.position = position
            for direction in ("inputs", "outputs"):
                exposed = item[f"exposed_{direction}"]
                allowed = {p.name for p in getattr(block, direction)}
                if len(set(exposed)) != len(exposed) or not set(exposed) <= allowed:
                    raise ValueError("Invalid exposed ports")
                setattr(block, f"exposed_{direction}", exposed)
            block.thumbnail = item.get("thumbnail", True)
            block.preview_output = item.get("preview_output", "")
            graph.add(block)
        for c in data["connections"]:
            graph.connect(**c)
        return graph
