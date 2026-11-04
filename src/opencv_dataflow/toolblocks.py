"""Reusable graphs implementing the same Block contract as individual tools."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import Block, Connection, Graph, Port, Result


@dataclass(frozen=True)
class Binding:
    block: str
    port: str


class ToolBlock(Block):
    type_id, title, category = "core.toolblock", "ToolBlock", "ToolBlocks"

    def __init__(self, graph, input_bindings, output_bindings, **kwargs):
        self.graph = graph
        self.input_bindings = dict(input_bindings)
        self.output_bindings = dict(output_bindings)
        self.inputs, self.outputs = self.validate_interface()
        super().__init__(**kwargs)

    def validate_interface(self):
        ports = []
        for direction, bindings in (
            ("inputs", self.input_bindings),
            ("outputs", self.output_bindings),
        ):
            declared = []
            for name, binding in bindings.items():
                if not isinstance(name, str) or not name.strip() or name != name.strip():
                    raise ValueError("Public terminal names must be nonempty and trimmed")
                block = self.graph.blocks.get(binding.block)
                port = next(
                    (p for p in getattr(block, direction, ()) if p.name == binding.port), None
                )
                if port is None:
                    raise ValueError(
                        f"Public {direction[:-1]} '{name}' has a missing internal port"
                    )
                if direction == "inputs" and any(
                    c.target == binding.block and c.input == binding.port
                    for c in self.graph.connections
                ):
                    raise ValueError(f"Public input '{name}' is also connected internally")
                declared.append(Port(name, port.kind, port.required))
            if direction == "inputs" and len(set(bindings.values())) != len(bindings):
                raise ValueError("An internal input can have only one public binding")
            ports.append(tuple(declared))
        self.graph.ordered()
        return tuple(ports)

    def clear_result(self):
        super().clear_result()
        self.graph.invalidate()

    def process(self, inputs):
        if self.validate_interface() != (self.inputs, self.outputs):
            raise ValueError("ToolBlock interface changed; update its public terminals")
        injected = {
            (b.block, b.port): inputs[name]
            for name, b in self.input_bindings.items()
            if name in inputs
        }
        self.graph.run(self._cancelled, injected)
        children = list(self.graph.blocks.values())
        unavailable = [b for b in children if b.state in ("error", "skipped")]
        if unavailable:
            raise RuntimeError("; ".join(f"{b.name}: {b.error}" for b in unavailable))
        outputs = {
            name: self.graph.blocks[b.block].last_result.outputs[b.port]
            for name, b in self.output_bindings.items()
        }
        scores = [b.last_result.score for b in children if b.last_result.score is not None]
        graphics = {
            f"{b.name} [{b.id[:8]}] / {key}": image
            for b in children
            for key, image in b.last_graphics.items()
        }
        return Result(
            outputs,
            graphics,
            passed=all(b.state == "pass" for b in children),
            score=min(scores) if scores else None,
            message=f"{len(children)} tools · {sum(b.state == 'fail' for b in children)} fail",
        )

    def close(self):
        self.graph.close()

    def update_interface(self, inputs, outputs, parent):
        """Rename/remap terminals atomically; preserve the meaning of existing wires."""
        candidate = copy.copy(self)
        candidate.input_bindings, candidate.output_bindings = dict(inputs), dict(outputs)
        candidate.inputs, candidate.outputs = candidate.validate_interface()

        def mapped(name, old, new):
            if name in new and new[name] == old[name]:
                return name
            replacement = next((n for n, binding in new.items() if binding == old[name]), None)
            if replacement is None:
                raise ValueError(f"Disconnect public terminal '{name}' before removing its binding")
            return replacement

        connections = []
        for c in parent.connections:
            output = (
                mapped(c.output, self.output_bindings, outputs) if c.source == self.id else c.output
            )
            input = mapped(c.input, self.input_bindings, inputs) if c.target == self.id else c.input
            source = candidate if c.source == self.id else parent.blocks[c.source]
            target = candidate if c.target == self.id else parent.blocks[c.target]
            out = next(p for p in source.outputs if p.name == output)
            inp = next(p for p in target.inputs if p.name == input)
            if out.kind != inp.kind and inp.kind != "any":
                raise ValueError("Updated interface would break a connected port type")
            connections.append(Connection(c.source, output, c.target, input))
        self.input_bindings, self.output_bindings = dict(inputs), dict(outputs)
        self.inputs, self.outputs = candidate.inputs, candidate.outputs
        self.exposed_inputs = list(inputs)
        self.exposed_outputs = list(outputs)
        if self.preview_output not in outputs:
            self.preview_output = ""
        parent.connections = connections
        parent.invalidate()

    def copy_results_from(self, other):
        super().copy_results_from(other)
        for key, block in self.graph.blocks.items():
            block.copy_results_from(other.graph.blocks[key])

    def definition(self):
        self.validate_interface()
        return {
            "toolblock_version": 1,
            "name": self.name,
            "graph": self.graph.to_dict(),
            "inputs": {n: asdict(b) for n, b in self.input_bindings.items()},
            "outputs": {n: asdict(b) for n, b in self.output_bindings.items()},
        }

    def serialize_extra(self):
        return {"toolblock": self.definition()}

    @classmethod
    def from_definition(cls, data, registry, depth=0, **kwargs):
        if data.get("toolblock_version") != 1:
            raise ValueError("Unsupported ToolBlock definition version")
        graph = Graph.from_dict(copy.deepcopy(data["graph"]), registry, depth)
        block = cls(
            graph,
            {n: Binding(**b) for n, b in data["inputs"].items()},
            {n: Binding(**b) for n, b in data["outputs"].items()},
            **kwargs,
        )
        block.name = data["name"]
        return block

    def export(self, path):
        path = Path(path)
        content = json.dumps(self.definition(), indent=2, allow_nan=False) + "\n"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path, registry):
        return cls.from_definition(json.loads(Path(path).read_text(encoding="utf-8")), registry)


def encapsulate(graph, selected, name, registry):
    """Atomically replace selected nodes with an independent embedded graph.

    Preserve crossing wires, expose unbound inputs and terminal output ports.
    Collapsing arbitrary nonconvex selections can introduce a cycle; reject it.
    """
    selected = set(selected)
    if not selected or not selected <= graph.blocks.keys():
        raise ValueError("Select at least one node")
    if not name.strip():
        raise ValueError("ToolBlock name cannot be empty")
    data = graph.to_dict()
    inner_data = {
        "schema_version": 1,
        "blocks": [b for b in data["blocks"] if b["id"] in selected],
        "connections": [
            c for c in data["connections"] if c["source"] in selected and c["target"] in selected
        ],
    }
    inner = Graph.from_dict(copy.deepcopy(inner_data), registry)
    ins, outs = {}, {}
    input_names, output_names = {}, {}

    def public_name(block, port, bindings):
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{block.name}_{port}").strip("_")
        candidate, suffix = base, 2
        while candidate in bindings:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    for block in inner.blocks.values():
        for port in block.inputs:
            if not any(c.target == block.id and c.input == port.name for c in inner.connections):
                terminal = public_name(block, port.name, ins)
                ins[terminal] = Binding(block.id, port.name)
                input_names[block.id, port.name] = terminal
        for port in block.outputs:
            crosses = any(
                c.source == block.id and c.output == port.name and c.target not in selected
                for c in graph.connections
            )
            consumed = any(
                c.source == block.id and c.output == port.name for c in inner.connections
            )
            if crosses or not consumed:
                terminal = public_name(block, port.name, outs)
                outs[terminal] = Binding(block.id, port.name)
                output_names[block.id, port.name] = terminal
    composite = ToolBlock(inner, ins, outs)
    composite.name = name.strip()
    composite.position = list(next(iter(inner.blocks.values())).position)
    replacements = []
    for c in graph.connections:
        if c.source in selected and c.target in selected:
            continue
        replacements.append(
            Connection(
                composite.id if c.source in selected else c.source,
                output_names[c.source, c.output] if c.source in selected else c.output,
                composite.id if c.target in selected else c.target,
                input_names[c.target, c.input] if c.target in selected else c.input,
            )
        )
    candidate = Graph()
    candidate.blocks = {k: b for k, b in graph.blocks.items() if k not in selected}
    candidate.add(composite)
    candidate.connections = replacements
    candidate.ordered()
    graph.blocks, graph.connections = candidate.blocks, replacements
    graph.invalidate()
    return composite
