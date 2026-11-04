"""GUI-independent typed graph contracts, execution, and project persistence."""

from __future__ import annotations

import copy
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .payloads import (
    NO_VALUE,
    FlowGroup,
    ImagePyramid,
    ValuePacket,
    compatible,
    finish_outputs,
    prepare_inputs,
    validate_payload,
)
from .tickets import DEFAULT_TICKET_SETTINGS, DIRECT_CLOCK, TICKET_MODES, TicketClock


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
    payloads: dict[str, dict] = field(default_factory=dict)
    emissions: dict[str, list[ValuePacket]] = field(default_factory=dict)


class Block:
    """Subclass and override process; runtime owns history and failure handling.

    Inputs are private snapshots: custom code may modify them without changing
    upstream history or a sibling branch. Keep only the latest run in memory.
    """

    type_id: ClassVar[str] = "base"
    title: ClassVar[str] = "Block"
    category: ClassVar[str] = "Custom"
    block_type: ClassVar[str] = ""
    subtype: ClassVar[str] = "General"
    inputs: ClassVar[tuple[Port, ...]] = ()
    outputs: ClassVar[tuple[Port, ...]] = ()
    defaults: ClassVar[dict[str, Any]] = {}
    parameter_choices: ClassVar[dict[str, tuple]] = {}
    parameter_help: ClassVar[dict[str, str]] = {}

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
        self.project_dir = None
        self.thumbnail = True
        self.preview_output = ""
        self.input_levels = {}
        self.ticket_source = self.type_id.startswith("input.")
        self.ticket_settings = copy.deepcopy(DEFAULT_TICKET_SETTINGS)
        if self.ticket_source:
            if not any(p.name == "ticket" for p in self.inputs):
                self.inputs = (*self.inputs, Port("ticket", "data", required=False))
            if not any(p.name == "ticket" for p in self.outputs):
                self.outputs = (*self.outputs, Port("ticket", "data"))
            self.exposed_inputs = [p.name for p in self.inputs]
            self.exposed_outputs = [p.name for p in self.outputs]
        self.clear_result()

    def clear_result(self):
        self.last_inputs: dict[str, Any] = {}
        self.input_payloads = {}
        self.last_result: Result | None = None
        self.last_output_images: dict[str, np.ndarray] = {}
        self.last_graphics: dict[str, np.ndarray] = {}
        self.state = "idle"
        self.error = ""
        self.elapsed_ms = 0.0

    def process(self, inputs: dict[str, Any]) -> Result:
        raise NotImplementedError

    def finalize_result(self, result, inputs):
        """Optional result policy for reusable block mixins."""
        return result

    def validate_parameters(self, parameters=None):
        allowed = {p.name for p in self.inputs if p.kind == "image"}
        if not self.input_levels.keys() <= allowed or any(
            level not in ("largest", *map(str, range(9))) for level in self.input_levels.values()
        ):
            raise ValueError("Invalid image input pyramid level selection")
        values = self.parameters if parameters is None else parameters
        for name, choices in self.parameter_choices.items():
            if str(values[name]) not in tuple(map(str, choices)):
                label = name.replace("_", " ").capitalize()
                raise ValueError(f"{label} must be one of: {', '.join(map(str, choices))}")

    def execute(self, values, cancelled=lambda: False):
        self.clear_result()
        self._source_ticket = None
        self._cancelled = cancelled
        started = time.perf_counter()
        try:
            self.validate_parameters()
            values, self.input_payloads = prepare_inputs(self, values)
            self.last_inputs = copy.deepcopy(values)
            for port in self.inputs:
                if port.required and port.name not in values:
                    raise ValueError(f"Missing input: {port.name}")
                if port.name in values:
                    validate_value(port, values[port.name])
            result = self.process(copy.deepcopy(self.last_inputs))
            if not isinstance(result, Result):
                raise TypeError("process() must return Result")
            result = self.finalize_result(result, self.last_inputs)
            if self.ticket_source:
                settings = self.ticket_settings
                if settings.get("mode") not in TICKET_MODES:
                    raise ValueError("Unknown source ticket mode")
                external = self.last_inputs.get(
                    "ticket", settings.get("value") if settings["mode"] == "fixed" else None
                )
                if settings["mode"] == "external" and "ticket" not in self.last_inputs:
                    raise ValueError("External ticket mode requires the ticket input")
                ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(
                    self.id, settings["mode"], external
                )
                result.outputs["ticket"] = ticket
                self._source_ticket = ticket
            result = finish_outputs(self, result)
            if result.score is not None and (
                not math.isfinite(result.score) or not 0 <= result.score <= 1
            ):
                raise ValueError("Score must be finite and between 0 and 1")
            expected = {p.name for p in self.outputs}
            if result.outputs.keys() != expected:
                raise ValueError(f"Result outputs must match {sorted(expected)}")
            for port in self.outputs:
                validate_value(port, result.outputs[port.name])
            if not result.emissions.keys() <= expected:
                raise ValueError("Emissions must name declared outputs")
            for name, packets in result.emissions.items():
                if len(packets) > 1024:
                    raise ValueError("Output exceeds 1024 emissions per invocation")
                for packet in packets:
                    if not isinstance(packet, ValuePacket):
                        raise ValueError("Emissions must be ValuePackets")
                    validate_value(next(p for p in self.outputs if p.name == name), packet)
            for name, value in result.graphics.items():
                validate_value(Port(name), value)
            self.last_result = copy.deepcopy(result)
            self.last_output_images = {
                p.name: self.last_result.outputs[p.name]
                for p in self.outputs
                if isinstance(self.last_result.outputs[p.name], (np.ndarray, ImagePyramid))
            }
            self.last_graphics = self.last_result.graphics
            self.state = "pass" if result.passed else "fail"
            if result.outputs and all(
                result.outputs[name] is NO_VALUE
                for name in getattr(self, "flow_primary", tuple(result.outputs))
            ):
                self.state = "waiting"
        except Exception as exc:
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            if getattr(self, "accept_events", False):
                self.reset_runtime()
        finally:
            self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self.last_result

    def close(self):
        """Release optional native resources, on the execution thread."""

    def reset_runtime(self):
        """Clear transient stream state after configuration changes or explicit reset."""

    def copy_runtime_from(self, other):
        """Copy only transient stream state into a separate execution/display graph."""

    def serialize_extra(self):
        return {}

    def restore_extra(self, data):
        """Restore optional JSON configuration beyond ordinary parameter controls."""

    def copy_results_from(self, other):
        self.copy_runtime_from(other)
        for name in (
            "last_inputs",
            "input_payloads",
            "last_result",
            "last_output_images",
            "last_graphics",
            "state",
            "error",
            "elapsed_ms",
        ):
            setattr(self, name, copy.deepcopy(getattr(other, name)))


def validate_value(port, value):
    if value is NO_VALUE:
        return
    if isinstance(value, ValuePacket):
        validate_payload(value.payload)
        value = value.value
    if isinstance(value, FlowGroup):
        value.__post_init__()
        if port.kind != "any":
            raise TypeError("Use Unbatch or Select group entry before consuming a flow group")
        return
    if port.kind == "pyramid" or (port.kind == "image" and isinstance(value, ImagePyramid)):
        if not isinstance(value, ImagePyramid):
            raise TypeError(f"{port.name} requires an image pyramid")
        value.__post_init__()
        return
    if port.kind == "depth":
        if (
            not isinstance(value, np.ndarray)
            or value.size == 0
            or value.ndim != 2
            or value.dtype not in (np.uint16, np.float32, np.float64)
        ):
            raise TypeError(f"{port.name} requires a 2D uint16 or floating depth image")
    elif port.kind == "image":
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
        block = self.types[type_id](**kwargs)
        if hasattr(block, "bind_registry"):
            block.bind_registry(self)
        return block


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
        self.project_dir = None
        self.clock = TicketClock()

    def bind_project_dir(self, directory):
        self.project_dir = Path(directory).resolve() if directory is not None else None
        for block in self.blocks.values():
            block.project_dir = self.project_dir
            from .toolblocks import ToolBlock

            if isinstance(block, ToolBlock) or getattr(block, "owns_workers", False):
                block.graph.bind_project_dir(self.project_dir)

    def add(self, block):
        if block.id in self.blocks:
            raise ValueError("Duplicate block id")
        self.blocks[block.id] = block
        block.project_dir = self.project_dir
        from .toolblocks import ToolBlock

        if isinstance(block, ToolBlock) or getattr(block, "owns_workers", False):
            block.graph.bind_project_dir(self.project_dir)
        return block

    def remove(self, block_id):
        self.connections = [c for c in self.connections if block_id not in (c.source, c.target)]
        del self.blocks[block_id]
        self.invalidate()

    def invalidate(self):
        self.clock = TicketClock()
        self.clear_results()
        for block in self.blocks.values():
            block.reset_runtime()

    def clear_results(self):
        for block in self.blocks.values():
            block.clear_result()

    def copy_runtime_from(self, other):
        self.clock = other.clock.clone()
        for key, block in self.blocks.items():
            block.copy_runtime_from(other.blocks[key])

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
        if not compatible(out.kind, inp.kind):
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
        from .toolblocks import ToolBlock

        self.clear_results()
        injected = injected or {}
        for (target, name), value in injected.items():
            block = self.blocks[target]
            port = next((p for p in block.inputs if p.name == name), None)
            if port is None or any(
                c.target == target and c.input == name for c in self.connections
            ):
                raise ValueError("Invalid or internally connected public input")
            validate_value(port, value)
        self.ordered()  # Reject cycles before acquiring sources.
        emitted = {}

        def run_block(key):
            block = self.blocks[key]
            block._clock = self.clock
            incoming = [c for c in self.connections if c.target == key]
            if cancelled():
                block.state, block.error = "skipped", "Run stopped"
                return
            if any(self.blocks[c.source].state in ("error", "skipped") for c in incoming):
                block.state, block.error = "skipped", "Upstream block unavailable"
                return
            streams = {c.input: emitted.get((c.source, c.output), []) for c in incoming}
            constants = {name: value for (target, name), value in injected.items() if target == key}
            outputs = {p.name: [] for p in block.outputs}
            if getattr(block, "accept_events", False):
                calls = [
                    {name: values[index]}
                    for index in range(max((len(v) for v in streams.values()), default=0))
                    for name, values in streams.items()
                    if index < len(values)
                ]
                if constants:
                    calls[0:0] = [{name: value} for name, value in constants.items()]
            elif any(
                not values and next(p for p in block.inputs if p.name == name).required
                for name, values in streams.items()
            ):
                calls = []
            else:
                had_streams = bool(streams)
                streams = {name: values for name, values in streams.items() if values}
                length = max((len(v) for v in streams.values()), default=1)
                if had_streams and not streams and not constants:
                    length = 0
                if any(len(v) not in (1, length) for v in streams.values()):
                    block.state, block.error = (
                        "error",
                        "Stream lengths differ; join/batch explicitly",
                    )
                    return
                calls = [
                    {
                        **constants,
                        **{
                            name: values[0 if len(values) == 1 else index]
                            for name, values in streams.items()
                        },
                    }
                    for index in range(length)
                ]
            if len(calls) > 1024:
                block.state, block.error = "error", "Block exceeds 1024 input events per run"
                return
            if not calls:
                block.state = "inactive"
                return
            failed = False
            for values in calls:
                if cancelled():
                    block.state, block.error = "skipped", "Run stopped"
                    return
                result = block.execute(values, cancelled)
                if result is None:
                    return
                failed |= block.state == "fail"
                for port in block.outputs:
                    if port.name in result.emissions:
                        outputs[port.name].extend(result.emissions[port.name])
                    elif result.outputs[port.name] is not NO_VALUE:
                        outputs[port.name].append(
                            ValuePacket(
                                result.outputs[port.name], result.payloads.get(port.name, {})
                            )
                        )
                    if len(outputs[port.name]) > 1024:
                        block.state, block.error = "error", "Output exceeds 1024 events per run"
                        return
            if failed:
                block.state = "fail"
            elif block.state == "waiting" and any(
                outputs[n] for n in getattr(block, "flow_primary", outputs)
            ):
                block.state = "pass"
            # Retain all emissions for ToolBlock boundaries, receipts and UI snapshots.
            block.last_result.emissions = outputs
            for name, packets in outputs.items():
                if packets:
                    block.last_result.outputs[name] = packets[-1].value
                    block.last_result.payloads[name] = packets[-1].payload
                    if isinstance(packets[-1].value, (np.ndarray, ImagePyramid)):
                        block.last_output_images[name] = packets[-1].value
                emitted[key, name] = packets

        pending = set(self.blocks)
        workers = max(
            (
                int(b.parameters.get("workers", 1))
                for b in self.blocks.values()
                if b.type_id == "flow.split"
            ),
            default=1,
        )
        if not 1 <= workers <= 8:
            raise ValueError("Parallel worker count must be 1..8")
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            while pending:
                ready = [
                    key
                    for key in self.blocks
                    if key in pending
                    and not any(c.target == key and c.source in pending for c in self.connections)
                ]
                # Source acquisition remains deterministic and on the owning execution thread.
                parallel = [
                    key
                    for key in ready
                    if executor is not None
                    and not self.blocks[key].ticket_source
                    and not isinstance(self.blocks[key], ToolBlock)
                    and not getattr(self.blocks[key], "owns_workers", False)
                    and not getattr(self.blocks[key], "accept_events", False)
                ]
                futures = [executor.submit(run_block, key) for key in parallel] if executor else []
                for key in ready:
                    if key not in parallel:
                        run_block(key)
                for future in futures:
                    future.result()
                pending.difference_update(ready)
        finally:
            if executor:
                executor.shutdown(wait=True)

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
                    **({"input_levels": b.input_levels} if b.input_levels else {}),
                    **(
                        {"ticket_settings": b.ticket_settings}
                        if b.ticket_source and b.ticket_settings != DEFAULT_TICKET_SETTINGS
                        else {}
                    ),
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
        self.bind_project_dir(path.resolve().parent)

    @classmethod
    def load(cls, path, registry):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, registry, project_dir=Path(path).resolve().parent)

    @classmethod
    def from_dict(cls, data, registry, _depth=0, project_dir=None):
        if _depth > 16:
            raise ValueError("ToolBlock nesting exceeds 16 levels")
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported project schema version")
        graph = cls()
        for item in data["blocks"]:
            if item["type"] == "flow.inspect_each":
                from .parallel_inspection import ParallelInspect

                block = ParallelInspect.from_definition(
                    item["inspection"],
                    registry,
                    _depth + 1,
                    block_id=item["id"],
                    parameters=item["parameters"],
                )
            elif item["type"] == "core.toolblock":
                from .toolblocks import ToolBlock

                block = ToolBlock.from_definition(
                    item["toolblock"], registry, _depth + 1, block_id=item["id"]
                )
            else:
                block = registry.create(
                    item["type"], block_id=item["id"], parameters=item["parameters"]
                )
            block.name = item["name"]
            if "ticket_settings" in item:
                settings = item["ticket_settings"]
                if (
                    not block.ticket_source
                    or not isinstance(settings, dict)
                    or set(settings) != {"mode", "value"}
                    or settings["mode"] not in TICKET_MODES
                ):
                    raise ValueError("Invalid source ticket settings")
                json.dumps(settings, allow_nan=False)
                block.ticket_settings = copy.deepcopy(settings)
            block.restore_extra(item)
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
            levels = item.get("input_levels", {})
            allowed_inputs = {p.name for p in block.inputs if p.kind == "image"}
            if (
                not isinstance(levels, dict)
                or not levels.keys() <= allowed_inputs
                or any(v not in ("largest", *map(str, range(9))) for v in levels.values())
            ):
                raise ValueError("Invalid image input pyramid level selections")
            block.input_levels = dict(levels)
            graph.add(block)
        for c in data["connections"]:
            graph.connect(**c)
        graph.bind_project_dir(project_dir)
        return graph
