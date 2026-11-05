"""GUI-independent typed graph contracts, execution, and project persistence."""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .activity import ActivityLog, report
from .diagnostics import BlockLogger
from .fail_reasons import normalize_reasons
from .outcomes import (
    INPUT_FAILURE_POLICIES,
    TicketResults,
    publish_result,
    rejected_inputs,
    stamp_result,
)
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
from .recipes import RecipeCatalog, apply_recipe, validate_bindings
from .run_profiles import PROFILES, captures, profile
from .tickets import DEFAULT_TICKET_SETTINGS, DIRECT_CLOCK, TICKET_MODES, TicketClock
from .timing import WorkTiming


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
    event_data: dict = field(default_factory=dict)
    final_judgment: bool | None = None
    fail_reasons: list = field(default_factory=list)


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
        self.input_failure_policies = {}
        self.current_emissions = {}
        self.input_rejected = False
        self.last_rejection = ""
        self._ticket_results = None
        self.recipe_bindings = {}
        self.recipe_model = ""
        self._recipes = None
        self._allow_recipe_activation = True
        self.ticket_source = self.type_id.startswith("input.")
        self.ticket_settings = copy.deepcopy(DEFAULT_TICKET_SETTINGS)
        if self.ticket_source:
            if not any(p.name == "ticket" for p in self.inputs):
                self.inputs = (*self.inputs, Port("ticket", "data", required=False))
            if not any(p.name == "ticket" for p in self.outputs):
                self.outputs = (*self.outputs, Port("ticket", "data"))
            self.exposed_inputs = [p.name for p in self.inputs]
            self.exposed_outputs = [p.name for p in self.outputs]
        self.timing = WorkTiming()
        self.run_profile = PROFILES["debug"]
        self._activity = None
        self.logger = BlockLogger(self)
        self._execution_lock = threading.Lock()
        self._execution_owner = None
        self._work_started = self._work_finished = None
        self.elapsed_ms = 0.0
        self.clear_result()

    def clear_result(self, preserve_history=False):
        self.current_emissions = {}
        self.input_rejected = False
        self.last_rejection = ""
        if preserve_history and "reject_failed" in self.input_failure_policies.values():
            self.state, self.error = "idle", ""
            return
        self.last_inputs: dict[str, Any] = {}
        self.input_payloads = {}
        self.last_result: Result | None = None
        self.last_output_images: dict[str, np.ndarray] = {}
        self.last_graphics: dict[str, np.ndarray] = {}
        self.last_recipe_ref = None
        self.last_recipe_parameters = {}
        self.state = "idle"
        self.error = ""

    def process(self, inputs: dict[str, Any]) -> Result:
        raise NotImplementedError

    def finalize_result(self, result, inputs):
        """Optional result policy for reusable block mixins."""
        return result

    def validate_parameters(self, parameters=None):
        validate_bindings(self, self.recipe_bindings, self.recipe_model)
        if (
            not isinstance(self.input_failure_policies, dict)
            or not self.input_failure_policies.keys() <= {p.name for p in self.inputs}
            or any(v not in INPUT_FAILURE_POLICIES for v in self.input_failure_policies.values())
        ):
            raise ValueError("Invalid input failure policy")
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
        invocation = uuid.uuid4().hex if self.run_profile.activity_spans else None
        self.report_event("input", values=values, invocation=invocation)
        acquired = self._execution_lock.acquire(blocking=False)
        if not acquired:
            invocation = invocation or uuid.uuid4().hex
            self.report_event(
                "blocked", {"reason": "input arrived while block active"}, values, invocation
            )
            if self._execution_owner == threading.get_ident():
                raise RuntimeError("A block cannot recursively execute itself")
            while not self._execution_lock.acquire(timeout=0.05):
                if cancelled():
                    self.report_event("cancelled_wait", values=values, invocation=invocation)
                    return None
            self.report_event("unblocked", values=values, invocation=invocation)
        self._execution_owner = threading.get_ident()
        self._activity_invocation = invocation
        self._activity_started = False
        try:
            return self._execute(values, cancelled)
        finally:
            try:
                if self._activity_started and captures(self, "finish"):
                    observed = dict(values)
                    if self.last_result is not None:
                        observed.update(
                            {
                                f"output:{name}:{index}": packet
                                for name, packets in self.current_emissions.items()
                                for index, packet in enumerate(packets)
                            }
                        )
                    if self._source_ticket is not None:
                        observed["source_ticket"] = self._source_ticket
                    self.report_event(
                        "finish", values=observed, invocation=invocation, at=self._work_finished
                    )
            finally:
                self._execution_owner = None
                self._execution_lock.release()

    def report_event(self, kind, data=None, values=None, invocation=None, at=None):
        """Publish a compact diagnostic marker to the shared job activity log."""
        report(self, kind, data, values, invocation, at)

    @property
    def generate_graphics(self):
        """Custom tools can avoid constructing auxiliary graphics when this is false."""
        return self.run_profile.graphics

    def _execute(self, values, cancelled=lambda: False):
        try:
            rejected = rejected_inputs(self, values)
        except (ValueError, TypeError):
            rejected = []
        if rejected:
            self.state, self.error = "inactive", ""
            self.current_emissions = {}
            self.input_rejected = True
            self.last_rejection = (
                f"Rejected failed input: {', '.join(rejected)}; retained last processed result"
            )
            publish_result(self, values, kind="input_rejected", message=self.last_rejection)
            self.report_event("rejected", {"message": self.last_rejection}, values)
            return None
        self.clear_result(preserve_history=True)
        Block.clear_result(self)
        self._source_ticket = None
        self._cancelled = cancelled
        self._work_started = self.timing.begin()
        self._work_finished = None
        self._diagnostic_exception = None
        self._activity_started = True
        self._activity_start_reported = False
        if not self.ticket_source:
            self._report_activity_start(values)
        configured_parameters = self.parameters
        configured_model = getattr(self, "trained_model", None)
        try:
            self.validate_parameters()
            values, self.input_payloads = prepare_inputs(self, values)
            self.last_inputs = copy.deepcopy(values)
            apply_recipe(self)
            self.validate_parameters()
            for port in self.inputs:
                if port.required and port.name not in values:
                    raise ValueError(f"Missing input: {port.name}")
                if port.name in values:
                    validate_value(port, values[port.name])
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
                self._source_ticket = ticket
                self._report_activity_start(values)
                self.report_event("ticket_assigned", values={"ticket": ticket})
            work_inputs = (
                self.last_inputs
                if (
                    self.run_profile.readonly_input_optimization
                    and type(self).__dict__.get("process_inputs_readonly", False)
                )
                else copy.deepcopy(self.last_inputs)
            )
            result = self.process(work_inputs)
            if not isinstance(result, Result):
                raise TypeError("process() must return Result")
            result = self.finalize_result(result, self.last_inputs)
            if self.ticket_source:
                result.outputs["ticket"] = self._source_ticket
            result = finish_outputs(self, result)
            if not isinstance(result.passed, (bool, np.bool_)) or (
                result.final_judgment is not None
                and not isinstance(result.final_judgment, (bool, np.bool_))
            ):
                raise ValueError("Tool result and final judgment must be boolean")
            result.passed = bool(result.passed)
            result.fail_reasons = normalize_reasons(result.fail_reasons)
            if result.passed and result.final_judgment is not False and result.fail_reasons:
                raise ValueError("Passing results cannot assign fail reasons")
            if result.final_judgment is not None:
                result.final_judgment = bool(result.final_judgment)
            if (
                not isinstance(result.event_data, dict)
                or len(json.dumps(result.event_data, allow_nan=False).encode("utf-8")) > 8192
            ):
                raise ValueError("Result event_data must be a JSON object of at most 8 KiB")
            if result.score is not None and (
                not math.isfinite(result.score) or not 0 <= result.score <= 1
            ):
                raise ValueError("Score must be finite and between 0 and 1")
            if result.score is not None:
                result.score = float(result.score)
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
            if not self.run_profile.graphics:
                result.graphics = {}
            for name, value in result.graphics.items():
                validate_value(Port(name), value)
            result = stamp_result(self, result)
            self.last_result = copy.deepcopy(result)
            self.current_emissions = self.last_result.emissions
            self.last_output_images = {
                p.name: self.last_result.outputs[p.name]
                for p in self.outputs
                if self.run_profile.output_previews
                and isinstance(self.last_result.outputs[p.name], (np.ndarray, ImagePyramid))
            }
            self.last_graphics = self.last_result.graphics
            self.state = "pass" if result.passed else "fail"
            if result.outputs and all(
                result.outputs[name] is NO_VALUE
                for name in getattr(self, "flow_primary", tuple(result.outputs))
            ):
                self.state = "waiting"
            self._finish_work()
            publish_result(
                self,
                {
                    name: ValuePacket(value, self.input_payloads.get(name, {}))
                    for name, value in self.last_inputs.items()
                },
                self.last_result,
                kind="final_judgment" if result.final_judgment is not None else "tool_result",
            )
        except Exception as exc:
            self._finish_work()
            if not self._activity_start_reported:
                self._report_activity_start(values)
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            self.logger.exception(self.error, exc_info=exc, values=values)
            self.last_result = None
            self.last_output_images, self.last_graphics, self.current_emissions = {}, {}, {}
            failure_values = {
                name: value
                if isinstance(value, ValuePacket)
                else ValuePacket(value, self.input_payloads.get(name, {}))
                for name, value in values.items()
            }
            if self._source_ticket is not None:
                failure_values["source_ticket"] = ValuePacket(
                    self._source_ticket, {"ticket": self._source_ticket}
                )
            publish_result(
                self,
                failure_values,
                kind="error",
                message=self.error,
            )
            if getattr(self, "accept_events", False) or getattr(self, "sequence_memory", False):
                self.reset_runtime()
        finally:
            self.parameters = configured_parameters
            if getattr(self, "supports_training", False):
                self.trained_model = configured_model
            self._finish_work()
            if not self.run_profile.input_previews:
                self.last_inputs = {}
        return self.last_result

    def _report_activity_start(self, values):
        observed = dict(values)
        if self._source_ticket is not None:
            observed["source_ticket"] = self._source_ticket
        self.report_event("start", values=observed, at=self._work_started)
        self._activity_start_reported = True

    def _finish_work(self):
        if self._work_finished is None:
            self._work_finished = self.timing.finish()
            self.elapsed_ms = self.timing.last_ms

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
        self.run_profile = other.run_profile
        if not other.run_profile.output_previews:
            from .payloads import describe_value

            for name in (
                "input_payloads",
                "state",
                "error",
                "elapsed_ms",
                "timing",
                "input_rejected",
                "last_rejection",
                "last_recipe_ref",
                "last_recipe_parameters",
            ):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
            self.last_inputs, self.last_output_images, self.last_graphics = {}, {}, {}
            self.current_emissions = {}
            result = other.last_result
            self.last_result = (
                Result(
                    outputs={
                        p.name: describe_value(result.outputs[p.name])
                        for p in self.outputs
                        if p.kind not in ("image", "depth", "pyramid", "any")
                    },
                    passed=result.passed,
                    score=result.score,
                    message=result.message,
                    fail_reasons=copy.deepcopy(result.fail_reasons),
                    final_judgment=result.final_judgment,
                )
                if result
                else None
            )
            return
        for name in (
            "last_inputs",
            "input_payloads",
            "last_result",
            "last_output_images",
            "last_graphics",
            "state",
            "error",
            "elapsed_ms",
            "timing",
            "current_emissions",
            "input_rejected",
            "last_rejection",
            "last_recipe_ref",
            "last_recipe_parameters",
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
        self.ticket_results = TicketResults()
        self.activity = ActivityLog()
        self._activity_prefix = ""
        self._activity_logical_prefix = ""
        self.production_completion = True
        self.validation_sets = []
        self.run_profile = PROFILES["debug"]
        self._event_prefix = ""
        self.recipes = RecipeCatalog()
        self._allow_recipe_activation = True

    def bind_recipes(self, catalog=None, allow_activation=None):
        if catalog is not None and catalog is not self.recipes:
            catalog.merge(self.recipes)
            self.recipes = catalog
        if allow_activation is not None:
            self._allow_recipe_activation = allow_activation
        for block in self.blocks.values():
            block._recipes = self.recipes
            block._allow_recipe_activation = self._allow_recipe_activation
            if hasattr(block, "graph"):
                block.graph.bind_recipes(
                    self.recipes,
                    self._allow_recipe_activation and not getattr(block, "owns_workers", False),
                )

    def bind_activity(self, log=None, prefix=None, logical_prefix=None):
        if log is not None:
            self.activity = log
        self.activity.capture_profile = self.run_profile.name
        if prefix is not None:
            self._activity_prefix = prefix
            self._activity_logical_prefix = prefix if logical_prefix is None else logical_prefix
        prefix = self._activity_prefix
        logical = self._activity_logical_prefix
        for block in self.blocks.values():
            block.run_profile = self.run_profile
            block._activity = self.activity
            block._activity_path = f"{prefix}/{block.id}" if prefix else block.id
            block._activity_logical_path = f"{logical}/{block.id}" if logical else block.id
            binding = (id(self.activity), block._activity_path, block._activity_logical_path)
            if getattr(block, "_activity_binding", None) != binding:
                block._activity_binding = binding
                block.logger.bind(self.activity)
                block.report_event("registered")
            if hasattr(block, "graph"):
                block.graph.set_run_profile(self.run_profile)
                block.graph.bind_activity(
                    self.activity, block._activity_path, block._activity_logical_path
                )

    def set_run_profile(self, value):
        self.run_profile = profile(value)
        self.activity.capture_profile = self.run_profile.name
        for block in self.blocks.values():
            block.run_profile = self.run_profile
            if not self.run_profile.input_previews:
                block.last_inputs = {}
            if not self.run_profile.output_previews:
                block.last_output_images = {}
            if not self.run_profile.graphics:
                block.last_graphics = {}
            if hasattr(block, "graph"):
                block.graph.set_run_profile(self.run_profile)

    def bind_ticket_results(self, consumer=None, prefix=None):
        if consumer is not None:
            self.ticket_results = consumer
        if prefix is not None:
            self._event_prefix = prefix
        prefix = self._event_prefix
        for block in self.blocks.values():
            block._ticket_results = self.ticket_results
            block._event_path = f"{prefix}/{block.id}" if prefix else block.id
            if hasattr(block, "graph"):
                block.graph.bind_ticket_results(self.ticket_results, block._event_path)

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
        if hasattr(block, "graph"):
            self.recipes.merge(block.graph.recipes)
        self.blocks[block.id] = block
        self.bind_recipes()
        self.bind_activity()
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

    def clear_results(self, preserve_history=False):
        for block in self.blocks.values():
            block.clear_result(preserve_history=preserve_history)

    def copy_runtime_from(self, other):
        self.clock = other.clock.clone()
        if self.recipes is not other.recipes:
            self.recipes.copy_runtime_from(other.recipes)
        if self.ticket_results is not other.ticket_results:
            self.ticket_results.restore(other.ticket_results.snapshot())
        for key, block in self.blocks.items():
            block.copy_runtime_from(other.blocks[key])
        self.copy_timing_from(other)

    def copy_timing_from(self, other):
        for key, block in self.blocks.items():
            if key not in other.blocks:
                continue
            block.timing = copy.deepcopy(other.blocks[key].timing)
            block.elapsed_ms = other.blocks[key].elapsed_ms
            if hasattr(block, "graph") and hasattr(other.blocks[key], "graph"):
                block.graph.copy_timing_from(other.blocks[key].graph)

    def restore_captured(self, provided):
        for key, snapshot in provided.items():
            if key not in self.blocks:
                raise ValueError("Captured upstream graph changed; reset flow state")
            block = self.blocks[key]
            for name, value in snapshot.items():
                if name == "_captured_graph":
                    block.graph.restore_captured(value)
                else:
                    setattr(block, name, copy.deepcopy(value))
            if not self.run_profile.input_previews:
                block.last_inputs = {}
            if not self.run_profile.output_previews:
                block.last_output_images = {}
            if not self.run_profile.graphics:
                block.last_graphics = {}
                if block.last_result is not None:
                    block.last_result.graphics = {}

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

    def run(self, cancelled=lambda: False, injected=None, provided=None):
        failed = False
        try:
            return self._run_iteration(cancelled, injected, provided)
        except BaseException:
            failed = True
            raise
        finally:
            for key, block in self.blocks.items():
                if getattr(block, "queue_boundary", False):
                    affected = {key}
                    while True:
                        following = {c.target for c in self.connections if c.source in affected}
                        if following <= affected:
                            break
                        affected.update(following)
                    block.complete(
                        failed=failed
                        or any(
                            self.blocks[k].state in ("fail", "error", "skipped")
                            for k in affected
                            if k in self.blocks
                        )
                    )
                    block.update_metrics()
            if (
                not failed
                and not cancelled()
                and self.production_completion
                and not self._event_prefix
            ):
                self._complete_tickets()

    def _complete_tickets(self):
        from .outcomes import failed_input, packets_in
        from .tickets import ticket_key

        if any(
            b.state == "waiting" and not getattr(b, "validation_skip_output", False)
            for b in self.blocks.values()
        ):
            return
        observed = {}
        for block in self.blocks.values():
            for packets in block.current_emissions.values():
                for packet in packets:
                    for member in packets_in(packet):
                        ticket = member.payload.get("ticket")
                        if ticket is not None and not member.payload.get("inspection", {}).get(
                            "parent_ticket"
                        ):
                            key = ticket_key(ticket)
                            previous = observed.get(
                                key, (ticket, True, member.payload.get("processing_time"))
                            )
                            observed[key] = (
                                ticket,
                                previous[1] and not failed_input(member),
                                member.payload.get("processing_time") or previous[2],
                            )
            if block.state == "error" and getattr(block, "_source_ticket", None):
                ticket = block._source_ticket
                observed[ticket_key(ticket)] = (ticket, False, None)
        for ticket, passed, timing in observed.values():
            self.ticket_results.complete(
                ticket, passed, timing, summary=not self.run_profile.ticket_details
            )

    def _run_iteration(self, cancelled=lambda: False, injected=None, provided=None):
        from .toolblocks import ToolBlock

        if provided is None and not cancelled():
            captures = []
            covered = set()
            for block in self.blocks.values():
                if getattr(block, "queue_boundary", False):
                    with block.condition:
                        pending = block.runtime["pending"]
                        if pending and pending[0][0].snapshots:
                            keys = set(pending[0][0].snapshots)
                            if keys & covered:
                                raise ValueError(
                                    "Queued upstream graphs overlap; queue before splitting"
                                )
                            covered.update(keys)
                            captures.append(block)
            if captures:
                provided = {}
                for block in captures:
                    capture = block.take(cancelled)
                    if capture is not None:
                        block.use_lease = True
                        provided.update(capture.snapshots)
        self.bind_ticket_results()
        self.bind_activity()
        self.bind_recipes()
        self.clear_results(preserve_history=True)
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
        provided = provided or {}
        self.restore_captured(provided)
        for key in provided:
            block = self.blocks[key]
            if block.last_result is not None:
                for port in block.outputs:
                    emitted[key, port.name] = block.current_emissions.get(port.name, [])

        def run_block(key):
            block = self.blocks[key]
            block._clock = self.clock
            incoming = [c for c in self.connections if c.target == key]
            if cancelled():
                block.state, block.error = "skipped", "Run stopped"
                return
            if any(
                self.blocks[c.source].state in ("error", "skipped")
                and not (getattr(block, "queue_boundary", False) and c.input == "image")
                and not getattr(block, "tolerates_unavailable_inputs", False)
                for c in incoming
            ):
                block.state, block.error = "skipped", "Upstream block unavailable"
                return
            streams = {c.input: emitted.get((c.source, c.output), []) for c in incoming}
            constants = {name: value for (target, name), value in injected.items() if target == key}
            outputs = {p.name: [] for p in block.outputs}
            if getattr(block, "queue_inputs", {}):
                queued_inputs = block.queue_inputs
                block._queued_public_events = {
                    name: [
                        *streams.get(name, []),
                        *([constants.pop(name)] if name in constants else []),
                    ]
                    for name in queued_inputs
                }
                remaining = {
                    name: values for name, values in streams.items() if name not in queued_inputs
                }
                if any(len(values) > 1 for values in remaining.values()):
                    block.state, block.error = (
                        "error",
                        "Queue changing ancillary data with the image payload",
                    )
                    return
                calls = [
                    {
                        **constants,
                        **{name: values[0] for name, values in remaining.items() if values},
                    }
                ]
                rejected_calls = []
                for name, packets in block._queued_public_events.items():
                    rejected_calls.extend(
                        {name: p} for p in packets if rejected_inputs(block, {name: p})
                    )
                    block._queued_public_events[name] = [
                        p for p in packets if not rejected_inputs(block, {name: p})
                    ]
                if rejected_calls:
                    if not any(block._queued_public_events.values()) and not any(
                        queue.stats()["queued"] or queue.stats()["busy"]
                        for queue in queued_inputs.values()
                    ):
                        calls = rejected_calls
                    else:
                        for rejected_call in rejected_calls:
                            publish_result(
                                block,
                                rejected_call,
                                kind="input_rejected",
                                message="Rejected failed public queue input",
                            )
            elif getattr(block, "queue_boundary", False):
                rejected_packets = []
                if not block.use_lease:
                    from .image_queue import QueuedCapture
                    from .queued_execution import ancestors, snapshots_for_frame

                    source_packets = streams.get("image", [])
                    preloaded = block.preloaded_events
                    block.preloaded_events = []
                    rejected_packets = [
                        p
                        for p in [*source_packets, *preloaded]
                        if rejected_inputs(block, {"image": p})
                    ]
                    block.pending_events = [
                        QueuedCapture(
                            block.normalize(packet),
                            snapshots_for_frame(
                                self, block.normalize(packet), ancestors(self, key)
                            ),
                        )
                        for packet in source_packets
                        if not rejected_inputs(block, {"image": packet})
                    ]
                    block.pending_events.extend(
                        QueuedCapture(block.normalize(p))
                        for p in preloaded
                        if not rejected_inputs(block, {"image": p})
                    )
                calls = [constants]
                if (
                    rejected_packets
                    and not block.pending_events
                    and not block.runtime["pending"]
                    and block.lease is None
                ):
                    calls = [{"image": p} for p in rejected_packets]
                else:
                    for packet in rejected_packets:
                        publish_result(
                            block,
                            {"image": packet},
                            kind="input_rejected",
                            message="Rejected failed queue input",
                        )
            elif getattr(block, "accept_events", False):
                calls = [
                    {name: values[index]}
                    for index in range(max((len(v) for v in streams.values()), default=0))
                    for name, values in streams.items()
                    if index < len(values)
                ]
                if constants:
                    calls[0:0] = [{name: value} for name, value in constants.items()]
                if getattr(block, "poll_without_input", False):
                    calls.append({})
            elif any(
                not values and next(p for p in block.inputs if p.name == name).required
                for name, values in streams.items()
            ):
                calls = []
            else:
                had_streams = bool(streams)
                streams = {name: values for name, values in streams.items() if values}
                length = max((len(v) for v in streams.values()), default=1)
                if (
                    had_streams
                    and not streams
                    and not constants
                    and not getattr(block, "poll_without_input", False)
                ):
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
            had_result = False
            last_processed_state = "inactive"
            for values in calls:
                if cancelled():
                    block.state, block.error = "skipped", "Run stopped"
                    return
                result = block.execute(values, cancelled)
                if result is None:
                    if block.input_rejected:
                        continue
                    return
                had_result = True
                last_processed_state = block.state
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
            if not had_result:
                block.state = "inactive"
                return
            block.state = last_processed_state
            if failed:
                block.state = "fail"
            elif block.state == "waiting" and any(
                outputs[n] for n in getattr(block, "flow_primary", outputs)
            ):
                block.state = "pass"
            # Retain all emissions for ToolBlock boundaries, receipts and UI snapshots.
            block.last_result.emissions = outputs
            block.current_emissions = outputs
            for name, packets in outputs.items():
                if packets:
                    block.last_result.outputs[name] = packets[-1].value
                    block.last_result.payloads[name] = packets[-1].payload
                    if block.run_profile.output_previews and isinstance(
                        packets[-1].value, (np.ndarray, ImagePyramid)
                    ):
                        block.last_output_images[name] = packets[-1].value
                emitted[key, name] = packets

        pending = set(self.blocks) - provided.keys()
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
                    and not getattr(self.blocks[key], "queue_boundary", False)
                    and not getattr(self.blocks[key], "accept_events", False)
                    and not getattr(self.blocks[key], "requires_serial_execution", False)
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

    def to_dict(self, include_recipes=True):
        catalog = self.recipes.definition() if include_recipes else None
        return {
            "schema_version": 1,
            **({"run_profile": self.run_profile.name} if self.run_profile.name != "debug" else {}),
            **(
                {"validation_sets": copy.deepcopy(self.validation_sets)}
                if self.validation_sets
                else {}
            ),
            **({"recipes": catalog} if catalog and catalog["recipes"] else {}),
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
                    **({"recipe_bindings": b.recipe_bindings} if b.recipe_bindings else {}),
                    **({"recipe_model": b.recipe_model} if b.recipe_model else {}),
                    **({"input_levels": b.input_levels} if b.input_levels else {}),
                    **(
                        {"input_failure_policies": b.input_failure_policies}
                        if b.input_failure_policies
                        else {}
                    ),
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
        graph.set_run_profile(data.get("run_profile", "debug"))
        if "validation_sets" in data:
            from .validation import validate_sets

            graph.validation_sets = validate_sets(data["validation_sets"])
        if "recipes" in data:
            graph.recipes = RecipeCatalog.from_dict(data["recipes"])
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
                if (
                    block.type_id == "inspect.final_judgment"
                    and direction == "inputs"
                    and exposed == ["value", "passed"]
                ):
                    exposed = [*exposed, "reasons"]
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
            policies = item.get("input_failure_policies", {})
            if (
                not isinstance(policies, dict)
                or not policies.keys() <= {p.name for p in block.inputs}
                or any(v not in INPUT_FAILURE_POLICIES for v in policies.values())
            ):
                raise ValueError("Invalid input failure policy")
            block.input_failure_policies = dict(policies)
            block.recipe_bindings = copy.deepcopy(item.get("recipe_bindings", {}))
            block.recipe_model = item.get("recipe_model", "")
            validate_bindings(block, block.recipe_bindings, block.recipe_model)
            graph.add(block)
        for c in data["connections"]:
            graph.connect(**c)
        graph.bind_project_dir(project_dir)
        return graph
