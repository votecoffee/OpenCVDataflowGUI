"""GUI-independent typed graph contracts, execution, and project persistence."""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .activity import ActivityLog, report
from .diagnostics import BlockLogger, exception_message
from .fail_reasons import normalize_reasons
from .graphics_layers import RESULT_GRAPHICS, GraphicDrawing, GraphicsBundle, graphics_bundle
from .outcomes import (
    INPUT_FAILURE_POLICIES,
    TicketResults,
    publish_result,
    rejected_inputs,
    stamp_result,
)
from .parameter_validation import NumericRange, ValidationError
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
from .port_types import SCALAR_KINDS, validate_scalar
from .recipes import RecipeCatalog, apply_recipe, validate_bindings
from .run_profiles import PROFILES, captures, graphics_mode, profile
from .scheduling import MAX_BRANCH_WORKERS, DependencyPlan, automatic_workers
from .tickets import DEFAULT_TICKET_SETTINGS, DIRECT_CLOCK, TICKET_MODES, TicketClock
from .timing import WorkTiming
from .visual_style import DEFAULT_NODE_SIZE, NODE_SIZES

_PLUGIN_DISPLAY_TITLES = {
    "inspection.ocv_segment_image": "SimpleSegmentation",
    "inspection.ocv_segment_parts": "SimpleSegmentation · Matched parts",
    "pattern.ogpm_recipe": "SimplePatternMatch",
    "pattern.ogpm_load_recipe": "SimplePatternMatch · Load recipe",
    "pattern.ogpm_train_recipe": "SimplePatternMatch · Train recipe",
    "pattern.ogpm_match_model": "SimplePatternMatch · Match wired recipe",
    "inspect.simple_path_width": "SimplePathWidth",
    "inspect.simple_reference": "SimpleReferenceInspect",
    "3d.simple_locate": "Simple3DLocate",
}

DISPLAY_RESULT_FIELDS = (
    "last_inputs",
    "input_payloads",
    "last_result",
    "last_output_images",
    "last_graphics",
    "state",
    "error",
    "error_kind",
    "current_emissions",
    "input_rejected",
    "last_rejection",
    "last_recipe_ref",
    "last_recipe_parameters",
)


def block_display_title(block):
    """Return the host title for a block class, instance, or optional plugin ID."""
    if isinstance(block, str):
        return _PLUGIN_DISPLAY_TITLES.get(block, block)
    return _PLUGIN_DISPLAY_TITLES.get(block.type_id, block.title)


@dataclass(frozen=True)
class Port:
    name: str
    kind: str = "image"
    required: bool = True
    visual_type: str = ""
    required_when_connected: bool = False


@dataclass
class Result:
    outputs: dict[str, Any] = field(default_factory=dict)
    # Standalone diagnostic previews, such as histograms and response maps.
    graphics: dict[str, np.ndarray] = field(default_factory=dict)
    passed: bool = True
    score: float | None = None
    message: str = ""
    payloads: dict[str, dict] = field(default_factory=dict)
    emissions: dict[str, list[ValuePacket]] = field(default_factory=dict)
    event_data: dict = field(default_factory=dict)
    final_judgment: bool | None = None
    fail_reasons: list = field(default_factory=list)
    # Named objects composited onto an image only when rendered.
    drawings: dict[str, GraphicDrawing] = field(default_factory=dict)


class Block:
    """Subclass and override process; runtime owns history and failure handling.

    Inputs are private snapshots: custom code may modify them without changing
    upstream history or a sibling branch. Keep only the latest run in memory.
    """

    type_id: ClassVar[str] = "base"
    title: ClassVar[str] = "Block"
    category: ClassVar[str] = "Custom"
    block_type: ClassVar[str] = ""
    parallel_safe: ClassVar[bool] = False
    layout_only: ClassVar[bool] = False
    subtype: ClassVar[str] = "General"
    inputs: ClassVar[tuple[Port, ...]] = ()
    outputs: ClassVar[tuple[Port, ...]] = ()
    defaults: ClassVar[dict[str, Any]] = {}
    parameter_choices: ClassVar[dict[str, tuple]] = {}
    parameter_help: ClassVar[dict[str, str]] = {}
    input_help: ClassVar[dict[str, str]] = {}
    preview_safe: ClassVar[bool | None] = None
    parameter_ranges: ClassVar[dict[str, NumericRange]] = {}
    plugin_dialog_label: ClassVar[str] = ""
    plugin_dialog_modeless: ClassVar[bool] = False

    def create_plugin_dialog(self, parent, parameters, context):
        """Return a fresh QDialog that edits the supplied inspector draft."""
        raise NotImplementedError("A block with a plugin dialog label must create a dialog")

    def __init__(self, *, block_id=None, parameters=None):
        self.id = block_id or uuid.uuid4().hex
        self.name = block_display_title(self)
        self._description = ""
        self._documentation = ""
        self.parameters = copy.deepcopy(self.defaults)
        if parameters:
            unknown = parameters.keys() - self.defaults.keys()
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            self.parameters.update(parameters)
        self.exposed_inputs = [p.name for p in self.inputs]
        self.exposed_outputs = [p.name for p in self.outputs]
        self.position = [0.0, 0.0]
        self.node_size = DEFAULT_NODE_SIZE
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
        self.graphics_mode = "inherit"
        self._job_graphics_mode = "inherit"
        self._graphics_for_parent = False
        self._activity = None
        self.logger = BlockLogger(self)
        self._execution_lock = threading.Lock()
        self._execution_owner = None
        self._work_started = self._work_finished = None
        self.elapsed_ms = 0.0
        self.clear_result()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        if not isinstance(value, str):
            raise ValueError("Block description must be text")
        self._description = value

    @property
    def documentation(self):
        return self._documentation

    @documentation.setter
    def documentation(self, value):
        if not isinstance(value, str):
            raise ValueError("Block documentation must be rich-text HTML")
        self._documentation = value

    def clear_result(self, preserve_history=False):
        self.current_emissions = {}
        self.error_kind = ""
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

    def issue_source_ticket(self):
        settings = self.ticket_settings
        if settings.get("mode") not in TICKET_MODES:
            raise ValueError("Unknown source ticket mode")
        external = self.last_inputs.get(
            "ticket", settings.get("value") if settings["mode"] == "fixed" else None
        )
        if settings["mode"] == "external" and "ticket" not in self.last_inputs:
            raise ValueError("External ticket mode requires the ticket input")
        ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(self.id, settings["mode"], external)
        return ticket

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
                choices_text = ", ".join(map(str, choices))
                raise ValidationError(f"{label} must be one of: {choices_text}", name)
        for name, limits in self.parameter_ranges.items():
            limits.check(values[name], name.replace("_", " ").capitalize(), name)

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
        self._execution_owner = threading.get_ident()
        self._activity_invocation = invocation
        self._activity_started = False
        try:
            if not acquired:
                self.report_event("unblocked", values=values, invocation=invocation)
            return self._execute(values, cancelled)
        except (Exception, SystemExit) as exc:
            self._record_execution_error(exc, values, active=self._activity_started)
            return None
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
    def output_ports(self):
        if self.layout_only:
            return ()
        if any(port.name == RESULT_GRAPHICS for port in self.outputs):
            return self.outputs
        return (*self.outputs, Port(RESULT_GRAPHICS, "graphics", required=False))

    @property
    def execution_outputs(self):
        return self.output_ports if RESULT_GRAPHICS in self.exposed_outputs else self.outputs

    @property
    def capture_graphics(self):
        """Whether to keep optional graphics for inspection."""
        mode = self.graphics_mode if self.graphics_mode != "inherit" else self._job_graphics_mode
        return self.run_profile.graphics if mode == "inherit" else mode == "on"

    @property
    def _graphics_required(self):
        return self._graphics_for_parent or RESULT_GRAPHICS in self.exposed_outputs

    @property
    def generate_graphics(self):
        """Custom tools can skip optional work when neither the UI nor a port needs it."""
        return self.capture_graphics or (
            self._graphics_required
            and (
                self._graphics_for_parent
                or self.run_profile.graphics
                or self._job_graphics_mode == "on"
            )
        )

    def set_graphics_mode(self, value):
        self.graphics_mode = graphics_mode(value)
        self._trim_graphics()

    def _trim_graphics(self):
        if not self.capture_graphics:
            self.last_graphics = {}
            if self.last_result is not None:
                self.last_result.graphics = {}
        if not self.generate_graphics and self.last_result is not None:
            self.last_result.drawings = {}

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
        validating = True
        try:
            values, self.input_payloads = prepare_inputs(self, values)
            self.last_inputs = copy.deepcopy(values)
            self.validate_parameters()
            apply_recipe(self)
            self.validate_parameters()
            for port in self.inputs:
                if port.required and port.name not in values:
                    raise ValueError(f"Missing input: {port.name}")
                if port.name in values:
                    validate_value(port, values[port.name])
            if self.ticket_source:
                self._source_ticket = self.issue_source_ticket()
                if self._source_ticket is not None:
                    self._report_activity_start(values)
                    self.report_event("ticket_assigned", values={"ticket": self._source_ticket})
            validating = False
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
                result.outputs["ticket"] = (
                    self._source_ticket if self._source_ticket is not None else NO_VALUE
                )
            if not self.capture_graphics:
                result.graphics = {}
            if not self.generate_graphics:
                result.drawings = {}
            for name, value in result.graphics.items():
                validate_value(Port(name), value)
            if RESULT_GRAPHICS in self.exposed_outputs and RESULT_GRAPHICS not in result.outputs:
                result.outputs[RESULT_GRAPHICS] = (
                    graphics_bundle(self, result)
                    if result.drawings or not self.outputs
                    or any(v is not NO_VALUE for v in result.outputs.values())
                    else NO_VALUE
                )
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
            expected = {p.name for p in self.execution_outputs}
            if result.outputs.keys() != expected:
                raise ValueError(f"Result outputs must match {sorted(expected)}")
            for port in self.execution_outputs:
                validate_value(port, result.outputs[port.name])
            if not result.emissions.keys() <= expected:
                raise ValueError("Emissions must name declared outputs")
            for name, packets in result.emissions.items():
                if len(packets) > 1024:
                    raise ValueError("Output exceeds 1024 emissions per invocation")
                for packet in packets:
                    if not isinstance(packet, ValuePacket):
                        raise ValueError("Emissions must be ValuePackets")
                    port = next(p for p in self.execution_outputs if p.name == name)
                    validate_value(port, packet)
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
            if self.outputs and all(
                result.outputs[name] is NO_VALUE
                for name in getattr(self, "flow_primary", tuple(p.name for p in self.outputs))
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
        except (Exception, SystemExit) as exc:
            invalid = validating and isinstance(exc, (ValueError, TypeError))
            self._record_execution_error(exc, values, active=True, invalid=invalid)
            if getattr(self, "accept_events", False) or getattr(self, "sequence_memory", False):
                try:
                    self.reset_runtime()
                except (Exception, SystemExit) as reset_error:
                    # Keep the processing failure as the block's result.
                    diagnostic = self._diagnostic_exception
                    self.logger.exception(
                        f"Runtime reset failed: {exception_message(reset_error)}",
                        exc_info=reset_error,
                    )
                    self._diagnostic_exception = diagnostic
        finally:
            self.parameters = configured_parameters
            if getattr(self, "supports_training", False):
                self.trained_model = configured_model
            self._finish_work()
            if not self.run_profile.input_previews:
                self.last_inputs = {}
        return self.last_result

    def _record_execution_error(self, exc, values, *, active=False, invalid=False):
        self.state, self.error = "error", exception_message(exc)
        self.error_kind = "invalid" if invalid or isinstance(exc, ValidationError) else "exception"
        self.input_rejected = False
        self.last_result = None
        self.last_output_images, self.last_graphics, self.current_emissions = {}, {}, {}
        if active:
            self._finish_work()
            if not self._activity_start_reported:
                self._report_activity_start(values)
        else:
            self._source_ticket = self._work_started = self._work_finished = None
        self.logger.exception(self.error, exc_info=exc, values=values)
        failure_values = dict(values)
        if active:
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
        try:
            publish_result(self, failure_values, kind="error", message=self.error)
        except (Exception, SystemExit) as report_error:
            # Invalid input metadata can fail both validation and ticket reporting.
            self.logger.error("Could not record ticket error: %s", exception_message(report_error))

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

    def _close_safely(self):
        try:
            self.close()
        except (Exception, SystemExit) as exc:
            self.logger.exception(f"Close failed: {exception_message(exc)}", exc_info=exc)

    def reset_runtime(self):
        """Clear transient stream state after configuration changes or explicit reset."""

    def copy_runtime_from(self, other):
        """Copy only transient stream state into a separate execution/display graph."""

    def serialize_extra(self):
        return {}

    def to_project_item(self):
        return {
            "id": self.id,
            "type": self.type_id,
            "name": self.name,
            **({"description": self.description} if self.description else {}),
            **({"documentation": self.documentation} if self.documentation else {}),
            "parameters": self.parameters,
            "position": self.position,
            **({"node_size": self.node_size} if self.node_size != DEFAULT_NODE_SIZE else {}),
            "exposed_inputs": self.exposed_inputs,
            "exposed_outputs": self.exposed_outputs,
            "thumbnail": self.thumbnail,
            "preview_output": self.preview_output,
            **({"graphics_mode": self.graphics_mode} if self.graphics_mode != "inherit" else {}),
            **({"recipe_bindings": self.recipe_bindings} if self.recipe_bindings else {}),
            **({"recipe_model": self.recipe_model} if self.recipe_model else {}),
            **({"input_levels": self.input_levels} if self.input_levels else {}),
            **(
                {"input_failure_policies": self.input_failure_policies}
                if self.input_failure_policies
                else {}
            ),
            **(
                {"ticket_settings": self.ticket_settings}
                if self.ticket_source and self.ticket_settings != DEFAULT_TICKET_SETTINGS
                else {}
            ),
            **self.serialize_extra(),
        }

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
                "error_kind",
                "elapsed_ms",
                "timing",
                "input_rejected",
                "last_rejection",
                "last_recipe_ref",
                "last_recipe_parameters",
            ):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
            self.last_inputs, self.last_output_images = {}, {}
            self.last_graphics = copy.deepcopy(other.last_graphics)
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
        for name in (*DISPLAY_RESULT_FIELDS, "elapsed_ms", "timing"):
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
    if port.kind == "cloud":
        from .pointclouds import PointCloud

        if not isinstance(value, PointCloud):
            raise TypeError(f"{port.name} requires a PointCloud, not XYZ image pixels")
        value.__post_init__()
    elif port.kind == "dct":
        from .spectral import DCTCoefficients

        if not isinstance(value, DCTCoefficients):
            raise TypeError(f"{port.name} requires DCT coefficients")
    elif port.kind == "dft":
        from .spectral import DFTSpectrum

        if not isinstance(value, DFTSpectrum):
            raise TypeError(f"{port.name} requires a Fourier spectrum")
    elif port.kind == "encoded_image":
        from .image_codecs import EncodedImage

        if not isinstance(value, EncodedImage):
            raise TypeError(f"{port.name} requires an encoded image packet")
    elif port.kind == "remap_map":
        from .remap_maps import PackedRemapMap

        if not isinstance(value, PackedRemapMap):
            raise TypeError(f"{port.name} requires packed remap coordinates")
    elif port.kind == "depth":
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
    elif port.kind in SCALAR_KINDS:
        validate_scalar(port, value)
    elif port.kind == "graphics":
        if not isinstance(value, GraphicsBundle):
            raise TypeError(f"{port.name} requires a result graphics bundle")
    elif port.kind == "data":
        json.dumps(value, allow_nan=False)
    elif port.kind != "any":
        raise ValueError(f"Unknown port type: {port.kind}")


class Registry:
    def __init__(self):
        self.types: dict[str, type[Block]] = {}
        self._providers = {}

    def add_provider(self, type_ids, load):
        """Register a host-selected loader for known optional block IDs."""
        self._providers.update(dict.fromkeys(type_ids, load))

    def register(self, block_type):
        if not issubclass(block_type, Block) or block_type.type_id in self.types:
            raise ValueError("Block must be a subclass with a unique type_id")
        self.types[block_type.type_id] = block_type

    def create(self, type_id, **kwargs):
        try:
            if type_id not in self.types and type_id in self._providers:
                self._providers[type_id](self)
            if type_id not in self.types:
                raise ValueError(f"Unavailable block type: {type_id}. Load its plugin first.")
            block = self.types[type_id](**kwargs)
            if hasattr(block, "bind_registry"):
                block.bind_registry(self)
        except SystemExit as exc:
            raise RuntimeError(f"Could not create {type_id}: {exception_message(exc)}") from exc
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
        from .filmstrip import Filmstrip

        self.filmstrip = Filmstrip()
        self.filmstrip_enabled = False
        self.groups = []
        self.viewport = {}
        self.run_profile = PROFILES["debug"]
        self.branch_workers = "auto"
        self.graphics_mode = "inherit"
        self._parent_graphics_mode = None
        self._graphics_for_parent = False
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
            self._bind_block_graphics(block)
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
            self._bind_block_graphics(block)
            if not self.run_profile.input_previews:
                block.last_inputs = {}
            if not self.run_profile.output_previews:
                block.last_output_images = {}
            if hasattr(block, "graph"):
                block.graph.set_run_profile(self.run_profile)

    def set_graphics_mode(self, value):
        self.graphics_mode = graphics_mode(value)
        self.set_run_profile(self.run_profile)

    def _inherit_graphics_from(self, owner):
        self._parent_graphics_mode = owner._job_graphics_mode
        self._graphics_for_parent = (
            owner.type_id == "core.toolblock"
            and owner._graphics_required
            and owner.generate_graphics
        )

    def _bind_block_graphics(self, block):
        block._job_graphics_mode = (
            self.graphics_mode if self._parent_graphics_mode is None else self._parent_graphics_mode
        )
        block._graphics_for_parent = self._graphics_for_parent
        block._trim_graphics()
        if hasattr(block, "graph"):
            block.graph._inherit_graphics_from(block)

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
        block = self.blocks.pop(block_id)
        self.groups = [
            {**g, "members": [m for m in g["members"] if m != block_id]}
            for g in self.groups
            if any(m != block_id for m in g["members"])
        ]
        if not block.layout_only:
            self.invalidate()

    def invalidate(self):
        self.clock = TicketClock()
        self.clear_results()
        for block in self.blocks.values():
            try:
                block.reset_runtime()
            except (Exception, SystemExit) as exc:
                block._record_execution_error(exc, {})

    def clear_results(self, preserve_history=False):
        for block in self.blocks.values():
            try:
                block.clear_result(preserve_history=preserve_history)
            except (Exception, SystemExit) as exc:
                block._record_execution_error(exc, {})

    def copy_runtime_from(self, other):
        self.clock = other.clock.clone()
        if self.recipes is not other.recipes:
            self.recipes.copy_runtime_from(other.recipes)
        if self.ticket_results is not other.ticket_results:
            self.ticket_results.restore(other.ticket_results.snapshot())
        for key, block in self.blocks.items():
            block.copy_runtime_from(other.blocks[key])
        self.copy_timing_from(other)

    def copy_source_runtime_from(self, other):
        for key, block in self.blocks.items():
            source = other.blocks.get(key)
            if source is None:
                continue
            if getattr(block, "file_source", False):
                block.copy_runtime_from(source)
            if hasattr(block, "graph") and hasattr(source, "graph"):
                block.graph.copy_source_runtime_from(source.graph)

    @property
    def buffered_stream(self):
        return any(getattr(b, "buffered_stream", False) for b in self.blocks.values())

    @property
    def pending_output(self):
        for block in self.blocks.values():
            if block.state in ("error", "skipped"):
                continue
            try:
                if getattr(block, "pending_output", False) or (
                    getattr(block, "queue_boundary", False) and block.stats()["queued"]
                ):
                    return True
            except (Exception, SystemExit) as exc:
                block._record_execution_error(exc, {})
        return False

    @property
    def pending_tickets(self):
        from .tickets import validate_ticket

        tickets = []
        for block in self.blocks.values():
            if block.state in ("error", "skipped"):
                continue
            try:
                pending = tuple(getattr(block, "pending_tickets", ()))
                for ticket in pending:
                    validate_ticket(ticket)
                tickets.extend(pending)
            except (Exception, SystemExit) as exc:
                block._record_execution_error(exc, {})
        return tuple(tickets)

    def _upstream_ended(self, key, external_end=False):
        """Follow image/set ancestors; scalar controls do not keep a stream open."""
        seen = set()

        def visit(node):
            if node in seen:
                return False, True
            seen.add(node)
            block = self.blocks[node]
            if getattr(block, "buffered_stream", False):
                return True, bool(
                    getattr(block, "end_of_input", False) and not block.pending_output
                )
            if getattr(block, "queue_boundary", False) and block.stats()["queued"]:
                return True, False
            if hasattr(block, "graph") and block.graph.pending_output:
                return True, False
            if getattr(block, "file_source", False):
                return True, bool(block.finite_source and block.exhausted)
            if block.ticket_source:
                return True, False
            if hasattr(block, "graph") and block.graph.has_finite_sources():
                return True, block.graph.sources_exhausted()
            return ancestors(node)

        def ancestors(node):
            statuses = []
            for edge in self.connections:
                if edge.target != node:
                    continue
                source = self.blocks[edge.source]
                port = next(p for p in source.output_ports if p.name == edge.output)
                if port.kind in ("image", "any"):
                    statuses.append(visit(edge.source))
            found = [done for present, done in statuses if present]
            return bool(found), all(found)

        found, done = ancestors(key)
        return done if found else external_end

    def sources_exhausted(self):
        sources = []
        for block in self.blocks.values():
            if getattr(block, "file_source", False) and block.finite_source:
                sources.append(block.exhausted)
            if hasattr(block, "graph") and block.graph.has_finite_sources():
                sources.append(block.graph.sources_exhausted())
        return bool(sources) and all(sources) and not self.pending_output

    def has_finite_sources(self):
        return any(
            (getattr(b, "file_source", False) and b.finite_source)
            or (hasattr(b, "graph") and b.graph.has_finite_sources())
            for b in self.blocks.values()
        )

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
                if name == "_source_runtime":
                    block.restore_runtime(value)
                elif name == "_captured_graph":
                    block.graph.restore_captured(value)
                else:
                    setattr(block, name, copy.deepcopy(value))
            if not self.run_profile.input_previews:
                block.last_inputs = {}
            if not self.run_profile.output_previews:
                block.last_output_images = {}
            block._trim_graphics()

    def connect(self, source, output, target, input):
        if source == target:
            raise ValueError("A block cannot connect to itself")
        self.validate_connection_ports(self.blocks[source], output, self.blocks[target], input)
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

    @staticmethod
    def validate_connection_ports(src, output, dst, input):
        out = next((p for p in src.output_ports if p.name == output), None)
        inp = next((p for p in dst.inputs if p.name == input), None)
        if out is None or inp is None:
            raise ValueError("Unknown port")
        if output not in src.exposed_outputs or input not in dst.exposed_inputs:
            raise ValueError("Expose both ports in the inspector before connecting")
        if not compatible(out.kind, inp.kind):
            raise ValueError(f"Cannot connect {out.kind} to {inp.kind}")

    def ordered(self):
        return DependencyPlan(self.blocks, self.connections).order

    def processing_blocks(self):
        """Iterate the tools that participate in execution and runtime results."""
        return (block for block in self.blocks.values() if not block.layout_only)

    def set_branch_workers(self, value):
        if value != "auto" and (type(value) is not int or not 1 <= value <= MAX_BRANCH_WORKERS):
            raise ValueError(
                f"Branch workers must be 'auto' or an integer from 1 to {MAX_BRANCH_WORKERS}"
            )
        self.branch_workers = value

    def run(
        self, cancelled=lambda: False, injected=None, provided=None, *,
        filmstrip_capture=None, end_of_input=False,
    ):
        if self.filmstrip_enabled and self.production_completion and not self._event_prefix:
            from .filmstrip import FilmstripCapture

            filmstrip_capture = FilmstripCapture(self.filmstrip.max_bytes)
        failed = False
        try:
            return self._run_iteration(
                cancelled, injected, provided, filmstrip_capture, end_of_input
            )
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
                if self.filmstrip_enabled:
                    self.filmstrip.capture(self, filmstrip_capture)

    def _complete_tickets(self):
        from .outcomes import failed_input, packets_in
        from .tickets import ticket_key

        if any(
            b.state == "waiting" and not getattr(b, "validation_skip_output", False)
            for b in self.blocks.values()
        ):
            return
        pending = {ticket_key(ticket) for ticket in self.pending_tickets}
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
            if ticket_key(ticket) in pending:
                continue
            self.ticket_results.complete(
                ticket, passed, timing, summary=not self.run_profile.ticket_details
            )

    def _run_iteration(self, cancelled, injected, provided, filmstrip_capture, end_of_input=False):
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
        for target, name in injected:
            block = self.blocks[target]
            port = next((p for p in block.inputs if p.name == name), None)
            if port is None or any(
                c.target == target and c.input == name for c in self.connections
            ):
                raise ValueError("Invalid or internally connected public input")
        plan = DependencyPlan((block.id for block in self.processing_blocks()), self.connections)
        splits = [block for block in self.blocks.values() if block.type_id == "flow.split"]
        workers = self.branch_workers
        if workers == "auto":
            if splits:
                workers = max(int(block.parameters.get("workers", 1)) for block in splits)
                if not 1 <= workers <= MAX_BRANCH_WORKERS:
                    raise ValueError(f"Parallel worker count must be 1..{MAX_BRANCH_WORKERS}")
            else:
                workers = automatic_workers()
        emitted = {}
        provided = provided or {}
        self.restore_captured(provided)
        for key in provided:
            block = self.blocks[key]
            if block.last_result is not None:
                if filmstrip_capture is not None:
                    filmstrip_capture.record_snapshot(block)
                for port in block.execution_outputs:
                    emitted[key, port.name] = block.current_emissions.get(port.name, [])

        def run_block(key):
            block = self.blocks[key]
            try:
                return dispatch(block, key)
            except (Exception, SystemExit) as exc:
                block._record_execution_error(exc, {})
                return None

        def dispatch(block, key):
            if block.state == "error":  # The block's initial clear hook failed.
                return
            block._clock = self.clock
            incoming = plan.incoming[key]
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
            outputs = {p.name: [] for p in block.execution_outputs}
            buffered = getattr(block, "buffered_stream", False)
            stream_ended = buffered and self._upstream_ended(key, end_of_input)
            draining = stream_ended and not any(
                streams.get(name) or name in constants
                for name in getattr(block, "stream_inputs", ())
            )
            if draining:
                calls = [constants]
            elif getattr(block, "queue_inputs", {}):
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
                    block.error_kind = "invalid"
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
                not streams[port.name] and (port.required or port.required_when_connected)
                for port in block.inputs
                if port.name in streams
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
                # A ticketed measurement cannot stand in for missing per-frame values.
                if any(len(v) not in (1, length) for v in streams.values()) or any(
                    port.required_when_connected
                    and port.name in streams
                    and len(streams[port.name]) == 1 < length
                    and "ticket" in streams[port.name][0].payload
                    for port in block.inputs
                ):
                    block.error_kind = "invalid"
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
                block.error_kind = "invalid"
                block.state, block.error = "error", "Block exceeds 1024 input events per run"
                return
            if not calls:
                block.state = "inactive"
                return
            failed = False
            had_result = False
            last_processed_state = "inactive"
            for call_index, values in enumerate(calls):
                if buffered:
                    block.end_of_input = bool(stream_ended and call_index == len(calls) - 1)
                if cancelled():
                    block.state, block.error = "skipped", "Run stopped"
                    return
                observe_children = filmstrip_capture is not None and isinstance(block, ToolBlock)
                if observe_children:
                    block._filmstrip_capture = filmstrip_capture
                try:
                    result = block.execute(values, cancelled)
                except (Exception, SystemExit) as exc:
                    block._record_execution_error(exc, values)
                    return
                finally:
                    if observe_children:
                        block._filmstrip_capture = None
                if result is None:
                    if block.input_rejected:
                        continue
                    return
                if filmstrip_capture is not None:
                    filmstrip_capture.record(block, result)
                had_result = True
                last_processed_state = block.state
                failed |= block.state == "fail"
                for port in block.execution_outputs:
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
                outputs[n] for n in getattr(block, "flow_primary", (p.name for p in block.outputs))
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
            return outputs

        def publish(key, outputs):
            if outputs is not None:
                for name, packets in outputs.items():
                    emitted[key, name] = packets

        def skip(key):
            self.blocks[key].state, self.blocks[key].error = "skipped", "Run stopped"

        parallel = {
            block.id
            for block in self.processing_blocks()
            if workers > 1
            and block.id not in provided
            and (
                splits
                or block.parallel_safe
                or type(block).__module__.startswith("opencv_dataflow.")
            )
            and not block.ticket_source
            and block.category not in ("Input", "Output")
            and block.block_type not in ("Input", "Output")
            and not block.recipe_bindings
            and not block.recipe_model
            and not isinstance(block, ToolBlock)
            and not getattr(block, "owns_workers", False)
            and not getattr(block, "queue_boundary", False)
            and not getattr(block, "accept_events", False)
            and not getattr(block, "requires_serial_execution", False)
        }
        plan.run(run_block, publish, provided, parallel, workers, cancelled, skip)

    def close(self):
        for block in self.blocks.values():
            block._close_safely()

    def to_dict(self, include_recipes=True):
        catalog = self.recipes.definition() if include_recipes else None
        return {
            "schema_version": 1,
            **({"groups": copy.deepcopy(self.groups)} if self.groups else {}),
            **({"viewport": copy.deepcopy(self.viewport)} if self.viewport else {}),
            **({"run_profile": self.run_profile.name} if self.run_profile.name != "debug" else {}),
            **({"branch_workers": self.branch_workers} if self.branch_workers != "auto" else {}),
            **({"graphics_mode": self.graphics_mode} if self.graphics_mode != "inherit" else {}),
            **(
                {"validation_sets": copy.deepcopy(self.validation_sets)}
                if self.validation_sets
                else {}
            ),
            **({"recipes": catalog} if catalog and catalog["recipes"] else {}),
            "blocks": [b.to_project_item() for b in self.blocks.values()],
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
        graph.set_branch_workers(data.get("branch_workers", "auto"))
        graph.set_graphics_mode(data.get("graphics_mode", "inherit"))
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
            block.description = item.get("description", block.description)
            block.documentation = item.get("documentation", block.documentation)
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
            size = item.get("node_size", DEFAULT_NODE_SIZE)
            if not isinstance(size, str) or size not in NODE_SIZES:
                raise ValueError("Invalid block size")
            block.node_size = size
            for direction in ("inputs", "outputs"):
                exposed = item[f"exposed_{direction}"]
                if (
                    block.type_id == "inspect.final_judgment"
                    and direction == "inputs"
                    and exposed == ["value", "passed"]
                ):
                    exposed = [*exposed, "reasons"]
                ports = block.inputs if direction == "inputs" else block.output_ports
                allowed = {p.name for p in ports}
                if len(set(exposed)) != len(exposed) or not set(exposed) <= allowed:
                    raise ValueError("Invalid exposed ports")
                setattr(block, f"exposed_{direction}", exposed)
            block.thumbnail = item.get("thumbnail", True)
            block.preview_output = item.get("preview_output", "")
            block.set_graphics_mode(item.get("graphics_mode", "inherit"))
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
        from .workspace import validate_layout

        graph.groups, graph.viewport = validate_layout(
            data.get("groups", []), data.get("viewport", {}), graph.blocks
        )
        graph.bind_project_dir(project_dir)
        return graph
