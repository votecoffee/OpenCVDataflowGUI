"""Versioned, file-based application authoring for assistants and Python clients.

This layer uses the editor's graph contracts. It does not attach to an open editor
or execute a project while inspecting, validating, editing, or saving it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from .blocks import builtin_registry
from .core import Graph, block_display_title
from .diagnostics import exception_message
from .opencv_reference import BINDINGS
from .parameter_validation import ValidationError
from .plugins import load_plugin
from .port_types import connection_mode
from .project_paths import rebase_project_paths
from .queued_execution import QueuedExecution, background_queue
from .runner import describe
from .visual_style import NODE_SIZES

API_VERSION = 1


class RevisionConflict(ValueError):
    """The saved project changed, or a destination already exists."""


class ProjectValidationError(ValidationError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("Project configuration is invalid")


def make_registry(plugins=()):
    """Load only explicitly selected Python files, plus host-known integrations."""
    registry = builtin_registry()
    for path in plugins:
        load_plugin(Path(path), registry)
    return registry


def file_revision(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def catalog(registry=None, query=""):
    """Return searchable summaries; optional providers are not loaded here."""
    registry = registry if registry is not None else make_registry()
    terms = query.casefold().split()
    records = []
    for type_id in registry.available_type_ids():
        cls = registry.types.get(type_id)
        record = {
            "type": type_id,
            "title": block_display_title(cls or type_id),
            "loaded": cls is not None,
            "category": cls.category if cls else "Simple Plugins / optional provider",
            "block_type": cls.block_type if cls else "",
            "subtype": cls.subtype if cls else "",
            "opencv_functions": BINDINGS.get(type_id, "").split(),
        }
        haystack = json.dumps(record).casefold()
        if all(term in haystack for term in terms):
            records.append(record)
    return records


def block_schema(type_id, registry=None):
    """Resolve one block, including dynamic/source ports and parameter metadata."""
    registry = registry if registry is not None else make_registry()
    block = registry.create(type_id)
    try:
        parameters = {}
        for name, default in block.defaults.items():
            limits = block.parameter_ranges.get(name)
            parameters[name] = {
                "default": copy.deepcopy(default),
                "choices": list(block.parameter_choices.get(name, ())),
                "help": block.parameter_help.get(name, ""),
                "range": asdict(limits) if limits else None,
            }
        return {
            "type": type_id,
            "title": block_display_title(block),
            "category": block.category,
            "block_type": block.block_type,
            "subtype": block.subtype,
            "description": block.description or (type(block).__doc__ or "").strip(),
            "inputs": [
                {
                    **asdict(p),
                    "help": block.input_help.get(p.name, ""),
                    "exposed": p.name in block.exposed_inputs,
                }
                for p in block.inputs
            ],
            "outputs": [
                {**asdict(p), "exposed": p.name in block.exposed_outputs}
                for p in block.output_ports
            ],
            "parameters": parameters,
            "opencv_functions": BINDINGS.get(type_id, "").split(),
            "capabilities": {
                "parallel_safe": block.parallel_safe,
                "preview_safe": block.preview_safe,
                "layout_only": block.layout_only,
                "ticket_source": block.ticket_source,
                "plugin_dialog": bool(block.plugin_dialog_label),
            },
            "project_path_parameters": list(block.project_path_parameters),
            "project_asset_parameters": list(block.project_asset_parameters),
            "template": copy.deepcopy(block.to_project_item()),
        }
    finally:
        block._close_safely()


def _graphs(graph, scope=(), bound=frozenset()):
    yield graph, scope, bound
    for block in graph.blocks.values():
        nested = getattr(block, "graph", None)
        if isinstance(nested, Graph):
            owner = getattr(block, "inspection", block)
            bindings = getattr(owner, "input_bindings", {})
            supplied = frozenset((b.block, b.port) for b in bindings.values())
            yield from _graphs(nested, (*scope, block.id), supplied)


def validate_graph(graph, *, require_complete=True):
    """Static diagnostics. Image-dependent rules are checked only during execution."""
    issues = []
    for current, scope, bound in _graphs(graph):
        current.ordered()
        wired = {(c.target, c.input) for c in current.connections} | bound
        for block in current.blocks.values():
            try:
                block.validate_parameters()
            except (Exception, SystemExit) as exc:
                issues.append(
                    {
                        "scope": list(scope),
                        "block": block.id,
                        "kind": "parameters",
                        "message": exception_message(exc),
                        "fields": list(getattr(exc, "fields", ())),
                    }
                )
            if require_complete and not block.layout_only:
                for port in block.inputs:
                    if port.required and (block.id, port.name) not in wired:
                        issues.append(
                            {
                                "scope": list(scope),
                                "block": block.id,
                                "kind": "missing_input",
                                "message": f"Required input {port.name!r} is not connected",
                                "fields": [port.name],
                            }
                        )
    return {"valid": not issues, "issues": issues, "executed": False}


_FIELDS = frozenset(
    "name description documentation position node_size exposed_inputs exposed_outputs "
    "thumbnail preview_output graphics_mode recipe_bindings recipe_model input_levels "
    "input_failure_policies ticket_settings parameters".split()
)


def _item(data, block_id):
    matches = [b for b in data["blocks"] if b["id"] == block_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one block with id {block_id!r}")
    return matches[0]


def _scope(data, path):
    if not isinstance(path, list) or any(not isinstance(p, str) for p in path):
        raise ValueError("Operation scope must be a list of containing block IDs")
    for block_id in path:
        item = _item(data, block_id)
        key = "inspection" if item["type"] == "flow.inspect_each" else "toolblock"
        data = item[key]["graph"]
    return data


def _keys(operation, required, optional=()):
    allowed = set(required) | set(optional) | {"op", "scope"}
    if set(operation) - allowed or set(required) - set(operation):
        raise ValueError(f"Invalid fields for {operation.get('op')!r}: expected {sorted(allowed)}")


def _apply_operation(data, operation, registry):
    if not isinstance(operation, dict):
        raise ValueError("Each operation must be an object")
    target = _scope(data, operation.get("scope", []))
    kind = operation.get("op")
    if kind == "add":
        _keys(operation, ("type", "id"), ("parameters", "name", "position"))
        if not isinstance(operation["id"], str) or not operation["id"].strip():
            raise ValueError("Block ID must be nonempty text")
        block = registry.create(
            operation["type"], block_id=operation["id"], parameters=operation.get("parameters")
        )
        try:
            item = copy.deepcopy(block.to_project_item())
            for name in ("name", "position"):
                if name in operation:
                    item[name] = operation[name]
            target["blocks"].append(item)
        finally:
            block._close_safely()
    elif kind == "update":
        _keys(operation, ("id", "fields"))
        fields = operation["fields"]
        if not isinstance(fields, dict) or fields.keys() - _FIELDS:
            raise ValueError("Unknown block field; replace nested definitions with insert/replace")
        item = _item(target, operation["id"])
        for name, value in fields.items():
            if name == "parameters":
                if not isinstance(value, dict):
                    raise ValueError("Parameters must be an object")
                item[name].update(value)
            else:
                item[name] = value
    elif kind in ("insert", "replace"):
        _keys(operation, ("block",))
        item = copy.deepcopy(operation["block"])
        if kind == "replace":
            target["blocks"][target["blocks"].index(_item(target, item["id"]))] = item
        else:
            target["blocks"].append(item)
    elif kind == "remove":
        _keys(operation, ("id",), ("disconnect",))
        item = _item(target, operation["id"])
        attached = [
            c for c in target["connections"] if operation["id"] in (c["source"], c["target"])
        ]
        if attached and operation.get("disconnect") is not True:
            raise ValueError("Removing a connected block requires disconnect=true")
        target["blocks"].remove(item)
        target["connections"] = [c for c in target["connections"] if c not in attached]
        for group in target.get("groups", []):
            group["members"] = [m for m in group["members"] if m != operation["id"]]
        target["groups"] = [g for g in target.get("groups", []) if g["members"]]
    elif kind in ("connect", "disconnect"):
        _keys(operation, ("source", "output", "target", "input"))
        wire = {name: operation[name] for name in ("source", "output", "target", "input")}
        if kind == "connect":
            target["connections"].append(wire)
        else:
            target["connections"].remove(wire)
    elif kind == "settings":
        _keys(operation, ("fields",))
        fields = operation["fields"]
        if not isinstance(fields, dict) or fields.keys() - {
            "run_profile",
            "branch_workers",
            "graphics_mode",
            "groups",
            "viewport",
            "validation_sets",
            "recipes",
        }:
            raise ValueError("Unknown project setting")
        target.update(fields)
    elif kind == "arrange":
        _keys(operation, ())
        from .workspace import arranged_positions

        graph = Graph.from_dict(target, registry)
        try:
            sizes = {
                b.id: (
                    NODE_SIZES[b.node_size][0],
                    90
                    + NODE_SIZES[b.node_size][1]
                    * max(len(b.exposed_inputs), len(b.exposed_outputs))
                    + NODE_SIZES[b.node_size][2],
                )
                for b in graph.blocks.values()
            }
            positions = arranged_positions(graph, sizes)
            for item in target["blocks"]:
                item["position"] = list(positions[item["id"]])
        finally:
            graph.close()
    else:
        raise ValueError(f"Unknown operation {kind!r}")


@contextmanager
def _write_lock(path):
    """Coordinate API writers. Desktop/third-party writers do not take this lock."""
    lock = path.with_name(f".{path.name}.assistant.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RevisionConflict(f"Another API writer holds {lock}") from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


class ProjectSession:
    """A working graph with transactional edits and optimistic file revisions."""

    def __init__(self, registry=None, *, directory=None):
        self.registry = registry if registry is not None else make_registry()
        self.graph = Graph()
        self.graph.bind_project_dir(directory)
        self.source_path = None
        self.revision = None

    @classmethod
    def open(cls, path, registry=None):
        session = cls(registry)
        path = Path(path).resolve()
        raw = path.read_bytes()
        session.graph = Graph.from_dict(json.loads(raw), session.registry, project_dir=path.parent)
        session.source_path = path
        session.revision = hashlib.sha256(raw).hexdigest()
        return session

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.graph.close()

    def inspect(self):
        """Return applied configuration plus instance-specific ports at every scope."""
        ports = []
        for current, scope, _ in _graphs(self.graph):
            for block in current.blocks.values():
                ports.append(
                    {
                        "scope": list(scope),
                        "id": block.id,
                        "type": block.type_id,
                        "inputs": [asdict(p) for p in block.inputs],
                        "outputs": [asdict(p) for p in block.output_ports],
                    }
                )
        return {
            "path": str(self.source_path) if self.source_path else None,
            "revision": self.revision,
            "project": copy.deepcopy(self.graph.to_dict()),
            "ports": ports,
            "validation": validate_graph(self.graph),
        }

    def apply(self, operations):
        """Commit all operations to this session, or preserve its previous graph."""
        if not isinstance(operations, list):
            raise ValueError("Operations must be a list")
        data = copy.deepcopy(self.graph.to_dict())
        for operation in operations:
            _apply_operation(data, operation, self.registry)
        candidate = Graph.from_dict(data, self.registry, project_dir=self.graph.project_dir)
        try:
            validation = validate_graph(candidate, require_complete=False)
            if not validation["valid"]:
                raise ProjectValidationError(validation["issues"])
        except BaseException:
            candidate.close()
            raise
        previous, self.graph = self.graph, candidate
        previous.close()
        return self.inspect()

    def save(self, path=None, *, expected_revision=None):
        """Atomically save, checking an existing destination against its known bytes.

        API writers share a sidecar lock. Other writers can still race between the
        final hash check and replace; use separate working copies for GUI co-work.
        """
        if path is None and self.source_path is None:
            raise ValueError("Choose a project destination")
        path = Path(path or self.source_path).resolve()
        if path.suffix.lower() != ".json":
            raise ValueError("Project destination must end in .json")
        if expected_revision is None and path == self.source_path:
            expected_revision = self.revision
        temporary = None
        with _write_lock(path):
            current = file_revision(path)
            if current != expected_revision:
                raise RevisionConflict(f"Project revision changed: {path}; inspect it again")
            try:
                with rebase_project_paths(self.graph, path.parent, path):
                    raw = (
                        json.dumps(self.graph.to_dict(), indent=2, allow_nan=False) + "\n"
                    ).encode("utf-8")
                    with NamedTemporaryFile(
                        dir=path.parent, prefix=".assistant-", delete=False
                    ) as f:
                        temporary = Path(f.name)
                        f.write(raw)
                        f.flush()
                        os.fsync(f.fileno())
                    if file_revision(path) != expected_revision:
                        raise RevisionConflict(f"Project changed while preparing save: {path}")
                    if current is None:
                        # An exclusive atomic publication also protects a new destination.
                        try:
                            os.link(temporary, path)
                        except FileExistsError:
                            raise RevisionConflict(f"Destination appeared: {path}") from None
                    else:
                        temporary.replace(path)
                    self.graph.bind_project_dir(path.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.source_path = path
        self.revision = hashlib.sha256(raw).hexdigest()
        return {"path": str(path), "revision": self.revision}

    def run(self, *, repeat=1, all_images=False, max_runs=1000, cancelled=lambda: False):
        """Explicitly execute this session and collect the runtime's normal receipts.

        Execution can perform configured file/device side effects. Cancellation is
        cooperative; it cannot interrupt a native call or arbitrary plugin code.
        """
        if type(repeat) is not int or type(max_runs) is not int or not 1 <= repeat <= max_runs:
            raise ValueError("Require 1 <= repeat <= max_runs, with whole numbers")
        if all_images and not self.graph.has_finite_sources():
            raise ValueError("all_images requires a finite image/video source with repeat=stop")
        validation = validate_graph(self.graph)
        if not validation["valid"]:
            raise ProjectValidationError(validation["issues"])
        queued = None
        runs = []
        limit = max_runs if all_images else repeat
        stop_reason = "requested_runs"
        try:
            if background_queue(self.graph) is not None:
                queued = QueuedExecution(self.graph, self.registry)
                queued.start()
            for index in range(limit):
                if cancelled():
                    stop_reason = "cancelled"
                    break
                if queued is None and self.graph.sources_exhausted():
                    stop_reason = "sources_exhausted"
                    break
                started = time.perf_counter()
                if queued is not None:
                    if not queued.run_one(cancelled):
                        stop_reason = "sources_exhausted"
                        break
                else:
                    self.graph.run(cancelled)
                blocks = list(self.graph.processing_blocks())
                runs.append(
                    {
                        "run": index + 1,
                        "passed": all(b.state in ("pass", "waiting", "inactive") for b in blocks),
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "blocks": describe(self.graph),
                    }
                )
            if all_images and len(runs) == limit:
                stop_reason = "sources_exhausted" if self.graph.sources_exhausted() else "max_runs"
            if cancelled():
                stop_reason = "cancelled"
        finally:
            if queued is not None:
                queued.close()
        complete = stop_reason not in ("cancelled", "max_runs")
        return {
            "project": str(self.source_path) if self.source_path else None,
            "revision": self.revision,
            "run_profile": self.graph.run_profile.name,
            "branch_workers": self.graph.branch_workers,
            "graphics_mode": self.graph.graphics_mode,
            "passed": bool(runs) and complete and all(r["passed"] for r in runs),
            "complete": complete,
            "stop_reason": stop_reason,
            "runs": runs,
            "ticket_results": self.graph.ticket_results.snapshot(),
            "activity": self.graph.activity.snapshot(),
            "active_recipe": self.graph.recipes.active_ref(),
        }

    def connection(self, source_kind, target_kind):
        """Describe a possible conversion; actual values still need runtime checks."""
        mode = connection_mode(source_kind, target_kind)
        return {"compatible": mode is not None, "conversion": mode}
