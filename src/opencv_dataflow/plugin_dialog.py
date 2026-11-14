"""Context for plugin dialogs, independent of Qt and the execution loop."""

import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from operator import index
from threading import get_ident

import numpy as np

from .core import validate_value
from .payloads import (
    NO_VALUE,
    ImagePyramid,
    ValuePacket,
    image_with_payload,
    metadata,
    prepare_inputs,
    pyramid_level_image,
)


def _is_image(value):
    return isinstance(value, ImagePyramid) or (
        isinstance(value, np.ndarray) and value.ndim in (2, 3) and value.size > 0
    )


def _snapshot(value, payload, *, level):
    level = index(level)
    if isinstance(value, ImagePyramid):
        if not 0 <= level < len(value.levels):
            raise ValueError("Image pyramid level is outside the available range")
        value, payload = pyramid_level_image(value, payload, level)
    else:
        if level != 0:
            raise ValueError("A single image only has level zero")
        payload = {**metadata(value), **payload}
    return ValuePacket(image_with_payload(value.copy(), payload), copy.deepcopy(payload))


@dataclass(frozen=True)
class PluginImageOutput:
    block_id: str
    block_name: str
    port: str
    level_count: int


@dataclass(frozen=True)
class PluginProcessedInputs:
    """Private packets from one completed execution, copied on the dialog thread."""

    snapshot_id: str
    block_id: str
    output_revision: int
    execution_count: int
    passed: bool
    inputs: dict[str, ValuePacket]


@dataclass(frozen=True)
class PluginConnectedInputs:
    """Private wired values from one editor publication, not a physical exposure."""

    snapshot_id: str
    publication_id: str
    block_id: str
    output_revision: int
    inputs: dict[str, ValuePacket]
    missing_inputs: tuple[str, ...]
    sources: dict[str, dict]


class PluginDialogContext:
    """Catalog cached image outputs published to the editor graph."""

    def __init__(self, graph, block, *, input_levels=None):
        self.block_id = block.id
        self.project_dir = graph.project_dir
        self._graph = graph
        self._block = block
        self._thread_id = get_ident()
        self._context_id = uuid.uuid4().hex
        self._input_levels = dict(block.input_levels if input_levels is None else input_levels)
        self._input_wires = {
            wire.input: (wire.source, wire.output)
            for wire in graph.connections
            if wire.target == block.id
        }
        self._output_revision = 0
        self._output_callbacks = {}
        self._dispatching_outputs = False
        self.refresh_images()

    @property
    def output_revision(self):
        """Number of editor output publications since this dialog opened."""
        return self._output_revision

    def subscribe_outputs(self, callback: Callable[[int], None]) -> Callable[[], None]:
        """Call back on the GUI thread with the newest published output revision."""
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        if not callable(callback):
            raise TypeError("The output subscriber must be callable")
        token = object()
        self._output_callbacks[token] = callback

        def unsubscribe():
            self._output_callbacks.pop(token, None)

        return unsubscribe

    def _outputs_published(self):
        self._output_revision += 1
        return bool(self._output_callbacks) and not self._dispatching_outputs

    def _notify_outputs(self, report_error):
        revision = self.output_revision
        self._dispatching_outputs = True
        try:
            for token, callback in tuple(self._output_callbacks.items()):
                if revision != self.output_revision:
                    break
                if token not in self._output_callbacks:
                    continue
                try:
                    callback(revision)
                except (Exception, SystemExit):
                    self._output_callbacks.pop(token, None)
                    report_error("Plugin output subscriber failed and was removed")
        finally:
            self._dispatching_outputs = False
        return bool(self._output_callbacks) and revision != self.output_revision

    def refresh_images(self):
        """Refresh the catalog on the GUI thread and return the available outputs."""
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        images = {}
        outputs = []
        for source in self._graph.blocks.values():
            result = source.last_result
            if result is None:
                continue
            for port in source.outputs:
                value = result.outputs.get(port.name)
                if not _is_image(value):
                    continue
                levels = len(value.levels) if isinstance(value, ImagePyramid) else 1
                images[source.id, port.name] = (value, result.payloads.get(port.name, {}))
                outputs.append(PluginImageOutput(source.id, source.name, port.name, levels))
        self._images = images
        self._processed_images = {
            name: (value, self._block.input_payloads.get(name, {}))
            for name, value in self._block.last_inputs.items()
            if _is_image(value)
        }
        self.image_outputs = tuple(outputs)
        return self.image_outputs

    def _close(self):
        self._graph = None
        self._block = None
        self._output_callbacks.clear()
        self._images.clear()
        self._processed_images.clear()
        self.image_outputs = ()

    def snapshot_image(self, block_id, port, *, level=0):
        """Copy a cached image and its wire payload. Pyramid levels are largest-first."""
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        entry = self._images.get((block_id, port))
        if entry is None:
            raise ValueError(
                "No cached image at that output. Run the graph and refresh the image catalog."
            )
        return _snapshot(*entry, level=level)

    def snapshot_input_image(self, port, *, source="connected", level=None):
        """Copy the wired or last processed image for the selected block's input."""
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        target = next((item for item in self._block.inputs if item.name == port), None)
        if target is None or target.kind not in ("image", "depth", "pyramid", "any"):
            raise ValueError(f"{port} is not an image-capable input")
        if source == "processed":
            entry = self._processed_images.get(port)
            if entry is None:
                raise ValueError(f"No processed image is cached for {port}")
        elif source == "connected":
            wire = self._input_wires.get(port)
            if wire is None:
                raise ValueError(f"Connect a source to {port} to capture its image")
            entry = self._images.get(wire)
            if entry is None:
                raise ValueError(f"No connected image is cached for {port}. Run the graph first.")
            if level is None:
                values, payloads = prepare_inputs(
                    self._block, {port: ValuePacket(*entry)}, input_levels=self._input_levels
                )
                validate_value(target, values[port])
                entry = values[port], payloads[port]
        else:
            raise ValueError("Image input source must be connected or processed")
        return _snapshot(*entry, level=0 if level is None else level)

    def snapshot_input_data(self, port):
        """Copy the latest published value and payload on a wired data input."""
        return self._snapshot_input_value(port, "data")

    def snapshot_input_cloud(self, port):
        """Copy the latest published point cloud and payload on a wired cloud input."""
        return self._snapshot_input_value(port, "cloud")

    def snapshot_processed_inputs(self, ports=None):
        """Copy retained inputs together; never combine separately read upstream outputs.

        Call on the context's owning GUI thread before handing the bundle to a worker.
        The snapshot ID identifies this private copy, not a synchronized acquisition.
        """
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        if get_ident() != self._thread_id:
            raise RuntimeError("Capture processed inputs on the plugin dialog's owning thread")
        block = self._block
        if not block._execution_lock.acquire(blocking=False):
            raise RuntimeError("The block is active; capture after its execution finishes")
        try:
            if block.timing.started is not None:
                raise RuntimeError("The block is active; capture after its execution finishes")
            if block.last_result is None or block.timing.count == 0:
                raise ValueError("No completed block execution is available to capture")
            if not block.run_profile.input_previews:
                raise ValueError("Processed inputs were not retained. Run again in Debug profile.")
            if (block.inputs and not block.last_inputs) or any(
                port.required and port.name not in block.last_inputs for port in block.inputs
            ):
                raise ValueError("Processed inputs are unavailable. Run again in Debug profile.")
            declared = {port.name for port in block.inputs}
            if ports is None:
                names = tuple(port.name for port in block.inputs if port.name in block.last_inputs)
            else:
                if isinstance(ports, str):
                    raise TypeError("Pass input names as a collection, not a single string")
                names = tuple(ports)
                if any(not isinstance(name, str) or name not in declared for name in names):
                    raise ValueError("Processed capture must name declared block inputs")
                if len(set(names)) != len(names):
                    raise ValueError("Processed capture input names must be unique")
                missing = [name for name in names if name not in block.last_inputs]
                if missing:
                    raise ValueError(f"No processed input is retained for: {', '.join(missing)}")
            packets = copy.deepcopy(
                {
                    name: ValuePacket(block.last_inputs[name], block.input_payloads.get(name, {}))
                    for name in names
                }
            )
            return PluginProcessedInputs(
                snapshot_id=uuid.uuid4().hex,
                block_id=block.id,
                output_revision=self.output_revision,
                execution_count=block.timing.count,
                passed=bool(block.last_result.passed),
                inputs=packets,
            )
        finally:
            block._execution_lock.release()

    def snapshot_connected_inputs(self, ports=None):
        """Capture wired values together, including for an unconfigured target.

        The owning thread serializes editor publications. Source execution locks
        reject direct concurrent execution without waiting. Missing optional
        values are reported; required inputs must have a retained connected value.
        No block is executed and physical acquisition synchronization is not inferred.
        """
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        if get_ident() != self._thread_id:
            raise RuntimeError("Capture connected inputs on the plugin dialog's owning thread")
        declared = {port.name: port for port in self._block.inputs}
        if isinstance(ports, str):
            raise TypeError("Pass input names as a collection, not a single string")
        names = tuple(declared) if ports is None else tuple(ports)
        if any(not isinstance(name, str) or name not in declared for name in names):
            raise ValueError("Connected capture must name declared block inputs")
        if len(set(names)) != len(names):
            raise ValueError("Connected capture input names must be unique")
        wires = {w.input: w for w in self._graph.connections if w.target == self.block_id}
        source_ids = {wires[name].source for name in names if name in wires}
        owners = {key: self._graph.blocks[key] for key in source_ids}
        owners[self.block_id] = self._block
        locked = []
        revision = self.output_revision
        try:
            for key in sorted(owners):
                owner = owners[key]
                if not owner._execution_lock.acquire(blocking=False):
                    raise RuntimeError("A connected block is active; capture after it finishes")
                locked.append(owner)
                if owner.timing.started is not None:
                    raise RuntimeError("A connected block is active; capture after it finishes")
            packets, sources, missing = {}, {}, []
            for name in names:
                wire = wires.get(name)
                source = owners[wire.source] if wire else None
                result = source.last_result if source is not None else None
                value = result.outputs.get(wire.output, NO_VALUE) if result else NO_VALUE
                if value is NO_VALUE:
                    if declared[name].required or (wire and declared[name].required_when_connected):
                        raise ValueError(f"No connected value is cached for required input {name}")
                    missing.append(name)
                    continue
                packets[name] = ValuePacket(value, result.payloads.get(wire.output, {}))
                sources[name] = {
                    "block_id": source.id, "port": wire.output,
                    "execution_count": source.timing.count, "state": source.state,
                    "retained": not bool(source.current_emissions.get(wire.output)),
                }
            values, payloads = prepare_inputs(
                self._block, packets, input_levels=self._input_levels
            )
            for name, value in values.items():
                validate_value(declared[name], value)
            copied = copy.deepcopy({
                name: ValuePacket(value, payloads[name]) for name, value in values.items()
            })
            if revision != self.output_revision:
                raise RuntimeError("Editor publication changed during connected capture; retry")
            return PluginConnectedInputs(
                snapshot_id=uuid.uuid4().hex,
                publication_id=f"{self._context_id}:{revision}",
                block_id=self.block_id, output_revision=revision,
                inputs=copied, missing_inputs=tuple(missing), sources=sources,
            )
        finally:
            for owner in reversed(locked):
                owner._execution_lock.release()

    def _snapshot_input_value(self, port, kind):
        if self._graph is None:
            raise RuntimeError("The plugin dialog has closed")
        target = next((item for item in self._block.inputs if item.name == port), None)
        if target is None or target.kind != kind:
            raise ValueError(f"{port} is not a {kind} input")
        wire = self._input_wires.get(port)
        if wire is None:
            raise ValueError(f"Connect a source to {port} to capture its {kind}")
        source, output = wire
        block = self._graph.blocks.get(source)
        result = block.last_result if block is not None else None
        if result is None or result.outputs.get(output, NO_VALUE) is NO_VALUE:
            raise ValueError(f"No connected {kind} is cached for {port}. Run the graph first.")
        packet = ValuePacket(result.outputs[output], result.payloads.get(output, {}))
        validate_value(target, packet)
        return copy.deepcopy(packet)
