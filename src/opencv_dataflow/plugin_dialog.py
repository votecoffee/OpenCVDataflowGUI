"""Context for plugin dialogs, independent of Qt and the execution loop."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np

from .core import validate_value
from .payloads import (
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


class PluginDialogContext:
    """Catalog cached image outputs published to the editor graph."""

    def __init__(self, graph, block, *, input_levels=None):
        self.block_id = block.id
        self.project_dir = graph.project_dir
        self._graph = graph
        self._block = block
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
