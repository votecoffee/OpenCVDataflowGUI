"""Context for plugin dialogs, independent of Qt and the execution loop."""

import copy
from dataclasses import dataclass
from operator import index

import numpy as np

from .payloads import ImagePyramid, ValuePacket, image_with_payload, metadata


@dataclass(frozen=True)
class PluginImageOutput:
    block_id: str
    block_name: str
    port: str
    level_count: int


class PluginDialogContext:
    """Catalog cached image outputs published to the editor graph."""

    def __init__(self, graph, block):
        self.block_id = block.id
        self.project_dir = graph.project_dir
        self._graph = graph
        self.refresh_images()

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
                if isinstance(value, ImagePyramid):
                    levels = len(value.levels)
                elif isinstance(value, np.ndarray) and value.ndim in (2, 3) and value.size:
                    levels = 1
                else:
                    continue
                images[source.id, port.name] = (value, result.payloads.get(port.name, {}))
                outputs.append(PluginImageOutput(source.id, source.name, port.name, levels))
        self._images = images
        self.image_outputs = tuple(outputs)
        return self.image_outputs

    def _close(self):
        self._graph = None
        self._images.clear()
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
        value, output_payload = entry
        level = index(level)
        if isinstance(value, ImagePyramid):
            if not 0 <= level < len(value.levels):
                raise ValueError("Image pyramid level is outside the available range")
            value = value.levels[level]
            # The chosen level owns its frame, while the output packet owns ticket data.
            output_payload = {
                key: item
                for key, item in output_payload.items()
                if key not in ("spatial", "pyramid")
            }
        elif level != 0:
            raise ValueError("A single image only has level zero")
        payload = copy.deepcopy({**metadata(value), **output_payload})
        return ValuePacket(image_with_payload(value.copy(), payload), payload)
