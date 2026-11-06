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
    """Catalog the current graph's cached image outputs at dialog creation."""

    def __init__(self, graph, block):
        self.block_id = block.id
        self.project_dir = graph.project_dir
        self._images = {}
        outputs = []
        for source in graph.blocks.values():
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
                self._images[source.id, port.name] = (value, result.payloads.get(port.name, {}))
                outputs.append(PluginImageOutput(source.id, source.name, port.name, levels))
        self.image_outputs = tuple(outputs)

    def snapshot_image(self, block_id, port, *, level=0):
        """Copy a cached image and its wire payload. Pyramid levels are largest-first."""
        entry = self._images.get((block_id, port))
        if entry is None:
            raise ValueError(
                "Run the graph to cache an image at that output before opening the dialog"
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
