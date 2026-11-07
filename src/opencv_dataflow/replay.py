"""Run-local repeat policies for recorded image sequences."""

from .core import Block


class ReplaySource(Block):
    file_source = True
    replayable = True
    repeat_override = None

    @property
    def finite_source(self):
        return self.replayable and (self.repeat_override or self.parameters["repeat"]) == "stop"

    @property
    def replay_stalled(self):
        return self._sequence["finished"] and self.state == "error"

    def configure_replay(self, repeat, *, rewind=False):
        if rewind or self._sequence["finished"]:
            self.reset_runtime()
        self.repeat_override = repeat


def replay_sources(graph, keys=None):
    blocks = graph.blocks.values() if keys is None else (graph.blocks[key] for key in keys)
    for block in blocks:
        if isinstance(block, ReplaySource) and block.replayable:
            yield block
        if hasattr(block, "graph") and not getattr(block, "owns_workers", False):
            yield from replay_sources(block.graph)
