"""Exercise Stop during the deterministic gap between clearing and processing."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from test_sequence import pump

from opencv_dataflow.app import MainWindow, Runner
from opencv_dataflow.core import Graph


def memory_graph(registry):
    graph = Graph()
    source = graph.add(registry.create("input.sequence_demo", block_id="source"))
    memory = graph.add(registry.create("sequence.image_memory", block_id="memory"))
    graph.connect(source.id, "image", memory.id, "image")
    return graph


@pytest.mark.parametrize(("interval", "stop_iteration"), [(0, 2), (60_000, 3)])
def test_stop_retains_last_complete_result_even_when_ui_is_throttled(
    app, monkeypatch, interval, stop_iteration
):
    window = MainWindow()
    window.replace_graph(memory_graph(window.registry))
    original_init = Runner.__init__
    original_clear = Graph.clear_results
    blocked, release = Event(), Event()
    iterations, expected_images = [], []

    def configure(runner, graph, continuous, registry):
        original_init(runner, graph, continuous, registry)
        graph.set_run_profile(
            replace(graph.run_profile, ui_interval_ms=interval, iteration_pause_ms=0)
        )

    def clear(graph, preserve_history=False):
        worker = window.runner
        if worker and graph is worker.graph and worker.continuous and preserve_history:
            iterations.append(1)
            if len(iterations) == stop_iteration:
                expected_images.append(graph.blocks["source"].last_output_images["image"].copy())
                original_clear(graph, preserve_history)
                blocked.set()
                release.wait(10)
                return
        original_clear(graph, preserve_history)

    monkeypatch.setattr(Runner, "__init__", configure)
    monkeypatch.setattr(Graph, "clear_results", clear)
    try:
        window.start_run(True)
        pump(app, blocked.is_set)
        memory = window.graph.blocks["memory"]
        assert memory.last_result.outputs["activations"] == 1
        window.stop_run()
        release.set()
        pump(app, lambda: window.runner is None)
        assert memory.state in ("pass", "waiting")
        assert memory.last_result.outputs["activations"] == stop_iteration - 1
        np.testing.assert_array_equal(
            window.graph.blocks["source"].last_output_images["image"], expected_images[0]
        )
        window.start_run(False)
        pump(app, lambda: window.runner is None)
        assert memory.last_result.outputs["activations"] == stop_iteration
    finally:
        release.set()
        if window.runner:
            window.stop_run()
            pump(app, lambda: window.runner is None)
        window.close()


def test_stop_after_completion_still_publishes_that_iteration(app, monkeypatch):
    window = MainWindow()
    window.replace_graph(memory_graph(window.registry))
    original_run = Graph.run

    def stop_after_completion(graph, *args, **kwargs):
        result = original_run(graph, *args, **kwargs)
        worker = window.runner
        if worker and graph is worker.graph:
            worker.stop()
        return result

    monkeypatch.setattr(Graph, "run", stop_after_completion)
    try:
        window.start_run(True)
        pump(app, lambda: window.runner is None)
        assert window.graph.blocks["memory"].last_result.outputs["activations"] == 1
    finally:
        if window.runner:
            window.stop_run()
            pump(app, lambda: window.runner is None)
        window.close()
