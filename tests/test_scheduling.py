"""Dependency-driven branch scheduling and saved worker policy."""

import threading

import pytest

from opencv_dataflow import core
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result


def test_auto_releases_fast_descendant_before_unrelated_branch_finishes(monkeypatch):
    monkeypatch.setattr(core, "automatic_workers", lambda: 2)
    slow_started = threading.Event()
    fast_finished = threading.Event()
    calls = []
    owner = threading.get_ident()

    class Source(Block):
        outputs = (Port("value", "number"),)

        def process(self, inputs):
            assert threading.get_ident() == owner
            return Result({"value": 1})

    class Step(Block):
        inputs = outputs = (Port("value", "number"),)
        parallel_safe = True

        def process(self, inputs):
            return Result({"value": inputs["value"] + 1})

    class Slow(Step):
        def process(self, inputs):
            slow_started.set()
            assert fast_finished.wait(3)
            calls.append("slow")
            return super().process(inputs)

    class FastEnd(Step):
        def process(self, inputs):
            assert slow_started.wait(3)
            calls.append("fast")
            fast_finished.set()
            return super().process(inputs)

    class Join(Block):
        inputs = (Port("a", "number"), Port("b", "number"))
        outputs = (Port("sum", "number"),)

        def process(self, inputs):
            assert threading.get_ident() == owner
            calls.append("join")
            return Result({"sum": inputs["a"] + inputs["b"]})

    graph = Graph()
    source, slow, fast, end, join = [
        graph.add(cls()) for cls in (Source, Slow, Step, FastEnd, Join)
    ]
    for target in (slow, fast):
        graph.connect(source.id, "value", target.id, "value")
    graph.connect(fast.id, "value", end.id, "value")
    graph.connect(slow.id, "value", join.id, "a")
    graph.connect(end.id, "value", join.id, "b")

    graph.run()

    assert all(block.state == "pass" for block in graph.blocks.values())
    assert calls == ["fast", "slow", "join"]
    assert join.last_result.outputs["sum"] == 5


def test_serial_setting_and_plugin_worker_policy():
    owner = threading.get_ident()
    threads = []

    class Probe(Block):
        def process(self, inputs):
            threads.append(threading.get_ident())
            return Result({})

    class Worker(Probe):
        parallel_safe = True

    class Writer(Worker):
        block_type = "Output"

    graph = Graph()
    for cls in (Probe, Writer, Worker):
        graph.add(cls())
    graph.set_branch_workers(2)
    graph.run()
    assert threads.count(owner) == 2

    threads.clear()
    graph.set_branch_workers(1)
    graph.run()
    assert threads == [owner] * 3


def test_branch_worker_setting_round_trips_in_editor(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    graph = Graph()
    assert "branch_workers" not in graph.to_dict()
    with pytest.raises(ValueError, match="Branch workers"):
        graph.set_branch_workers(True)

    window = MainWindow()
    window.replace_graph(graph)
    window.branch_workers_choice.setCurrentText("Serial")
    assert graph.branch_workers == 1
    assert window.dirty
    path = tmp_path / "job.json"
    graph.save(path)
    restored = Graph.load(path, builtin_registry())
    window.replace_graph(restored)
    assert window.branch_workers_choice.currentText() == "Serial"
    assert not window.dirty
    window.close()


def test_stop_skips_ready_and_dependent_work_and_joins_active_worker():
    stop = threading.Event()
    started = threading.Event()
    finished = threading.Event()

    class Worker(Block):
        inputs = (Port("value", "number", required=False),)
        outputs = (Port("value", "number"),)
        parallel_safe = True

        def process(self, inputs):
            pytest.fail("A skipped block was executed")

    class Active(Worker):
        def process(self, inputs):
            started.set()
            assert stop.wait(3)
            finished.set()
            return Result({"value": 1})

    class Stopper(Worker):
        def process(self, inputs):
            assert started.wait(3)
            stop.set()
            return Result({"value": 1})

    graph = Graph()
    graph.set_branch_workers(2)
    # Insert the shallow branch first to exercise priority when only two slots are available.
    _, ready, active, child = [graph.add(cls()) for cls in (Stopper, Worker, Active, Worker)]
    graph.connect(active.id, "value", child.id, "value")
    graph.run(cancelled=stop.is_set)
    assert finished.is_set()
    assert ready.state == child.state == "skipped"
