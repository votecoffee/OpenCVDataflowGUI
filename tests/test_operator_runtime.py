"""Exercise the operator entry points and the actual Windows process tree boundary."""

import json
import os
import time
from pathlib import Path
from threading import Event

import pytest

from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.operator_runtime import RunOptions, run
from opencv_dataflow.ticket_history import recent

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/user_tools/assistant_workflow"


def test_operator_reopen_and_retain_ticket_history(tmp_path):
    history = tmp_path / "runtime.sqlite3"
    source = EXAMPLE / "assistant_workflow.json"
    negative = tmp_path / "fail.json"
    with ProjectSession.open(source) as session:
        session.apply(json.loads((EXAMPLE / "fail_edits.json").read_bytes()))
        session.save(negative)
    assert run(source, history) == 0
    assert run(negative, history) == 1
    assert [r["status"] for r in recent(history, sessions=True)] == ["fail", "pass"]
    rejected, accepted = recent(history)
    assert rejected["status"] == "final_fail"
    assert rejected["detail"]["final_fail_reasons"][0]["code"] == "mean_intensity"
    assert accepted["status"] == "final_pass"
    assert accepted["detail"]["completion"]["elapsed_ms"] >= 0
    assert accepted["revision"] != rejected["revision"]
    assert run(source, history, options=RunOptions(retention=1)) == 0
    assert len(recent(history)) == 1


def test_continuous_can_stop_and_run_shutdown_hook(tmp_path):
    history = tmp_path / "runtime.sqlite3"
    project = tmp_path / "run.json"
    shutdown = tmp_path / "closed"
    with ProjectSession.open(EXAMPLE / "assistant_workflow.json") as session:
        session.graph.set_program(
            {"enabled": True, "scripts": {"shutdown": f"open({str(shutdown)!r}, 'w').close()"}}
        )
        session.save(project)
    stop = Event()

    def progress(message):
        if message.startswith("Cycle 1 complete"):
            stop.set()

    assert (
        run(
            project,
            history,
            options=RunOptions(mode="continuous", interval_ms=500),
            cancelled=stop.is_set,
            progress=progress,
        )
        == 130
    )
    assert shutdown.is_file()
    assert recent(history, sessions=True)[0]["status"] == "stopped"
    assert recent(history)[0]["detail"]["completion"]["passed"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_blocked_acquisition_cannot_leave_child_running(tmp_path):
    history = tmp_path / "runtime.sqlite3"
    plugin = tmp_path / "slow.py"
    leak = tmp_path / "leaked"
    started = tmp_path / "child_started"
    child_code = (
        f"from pathlib import Path; import time; Path({str(started)!r}).touch(); "
        f"time.sleep(4); Path({str(leak)!r}).touch()"
    )
    plugin.write_text(
        "import subprocess, sys, time\n"
        "from opencv_dataflow.core import Block, Port\n"
        "class Slow(Block):\n"
        "    type_id = 'input.test_hung'\n"
        "    outputs = (Port('image'),)\n"
        "    def process(self, inputs):\n"
        f"        subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "        time.sleep(60)\n"
        "def register(registry):\n"
        "    registry.register(Slow)\n",
        encoding="utf-8",
    )
    project = tmp_path / "blocked.json"
    from opencv_dataflow.assistant_api import make_registry

    with ProjectSession(make_registry([plugin])) as session:
        session.apply([{"op": "add", "id": "source", "type": "input.test_hung"}])
        session.save(project)
    before = time.monotonic()
    assert run(project, history, plugins=[plugin], options=RunOptions(cycle_seconds=1)) == 2
    assert time.monotonic() - before < 15
    assert recent(history, sessions=True)[0]["status"] == "timeout"
    assert started.is_file(), "The simulated acquisition launched a real descendant"
    time.sleep(4)
    assert not leak.exists()
