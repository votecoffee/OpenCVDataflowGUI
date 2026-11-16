"""Completion/error/cancellation details, block cleanup ordering and CLI final receipts."""

import json
import threading
import time

import pytest
from test_program_lifecycle import program

from opencv_dataflow.assistant_cli import main
from opencv_dataflow.core import Block, Graph, Result


@pytest.mark.parametrize("status", ["fail", "error"])
def test_after_run_receives_fail_and_shutdown_sees_closed_blocks(tmp_path, status):
    class Failed(Block):
        type_id = "test.failed"

        def process(self, inputs):
            if status == "error":
                raise ValueError("block failure")
            return Result({}, passed=False)

        def close(self):
            (tmp_path / "block-closed").touch()

    graph = Graph()
    graph.bind_project_dir(tmp_path)
    graph.add(Failed())
    graph.set_program(
        {
            "enabled": True,
            "scripts": {
                "after_run": f"assert context['status'] == {status!r}; state['failed'] = True",
                "shutdown": "from pathlib import Path; assert Path('block-closed').exists()",
            },
        }
    )
    try:
        graph.run()
        assert graph.program_status()["state"]["failed"] is True
    finally:
        graph.close()
    assert not graph.program_status()["error"]


def test_service_exit_still_calls_completion_hook_with_error_status(tmp_path, monkeypatch):
    graph, _ = program(tmp_path, scripts={"after_run": "state['status'] = context['status']"})
    graph.start_program()
    runtime = graph._program_runtime
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("service disappeared")

    monkeypatch.setattr(runtime, "_check_services", check)
    try:
        with pytest.raises(RuntimeError, match="service disappeared"):
            graph.run()
        assert graph.program_status()["state"]["status"] == "error"
    finally:
        graph.close()


def test_original_graph_exception_survives_failing_after_run(tmp_path, monkeypatch):
    graph, _ = program(tmp_path, scripts={"after_run": "raise ValueError('completion hook')"})

    def broken(*args):
        raise LookupError("original processing failure")

    monkeypatch.setattr(graph, "_run_iteration", broken)
    try:
        with pytest.raises(LookupError, match="original processing failure"):
            graph.run()
        assert "completion hook" in graph.program_status()["error"]
    finally:
        graph.close()


def test_cancelled_startup_unwinds_then_manual_run_can_restart(tmp_path):
    cancelled = threading.Event()
    graph, counter = program(
        tmp_path,
        scripts={
            "startup": (
                "from pathlib import Path; import time\n"
                "if not Path('entered').exists():\n    Path('entered').touch()\n    time.sleep(30)"
            )
        },
    )

    def cancel():
        end = time.monotonic() + 8
        while not (tmp_path / "entered").exists() and time.monotonic() < end:
            time.sleep(0.02)
        cancelled.set()

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        graph.run(cancelled.is_set)
        assert counter.calls == 0 and not graph.program_status()["error"]
        assert graph.program_status()["closed"]
        graph.run()
        assert counter.calls == 1
    finally:
        thread.join(timeout=10)
        graph.close()


def test_cancelled_processing_still_gets_after_run_status(tmp_path, monkeypatch):
    cancelled = threading.Event()
    graph, _ = program(tmp_path, scripts={"after_run": "state['status'] = context['status']"})
    monkeypatch.setattr(graph, "_run_iteration", lambda *args: cancelled.set())
    try:
        graph.run(cancelled.is_set)
        assert graph.program_status()["state"]["status"] == "cancelled"
        assert not graph.program_status()["error"]
    finally:
        graph.close()


@pytest.mark.parametrize(
    "shutdown,exit_code", [("print('closing')", 0), ("raise ValueError('close error')", 1)]
)
def test_cli_report_contains_shutdown_and_reports_cleanup_failure(
    tmp_path, capsys, shutdown, exit_code
):
    graph = Graph()
    graph.set_program({"enabled": True, "scripts": {"shutdown": shutdown}})
    path, report = tmp_path / "program.json", tmp_path / "receipt.json"
    graph.save(path)
    graph.close()
    # A layout-only program has no processing blocks: use an ordinary source.
    from opencv_dataflow.blocks import builtin_registry, demo_graph

    graph = demo_graph(builtin_registry())
    graph.set_program({"enabled": True, "scripts": {"shutdown": shutdown}})
    graph.save(path)
    graph.close()
    assert main(["run", str(path), "--report", str(report), "--summary"]) == exit_code
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is (exit_code == 0)
    receipt = json.loads(report.read_bytes())
    assert receipt["program"]["closed"]
    assert any("Program stopped" in event["message"] for event in receipt["activity"]["events"])
    if exit_code:
        assert "close error" in receipt["program"]["error"]
