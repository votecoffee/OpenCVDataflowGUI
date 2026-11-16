"""Real owned-process cleanup and program hooks, without ROS or equipment."""

import copy
import ctypes
import json
import os
import sys
import time

import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.managed_process import ManagedProcess
from opencv_dataflow.program_config import validate_program
from opencv_dataflow.project_xml import export_project_xml, read_project_xml


class Counter(Block):
    type_id, title = "test.counter", "Counter"
    outputs = (Port("count", "integer"),)

    def __init__(self):
        super().__init__(block_id="counter")
        self.calls = 0

    def process(self, inputs):
        self.calls += 1
        return Result({"count": self.calls})


def alive(pid):
    if os.name == "nt":
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        api.OpenProcess.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for(predicate, timeout=8):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "Timed out waiting for test process state"
        time.sleep(0.02)


def program(tmp_path, **settings):
    graph = Graph()
    graph.bind_project_dir(tmp_path)
    counter = graph.add(Counter())
    graph.set_program({"enabled": True, **settings})
    return graph, counter


def service(tmp_path, source=None, **options):
    path = tmp_path / "service.py"
    path.write_text(
        source
        or (
            "import os, pathlib, subprocess, sys\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "pids = [os.getpid(), child.pid]\n"
            "pathlib.Path('pids.json').write_text(__import__('json').dumps(pids))\n"
            "print('TEST_READY', flush=True)\n"
            "sys.stdin.readline()\n"
            "pathlib.Path('stopped.txt').write_text('graceful')\n"
        ),
        encoding="utf-8",
    )
    return {
        "id": "demo",
        "argv": ["{python}", "-u", "{project_dir}/service.py"],
        "ready_text": "TEST_READY",
        "startup_timeout_seconds": 3,
        "shutdown_timeout_seconds": 0.2,
        **options,
    }


def test_disabled_and_never_run_programs_do_not_execute(tmp_path):
    graph = Graph()
    assert "program" not in graph.to_dict()
    graph.bind_project_dir(tmp_path)
    graph.set_program({"scripts": {"startup": "raise AssertionError('must not execute')"}})
    graph.run()
    graph.close()
    enabled = Graph.from_dict(
        {
            **graph.to_dict(),
            "program": {
                "enabled": True,
                "scripts": {"shutdown": "raise AssertionError('never started')"},
            },
        },
        builtin_registry(),
        project_dir=tmp_path,
    )
    assert validate_graph(enabled)["valid"]
    enabled.close()
    assert not enabled.activity.snapshot()["events"]


def test_hooks_start_once_wrap_each_cycle_and_shutdown_once(tmp_path):
    scripts = {
        event: f"state.setdefault('events', []).append('{event}')"
        for event in ("startup", "before_run", "after_run", "shutdown")
    }
    scripts["after_run"] += "; assert context['status'] == 'pass'; print('inspection complete')"
    scripts["shutdown"] += (
        "\nfrom pathlib import Path\n"
        "Path(context['project_dir'], 'shutdown.json').write_text(__import__('json').dumps(state))"
    )
    graph, counter = program(tmp_path, scripts=scripts, variables={"seed": 7})
    try:
        assert not (tmp_path / "shutdown.json").exists()
        graph.run()
        graph.run()
        state = graph.program_status()
        assert counter.calls == 2 and state["run_index"] == 2
        assert state["state"]["events"] == [
            "startup",
            "before_run",
            "after_run",
            "before_run",
            "after_run",
        ]
        assert graph.program["variables"] == {"seed": 7}
        assert "inspection complete" in str(graph.activity.snapshot())
    finally:
        graph.close()
    saved = (tmp_path / "shutdown.json").read_bytes()
    assert json.loads(saved)["events"][-1] == "shutdown"
    graph.close()
    assert (tmp_path / "shutdown.json").read_bytes() == saved


def test_service_survives_runs_and_cleans_owned_descendants_only(tmp_path):
    graph, counter = program(tmp_path, services=[service(tmp_path)])
    unrelated = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        tmp_path,
        os.environ.copy(),
    )
    pids = []
    try:
        graph.run()
        pids = json.loads((tmp_path / "pids.json").read_bytes())
        assert all(alive(pid) for pid in pids)
        command_pid = graph.program_status()["services"]["demo"]["pid"]
        graph.run()
        assert graph.program_status()["services"]["demo"]["pid"] == command_pid
        assert counter.calls == 2
        graph.close()
        wait_for(lambda: not any(alive(pid) for pid in pids))
        assert (tmp_path / "stopped.txt").read_text() == "graceful"
        assert unrelated.process.poll() is None
    finally:
        graph.close()
        unrelated.close()


@pytest.mark.parametrize(
    "source,pattern",
    [
        ("print('started', flush=True)\n", "exited during startup"),
        ("import time\nprint('started', flush=True)\ntime.sleep(60)", "readiness timed out"),
    ],
)
def test_failed_service_start_unwinds_and_runs_shutdown(tmp_path, source, pattern):
    graph, counter = program(
        tmp_path,
        services=[
            service(tmp_path, source, startup_timeout_seconds=3, shutdown_timeout_seconds=0.05)
        ],
        scripts={"shutdown": "from pathlib import Path; Path('unwound').touch()"},
    )
    try:
        with pytest.raises((RuntimeError, TimeoutError), match=pattern):
            graph.run()
        assert counter.calls == 0
        status = graph.program_status()
        assert status["closed"] and status["error"]
        assert not status["services"]["demo"]["running"]
        assert (tmp_path / "unwound").exists()
    finally:
        graph.close()


def test_script_timeout_kills_its_children_and_does_not_run_blocks(tmp_path):
    graph, counter = program(
        tmp_path,
        script_timeout_seconds=3,
        scripts={
            "startup": (
                "import os, pathlib, subprocess, sys, time\n"
                "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                "pathlib.Path('hook-child.txt').write_text(str(p.pid))\ntime.sleep(60)"
            )
        },
    )
    try:
        with pytest.raises(TimeoutError, match="script timed out"):
            graph.run()
        assert counter.calls == 0
        pid = int((tmp_path / "hook-child.txt").read_text())
        wait_for(lambda: not alive(pid))
        assert any(
            e["severity"] == "ERROR" and e["exception"] for e in graph.activity.snapshot()["events"]
        )
    finally:
        graph.close()


@pytest.mark.parametrize(
    "source,pattern",
    [
        ("raise ValueError('bad hook')", "bad hook"),
        ("raise SystemExit(5)", "SystemExit"),
        ("state['invalid'] = float('nan')", "JSON"),
        ("state['large'] = 'x' * 70000", "64 KiB"),
        ("import os; os._exit(0)", "bounded result"),
    ],
)
def test_contained_hook_failures_keep_previous_state(tmp_path, source, pattern):
    graph, counter = program(tmp_path, variables={"kept": True}, scripts={"before_run": source})
    try:
        with pytest.raises(RuntimeError, match=pattern):
            graph.run()
        assert counter.calls == 0 and graph.program_status()["state"] == {"kept": True}
    finally:
        graph.close()


def test_shutdown_error_still_cleans_services_and_is_logged(tmp_path):
    graph, _ = program(
        tmp_path,
        services=[service(tmp_path)],
        scripts={"shutdown": "raise RuntimeError('shutdown failed')"},
    )
    graph.run()
    pids = json.loads((tmp_path / "pids.json").read_bytes())
    graph.close()
    wait_for(lambda: not any(alive(pid) for pid in pids))
    events = graph.activity.snapshot()["events"]
    assert any(e["severity"] == "ERROR" and "shutdown failed" in e["message"] for e in events)


def test_borrowed_gui_style_execution_does_not_close_owner_service(tmp_path):
    owner, _ = program(tmp_path, services=[service(tmp_path)])
    worker = Graph()
    worker.program = copy.deepcopy(owner.program)
    worker.bind_project_dir(tmp_path)
    worker.borrow_program_runtime(owner)
    try:
        worker.run()
        worker.close()
        assert owner.program_status()["started"]
        assert owner.program_status()["services"]["demo"]["running"]
        owner.run()
        assert owner.program_status()["run_index"] == 2
    finally:
        owner.close()


def test_cancelled_start_does_not_launch_or_execute_hooks(tmp_path):
    graph, counter = program(
        tmp_path,
        services=[service(tmp_path)],
        scripts={"startup": "raise AssertionError('cancelled')"},
    )
    try:
        graph.run(lambda: True)
        assert counter.calls == 0 and not graph.program_status()["started"]
        assert not (tmp_path / "pids.json").exists()
    finally:
        graph.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"unknown": True},
        {"enabled": "true"},
        {"variables": []},
        {"script_timeout_seconds": True},
        {"scripts": {"opening": "pass"}},
        {"assets": ["../private.py"]},
        {"services": [{"id": "1", "argv": ["python"]}]},
        {"services": [{"argv": []}]},
        {"services": [{"argv": ["python"], "runtime": "shell"}]},
        {"services": [{"argv": ["python"], "environment": {"BAD=KEY": "value"}}]},
        {"services": [{"argv": ["python"], "shutdown_timeout_seconds": 100}]},
    ],
)
def test_bad_program_settings_rejected_without_execution(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_program(bad)


def test_static_validation_json_xml_assets_and_save_as(tmp_path):
    registry = builtin_registry()
    graph = demo_graph(registry)
    graph.bind_project_dir(tmp_path)
    asset = tmp_path / "helper.py"
    asset.write_bytes(b"# retained helper\r\n")
    graph.set_program(
        {
            "name": "Example",
            "enabled": True,
            "assets": ["helper.py"],
            "scripts": {"startup": "state['ready'] = True"},
        }
    )
    try:
        assert validate_graph(graph)["valid"]
        from opencv_dataflow.program_documentation import render_program_documentation

        assert "Program settings and lifecycle" in render_program_documentation(graph, "Example")
        xml = tmp_path / "saved.xml"
        export_project_xml(graph.to_dict(), xml)
        restored = Graph.from_dict(read_project_xml(xml), registry, project_dir=tmp_path)
        assert restored.program == graph.program and not restored.program_status()["started"]
        restored.close()
        destination = tmp_path / "new"
        destination.mkdir()
        graph.save(destination / "program.json")
        assert (destination / "helper.py").read_bytes() == asset.read_bytes()
        assert not graph.program_status()["started"]
        graph.program["scripts"]["before_run"] = "if broken"
        validation = validate_graph(graph)
        assert not validation["valid"] and validation["issues"][0]["kind"] == "program"
    finally:
        graph.close()


def test_assistant_api_settings_and_receipts(tmp_path):
    with ProjectSession(directory=tmp_path) as session:
        session.apply(
            [
                {
                    "op": "settings",
                    "fields": {
                        "program": {
                            "enabled": True,
                            "scripts": {"before_run": "state['runs'] = state.get('runs', 0) + 1"},
                        }
                    },
                }
            ]
        )
        receipt = session.run(repeat=2)
        assert receipt["program"]["run_index"] == 2
        assert receipt["program"]["state"]["runs"] == 2


def test_process_constructor_failure_does_not_launch_command(monkeypatch, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows ownership assignment")
    from opencv_dataflow.managed_process import WindowsJob

    def reject(*args):
        raise OSError("ownership unavailable")

    monkeypatch.setattr(WindowsJob, "assign", reject)
    marker = tmp_path / "should-not-exist"
    with pytest.raises(OSError, match="ownership unavailable"):
        ManagedProcess(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
            tmp_path,
            os.environ.copy(),
        )
    assert not marker.exists()
