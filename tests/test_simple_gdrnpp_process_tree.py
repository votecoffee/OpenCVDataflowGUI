"""Real owned child processes, busy cancellation, closure and request recovery.

The stand-in worker needs only Python's standard library, so the installed host
can exercise the same bridge without importing or installing a neural runtime.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from opencv_dataflow.simple_gdrnpp import bridge
from opencv_dataflow.simple_gdrnpp.process_tree import ProcessTree

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows native lifecycle qualification")

STAND_IN = """
import json, os, subprocess, sys, time
from pathlib import Path
from zipfile import ZipFile
root = Path(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    job = root / request['id']
    metadata = json.loads((job / 'metadata.json').read_text())
    if request['op'] in ('busy', 'exit'):
        child = subprocess.Popen([sys.argv[2], '-c', 'import time; time.sleep(60)'],
                                 stdin=subprocess.DEVNULL)
        marker = root / 'ready.tmp'
        marker.write_text(json.dumps(dict(parent=os.getpid(), child=child.pid)))
        marker.replace(root / 'ready.json')
        if request['op'] == 'exit':
            while not (root / 'exit.now').exists():
                time.sleep(.01)
            os._exit(7)
        time.sleep(60)
    if request['op'] == 'error':
        print(json.dumps(dict(id=request['id'], ok=False, error='controlled failure')), flush=True)
        continue
    result = job / 'result'
    result.mkdir()
    (result / 'metadata.json').write_text(json.dumps(dict(nonce=metadata.get('nonce'),
                                                        session=os.getpid())))
    with ZipFile(result / 'arrays.npz', 'w'):
        pass
    print(json.dumps(dict(id=request['id'], ok=True)), flush=True)
"""


class Observation:
    """Retain a handle to a freshly observed fixture, never a PID-only kill."""

    def __init__(self, pid):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.open = kernel.OpenProcess
        self.open.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        self.open.restype = ctypes.c_void_p
        self.wait = kernel.WaitForSingleObject
        self.wait.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.close = kernel.CloseHandle
        self.close.argtypes = [ctypes.c_void_p]
        self.kill = kernel.TerminateProcess
        self.kill.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.handle = self.open(0x100001, False, pid)  # SYNCHRONIZE | PROCESS_TERMINATE
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def stopped(self, timeout=5000):
        return self.wait(self.handle, timeout) == 0

    def release(self):
        # Failure cleanup is restricted to this retained, owned fixture handle.
        if not self.stopped(0):
            self.kill(self.handle, 1)
            assert self.stopped()
        self.close(self.handle)


def wait_ready(client):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        root = getattr(client, "root", None)
        if root is not None and (root / "ready.json").exists():
            return json.loads((root / "ready.json").read_text())
        time.sleep(0.01)
    raise AssertionError("Controlled worker did not become busy")


@pytest.fixture(params=["direct", "venv_launcher"])
def client(request, monkeypatch):
    executable = sys._base_executable if request.param == "direct" else sys.executable

    def start(command, **kwargs):
        return ProcessTree(
            [executable, "-u", "-c", STAND_IN, command[-1], sys._base_executable], **kwargs
        )

    monkeypatch.setattr(bridge, "ProcessTree", start)
    value = bridge.WorkerClient(python=executable, timeout=15)
    try:
        yield value
    finally:
        value.close()


def launch_request(client, operation, cancelled=lambda: False):
    outcome = {}

    def run():
        try:
            outcome["result"] = client.call(operation, {"nonce": "old"}, cancelled=cancelled)
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize("action", ["cancel", "timeout", "close"])
def test_busy_request_retires_owned_children_and_readers_before_storage(client, action):
    cancel = threading.Event()
    if action == "timeout":
        client.timeout = 0.75
    thread, outcome = launch_request(client, "busy", cancel.is_set)
    pids = wait_ready(client)
    child = Observation(pids["child"])
    parent = Observation(pids["parent"])
    root, process, readers = client.root, client.process, list(client._readers)
    try:
        if action == "cancel":
            cancel.set()
        elif action == "close":
            client.close()
        thread.join(10)
        assert not thread.is_alive() and "result" not in outcome
        assert isinstance(outcome["error"], TimeoutError if action == "timeout" else RuntimeError)
        assert parent.stopped() and child.stopped() and process.poll() is not None
        assert all(not reader.is_alive() for reader in readers)
        assert client.process is None and not root.exists()
        if action == "close":
            with pytest.raises(RuntimeError, match="closed"):
                client.call("echo")
            client.close()
        else:
            client.timeout = 15
            result, arrays = client.call("echo", {"nonce": "fresh"})
            assert result["nonce"] == "fresh" and arrays == {}
            assert result["session"] != pids["parent"] and list(client.root.iterdir()) == []
    finally:
        child.release()
        parent.release()
        thread.join(5)


@pytest.mark.parametrize("operation", ["exit", "error"])
def test_failed_or_exited_worker_can_restart_without_stale_results(client, operation):
    if operation == "exit":
        thread, outcome = launch_request(client, operation)
        pids = wait_ready(client)
        child = Observation(pids["child"])
        try:
            (client.root / "exit.now").touch()
            thread.join(10)
            assert not thread.is_alive() and "result" not in outcome
            assert isinstance(outcome["error"], RuntimeError) and child.stopped()
        finally:
            child.release()
    else:
        with pytest.raises(ValueError, match="controlled failure"):
            client.call(operation)
    assert client.process is None and not client.root.exists()
    result, _ = client.call("echo", {"nonce": "after failure"})
    assert result["nonce"] == "after failure"


def test_idle_close_is_idempotent_and_releases_pipe_readers(client):
    client.call("echo")
    root, process, readers = client.root, client.process, list(client._readers)
    client.close()
    client.close()
    assert process.poll() is not None and not root.exists()
    assert all(not reader.is_alive() for reader in readers)


def test_failed_retirement_keeps_storage_until_a_successful_retry(client, monkeypatch):
    client.call("echo")
    tree, root, process = client._tree, client.root, client.process

    def fail():
        raise OSError("controlled retirement failure")

    with monkeypatch.context() as patch:
        patch.setattr(tree, "stop", fail)
        with pytest.raises(OSError, match="controlled retirement failure"):
            client.call("error")
    assert client._closed and process.poll() is None and root.exists() and list(root.iterdir())
    client.close()
    assert process.poll() is not None and not root.exists()


def test_cancel_during_result_read_discards_the_completed_packet(client, monkeypatch):
    original = bridge.read_packet
    reading, release, cancel = threading.Event(), threading.Event(), threading.Event()

    def read(path):
        packet = original(path)
        reading.set()
        assert release.wait(5)
        return packet

    monkeypatch.setattr(bridge, "read_packet", read)
    thread, outcome = launch_request(client, "echo", cancel.is_set)
    assert reading.wait(10)
    cancel.set()
    release.set()
    thread.join(10)
    assert not thread.is_alive() and "result" not in outcome
    assert isinstance(outcome["error"], RuntimeError)
    assert client.process is None and not client.root.exists()
    monkeypatch.setattr(bridge, "read_packet", original)
    result, _ = client.call("echo", {"nonce": "new packet"})
    assert result["nonce"] == "new packet"


def test_cancelled_lock_waiter_does_not_stop_the_active_request(client):
    cancel_owner = threading.Event()
    owner, outcome = launch_request(client, "busy", cancel_owner.is_set)
    pids = wait_ready(client)
    child = Observation(pids["child"])
    process = client.process
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            client.call("echo", cancelled=lambda: True)
        assert process.poll() is None and not child.stopped(0) and client.process is process
    finally:
        cancel_owner.set()
        owner.join(10)
        child.release()
    assert not owner.is_alive() and "result" not in outcome


def test_timed_out_lock_waiter_does_not_stop_the_active_request(client):
    cancel_owner = threading.Event()
    owner, outcome = launch_request(client, "busy", cancel_owner.is_set)
    pids = wait_ready(client)
    child = Observation(pids["child"])
    process = client.process
    try:
        client.timeout = 0.05
        with pytest.raises(TimeoutError, match="busy"):
            client.call("echo")
        assert process.poll() is None and not child.stopped(0) and client.process is process
    finally:
        cancel_owner.set()
        owner.join(10)
        child.release()
    assert not owner.is_alive() and "result" not in outcome


def test_startup_failure_never_runs_uncontained_worker(tmp_path, monkeypatch):
    from opencv_dataflow.simple_gdrnpp import process_tree

    marker = tmp_path / "must_not_run"
    original = process_tree._resume_primary

    def fail(process):
        assert not marker.exists()
        raise RuntimeError("controlled startup failure")

    monkeypatch.setattr(process_tree, "_resume_primary", fail)
    with pytest.raises(RuntimeError, match="controlled startup failure"):
        ProcessTree(
            [
                sys._base_executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                str(marker),
            ]
        )
    assert not marker.exists()
    monkeypatch.setattr(process_tree, "_resume_primary", original)
    tree = ProcessTree([sys._base_executable, "-c", "pass"])
    tree.process.wait(timeout=5)
    tree.stop()
    tree.stop()


def test_abrupt_owner_exit_closes_job_and_retires_descendants(tmp_path):
    marker = tmp_path / "ready.json"
    package_root = str(Path(bridge.__file__).resolve().parents[len(bridge.__package__.split("."))])
    code = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from opencv_dataflow.simple_gdrnpp.process_tree import ProcessTree
tree = ProcessTree([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[2]).write_text(json.dumps(dict(child=tree.process.pid)))
time.sleep(60)
"""
    owner = subprocess.Popen(
        [sys._base_executable, "-c", code, package_root, str(marker)],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child = Observation(json.loads(marker.read_text())["child"])
        owner.kill()
        owner.wait(timeout=5)
        assert child.stopped()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if child is not None:
            child.release()
