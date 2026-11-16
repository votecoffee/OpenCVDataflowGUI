"""One program session shared by execution copies, never by unrelated programs."""

import copy
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

from .managed_process import ManagedProcess
from .program_config import json_object, validate_program, validate_scripts
from .timing import mark


class ProgramRuntime:
    def __init__(self, settings):
        self.settings = validate_program(settings)
        self.state = copy.deepcopy(self.settings["variables"])
        self.started = self.closed = False
        self.run_index = 0
        self.services = {}
        self.events = []
        self.error = ""
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            return {
                "started": self.started,
                "closed": self.closed,
                "run_index": self.run_index,
                "state": copy.deepcopy(self.state),
                "error": self.error,
                "services": {
                    name: {
                        "pid": process.child_pid,
                        "launcher_pid": process.process.pid,
                        "running": not process.closed and process.process.poll() is None,
                        "output_tail": process.tail()[-8192:],
                    }
                    for name, (_, process) in self.services.items()
                },
                "events": copy.deepcopy(self.events),
            }

    def _report(self, graph, message, severity="INFO", exc=None):
        event = {
            "kind": "program",
            "time": mark(),
            "tickets": [],
            "parent_tickets": [],
            "source": {
                "id": "__program__",
                "name": self.settings["name"] or "Program",
                "type": "program.lifecycle",
                "path": "__program__",
            },
            "data": {"run_index": self.run_index},
            "message": str(message)[-8192:],
            "severity": severity,
            "exception": None
            if exc is None
            else {
                "type": type(exc).__name__,
                "message": str(exc)[-4096:],
                "traceback": traceback.format_exc()[-16384:],
            },
        }
        graph.activity.emit(event)
        self.events.append({"message": event["message"], "severity": severity})
        del self.events[:-200]

    def _hook(self, event, graph, cancelled=lambda: False, **details):
        source = self.settings["scripts"][event]
        if not source.strip():
            return
        context = {
            "event": event,
            "project_dir": str(graph.project_dir) if graph.project_dir else None,
            "program_name": self.settings["name"],
            "run_index": self.run_index,
            "services": self.snapshot()["services"],
            **details,
        }
        with TemporaryDirectory(prefix="dataflow-hook-") as temporary:
            request, response = Path(temporary) / "request.json", Path(temporary) / "response.json"
            request.write_bytes(
                json.dumps(
                    {"source": source, "context": context, "state": self.state}, allow_nan=False
                ).encode()
            )
            process = ManagedProcess(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("program_script.py")),
                    str(request),
                    str(response),
                ],
                graph.project_dir or Path.cwd(),
                os.environ.copy(),
            )
            try:
                process.wait(self.settings["script_timeout_seconds"], cancelled)
                output = process.tail().split("\n", 1)[-1].strip()
                if output:
                    self._report(graph, f"{event}: {output[-4096:]}")
                if not response.is_file() or response.stat().st_size > 65536:
                    raise RuntimeError(f"{event} script did not return a bounded result")
                result = json.loads(response.read_bytes())
                if not result.get("ok"):
                    raise RuntimeError(
                        f"{event}: {result.get('type', 'ScriptError')}: {result.get('message', '')}"
                        f"\n{result.get('traceback', '')}"
                    )
                self.state = json_object(result["state"], "Program state")
                self._report(graph, f"{event} script complete")
            finally:
                process.close()

    def _start_service(self, service, graph, cancelled):
        python, environment = sys.executable, os.environ.copy()
        if service["runtime"] == "ros2":
            from .ros2_client import runtime_environment

            python, environment = runtime_environment()

        def expand(value):
            if "{project_dir}" in value and graph.project_dir is None:
                raise ValueError("Save the program before using {project_dir} in a service")
            return value.replace("{python}", python).replace(
                "{project_dir}", str(graph.project_dir or "")
            )

        argv = [expand(value) for value in service["argv"]]
        cwd = Path(expand(service["cwd"]))
        if not cwd.is_absolute():
            cwd = (graph.project_dir or Path.cwd()) / cwd
        environment.update({key: expand(value) for key, value in service["environment"].items()})
        process = ManagedProcess(argv, cwd, environment)
        self.services[service["id"]] = (service, process)
        process.wait_ready(service["ready_text"], service["startup_timeout_seconds"], cancelled)
        self._report(graph, f"Service {service['id']} ready (PID {process.child_pid})")

    def start(self, graph, cancelled=lambda: False):
        with self.lock:
            if self.closed:
                raise RuntimeError(
                    "Program session is closed; stop/reset program services to retry"
                )
            if self.started:
                return
            if cancelled():
                raise InterruptedError("Program startup cancelled")
            validate_scripts(self.settings)
            self.started = True  # Shutdown also applies to partial startup.
            try:
                self._report(graph, "Program starting")
                self._hook("startup", graph, cancelled)
                for service in self.settings["services"]:
                    if service["enabled"]:
                        self._start_service(service, graph, cancelled)
                self._report(graph, "Program ready")
            except BaseException as exc:
                self.error = "" if isinstance(exc, InterruptedError) else str(exc)[-8192:]
                self._report(
                    graph,
                    f"Program startup stopped: {exc}",
                    "WARNING" if isinstance(exc, InterruptedError) else "ERROR",
                    exc,
                )
                self.close(graph)
                raise

    def before_run(self, graph, cancelled):
        self.start(graph, cancelled)
        with self.lock:
            self.run_index += 1
            try:
                self._check_services()
                self._hook("before_run", graph, cancelled)
            except BaseException as exc:
                self.error = "" if isinstance(exc, InterruptedError) else str(exc)[-8192:]
                self._report(
                    graph,
                    f"Before run stopped: {exc}",
                    "WARNING" if isinstance(exc, InterruptedError) else "ERROR",
                    exc,
                )
                raise

    def _check_services(self):
        for name, (_, process) in self.services.items():
            if process.process.poll() is not None:
                raise RuntimeError(f"Program service {name} exited: {process.tail()[-4096:]}")

    def after_run(self, graph, cancelled, error=None):
        with self.lock:
            blocks = list(graph.processing_blocks())
            original_error = error
            service_error = None
            if error is None and not cancelled():
                try:
                    self._check_services()
                except Exception as exc:
                    service_error = error = exc
            status = (
                "error"
                if error or any(b.state == "error" for b in blocks)
                else "cancelled"
                if cancelled()
                else (
                    "fail"
                    if any(b.state in ("fail", "error", "skipped") for b in blocks)
                    else "pass"
                )
            )
            try:
                self._hook(
                    "after_run",
                    graph,
                    lambda: False,  # A stopped cycle still gets its bounded completion hook.
                    status=status,
                    error=str(error) if error else None,
                    blocks=[{"id": b.id, "name": b.name, "state": b.state} for b in blocks],
                )
                if service_error is not None:
                    raise service_error
            except BaseException as exc:
                self.error = str(exc)[-8192:]
                self._report(graph, f"After run failed: {exc}", "ERROR", exc)
                if original_error is None:
                    raise

    def close(self, graph):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if self.started:
                    self._hook("shutdown", graph, status="shutdown", error=self.error or None)
            except BaseException as exc:
                self.error = str(exc)[-8192:]
                self._report(graph, f"Shutdown script failed: {exc}", "ERROR", exc)
            finally:
                for name, (service, process) in reversed(tuple(self.services.items())):
                    try:
                        process.close(service["shutdown_timeout_seconds"], service["stop_input"])
                        self._report(graph, f"Service {name} stopped")
                    except BaseException as exc:
                        self._report(graph, f"Service {name} cleanup failed: {exc}", "ERROR", exc)
                if self.started:
                    self._report(graph, "Program stopped")
