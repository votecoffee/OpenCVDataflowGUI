"""Owned subprocess trees with bounded output, readiness, waits and cleanup."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [
                ("ProcessTime", ctypes.c_int64),
                ("JobTime", ctypes.c_int64),
                ("Flags", w.DWORD),
                ("MinWorkingSet", ctypes.c_size_t),
                ("MaxWorkingSet", ctypes.c_size_t),
                ("ActiveProcesses", w.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("Priority", w.DWORD),
                ("Scheduling", w.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOps",
                    "WriteOps",
                    "OtherOps",
                    "ReadBytes",
                    "WriteBytes",
                    "OtherBytes",
                )
            ]

        class Limits(ctypes.Structure):
            _fields_ = [
                ("Basic", Basic),
                ("IO", IO),
                ("ProcessMemory", ctypes.c_size_t),
                ("JobMemory", ctypes.c_size_t),
                ("PeakProcess", ctypes.c_size_t),
                ("PeakJob", ctypes.c_size_t),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Limits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class ManagedProcess:
    """The target cannot spawn children before ownership is established."""

    def __init__(self, argv, cwd, environment):
        self.job = None
        self.process = None
        self.child_pid = None
        self._tail = bytearray()
        self._lock = threading.Lock()
        self._ready_text = b""
        self._ready = threading.Event()
        self._reader = None
        self.closed = False
        try:
            if os.name == "nt":
                self.job = WindowsJob()
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-u",
                    str(Path(__file__).with_name("process_host.py")),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt",
            )
            if self.job is not None:
                self.job.assign(self.process)
            self._reader = threading.Thread(target=self._drain, name="program-output", daemon=True)
            self._reader.start()
            self._send({"argv": argv, "cwd": str(cwd), "environment": environment})
        except BaseException:
            self.close(0.05)
            raise

    def _send(self, value):
        self.process.stdin.write((json.dumps(value, allow_nan=False) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def _drain(self):
        try:
            while chunk := self.process.stdout.read1(4096):
                with self._lock:
                    self._tail.extend(chunk)
                    del self._tail[:-65536]
                    if self._ready_text and self._ready_text in self._tail:
                        self._ready.set()
                    if self.child_pid is None:
                        import re

                        match = re.search(rb"DATAFLOW_CHILD_PID=(\d+)", self._tail)
                        if match:
                            self.child_pid = int(match.group(1))
        except OSError, ValueError:
            pass

    def tail(self):
        with self._lock:
            return self._tail.decode("utf-8", errors="replace")

    def wait_ready(self, text, timeout, cancelled):
        self._ready_text = text.encode("utf-8")
        end = time.monotonic() + timeout
        while True:
            if cancelled():
                raise InterruptedError("Program startup cancelled")
            if self.process.poll() is not None:
                raise RuntimeError(f"Service exited during startup: {self.tail()[-4096:]}")
            if (text and (self._ready.is_set() or text in self.tail())) or (
                not text and self.child_pid is not None
            ):
                return
            if time.monotonic() >= end:
                raise TimeoutError(f"Service readiness timed out: {self.tail()[-4096:]}")
            time.sleep(0.02)

    def wait(self, timeout, cancelled):
        end = time.monotonic() + timeout
        while self.process.poll() is None:
            if cancelled():
                raise InterruptedError("Program script cancelled")
            if time.monotonic() >= end:
                raise TimeoutError("Program script timed out")
            time.sleep(0.02)
        if self._reader is not None:
            self._reader.join(timeout=0.5)
        return self.process.returncode

    def close(self, timeout=0.1, stop_input=""):
        if self.closed:
            return
        self.closed = True
        process = self.process
        try:
            if process is not None and process.poll() is None:
                try:
                    self._send({"stop_input": stop_input, "timeout": timeout})
                    process.wait(timeout=timeout + 0.5)
                except OSError, ValueError, subprocess.TimeoutExpired:
                    pass
        finally:
            if self.job is not None:
                self.job.close()  # Also kills descendants of an already-exited launcher.
            elif process is not None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
                if self._reader is not None:
                    self._reader.join(timeout=1)
                process.stdin.close()
                process.stdout.close()
