"""Own a worker and its descendants without a host-side process-monitor dependency.

Windows starts the root suspended, attaches a kill-on-close Job Object, then
resumes its primary thread. A venv launcher cannot spawn outside our job during
startup. POSIX uses a new session/process group. Only trusted worker commands
belong here; this is resource ownership, not an untrusted-code sandbox.
"""

import os
import signal
import subprocess
import time

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t),
            ("active_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", _BasicLimits),
            ("io_counters", ctypes.c_uint64 * 6),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("times", ctypes.c_int64 * 4),
            ("page_faults", wintypes.DWORD),
            ("total", wintypes.DWORD),
            ("active", wintypes.DWORD),
            ("terminated", wintypes.DWORD),
        ]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in ("size", "usage", "tid", "pid")]
        _fields_ += [
            ("priority", wintypes.LONG),
            ("base", wintypes.LONG),
            ("flags", wintypes.DWORD),
        ]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    def _function(name, arguments, result=wintypes.BOOL):
        function = getattr(_kernel, name)
        function.argtypes, function.restype = arguments, result
        return function

    _create_job = _function(
        "CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE
    )
    _set_job = _function(
        "SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    )
    _assign = _function("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE])
    _terminate = _function("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT])
    _query = _function(
        "QueryInformationJobObject",
        [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
    )
    _close = _function("CloseHandle", [wintypes.HANDLE])
    _snapshot = _function(
        "CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE
    )
    _first_thread = _function("Thread32First", [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)])
    _next_thread = _function("Thread32Next", [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)])
    _open_thread = _function(
        "OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    )
    _thread_pid = _function("GetProcessIdOfThread", [wintypes.HANDLE], wintypes.DWORD)
    _resume = _function("ResumeThread", [wintypes.HANDLE], wintypes.DWORD)

    def _checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def _resume_primary(process):
        # Popen closes the primary thread handle. Enumerate while its retained
        # process handle keeps the newly created, suspended process identity.
        snapshot = _snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            threads = []
            available = _first_thread(snapshot, ctypes.byref(entry))
            while available:
                if entry.pid == process.pid:
                    threads.append(entry.tid)
                available = _next_thread(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise ctypes.WinError(ctypes.get_last_error())
            if len(threads) != 1 or process.poll() is not None:
                raise RuntimeError("Cannot identify the suspended worker primary thread")
            thread = _checked(_open_thread(0x802, False, threads[0]))
            try:
                if _thread_pid(thread) != process.pid:
                    raise RuntimeError("Worker primary thread identity changed")
                if _resume(thread) != 1:
                    raise RuntimeError("Cannot resume the contained worker primary thread")
            finally:
                _close(thread)
        finally:
            _close(snapshot)


class ProcessTree:
    """Popen-compatible process plus an idempotent, synchronous tree stop."""

    def __init__(self, command, **kwargs):
        self.process = None
        self._job = None
        self._assigned = False
        try:
            if os.name == "nt":
                self._job = _checked(_create_job(None, None))
                limits = _ExtendedLimits()
                limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                _checked(_set_job(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | 4  # CREATE_SUSPENDED
            else:
                kwargs["start_new_session"] = True
            self.process = subprocess.Popen(command, **kwargs)
            if os.name == "nt":
                # CPython's Windows Popen owns this handle until process disposal;
                # using it avoids reopening a potentially reused numeric PID.
                _checked(_assign(self._job, int(self.process._handle)))
                self._assigned = True
                _resume_primary(self.process)
        except BaseException:
            process = self.process
            try:
                self.stop()
            finally:
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
            raise

    def stop(self):
        """Retire descendants even if the root already exited; never target a name."""
        process = self.process
        try:
            if os.name == "nt" and self._job is not None:
                _checked(_terminate(self._job, 1))
                deadline = time.monotonic() + 5
                while True:
                    accounting = _Accounting()
                    _checked(
                        _query(
                            self._job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
                        )
                    )
                    if not accounting.active:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Owned worker processes did not retire")
                    time.sleep(0.01)
            elif os.name != "nt" and process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                if os.name == "nt" and not self._assigned and process.poll() is None:
                    process.kill()  # Startup failed before assignment; still suspended.
                process.wait(timeout=5)
        finally:
            if os.name == "nt" and self._job is not None:
                job, self._job = self._job, None
                _checked(_close(job))  # Kill-on-close also covers an exceptional stop.
        self.process = None
