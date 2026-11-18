"""Persistent subprocess bridge; no Torch/Qt imports and no executable paths in projects."""

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .process_tree import ProcessTree
from .wire import read_packet, strict_json, write_packet


def runtime_profile():
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config"))
    return base / "OpenCVSimpleTools" / "gdrnpp-runtime.json"


def runtime_python():
    value = os.environ.get("SIMPLE_GDRNPP_PYTHON", "")
    if not value and runtime_profile().is_file():
        value = strict_json(runtime_profile().read_bytes())["python"]
    if not value:
        raise ValueError("Choose the isolated GDRNPP Python in Setup / Runtime")
    if not Path(value).is_absolute():
        raise ValueError("Choose an absolute installed Python path")
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Choose an installed Python executable")
    return str(path)


def configure_runtime(python):
    path = Path(python).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Choose an installed Python executable")
    target = runtime_profile()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"python": str(path)}) + "\n", encoding="utf-8", newline="\n")
    os.environ["SIMPLE_GDRNPP_PYTHON"] = str(path)


def project_recipe_path(project_dir, relative):
    if project_dir is None:
        raise ValueError("Save the project before staging setup")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or Path(relative).suffix != ".gdrnpp"
    ):
        raise ValueError("Use a project-relative .gdrnpp recipe")
    root = Path(project_dir).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Recipe resolves outside this project")
    return path


class WorkerClient:
    def __init__(self, *, python=None, timeout=180):
        self.python = python
        self.timeout = timeout
        self.process = None
        self._closed = False
        self._lock = threading.Lock()
        self._root = None
        self._responses = None
        self._tree = None
        self._readers = []

    def _start(self):
        parent = Path(tempfile.gettempdir()).resolve()
        self._root = tempfile.TemporaryDirectory(prefix="simple-gdrnpp-", dir=parent)
        self.root = Path(self._root.name).resolve()
        if not self.root.is_relative_to(parent):
            raise ValueError("Invalid private worker directory")
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        self._tree = ProcessTree(
            [
                self.python or runtime_python(),
                "-u",
                "-m",
                "opencv_simple_gdrnpp.worker",
                "--root",
                str(self.root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.process = self._tree.process
        process = self.process
        responses = queue.Queue(maxsize=4)
        self._responses = responses
        stderr = bytearray()
        self._stderr = stderr

        def read():
            try:
                while line := process.stdout.readline(65537):
                    if len(line) > 65536:
                        raise ValueError("Oversized worker protocol message")
                    responses.put_nowait(strict_json(line))
                responses.put_nowait(
                    RuntimeError(
                        "GDRNPP worker exited: " + stderr.decode("utf-8", "replace")[-2000:]
                    )
                )
            except Exception as error:
                try:
                    responses.put_nowait(error)
                except queue.Full:
                    pass

        def drain():
            try:
                while chunk := process.stderr.read1(4096):
                    stderr.extend(chunk)
                    if len(stderr) > 4000:
                        del stderr[:-4000]
            except (OSError, ValueError):
                pass

        self._readers = [threading.Thread(target=target, daemon=True) for target in (read, drain)]
        for reader in self._readers:
            reader.start()

    def _stop(self):
        process = self.process
        if self._tree is not None:
            self._tree.stop()
            self._tree = None
        if process is not None:
            for reader in self._readers:
                if reader.ident is not None:
                    reader.join(timeout=5)
                if reader.is_alive():
                    raise TimeoutError("GDRNPP worker pipe reader did not retire")
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self.process = None
        self._readers = []
        self._responses = None
        if self._root is not None:
            root = self._root
            if not Path(root.name).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
                raise RuntimeError("Private worker cleanup escaped its temporary parent")
            root.cleanup()
            self._root = None

    def call(self, operation, metadata=None, arrays=None, *, cancelled=lambda: False):
        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout
        job = None
        response_metadata = None
        failed = False
        rejected_read = False
        while not self._lock.acquire(timeout=0.05):
            if cancelled():
                raise RuntimeError("GDRNPP request cancelled")
            if time.monotonic() > deadline:
                raise TimeoutError("GDRNPP worker is busy")
        try:
            acquired = time.perf_counter()
            if self._closed:
                raise RuntimeError("GDRNPP worker has closed")
            if cancelled():
                raise RuntimeError("GDRNPP request cancelled")
            if time.monotonic() > deadline:
                raise TimeoutError("GDRNPP worker timed out")
            if self.process is None:
                self._start()
            process_ready = time.perf_counter()
            identifier = uuid.uuid4().hex
            job = self.root / identifier
            job.mkdir()
            write_packet(job, metadata or {}, arrays or {})
            if self._closed or cancelled():
                raise RuntimeError("GDRNPP request cancelled; no result published")
            if time.monotonic() > deadline:
                raise TimeoutError("GDRNPP worker timed out")
            message = json.dumps(dict(id=identifier, op=operation)).encode() + b"\n"
            self.process.stdin.write(message)
            self.process.stdin.flush()
            sent = time.perf_counter()
            while True:
                if self._closed or cancelled():
                    raise RuntimeError("GDRNPP request cancelled; no result published")
                if time.monotonic() > deadline:
                    raise TimeoutError("GDRNPP worker timed out")
                try:
                    response = self._responses.get(timeout=0.05)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise RuntimeError("GDRNPP worker exited before returning a result")
                    continue
                if isinstance(response, Exception):
                    raise response
                if response.get("id") != identifier:
                    raise RuntimeError("Stale worker response")
                if not response.get("ok"):
                    rejected_read = operation in (
                        "capture_completed",
                        "capture_completed_candidate",
                        "capture_completed_graphics",
                    ) and str(response.get("error", "")).startswith("ValueError:")
                    raise ValueError(response.get("error", "GDRNPP worker failed"))
                received = time.perf_counter()
                response_metadata, result_arrays = read_packet(job / "result")
                if self._closed or cancelled():
                    raise RuntimeError("GDRNPP request cancelled; no result published")
                if time.monotonic() > deadline:
                    raise TimeoutError("GDRNPP worker timed out")
                if not isinstance(response_metadata, dict):
                    raise ValueError("Worker metadata must be an object")
                response_metadata["request_timing"] = dict(
                    operation=operation,
                    client_lock_wait_seconds=acquired - started,
                    client_process_launch_seconds=process_ready - acquired,
                    client_packet_write_seconds=sent - process_ready,
                    client_reply_wait_seconds=received - sent,
                    client_packet_read_seconds=time.perf_counter() - received,
                    worker=response.get("timing"),
                    scope=(
                        "Serialized worker request through private packet cleanup; "
                        "excludes acquisition/GUI"
                    ),
                )
                if isinstance(response_metadata.get("report"), dict):
                    response_metadata["report"]["request_timing"] = response_metadata[
                        "request_timing"
                    ]
                return response_metadata, result_arrays
        except BaseException:
            failed = not rejected_read
            if rejected_read:
                raise
            try:
                self._stop()
            except BaseException:
                self._closed = True  # Unconfirmed retirement must never reuse this session.
                raise
            raise
        finally:
            cleanup_started = time.perf_counter()
            try:
                if not failed and job is not None and job.exists():
                    if not job.resolve().is_relative_to(self.root):
                        raise RuntimeError("Request cleanup escaped private worker storage")
                    shutil.rmtree(job)
            finally:
                if isinstance(response_metadata, dict) and "request_timing" in response_metadata:
                    response_metadata["request_timing"].update(
                        client_cleanup_seconds=time.perf_counter() - cleanup_started,
                        client_total_seconds=time.perf_counter() - started,
                    )
                self._lock.release()

    def close(self):
        self._closed = True
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
