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
        self.process = subprocess.Popen(
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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
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

        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=drain, daemon=True).start()

    def _stop(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        if self._root is not None:
            root = self._root
            self._root = None
            if not Path(root.name).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
                raise RuntimeError("Private worker cleanup escaped its temporary parent")
            root.cleanup()

    def call(self, operation, metadata=None, arrays=None, *, cancelled=lambda: False):
        deadline = time.monotonic() + self.timeout
        job = None
        while not self._lock.acquire(timeout=0.05):
            if cancelled():
                raise RuntimeError("GDRNPP request cancelled")
            if time.monotonic() > deadline:
                raise TimeoutError("GDRNPP worker is busy")
        try:
            if self._closed:
                raise RuntimeError("GDRNPP worker has closed")
            if cancelled():
                raise RuntimeError("GDRNPP request cancelled")
            if self.process is None:
                self._start()
            identifier = uuid.uuid4().hex
            job = self.root / identifier
            job.mkdir()
            write_packet(job, metadata or {}, arrays or {})
            message = json.dumps(dict(id=identifier, op=operation)).encode() + b"\n"
            self.process.stdin.write(message)
            self.process.stdin.flush()
            while True:
                if cancelled():
                    raise RuntimeError("GDRNPP request cancelled; no result published")
                if time.monotonic() > deadline:
                    raise TimeoutError("GDRNPP worker timed out")
                try:
                    response = self._responses.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(response, Exception):
                    raise response
                if response.get("id") != identifier:
                    raise RuntimeError("Stale worker response")
                if not response.get("ok"):
                    raise ValueError(response.get("error", "GDRNPP worker failed"))
                return read_packet(job / "result")
        except BaseException:
            self._stop()
            raise
        finally:
            try:
                if job is not None and job.exists():
                    if not job.resolve().is_relative_to(self.root):
                        raise RuntimeError("Request cleanup escaped private worker storage")
                    shutil.rmtree(job)
            finally:
                self._lock.release()

    def close(self):
        self._closed = True
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
