"""One owned native decoder and one shared frame slot, with cancellable deadlines."""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from .managed_process import WindowsJob


class CaptureFailure(ConnectionError):
    def __init__(self, code, retryable=True):
        super().__init__(f"Network video: {code}")
        self.code, self.retryable = code, retryable


class CaptureWorker:
    """No I/O at construction. Handles never enter graph snapshots or project files."""

    def __init__(self):
        self.process = self.memory = self.job = self.reader = None
        self.responses = queue.Queue(maxsize=1)

    def _drain(self):
        stream = self.process.stdout
        try:
            while line := stream.readline(8193):
                if len(line) > 8192:
                    raise ValueError("protocol")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("protocol")
                self.responses.put_nowait(value)
        except OSError, ValueError, queue.Full:
            try:
                self.responses.put_nowait({"status": "error", "code": "worker_protocol"})
            except queue.Full:
                pass

    def _wait(self, timeout_ms, cancelled, operation):
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            if cancelled():
                raise InterruptedError("Network video cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureFailure(f"{operation}_timeout")
            try:
                response = self.responses.get(timeout=min(0.01, remaining))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise CaptureFailure("worker_exited") from None
                continue
            if response.get("status") == "error":
                codes = {
                    "open_failed",
                    "read_failed",
                    "frame_limit_or_format",
                    "frame_size_changed",
                    "decoder_error",
                    "worker_protocol",
                }
                code = response.get("code")
                raise CaptureFailure(
                    code if code in codes else "worker_protocol",
                    response.get("retryable", False) is True,
                )
            return response

    def request(self, operation, timeout_ms, cancelled):
        if cancelled():
            raise InterruptedError("Network video cancelled")
        self._send({"op": operation})
        response = self._wait(timeout_ms, cancelled, operation)
        expected = "opened" if operation == "open" else "frame"
        if response.get("status") != expected:
            raise CaptureFailure("worker_protocol", False)
        return response

    def _send(self, value):
        try:
            self.process.stdin.write((json.dumps(value, allow_nan=False) + "\n").encode("utf8"))
            self.process.stdin.flush()
        except OSError, ValueError:
            raise CaptureFailure("worker_exited") from None

    def open(self, settings, cancelled):
        try:
            self.memory = SharedMemory(create=True, size=settings["maximum_frame_bytes"])
            if os.name == "nt":
                self.job = WindowsJob()
            environment = os.environ.copy()
            environment.update(
                OPENCV_LOG_LEVEL="OFF",
                OPENCV_VIDEOIO_DEBUG="0",
                OPENCV_FFMPEG_DEBUG="0",
                OPENCV_FFMPEG_CAPTURE_OPTIONS="",
            )
            if settings["url"].lower().startswith("rtsp:"):
                environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{settings['rtsp_transport']}"
                )
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).with_name("network_video_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt",
            )
            if self.job is not None:
                self.job.assign(self.process)
            self.reader = threading.Thread(target=self._drain, name="network-video", daemon=True)
            self.reader.start()
            self._send({**settings, "memory": self.memory.name})
            ready = self._wait(settings["startup_timeout_ms"], cancelled, "startup")
            if ready.get("status") != "ready":
                raise CaptureFailure("worker_protocol", False)
            self.request("open", settings["open_timeout_ms"], cancelled)
        except BaseException:
            self.close()
            raise

    def read(self, settings, cancelled):
        info = self.request("read", settings["read_timeout_ms"], cancelled)
        shape = info.pop("shape", None)
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[2] != 3
            or shape[0] * shape[1] > settings["maximum_frame_pixels"]
            or shape[0] * shape[1] * 3 > settings["maximum_frame_bytes"]
        ):
            raise CaptureFailure("worker_protocol", False)
        image = np.ndarray(shape, np.uint8, buffer=self.memory.buf).copy()
        info.pop("status", None)
        return image, info

    def close(self):
        process, reader = self.process, self.reader
        try:
            if self.job is not None:
                self.job.close()
            elif process is not None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
                if reader is not None:
                    reader.join(timeout=1)
                process.stdin.close()
                process.stdout.close()
        finally:
            self.process = self.job = self.reader = None
            if self.memory is not None:
                self.memory.close()
                self.memory.unlink()
                self.memory = None
